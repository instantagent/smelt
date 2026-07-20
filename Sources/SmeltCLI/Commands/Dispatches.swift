import Foundation
import SmeltSchema

/// `smelt dispatches model.smeltpkg [--prefill] [--filter SUBSTRING] [--sequence]`
///
/// Static dump of a package's dispatch table: per-pipeline dispatch counts,
/// grid/threadgroup geometry, and position guards. With `--sequence`, prints
/// records in table order with concrete buffer/constant bindings. No Metal
/// device needed — this reads manifest.json + dispatches.bin directly, so it
/// works for diagnosing a package's decode plan without running it.
func runDispatchesCommand() {
    guard args.count >= 3 else {
        fputs("Usage: smelt dispatches <model.smeltpkg> [--prefill] [--filter SUBSTRING] [--sequence]\n", stderr)
        exit(1)
    }
    let pkgPath = args[2]
    let usePrefill = args.contains("--prefill")
    let showSequence = args.contains("--sequence")
    let filter = parseArg("--filter")
    let capabilities = requireCAMPackageCapabilitiesOrExit(
        packagePath: pkgPath,
        verb: "dispatches"
    )
    let request: SmeltCAMCapabilityRequest = usePrefill
        ? .inspectPrefillDispatchTable
        : .inspectDecodeDispatchTable
    do {
        _ = try capabilities.resolve(request)
    } catch SmeltCAMPackageCapabilitiesError.noMatchingExport {
        fputs("smelt dispatches: no CAM export satisfies dispatch table request\n", stderr)
        exit(1)
    } catch {
        fputs("smelt dispatches: \(error)\n", stderr)
        exit(1)
    }
    requireCAMCapabilityFilesOrExit(
        request.requiredPackageFiles,
        packagePath: pkgPath,
        verb: "dispatches"
    )

    let tablePath = usePrefill
        ? "\(pkgPath)/prefill_dispatches.bin"
        : "\(pkgPath)/dispatches.bin"

    let pipelines = loadOptionalDispatchPipelineNames(packagePath: pkgPath)

    let raw: Data
    do {
        raw = try Data(contentsOf: URL(fileURLWithPath: tablePath))
    } catch {
        fputs("Failed to load \(tablePath): \(error)\n", stderr)
        exit(1)
    }
    let recordStride = MemoryLayout<SmeltDispatchRecord>.stride
    guard raw.count % recordStride == 0 else {
        fputs(
            "Table size \(raw.count) is not a multiple of record stride "
                + "\(recordStride); package was built by an incompatible compiler\n",
            stderr
        )
        exit(1)
    }
    // Copy into properly-aligned storage rather than binding Data's bytes
    // in place — Data makes no alignment guarantee for the record type.
    let recordCount = raw.count / recordStride
    var records = [SmeltDispatchRecord](
        repeating: .empty(), count: recordCount
    )
    _ = records.withUnsafeMutableBytes { dst in
        raw.copyBytes(to: dst)
    }

    // Render one grid dimension, decoding the packed dynamic-grid kind so a
    // prefill record like (literal 8, seqLen, literal 1) doesn't print as a
    // zero-height dispatch.
    func gridDim(_ literal: UInt32, kind: UInt8) -> String {
        switch kind {
        case SmeltDispatchRecord.gridSeqLen:
            return "seqLen"
        case SmeltDispatchRecord.gridSeqLenMulLiteral:
            return "seqLen*\(literal)"
        case SmeltDispatchRecord.gridSeqLenCeilDivLiteral:
            let divisor = literal & 0x7FFF_FFFF
            return (literal & 0x8000_0000) != 0
                ? "seqLen/\(divisor)"
                : "ceil(seqLen/\(divisor))"
        default:
            return "\(literal)"
        }
    }

    func bufferSummary(_ rec: SmeltDispatchRecord) -> String {
        guard rec.bufferCount > 0 else { return "[]" }
        var parts: [String] = []
        parts.reserveCapacity(Int(rec.bufferCount))
        for index in 0..<Int(rec.bufferCount) {
            let buffer = getBuffer(rec, index: index)
            let slot: String
            switch buffer.slot {
            case SmeltBufferRecord.slotCur:
                slot = "cur"
            case SmeltBufferRecord.slotAlt:
                slot = "alt"
            default:
                slot = "\(buffer.slot)"
            }
            let offset = buffer.offsetKind == 0
                ? "\(buffer.offset)"
                : "kind\(buffer.offsetKind):\(buffer.offset)"
            parts.append("\(buffer.bindingIndex)=s\(slot)+\(offset)")
        }
        return "[" + parts.joined(separator: ", ") + "]"
    }

    func constantSummary(_ rec: SmeltDispatchRecord) -> String {
        guard rec.constantCount > 0 else { return "[]" }
        var parts: [String] = []
        parts.reserveCapacity(Int(rec.constantCount))
        for index in 0..<Int(rec.constantCount) {
            let constant = getConstant(rec, index: index)
            parts.append(
                "\(constant.bindingIndex)=k\(constant.kind):\(constant.value)"
            )
        }
        return "[" + parts.joined(separator: ", ") + "]"
    }

    func recordName(_ rec: SmeltDispatchRecord) -> String {
        let idx = Int(rec.pipeline)
        return idx < pipelines.count ? pipelines[idx] : "pipeline_\(idx)"
    }

    if showSequence {
        print("table: \(tablePath)")
        print("records=\(records.count)")
        for (index, rec) in records.enumerated() {
            if rec.opKind == SmeltDispatchRecord.opSwap {
                if filter.isEmpty || "swap".contains(filter) {
                    print("\(index): swap")
                }
                continue
            }
            let name = recordName(rec)
            if !filter.isEmpty, !name.contains(filter) { continue }
            let style = rec.dispatchStyle == SmeltDispatchRecord.styleThreads
                ? "threads" : "tgroups"
            print(
                "\(index): \(name) grid=(\(gridDim(rec.gridW, kind: rec.gridWKind)),\(gridDim(rec.gridH, kind: rec.gridHKind)),\(gridDim(rec.gridD, kind: rec.gridDKind)))"
                    + " tg=(\(rec.tgW),\(rec.tgH),\(rec.tgD)) \(style)"
                    + " buffers=\(bufferSummary(rec)) constants=\(constantSummary(rec))"
            )
        }
        return
    }

    // Group dispatches by pipeline, then by (geometry, guards) signature.
    struct Variant: Hashable {
        let grid: [String]
        let tg: [UInt32]
        let style: UInt8
        let guards: String
    }
    var counts: [String: Int] = [:]
    var variants: [String: [Variant: Int]] = [:]
    var order: [String] = []
    var swaps = 0

    for rec in records {
        if rec.opKind == SmeltDispatchRecord.opSwap {
            swaps += 1
            continue
        }
        let name = recordName(rec)
        var guards: [String] = []
        if rec.minSeqLen > 0 {
            guards.append("seqLen>=\(rec.minSeqLen)")
        }
        for c in 0..<Int(rec.constantCount) {
            let con = getConstant(rec, index: c)
            switch con.kind {
            case SmeltConstantRecord.kindPositionPlus1LessThanLiteralSkipIfFalse:
                guards.append("pos+1<\(con.value)")
            case SmeltConstantRecord.kindPositionPlus1GreaterEqualLiteralSkipIfFalse:
                guards.append("pos+1>=\(con.value)")
            case SmeltConstantRecord.kindSeqLenModLiteralSkipIfZero:
                guards.append("seqLen%\(con.value)!=0")
            case SmeltConstantRecord.kindSeqLenLessThanLiteralSkipIfFalse:
                guards.append("seqLen<\(con.value)")
            default:
                break
            }
        }
        let variant = Variant(
            grid: [
                gridDim(rec.gridW, kind: rec.gridWKind),
                gridDim(rec.gridH, kind: rec.gridHKind),
                gridDim(rec.gridD, kind: rec.gridDKind),
            ],
            tg: [rec.tgW, rec.tgH, rec.tgD],
            style: rec.dispatchStyle,
            guards: guards.joined(separator: ",")
        )
        if counts[name] == nil { order.append(name) }
        counts[name, default: 0] += 1
        variants[name, default: [:]][variant, default: 0] += 1
    }

    let total = counts.values.reduce(0, +)
    print("table: \(tablePath)")
    print("records=\(records.count) swaps=\(swaps) dispatches=\(total)")
    for name in order.sorted(by: { counts[$0]! > counts[$1]! }) {
        if !filter.isEmpty, !name.contains(filter) { continue }
        print("\n\(name): \(counts[name]!)")
        let sorted = variants[name]!.sorted { $0.value > $1.value }
        for (v, count) in sorted {
            let style = v.style == SmeltDispatchRecord.styleThreads
                ? "threads" : "tgroups"
            let guardSuffix = v.guards.isEmpty ? "" : "  guard[\(v.guards)]"
            print(
                "    grid=(\(v.grid[0]),\(v.grid[1]),\(v.grid[2]))"
                    + " tg=(\(v.tg[0]),\(v.tg[1]),\(v.tg[2]))"
                    + " \(style)\(guardSuffix)  x\(count)"
            )
        }
    }
}

private struct DispatchPipelineNames: Decodable {
    let pipelines: [String]?
}

private func loadOptionalDispatchPipelineNames(packagePath: String) -> [String] {
    let manifestPath = "\(packagePath)/manifest.json"
    guard FileManager.default.fileExists(atPath: manifestPath),
          let manifestData = try? Data(contentsOf: URL(fileURLWithPath: manifestPath)),
          let decoded = try? JSONDecoder().decode(DispatchPipelineNames.self, from: manifestData)
    else {
        return []
    }
    return decoded.pipelines ?? []
}
