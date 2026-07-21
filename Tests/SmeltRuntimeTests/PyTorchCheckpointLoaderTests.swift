import Foundation
import XCTest

@testable import SmeltRuntime

final class PyTorchCheckpointLoaderTests: XCTestCase {
    func testSyntheticDenseBF16TensorIsZeroCopyAndExact() throws {
        let tensorBytes: [UInt8] = [0x00, 0x3f, 0x80, 0x3f, 0x00, 0x40, 0x40, 0x40]
        let archive = makeStoredZip(entries: [
            ("fixture/data.pkl", makeTensorPickle()),
            ("fixture/byteorder", Data("little".utf8)),
            ("fixture/data/0", Data(tensorBytes)),
        ])
        let path = FileManager.default.temporaryDirectory
            .appendingPathComponent("smelt-pytorch-checkpoint-\(UUID().uuidString).ckpt")
        try archive.write(to: path)
        defer { try? FileManager.default.removeItem(at: path) }

        let loader = try PyTorchCheckpointLoader(path: path.path)
        XCTAssertEqual(loader.tensors.count, 1)
        XCTAssertEqual(loader.tensors[0].name, "model.weight")
        XCTAssertEqual(loader.tensors[0].dtype, "BF16")
        XCTAssertEqual(loader.tensors[0].shape, [2, 2])
        XCTAssertEqual(loader.tensors[0].strides, [2, 1])
        XCTAssertEqual(loader.tensors[0].byteCount, tensorBytes.count)
        let descriptor = try XCTUnwrap(loader.checkpointTensors.first)
        let pointer = loader.checkpointTensorData(descriptor)
            .assumingMemoryBound(to: UInt8.self)
        XCTAssertEqual(
            Array(UnsafeBufferPointer(start: pointer, count: tensorBytes.count)),
            tensorBytes
        )
    }

  func testCanonicalSkinTokensCheckpointInventoryAndAlias() throws {
    guard let path = ProcessInfo.processInfo.environment["SMELT_SKINNING_CHECKPOINT"] else {
      throw XCTSkip("SMELT_SKINNING_CHECKPOINT is not set")
        }
        let loader = try PyTorchCheckpointLoader(path: path)
        XCTAssertEqual(loader.tensors.count, 672)
        XCTAssertTrue(loader.tensors.allSatisfy { $0.dtype == "BF16" })
        let totalElements = loader.tensors.reduce(0) { partial, tensor in
            partial + tensor.shape.reduce(1, *)
        }
        XCTAssertEqual(totalElements, 595_262_598)

        let byName = Dictionary(uniqueKeysWithValues: loader.tensors.map { ($0.name, $0) })
        let embedding = try XCTUnwrap(byName["transformer.model.embed_tokens.weight"])
        let head = try XCTUnwrap(byName["transformer.lm_head.weight"])
        XCTAssertEqual(embedding.shape, [33_036, 896])
        XCTAssertEqual(head.shape, embedding.shape)
        XCTAssertEqual(head.storageKey, embedding.storageKey)
        XCTAssertEqual(head.storageOffset, embedding.storageOffset)

        let prefixes = ["vae.", "mesh_encoder.", "transformer.", "output_proj."]
        let prefixCounts = prefixes.map { prefix in
            loader.tensors.filter { $0.name.hasPrefix(prefix) }.count
        }
        XCTAssertEqual(prefixCounts, [252, 106, 311, 3])
    }

    private func makeTensorPickle() -> Data {
        var data = Data([0x80, 0x02]) // PROTO 2
        appendUnicode("model.weight", to: &data)
        appendGlobal(module: "torch._utils", name: "_rebuild_tensor_v2", to: &data)
        data.append(0x28) // MARK: rebuild arguments
        data.append(0x28) // MARK: persistent storage tuple
        appendUnicode("storage", to: &data)
        appendGlobal(module: "torch", name: "BFloat16Storage", to: &data)
        appendUnicode("0", to: &data)
        appendUnicode("cpu", to: &data)
        data.append(contentsOf: [0x4b, 0x04, 0x74, 0x51]) // BININT1 4, TUPLE, BINPERSID
        data.append(contentsOf: [0x4b, 0x00]) // storage offset 0
        data.append(contentsOf: [0x4b, 0x02, 0x4b, 0x02, 0x86]) // shape (2, 2)
        data.append(contentsOf: [0x4b, 0x02, 0x4b, 0x01, 0x86]) // strides (2, 1)
        data.append(0x89) // NEWFALSE
        appendGlobal(module: "collections", name: "OrderedDict", to: &data)
        data.append(contentsOf: [0x29, 0x52, 0x74, 0x52, 0x2e])
        return data
    }

    private func appendUnicode(_ value: String, to data: inout Data) {
        let bytes = Data(value.utf8)
        data.append(0x58)
        appendUInt32(UInt32(bytes.count), to: &data)
        data.append(bytes)
    }

    private func appendGlobal(module: String, name: String, to data: inout Data) {
        data.append(0x63)
        data.append(Data("\(module)\n\(name)\n".utf8))
    }

    private func makeStoredZip(entries: [(String, Data)]) -> Data {
        struct CentralEntry {
            let name: String
            let byteCount: Int
            let localOffset: Int
        }
        var archive = Data()
        var centralEntries: [CentralEntry] = []
        for (name, payload) in entries {
            let localOffset = archive.count
            let nameData = Data(name.utf8)
            appendUInt32(0x0403_4b50, to: &archive)
            appendUInt16(20, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt32(0, to: &archive)
            appendUInt32(UInt32(payload.count), to: &archive)
            appendUInt32(UInt32(payload.count), to: &archive)
            appendUInt16(UInt16(nameData.count), to: &archive)
            appendUInt16(0, to: &archive)
            archive.append(nameData)
            archive.append(payload)
            centralEntries.append(
                CentralEntry(name: name, byteCount: payload.count, localOffset: localOffset)
            )
        }

        let centralOffset = archive.count
        for entry in centralEntries {
            let nameData = Data(entry.name.utf8)
            appendUInt32(0x0201_4b50, to: &archive)
            appendUInt16(20, to: &archive)
            appendUInt16(20, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt32(0, to: &archive)
            appendUInt32(UInt32(entry.byteCount), to: &archive)
            appendUInt32(UInt32(entry.byteCount), to: &archive)
            appendUInt16(UInt16(nameData.count), to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt16(0, to: &archive)
            appendUInt32(0, to: &archive)
            appendUInt32(UInt32(entry.localOffset), to: &archive)
            archive.append(nameData)
        }
        let centralSize = archive.count - centralOffset
        appendUInt32(0x0605_4b50, to: &archive)
        appendUInt16(0, to: &archive)
        appendUInt16(0, to: &archive)
        appendUInt16(UInt16(centralEntries.count), to: &archive)
        appendUInt16(UInt16(centralEntries.count), to: &archive)
        appendUInt32(UInt32(centralSize), to: &archive)
        appendUInt32(UInt32(centralOffset), to: &archive)
        appendUInt16(0, to: &archive)
        return archive
    }

    private func appendUInt16(_ value: UInt16, to data: inout Data) {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
    }

    private func appendUInt32(_ value: UInt32, to data: inout Data) {
        data.append(UInt8(truncatingIfNeeded: value))
        data.append(UInt8(truncatingIfNeeded: value >> 8))
        data.append(UInt8(truncatingIfNeeded: value >> 16))
        data.append(UInt8(truncatingIfNeeded: value >> 24))
    }
}
