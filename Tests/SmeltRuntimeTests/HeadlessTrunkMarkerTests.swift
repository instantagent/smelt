// HeadlessTrunkMarkerTests — graph identity comes from the explicit manifest
// marker; BF16 and FP32 remain storage cells of the same port contract.

import Testing
import SmeltSchema
@testable import SmeltRuntime

private let fp32A = (index: 7, dtype: SmeltDType.fp32)
private let fp32B = (index: 9, dtype: SmeltDType.fp32)
private let bf16A = (index: 11, dtype: SmeltDType.bf16)
private let bf16B = (index: 13, dtype: SmeltDType.bf16)

@Test func denseTrunkPortsTreatBF16AndFP32AsStorageFamilies() throws {
    let fp32 = try SmeltRuntime.resolveDenseTrunkPorts(
        caller: "t",
        headlessTrunkABI: true,
        kind: nil,
        hiddenA: fp32A,
        normOut: fp32B
    )
    #expect(fp32 == (hiddenA: 7, normOut: 9, dtype: .fp32))

    let bf16 = try SmeltRuntime.resolveDenseTrunkPorts(
        caller: "t",
        headlessTrunkABI: true,
        kind: nil,
        hiddenA: bf16A,
        normOut: bf16B
    )
    #expect(bf16 == (hiddenA: 11, normOut: 13, dtype: .bf16))
}

@Test func denseTrunkPortsRejectMixedStorageFamilies() {
    #expect(throws: SmeltRuntimeError.self) {
        try SmeltRuntime.resolveDenseTrunkPorts(
            caller: "t",
            headlessTrunkABI: true,
            kind: nil,
            hiddenA: bf16A,
            normOut: fp32B
        )
    }
}

@Test func headlessTrunkMarkerGatesTheDenseTrunkAPI() throws {
    let ports = try SmeltRuntime.resolveDenseTrunkPorts(
        caller: "t", headlessTrunkABI: true, kind: nil, hiddenA: fp32A, normOut: fp32B)
    #expect(ports == (hiddenA: 7, normOut: 9, dtype: .fp32))
}

@Test func kindLabelDoesNotBypassTheMarker() throws {
    for kind in ["headless-trunk", "tts-trunk"] {
        #expect(throws: SmeltRuntimeError.self) {
            try SmeltRuntime.resolveDenseTrunkPorts(
                caller: "t", headlessTrunkABI: nil, kind: kind, hiddenA: fp32A, normOut: fp32B)
        }
    }
}

@Test func unmarkedPackageWithFP32PortsIsRejected() {
    // A non-trunk package that happens to expose compatible ports is rejected
    // because it is not marked.
    for kind in [String?.none, "headless-trunk", "graphless-package"] {
        #expect(throws: SmeltRuntimeError.self) {
            try SmeltRuntime.resolveDenseTrunkPorts(
                caller: "t", headlessTrunkABI: false, kind: kind, hiddenA: fp32A, normOut: fp32B)
        }
        #expect(throws: SmeltRuntimeError.self) {
            try SmeltRuntime.resolveDenseTrunkPorts(
                caller: "t", headlessTrunkABI: nil, kind: kind, hiddenA: fp32A, normOut: fp32B)
        }
    }
}

@Test func markedTrunkMissingPortsIsRejected() {
    #expect(throws: SmeltRuntimeError.self) {
        try SmeltRuntime.resolveDenseTrunkPorts(
            caller: "t", headlessTrunkABI: true, kind: nil, hiddenA: nil, normOut: fp32B)
    }
}

@Test func markedTrunkWithUnregisteredStorageCellIsRejected() {
    #expect(throws: SmeltRuntimeError.self) {
        try SmeltRuntime.resolveDenseTrunkPorts(
            caller: "t", headlessTrunkABI: true, kind: nil,
            hiddenA: (index: 7, dtype: .fp16), normOut: fp32B)
    }
}
