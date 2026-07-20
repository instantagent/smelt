import Testing

@testable import SmeltCompiler

@Suite("Bounded HF checkpoint inventory")
struct HFCheckpointInventoryTests {
    @Test("Header metadata summarizes without checkpoint bodies")
    func summarizesHeaderMetadata() {
        let tensors = [
            HFCheckpointTensorMetadata(
                name: "a",
                dtype: "BF16",
                shape: [2, 4],
                filename: "model-00001-of-00002.safetensors",
                fileOffsetStart: 128,
                fileOffsetEnd: 144
            ),
            HFCheckpointTensorMetadata(
                name: "b",
                dtype: "BF16",
                shape: [3],
                filename: "model-00001-of-00002.safetensors",
                fileOffsetStart: 144,
                fileOffsetEnd: 150
            ),
            HFCheckpointTensorMetadata(
                name: "c",
                dtype: "F32",
                shape: [2],
                filename: "model-00002-of-00002.safetensors",
                fileOffsetStart: 96,
                fileOffsetEnd: 104
            ),
        ]
        let inventory = SmeltHFCheckpointInventoryProbe.summarize(
            modelID: "fixture/model",
            revision: "exact",
            tensors: tensors
        )

        #expect(inventory.tensorCount == 3)
        #expect(inventory.tensorBytes == 30)
        #expect(inventory.files.map(\.name) == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ])
        #expect(inventory.files.map(\.tensorBytes) == [22, 8])
        #expect(inventory.dtypes == [
            .init(name: "BF16", count: 2),
            .init(name: "F32", count: 1),
        ])
        #expect(inventory.transferPolicy == "safetensors-prefix-and-json-header-ranges-only")
    }
}
