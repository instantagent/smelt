// Kernel capabilities are the compiler-facing contract between planned ops,
// generated kernels, and future weight-layout planning.

import SmeltRuntime

enum SmeltKernelPhase: String, Sendable, Equatable, Hashable {
    case decode
    case prefill
    case storage
}

enum SmeltKernelOperationPattern: String, Sendable, Equatable, Hashable {
    case affineMatvecResidualAdd
    case fusedDualAffineMatvecResidualAdd
    case affineMatvecPrefillFull
    case fusedGateUpSwigluPrefillFull
    case affineMatvecPrefillSmallBatch
    case fusedDualAffineMatvecPrefillSmallBatch
    case fusedGateUpSwigluPrefillSmallBatch
    case affineVerifyArgmaxPrefill
    case verifyArgmaxReduce
    case affineStorageRead
    case signedStorageRead
}

enum SmeltWeightStorageKind: Sendable, Equatable, Hashable {
    case affineU4RowMajor(groupSize: Int)
    case signedRowMajor(format: SmeltSignedQuantFormat, groupSize: Int)
}

enum SmeltKernelWeightRole: String, Sendable, Equatable, Hashable {
    case affine
    case signed
    case key
    case value
    case gate
    case up
    case first
    case second
}

enum SmeltKernelCapabilitySource: String, Sendable, Equatable {
    case builtInCatalog = "built_in_catalog"
    case packageLocalGenerated = "package_local_generated"
    case storage
}

enum SmeltGeneratedKernelSourceTemplate: String, Sendable, Equatable, Hashable {
    case affineMatvecResidualAddFixed
    case affineMatvecResidualAddFixedRows4
    case fusedDualAffineMatvecResidualAddFixedRows4
    case affineMatvecPrefillFull
    case fusedGateUpSwigluPrefillFull
    case affineMatvecPrefillSmallBatch
    case fusedDualAffineMatvecPrefillSmallBatch
    case fusedGateUpSwigluPrefillSmallBatch
    case affineVerifyArgmaxPrefill
    case verifyArgmaxReduce
}

struct SmeltKernelWeightRequirement: Sendable, Equatable {
    let role: SmeltKernelWeightRole
    let acceptedLayouts: [SmeltWeightStorageKind]
}

struct SmeltKernelShape: Sendable, Equatable, Hashable {
    let rows: Int
    let cols: Int
    let groupSize: Int
}

struct SmeltKernelCapability: Sendable, Equatable {
    let id: String
    let phase: SmeltKernelPhase
    let operation: SmeltKernelOperationPattern
    let shape: SmeltKernelShape
    let source: SmeltKernelCapabilitySource
    let sourceTemplate: SmeltGeneratedKernelSourceTemplate?
    let weightRequirements: [SmeltKernelWeightRequirement]
    let rowTile: Int?
    let batchTile: Int?
    let threadgroupWidth: Int?

    init(
        id: String,
        phase: SmeltKernelPhase,
        operation: SmeltKernelOperationPattern,
        shape: SmeltKernelShape,
        source: SmeltKernelCapabilitySource,
        sourceTemplate: SmeltGeneratedKernelSourceTemplate? = nil,
        weightRequirements: [SmeltKernelWeightRequirement],
        rowTile: Int?,
        batchTile: Int?,
        threadgroupWidth: Int?
    ) {
        self.id = id
        self.phase = phase
        self.operation = operation
        self.shape = shape
        self.source = source
        self.sourceTemplate = sourceTemplate
        self.weightRequirements = weightRequirements
        self.rowTile = rowTile
        self.batchTile = batchTile
        self.threadgroupWidth = threadgroupWidth
    }

    var requiresPackageLocalGeneratedSource: Bool {
        source == .packageLocalGenerated
    }
}

struct SmeltGeneratedKernelCapabilityDescriptor: Sendable {
    let operation: SmeltKernelOperationPattern
    let phase: SmeltKernelPhase
    let sourceTemplate: SmeltGeneratedKernelSourceTemplate
    let weightRoles: [SmeltKernelWeightRole]
    let rowTile: Int
    let batchTile: Int?
    let threadgroupWidth: Int
    let supportsShape: @Sendable (SmeltKernelShape) -> Bool
    let capabilityID: @Sendable (SmeltKernelShape) -> String

    func capability(
        for shape: SmeltKernelShape,
        source: SmeltKernelCapabilitySource
    ) -> SmeltKernelCapability? {
        guard supportsShape(shape) else {
            return nil
        }
        let acceptedLayout = SmeltWeightStorageKind.affineU4RowMajor(
            groupSize: shape.groupSize
        )
        return SmeltKernelCapability(
            id: capabilityID(shape),
            phase: phase,
            operation: operation,
            shape: shape,
            source: source,
            sourceTemplate: sourceTemplate,
            weightRequirements: weightRoles.map {
                SmeltKernelWeightRequirement(
                    role: $0,
                    acceptedLayouts: [acceptedLayout]
                )
            },
            rowTile: sourceTemplate == .affineMatvecPrefillSmallBatch
                ? SmeltGeneratedKernelVariants.prefillAffineSmallBatchRowTile(
                    cols: shape.cols
                )
                : rowTile,
            batchTile: batchTile,
            threadgroupWidth: threadgroupWidth
        )
    }
}

enum SmeltKernelCapabilityRegistry {
    private static let generatedCapabilityDescriptors: [SmeltGeneratedKernelCapabilityDescriptor] = [
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .affineMatvecResidualAdd,
            phase: .decode,
            sourceTemplate: .affineMatvecResidualAddFixedRows4,
            weightRoles: [.affine],
            rowTile: 4,
            batchTile: nil,
            threadgroupWidth: 64,
            supportsShape: { shape in
                SmeltGeneratedKernelVariants.canGenerateAffineU4FixedRows4(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.decodeFusedAffineMatvecAddRows4Name(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            }
        ),
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .affineMatvecPrefillFull,
            phase: .prefill,
            sourceTemplate: .affineMatvecPrefillFull,
            weightRoles: [.affine],
            rowTile: 32,
            batchTile: 16,
            threadgroupWidth: 128,
            supportsShape: { shape in
                SmeltGeneratedKernelVariants.canGenerateAffineU4Full(
                    groupSize: shape.groupSize
                )
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.prefillAffineFullName(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            }
        ),
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .fusedDualAffineMatvecResidualAdd,
            phase: .decode,
            sourceTemplate: .fusedDualAffineMatvecResidualAddFixedRows4,
            weightRoles: [.key, .value],
            rowTile: 4,
            batchTile: nil,
            threadgroupWidth: 64,
            supportsShape: { shape in
                SmeltGeneratedKernelVariants.canGenerateAffineU4FixedRows4(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.decodeFusedDualAffineMatvecAddRows4Name(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            }
        ),
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .fusedGateUpSwigluPrefillFull,
            phase: .prefill,
            sourceTemplate: .fusedGateUpSwigluPrefillFull,
            weightRoles: [.gate, .up],
            rowTile: 32,
            batchTile: 16,
            threadgroupWidth: 128,
            supportsShape: { shape in
                SmeltGeneratedKernelVariants.canGenerateAffineU4Full(
                    groupSize: shape.groupSize
                )
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.prefillFusedGateUpSwigluFullName(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            }
        ),
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .affineMatvecPrefillSmallBatch,
            phase: .prefill,
            sourceTemplate: .affineMatvecPrefillSmallBatch,
            weightRoles: [.affine],
            rowTile: 8,
            batchTile: 4,
            threadgroupWidth: 64,
            supportsShape: { shape in
                SmeltGeneratedKernelVariants.canGenerateAffineU4Fixed(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.prefillAffineSmallBatchName(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            }
        ),
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .fusedGateUpSwigluPrefillSmallBatch,
            phase: .prefill,
            sourceTemplate: .fusedGateUpSwigluPrefillSmallBatch,
            weightRoles: [.gate, .up],
            rowTile: 8,
            batchTile: 4,
            threadgroupWidth: 64,
            supportsShape: { shape in
                SmeltGeneratedKernelVariants.canGenerateAffineU4Fixed(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.prefillFusedGateUpSwigluSmallBatchName(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            }
        ),
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .fusedDualAffineMatvecPrefillSmallBatch,
            phase: .prefill,
            sourceTemplate: .fusedDualAffineMatvecPrefillSmallBatch,
            weightRoles: [.first, .second],
            rowTile: 8,
            batchTile: 4,
            threadgroupWidth: 64,
            supportsShape: { shape in
                SmeltGeneratedKernelVariants.canGenerateAffineU4Fixed(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.prefillFusedDualAffineSmallBatchName(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            }
        ),
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .affineVerifyArgmaxPrefill,
            phase: .prefill,
            sourceTemplate: .affineVerifyArgmaxPrefill,
            weightRoles: [.affine],
            rowTile: 8,
            batchTile: 4,
            threadgroupWidth: 64,
            supportsShape: { shape in
                shape.rows <= (1 << 18) - 1
                    && SmeltGeneratedKernelVariants.canGenerateAffineU4Fixed(
                        rows: shape.rows,
                        cols: shape.cols,
                        groupSize: shape.groupSize
                    )
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.prefillVerifyArgmaxName(
                    rows: shape.rows,
                    cols: shape.cols,
                    groupSize: shape.groupSize
                )
            }
        ),
        SmeltGeneratedKernelCapabilityDescriptor(
            operation: .verifyArgmaxReduce,
            phase: .prefill,
            sourceTemplate: .verifyArgmaxReduce,
            weightRoles: [],
            rowTile: 1,
            batchTile: nil,
            threadgroupWidth: 256,
            supportsShape: { shape in
                shape.rows > 0
                    && shape.rows % 8 == 0
                    && shape.rows <= (1 << 18) - 1
            },
            capabilityID: { shape in
                SmeltGeneratedKernelVariants.prefillVerifyArgmaxReduceName(
                    rows: shape.rows
                )
            }
        ),
    ]

    static func generatedCapability(
        operation: SmeltKernelOperationPattern,
        shape: SmeltKernelShape
    ) -> SmeltKernelCapability? {
        for descriptor in generatedCapabilityDescriptors where descriptor.operation == operation {
            let id = descriptor.capabilityID(shape)
            if let capability = descriptor.capability(
                for: shape,
                source: generatedCapabilitySource(id: id)
            ) {
                return capability
            }
        }
        return nil
    }

    static func affineStorageRead(
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltKernelCapability {
        SmeltKernelCapability(
            id: "affine_u4_row_major_storage_r\(rows)_c\(cols)_g\(groupSize)",
            phase: .storage,
            operation: .affineStorageRead,
            shape: SmeltKernelShape(rows: rows, cols: cols, groupSize: groupSize),
            source: .storage,
            sourceTemplate: nil,
            weightRequirements: [
                SmeltKernelWeightRequirement(
                    role: .affine,
                    acceptedLayouts: [.affineU4RowMajor(groupSize: groupSize)]
                ),
            ],
            rowTile: nil,
            batchTile: nil,
            threadgroupWidth: nil
        )
    }

    static func signedStorageRead(
        format: SmeltSignedQuantFormat,
        rows: Int,
        cols: Int,
        groupSize: Int
    ) -> SmeltKernelCapability {
        let layout = SmeltWeightStorageKind.signedRowMajor(
            format: format, groupSize: groupSize)
        return SmeltKernelCapability(
            id: "\(format.rawValue)_row_major_storage_r\(rows)_c\(cols)_g\(groupSize)",
            phase: .storage,
            operation: .signedStorageRead,
            shape: SmeltKernelShape(rows: rows, cols: cols, groupSize: groupSize),
            source: .storage,
            sourceTemplate: nil,
            weightRequirements: [
                SmeltKernelWeightRequirement(role: .signed, acceptedLayouts: [layout]),
            ],
            rowTile: nil,
            batchTile: nil,
            threadgroupWidth: nil
        )
    }

    private static func generatedCapabilitySource(id: String) -> SmeltKernelCapabilitySource {
        SmeltKernelCatalog.pipelineIndex(named: id) == nil
            ? .packageLocalGenerated
            : .builtInCatalog
    }

}
