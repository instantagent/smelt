enum SmeltBuiltInFileTransforms {
    static let registry = SmeltFileTransformRegistry(registrations: [
        SmeltCopyFileTransform.registration,
        SmeltMeshAutoRigFileTransform.registration,
    ])
}
