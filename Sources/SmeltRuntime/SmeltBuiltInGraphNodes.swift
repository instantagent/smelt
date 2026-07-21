enum SmeltBuiltInGraphNodes {
    static let registry = SmeltGraphNodeRegistry(
        registrations: [SmeltCopyFileTransform.registration]
            + SmeltSkinningGraphNodes.registrations
    )
}
