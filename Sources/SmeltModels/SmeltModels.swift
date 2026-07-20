import SmeltModuleAuthoring

/// The registry of model definitions. `smelt-models emit` writes one
/// `<module-id>.module.json` per entry; parity and determinism tests consume
/// the values directly.
public enum SmeltModels {
    /// Every model definition, as fully-lowered IR values.
    public static var all: [SmeltCAMIR] {
        [
            qwen35Text(),
            qwen35Fast(),
            qwen36TwentySevenB(),
            qwen36TwentySevenBMTP(),
            bonsaiTwentySevenBBinary(),
            bonsaiTwentySevenBTernary(),
            qwen35FourB(),
            qwen35Reasoner(),
            ds4HeavyQuant(),
            qwen3TTS(),
            inkling(),
            inklingMTP(),
        ]
    }

    /// Look up a definition by its module id.
    public static func definition(id: String) -> SmeltCAMIR? {
        all.first { $0.module.id == id }
    }
}
