import Foundation

extension SmeltCompiler {
    /// Compile the package's Metal sources. This changes executable behavior,
    /// never packed checkpoint bytes, so it intentionally lives outside the
    /// weight-building source fingerprint.
    static func compileMetalLibrary(
        shaderDir: String,
        outputPath: String,
        generatedLutMatvecSuffix: String = ""
    ) throws {
        let fm = FileManager.default
        let files = try fm.contentsOfDirectory(atPath: shaderDir)
            .filter { $0.hasSuffix(".metal") }
            .sorted()

        try compileMetalLibrary(
            shaderDir: shaderDir,
            shaderFiles: files,
            outputPath: outputPath,
            generatedLutMatvecSuffix: generatedLutMatvecSuffix
        )
    }

    /// Compile only an explicitly named Metal source closure. Component
    /// packages use this overload so their executable routing cannot change
    /// merely because an unrelated shader is added to Resources/Shaders.
    static func compileMetalLibrary(
        shaderDir: String,
        shaderFiles: [String],
        outputPath: String,
        generatedLutMatvecSuffix: String = ""
    ) throws {
        let fm = FileManager.default
        let files = shaderFiles.sorted()

        guard !files.isEmpty else {
            throw SmeltCompilerError.noShaders(shaderDir)
        }
        guard Set(files).count == files.count,
              files.allSatisfy({
                  $0.hasSuffix(".metal")
                      && !$0.contains("/")
                      && !$0.contains("..")
                      && fm.fileExists(atPath: "\(shaderDir)/\($0)")
              })
        else {
            throw SmeltCompilerError.unsupportedConfiguration(
                "named shader closure contains a duplicate, path, or missing file"
            )
        }

        let tempDir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent(
                "agent_metal_\(ProcessInfo.processInfo.globallyUniqueString)"
            )
        try fm.createDirectory(atPath: tempDir, withIntermediateDirectories: true)
        defer { try? fm.removeItem(atPath: tempDir) }
        let moduleCacheDir = "\(tempDir)/module-cache"
        try fm.createDirectory(atPath: moduleCacheDir, withIntermediateDirectories: true)

        var airFiles: [String] = []
        for file in files {
            let metalPath: String
            if file == "lut_matvec.metal", !generatedLutMatvecSuffix.isEmpty {
                metalPath = "\(tempDir)/\(file)"
                let base = try String(
                    contentsOfFile: "\(shaderDir)/\(file)",
                    encoding: .utf8
                )
                try (base + generatedLutMatvecSuffix).write(
                    toFile: metalPath,
                    atomically: true,
                    encoding: .utf8
                )
            } else {
                metalPath = "\(shaderDir)/\(file)"
            }
            let airPath = "\(tempDir)/\(file.replacingOccurrences(of: ".metal", with: ".air"))"

            var metalArguments = [
                "-sdk", "macosx", "metal",
                "-fmodules-cache-path=\(moduleCacheDir)",
                "-I", shaderDir,
            ]
            // Reference-compatible arithmetic is a shader-brick contract.
            // Safe math mode alone still permits contraction; MLX freezes
            // these operations with this exact compiler option.
            let preciseArithmetic = file.hasSuffix("_precise.metal")
            if preciseArithmetic
                || file == "attention.metal"
                || file == "conv1d.metal"
                || file == "norms.metal"
                || file == "prefill_attention.metal"
                || file == "prefill_recurrence.metal"
                || file == "recurrence.metal"
                || file == "signed_quant.metal"
            {
                metalArguments.append("-fno-fast-math")
            }
            if preciseArithmetic {
                metalArguments += [
                    "-fmetal-math-mode=safe",
                    "-fmetal-math-fp32-functions=precise",
                    "-ffp-contract=off",
                ]
            }
            metalArguments += ["-c", metalPath, "-o", airPath]
            let compileResult = try runMetalCompiler(
                "/usr/bin/xcrun",
                arguments: metalArguments
            )
            guard compileResult == 0 else {
                throw SmeltCompilerError.metalCompileFailed(file)
            }
            airFiles.append(airPath)
        }

        let linkResult = try runMetalCompiler(
            "/usr/bin/xcrun",
            arguments: ["-sdk", "macosx", "metallib"]
                + airFiles + ["-o", outputPath]
        )
        guard linkResult == 0 else {
            throw SmeltCompilerError.metallibLinkFailed
        }
    }

    @discardableResult
    private static func runMetalCompiler(
        _ executable: String,
        arguments: [String]
    ) throws -> Int32 {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        try process.run()
        process.waitUntilExit()
        return process.terminationStatus
    }
}
