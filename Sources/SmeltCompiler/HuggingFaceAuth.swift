import Foundation

enum HuggingFaceAuth {
    static func applyAuthorization(to request: inout URLRequest) {
        guard let token = resolveToken() else { return }
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
    }

    static func resolveToken() -> String? {
        let env = ProcessInfo.processInfo.environment
        for key in ["HF_TOKEN", "HF_HUB_TOKEN"] {
            if let token = cleanToken(env[key]) {
                return token
            }
        }

        if let path = env["HF_TOKEN_PATH"],
           let token = tokenFromFile(path)
        {
            return token
        }

        let hfHome = env["HF_HOME"] ?? "\(homeDirectory())/.cache/huggingface"
        for path in [
            "\(hfHome)/token",
            "\(hfHome)/stored_tokens",
            "\(homeDirectory())/.huggingface/token",
        ] {
            if let token = tokenFromFile(path) {
                return token
            }
        }

        return tokenFromGitCredentialHelper()
    }

    private static func homeDirectory() -> String {
        FileManager.default.homeDirectoryForCurrentUser.path
    }

    private static func cleanToken(_ raw: String?) -> String? {
        guard let raw else { return nil }
        let token = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        return token.isEmpty ? nil : token
    }

    private static func tokenFromFile(_ path: String) -> String? {
        guard let data = try? Data(contentsOf: URL(fileURLWithPath: path)),
              let content = String(data: data, encoding: .utf8)
        else { return nil }

        for line in content.split(whereSeparator: \.isNewline) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if let token = cleanToken(String(trimmed)), token.hasPrefix("hf_") {
                return token
            }
            if trimmed.hasPrefix("hf_token") || trimmed.hasPrefix("token") {
                let parts = trimmed.split(separator: "=", maxSplits: 1).map(String.init)
                if parts.count == 2, let token = cleanToken(parts[1]), token.hasPrefix("hf_") {
                    return token
                }
            }
        }

        return nil
    }

    private static func tokenFromGitCredentialHelper() -> String? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = ["credential", "fill"]

        let input = Pipe()
        let output = Pipe()
        process.standardInput = input
        process.standardOutput = output
        process.standardError = Pipe()

        do {
            try process.run()
            input.fileHandleForWriting.write(
                Data("protocol=https\nhost=huggingface.co\n\n".utf8)
            )
            input.fileHandleForWriting.closeFile()
            process.waitUntilExit()
        } catch {
            return nil
        }

        guard process.terminationStatus == 0 else { return nil }
        let data = output.fileHandleForReading.readDataToEndOfFile()
        guard let content = String(data: data, encoding: .utf8) else { return nil }
        for line in content.split(whereSeparator: \.isNewline) {
            let parts = line.split(separator: "=", maxSplits: 1).map(String.init)
            guard parts.count == 2, parts[0] == "password" else { continue }
            return cleanToken(parts[1])
        }
        return nil
    }
}
