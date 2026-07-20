import CryptoKit
import Foundation
import SmeltSchema

public struct SmeltVerifiedPackageFile: Sendable {
    public let name: String
    public let path: String
    public let expectedSHA256: String
    public let actualSHA256: String
}

public struct SmeltPackageIntegrityReport: Sendable {
    public let packagePath: String
    public let verifiedFiles: [SmeltVerifiedPackageFile]
    public let skippedFiles: [String]
    public let buildProvenance: SmeltBuildProvenance?
}

public enum SmeltPackageIntegrity {
    public static func verify(
        packagePath: String,
        includeWeights: Bool = true
    ) throws -> SmeltPackageIntegrityReport {
        let manifestPath = "\(packagePath)/manifest.json"
        let manifestData = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try SmeltManifest.decode(from: manifestData)
        return try verify(
            packagePath: packagePath,
            manifest: manifest,
            includeWeights: includeWeights
        )
    }

    static func verify(
        packagePath: String,
        manifest: SmeltManifest,
        includeWeights: Bool
    ) throws -> SmeltPackageIntegrityReport {
        var verifiedFiles: [SmeltVerifiedPackageFile] = []
        var skippedFiles: [String] = []

        try verifyFile(
            packagePath: packagePath,
            fileName: "weights.bin",
            expectedSHA256: manifest.checksums.weightsBin,
            include: includeWeights,
            verifiedFiles: &verifiedFiles,
            skippedFiles: &skippedFiles
        )
        try verifyFile(
            packagePath: packagePath,
            fileName: "model.metallib",
            expectedSHA256: manifest.checksums.metallib,
            include: true,
            verifiedFiles: &verifiedFiles,
            skippedFiles: &skippedFiles
        )
        try verifyFile(
            packagePath: packagePath,
            fileName: "SmeltGenerated.swift",
            expectedSHA256: manifest.checksums.generatedSwift,
            include: true,
            verifiedFiles: &verifiedFiles,
            skippedFiles: &skippedFiles
        )
        try verifyFile(
            packagePath: packagePath,
            fileName: "dispatches.bin",
            expectedSHA256: manifest.checksums.dispatchesBin,
            include: true,
            verifiedFiles: &verifiedFiles,
            skippedFiles: &skippedFiles
        )
        try verifyFile(
            packagePath: packagePath,
            fileName: "prefill_dispatches.bin",
            expectedSHA256: manifest.checksums.prefillDispatchesBin,
            include: true,
            verifiedFiles: &verifiedFiles,
            skippedFiles: &skippedFiles
        )
        try verifyFile(
            packagePath: packagePath,
            fileName: "prefill_verify_argmax_dispatches.bin",
            expectedSHA256: manifest.checksums.prefillVerifyArgmaxDispatchesBin,
            include: true,
            verifiedFiles: &verifiedFiles,
            skippedFiles: &skippedFiles
        )
        try verifyFile(
            packagePath: packagePath,
            fileName: "tokenizer.json",
            expectedSHA256: manifest.checksums.tokenizerJSON,
            include: true,
            verifiedFiles: &verifiedFiles,
            skippedFiles: &skippedFiles
        )

        return SmeltPackageIntegrityReport(
            packagePath: packagePath,
            verifiedFiles: verifiedFiles,
            skippedFiles: skippedFiles,
            buildProvenance: manifest.buildProvenance
        )
    }

    private static func verifyFile(
        packagePath: String,
        fileName: String,
        expectedSHA256: String?,
        include: Bool,
        verifiedFiles: inout [SmeltVerifiedPackageFile],
        skippedFiles: inout [String]
    ) throws {
        guard include else {
            skippedFiles.append(fileName)
            return
        }
        guard let expectedSHA256, !expectedSHA256.isEmpty else {
            skippedFiles.append(fileName)
            return
        }

        let path = "\(packagePath)/\(fileName)"
        guard FileManager.default.fileExists(atPath: path) else {
            throw SmeltRuntimeError.checksumMismatch(
                "\(fileName) missing but manifest expects checksum"
            )
        }

        let actualSHA256 = try sha256Hex(ofFileAt: path)
        guard actualSHA256 == expectedSHA256 else {
            throw SmeltRuntimeError.checksumMismatch(
                "\(fileName) expected \(expectedSHA256.prefix(12))..., got \(actualSHA256.prefix(12))..."
            )
        }

        verifiedFiles.append(
            SmeltVerifiedPackageFile(
                name: fileName,
                path: path,
                expectedSHA256: expectedSHA256,
                actualSHA256: actualSHA256
            )
        )
    }

    private static func sha256Hex(ofFileAt path: String) throws -> String {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }

        var hasher = SHA256()
        while true {
            let chunk = try handle.read(upToCount: 1 << 20) ?? Data()
            if chunk.isEmpty {
                break
            }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}
