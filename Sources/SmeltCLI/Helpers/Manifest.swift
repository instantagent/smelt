import Foundation
import SmeltRuntime
import SmeltSchema

func loadInferenceConfig(
    packagePath: String
) throws -> (manifest: SmeltManifest, inference: SmeltInferenceManifest) {
    let manifestPath = "\(packagePath)/manifest.json"
    let data = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
    let manifest = try SmeltManifest.decode(from: data)
    return (manifest, try manifest.resolvedInferencePolicy().inference)
}

func loadTraceMarkers(packagePath: String) throws -> SmeltTraceMarkers {
    let path = "\(packagePath)/trace_markers.json"
    let data = try Data(contentsOf: URL(fileURLWithPath: path))
    return try JSONDecoder().decode(SmeltTraceMarkers.self, from: data)
}
