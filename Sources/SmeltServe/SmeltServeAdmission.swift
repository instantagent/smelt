import Foundation
import SmeltRuntime
import SmeltSchema

/// The admitted serve route. Package access keeps the concrete construction
/// plans private to Smelt while allowing the `smelt` executable target to
/// dispatch text and audio adapters without loading or resolving twice.
package enum SmeltServeRuntimePlan: Sendable {
    case textToText(CAMTextRuntimeConstruction)
    case textToPCM(SmeltTextToPCMServeAdmission)
}

/// The route evidence needed by Smelt's text-to-PCM CLI adapter after the
/// shared serving layer has already completed package admission.
package struct SmeltTextToPCMServeAdmission: Sendable {
    package let packagePath: String
    package let runtimeRoute: CAMRuntimeRoute
    package let capabilities: SmeltCAMPackageCapabilities
    package let decision: SmeltCAMPackageCapabilities.Decision
}

/// Serve adapter over SmeltRuntime's single package admission authority.
package enum SmeltServeAdmission {
    package static func resolve(packagePath: String) throws -> SmeltServeRuntimePlan {
        let admission: SmeltRuntimeAdmission
        do {
            admission = try SmeltRuntimeAdmission.resolve(
                packagePath: packagePath,
                requests: [.serveText, .serveAudio, .serveAudioStream]
            )
        } catch SmeltRuntimeAdmissionError.noMatchingExport {
            throw SmeltServeError("no CAM export satisfies serve request")
        } catch {
            throw SmeltServeError("\(error)")
        }

        switch admission.runtimeRoute {
        case .textToText:
            do {
                return .textToText(try admission.makeTextConstruction())
            } catch {
                throw SmeltServeError("CAM text construction failed: \(error)")
            }
        case .textToPCM(let outputRate):
            guard outputRate == "24khz" else {
                throw SmeltServeError(
                    "unsupported CAM audio output rate '\(outputRate)'"
                )
            }
            return .textToPCM(SmeltTextToPCMServeAdmission(
                packagePath: admission.packagePath,
                runtimeRoute: admission.runtimeRoute,
                capabilities: admission.capabilities,
                decision: admission.decision
            ))
        }
    }
}
