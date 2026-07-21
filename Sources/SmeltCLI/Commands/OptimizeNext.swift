import Foundation

func runOptimizeNextCommand(_ args: [String]) {
    do {
        let status = try optimizeNextStatus(arguments: args)
        if status != 0 {
            exit(status)
        }
    } catch {
        fputs("Optimize-next failed: \(error)\n", stderr)
        exit(1)
    }
}
