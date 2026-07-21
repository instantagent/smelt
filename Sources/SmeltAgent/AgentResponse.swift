package struct AgentResponse: Sendable, Equatable {
    package let text: String
    package let tokenIDs: [Int32]
    package let promptTokenCount: Int
    package let prefillTime: Double
    package let generateTime: Double

    package init(
        text: String,
        tokenIDs: [Int32],
        promptTokenCount: Int,
        prefillTime: Double,
        generateTime: Double
    ) {
        self.text = text
        self.tokenIDs = tokenIDs
        self.promptTokenCount = promptTokenCount
        self.prefillTime = prefillTime
        self.generateTime = generateTime
    }
}
