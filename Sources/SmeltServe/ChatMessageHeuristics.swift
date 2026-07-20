import Foundation

// OpenAI-compatible request heuristics shared by every SmeltServe consumer.

/// Walk messages in reverse to find the first non-system entry's
/// role. Used by the auto-mode post-tool-result fallback: when
/// the last non-system message is a tool result, current IT
/// models can't reliably navigate the union grammar, so we drop
/// tools for that turn and force a free-text answer.
func lastNonSystemRoleIsTool(
    _ messages: [OpenAIChatMessage]
) -> Bool {
    for message in messages.reversed() {
        if message.role == .system { continue }
        return message.role == .tool
    }
    return false
}

/// True if the request has tools in this turn OR any tool_calls
/// / tool_result entry in history. Used to bypass the prefix
/// cache for tool-involving requests.
///
/// Deliberately broader than "current request has tools":
/// The prompt-state cache matches by token-id LCP, so a changed tools
/// system-message block is itself token-visible and stops the
/// match at the first differing token — that case is already
/// safe. The unsafe case is post-tool / tool-history prompts that
/// still share a long valid token prefix with an older non-tool
/// entry; the partial-restore + suffix-prefill path produces
/// degenerate output there (Pi gate setup + tool-torture reproduce
/// it). "Current turn has tools" is too narrow — it would re-include
/// the tool-history-but-no-active-tools turn that triggers this.
func requestInvolvesTools(
    _ request: OpenAIChatCompletionsRequest
) -> Bool {
    if let tools = request.tools, !tools.isEmpty { return true }
    for message in request.messages {
        if message.role == .tool { return true }
        if let calls = message.toolCalls, !calls.isEmpty { return true }
    }
    return false
}

/// Return the previous assistant answer for Pi-style resume
/// requests: the same user prompt appears again after a
/// completed assistant text answer in a tool-involving
/// conversation. Used by both the tool-disable heuristic (drop
/// descriptors when this is non-nil) and the deterministic
/// replay path (return the cached text instead of regenerating).
///
/// Intentionally narrower than a semantic cache — requires
/// byte-for-byte prompt equality after whitespace trimming. Pi's
/// session-resume flow re-sends the original prompt verbatim;
/// without this short-circuit an instruct model under union grammar
/// often emits a degenerate single token (`}` is most common;
/// the text-arm regex `\s*[^{\s]...` accepts it).
func repeatedAnsweredQuestionAnswer(
    _ messages: [OpenAIChatMessage]
) -> String? {
    guard messages.last?.role == .user else { return nil }
    let lastIdx = messages.count - 1
    let lastText = (messages[lastIdx].content ?? "")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    guard !lastText.isEmpty else { return nil }
    for earlierIdx in stride(from: lastIdx - 1, through: 0, by: -1) {
        let earlier = messages[earlierIdx]
        guard earlier.role == .user else { continue }
        let earlierText = (earlier.content ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard earlierText == lastText else { continue }
        var answer: String? = nil
        for innerIdx in (earlierIdx + 1)..<lastIdx {
            let m = messages[innerIdx]
            if m.role == .assistant,
               let content = m.content,
               !content.trimmingCharacters(
                    in: .whitespacesAndNewlines
               ).isEmpty
            {
                answer = content.trimmingCharacters(
                    in: .whitespacesAndNewlines
                )
            }
        }
        if let answer { return answer }
    }
    return nil
}
