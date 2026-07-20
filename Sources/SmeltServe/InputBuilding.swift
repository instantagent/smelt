import Foundation
import SmeltRuntime
import SmeltSchema

// Prompt construction shared by direct runtime and serving adapters.

enum InputBuildingError: Error, CustomStringConvertible {
    case unknownTemplate(String)
    case missingChatTemplateTokens(template: String)
    case toolDescriptorMissingName
    case toolDescriptorMissingNameAt(index: Int)
    case toolDescriptorMissingSchema(name: String)
    case toolDescriptorInvalidSchema(name: String)
    case toolsFileReadFailed(path: String, underlying: Error)
    case toolsFileInvalidJSON(path: String, underlying: Error)
    case toolMessageMissingIdentifier
    case systemMessageMustBeFirst

    var description: String {
        switch self {
        case .unknownTemplate(let name):
            return "Unknown template: \(name). Available: \(SmeltPromptTemplateName.availablePromptTemplates)"
        case .missingChatTemplateTokens(let template):
            return "Tokenizer is missing \(template) chat special tokens"
        case .toolDescriptorMissingName:
            return "Tool descriptor is missing a non-empty name"
        case .toolDescriptorMissingNameAt(let index):
            return "Tool descriptor \(index) is missing a non-empty name"
        case .toolDescriptorMissingSchema(let name):
            return "Tool descriptor \(name) is missing schema, parameters, or schemaJSON"
        case .toolDescriptorInvalidSchema(let name):
            return "Tool descriptor \(name) has an invalid JSON schema object"
        case .toolsFileReadFailed(let path, let underlying):
            return "Could not read tools file \(path): \(underlying)"
        case .toolsFileInvalidJSON(let path, let underlying):
            return "Tools file \(path) is not valid JSON: \(underlying)"
        case .toolMessageMissingIdentifier:
            return "Tool message must include `tool_call_id` (referencing a prior assistant tool_call) or `name`"
        case .systemMessageMustBeFirst:
            return "System message must be the first chat message"
        }
    }
}

package func buildInputIds(
    prompt: String,
    tokenizer: SmeltTokenizer,
    template: String,
    thinkingPolicy: SmeltThinkingPolicy = .disabled
) throws -> [Int32] {
    if prompt.isEmpty {
        return [Int32(tokenizer.bosTokenId ?? 1)]
    }

    switch template {
    case "":
        var ids = tokenizer.encode(prompt)
        if let bos = tokenizer.bosTokenId {
            ids.insert(Int32(bos), at: 0)
        }
        return ids

    case SmeltPromptTemplateName.headerTurns:
        let bos: Int32 = 128_000
        let startHeader: Int32 = 128_006
        let endHeader: Int32 = 128_007
        let eotId: Int32 = 128_009
        var ids: [Int32] = [bos, startHeader]
        ids += tokenizer.encode("user")
        ids += [endHeader]
        ids += tokenizer.encode("\n\n")
        ids += tokenizer.encode(prompt)
        ids += [eotId, startHeader]
        ids += tokenizer.encode("assistant")
        ids += [endHeader]
        ids += tokenizer.encode("\n\n")
        return ids

    case SmeltPromptTemplateName.channelTurns:
        let bos: Int32 = 2
        let turnStart: Int32 = 105
        let turnEnd: Int32 = 106
        let channelStart: Int32 = 100
        let channelEnd: Int32 = 101
        let newline: Int32 = 107
        var ids: [Int32] = [bos, turnStart]
        ids += tokenizer.encode("user")
        ids += [newline]
        ids += tokenizer.encode(prompt)
        ids += [turnEnd, newline, turnStart]
        ids += tokenizer.encode("model")
        ids += [newline, channelStart]
        ids += tokenizer.encode("thought")
        ids += [newline, channelEnd]
        return ids

    case SmeltPromptTemplateName.chatML,
         SmeltPromptTemplateName.chatMLXMLTools:
        guard let imStart = tokenizer.addedTokenId(for: "<|im_start|>"),
              let imEnd = tokenizer.addedTokenId(for: "<|im_end|>")
        else {
            throw InputBuildingError.missingChatTemplateTokens(template: "chatml")
        }
        var ids: [Int32] = [Int32(imStart)]
        ids += tokenizer.encode("user")
        ids += tokenizer.encode("\n")
        ids += tokenizer.encode(prompt)
        ids += [Int32(imEnd)]
        ids += tokenizer.encode("\n")
        ids += try chatMLAssistantPrelude(thinkingPolicy: thinkingPolicy, tokenizer: tokenizer)
        return ids

    default:
        throw InputBuildingError.unknownTemplate(template)
    }
}

typealias BakedSystemPromptContinuationPlan = SmeltBakedSystemPromptContinuationPlan

func bakedSystemPromptContinuationPlan(
    tokenizer: SmeltTokenizer,
    template: String,
    thinkingPolicy: SmeltThinkingPolicy
) throws -> BakedSystemPromptContinuationPlan? {
    do {
        return try SmeltBakedPromptContinuationBuilder.systemPromptPlan(
            tokenizer: tokenizer,
            template: template,
            thinkingPolicy: thinkingPolicy
        )
    } catch SmeltPromptTemplateError.missingChatTemplateTokens(let template) {
        throw InputBuildingError.missingChatTemplateTokens(template: template)
    }
}

func buildInputIdsFromBakedContinuation(
    prompt: String,
    tokenizer: SmeltTokenizer,
    bakedPrefixTokenIds: [Int32],
    continuation: SmeltBakedPromptContinuation,
    template: String,
    thinkingPolicy: SmeltThinkingPolicy
) -> [Int32]? {
    SmeltBakedPromptContinuationBuilder.inputIds(
        prompt: prompt,
        tokenizer: tokenizer,
        bakedPrefixTokenIds: bakedPrefixTokenIds,
        continuation: continuation,
        template: template,
        thinkingPolicy: thinkingPolicy
    )
}

package func buildInputIdsApplyingBakedPrefix(
    prompt: String,
    tokenizer: SmeltTokenizer,
    unbakedInputIds: [Int32],
    bakedPrefixTokenIds: [Int32],
    continuation: SmeltBakedPromptContinuation?,
    template: String,
    thinkingPolicy: SmeltThinkingPolicy
) -> [Int32] {
    SmeltBakedPromptContinuationBuilder.inputIds(
        prompt: prompt,
        tokenizer: tokenizer,
        bakedPrefixTokenIds: bakedPrefixTokenIds,
        continuation: continuation,
        template: template,
        thinkingPolicy: thinkingPolicy,
        unbakedInputIds: unbakedInputIds
    )
}

/// ChatML assistant-generation prelude: opens the assistant turn and, when
/// thinking is disabled, pre-closes the `<think>` channel so generation starts
/// after `</think>`. Shared by `buildInputIds` (run) and
/// `buildChatCompletionsInputIds` (serve) so the two paths can't drift.
private func chatMLAssistantPrelude(
    thinkingPolicy: SmeltThinkingPolicy,
    tokenizer: SmeltTokenizer
) throws -> [Int32] {
    guard let imStart = tokenizer.addedTokenId(for: "<|im_start|>") else {
        throw InputBuildingError.missingChatTemplateTokens(template: "chatml")
    }
    var ids: [Int32] = [Int32(imStart)]
    ids += tokenizer.encode("assistant")
    ids += tokenizer.encode("\n")
    guard thinkingPolicy == .disabled else { return ids }
    guard let think = tokenizer.addedTokenId(for: "<think>"),
          let thinkEnd = tokenizer.addedTokenId(for: "</think>")
    else {
        throw InputBuildingError.missingChatTemplateTokens(template: "chatml")
    }
    ids += [Int32(think)]
    ids += tokenizer.encode("\n\n")
    ids += [Int32(thinkEnd)]
    ids += tokenizer.encode("\n\n")
    return ids
}

/// Resolve the chat-template name from explicit config only. An empty result
/// means raw concatenation, not model-name autodetection.
package func resolveChatTemplate(
    cliOverride: String?,
    packageTemplate: String?
) -> String {
    if let cliOverride, !cliOverride.isEmpty {
        return cliOverride
    }
    if let packageTemplate, !packageTemplate.isEmpty {
        return packageTemplate
    }
    return ""
}

/// The package's thinking policy, defaulting to `.disabled` when unset — the
/// single place that default lives, so run and serve can't drift on it.
package func resolvedThinkingPolicy(_ inference: SmeltInferenceManifest) -> SmeltThinkingPolicy {
    inference.thinkingPolicy ?? .disabled
}

/// Build prompt token IDs for OpenAI chat-completions multi-message
/// requests. The ChatML assistant prelude honors `thinkingPolicy` via the same
/// `chatMLAssistantPrelude` helper as `buildInputIds`, so `serve` and `run` render
/// identical prompts: `.disabled` pre-closes the `<think>` channel in the prompt
/// (rather than relying on the runtime's after-the-fact think-skip, which still
/// samples the open-thinking distribution and is unreliable on quantized models).
func buildChatCompletionsInputIds(
    messages: [OpenAIChatMessage],
    tokenizer: SmeltTokenizer,
    template: String,
    thinkingPolicy: SmeltThinkingPolicy = .disabled,
    tools: [SmeltToolDescriptor]? = nil,
    toolTranscriptCodec: SmeltNativeToolTranscriptCodec? = nil
) throws -> [Int32] {
    guard !messages.isEmpty else {
        return try buildInputIds(prompt: "", tokenizer: tokenizer, template: template)
    }

    // The OpenAI wire format separates assistant tool calls (content=null,
    // tool_calls=[...]) from tool results (role="tool", tool_call_id, content).
    // Our chat templates carry only text inside model/user turns, so each
    // template renders those structured fields inline as text the model can
    // see — otherwise turn-2 looks like a fresh user message and the model
    // re-calls the tool indefinitely.
    var toolCallNames: [String: String] = [:]

    let legacyNativeCodec = template == SmeltPromptTemplateName.chatMLXMLTools
        ? try SmeltNativeToolTranscriptCodec.resolve(
            SmeltToolTranscriptCodecName.xmlFunctionParameters
        ) : nil
    let nativeCodec = toolTranscriptCodec ?? legacyNativeCodec
    let promptTemplate = template == SmeltPromptTemplateName.chatMLXMLTools
        ? SmeltPromptTemplateName.chatML : template

    switch promptTemplate {
    case "":
        let transcript = messages
            .map { "\($0.role.rawValue):\n\(renderToolAwareBody($0, toolCallNames: &toolCallNames))" }
            .joined(separator: "\n\n")
        return try buildInputIds(prompt: transcript, tokenizer: tokenizer, template: template)

    case SmeltPromptTemplateName.headerTurns:
        let bos: Int32 = 128_000
        let startHeader: Int32 = 128_006
        let endHeader: Int32 = 128_007
        let eotId: Int32 = 128_009
        var ids: [Int32] = [bos]
        for message in messages {
            ids += [startHeader]
            ids += tokenizer.encode(llamaRoleName(message.role))
            ids += [endHeader]
            ids += tokenizer.encode("\n\n")
            ids += tokenizer.encode(renderToolAwareBody(message, toolCallNames: &toolCallNames))
            ids += [eotId]
        }
        ids += [startHeader]
        ids += tokenizer.encode("assistant")
        ids += [endHeader]
        ids += tokenizer.encode("\n\n")
        return ids

    case SmeltPromptTemplateName.channelTurns:
        // Tool history uses the model's native special tokens. Tool
        // results must be wrapped in the native response:name{...}
        // envelope; otherwise raw JSON file bytes sit directly next
        // to the assistant continuation and bias resume turns toward
        // punctuation.
        //
        // Token IDs come from tokenizer.addedTokenId(for:) lookups
        // so a model variant that renumbers added-tokens-decoder
        // entries fails loudly at request time instead of silently
        // encoding garbage IDs.
        let bos: Int32 = 2
        guard let turnStartRaw = tokenizer.addedTokenId(for: "<|turn>"),
              let turnEndRaw = tokenizer.addedTokenId(for: "<turn|>"),
              let toolCallStartRaw = tokenizer.addedTokenId(for: "<|tool_call>"),
              let toolCallEndRaw = tokenizer.addedTokenId(for: "<tool_call|>"),
              let toolResponseStartRaw = tokenizer.addedTokenId(for: "<|tool_response>"),
              let toolResponseEndRaw = tokenizer.addedTokenId(for: "<tool_response|>"),
              let quoteRaw = tokenizer.addedTokenId(for: "<|\"|>")
        else {
            throw InputBuildingError.missingChatTemplateTokens(
                template: SmeltPromptTemplateName.channelTurns)
        }
        let turnStart = Int32(turnStartRaw)
        let turnEnd = Int32(turnEndRaw)
        let toolCallStart = Int32(toolCallStartRaw)
        let toolCallEnd = Int32(toolCallEndRaw)
        let toolResponseStart = Int32(toolResponseStartRaw)
        let toolResponseEnd = Int32(toolResponseEndRaw)
        let quote = Int32(quoteRaw)
        // Newline is the single-byte literal "\n" in the BPE vocab,
        // not an added-tokens-decoder entry; encoding gives token 107
        // for the canonical tokenizer. Keeping it hardcoded
        // mirrors the existing pattern for BPE-level constants.
        let newline: Int32 = 107
        var ids: [Int32] = [bos]
        var systemPrefix = ""
        // The model's native template maps OpenAI's split
        // assistant{tool_calls} + role:tool + assistant{content}
        // sequence back into one model turn. The tool response is
        // not raw bytes: it is wrapped as
        // `response:name{value:<|"|>...<|"|>}`, which keeps JSON file
        // contents from looking like the assistant's next-token
        // continuation. When the request ends with a tool result, the
        // canonical template leaves that model turn open and
        // generation resumes immediately after `<tool_response|>`.
        var modelTurnOpen = false
        var awaitingAnswerAfterToolResponse = false
        func closeModelTurnIfOpen() {
            if modelTurnOpen {
                ids += [turnEnd, newline]
                modelTurnOpen = false
            }
        }
        func openModelTurn() {
            ids += [turnStart]
            ids += tokenizer.encode("model")
            ids += [newline]
            modelTurnOpen = true
        }
        // System content is normally folded into the next user turn
        // (this template has no `<|turn>system` header). If the first
        // non-system message is an assistant or tool message — common
        // in Pi resume flows where the conversation begins with a
        // tool exchange — there's no user turn to fold the systems
        // into, and the prefix would be silently dropped. Emit a
        // synthetic user turn carrying the system content so the
        // caller's instructions reach the model regardless of the
        // conversation shape.
        func flushSystemPrefixAsUserTurnIfNeeded() {
            guard !systemPrefix.isEmpty else { return }
            closeModelTurnIfOpen()
            ids += [turnStart]
            ids += tokenizer.encode("user")
            ids += [newline]
            ids += tokenizer.encode(systemPrefix)
            ids += [turnEnd, newline]
            systemPrefix = ""
        }
        for message in messages {
            if message.role == .system {
                if !systemPrefix.isEmpty { systemPrefix += "\n\n" }
                systemPrefix += message.content ?? ""
                continue
            }
            if message.role == .tool {
                // Tool response continues the open model turn — do
                // NOT open a new user turn. Defensive: if no model
                // turn is open (history starts with a tool message,
                // shouldn't happen but possible), open one.
                // Flush pending systemPrefix BEFORE the assistant
                // turn so the system instructions land as proper
                // conversation context, not as a synthetic user turn
                // appended AFTER the tool exchange.
                flushSystemPrefixAsUserTurnIfNeeded()
                if !modelTurnOpen { openModelTurn() }
                // The model's trained `response:NAME{value:...}`
                // envelope hardcodes the tool name; we previously
                // fell back to the literal string "unknown" when
                // both tool_call_id and name were missing, but
                // `response:unknown{...}` is a sequence the model
                // was never trained on — its next-token
                // distribution there is undefined. Require at
                // least one identifier and 400 the request when
                // neither is present.
                let resolvedName = message.toolCallId
                    .flatMap { toolCallNames[$0] }
                    ?? message.name
                guard let toolName = resolvedName, !toolName.isEmpty else {
                    throw InputBuildingError.toolMessageMissingIdentifier
                }
                ids += [toolResponseStart]
                ids += tokenizer.encode("response:\(toolName){value:")
                ids += [quote]
                ids += tokenizer.encode(message.content ?? "")
                ids += [quote]
                ids += tokenizer.encode("}")
                ids += [toolResponseEnd]
                awaitingAnswerAfterToolResponse = true
                continue
            }
            if message.role == .assistant {
                flushSystemPrefixAsUserTurnIfNeeded()
                if !modelTurnOpen { openModelTurn() }
                // Per response_schema: tool_calls come BEFORE content
                // within an assistant turn. OpenAI's wire shape allows
                // both fields on the same message; we emit tool_calls
                // first to match the trained order.
                if let toolCalls = message.toolCalls, !toolCalls.isEmpty {
                    for call in toolCalls {
                        // Pass raw args through unless empty or literal
                        // "null" (OpenAI's no-arg-tool shape); whitespace-
                        // prefixed or pretty-printed args stay as-is.
                        let raw = call.function.arguments
                        let argsText: String = (raw.isEmpty || raw == "null")
                            ? "{}" : raw
                        ids += [toolCallStart]
                        ids += tokenizer.encode("call:\(call.function.name)\(argsText)")
                        ids += [toolCallEnd]
                        toolCallNames[call.id] = call.function.name
                    }
                }
                if let content = message.content, !content.isEmpty {
                    ids += tokenizer.encode(content)
                }
                // An assistant turn with content (final answer)
                // closes the model turn, even if OpenAI carried
                // tool_calls and content on the same message. An
                // assistant turn with ONLY tool_calls stays open — a
                // tool_response is expected to follow.
                let hasToolCalls = !(message.toolCalls?.isEmpty ?? true)
                let hasContent = !(message.content?.isEmpty ?? true)
                if hasContent || !hasToolCalls {
                    closeModelTurnIfOpen()
                }
                awaitingAnswerAfterToolResponse = false
                continue
            }
            // role == .user (or other) — close any open model turn
            // before opening a new user block.
            closeModelTurnIfOpen()
            awaitingAnswerAfterToolResponse = false
            ids += [turnStart]
            ids += tokenizer.encode("user")
            ids += [newline]
            // Single encode preserves BPE merges across the system-
            // prefix/body boundary.
            var body = message.content ?? ""
            if message.role == .user && !systemPrefix.isEmpty {
                body = "\(systemPrefix)\n\n\(body)"
                systemPrefix = ""
            }
            ids += tokenizer.encode(body)
            ids += [turnEnd, newline]
        }
        if awaitingAnswerAfterToolResponse && modelTurnOpen {
            // systemPrefix is empty here — flushed earlier by the
            // first assistant/tool branch via
            // flushSystemPrefixAsUserTurnIfNeeded, which keeps the
            // canonical resume-mid-turn shape intact (assistant
            // continues right after `<tool_response|>`).
            return ids
        }
        // Plain priming for the next assistant generation. If
        // systemPrefix is still non-empty (conversation has no user
        // message at all), splice it as a synthetic user turn before
        // the assistant prime so the system instructions reach the
        // model.
        flushSystemPrefixAsUserTurnIfNeeded()
        closeModelTurnIfOpen()
        ids += [turnStart]
        ids += tokenizer.encode("model")
        ids += [newline]
        return ids

    case SmeltPromptTemplateName.chatML:
        if let nativeCodec {
            return try buildChatMLNativeToolInputIds(
                messages: messages,
                tools: tools ?? [],
                tokenizer: tokenizer,
                thinkingPolicy: thinkingPolicy,
                codec: nativeCodec
            )
        }
        guard let imStart = tokenizer.addedTokenId(for: "<|im_start|>"),
              let imEnd = tokenizer.addedTokenId(for: "<|im_end|>")
        else {
            throw InputBuildingError.missingChatTemplateTokens(template: "chatml")
        }
        var ids: [Int32] = []
        for message in messages {
            ids += [Int32(imStart)]
            ids += tokenizer.encode(chatMLRoleName(message.role))
            ids += tokenizer.encode("\n")
            ids += tokenizer.encode(renderToolAwareBody(message, toolCallNames: &toolCallNames))
            ids += [Int32(imEnd)]
            ids += tokenizer.encode("\n")
        }
        ids += try chatMLAssistantPrelude(thinkingPolicy: thinkingPolicy, tokenizer: tokenizer)
        return ids

    default:
        throw InputBuildingError.unknownTemplate(promptTemplate)
    }
}

/// Package-owned ChatML transcript with a native tool codec.
///
/// This is deliberately selected by the package's prompt-format capability,
/// never by model or repository name. The transcript mirrors the pinned native
/// template's role, tool-call, and tool-response shapes. Prior assistant turns
/// retain an explicit empty thinking block when thinking is disabled; that is
/// the template's `preserve_thinking=true` mode and gives a stable token prefix
/// across turns, which in turn makes exact recurrent-state checkpoints safe.
private func buildChatMLNativeToolInputIds(
    messages: [OpenAIChatMessage],
    tools: [SmeltToolDescriptor],
    tokenizer: SmeltTokenizer,
    thinkingPolicy: SmeltThinkingPolicy,
    codec: SmeltNativeToolTranscriptCodec
) throws -> [Int32] {
    if messages.dropFirst().contains(where: { $0.role == .system }) {
        throw InputBuildingError.systemMessageMustBeFirst
    }
    guard let imStartRaw = tokenizer.addedTokenId(for: "<|im_start|>"),
          let imEndRaw = tokenizer.addedTokenId(for: "<|im_end|>"),
          let toolResponseStartRaw = tokenizer.addedTokenId(
              for: "<tool_response>"
          ),
          let toolResponseEndRaw = tokenizer.addedTokenId(
              for: "</tool_response>"
          )
    else {
        throw InputBuildingError.missingChatTemplateTokens(
            template: codec.name
        )
    }
    let imStart = Int32(imStartRaw)
    let imEnd = Int32(imEndRaw)
    let callerSystem = messages.first(where: { $0.role == .system })?
        .content?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    var ids: [Int32] = []

    if !tools.isEmpty {
        var body = try codec.renderSystemMessage(
            tools: tools
        )
        if !callerSystem.isEmpty { body += "\n\n" + callerSystem }
        ids += [imStart]
        ids += tokenizer.encodeWithSpecials("system\n" + body)
        ids += [imEnd]
        ids += tokenizer.encode("\n")
    } else if !callerSystem.isEmpty {
        ids += [imStart]
        ids += tokenizer.encode("system\n" + callerSystem)
        ids += [imEnd]
        ids += tokenizer.encode("\n")
    }

    for (index, message) in messages.enumerated() {
        switch message.role {
        case .system:
            continue
        case .user:
            ids += [imStart]
            ids += tokenizer.encode(
                "user\n" + (message.content ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines)
            )
            ids += [imEnd]
            ids += tokenizer.encode("\n")
        case .assistant:
            ids += [imStart]
            ids += tokenizer.encode("assistant\n")
            if thinkingPolicy == .disabled {
                guard let think = tokenizer.addedTokenId(for: "<think>"),
                      let thinkEnd = tokenizer.addedTokenId(for: "</think>")
                else {
                    throw InputBuildingError.missingChatTemplateTokens(
                        template: codec.name
                    )
                }
                ids += [Int32(think)]
                ids += tokenizer.encode("\n\n")
                ids += [Int32(thinkEnd)]
                ids += tokenizer.encode("\n\n")
            }
            let content = (message.content ?? "")
                .trimmingCharacters(in: .whitespacesAndNewlines)
            if !content.isEmpty { ids += tokenizer.encode(content) }
            if let calls = message.toolCalls, !calls.isEmpty {
                if !content.isEmpty { ids += tokenizer.encode("\n\n") }
                ids += tokenizer.encodeWithSpecials(
                    try codec.renderCalls(
                    calls.map {
                        SmeltDecodedXMLToolCall(
                            name: $0.function.name,
                            argumentsJSON: $0.function.arguments
                        )
                    })
                )
            }
            ids += [imEnd]
            ids += tokenizer.encode("\n")
        case .tool:
            let previousIsTool = index > 0 && messages[index - 1].role == .tool
            let nextIsTool = index + 1 < messages.count
                && messages[index + 1].role == .tool
            if !previousIsTool {
                ids += [imStart]
                ids += tokenizer.encode("user")
            }
            ids += tokenizer.encode("\n")
            ids += [Int32(toolResponseStartRaw)]
            ids += tokenizer.encode(
                "\n" + (message.content ?? "")
                    .trimmingCharacters(in: .whitespacesAndNewlines) + "\n"
            )
            ids += [Int32(toolResponseEndRaw)]
            if !nextIsTool {
                ids += [imEnd]
                ids += tokenizer.encode("\n")
            }
        }
    }
    ids += try chatMLAssistantPrelude(
        thinkingPolicy: thinkingPolicy,
        tokenizer: tokenizer
    )
    return ids
}

/// Apply the package's sealed persona to OpenAI chat history. A single user
/// turn is intentionally routed through the same continuation builder as
/// `smelt run --once`, making first-turn token IDs identical. ChatML history then
/// extends that exact prefix for later interactive turns.
func buildChatCompletionsInputIdsApplyingBakedPrefix(
    messages: [OpenAIChatMessage],
    tokenizer: SmeltTokenizer,
    template: String,
    thinkingPolicy: SmeltThinkingPolicy,
    bakedPrefix: SmeltBakedPromptPrefix?,
    tools: [SmeltToolDescriptor]? = nil,
    toolTranscriptCodec: SmeltNativeToolTranscriptCodec? = nil
) throws -> [Int32] {
    let messages = messages.filter {
        !($0.role == .system && ($0.content ?? "").isEmpty)
    }
    let unbaked = try buildChatCompletionsInputIds(
        messages: messages,
        tokenizer: tokenizer,
        template: template,
        thinkingPolicy: thinkingPolicy,
        tools: tools,
        toolTranscriptCodec: toolTranscriptCodec
    )
    guard let bakedPrefix else { return unbaked }
    // A real caller-supplied system prompt is an explicit persona override,
    // exactly like `smelt run --system`; do not silently prepend the sealed one.
    guard !messages.contains(where: {
        $0.role == .system && !($0.content ?? "").isEmpty
    }) else {
        return unbaked
    }

    if messages.count == 1,
       let first = messages.first,
       first.role == .user {
        return buildInputIdsApplyingBakedPrefix(
            prompt: first.content ?? "",
            tokenizer: tokenizer,
            unbakedInputIds: unbaked,
            bakedPrefixTokenIds: bakedPrefix.tokenIds,
            continuation: bakedPrefix.continuation,
            template: template,
            thinkingPolicy: thinkingPolicy
        )
    }

    guard template == SmeltPromptTemplateName.chatML,
          toolTranscriptCodec == nil,
          let continuation = bakedPrefix.continuation,
          continuation.matches(
            template: template,
            thinkingPolicy: thinkingPolicy
          ),
          messages.first?.role == .user,
          let imStart = tokenizer.addedTokenId(for: "<|im_start|>"),
          let imEnd = tokenizer.addedTokenId(for: "<|im_end|>")
    else {
        // A package carrying continuation metadata must match exactly. This is
        // the same safe fallback used by one-shot run on template drift.
        return bakedPrefix.continuation == nil
            ? bakedPrefix.tokenIds + unbaked
            : unbaked
    }

    var toolCallNames: [String: String] = [:]
    var ids = bakedPrefix.tokenIds
    ids += tokenizer.encode(renderToolAwareBody(
        messages[0],
        toolCallNames: &toolCallNames
    ))
    ids += [Int32(imEnd)]
    ids += tokenizer.encode("\n")
    for message in messages.dropFirst() {
        ids += [Int32(imStart)]
        ids += tokenizer.encode(chatMLRoleName(message.role))
        ids += tokenizer.encode("\n")
        ids += tokenizer.encode(renderToolAwareBody(
            message,
            toolCallNames: &toolCallNames
        ))
        ids += [Int32(imEnd)]
        ids += tokenizer.encode("\n")
    }
    ids += try chatMLAssistantPrelude(
        thinkingPolicy: thinkingPolicy,
        tokenizer: tokenizer
    )
    return ids
}

/// Returns the message body to render inside the template's role turn,
/// inlining assistant tool calls and tool-result markers so the model
/// sees the tool-use trace. `toolCallNames` is threaded across messages
/// so tool-result turns can reference the function name from the
/// preceding assistant call.
private func renderToolAwareBody(
    _ message: OpenAIChatMessage,
    toolCallNames: inout [String: String]
) -> String {
    var body = message.content ?? ""
    if message.role == .assistant, let toolCalls = message.toolCalls, !toolCalls.isEmpty {
        if !body.isEmpty { body += "\n\n" }
        body += toolCalls.map { call in
            "[tool_call] \(call.function.name)(\(call.function.arguments))"
        }.joined(separator: "\n")
        for call in toolCalls { toolCallNames[call.id] = call.function.name }
    }
    if message.role == .tool {
        let name = message.toolCallId.flatMap { toolCallNames[$0] } ?? "tool"
        body = "[tool_result for \(name)]\n\(body)"
    }
    return body
}

private func llamaRoleName(_ role: OpenAIRole) -> String {
    switch role {
    case .assistant: return "assistant"
    case .system:    return "system"
    case .tool:      return "ipython"
    case .user:      return "user"
    }
}

private func chatMLRoleName(_ role: OpenAIRole) -> String {
    switch role {
    case .assistant: return "assistant"
    case .system:    return "system"
    case .tool:      return "tool"
    case .user:      return "user"
    }
}

package func buildSystemIds(
    systemPrompt: String,
    tokenizer: SmeltTokenizer,
    template: String
) throws -> [Int32] {
    guard !systemPrompt.isEmpty else { return [] }

    switch template {
    case SmeltPromptTemplateName.chatML,
         SmeltPromptTemplateName.chatMLXMLTools:
        guard let imStart = tokenizer.addedTokenId(for: "<|im_start|>"),
              let imEnd = tokenizer.addedTokenId(for: "<|im_end|>")
        else {
            throw InputBuildingError.missingChatTemplateTokens(template: "chatml")
        }
        var ids: [Int32] = [Int32(imStart)]
        ids += tokenizer.encode("system")
        ids += tokenizer.encode("\n")
        ids += tokenizer.encode(systemPrompt)
        ids += [Int32(imEnd)]
        ids += tokenizer.encode("\n")
        return ids

    default:
        return tokenizer.encode(systemPrompt)
    }
}

private struct SmeltToolFileDescriptor: Decodable {
    let name: String
    let description: String?
    let schemaJSON: String?
    let schemaJson: String?
    let schema: SmeltJSONValue?
    let parameters: SmeltJSONValue?

    enum CodingKeys: String, CodingKey {
        case name
        case description
        case schemaJSON
        case schemaJson = "schema_json"
        case schema
        case parameters
    }
}

private func toolDescriptor(from descriptor: SmeltToolFileDescriptor) throws -> SmeltToolDescriptor {
    guard !descriptor.name.isEmpty else {
        throw InputBuildingError.toolDescriptorMissingName
    }

    let schemaJSON: String
    if let rawSchema = descriptor.schemaJSON ?? descriptor.schemaJson {
        do {
            _ = try JSONSerialization.jsonObject(with: Data(rawSchema.utf8))
        } catch {
            throw InputBuildingError.toolDescriptorInvalidSchema(name: descriptor.name)
        }
        schemaJSON = rawSchema
    } else if let schema = descriptor.schema ?? descriptor.parameters {
        schemaJSON = try SmeltJSON.canonicalString(schema)
    } else {
        throw InputBuildingError.toolDescriptorMissingSchema(name: descriptor.name)
    }

    return SmeltToolDescriptor(
        name: descriptor.name,
        schemaJSON: schemaJSON,
        description: descriptor.description
    )
}

func loadToolsFile(_ path: String) throws -> [SmeltToolDescriptor] {
    let data: Data
    do {
        data = try Data(contentsOf: URL(fileURLWithPath: path))
    } catch {
        throw InputBuildingError.toolsFileReadFailed(path: path, underlying: error)
    }

    let descriptors: [SmeltToolFileDescriptor]
    do {
        descriptors = try JSONDecoder().decode([SmeltToolFileDescriptor].self, from: data)
    } catch {
        throw InputBuildingError.toolsFileInvalidJSON(path: path, underlying: error)
    }

    return try descriptors.enumerated().map { index, descriptor in
        do {
            return try toolDescriptor(from: descriptor)
        } catch InputBuildingError.toolDescriptorMissingName {
            throw InputBuildingError.toolDescriptorMissingNameAt(index: index)
        }
    }
}
