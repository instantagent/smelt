import Testing
import SmeltRuntime

@Suite struct SmeltXMLFunctionToolCodecTests {
    private let read = SmeltToolDescriptor(
        name: "read",
        schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"},"line":{"type":"integer"}},"required":["path"],"additionalProperties":false}"#,
        description: "Read a file"
    )

    @Test func decodesNativeParametersToCanonicalOpenAIArguments() throws {
        let decoded = try SmeltXMLFunctionToolCodec.decode(
            """
            I will inspect it.
            <tool_call>
            <function=read>
            <parameter=path>
            Sources/main.swift
            </parameter>
            <parameter=line>
            7
            </parameter>
            </function>
            </tool_call>
            """,
            tools: [read]
        )
        #expect(decoded.leadingText == "I will inspect it.")
        #expect(decoded.calls == [SmeltDecodedXMLToolCall(
            name: "read",
            argumentsJSON: #"{"line":7,"path":"Sources\/main.swift"}"#
        )])
    }

    @Test func rendersPinnedToolsSystemBlockExactly() throws {
        let rendered = try SmeltXMLFunctionToolCodec.renderSystemMessage(
            tools: [read]
        )
        let descriptor = #"{"type": "function", "function": {"name": "read", "description": "Read a file", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "line": {"type": "integer"}}, "required": ["path"], "additionalProperties": false}}}"#
        let protocolSuffix = """

        </tools>

        If you choose to call a function ONLY reply in the following format with NO suffix:

        <tool_call>
        <function=example_function_name>
        <parameter=example_parameter_1>
        value_1
        </parameter>
        <parameter=example_parameter_2>
        This is the value for the second parameter
        that can span
        multiple lines
        </parameter>
        </function>
        </tool_call>

        <IMPORTANT>
        Reminder:
        - Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
        - Required parameters MUST be specified
        - You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
        - If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
        </IMPORTANT>
        """
        #expect(rendered == """
        # Tools

        You have access to the following functions:

        <tools>
        \(descriptor)\(protocolSuffix)
        """)
    }

    @Test func priorCallsRenderDeterministicallyAndRoundTrip() throws {
        let rendered = try SmeltXMLFunctionToolCodec.renderCalls([
            SmeltDecodedXMLToolCall(
                name: "read",
                argumentsJSON: #"{"path":"Sources/main.swift","line":7}"#
            ),
        ])
        #expect(rendered == """
        <tool_call>
        <function=read>
        <parameter=line>
        7
        </parameter>
        <parameter=path>
        Sources/main.swift
        </parameter>
        </function>
        </tool_call>
        """)
        let decoded = try SmeltXMLFunctionToolCodec.decode(
            rendered, tools: [read]
        )
        #expect(decoded.calls == [SmeltDecodedXMLToolCall(
            name: "read",
            argumentsJSON: #"{"line":7,"path":"Sources\/main.swift"}"#
        )])
    }

    @Test func preservesMultilineStringValues() throws {
        let write = SmeltToolDescriptor(
            name: "write",
            schemaJSON: #"{"type":"object","properties":{"path":{"type":"string"},"content":{"type":"string"}},"required":["path","content"]}"#
        )
        let decoded = try SmeltXMLFunctionToolCodec.decode(
            """
            <tool_call>
            <function=write>
            <parameter=path>
            /tmp/answer.txt
            </parameter>
            <parameter=content>
            first
            second
            </parameter>
            </function>
            </tool_call>
            """,
            tools: [write]
        )
        #expect(decoded.calls[0].argumentsJSON.contains(#""content":"first\nsecond""#))
    }

    @Test func rejectsUnknownMissingAndIllTypedArguments() {
        let invalid = [
            """
            <tool_call>
            <function=read>
            <parameter=other>
            x
            </parameter>
            </function>
            </tool_call>
            """,
            """
            <tool_call>
            <function=read>
            <parameter=line>
            7
            </parameter>
            </function>
            </tool_call>
            """,
            """
            <tool_call>
            <function=read>
            <parameter=path>
            x
            </parameter>
            <parameter=line>
            seven
            </parameter>
            </function>
            </tool_call>
            """,
        ]
        for text in invalid {
            #expect(throws: SmeltXMLFunctionToolCodecError.self) {
                try SmeltXMLFunctionToolCodec.decode(text, tools: [read])
            }
        }
    }

    @Test func grammarIsBoundToActiveSemanticNames() throws {
        let grammar = try SmeltXMLFunctionToolCodec.larkGrammar(
            for: [read], allowText: true
        )
        #expect(grammar.contains(#"function_name: "read""#))
        #expect(grammar.contains(#""line""#))
        #expect(grammar.contains(#""path""#))
        #expect(grammar.contains("start: tool_call | text_response"))
        #expect(grammar.contains("text_tail: /[^<]*/"))
    }

    @Test func streamingDecoderEmitsFunctionStartsAcrossArbitraryChunking() throws {
        let text = """
        I will inspect both files.
        <tool_call>
        <function=read>
        <parameter=path>
        Sources/main.swift
        </parameter>
        </function>
        </tool_call>
        <tool_call>
        <function=read>
        <parameter=path>
        Tests/main.swift
        </parameter>
        </function>
        </tool_call>
        """
        let decoder = SmeltXMLFunctionToolStreamDecoder(toolNames: ["read"])
        var events: [SmeltXMLFunctionToolStreamCallStart] = []
        for character in text {
            events += try decoder.consume(String(character))
        }
        #expect(events == [
            SmeltXMLFunctionToolStreamCallStart(
                index: 0,
                name: "read",
                leadingText: "I will inspect both files."
            ),
            SmeltXMLFunctionToolStreamCallStart(index: 1, name: "read"),
        ])
    }

    @Test func streamingDecoderRejectsUndeclaredFunctionAsSoonAsNameCloses() throws {
        let decoder = SmeltXMLFunctionToolStreamDecoder(toolNames: ["read"])
        #expect(throws: SmeltXMLFunctionToolCodecError.self) {
            try decoder.consume("<tool_call>\n<function=write>")
        }
    }
}
