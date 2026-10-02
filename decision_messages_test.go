package cogito

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"github.com/sashabaranov/go-openai"
	"reflect"
	"strings"
	"testing"
)

type envelopeTool struct{ calls int }

func (e *envelopeTool) Tool() openai.Tool {
	return openai.Tool{Type: openai.ToolTypeFunction, Function: &openai.FunctionDefinition{Name: "record"}}
}
func (e *envelopeTool) Execute(map[string]any) (string, any, error) {
	e.calls++
	return "recorded", nil, nil
}
func envelopeReply(m Message) CompletionReply {
	return CompletionReply{Response: LLMReply{ChatCompletionResponse: openai.ChatCompletionResponse{Choices: []openai.ChatCompletionChoice{{Message: m.Portable, FinishReason: openai.FinishReasonStop}}}}, Choices: []CompletionChoice{{Message: m, FinishReason: openai.FinishReasonStop}}}
}
func TestExecuteToolsRichEnvelope(t *testing.T) {
	tool := &envelopeTool{}
	original := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "original text", ToolCalls: []openai.ToolCall{
		{ID: "one", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{ "n": 900719925474099312345 }`}},
		{ID: "two", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{"n":2}`}},
	}}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"thinking","signature":"signed"}`), json.RawMessage(`{"type":"redacted_thinking","data":"hidden"}`), json.RawMessage(`{"type":"future","n":900719925474099312345}`)}}}
	final := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"text","text":"done","future":42}`)}}}
	llm := &completionRichFake{}
	llm.run = func(context.Context, openai.ChatCompletionRequest) (LLMReply, LLMUsage, error) {
		t.Error("legacy called")
		return envelopeReply(final).Response, LLMUsage{}, nil
	}
	llm.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
		if llm.richCalls == 1 {
			return envelopeReply(original), LLMUsage{TotalTokens: 3}, nil
		}
		found := false
		results := []string{}
		for _, m := range req.Messages {
			if m.Provider != nil {
				if !reflect.DeepEqual(m, original) {
					t.Fatal("original envelope changed")
				}
				found = true
			}
			if m.Portable.Role == "tool" {
				results = append(results, m.Portable.ToolCallID)
			}
		}
		if !found || !reflect.DeepEqual(results, []string{"one", "two"}) {
			t.Fatalf("missing original/correlation: %v %v", found, results)
		}
		return envelopeReply(final), LLMUsage{TotalTokens: 4}, nil
	}
	f, err := ExecuteTools(llm, NewEmptyFragment().AddMessage(UserMessageRole, "go"), WithTools(tool), WithIterations(3))
	if err != nil {
		t.Fatal(err)
	}
	messages, err := f.Conversation()
	if err != nil {
		t.Fatal(err)
	}
	if llm.calls != 0 || llm.richCalls != 2 || tool.calls != 2 {
		t.Fatalf("calls legacy/rich/tools %d/%d/%d", llm.calls, llm.richCalls, tool.calls)
	}
	if len(messages) != 5 || !reflect.DeepEqual(messages[1], original) || !reflect.DeepEqual(messages[4], final) {
		t.Fatalf("lost returned envelopes: %+v", messages)
	}
	if f.Status.CumulativeUsage.TotalTokens != 7 {
		t.Fatal("usage lost")
	}
}

// The rich method must win even when a streaming callback is configured.
type envelopeStreamingFake struct {
	*completionRichFake
	streams int
}

func (f *envelopeStreamingFake) CreateChatCompletionStream(context.Context, openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	f.streams++
	panic("lossy stream called")
}
func TestExecuteToolsRichPreferenceAndCancellation(t *testing.T) {
	for _, cancelReply := range []bool{false, true} {
		t.Run(fmt.Sprint(cancelReply), func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			m := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}, Provider: &ProviderMessage{Blocks: []json.RawMessage{json.RawMessage(`{"opaque":true}`)}}}
			fake := &envelopeStreamingFake{completionRichFake: &completionRichFake{}}
			fake.rich = func(context.Context, CompletionRequest) (CompletionReply, LLMUsage, error) {
				if cancelReply {
					cancel()
				}
				return envelopeReply(m), LLMUsage{}, nil
			}
			f, err := ExecuteTools(fake, NewEmptyFragment(), WithTools(&envelopeTool{}), WithContext(ctx), WithStreamCallback(func(StreamEvent) {}))
			if fake.calls != 0 || fake.streams != 0 || fake.richCalls != 1 {
				t.Fatal("wrong dispatch")
			}
			if cancelReply {
				if !errors.Is(err, context.Canceled) || len(f.Messages) != 0 {
					t.Fatalf("cancel committed: %v %+v", err, f.Messages)
				}
			} else if err != nil || len(f.Messages) != 1 {
				t.Fatalf("no-tool reply lost: %v", err)
			}
		})
	}
}
func TestExecuteToolsRejectLegacyOpaque(t *testing.T) {
	fake := &completionLegacyFake{}
	f := NewEmptyFragment().AppendMessages(Message{Portable: openai.ChatCompletionMessage{Role: "assistant"}, Provider: &ProviderMessage{}})
	_, err := ExecuteTools(fake, f, WithTools(&envelopeTool{}))
	var unsupported *ErrProviderMessagesUnsupported
	if !errors.As(err, &unsupported) || fake.calls != 0 {
		t.Fatalf("must reject before backend: %v", err)
	}
}

func TestExecuteToolsRichPolicy(t *testing.T) {
	for _, mode := range []string{"skip", "modified-id", "mutated-id", "modified-args", "adjust", "cancel", "cancel-skip"} {
		t.Run(mode, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			tool := &envelopeTool{}
			original := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "signed selection", ToolCalls: []openai.ToolCall{
				{ID: "one", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{"n":1}`}},
				{ID: "two", Type: openai.ToolTypeFunction, Function: openai.FunctionCall{Name: "record", Arguments: `{"n":2}`}},
			}}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"thinking","signature":"policy-signature"}`), json.RawMessage(`{"type":"tool_use","id":"one","name":"record","input":{"n":1}}`), json.RawMessage(`{"type":"tool_use","id":"two","name":"record","input":{"n":2}}`)}}}
			final := Message{Portable: openai.ChatCompletionMessage{Role: "assistant", Content: "done"}, Provider: &ProviderMessage{Version: 1, Blocks: []json.RawMessage{json.RawMessage(`{"type":"text","text":"done"}`)}}}
			check := func(messages []Message) {
				t.Helper()
				if len(messages) < 4 || !reflect.DeepEqual(messages[1], original) {
					t.Fatalf("original assistant must precede results, unchanged: %+v", messages)
				}
				ids := map[string]int{}
				for _, m := range messages[2:] {
					if m.Portable.Role == "tool" {
						ids[m.Portable.ToolCallID]++
					}
				}
				if !reflect.DeepEqual(ids, map[string]int{"one": 1, "two": 1}) {
					t.Fatalf("orphaned results: %v", ids)
				}
			}
			fake := &completionRichFake{}
			fake.rich = func(_ context.Context, req CompletionRequest) (CompletionReply, LLMUsage, error) {
				if fake.richCalls == 1 {
					return envelopeReply(original), LLMUsage{}, nil
				}
				if mode == "skip" || mode == "modified-args" {
					check(req.Messages)
				}
				return envelopeReply(final), LLMUsage{}, nil
			}
			input := NewEmptyFragment().AddMessage(UserMessageRole, "go")
			f, err := ExecuteTools(fake, input, WithTools(tool), WithContext(ctx), WithIterations(3), WithToolCallBack(func(tc *ToolChoice, _ *SessionState) ToolCallDecision {
				d := ToolCallDecision{Approved: true}
				if tc.ID != "one" {
					return d
				}
				switch mode {
				case "skip":
					d.Skip = true
				case "modified-id":
					c := *tc
					c.ID = "orphan"
					d.Modified = &c
				case "mutated-id":
					tc.ID = "orphan"
				case "modified-args":
					c := *tc
					c.Arguments = map[string]any{"n": 42}
					d.Modified = &c
				case "adjust":
					d.Adjustment = "do not use tools"
				case "cancel":
					cancel()
				case "cancel-skip":
					cancel()
					d.Skip = true
				}
				return d
			}))
			messages, convErr := f.Conversation()
			if convErr != nil {
				t.Fatal(convErr)
			}
			switch mode {
			case "skip", "modified-args":
				if err != nil {
					t.Fatal(err)
				}
				check(messages)
				if mode == "modified-args" && f.Status.ToolResults[0].ToolArguments.Arguments["n"] != 42 {
					t.Fatal("execution did not use modified arguments")
				}
				want := 2
				if mode == "skip" {
					want = 1
				}
				if tool.calls != want || fake.richCalls != 2 {
					t.Fatalf("calls tools/backend: %d/%d", tool.calls, fake.richCalls)
				}
			case "adjust":
				if err != nil || tool.calls != 0 || fake.richCalls != 2 {
					t.Fatalf("adjustment: err=%v calls=%d selections=%d", err, tool.calls, fake.richCalls)
				}
				check(messages)
			default:
				if err == nil || tool.calls != 0 || fake.richCalls != 1 || fake.calls != 0 {
					t.Fatalf("policy must fail closed before side effects: err=%v tools=%d rich=%d legacy=%d", err, tool.calls, fake.richCalls, fake.calls)
				}
				if mode == "cancel" || mode == "cancel-skip" {
					if !errors.Is(err, context.Canceled) {
						t.Fatalf("wrong cancellation error: %v", err)
					}
				}
				// Nonstream rich policy now durably closes the received batch.
				// Legacy streaming cancellation retains its existing boundary.
				check(messages)
			}
		})
	}
}

// TestForcedPickAfterAssistantEndsWithUserTurn pins the fix for the gemma-4
// sampler failure: a forced tool choice right after an assistant (reasoning)
// turn must close with a user turn; without forcing, or when the last turn is
// already a user turn, the conversation is passed through unchanged.
func TestForcedPickAfterAssistantEndsWithUserTurn(t *testing.T) {
	conv := []openai.ChatCompletionMessage{
		{Role: "user", Content: "what's the weather in Boston?"},
		{Role: "assistant", Content: "I should call the weather tool."},
	}
	got := closeWithUserTurnForForcedPick(conv, "pick_tool")
	if len(got) != 3 || got[2].Role != "user" || got[2].Content == "" {
		t.Fatalf("expected a closing user turn after the assistant turn, got %+v", got)
	}
	// The closing turn must refer back to the existing request; a standalone
	// "Pick the tool now." was read by models as a new, ambiguous user request.
	if got[2].Content != forcedPickClosingTurn {
		t.Fatalf("unexpected closing turn %q", got[2].Content)
	}
	if !strings.Contains(got[2].Content, "request above") || strings.Contains(got[2].Content, "Pick the tool now") {
		t.Fatalf("closing turn must point back at the request above, got %q", got[2].Content)
	}
	if len(conv) != 2 {
		t.Fatalf("input must not be mutated, got %d messages", len(conv))
	}
	if got := closeWithUserTurnForForcedPick(conv, ""); len(got) != 2 {
		t.Fatalf("without a forced tool the conversation must pass through, got %d", len(got))
	}
	user := []openai.ChatCompletionMessage{{Role: "user", Content: "hi"}}
	if got := closeWithUserTurnForForcedPick(user, "pick_tool"); len(got) != 1 {
		t.Fatalf("a conversation already ending with a user turn must pass through, got %d", len(got))
	}
}

// The non-streaming decision path must close a forced pick that follows an
// assistant turn with a user turn on the request the model receives; an
// unforced decision is sent unchanged.
func TestDecisionForcedPickAfterAssistantRequestEndsWithUser(t *testing.T) {
	conv := []openai.ChatCompletionMessage{
		{Role: "user", Content: "Wann wurde Siemens gegründet?"},
		{Role: "assistant", Content: "I should look this up."},
	}
	forced := &forcedFallbackLLM{}
	if _, err := decision(context.Background(), forced, conv, testTool(), "assist", 1); err != nil {
		t.Fatalf("forced decision failed: %v", err)
	}
	sent := forced.calls[0].Messages
	if last := sent[len(sent)-1]; last.Role != openai.ChatMessageRoleUser {
		t.Fatalf("forced pick after an assistant turn must end with a user turn, got %q", last.Role)
	}

	free := &forcedFallbackLLM{}
	if _, err := decision(context.Background(), free, conv, testTool(), "", 1); err != nil {
		t.Fatalf("unforced decision failed: %v", err)
	}
	if got := free.calls[0].Messages; len(got) != len(conv) || got[len(got)-1].Role != openai.ChatMessageRoleAssistant {
		t.Fatalf("unforced decision must be sent unchanged, got %d messages ending with %q", len(got), got[len(got)-1].Role)
	}
}
