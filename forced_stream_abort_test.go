package cogito

import (
	"context"
	"strings"
	"testing"

	"github.com/sashabaranov/go-openai"
)

// proseStreamLLM streams a forced decision the way backends that ignore
// tool_choice do: plain prose, chunk after chunk, until the request context is
// cancelled. With toolCall set it streams a proper tool-call delta instead.
type proseStreamLLM struct {
	forcedFallbackLLM
	toolCall   bool
	sentChunks int
}

func (l *proseStreamLLM) CreateChatCompletionStream(ctx context.Context, req openai.ChatCompletionRequest) (<-chan StreamEvent, error) {
	ch := make(chan StreamEvent)
	go func() {
		defer close(ch)
		if l.toolCall {
			ch <- StreamEvent{Type: StreamEventToolCall, ToolCallIndex: 0, ToolCallID: "call_1", ToolName: "assist", ToolArgs: `{"query":"Siemens"}`}
			ch <- StreamEvent{Type: StreamEventDone, FinishReason: "tool_calls"}
			return
		}
		chunk := strings.Repeat("Siemens wurde 1847 gegründet. ", 4)
		for i := 0; i < 10000; i++ {
			select {
			case <-ctx.Done():
				return
			case ch <- StreamEvent{Type: StreamEventContent, Content: chunk}:
				l.sentChunks++
			}
		}
		ch <- StreamEvent{Type: StreamEventDone, FinishReason: "length"}
	}()
	return ch, nil
}

// A forced decision whose stream turns into prose is aborted within the first
// few hundred bytes and the arguments come from the schema fallback, carried
// by the assistant message under the choice's ID.
func TestForcedStreamProseIsAbortedAndRecovered(t *testing.T) {
	llm := &proseStreamLLM{}
	conv := []openai.ChatCompletionMessage{{Role: "user", Content: "Wann wurde Siemens gegründet?"}}
	res, err := decisionWithStreaming(context.Background(), llm, conv, testTool(), "assist", 1, func(StreamEvent) {})
	if err != nil {
		t.Fatalf("decision failed: %v", err)
	}
	if llm.sentChunks > 20 {
		t.Fatalf("stream was not aborted early: %d chunks sent", llm.sentChunks)
	}
	if len(res.toolChoices) != 1 || res.toolChoices[0].Name != "assist" {
		t.Fatalf("expected the recovered forced tool, got %+v", res.toolChoices)
	}
	if res.assistant == nil || len(res.assistant.Portable.ToolCalls) != 1 || res.assistant.Portable.ToolCalls[0].ID != res.toolChoices[0].ID {
		t.Fatalf("assistant message does not carry the recovered call: %+v", res.assistant)
	}
	if len(llm.calls) != 1 || llm.calls[0].ResponseFormat == nil {
		t.Fatalf("expected exactly one schema fallback call, got %d", len(llm.calls))
	}
}

// Counter-check: a backend that honors the forced tool_choice is neither
// aborted nor sent to the fallback.
func TestForcedStreamWithToolCallIsNotAborted(t *testing.T) {
	llm := &proseStreamLLM{toolCall: true}
	conv := []openai.ChatCompletionMessage{{Role: "user", Content: "Wann wurde Siemens gegründet?"}}
	res, err := decisionWithStreaming(context.Background(), llm, conv, testTool(), "assist", 1, func(StreamEvent) {})
	if err != nil {
		t.Fatalf("decision failed: %v", err)
	}
	if len(res.toolChoices) != 1 || res.toolChoices[0].ID != "call_1" {
		t.Fatalf("expected the streamed tool call, got %+v", res.toolChoices)
	}
	if len(llm.calls) != 0 {
		t.Fatalf("fallback must not run for an honored tool_choice, got %d calls", len(llm.calls))
	}
}
