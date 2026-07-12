package cogito

import "testing"

func TestParseStreamedToolArgs_Defensive(t *testing.T) {
	cases := []struct {
		name, raw, wantKey string
		wantVal            any
		wantErr            bool
	}{
		{"plain valid", `{"query":"photosynthesis"}`, "query", "photosynthesis", false},
		{"concatenated duplicate (frontier defect)", `{"query":"photosynthesis"}{"query":"photosynthesis"}`, "query", "photosynthesis", false},
		{"concatenated different -> first wins", `{"city":"NYC"}{"city":"LA"}`, "city", "NYC", false},
		{"truly malformed", `{"query":`, "", nil, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			out := make(map[string]any)
			err := parseStreamedToolArgs(c.raw, &out)
			if c.wantErr {
				if err == nil {
					t.Fatalf("expected error, got nil (out=%v)", out)
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if out[c.wantKey] != c.wantVal {
				t.Fatalf("got %v=%v, want %v", c.wantKey, out[c.wantKey], c.wantVal)
			}
		})
	}
}
