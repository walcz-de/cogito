package cogito

import (
	"encoding/json"
	"strings"
	"testing"
)

// A property schema that uses anyOf, $ref or a nested additionalProperties
// must reach the model as the MCP server declared it. Routing the
// properties through jsonschema.Definition dropped all three: Definition has
// no field for them, so the model saw an argument with no allowed shape and
// could only send {} (walcz-de/agntsio #453).
func TestMCPToolKeepsAnyOfRefAndDefsInProperties(t *testing.T) {
	var schema toolInputSchema
	raw := `{
	  "type": "object",
	  "properties": {
	    "betrag": {"anyOf": [{"type": "number"}, {"type": "string"}], "description": "Betrag"},
	    "empfaenger": {"$ref": "#/$defs/Konto"},
	    "optionen": {"type": "object", "additionalProperties": {"type": "string"}}
	  },
	  "required": ["betrag"],
	  "$defs": {"Konto": {"type": "object", "properties": {"iban": {"type": "string"}}, "required": ["iban"]}}
	}`
	if err := json.Unmarshal([]byte(raw), &schema); err != nil {
		t.Fatalf("unmarshal input schema: %v", err)
	}
	coerceNullableTypes(schema.Properties)
	tool := &mcpTool{name: "ueberweisung", inputSchema: schema}

	dat, err := json.Marshal(tool.Tool().Function.Parameters)
	if err != nil {
		t.Fatalf("marshal parameters: %v", err)
	}
	for _, want := range []string{`"anyOf"`, `"$ref":"#/$defs/Konto"`, `"$defs"`, `"iban"`, `"additionalProperties":{"type":"string"}`} {
		if !strings.Contains(string(dat), want) {
			t.Errorf("advertised parameters lost %s: %s", want, dat)
		}
	}
}

// Counter-check: a plain schema still arrives unchanged and without the
// optional keys, so the fix does not invent structure.
func TestMCPToolPlainSchemaUnchanged(t *testing.T) {
	var schema toolInputSchema
	raw := `{"type":"object","properties":{"q":{"type":"string","description":"Suche"}},"required":["q"]}`
	if err := json.Unmarshal([]byte(raw), &schema); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	tool := &mcpTool{name: "suche", inputSchema: schema}
	dat, _ := json.Marshal(tool.Tool().Function.Parameters)
	var got map[string]any
	_ = json.Unmarshal(dat, &got)
	if _, ok := got["$defs"]; ok {
		t.Errorf("no $defs expected: %s", dat)
	}
	p := got["properties"].(map[string]any)["q"].(map[string]any)
	if p["type"] != "string" || p["description"] != "Suche" {
		t.Errorf("plain property changed: %s", dat)
	}
}
