package config

import "testing"

// splitDirs must parse a directory list authored for either platform (issue
// #61): ":" and ";" both separate entries, and a Windows drive path cut by a
// ":" split ("C" + "\a") is glued back together. Every case runs as a plain
// string transform, so all platforms' styles are covered on any GOOS.
func TestSplitDirs(t *testing.T) {
	tests := []struct {
		name string
		in   string
		want []string
	}{
		{"unix style", ":/a:/b", []string{"/a", "/b"}},
		{"windows style", `C:\a;C:\b`, []string{`C:\a`, `C:\b`}},
		{"drive glue across unix separator", `C:\a:C:\b`, []string{`C:\a`, `C:\b`}},
		{"empty segments dropped", "/a::/b:", []string{"/a", "/b"}},
		{"mixed separators and platforms", `/a;C:\b:/c`, []string{"/a", `C:\b`, "/c"}},
		{"forward-slash drive glued", "C:/a;C:/b", []string{"C:/a", "C:/b"}},
		{"whitespace trimmed", "  /a :  /b\t;\n/c ", []string{"/a", "/b", "/c"}},
		{"single entry", "/only", []string{"/only"}},
		{"all empty", "::;", nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := splitDirs(tt.in)
			if len(got) != len(tt.want) {
				t.Fatalf("splitDirs(%q) = %q, want %q", tt.in, got, tt.want)
			}
			for i := range got {
				if got[i] != tt.want[i] {
					t.Fatalf("splitDirs(%q) = %q, want %q", tt.in, got, tt.want)
				}
			}
		})
	}
}
