package provider

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"google.golang.org/adk/v2/model"
	"google.golang.org/genai"
)

// zombieSSERequest is a complete Anthropic messages SSE turn: the prelude from
// stream_cancel_paths_test.go plus the closing events, so a streamed request
// can run to TurnComplete against the fixture server. The body carries no
// Content-Length and no chunked framing, so its end is signaled by the
// connection's write side closing — EOF terminates the SSE stream cleanly.
const zombieSSERequest = antSSEPrelude + `event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}

`

// hijacker takes the fixture server over at the socket level: each hijacked
// connection leaves http.Server's control entirely, so keep-alive reuse never
// re-enters ServeHTTP — the caller owns the socket for its whole life and
// must read subsequent requests off it itself. The returned bufio.ReadWriter
// is the server's own buffered view of the socket and may already hold bytes
// of the request that triggered the hijack, so it (not a fresh bufio.Reader
// over the conn) is what further reads must go through.
func hijacker(w http.ResponseWriter) (net.Conn, *bufio.ReadWriter, error) {
	hj, ok := w.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("server does not support hijacking")
	}
	return hj.Hijack()
}

// readHTTPRequest consumes one request from a hijacked connection: the head
// plus whatever body its Content-Length announces. The SDK sends a JSON body
// with every messages call, and the response for the next exchange must not
// overtake unread request bytes on the socket.
func readHTTPRequest(br *bufio.Reader) error {
	clen := 0
	for {
		line, err := br.ReadString('\n')
		if err != nil {
			return err
		}
		if line == "\r\n" {
			break
		}
		if v, ok := strings.CutPrefix(line, "Content-Length:"); ok {
			clen, _ = strconv.Atoi(strings.TrimSpace(v))
		}
	}
	if clen > 0 {
		if _, err := io.CopyN(io.Discard, br, int64(clen)); err != nil {
			return err
		}
	}
	return nil
}

// serveKeepAliveConn owns one hijacked connection for its whole life and
// serves every request the client sends on it, in order — including the
// keep-alive reuse that never re-enters ServeHTTP. The first request arrived
// through http.Server: its head is already parsed (and only its body bytes,
// if any were over-read, sit in br), so the body is drained by the
// Content-Length the server reports and the turn is answered before anything
// else is read — the client will not send another request on this socket
// until the response arrives. Every later request is read raw off the
// socket, head and body. The behavior per request models the gateway of
// issue #45: fresh connections get a complete SSE turn with chunked framing,
// whose terminating chunk hands the socket back to the client's pool as
// idle; every later request on the same socket is the incident — a gateway
// that has stopped serving its established connections accepts and goes
// silent. So a retry that draws a parked socket sits out another idle
// budget, and only a retry on a fresh dial completes.
func serveKeepAliveConn(conn net.Conn, br *bufio.Reader, firstContentLength int64) {
	defer conn.Close()
	if firstContentLength > 0 {
		if _, err := io.CopyN(io.Discard, br, firstContentLength); err != nil {
			return
		}
	}
	serveCompleteKeepAlive(conn, zombieSSERequest)
	// The socket is parked in the client's pool now. The next thing that
	// can arrive on it is a reused request — the zombie stage.
	if err := readHTTPRequest(br); err != nil {
		return // the peer (or the test cleanup) went away
	}
	holdSilent(conn, antSSEPrelude)
}

// serveCompleteKeepAlive writes a full SSE turn as one chunk plus the
// terminating zero chunk. The framing — not a connection close — delivers the
// body's EOF, so the client finishes the stream and returns the socket to its
// pool with the connection still open: exactly the parking that puts an idle
// keep-alive socket in play for the zombie stage.
func serveCompleteKeepAlive(conn net.Conn, body string) {
	fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n%x\r\n%s\r\n0\r\n\r\n", len(body), body)
}

// holdSilent writes a streaming response head plus prelude and then goes
// silent — the zombie of issue #45: a gateway that accepted the request and
// stopped talking. It unblocks only when the client tears the connection down
// (the idle abort cancels the request context, and closing the in-flight
// socket is what the side read reports) or on a safety timer, so a broken
// abort fails the test instead of hanging it. The timer sits 20× above the
// scaled idle budget (25s vs 1250ms), so a working abort always wins the
// race; only a broken abort path makes the test wait it out.
func holdSilent(conn net.Conn, prelude string) {
	fmt.Fprintf(conn, "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n\r\n%s", prelude)
	gone := make(chan struct{})
	go func() {
		defer close(gone)
		var sink [64]byte
		for {
			if _, err := conn.Read(sink[:]); err != nil {
				return
			}
		}
	}()
	select {
	case <-gone:
	case <-time.After(25 * time.Second):
	}
	conn.Close()
}

// TestStreamIdleAbortDropsZombieConnectionBeforeRetry is the incident from
// issue #45, end to end on a real HTTP client: the gateway has served this
// client before, so the pool holds parked keep-alive sockets; the gateway
// then starts discarding its established connections — a request on a reused
// socket is accepted and answered with silence, while fresh dials still get
// a full turn. The idle watch aborts the silent stream with "llm stream idle"
// (classified transient — this is what a retry budget consumes), and only
// closeIdle draining the pool makes the retry dial fresh: without it the
// retry draws the next parked socket and sits out a second idle budget.
func TestStreamIdleAbortDropsZombieConnectionBeforeRetry(t *testing.T) {
	var dials atomic.Int64
	var mu sync.Mutex
	var live []net.Conn
	t.Cleanup(func() {
		mu.Lock()
		defer mu.Unlock()
		for _, c := range live {
			c.Close()
		}
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, br, err := hijacker(w)
		if err != nil {
			return
		}
		mu.Lock()
		live = append(live, conn)
		mu.Unlock()
		dials.Add(1)
		serveKeepAliveConn(conn, br.Reader, r.ContentLength)
	}))
	t.Cleanup(srv.Close)

	opts := &LLMOptions{InsecureSkipTLS: true}
	m, err := NewAnthropic(context.Background(), "claude-sonnet-4-6", "sk-test", srv.URL, "none", opts)
	if err != nil {
		t.Fatalf("NewAnthropic: %v", err)
	}
	// The clone branch ran for the idle watch, so NewAnthropic must have
	// captured the pool closer on opts and NewLLM must have threaded it to
	// the watch — this assert is what makes the test fail if the plumbing
	// between BuildTransport and idleStreamModel is ever dropped.
	if opts.closeIdle == nil {
		t.Fatal("NewAnthropic left opts.closeIdle unset; the idle watch would have nothing to close")
	}
	// The idle budget is 1250ms — the original 250ms scaled ×5, same move
	// as the issue #61 fix: every duration in this test (the budget, the
	// holdSilent and retry guards below) moved together, so the
	// budget-vs-work proportions are unchanged. The parking turns are
	// healthy traffic that must complete inside the budget, and on a
	// loaded CI runner even a fully served turn (dial, request, one-shot
	// response, drain) overran 250ms — the idle watch fired on the
	// parking stage itself and the test died before its zombie stage
	// (CI runs 37302981844, 37326239257). The zombie stage needs the
	// budget only as silence to detect, so scaling it keeps that
	// detection and both failure injections intact.
	watched := idleStreamModel{inner: m, timeout: 1250 * time.Millisecond, tick: 0, closeIdle: opts.closeIdle}

	req := &model.LLMRequest{Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Hi"}}}}}

	// Parking stage: two concurrent streaming turns complete and leave two
	// idle keep-alive sockets in this client's pool. Two, not one — the
	// zombie attempt below consumes the socket it draws, and the retry must
	// still find one parked behind it to be able to draw a dead socket.
	turn := func() {
		_, errs := drainWatch(watched.GenerateContent(context.Background(), req, true))
		if len(errs) != 0 {
			t.Errorf("parking turn failed: %v", errs)
		}
	}
	var wg sync.WaitGroup
	wg.Add(2)
	for range 2 {
		go func() {
			defer wg.Done()
			turn()
		}()
	}
	wg.Wait()
	if got := dials.Load(); got != 2 {
		t.Fatalf("parking stage saw %d dials, want 2 — the pool did not end up with two idle sockets", got)
	}

	// Attempt 1: the zombie. The request reuses a parked socket, and the
	// gateway answers reused sockets with silence. Must yield exactly the
	// idle failure, and must not dial — a fresh dial here would mean the
	// pool was empty and the scenario stopped covering reuse.
	_, errs := drainWatch(watched.GenerateContent(context.Background(), req, true))
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "llm stream idle") {
		t.Fatalf("attempt 1: want exactly the idle failure, got %v", errs)
	}
	if got := dials.Load(); got != 2 {
		t.Errorf("the zombie attempt dialed a new connection (%d total, want 2): it did not draw a parked socket, so the retry below no longer exercises pool reuse", got)
	}

	// Attempt 2: the retry. With the pool drained by the idle abort, this
	// must dial a fresh connection — which the gateway still serves — and
	// complete normally. With the closeIdle call dropped, the retry draws
	// the remaining parked socket instead, is answered with the same
	// silence, and fails here with a second idle error.
	done := make(chan struct{})
	go func() {
		defer close(done)
		resps, errs := drainWatch(watched.GenerateContent(context.Background(), req, true))
		if len(errs) != 0 {
			t.Errorf("retry after the idle abort failed: %v", errs)
		}
		if len(resps) == 0 {
			t.Error("retry completed without a response")
		}
	}()
	// 25s mirrors the old 5s guard scaled ×5 with the budget — still 20×
	// above the idle window, and only reachable when the retry hangs on a
	// dead pooled socket.
	select {
	case <-done:
	case <-time.After(25 * time.Second):
		t.Fatal("retry after the idle abort did not complete: the pool handed the dead connection back")
	}
	if got := dials.Load(); got != 3 {
		t.Errorf("server saw %d connections, want 3 (two parked + the retry on a fresh socket)", got)
	}
}

// The same scenario without a captured transport closer: behavior must be
// unchanged from before issue #45 — the same idle failure, no panic on a nil
// closeIdle. (Whether the pool then reuses a socket is the transport's call;
// that is exactly the pre-fix status quo this package leaves alone.)
func TestStreamIdleAbortNilCloseIdleIsSafe(t *testing.T) {
	var conns atomic.Int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conns.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		<-r.Context().Done()
	}))
	t.Cleanup(srv.Close)

	m, err := NewAnthropic(context.Background(), "claude-sonnet-4-6", "sk-test", srv.URL, "none", nil)
	if err != nil {
		t.Fatalf("NewAnthropic: %v", err)
	}
	watched := idleStreamModel{inner: m, timeout: 150 * time.Millisecond, tick: 0}

	req := &model.LLMRequest{Contents: []*genai.Content{{Role: "user", Parts: []*genai.Part{{Text: "Hi"}}}}}
	_, errs := drainWatch(watched.GenerateContent(context.Background(), req, true))
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), "llm stream idle") {
		t.Fatalf("want exactly the idle failure with no closeIdle wired, got %v", errs)
	}
}

// TestCloseIdleConnectionsClearsClientPool isolates what the forwarding is
// for, at the http.Client level: after an idle socket has settled in the
// pool, CloseIdleConnections — reached through a wrapper stack of the same
// shape BuildTransport assembles (header injection, trace, rate limiting) —
// evicts it, so the next request dials fresh. Without the close, the same
// request reuses the pooled socket and the dial count does not move.
func TestCloseIdleConnectionsClearsClientPool(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("mode") == "silent" {
			// A zombie endpoint: accept, respond with a head, never speak.
			w.WriteHeader(http.StatusOK)
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
			<-r.Context().Done()
			return
		}
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	}))
	t.Cleanup(srv.Close)

	var dials atomic.Int64
	tr := http.DefaultTransport.(*http.Transport).Clone()
	tr.DialContext = func(ctx context.Context, network, addr string) (net.Conn, error) {
		dials.Add(1)
		var d net.Dialer
		return d.DialContext(ctx, network, addr)
	}
	client := &http.Client{Transport: &headerTransport{
		base: &traceTransport{
			base: &rateLimitShim{Base: tr},
		},
		headers: map[string]string{"X-Test": "1"},
	}}

	// Park one request on the silent endpoint and let another complete, so
	// the pool holds one idle socket.
	silentDone := make(chan struct{})
	go func() {
		defer close(silentDone)
		resp, err := client.Get(srv.URL + "/?mode=silent")
		if err == nil {
			_ = resp.Body.Close()
		}
	}()
	resp, err := client.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET: %v", err)
	}
	_ = resp.Body.Close()
	if got := dials.Load(); got != 2 {
		t.Fatalf("got %d dials before the close, want 2 (silent + ok)", got)
	}

	client.CloseIdleConnections()
	<-silentDone

	// The pool must be empty now: the next request dials a third connection.
	resp, err = client.Get(srv.URL + "/")
	if err != nil {
		t.Fatalf("GET after close: %v", err)
	}
	_ = resp.Body.Close()
	if got := dials.Load(); got != 3 {
		t.Errorf("got %d dials after CloseIdleConnections, want 3 — the pool still handed back a socket", got)
	}

	// Control: a second close is harmless.
	client.CloseIdleConnections()
}

// rateLimitShim is a minimal RoundTripper wrapper standing in for
// ratelimit.Transport (which lives in another package): it exercises the same
// guarded type-assert-and-forward CloseIdleConnections pattern the real
// transport implements, so the client-level test covers a three-layer stack.
type rateLimitShim struct{ Base http.RoundTripper }

func (t *rateLimitShim) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.Base.RoundTrip(req)
}

func (t *rateLimitShim) CloseIdleConnections() {
	if t.Base == nil || t.Base == http.DefaultTransport {
		return
	}
	if ci, ok := t.Base.(interface{ CloseIdleConnections() }); ok {
		ci.CloseIdleConnections()
	}
}
