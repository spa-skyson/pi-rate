package session

import (
	"context"
	"fmt"
	"time"

	"github.com/google/uuid"

	adkmodel "google.golang.org/adk/v2/model"
	"google.golang.org/adk/v2/session"
	"google.golang.org/genai"

	"github.com/spa-skyson/pi-rate/internal/logger"
)

// SynthesizeStoppedResponses answers every function call in the session that
// no function response follows with a synthetic error response naming the
// stop reason. A run that ends between persisting a call and receiving its
// response — a turn that died with the process (OOM kill), a resume after a
// crash, the loop auto-stop (issue #60), a cancel, a terminal failure —
// otherwise leaves the call dangling forever: ADK drops it from every later
// context build with a "dropping function calls with no matching function
// response" warning, and the model never learns what happened to the tool.
//
// The sweep is idempotent: after one pass no orphans remain, so calling it
// again — on the next turn, the next run — appends nothing. This is what lets
// a pre-turn caller run it before every dispatch without duplicating what a
// post-turn caller (the TUI loop's fail path) already healed.
//
// The scan is one reverse pass over the persisted events: a response marks
// its id answered, a call with no later answer is an orphan, so the cost is
// linear in history length and old orphans from earlier turns are healed
// too. Appended events copy the shape of a real tool response (author,
// invocation and node info come from the call's event), so ADK matches them
// by id like any other pair. Best-effort: a failed append is logged and the
// orphan is left for the next sweep.
func SynthesizeStoppedResponses(ctx context.Context, svc session.Service, appName, userID, sessionID, reason string, log *logger.Logger) {
	if svc == nil || sessionID == "" {
		return
	}
	resp, err := svc.Get(ctx, &session.GetRequest{
		AppName:   appName,
		UserID:    userID,
		SessionID: sessionID,
	})
	if err != nil || resp == nil || resp.Session == nil {
		log.Errorf("synthetic responses: reading session %s: %v", sessionID, err)
		return
	}
	type orphan struct {
		call *genai.FunctionCall
		from *session.Event
	}
	events := resp.Session.Events()
	answered := make(map[string]struct{}, events.Len())
	var orphans []orphan
	for i := events.Len() - 1; i >= 0; i-- {
		ev := events.At(i)
		if ev == nil || ev.Content == nil {
			continue
		}
		for _, part := range ev.Content.Parts {
			if part == nil {
				continue
			}
			if fr := part.FunctionResponse; fr != nil && fr.ID != "" {
				answered[fr.ID] = struct{}{}
			}
			if fc := part.FunctionCall; fc != nil && fc.ID != "" {
				if _, ok := answered[fc.ID]; !ok {
					orphans = append(orphans, orphan{call: fc, from: ev})
				}
			}
		}
	}
	for _, o := range orphans {
		if err := svc.AppendEvent(ctx, resp.Session, stoppedResponseEvent(o.call, o.from, reason)); err != nil {
			log.Errorf("synthetic response for %s: %v", o.call.ID, err)
			continue
		}
		log.Info(fmt.Sprintf("synthesized function response for %s (%s): turn was stopped: %s", o.call.ID, o.call.Name, reason))
	}
}

// stoppedResponseErrorFmt is the error payload of the synthetic response.
// Beyond naming the stop reason it must tell the model that the operation's
// outcome is unknown: the call was persisted but never answered, so whether
// the tool's side effect (a file write, a command, an external request)
// happened cannot be known from the session — the user has to verify it.
// Without that note the model treats the synthetic error as ground truth
// about the operation and may redo or skip a step whose real state differs.
const stoppedResponseErrorFmt = "turn was stopped: %s. The tool call never completed, " +
	"so whether its side effect already happened is unknown — verify the operation's state " +
	"with the user before retrying the tool or acting on its result."

// stoppedResponseEvent builds the synthetic function-response event for one
// orphaned call: the same shape a real tool response event carries (role
// "user", the call's author, invocation and node info), with the error key
// ADK tools use to report failure. ADK pairs it with the call by id.
func stoppedResponseEvent(call *genai.FunctionCall, from *session.Event, reason string) *session.Event {
	return &session.Event{
		ID: uuid.NewString(),
		LLMResponse: adkmodel.LLMResponse{
			Content: &genai.Content{
				Role: genai.RoleUser,
				Parts: []*genai.Part{{
					FunctionResponse: &genai.FunctionResponse{
						ID:       call.ID,
						Name:     call.Name,
						Response: map[string]any{"error": fmt.Sprintf(stoppedResponseErrorFmt, reason)},
					},
				}},
			},
		},
		InvocationID: from.InvocationID,
		Author:       from.Author,
		Timestamp:    time.Now(),
		Actions:      session.EventActions{StateDelta: map[string]any{}, ArtifactDelta: map[string]int64{}},
		NodeInfo:     from.NodeInfo,
	}
}
