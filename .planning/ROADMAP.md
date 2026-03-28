# Roadmap: Casper Charades Agent

## Overview

Transform the starter template into a competition-winning charades agent through four incremental phases, each producing a fully working agent. Phase 1 delivers a single-model bilingual baseline that can compete immediately. Phase 2 adds the biggest accuracy win: temporal reasoning across frame sequences with keyframe gating and wrong-guess feedback. Phase 3 adds sophisticated guess-timing through rank-vote accumulation and adaptive thresholds. Phase 4 adds multi-model parallelism and resilience. All code lives in a single file: `agent/src/agent/prompt.py`.

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

Decimal phases appear between their surrounding integers in numeric order.

- [ ] **Phase 1: Bilingual Foundation** - Single-model agent with charades prompting, bilingual guesses, frame preprocessing, and confidence gating
- [ ] **Phase 2: Temporal Intelligence** - Keyframe extraction, multi-frame reasoning, structured confidence output, and wrong-guess feedback loop
- [ ] **Phase 3: Confidence Sophistication** - Rank-vote accumulation, union-only candidate tracking, adaptive threshold, and semantic diversity
- [ ] **Phase 4: Multi-Model Resilience** - Parallel multi-model inference, agreement signals, role differentiation, and graceful degradation

## Phase Details

### Phase 1: Bilingual Foundation
**Goal**: Agent can watch a charades livestream, send preprocessed frames to a vision LLM, and submit bilingual guesses (English or Tagalog phrases) while preserving its 10-guess budget through confidence gating
**Depends on**: Nothing (first phase)
**Requirements**: FOUND-01, FOUND-02, FOUND-03, FOUND-04, FOUND-05, LANG-01, LANG-02, LANG-03, LANG-04
**Success Criteria** (what must be TRUE):
  1. Running `uv run -m agent --practice` produces text guesses from the vision LLM based on camera frames
  2. Agent returns None (skips) when it is not confident, rather than burning a guess on every frame
  3. Agent outputs multi-word phrases (not just single words) and can produce guesses in Tagalog when the action fits Filipino context
  4. Frames are visibly preprocessed (resized, compressed) before LLM calls -- raw 1080p frames are never sent
  5. Module-level state persists across analyze() calls (frame count, guess history are maintained between invocations)
**Plans**: TBD

Plans:
- [ ] 01-01: TBD

### Phase 2: Temporal Intelligence
**Goal**: Agent reasons across sequences of frames instead of treating each frame independently, skips redundant frames via keyframe detection, and feeds wrong-guess feedback into subsequent prompts
**Depends on**: Phase 1
**Requirements**: TEMP-01, TEMP-02, TEMP-03, TEMP-04, CONF-01, FEED-01, FEED-02, FEED-04
**Success Criteria** (what must be TRUE):
  1. Agent detects near-identical frames and skips them (observable: fewer LLM calls than frames received when camera is still)
  2. Agent sends multiple keyframes as context to the LLM, not just the latest single frame
  3. LLM returns structured output containing a guess, confidence score, and reasoning -- not just raw text
  4. After a wrong guess (409), the agent includes that rejected guess as a "NOT this" constraint in its next prompt
  5. Agent infers a 409 occurred when analyze() is called again after previously returning a non-None value
**Plans**: TBD

Plans:
- [ ] 02-01: TBD

### Phase 3: Confidence Sophistication
**Goal**: Agent makes mathematically disciplined guess-timing decisions through rank-weighted evidence accumulation across frames and an adaptive threshold that balances speed against guess conservation
**Depends on**: Phase 2
**Requirements**: CONF-02, CONF-03, CONF-04, FEED-03
**Success Criteria** (what must be TRUE):
  1. Agent maintains a ranked candidate list that accumulates votes across frames -- candidates are never dropped, only re-ranked
  2. Guess threshold adapts over time: starts conservative (waits for strong evidence), becomes more aggressive as time passes, and tightens when few guesses remain
  3. After a wrong guess, the next guess comes from a meaningfully different category (not a synonym or variant of the rejected guess)
  4. In practice mode, agent visibly accumulates confidence over multiple frames before committing to a guess
**Plans**: TBD

Plans:
- [ ] 03-01: TBD

### Phase 4: Multi-Model Resilience
**Goal**: Agent leverages parallel inference across multiple vision models for higher accuracy through agreement signals and survives individual model failures without interruption
**Depends on**: Phase 3
**Requirements**: MULTI-01, MULTI-02, MULTI-03, MULTI-04, RESIL-01, RESIL-02, RESIL-03
**Success Criteria** (what must be TRUE):
  1. Agent fires requests to 2-3 models in parallel and total latency equals the slowest model, not the sum
  2. When multiple models agree on a guess, confidence is boosted; when they disagree, the agent waits for more evidence
  3. Different models serve different roles (fast screener vs deep reasoner), not identical queries with majority vote
  4. If a secondary model times out or returns an error, the agent continues operating on the remaining model(s) without crashing
  5. Rate limits and API errors are handled gracefully with fallback behavior, not exceptions
**Plans**: TBD

Plans:
- [ ] 04-01: TBD

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Bilingual Foundation | 0/0 | Not started | - |
| 2. Temporal Intelligence | 0/0 | Not started | - |
| 3. Confidence Sophistication | 0/0 | Not started | - |
| 4. Multi-Model Resilience | 0/0 | Not started | - |
