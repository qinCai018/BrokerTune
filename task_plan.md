# Task Plan: BrokerTuner APN-DDPG Audit and Paper Pseudocode Rewrite

## Goal
Audit the real BrokerTuner implementation and rewrite two Mosquitto tuning paper algorithms so they match the codebase while using the paper name `APN-DDPG`.

## Current Phase
Phase 2

## Phases
### Phase 1: Requirements & Discovery
- [x] Understand user intent
- [x] Identify constraints and requirements
- [x] Document findings in findings.md
- **Status:** complete

### Phase 2: Source Audit
- [x] Audit requested source files and relevant tests/docs
- [x] Extract implementation evidence for environment, policy, replay, and update flow
- [ ] Separate always-on mechanisms from config-gated mechanisms
- **Status:** in_progress

### Phase 3: Synthesis
- [ ] Consolidate mismatch list against paper-style DDPG pseudocode
- [ ] Map four modules to code locations and algorithm steps
- [ ] Mark any uncertain details for manual review
- **Status:** pending

### Phase 4: LaTeX Rewrite
- [ ] Draft algorithm 1 in `algorithm2e`
- [ ] Draft algorithm 2 in `algorithm2e`
- [ ] Enforce naming rule: pseudocode uses only `APN-DDPG`
- **Status:** pending

### Phase 5: Verification & Delivery
- [ ] Re-check source alignment and naming constraints
- [ ] Ensure required mechanisms appear in audit and pseudocode
- [ ] Deliver final answer
- **Status:** pending

## Key Questions
1. How do `MosquittoBrokerEnv`, `EnhancedDDPG`, `FeatureWiseAttentionExtractor`, and `PrioritizedNStepReplayBuffer` map to the paper's four modules?
2. Which mechanisms are always active, and which require configuration switches to enable in the current code?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Audit source before writing pseudocode | User explicitly requires code-first correction and forbids guessed algorithms |
| Maintain a file-backed audit trail | Task is multi-step and evidence-heavy; persistent notes reduce drift |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| None so far | 1 | N/A |

## Notes
- Use source code as ground truth when docs disagree.
- Keep `EnhancedDDPG` out of final algorithm bodies.
