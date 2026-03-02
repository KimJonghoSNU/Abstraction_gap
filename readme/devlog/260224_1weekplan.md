# 2026-02-24 Plan: Traversal-first (A안)

## Locked Decisions
- Primary metric: `nDCG@10`
- Fast loop scope: `biology` only
- Baseline: original `LATTICE` tree
- This 5-day scope: train/evaluate traversal policy first
- Deferred: clustering model training (next phase)

## Why Traversal-first
- Traversal supervision can be built with current tree and retrieval logs.
- Iteration speed is higher than clustering-tree regeneration.
- `nDCG` improvement attribution is cleaner (policy quality vs tree quality).

## 5-Day Schedule (2026-02-25 ~ 2026-03-01)

### Day 1 (2026-02-25): Baseline Fix + Eval Harness Freeze
- Goal: lock reproducible baseline and evaluation contract.
- Tasks:
    - run original `LATTICE` tree baseline on `biology`.
    - freeze eval config: dataset split, `k`, seed, rerank option, budget.
    - store baseline artifacts and query-level `nDCG@10`.
- Deliverables:
    - `biology` baseline score table.
    - query-level error slice (top loss queries).

### Day 2 (2026-02-26): Traversal Supervision Data Build
- Goal: build training labels for traversal decisions.
- Tasks:
    - create per-step supervision tuple:
      `state(name+desc, query, history) -> target child node`.
    - create hard negatives from sibling/nearby branches.
    - split train/val with fixed seed.
- Deliverables:
    - traversal train/val jsonl.
    - label stats (depth distribution, class balance, OOV node ratio).

### Day 3 (2026-02-27): Traversal Model SFT (v1)
- Goal: get first learned traversal policy.
- Tasks:
    - train SFT model for next-node selection.
    - run offline validation (top-1 / top-k node accuracy by depth).
    - calibrate decoding/retry policy for stable online behavior.
- Deliverables:
    - best checkpoint id.
    - validation report (depth-wise accuracy + failure cases).

### Day 4 (2026-02-28): Online Retrieval Integration + Ablation
- Goal: verify end-to-end gain on retrieval metric.
- Tasks:
    - integrate learned traversal into retrieval loop.
    - compare against baseline traversal policy.
    - ablation:
      `history on/off`, `desc anchor mode on/off`, `beam width`.
- Deliverables:
    - end-to-end `nDCG@10` comparison table.
    - ablation summary with win/loss conditions.

### Day 5 (2026-03-01): Stabilization + Report
- Goal: finalize weekly result and next action.
- Tasks:
    - rerun best config with fixed seeds for variance check.
    - package artifacts for reproducibility.
    - document blocker list and next-phase entry criteria (clustering phase).
- Deliverables:
    - final weekly summary (`baseline vs traversal-v1`).
    - next-week backlog with priority.

## Success Criteria (This Week)
- Primary: `biology nDCG@10` improves over original `LATTICE` tree baseline.
- Secondary: depth-3/4 traversal accuracy increases without large latency blow-up.
- Tertiary: failure cases are categorized with actionable fixes.

## Stop Rules
- If Day 3 validation node accuracy does not beat baseline heuristic, pause integration and fix supervision quality first.
- If online gain is inconsistent across seeds on Day 5, do not move to clustering training yet.

## Phase Boundary (Explicit)
- Current phase ends at traversal policy validation.
- Clustering model training starts only after stable traversal gain is confirmed.
