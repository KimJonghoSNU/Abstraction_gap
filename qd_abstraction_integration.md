# QD Abstraction Integration (Working Note)

# Round 1

## 1) Motivation (Why this integration)
The current concern is whether **abstraction gap** can be solved by:
- tighter coupling between the document abstraction (LATTICE's tree structure) and query abstraction (query rewriting to define what evidence to show)

We want a pipeline that reliably jumps to the correct abstraction level without getting trapped by feedback locality.


## 2) Current pipeline variants being compared

### A. Baseline interaction loop (status quo)
- QE → flat retrieval → traversal
- rewrite happens during traversal every `rewrite_every`

### B. Pre-flat rewrite (no traversal rewrite)
- flat retrieval on original query
- rewrite once using flat context
- re-run flat retrieval → gate → traversal (no rewrite in traversal)
- context options:
    - **branch-only**
    - **leaf-only**
    - **all-nodes**

### C. Pre-flat rewrite + traversal rewrite
- Same as B, plus iterative rewrite during traversal

### D. Branch-seeded traversal (new variant)
- After flat retrieval, use selected branch gates as **initial beam states**
- run traversal for iter 0 inside those subtrees
- after iter 0, traverse normally
- controlled by `--seed_from_flat_gates`


## 3) Observations so far (from ndcg_summary.csv)
Summary trends:
- **Pre-flat rewrite (all-nodes) > branch-only** on average.
- **Leaf-only pre-flat rewrite** looks strongest.
- **Iterative rewrite on top of branch-only** does not add value and often hurts.

Interpretation:
- **Branch-only context for rewriting is likely harmful or too abstract**, especially when rewrite drives the retriever into the wrong region.
- Leaf evidence seems to ground rewrite better than branch summaries.
- 즉 query rewrite -> Tree traversal 을 더 잘하게 해줌. 하지만 tree의 정보가 query rewrite를 더 잘하게 해주지는 않음.


## 4) Current hypothesis alignment
From `abstraction_gap2.md`, abstraction gap is driven by:
1) bridge representation bottleneck, and
2) feedback locality from top‑K → rewrite loops.

The current results suggest:
- **branch summaries alone are not reliable bridge signals** (may cause “logic poisoning”)
- **leaf evidence is a better anchor for rewrite**, at least in the current model + prompt setup





# Round 2

## 5) Implementation Plan (Draft)
Goal: use tree summaries to define **abstraction schema** (categories) while keeping **rewrite evidence grounded in leaf docs**, based on current results.

### 5.1 Design choices (default + ablations)
1) **Schema source**
    - Default: **LLM-generated schema** from flat-retrieved branch descriptions
    - Future (not implement now): **hybrid** (tree schema + LLM prune/merge)
2) **Schema scope**
    - Default: **intermediate children of flat-retrieved branches** (no breadth cap for now)
    - Ablation: **top branches only**
3) **Schema form**
    - Default: **free-text labels** derived from branch desc/path
4) **Schema size**
    - **dynamic per query**, between 3 and 5 categories
5) **Rewrite evidence**
    - Default: **leaf-only evidence** (matches current best results)
    - Branch summaries are used only to define schema, not as rewrite evidence
6) **Rewrite output**
    - Single final rewrite (structured by schema)
    - Per-category candidates are generated for **logging only**, but keep code extensible to fuse later
7) **Iteration policy**
    - Schema is generated **once pre-flat** (not refreshed every iter)
    - Schema is **logged per iter** for analysis
    - TODO: ablate **schema refresh every iter**
    - Rewrite timing is still controlled by `rewrite_every` for ablation

### 5.2 Proposed pipeline (high level)
1) **Flat retrieval (all nodes)** on original query
2) **Schema induction** from flat branch hits
3) **Rewrite** using:
    - schema as structure
    - **leaf-only** evidence from flat results as grounding
4) **Leaf-only retrieval** (evaluation uses leaf-only)
5) Optional: **traversal** with `--seed_from_flat_gates` (ablation)

### 5.3 Concrete implementation points
1) `previous/shortcut_reranker/scripts/query_generator.py`
    - Add optional `schema` input block
    - Generate single structured rewrite plus per-category candidates (logging)
2) `src/run.py`
    - Log schema per iter alongside rewrite cache
    - Keep leaf-only evidence for rewrite context
    - No hybrid conflict-resolution logic for now
3) Prompts
    - New prompt template: schema-guided rewrite
    - Separate prompt for schema induction (tree-only vs LLM)

### 5.4 Risks / critique (why this direction is safer)
1) Branch-only evidence previously degraded results; schema-only use avoids that
2) LLM schema can drift from branch evidence; log schema drift + consider hybrid
3) Too many categories increase noise → keep 3–5 and dynamic per query

### 5.5 TODO (follow-up)
1) Add hybrid schema prune/merge when ready (tree + LLM)
2) Add optional per-category fusion into retrieval (currently logging-only)
3) Add schema-specific metrics/logging (schema drift, per-category support)
4) Ablate schema refresh every iter






# 6) Open questions / decisions to make
1) **Rewrite context choice**
    - Should rewrite use leaf-only or mixed (leaf + small branch)?
2) **Rewrite timing**
    - Should we stop iterating rewrite after pre-flat? (current evidence says yes)
3) **Gate vs beam seeding**
    - Is `--seed_from_flat_gates` consistently better than allowed_prefix-only?
4) **Final ranking fusion**
    - RRF works, but should we tune k or add learned weighting?
