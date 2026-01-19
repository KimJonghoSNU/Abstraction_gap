# Shortcut Reranker (Leaf-only, No Tree)

## Goal
Test whether query rewriting alone can reduce the abstraction gap without using the tree.

Key idea:
Compare static rewrite prompts vs agentic rewrite prompts in a flat retrieval loop.


## Scope (Important constraints)
- No tree traversal is used.
- Flat retrieval targets leaf nodes only.
- Top-K documents for rewrite are leaf documents only.


## Pipeline (Conceptual)
1) Flat retrieval on leaf nodes using the current query
2) Rewrite the query using top-K leaf documents
3) Retrieve again on leaf nodes
4) Optionally repeat


## Core experiment axis
### Static prompt vs Agentic prompt
- Static: a fixed rewrite template, no iterative reasoning or control
- Agentic: prompt explicitly asks for reasoning/plan + diverse evidence targets (theory/entity/example), then rewrites

Main question:
Does agentic rewriting improve flat-only retrieval when tree signals are removed?


## What this is NOT
- Not testing gate/traversal
- Not using branch summaries
- Not mixing leaf/branch context


## Expected outputs
- Iteration-level nDCG / Recall from the leaf-only retrieval loop
- Comparison of static vs agentic prompts under identical retrieval budget
