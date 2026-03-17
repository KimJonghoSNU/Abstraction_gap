# Method

## Round5: Retriever-Guided Iterative Rewrite and Branch Selection

We describe the exact procedure implemented by `src/bash/round5/run_round5.sh` and executed in `src/run_round5.py`.
This section is intentionally implementation-faithful for reproducibility.

### Experimental Protocol

The launcher iterates over a set of BRIGHT subsets and runs one full Round5 pipeline per subset.
In the current script version, the default configuration is:

- Retriever: `Diver-Retriever-4B`
- LLM backend: `vllm`
- LLM: `Qwen3-4B-Instruct-2507`
- Number of iterations: `10`
- Number of evaluation queries per subset: `1000`
- Beam size: `10`
- Flat retrieval size for evaluation: `1000`
- Rewrite context size: `10`
- Local pool size for branch scoring: `100`
- Calibration: disabled by default

The launcher loops over:

1. Selector mode (`ROUND5_SELECTOR_MODES`, default `meanscore_global`)
2. Query source (currently fixed to `original` in the script)
3. Subset (`biology`, `psychology`, `economics`, `earth_science`, `robotics`, `sustainable_living`, `stackoverflow`, `theoremqa_questions`, `theoremqa_theorems`, `pony`)

Each subset is mapped to a fixed tree version (bottom-up or top-down), and precomputed node embeddings are required.
The script is fail-fast: if any single run fails, execution stops.

### Rewrite Prompt Setting

By default, `run_round5.sh` passes:

- `--rewrite_prompt_name agent_executor_v1_icl2`

This is an important implementation detail because reported results are sensitive to this default.
In Round5 legacy mode, the prompt is used directly if it exists in `REWRITE_PROMPT_TEMPLATES`.

### Runtime Objective

Given a query, Round5 alternates between:

1. Evidence-aware query rewriting
2. Leaf retrieval for evaluation
3. Beam update over branch states (baseline retriever-slate update)
4. Optional global branch override by a selector policy

The system stores per-iteration retrieval and branch diagnostics in `iter_records`.

## Iterative Inference Procedure

Let `t = 0, ..., T-1` with `T = 10`.
For each query and iteration `t`, the implementation performs:

1. **Cumulative leaf-pool update**
   - Leaves reached so far are accumulated in a per-query set.

2. **Rewrite-context retrieval (pre-rewrite query)**
   - Retrieve top-`K_local` leaf hits from the cumulative pool using the current query, where `K_local = round5_mrr_pool_k` (default `100`).
   - Use top `K_ctx = rewrite_context_topk` leaf descriptions (default `10`) as rewrite context.

3. **Query rewrite**
   - Construct rewrite prompt from:
     - original query,
     - previous rewrite,
     - retrieved leaf summaries.
   - Apply rewrite cache if available; otherwise call LLM with deterministic decoding (`temperature = 0.0`).
   - Parse `Possible_Answer_Docs` and flatten to a rewrite string.

4. **Evaluation retrieval (post-rewrite query)**
   - Compose retrieval query as `original_query + rewrite`.
   - Retrieve top-`K_flat` leaves from the cumulative pool (`K_flat = flat_topk`, default `1000`).
   - Compute per-query metrics: `nDCG@10`, `Recall@10`, `Recall@100`, `Recall@all`, `Coverage`.

5. **Baseline branch update (retriever-slate)**
   - Construct traversal slates from current beam state.
   - Score slate candidates with retriever embeddings.
   - Update beam using retriever-ranked slates.

6. **Optional selector override**
   - If selector mode is not `retriever_slate`, override the next branch set using global branch scoring from local leaf evidence.

7. **Logging and persistence**
   - Save rewrite outputs, selected branches, candidate branch scores, retrieved doc IDs, and metrics into `iter_records`.

## Selector Policies

Round5 supports four selector modes:

- `retriever_slate`
  - No global override; uses baseline retriever-slate beam update.

- `maxscore_global`
  - For each candidate child branch, score by the maximum matched leaf score.

- `meanscore_global`
  - For each candidate child branch, score by the mean matched leaf score.

- `max_hit_global`
  - For each candidate child branch, score by the number of matched local hits.

Candidate branches are the direct children of currently selected branches (or root children at iteration 0).
Top-`B` branches (`B = max_beam_size`) are selected globally; if fewer than `B` are matched, the implementation partially fills with baseline beam paths.

## Output Artifacts

For each run, the runtime stores:

- `all_eval_sample_dicts.pkl`
- `all_eval_metrics.pkl`
- `llm_api_history.pkl`
- `hparams.json`
- `run.log`

under:

`results/BRIGHT/<subset>/round5/<exp_dir>/`

where `<exp_dir>` is generated from serialized hyperparameters.

## Reproducibility-Critical Notes

- The current launcher default uses `agent_executor_v1_icl2`; this is not neutral and should be explicitly declared in experiments.
- Query source is currently fixed to `original` in `run_round5.sh`.
- Calibration is disabled by default via launcher environment variable.
- The script expects valid node embedding files per subset (`node_embs.diver.npy`).

## Main-Paper Version (Condensed)

### Method Overview
We use an iterative retrieval framework that alternates between query rewriting and branch selection over a hierarchical corpus tree.
At iteration `t`, the model first gathers leaf-level evidence from a cumulative pool of previously reached leaves and rewrites the query using this evidence.
The rewritten query is then used for leaf retrieval and for branch expansion under a global selector policy.

### Iterative Procedure
For each query and iteration:

1. Retrieve local leaf evidence from the cumulative reachable leaf pool.
2. Generate a rewrite conditioned on the original query, previous rewrite, and retrieved evidence.
3. Form the retrieval query by concatenating the original query and rewrite.
4. Retrieve top leaf candidates with this rewritten query.
5. Update branch expansion using one of the global selector policies:
    - `maxscore_global`: branch score = maximum matched leaf score
    - `meanscore_global`: branch score = mean matched leaf score
    - `max_hit_global`: branch score = number of matched local hits

### nDCG@10 Definition and Reuse
`nDCG@10` is computed from the ranked leaf documents returned in **Step 4** at each iteration, using query-specific gold documents.
The per-query score is then aggregated across queries (iteration mean), and final reporting uses the standard run-level aggregation protocol.

A practical implication is that Step 1 and Step 4 use the same retrieval query across adjacent iterations:
- Step 4 at iteration `t-1` and Step 1 at iteration `t` share the same rewritten query.
- Therefore, Step 1 retrieval can be reused for `t>0`.
- Iteration `t=0` is the only exception, because no previous rewritten-query retrieval exists.

## Main-Paper Draft (Merged Retrieval Version)

The intent is to make explicit that both stages use the same retriever, while the query state and reachable pool change with the search state.

```latex
\subsection{Problem Setup}
\label{sec:problem_setup}

Let $q$ be an input query and let $\mathcal{T}$ be a fixed hierarchical corpus tree whose leaves correspond to documents and whose internal nodes correspond to clusters of documents. We assume that the tree is given and focus on inference over the tree, rather than tree construction itself.

At iteration $t$, the system maintains a beam of active branches $B_t$, a reachable leaf pool for rewrite-context retrieval $\mathcal{P}^{\mathrm{ctx}}_t$, a reachable leaf pool for ranking and branch expansion $\mathcal{P}^{\mathrm{ret}}_t$, and a rewritten evidence description $r_t$. The initial rewrite is empty, and the initial retrieval query is the original user query.

The key design choice is that retrieval is not performed over the whole corpus after the first step. Instead, the retrieval pool is constrained by the current search state. At the first iteration, the reachable pools contain all leaves in the corpus. At later iterations, the pools are restricted by the currently selected branches in the beam. Once traversal reaches the leaf level, the reachable pool can be maintained as the union of previously reached leaves and the current frontier. This turns the tree into a state-dependent feedback boundary for iterative rewriting.

\subsection{Tree-Constrained Adaptive Retrieval and Branch Expansion}
\label{sec:iterative_retrieval}

We run the iterative process for $T$ steps. At each iteration, the method performs two retrieval calls of the same form: one before rewriting to gather local evidence, and one after rewriting to rank leaves and expand branches. The retrieval model is the same in both cases; only the query state and the reachable pool differ.

Let $u_t$ denote the current retrieval query before rewriting. At the first iteration, $u_0 = q$. At later iterations, the current query is the rewritten query from the previous step,
\begin{equation}
u_t = q \oplus r_{t-1} \qquad (t > 0),
\end{equation}
where $\oplus$ denotes string concatenation.

We first retrieve local evidence leaves from the currently reachable context pool:
\begin{equation}
\mathcal{E}_t = \mathrm{Ret}_k(u_t, \mathcal{P}^{\mathrm{ctx}}_t),
\end{equation}
where $\mathrm{Ret}_k(\cdot,\cdot)$ returns the top-$k$ leaves under the chosen retriever.

After obtaining the rewrite $r_t$, we form the post-rewrite retrieval query
\begin{equation}
\tilde{q}_t = q \oplus r_t,
\end{equation}
and apply the same retriever again over the current ranking pool:
\begin{equation}
\mathcal{H}_t = \mathrm{Ret}_m(\tilde{q}_t, \mathcal{P}^{\mathrm{ret}}_t),
\end{equation}
where $\mathcal{H}_t$ denotes the top matched leaves. These leaves serve two roles: they are the ranked retrieval outputs used for evaluation, and they provide the matched evidence used to score candidate branches.

Let $\mathcal{C}_t$ denote the candidate branches at iteration $t$, defined as the children of the branches currently selected in the beam. For each candidate branch $c \in \mathcal{C}_t$, we score the branch using the matched leaves in $\mathcal{H}_t$ that fall under $c$. Let $\mathcal{L}(c) \subseteq \mathcal{H}_t$ be those descendant hits. A max-based selector is
\begin{equation}
s_{\max}(c) = \max_{\ell \in \mathcal{L}(c)} \mathrm{sim}(\tilde{q}_t, \ell),
\end{equation}
and a mean-based selector is
\begin{equation}
s_{\mathrm{mean}}(c) = \frac{1}{|\mathcal{L}(c)|} \sum_{\ell \in \mathcal{L}(c)} \mathrm{sim}(\tilde{q}_t, \ell),
\end{equation}
where $\mathrm{sim}(\cdot,\cdot)$ is the retriever similarity score. The top $B$ branches are then selected as the beam for the next iteration.

This design makes the tree a state-dependent feedback boundary for both rewriting and retrieval. The locality constraint improves grounding, but it also introduces path dependence: if an early branch choice is wrong, subsequent evidence and rewrites are conditioned on a biased local region.

\subsection{Grounded Query Rewriting}
\label{sec:rewriting}

The rewrite module is designed to produce not an answer, but descriptions of the evidence that would support the answer. Given the original query $q$, the previous rewrite $r_{t-1}$, and the locally retrieved evidence leaves $\mathcal{E}_t$, the model generates
\begin{equation}
r_t = f_{\mathrm{rw}}(q, r_{t-1}, \mathcal{E}_t),
\end{equation}
where $f_{\mathrm{rw}}$ is an instruction-tuned language model.

This design directly targets the abstraction gap by encouraging the model to search for answer-supporting evidence rather than documents that merely overlap lexically with the query. The useful evidence may exist at a more abstract level than the user query itself. For example, a query about counting bees entering and leaving a hive may require a theoretical concept such as a Poisson distribution, even though the query never mentions that concept explicitly.

At the same time, unconstrained rewriting can generate abstractions that do not correspond to any recoverable corpus concept. Our method mitigates this by conditioning rewriting only on evidence retrieved from the currently reachable tree region.
```

## Algorithm Draft (LaTeX)

This version is intended for the main paper and follows the merged retrieval formulation above.
It uses standard `algorithm` + `algpseudocode` notation.

```latex
% Requires:
% \usepackage{algorithm}
% \usepackage{algpseudocode}

\begin{algorithm}[t]
\caption{Tree-Constrained Adaptive Retrieval with Grounded Query Rewriting}
\label{alg:tree_constrained_adaptive_retrieval}
\begin{algorithmic}[1]
\Require query $q$, corpus tree $\mathcal{T}$, retriever $\mathrm{Ret}$, rewrite model $f_{\mathrm{rw}}$, beam size $B$, iterations $T$, evidence budget $k$, ranking budget $m$
\State $r_{-1} \gets \emptyset$
\State $B_0 \gets \{\mathrm{root}\}$
\State $\mathcal{L}_{\mathrm{seen}} \gets \emptyset$
\For{$t = 0$ to $T-1$}
    \State $(\mathcal{P}^{\mathrm{ctx}}_t, \mathcal{P}^{\mathrm{ret}}_t) \gets \textsc{ReachablePools}(B_t, \mathcal{L}_{\mathrm{seen}}, \mathcal{T})$
    \If{$t = 0$}
        \State $u_t \gets q$
    \Else
        \State $u_t \gets q \oplus r_{t-1}$
    \EndIf
    \State $\mathcal{E}_t \gets \mathrm{Ret}_k(u_t, \mathcal{P}^{\mathrm{ctx}}_t)$
    \State $r_t \gets f_{\mathrm{rw}}(q, r_{t-1}, \mathcal{E}_t)$
    \State $\tilde{q}_t \gets q \oplus r_t$
    \State $\mathcal{H}_t \gets \mathrm{Ret}_m(\tilde{q}_t, \mathcal{P}^{\mathrm{ret}}_t)$
    \State compute retrieval metrics from $\mathcal{H}_t$
    \State $\mathcal{C}_t \gets \textsc{Children}(B_t)$
    \ForAll{$c \in \mathcal{C}_t$}
        \State $\mathcal{L}(c) \gets \{\ell \in \mathcal{H}_t : \ell \text{ is a descendant of } c\}$
        \State $s(c) \gets \textsc{BranchScore}(\tilde{q}_t, \mathcal{L}(c))$
    \EndFor
    \State $B_{t+1} \gets \textsc{TopB}(\mathcal{C}_t, s, B)$
    \State $\mathcal{L}_{\mathrm{seen}} \gets \mathcal{L}_{\mathrm{seen}} \cup \mathcal{H}_t$
\EndFor
\State \Return $\{\mathcal{H}_t\}_{t=0}^{T-1}$, $\{B_t\}_{t=1}^{T}$, $\{r_t\}_{t=0}^{T-1}$
\end{algorithmic}
\end{algorithm}
```

Notes:

- `\textsc{ReachablePools}` returns all corpus leaves at `t=0`; afterwards it restricts retrieval to the tree region induced by the current beam and the reached leaf set.
- `\textsc{BranchScore}` can instantiate the selector in the text, e.g. `max` or `mean` over descendant-hit similarities.
- If space is tight, the metric-computation line can be dropped from the algorithm and left in the main text.
