# QD Abstraction Integration (Working Note)

# Round 1

## 1) Motivation (Why this integration)
The current concern is whether **abstraction gap** can be solved by:
- tighter coupling between the document abstraction (LATTICE's tree structure) and query abstraction (query rewriting to define what evidence to show)

We want a pipeline that reliably jumps to the correct abstraction level without getting trapped by feedback locality.


## 2) pipeline variants being compared

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

## baselines
### 1) Tree traversal only (no flat, no rewrite)
    - Goal: isolate pure tree navigation without any flat anchors or query rewriting.
    - Implementation: `src/run.py` without `--flat_then_tree`, no `--qe_*`, no `--rewrite_*`.
    - Script: `src/bash/baselines/run_baseline1_tree_only.sh`.

### 1b) Tree traversal + iterative rewrite (leafslate context)
    - Goal: test per-iteration rewrite in traversal without flat, using leaf hits + branch slate context.
    - Implementation: `src/run.py` with `--rewrite_prompt_name thinkqe`, `--rewrite_every 1`, and `--rewrite_context_source leafslate`.
    - Context: traversal leaf hits (if any) + branch nodes from current slates.
    - Script: `src/bash/baselines/run_baseline1_tree_iter_rewrite.sh`.

### 2) QE-only rewrite, then pure traversal (no flat, no context)
    - Goal: test context-free query rewriting impact before traversal.
    - Implementation: `src/run.py` with `--qe_prompt_name thinkqe`, `--flat_then_tree` off, and no traversal rewrite.
    - Script: `src/bash/baselines/run_baseline2_qe_noctx.sh`.

### 2b) QE-only rewrite (agent_executor_v1), then pure traversal (no flat, no context)
    - Goal: compare a different no-context QE prompt against thinkqe.
    - Implementation: `src/run.py` with `--qe_prompt_name agent_executor_v1`, `--flat_then_tree` off, and no traversal rewrite.
    - Script: `src/bash/baselines/run_baseline2_qe_agent_executor.sh`.

### 3) Leaf-only retrieve ↔ rewrite loop (leaf-only context)
    - Goal: evaluate iterative rewrite with only leaf evidence (no branch leakage).
    - Implementation: `src/run_leaf_rank.py` with `--leaf_only_retrieval` and `--rewrite_prompt_name thinkqe`.
    - Script: `src/bash/baselines/run_baseline3_leaf_only_loop.sh`.

## 3) Observations so far (from ndcg_summary.csv)
Summary trends:
- **Leaf-only pre-flat rewrite is strongest** on average...
- **Iterative rewrite on top of branch-only** does not add value and often hurts.

Interpretation:
- **Branch-only context for rewriting is likely harmful or too abstract**, especially when rewrite drives the retriever into the wrong region.
- Leaf evidence seems to ground rewrite better than branch summaries.
- 즉 query rewrite -> Tree traversal 을 더 잘하게 해줌. 하지만 역방향으로, tree의 정보가 query rewrite를 더 잘하게 해주지는 않음.

The current results suggest:
- **branch summaries alone are not reliable bridge signals**
- **leaf evidence is a better anchor for rewrite**, at least in the current model + prompt setup



# Round 2

## 5) Results (schema_d1 runs in ndcg_summary.csv)
Scope: schema_d1 experiments (LLM-generated schema from depth=1 branch retrieval).

### Overall trends (mean across biology + psychology)
- **PreFRS=leaf** is best: ndcg@10 iter0/max = **49.55 / 52.66**
- **PreFRS=all** is second: **50.40 / 52.19**
- **PreFRS=branch** is worst: **47.64 / 50.55**
- **Rewrite_every=0** slightly > **rewrite_every=1** (means: **49.46 / 52.14** vs **48.80 / 51.52**)

### Per-subset trends
- **Biology**: leaf > all > branch (iter0/max)
    - leaf **56.74 / 59.77**, all **57.79 / 59.25**, branch **54.22 / 57.17**
- **Psychology**: leaf > all > branch (iter0/max)
    - leaf **42.36 / 45.54**, all **43.01 / 45.13**, branch **41.05 / 43.93**

### Interpretation
- Even with schema (depth=1), **branch-only rewrite context is still weakest**.
- **Leaf-only remains the safest grounding signal**, schema or not.
- Iterative rewrite (RE=1) does **not** consistently help; average is slightly worse than RE=0.

## 5) Implementation Plan (Draft)
Goal: use tree summaries to define **abstraction schema** (categories) while keeping **rewrite evidence grounded in leaf docs**, based on current results.

### 5.1 Design choices (default + ablations)
2) **Schema scope**
    - Default: **intermediate children of flat-retrieved branches** (using all depth=1 branches)
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
    - Rewrite timing is still controlled by `rewrite_every` for ablation

### 5.2 Proposed pipeline (high level)
1) **Flat retrieval (all nodes)** on original query
2) **Schema induction** from flat branch hits
3) **Rewrite** using:
    - schema as structure
    - **leaf-only** evidence from flat results as grounding
4) **Leaf-only retrieval** (evaluation uses leaf-only)
5) Optional: **traversal** with `--seed_from_flat_gates` (ablation) -> horrible performance. I assume that it is diffcult to mix score of flat retrieval + from tree traversal


### 5.4 Risks / critique (why this direction is safer)
1) Branch-only evidence previously degraded results; schema-only use avoids that
2) LLM schema can drift from branch evidence; log schema drift + consider hybrid
3) Too many categories increase noise → keep 3–5 and dynamic per query

### Results
- 전체 branch로 schema rewrite하는 것보다는 성능 오름. 그런데 안쓰느니만 못함.
- 분석: depth=1에서 제공해주는 내용이 query abstraction과는 맞지 않음. 
    - LATTICE에서 트리를 만드는 방식: 같은 내용을 묶음. abstraction category (theory, entity..) 로 묶이지 않음.
    - 그래도 tree 쓰는 거 정당화 가능한지? corpus keyword extraction (rewritten query content 채우기 위해 필요) vs category abstraction 각각의 효과 모두 필요하다?

```
### LATTICE tree generation
You are an expert AI analyst and summarizer. Your mission is to create a highly informative and "
discriminative signpost" for a navigating search agent. This signpost (a summary) must guide the agent to
the correct cluster of nodes to answer a user’s query.
You will follow a strict, step-by-step cognitive process. You must analyze the children nodes in a target
parent node (the "Positive Set").
Prompt ID: {prompt_id} (ignore, this is just for watermarking purposes).
## INPUTS
### POSITIVE SET: Information about the target parent node to be summarized
{positive_set_descriptions}
---
## YOUR TASK & OUTPUT FORMAT
Your entire output must be a single, valid JSON object. Inside this JSON, you will follow the 3-step
thinking process outlined below, populating each field as instructed.
### JSON Structure and Instructions:
{{
"detailed_fingerprints": [
// For EACH children node in the POSITIVE SET (target parent node), extract a structured object of its
key, queryable facts.
{{
"one_line_summary": "...", // write a very information dense and very concise one-line summary for
the information contained in this node
"key_entities": ["..."], // List a very few key entities which is central to this node
"genre_or_category": ["..."], // List a few key genre / categories this node can be classified into
"name": "...", // Name the node
}}
],
"common_theme": "...", // Reason deeply what are the common themes between the nodes in the POSITIVE SET
"summary": "...", // Based on step 1 and step 2, write a very information dense description of the target
node, **make sure to include all key entities**.
```

### Next Round

- tree -> query rewrite쓰는 효과: 검색 pool이 좁아지는 효과 (1 depth tree로 전부 변경해서 검색한다고 생각하자. flat retrieval -> feedback -> flat retrieval 반복.)
- Algorithm
    - flat retrieval -> leaf node들 + 걸리는 모든 branch들의 leaf node들로 ndcg@10 측정 = final metric. branch의 leaf node들은 점수를 어떻게 다시 매길지 고민.
    - topk leaf node들 (or topk leaf + topk branch 조합. 각각 leaf, branch라고 표시해야할지 고민.)이 query rewriter feedback으로 들어감
    - 여기서부터가 고민인데, 다음 retriever corpus pool은 이전 flat retrieval 결과와 같은 prefix를 공유하는 노드들에 대해서만 할 지, 아니면 iteration i 마다 depth-i인 노드들을 제외하고 점점 deeper tree node search를 할 지.
    - 위 내용 반복


- rewrite할 때 branch node / leaf node 구분해서 주기 vs leaf node만 주기 ablation


## Round 3


Phase 1: Initial Observation and Gate Induction
1. Global Flat Retrieval

Global flat retrieval을 수행한다.
쿼리는 Query_t를 사용한다.
Query_t는 Query0와 현재까지 생성된 Q_rewritten을 결합한 것이다.
인덱스는 leaf와 branch가 함께 있을 수 있다.

2. Output Separation

Top K_anchor 결과에서 leaf와 branch를 분리한다.
L_init는 Top K_anchor leaf 결과다.
B_hit는 Top K_anchor 결과 안에 포함된 branch 결과의 prefix 집합이다.
branch node가 검색되더라도 L_init에는 포함하지 않는다.

3. Active Branches Construction

L_init에 포함된 leaf들의 prefix 전부를 모아 P_leaf로 둔다.
ActiveBranches B_active는 P_leaf와 B_hit의 합집합이다.

- anchor = top‑K over all nodes
- global = top‑K over leaf subset
- local = top‑K over prefix‑filtered leaf subset

4. Density Check

각 candidate prefix u에 대해 density(u)를 계산한다.
density(u)는 L_init 중 prefix가 u인 비율이다.
density는 Phase 2에서 explore와 exploit의 분기 신호로 사용한다.
density는 Phase 3에서 local과 global fusion 조절 신호로 사용할 수 있다.

TODO drift signal을 정의해서 explore와 exploit 분기를 더 안정화한다.
TODO conflict signal을 정의해서 local과 global 충돌을 반영한다.

Phase 2: Interaction and Rewriting
5. Schema Optional Rewrite

입력은 다음으로 구성한다.
Original Query0.
Evidence로서 L_init의 leaf 텍스트.
Optional context로서 B_active의 branch description.

Rewriter는 다음을 출력한다.
Rewritten Query Q_rewritten.
Action a는 explore 또는 exploit 중 하나다.

exploit은 현재 B_active가 맞는 지역이라고 보고 구체화를 시도한다.
explore는 B_active가 불완전하거나 틀렸을 수 있다고 보고 탈출을 시도한다.

Phase 3: Hybrid Retrieval
6. Dual Path Retrieval

두 개의 경로로 retrieval을 수행한다.

Path A: Local Exploit Search.
Pool은 B_active의 descendants로 만든다.
descendants에는 하위 branch와 leaf를 포함한다.
단 B_active에 속한 branch 노드 자체와 그 상위 노드들은 pool에서 제외한다.
쿼리는 Query_t를 사용한다.

Depth 제약을 둔다.
처음 선택된 active branch들의 depth가 d라면, Path A에서 depth d에 해당하는 branch 노드들은 제외한다.
그 아래 노드들만 대상으로 한다.
목적은 같은 추상 레벨에서 맴도는 locality를 줄이고 더 구체 레벨로 내려가게 하는 것이다.

Path A 결과 처리 규칙을 둔다.
Path A의 retrieval 결과에서 branch 노드는 즉시 제거한다.
leaf 결과만 fusion 단계로 전달한다.

Path B: Global Escape Search.
Pool은 전체 leaf다.
쿼리는 Query_t를 사용한다.
출력은 Top K_global만 유지한다.
K_global은 10 정도로 둔다.

7. Score Fusion

Path A와 Path B 결과를 RRF로 결합한다.
결합 결과를 Final Ranked List로 둔다.
Final Ranked List는 leaf만 포함한다.

density 기반으로 local과 global 중 어느 쪽을 더 믿을지 조절하는 것은 옵션으로 둔다.
우선은 고정 RRF로 시작한다.
추후 density를 이용한 가중 fusion은 ablation으로 넣는다.

Phase 4: Evaluation and Looping
8. Offline Metric Calculation

Final Ranked List의 Top K_eval leaf를 기준으로 ndcg를 계산한다.
branch 노드는 평가에서 제외한다.

9. Next Round Update

다음 라운드는 Phase 1부터 다시 수행한다.
gate는 매 라운드 새로 유도된다.
Query_t는 매 라운드 업데이트된 Q_rewritten을 반영한다.

Action이 exploit이면, 다음 라운드에서도 local이 강하게 작동할 가능성이 있으므로 depth 제약을 유지한다.
Action이 explore이면 local 신뢰가 낮다고 보고 global 결과가 다음 라운드 B_active 선정에 더 크게 반영되도록 한다.
구체 규칙은 TODO로 남긴다.

TODO explore에서 B_active 업데이트 규칙을 정량화한다.
TODO drift signal.
TODO conflict signal.

## Round 3 Implementation Notes (260122)

- 구현 파일: `src/run_round3.py`
    - traversal 없이 round3 전용 파이프라인만 수행
    - Query_t = original + rewrite (explore에서도 concat? replace가 더 성능 잘 나옴)
    - rewrite_every=0이면 **rewrite 비활성화**
    - leaf depth는 현재 > 1 전제 (prefix 누락 가능성 주석 추가)
- Prompt: `round3_action_v1` in `src/rewrite_prompts.py`
    - stepback_json 스타일 (Plan + Possible_Answer_Docs)
    - abstract evidence 강조, lexical overlap 불필요
    - exploit은 evidence key term을 anchor로 유지하되 abstractive 유지
    - explore는 이전 rewrite를 negative constraint로 취급
- Rewrite context ablation:
    - `--round3_rewrite_context leaf` (leaf-only)
    - `--round3_rewrite_context leaf_branch` (leaf + branch, prompt에서 label 분리)
    - branch_descs가 비면 prompt에서 Branch Context 블록 제거
- Retrieval:
    - Anchor flat retrieval로 L_init / B_hit 분리
    - ActiveBranches = leaf prefixes + branch hit prefixes
    - Local pool = B_active descendants (leaf-only), global pool = all leaf
    - Local + Global RRF fusion only (tree traversal 없음)
- Cache:
    - rewrite cache에 action + rewritten_query 저장
- Scripts:
    - `src/bash/run_round3_ablation.sh` (leaf vs leaf_branch ablation)

### Round 3 (Current Implementation)

- Rewrite는 `Possible_Answer_Docs`를 concat한 문자열로 구성하고, 항상 `original + rewrite`로 검색한다.
- Action은 prompt에 포함되지만, 현재는 **pool/weight에 영향 없음** (local/global 균형은 고정).
- Local pool은 `B_active` prefix의 leaf만 대상으로 한다.
- Depth 제약(같은 depth branch 제외)은 **미구현**.
- density/drift/conflict signal은 계산/로그만 하고 **행동에 반영되지 않음**.

### Results [WIP]

- Leaf‑only context still wins.
    round3_leaf > round3_leaf_branch in both biology and psychology.
    That matches earlier results: branch evidence tends to hurt rewrite quality.
- Explore‑mode = original is clearly bad.
    round3_explore_original is the worst in both biology (54.35 max) and psychology (41.76 max).
    Keeping original for explore seems to prevent escaping bad anchors.
- Per‑level actions underperform.
    round3_action_levels_v1 is noticeably worse than single‑action (round3_action_v1) in both domains.
    Likely too complex for the model right now or PRUNE is dropping useful categories.
- Psychology stagnates after iter 0.
    Best psychology runs are effectively iter‑0 best (no improvement across iterations).
    Biology improves with iterations, psychology doesn’t → rewrite drift or mismatch.

## Round 3 Comparison Table (ndcg_summaryround3.csv)

round3_action_v1 explore=replace	58.58,62.87,3	44.17,44.17,0
round3_action_v1 explore=concat	57.18,61.56,1	41.92,44.73,2
round3_action_levels_v1 explore=replace	54.57,58.77,6	43.41,43.81,2

DONE
- Per-level action (category별 EXPLORE/EXPLOIT) 적용

TODO
- density 기반 weighted fusion

## Round 3-1 (Missing to Implement)

1. Depth constraint in local pool
    - Active branch depth가 d일 때, local pool에서 depth d의 branch layer는 제외하고 더 아래 depth만 검색.
2. Action-aware pool balancing
    - EXPLORE: global 비중 증가 (RRF 가중치 or pool 구성 변경).
    - EXPLOIT: local 비중 증가.
3. Action-aware B_active update
    - EXPLORE일 때 global top-K prefix가 다음 라운드 B_active에 더 크게 반영.
4. Density/Drift/Conflict signals
    - density 기반으로 explore/exploit 결정 안정화.
    - drift/conflict 지표 정의 후 action/weight에 반영.
5. Weight fusion ablation
    - density를 이용한 가중 RRF vs 고정 RRF 비교.


### Round 3-1 방향 (Round 3와의 차이)

- Rewrite context가 action에 따라 분기됨.
    - EXPLOIT: local leaf (B_active descendants) top‑K만 사용.
    - EXPLORE: local leaf top‑K + (global−local) top‑K를 함께 사용.
    - 두 컨텍스트는 프롬프트에서 라벨로 분리해서 제공.
- global−local 컨텍스트가 부족하지 않도록 global 후보는 더 크게 뽑고(확장 top‑K),
  거기서 local을 제외한 상위 K개를 사용.
- Action 결정은 LLM 출력(action)만 사용. (density/drift/heuristic override 없음)
- Local/Global fusion은 기존과 동일하게 고정 RRF로 유지. (비중 조절 미적용)

#### --round3_anchor_local_rank
- 기본: local rank는 local leaf top‑K를 점수 순으로 정렬.
- 이 옵션을 켜면: **anchor top‑K의 순서를 따라 local rank를 재구성**.
    - anchor top‑K에 leaf가 있으면 그대로 포함.
    - anchor top‑K에 branch가 있으면, 그 branch 하위 leaf 중 **점수가 가장 높은 leaf**로 대체.
    - 중복이면 다음으로 높은 leaf를 선택.
    - 부족하면 기존 local hits에서 높은 점수 순으로 채움.

#### Oracle branch (run_oracle_branch.py)
- 목적: `--round3_anchor_local_rank`의 branch→best‑leaf 규칙이 **정확히 작동했을 때** 성능이 얼마나 개선되는지 상한을 측정.
- 구현 개요:
    - anchor top‑K에서 **branch는 해당 branch 하위 gold leaf 중 점수가 가장 높은 leaf로 치환**.
    - branch 하위에 gold가 없으면 **그 branch는 버림**.
    - 중복 leaf는 제거하고, **anchor 순서를 유지**.
    - 이 oracle paths 중 **top‑K(`rewrite_context_topk`)가 metric을 계산하는 top-K가 됨. 다음 iteration rewrite context로 사용** (v2일 때만).
- 보고 지표:
    - `AnchorBranchHitRatio@K`: top‑K branch 중 gold를 포함하는 비율(샘플 평균).
    - `Oracle_nDCG@10`: oracle‑치환 리스트의 nDCG@10.
    - @100 버전도 함께 기록(`AnchorBranchHitRatio@100`, `Oracle_nDCG@10@100`).

#### Oracle (upper bound) variant:
##### 1: local/global
    - per‑query로 local/global/rrf 중 nDCG@10이 최대인 리스트를 선택.
    - 다음 iteration의 rewrite context는 이 oracle‑selected 리스트의 leaf들로 구성.
    - gate/pool은 기존 anchor 기반 유지 (oracle는 context만 교체).
##### 2: exploit/explore
    - (action‑oracle) EXPLOIT/EXPLORE용 rewrite를 각각 생성하고,
      두 쿼리로 retrieval을 수행한 뒤 nDCG@10이 더 높은 쪽을 선택.
      선택된 rewrite가 다음 iteration의 context를 결정한다.

### 정리, TODO 고민

- explore/exploit 잘 고르면 효과 있음 (44.12 -> 50.71). local/global 중 어떤 걸 더 볼지는 효과 없음 
- branch를 잘 고르기 <- 잘 고르면 효과 있는데 어떻게 잘 고를지가 어려운 문제 
- 성능 빠르게 올릴 방법: 

#### query signal로 rewrite를 어떻게 할지 확인
- branch signal이 explore / exploit과 관련이 있을지 확인
1) Dmax로 oracle 예측이 되는지 확인
    - Dmax: L_init에서 가장 많이 나온 prefix의 비율
    - Density (Dmax): Dmax는 한 라운드에서 나온 **Top K_anchor leaf 결과가 특정 prefix 한 곳에 얼마나 “몰려 있는지”**를 수치로 만든 거
    - 결과: 실패 (correlation 0)
2) prefix entropy
    - L_init prefix 분포의 엔트로피
    - 높을수록 방향이 불안정
    - oracle explore와 양의 상관이 기대됨
    - 결과: 실패 (correlation 0)

- action별 평균 overlap/drift (explore vs exploit) 를 집계: 구현 중 




<memo>
핵심
컨트롤러 목표: 현재 라운드 관측이 제대로 된 region에 밀집했는지 아니면 탈출이 필요한지 판단한다

- 왜 corpus tree를 써야 하냐?
    - locality에서 벗어나기 위한 신호 (branch node + leaf node 같이 제공)
    - 현재 스캔 중인 문서 풀이 아닌 문서 풀들은 어디 있냐? 에 대한 답
    - rewrite 할 때는 leaf node만 보여주고 rewrite
    - anchor (leaf+branch) node의 하위 leaf 노드에 대해서만 retrieval: local search
    - 전체 leaf 노드에 대해서만 retrieval: global search
    - (현재는 둘을 RRF 하고있는 상황)

- local / global, explore / exploit 활용법:
    - 기본 idea: 
        - explore뜨면 global에서 정보를 더 가져오기?
        - exploit에서는 local에서 query refine?
- 사실 지금 3개로 나눠서 봐야 함
    - local leaf(=anchor에서 leaf node들)
    - anchor's branch leaf - local leaf (=anchor에서 branch node밑에 딸려 있는 leaf node들)
    - global leaf - (anchor+local) leaf
- explore해야 한다고 뜨면, rewrite context를 top anchor leaf에서 받아서 rewrite context로 넣어주기 or global leaf에서 받아서 넣어주기?
    - top anchor leaf 점수 계산 방법: anchor branch로부터 leaf 까지 path에 등장하는 값들의 평균
    - 지금 확인한 건, explore, exploit을 개잘하면 성능이 오른다












<!-- ## Current best setting
- biology: S=flat_gate_qe_iter-FTT=True-FT=100-GBT=10-QePN=pre_flat_rewrite_v1-QeCP=biology_pre_flat_rewrite-RPN=gate_rewrite_v1-RM=concat-RE=1-RCT=5-RCS=fused-RAtS=True 63.xx
- psychology: S=flat_gate-FTT=True-FT=100-GBT=10-QeCP=psychology_converted_qe_woplan.log 48.xx -->




<!-- # 6) Open questions / decisions to make
1) **Rewrite context choice**
    - Should rewrite use leaf-only or mixed (leaf + small branch)? -> leaf only
2) **Rewrite timing**
    - Should we stop iterating rewrite after pre-flat? (current evidence says yes) 
3) **Gate vs beam seeding**
    - Is `--seed_from_flat_gates` consistently better than allowed_prefix-only? -> no
4) **Final ranking fusion**
    - RRF works, but should we tune k or add learned weighting? -->


