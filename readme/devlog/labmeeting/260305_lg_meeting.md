# 2026-03-05

- 목표: abstraction gap을 해결하자
- 1) LLM의 지식 한계 때문에 구체적으로 원하는 내용을 못만든다, 2) corpus에 없는 근거를 만들면 안됨.
Tree를 쓰는 게 여전히 좋은 방향이다: 1) tree depth 2) tree width

지금 해결하고자 하는 문제: tree의 정보를 corpus feedback으로 사용할 수 있을까?
1) 어떤 내용으로 rewrite하면 좋을지 줄 수 있을까? 지금은 abstraction category: theory/entity/example 고정해서 쓰고 있음. 근데 각 branch (=하위 cluster의 요약)은 코퍼스에 어떤 내용이 들어있는지 요약해서 알려주는 역할임. 여기서 뽑은 정보로 대체할 수 있을까? -> readme/devlog/260303_round5_category_mode.md
2) 모델이 탐색해야 하는 방향을 점점 구체적으로 설정하는 역할로 쓸 수 있을까? -> readme/devlog/260302_round5.md

- tree graph는 이전에 (evolutionary mechanism, theory, example, entity...) 이렇게 묶이는 contents tree보다 (evolutionary mechanism, cognitive mechanism...) 이렇게 묶이는 category tree를 만들려고 했었는데 직접 생성 실패해서 일단 선행연구에서 제공해주는 tree를 계속 쓰고 있음.

## 실험 1. tree를 어떤 부분에서 Rewrite해야할지 결정해주는 용도로 쓰자
tree depth가 깊어지면서 retrieval search pool이 점점 줄어드는 효과임. 그럼 depth에 noise들어오는 문제는 막을 수 있다. width문제를 여전히 해결 못하고 있음. 

### 알고리즘

1. `query_pre` 결정
    - t>0에서는 이전 iteration의 `Rewritten query`, t=1일 때는 `original_query` 사용.

2. pre-rewrite retrieval (rewrite 컨텍스트 수집)
    - 검색 대상 pool은 `cumulative reached leaf pool`.
    - pool이 비어 있으면 전체 leaf를 사용.
    - `query_pre`로 top-`round5_mrr_pool_k` retrieval 수행.
    - 상위 hit의 desc를 `leaf_descs`로 만들어 rewrite 입력에 사용.

3. rewrite (`agent_executor_v1`)
    - (legacy 기준) 입력: `original_query`, `previous_rewrite`, `leaf_descs`.
    - `branch_descs`는 legacy rewrite 경로에서 비워서 전달된다.
    - 출력: `Possible_Answer_Docs`를 파싱하고 `rewrite` 문자열 생성.
    - `query_post = original_query + rewrite` (rewrite가 비면 `query_pre` 유지).

4. post-rewrite retrieval (평가용)
    - 같은 pool(`cumulative reached leaf pool`, 비면 전체 leaf)에서 `query_post`로 retrieval.
    - top-k는 `flat_topk`를 사용한다.
    - 이 결과로 nDCG/Recall/Coverage 계산.

5. branch 선택 (retriever-slate)
    - 각 sample의 현재 beam state마다 `slate`를 생성한다.
    - 기본 slate는 현재 state의 child들로 만들고, leaf-cluster 조건에서는 predicted leaf 기반 항목이 추가될 수 있다.
    - 각 slate 후보 노드 점수:
        - `score = dot(query_emb, node_emb)` (임베딩 정규화 상태라 cosine과 동일)
        - `score_01 = clip((score + 1) / 2, 0, 1)`
    - `score_01` 내림차순으로 slate ranking 생성 후, 이를 `relevance_scores`로 `sample.update(...)`에 전달.
    - `update` 내부에서 child `local_relevance`로 prediction node가 생성될 때 `path_relevance`가 계산된다:
        - `cur_rel > 0.09`이면 `alpha * cur_rel + (1 - alpha) * parent_path_rel`
        - 아니면 `cur_rel` 유지
    - branch beam 선택은 `path_relevance` 기준이다.
    - 마지막으로 expandable path를 `path_relevance` 기준으로 정렬해 top-`max_beam_size`를 다음 selected branches로 선택.


## 실험 2. category를 잘 고를 수 있냐?
- 질문: "LLM이 답 자체를 생성하기 어렵더라도, **어떤 근거 문서가 필요한지(카테고리/방향)**는 판단할 수 있는가?" & 어떤 걸 학습해야 하는가?
- 실험 초점:
    1. tree-only traversal로 retrieval을 대체할 수 있는지
    2. traversal 정책을 DPO로 학습하면 성능이 오르는지
결과: DPO 성능 학습하면 좀 오르긴 하는데, 30B의 성능과 비교해보면, 그냥 Retriever기반 선택 (as we do in round5) 이 더 성능이 좋음. + 실험 2의 결론까지 더하면 category를 잘고르는건 효과 없는걸까?

## 실험 3. category를 고른다 해서, or gold category를 고른다 해서 모델이 rewrite을 더 잘하는가?
- 아님! gold leaf-node 만 넣어주는 실험도 실패.. (agent_executor_category)
- 이전의 category selection 성능 향상은 어떻게 왔나? 임의의 3개 rewriting 후 best를 하나씩 골랐으니까?




## 2) 실험/가설/결과

### 실험 1. DPO 기반 branch 선택 학습
- 가설: branch 선택을 preference learning으로 학습하면 tree-only 성능이 오른다.
- 설정:
    - 데이터: reason-embed 기반 (각 step에서 gold sub-branch 포함 여부로 pos/neg 구성)
    - 학습: QLoRA 4bit + DPO
- 결과 (`results/BRIGHT/ndcg_summaryS=baseline1_tree_only.csv`):
    - Base(30B): overall **42.03**
    - prompt 교체 실험 row overall: **34.21**
    - Base(4B): overall **34.69**
    - DPO merged: overall **35.16**
- 해석:
    - 프롬프트 자체는 성능 향상 없음 (34.69 -> 34.21)
    - 그래도 DPO 학습이 어떤 branch가 좋은 branch인지 신호를 제공해줌 (34.21 -> 35.16)

## 실험 2.
- @run_round5_gold.py
- results/BRIGHT/ndcg_summaryround5.csv 
round5_gold-RInTP=-1-PlTau=5.0-RCF=0.5-Llm=Qwen3-30B-A3B-Instruct-2507-NumI=10-MaxBS=5-S=round5_gold_gold_branch_v1-FT=1000-PreFRS=branch-RPN=round5_agent_executor_category_v1-RM=concat-RE=1-RCT=5-RCS=mixed-RGT=10-RM=category-RCO=gold_branch_v1-RRrfK=60-RRC=leaf-REM=replace,55.87,57.51,1.0,54.52,56.59,4.0,27.58,28.71,9.0,42.35,45.24,4.0,23.87,34.41,4.0,44.492000000000004 이 실험. results/BRIGHT/biology/round5_gold 의 결과임. 성능이 오르지는 않음. 오히려 안좋음.

## 실험 3. 

상위 런 기준으로는:
- `round5_mrr_selector_accum` 런 A: overall `45.30`
- `round5_mrr_selector_accum` 런 B(최고): overall `45.77`

=> 현재 round5 legacy는 **평균 45.77까지 도달**해서, 실험적으로 효과 있는 방향이라고 볼 수 있다.

# 결론:

# Next TODO
- tree width 문제는 더 이상 해결되지 않았다?
- 학습을 한다면 category selector 만 잘해서 해결될 거 같지 않고 category select -> rewrite 까지 해야 함 
- 언제 STOP할지

Tree branch depth는 rewrite 포커스를 직접 지정하기보다, traversal이 도달하는 leaf 분포를 통해 rewrite evidence를 점진적으로 집중시키는 간접 corpus feedback 역할을 한다.
