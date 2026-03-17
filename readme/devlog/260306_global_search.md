네 구현을 보면 rewrite context는 현재까지 도달한 leaf들의 cumulative pool에서 가져오고, 다음 step의 후보 branch도 현재 선택된 branch의 child들로 제한돼 있어. 그래서 한 번 잘못 들어간 branch는 이후 rewrite와 branch selection 둘 다 같은 지역 정보에 의해 계속 갱신될 가능성이 크다. 즉, 오류가 한 단계짜리 오류가 아니라 self-reinforcing 오류가 된다. 

## Pagerank (random jump)
왜 corpus grounded neighborhood 밖의 feedback이 문제인가

쉽게 말하면, rewrite는 검색 엔진에게 “이런 종류의 근거를 찾아라”라고 방향을 주는 단계야. 그런데 그 방향을 정할 때 지금 실제로 탐색 중인 코퍼스 주변과 무관한 힌트가 섞이면, 모델은 그럴듯하지만 현재 코퍼스에서 회수 불가능한 방향으로 query를 바꿔버릴 수 있어.

예를 들어 원래 질문이 어떤 현상을 설명하는 원리를 찾는 건데, 현재 선택된 branch 아래에는 그 원리를 직접 설명하는 문서가 없고 대신 표면적으로 비슷한 사례 문서들만 있다고 해보자. 이때 neighborhood 밖에서 들어온 힌트가 우연히 강하게 보이면, 모델은 “아, 이 질문은 이 사례를 더 파면 되겠구나”라고 rewrite할 수 있다. 문제는 그 순간부터 retrieval은 그 rewrite를 따라가고, branch selection도 그 retrieval score를 다시 먹기 때문에, 잘못된 추상화가 다음 iteration의 탐색 공간까지 바꿔버린다는 점이야. 그러면 abstraction gap이 줄어드는 게 아니라 오히려 drift가 커진다.

조금 더 직관적으로 말하면 이거야.

현재 search state는 “지금 내가 어느 동네를 뒤지고 있는가”를 뜻해.
feedback은 “그 동네에서 다음에 뭘 더 찾아야 하는가”를 정해줘.
그런데 feedback이 다른 동네 이야기면, rewrite는 내비게이션을 다른 목적지로 바꾸고, 실제 이동은 여전히 현재 동네 안에서만 하게 된다.
그러면 query와 pool 사이에 mismatch가 생긴다.
이 mismatch가 반복되면 retrieval quality가 떨어지고, branch도 잘못 고착된다.


이건 구현도 쉬워.
현재 구조에서 selector override 전에 global probe를 한 번 더 돌리고, top scoring unexplored branch를 beam 마지막 몇 칸에 넣으면 된다.

2. uncertainty triggered rollback을 넣기

항상 backtracking할 필요는 없어. 대신 “지금 branch가 틀렸을 가능성”이 높을 때만 rollback하면 된다.

트리거는 이런 걸 쓰면 돼.

현재 iteration에서 rewrite 전후 nDCG proxy가 안 오름
top local hits의 score margin이 작음
branch 후보들의 score가 서로 비슷함
off-branch 비율이 갑자기 증가함
iteration이 진행되는데 retrieval diversity가 줄고 coverage가 정체됨

이런 신호가 나오면 한 depth 위로 올라가서 sibling branch를 다시 열어보는 거야.
즉, local search를 유지하되 commitment를 irreversible하게 만들지 않는 거지.

## method 1

지금 beam 전부를 local child expansion에 쓰지 말고, beam 하나 정도는 항상 전역 탈출용으로 남겨.
예를 들면 beam size가 10이면 8개는 local child에서 뽑고, 2개는 root 혹은 상위 depth의 global retrieval로 다시 뽑는 거야.

이렇게 하면 main search는 local하게 가되, 한두 개 후보는 늘 “혹시 완전히 다른 가지가 맞나”를 확인해준다. diverse beam search가 greedy beam의 mode collapse를 줄이려는 논리와 비슷해.

1. max. beam size=K이라면, K-2은 일반적인 beam search에 따라서 다음 frontier를 고르고(by mean score)(Origin_mean), 나머
  지 2개는 beam search로 골라지지 않은 frontier들 중 가장 높은 score를 가진 Frontier (by max score) (Rest_max) 를
  가져오자. 이 때 Rest_max 점수가 origin_max 점수보다 낮으면 가져올 필요 없음

2. Random. beam size=K이라면, K-2은 일반적인 beam search에 따라서 다음 frontier를 고르고(by mean score)(Origin_mean), 나머
  지 2개는 beam search로 골라지지 않은 frontier들 중 random selection.

여기서 한 단계만 바꾸면 훨씬 나아진다.

현재 beam만 유지하지 말고, 이전 iteration에서 점수가 괜찮았던 sibling branch 몇 개를 open list처럼 따로 저장해.
다음 iteration에서는 current children만 보지 말고, saved siblings와 current children을 같이 경쟁시키는 거야.

이건 best-first search의 핵심 직관과 가깝다. 한 경로에만 올인하지 않고, 여러 frontier 후보를 남겨두는 방식이다.

논문적으로도 좋다.
“tree constrained local search”가 아니라 “tree constrained search with recoverable frontier”라고 쓸 수 있어서, locality의 장점은 살리고 irreversibility criticism을 피할 수 있어.

### 2026-03-11 구현 메모

- `src/run_round6.py`에 method 1의 deterministic 버전을 구현함.
- 구현 원칙:
    - rewrite context는 그대로 local / cumulative reached leaf pool만 사용
    - global signal은 rewrite 단계가 아니라 **branch selection 직전**에만 추가
    - 즉 query drift는 막고, branch commitment만 recoverable하게 만듦
- 현재 동작:
    - local selector는 기존 `--round5_selector_mode`를 그대로 사용 (`maxscore_global`, `meanscore_global`, `max_hit_global`)
    - 그 뒤 full leaf pool에 대해 rewritten query로 global probe를 한 번 더 수행
    - 현재 sample에서 expand 가능한 전체 branch endpoint 중, local top-K에 안 들어간 후보들을 global escape 후보로 둠
    - beam size가 `K`이고 `--round6_global_escape_slots=S`이면:
        - local selector top-K를 먼저 만들고
        - 그 중 뒤 `S`칸을 replaceable tail로 보고
        - global 후보의 `Rest_max`가 tail의 `origin_max`보다 큰 경우에만 교체
    - 비교 축은 selector mode와 무관하게 `max_score`로 통일함. 그래야 local tail vs global escape를 같은 scale에서 비교할 수 있음.
- 제약:
    - `retriever_slate`에는 적용하지 않음. 현재 구현에서는 점수 축이 직접 비교 가능하지 않아서 fail-fast로 막음.
    - 최소 1개 local branch는 항상 유지함.
- 추가 인자:
    - `--round6_global_escape`
    - `--round6_global_escape_slots` (default 2)
- 로그:
    - `iter_records`에 `global_escape_pick_reason`, `global_escape_scored_top`, `global_escape_selected_paths`, `global_escape_replaced_local_paths` 등을 저장
    - 그래서 “실제로 replacement가 얼마나 발생했는가”를 후처리로 바로 볼 수 있음
- 분석 스크립트:
    - `scripts/analyze_round6_global_escape.py`
    - run별 / iter별 replacement rate, 평균 교체 slot 수, `pick_reason` 분포를 집계하도록 구현함

## method 2

go-explore의 아이디어에 따라 구현하기. 

Go-Explore의 핵심은 유망한 상태를 기억해 두고, 그 상태로 먼저 정확히 돌아간 다음, 그 지점에서 의도적으로 새로운 방향을 탐색하는 것이야. 즉, promising state를 archive에 남겨 두고, 다른 방향을 탐색한 뒤에도 다시 쓸 수 있어야 한다는 점이 중요하다.

explore

leaf에 도달했을 때 다음 iteration에서 발동
query를 original query로 reset한다.
이전 rewrite는 보여주고, 이건 이미 본 방향이라 새로운 evidence가 없을지 Explore하라고 프롬프팅해.
직전에 exploit했던 subtree는 한 번만 retrieval pool과 rewrite evidence에서 뺀다.
rewrite할 때 original query, 이전 rewrite+evidence는 이미 본 방향이라 주고 rewrite. 현재 Branch frontier들 중 다시 고르기
다음 iteration부터는 그 subtree도 다시 retrieval pool에 들어올 수 있게 한다.

만약 다음에도 leaf에 도달했다면? 이전에 도달한 leaf + 또 도달한 leaf concat하고 똑같이 처리

### 2026-03-11 구현 메모

- `src/run_round6.py`에 method2를 `Go-Explore inspired archive-conditioned explore step` 형태로 구현함.
- trigger:
    - 어떤 iteration에서 new leaf가 추가되면, 다음 iteration이 `explore_step`이 됨
- explore step 동작:
    - 검색 쿼리 seed는 `original query`로 reset
    - previous rewrite는 이미 본 방향으로 간주
    - `agent_executor_v1_icl2_explore` 프롬프트를 사용함
    - 이 프롬프트는 `agent_executor_v1_icl2`를 거의 그대로 유지하고, `Previous Rewritten Query + Seen-Direction Evidence`를 이미 본 방향으로 보고 가능하면 다른 plausible evidence direction을 찾으라는 가이드만 추가함
- archive / mask:
    - new leaf마다 `selected_branches_before` 중 가장 깊은 ancestor branch를 archive prefix로 저장
    - archive는 누적 저장하지만, masking은 explore step에서만 적용
    - masking scope는 `rewrite evidence + selector scoring`이고, final eval retrieval에는 적용하지 않음
- fusion memory:
    - 각 iteration의 retrieved top leaf ranking을 bank에 저장
    - leaf-trigger run도 bank에 포함
    - query는 누적하지 않고, retrieval 결과만 누적
- fusion mode:
    - `rrf`
    - `max_score`
    - `sum_score`
    - active mode 하나가 실제 controller / final ranking에 사용되고, 나머지는 diagnostic으로만 nDCG를 계산함
- 로그:
    - `iter_records`에 `explore_step`, `explore_archive_prefixes`, `new_leaf_paths`, `fusion_bank_runs`, `fusion_metrics_by_mode` 등을 저장
    - 각 banked run의 nDCG@10과, 각 fusion mode의 fused nDCG@10을 따로 남김

### 2026-03-11 구현 메모 추가

- method2의 fusion memory를 두 개로 분리함.
    - `all_iters bank`: 모든 completed iteration의 retrieval run을 저장
    - `leaf_trigger_only bank`: `new_leaf_paths`가 생긴 run만 저장
- controller와 metric의 역할을 분리함.
    - controller는 `all_iters bank`만 사용
    - 같은 iteration에서 방금 생성된 retrieval run은 controller가 바로 쓰지 않음
    - 즉 run은 iteration 끝에서 bank에 append되고, 다음 iteration부터 controller memory로 사용됨
- 같은 실행에서 metric을 두 개 같이 계산함.
    - `nDCG@10`: 공식 metric. `all_iters bank`를 fuse한 결과
    - `nDCG_leaf_trigger_only`: diagnostic metric. 같은 trajectory 위에서 `leaf_trigger_only bank`만 fuse한 결과
- 따라서 `nDCG_leaf_trigger_only`는 fair full-run ablation은 아님.
    - branch selection trajectory는 여전히 `all_iters bank` controller가 만들기 때문
    - 이 값은 “같은 trajectory에서 bank scope만 바꿔 re-fuse하면 어떻게 보이는가”를 보는 counterfactual diagnostic임

### 용어 정의

- **run**
    - 한 sample에 대해, 한 iteration에서 rewrite 후 수행한 retrieval 결과
- **bank**
    - 이후 iteration에서 재사용하기 위해 저장해둔 과거 run들의 집합
- **all_iters bank**
    - 모든 completed run을 저장하는 공식 memory bank
- **leaf_trigger_only bank**
    - 새로운 leaf를 처음 도달하게 만든 run만 저장하는 diagnostic memory bank
- **memory**
    - bank에 저장된 과거 retrieval history 전체를 뜻함
    - 현재 구현에서는 query text를 누적하지 않고 retrieval result만 누적함
- **fusion**
    - 여러 banked run의 ranked leaf retrieval 결과를 하나의 ranked list로 합치는 연산
    - 현재 구현된 방식은 `rrf`, `max_score`, `sum_score`
- **controller**
    - 다음 iteration에서 어떤 branch frontier를 expand할지 정하는 선택 로직
    - method2에서는 `all_iters bank`를 fuse한 retrieval result를 입력으로 사용함
- **official metric**
    - 현재 실행의 branch decision과 연결된 metric
    - 현재는 `all_iters bank` 기반 `nDCG@10`
- **diagnostic metric**
    - 같은 실행 trajectory는 유지한 채, bank scope나 fusion rule만 바꿔 계산한 보조 metric
    - 현재는 `nDCG_leaf_trigger_only`
- **leaf-trigger run**
    - 어떤 iteration에서 이전에 도달하지 못했던 leaf가 하나 이상 새로 추가된 run
    - 이 run은 다음 iteration의 `explore_step`을 유발함

  - history:
      - 이전 beam에 한 번이라도 들어왔던 branch endpoints
  - recoverable frontier:
      - history branch들로부터 한 step 더 내려간 candidate sub-branches
      - ancestor-dominated candidate는 dedup
  - explore evidence pool:
      - recoverable frontier candidate들의 descendant leaf union
  - explore branch selection:
      - 이 recoverable frontier candidate들 중 beam size만큼 선택

### iteration t 기준 타임라인

- **Step 0. 현재 local memory 준비**
    - sample마다 지금까지 실제로 도달한 leaf들을 `cumulative reached leaf pool`로 유지함
    - iteration `t`의 rewrite와 retrieval은 기본적으로 이 pool 위에서 수행됨

- **Step 1. rewrite 직전 evidence retrieval**
    - `query_pre`로 현재 pool에서 leaf retrieval을 한 번 수행함
    - 여기서 얻은 top-k leaf 요약이 `leaf_descs`
    - 현재 beam에 잡힌 branch summary가 `branch_descs`
    - explore step이면 archive된 subtree 쪽 evidence를 따로 모아 `seen_direction_evidence`로 넣음

- **Step 2. rewrite**
    - normal step이면 `query_pre = 이전 iteration의 query_post`
    - explore step이면 `query_pre = original query`
    - prompt는 evidence를 보고 `rewrite`를 생성함
    - 최종 검색 query는 `original query + rewrite`

- **Step 3. rewrite 후 retrieval**
    - `query_post = original query + rewrite`
    - 이 query로 `cumulative reached leaf pool`에서 retrieval 수행
    - 이 결과가 현재 iteration의 run이 됨
    - method2에서는 이 run을 바로 controller에 쓰지 않고, iteration 끝에서 bank에 append할 후보로만 저장함

- **Step 4. candidate branch 확장 대상 생성**
    - 현재 선택된 branch들의 direct child만 다음 확장 후보가 됨
    - 아직 선택된 branch가 없으면 root child들이 후보가 됨
    - 즉 candidate set은 항상 “현재 frontier의 한 단계 아래”로 제한됨

- **Step 5. candidate branch 점수화 / 선택**
    - method2 off:
        - 현재 iteration local retrieval 결과로 candidate branch를 점수화
    - method2 on:
        - `all_iters bank`를 fusion한 retrieval 결과로 candidate branch를 점수화
        - 단 bank가 비어 있으면 `current_local_retrieval_fallback`으로 내려감
    - 점수화 방식은 selector mode에 따라 `max`, `mean`, `hit-count` 중 하나를 사용
    - top-B branch를 골라 실제 frontier를 갱신함

- **Step 6. 새 leaf 도달 여부 확인**
    - branch를 확장한 뒤, 이전까지 없던 leaf가 새로 추가되었는지 검사함
    - 새 leaf가 있으면 그 leaf의 deepest selected ancestor를 archive prefix로 저장
    - 그리고 다음 iteration을 `explore_step=True`로 예약함

- **Step 7. iteration 끝에서 memory bank append**
    - `all_iters bank`에는 이번 run을 항상 append
    - `leaf_trigger_only bank`에는 이번 iteration에서 `new_leaf_paths`가 있을 때만 append
    - 중요한 점: controller는 같은 iteration에서 방금 만든 run을 바로 쓰지 않음
    - 즉 bank는 항상 “completed runs only” semantics를 가짐

- **Step 8. metric 계산**
    - 공식 metric `nDCG@10`은 `all_iters bank`를 fuse한 결과
    - diagnostic metric `nDCG_leaf_trigger_only`는 같은 trajectory 위에서 `leaf_trigger_only bank`만 fuse한 결과

### method2 off vs on 비교

- **method2 off**
    - rewrite evidence는 현재 `cumulative reached leaf pool`의 top leaf evidence를 사용
    - candidate branch 선택도 현재 iteration local retrieval 결과만 사용
    - memory bank, fusion, archive-conditioned explore가 없음
    - 즉 현재 local state만 보고 다음 branch를 고르는 순수 local search에 가까움

- **method2 on**
    - rewrite 단계는 여전히 local evidence 기반이지만, leaf-trigger 이후에는 query를 `original query`로 reset한 explore step이 들어감
    - archive된 subtree evidence는 `seen_direction_evidence`로 따로 제공되고, explore step에서 rewrite evidence / selector scoring에서만 일시적으로 mask됨
    - candidate branch 선택은 현재 iteration retrieval 하나만 보지 않고, 과거 completed runs를 bank에 저장한 뒤 fuse한 결과를 사용
    - 따라서 local search 위에 “기억된 retrieval history”를 얹어서 branch 선택을 더 안정화하는 구조임

- **핵심 차이 한 줄**
    - method2 off = 현재 local retrieval만 보는 local controller
    - method2 on = local rewrite + banked retrieval history를 보는 fusion controller

### 코드 최적화 포인트

- 현재 iteration의 `rewrite 후 retrieval`과 다음 iteration의 `rewrite 전 retrieval`은 둘 다 leaf retrieval이라는 점에서는 비슷함
- 하지만 완전히 같은 연산은 아님
    - 다음 iteration rewrite 전 retrieval은 `query_pre`를 사용
    - normal step에서는 이 `query_pre`가 이전 iteration의 `query_post`와 같아서 재활용 여지가 있음
    - explore step에서는 `query_pre = original query`로 reset되므로 재활용이 어려움
    - 또한 explore step에서는 archive mask 때문에 retrieval pool 자체도 달라짐

- 따라서 최적화 가능성은 이렇게 정리됨
    - normal -> normal 전이:
        - 이전 iteration의 retrieval score를 재사용하고, 새로 추가된 leaf만 incremental scoring하는 최적화가 가능함
    - explore 관련 전이:
        - query와 pool이 둘 다 바뀔 수 있으므로 재계산이 필요함

- 즉 실용적으로는
    - local 누적 search 구간에서는 incremental retrieval cache 최적화 가능
    - explore 구간에서는 correctness를 위해 새로 retrieval하는 쪽이 안전함



<!-- 3. rewrite는 local only로 두되, branch selection에는 tiny global signal을 섞기

나는 이게 특히 좋아 보여.

rewrite context는 계속 local pool에서만 가져와. 그래야 abstraction이 corpus grounded하게 유지된다.
대신 branch scoring에는 작은 전역 신호를 섞어.

예를 들면

branch score
= local matched leaf score
λ × global retriever prior
μ × exploration bonus for under-visited branches

이렇게 하면 rewrite drift는 막으면서도, branch selection이 완전히 local trap에 갇히지는 않는다. UCT 계열이 exploitation과 exploration을 같이 보려는 이유도 이 균형 때문이야.

네 세팅에서 가장 추천하는 버전

지금 바로 실험 가능한 건 이 조합이야.

rewrite는 지금처럼 cumulative reachable leaf pool만 사용
branch selection은 80퍼센트 local, 20퍼센트 global escape
uncertainty가 높을 때만 rollback
그리고 open list에 previous sibling 3개 정도 유지

이 조합의 장점은 네 메인 메시지를 안 깨는 거야.
tree를 corpus constraint로 쓴다는 핵심은 유지된다.
동시에 리뷰어가 “한 번 branch 틀리면 끝나는 거 아니냐”라고 물었을 때, 아니라고 답할 수 있다.
우리는 local grounding을 유지하되 recoverable search를 설계했다고 말할 수 있으니까. -->

## 2026-03-11 구현 메모 추가

### 이번에 반영한 method2 의미

- `method2=1`은 이제 단순 bank-conditioned controller가 아니라, **Go-Explore style reopen step**을 포함한다.
- baseline(`method2=0`)은 그대로 둔다.
- 핵심은 `leaf에 도달 -> 다음 step에서 과거 branching decision으로 돌아가 missed sibling을 다시 연다`는 점이다.

### 용어 정의

- **decision state**
    - 실제로 beam이 어떤 child branch를 고른 branching node.
    - 구현에서는 `selected_before`에 있었던 parent branch state가 이에 해당한다.

- **archive**
    - 과거 decision state들을 누적 저장한 메모리.
    - 각 record는 `parent branching state`와, 그 state에서 **이미 선택했던 child branch들**을 기억한다.
    - 즉 archive는 “어디를 갔었는가”가 아니라 “그 decision point에서 어떤 action들을 이미 썼는가”를 저장한다.

- **recoverable sibling candidate**
    - archive에 있는 어떤 parent branching state의 direct child 중,
    - 아직 그 parent에서 선택된 적이 없는 unexplored sibling branch.
    - 첫 explore step의 action space는 이 candidate들의 union이다.

- **memory bank**
    - query rewrite 후 retrieval 결과를 iteration 단위 run으로 저장한 bank.
    - `all_iters`는 공식 controller용 memory, `leaf_trigger_only`는 diagnostic memory다.

- **fusion**
    - 여러 retrieval run의 ranked leaf 결과를 하나로 합치는 연산.
    - 현재 `rrf`, `max_score`, `sum_score`를 지원한다.

- **controller**
    - 다음에 어떤 branch를 beam에 둘지 결정하는 selection logic.
    - normal step에서는 기존 round6 score-based selector를 유지하고,
    - explore step에서는 sibling reopen candidate들만 대상으로 같은 selector를 적용한다.

### 첫 explore step의 실제 동작

- trigger
    - 직전 iteration에서 `new_leaf_paths`가 하나라도 생기면 다음 iteration을 explore로 예약한다.

- query
    - `query_pre = original query`

- rewrite input
    - `original query`
    - `unexplored sibling evidence`
    - `previous rewrite`는 already-covered direction memo 용도
    - `seen_direction_evidence`는 직전 leaf trigger를 만든 selected prefix 쪽 evidence를 짧게 제공

- evidence pool
    - 전체 leaf를 쓰지 않는다.
    - archive 전체에서 얻은 `recoverable sibling candidate`들의 descendant leaf union만 사용한다.
    - 따라서 evidence space와 action space를 맞춘다.

- exploit subtree 처리
    - 직전 leaf trigger를 만든 selected prefix descendant는 첫 explore step의 positive evidence에서 제외한다.
    - 영구 exclusion은 하지 않는다.
    - 다음 normal step부터는 retrieval pool에 다시 들어오고, 실제로 재검색되면 다시 evidence로 승격될 수 있다.

- branch selection
    - `all_expandable_paths` raw 전체를 그대로 쓰지 않는다.
    - archive에서 만든 sibling candidate union만 score한다.
    - 점수 규칙은 기존 `round5_selector_mode`를 그대로 재사용한다.
    - beam은 merge가 아니라 **replace**다. 즉 첫 explore step은 archived state로 실제로 돌아간다.

### normal step과의 차이

- normal step
    - rewrite evidence: `cumulative reached leaf pool`
    - branch candidate: 현재 live beam의 direct child
    - beam transition: baseline처럼 child selection + fallback fill

- first explore step
    - rewrite evidence: recoverable sibling descendant pool
    - branch candidate: archive에서 모은 unexplored sibling union
    - beam transition: selected sibling states로 replace

### 구현상 주의점

- explore step에서는 current live beam을 한 번 더 expand하지 않는다.
    - 즉 `sample.update(...)`를 건너뛰고,
    - 현재 prediction tree 위에서 already-expandable한 sibling state들을 reopen 대상으로 삼는다.
    - 이게 “현재 frontier를 더 exploit한 뒤 explore”가 아니라, “archive state로 돌아가서 explore”라는 의미를 보존한다.

- archive는 cumulative지만, official retrieval metric은 여전히 기존 방식대로 계산한다.
    - `nDCG@10`: `all_iters` bank 기반 official metric
    - `nDCG_leaf_trigger_only`: 같은 trajectory에서의 diagnostic metric

## Analysis (RRF 중심)

`method2`의 fusion mode 중에서는 현재 `RRF`가 가장 잘 나온다. 그래서 분석도 `round6_mrr_selector_accum_meanscore_global_method2_rrf` 결과를 기준으로 진행한다.

이번 버전에서는 `stackoverflow`는 제외했다. coding subset 특성이 다른데, 현재 관심은 science/general subset에서 “RRF 자체가 문제인지 vs explore step이 문제인지”를 먼저 분리해 보는 것이다.

### 분석 방법

- 사용 데이터:
    - 현재 존재하는 6개 subset의 `method2_rrf` run
    - `biology`, `earth_science`, `economics`, `psychology`, `robotics`, `sustainable_living`
- 비교한 두 metric:
    - **official nDCG@10**
        - `iter_records[i]["metrics"]["nDCG@10"]`
        - 즉 실제 실행에서 controller가 사용한 `all_iters bank + RRF` 결과
    - **current-run nDCG@10**
        - 같은 iteration의 `local_doc_ids`를 gold doc id와 직접 비교해 다시 계산한 값
        - 즉 bank fusion 없이, 방금 생성된 retrieval 결과만 본 값
- 이 둘을 나누면 다음을 분리해서 볼 수 있다.
    - `official - current-run > 0` 이면 RRF fusion이 보정 역할을 함
    - explore step에서 `current-run`이 급락하면, 문제의 1차 원인은 explore step 자체에 있음

재현 방법:

```bash
python scripts/analyze_round6_rrf_explore_vs_fusion.py \
    --base_dir results/BRIGHT \
    --exclude_subsets stackoverflow \
    --output_prefix results/BRIGHT/analysis/round6_rrf_explore_vs_fusion_no_stackoverflow
```

출력 파일:

- `results/BRIGHT/analysis/round6_rrf_explore_vs_fusion_no_stackoverflow_rows.csv`
- `results/BRIGHT/analysis/round6_rrf_explore_vs_fusion_no_stackoverflow_subset_summary.csv`
- `results/BRIGHT/analysis/round6_rrf_explore_vs_fusion_no_stackoverflow_overall_summary.csv`

### 핵심 결과

- **결론 1. 성능 하락의 1차 원인은 RRF 자체보다 explore step의 current retrieval 저하다.**
    - 전체 평균에서:
        - non-explore row의 `current_step_delta_mean = +0.50`
        - explore row의 `current_step_delta_mean = -1.98`
    - 즉 explore step이 발동한 iteration에서는, 방금 생성한 retrieval 결과만 놓고 보면 평균적으로 성능이 내려간다.

- **결론 2. RRF는 오히려 그 explore-time drop을 완화하는 쪽이다.**
    - 전체 평균에서:
        - non-explore row의 `fusion_delta_mean = +1.00`
        - explore row의 `fusion_delta_mean = +4.35`
    - 즉 explore step일수록 `official nDCG@10`이 `current-run nDCG@10`보다 더 높다.
    - 해석하면, first explore step이 immediate retrieval quality를 떨어뜨리지만, RRF가 이전 good run들을 끌어와 그 손실을 메운다.

- **결론 3. explore step은 local retrieval에는 꽤 공격적이고, RRF가 그 충격을 가린다.**
    - 전체 평균에서:
        - explore row의 `official_step_delta_mean = +0.34`
        - explore row의 `current_step_delta_mean = -1.98`
    - 즉 official metric만 보면 explore step이 꼭 나빠 보이지 않는다.
    - 하지만 current-run metric은 실제로 급락한다.
    - 따라서 “explore가 문제 없는 것처럼 보이는” 이유는 RRF bank memory가 이전 run을 재활용하기 때문이지, explore step이 자체적으로 retrieval을 잘하는 것은 아니다.

### subset별 관찰

- `psychology`, `sustainable_living`
    - explore row에서 `official_step_delta_mean`까지 음수
    - 즉 이 subset들은 RRF가 보정해도 explore step의 충격을 다 못 막는다.

- `biology`, `earth_science`, `robotics`
    - explore row에서 `official_step_delta_mean`은 양수
    - 하지만 `current_step_delta_mean`은 여전히 음수이거나 매우 약함
    - 즉 explore 직후 retrieval은 흔들리지만, banked RRF가 전체 metric을 방어한다.

- `robotics`
    - 예외적으로 explore row의 `current_step_delta_mean`도 약한 양수
    - 이 subset에서는 sibling reopen이 실제로 도움이 되는 케이스가 더 많을 가능성이 있다.

- `economics`
    - `official_step_delta_mean`은 거의 0에 가깝지만, `current_step_delta_mean`은 명확히 음수
    - 즉 explore가 current retrieval을 흔들고, RRF가 그 손실을 겨우 상쇄하는 케이스에 가깝다.

### 해석

- 지금 observed degradation을 두 부분으로 나누면:
    - **RRF fusion score aggregation 문제**
        - 현재 증거상 주원인 아님
        - explore row에서조차 `official - current-run`이 평균적으로 크게 양수
        - 즉 RRF는 보통 해를 주기보다 보정 역할을 함
    - **explore step 자체의 distribution shift**
        - 더 핵심 원인
        - `original query + sibling evidence`로 만든 first explore rewrite가, immediate retrieval quality를 자주 떨어뜨림
        - 그리고 beam을 replace하는 구조라 이후 controller trajectory도 바뀜

- 따라서 현재 method2의 약점은 “RRF를 써서 점수를 합친 것”보다,
    - **first explore step의 rewrite / branch reopen이 너무 공격적이라 현재 retrieval이 흔들리는 것**
    - 그리고 그 상태에서 **새 beam으로 교체되며 이후 trajectory가 불안정해지는 것**
  으로 보는 편이 맞다.

### 지금 단계의 working hypothesis

- RRF는 유지하는 편이 맞다.
    - 현재 데이터에서는 explore-time damage를 줄여주는 쪽이다.
- 다음 개선 포인트는 fusion rule이 아니라 explore transition이다.
    - 첫 explore step의 rewrite를 더 보수적으로 만들기
    - sibling reopen candidate를 더 좁게/더 잘 점수화하기
    - beam replace를 완화할지, 혹은 replace 이후 one-step stabilizer를 둘지 검토하기

## `global_method2_rrf` vs `global_gescape2`

이번에는 `results/BRIGHT/ndcg_end_summaryround6.csv`에서 아래 두 실험을 직접 비교한다.

- `round6_mrr_selector_accum_meanscore_global_gescape2`
- `round6_mrr_selector_accum_meanscore_global_method2_rrf`

둘 다 공통적으로:

- base selector는 `meanscore_global`
- rewrite prompt는 `agent_executor_v1_icl2`
- local cumulative leaf memory를 사용

차이는 **global signal을 언제, 얼마나 강하게 넣는지**에 있다.

### 코드 동작 차이

#### 1. `global_gescape2`

이 경로는 기본적으로 **local trajectory를 유지**한다.

- normal rewrite / normal local retrieval / normal `sample.update(...)`를 그대로 수행
- branch selector 단계에서만 추가로 full-leaf global probe를 한 번 더 수행
- local top beam을 먼저 만들고, 그 뒤 `max_score` 기준으로 local tail보다 더 좋은 global candidate가 있을 때만 최대 2개 slot을 교체
- 즉:
    - local branch는 최소 일부 유지
    - global branch는 selector tail만 부분 교체
    - trajectory는 local exploit 중심으로 계속 이어짐

코드상 핵심은:

- `round6_global_escape` 활성화
- `_merge_local_with_global_escape(...)`
- `sample.update(...)`는 그대로 실행됨

정리하면, `gescape2`는 **"local exploit + conservative global escape"**다.

#### 2. `global_method2_rrf`

이 경로는 **archive-conditioned explore + RRF memory controller**다.

- leaf-trigger가 발생하면 다음 iteration에서 `explore_step`으로 전환
- 첫 explore step에서는:
    - `query_pre = original query`
    - evidence는 archive에서 모은 `unexplored sibling` 하위 leaf들만 사용
    - 직전 exploit subtree는 first explore step에서 evidence/retrieval pool에서 제외
- controller는 current local retrieval이 아니라 `all_iters` fusion bank를 `RRF`로 fuse한 ranking을 우선 사용
- 그리고 explore step에서는 current live beam을 더 expand하지 않고, recovered sibling candidate들 중 top-B를 골라 **beam 자체를 replace**한다

즉:

- search state를 archive decision point로 되돌림
- 놓친 sibling direction을 다시 엶
- selection도 current-only가 아니라 banked RRF memory가 개입

정리하면, `method2_rrf`는 **"return-to-decision-point explore + beam replacement + banked controller"**다.

### 결과 비교

`ndcg_end_summaryround6.csv` 기준 overall 결과:

- `global_gescape2`
    - `avg_ndcg_end = 35.551`
    - `avg_ndcg_max = 38.000`
- `global_method2_rrf`
    - `avg_ndcg_end = 30.746`
    - `avg_ndcg_max = 37.714`

즉 overall delta는:

- `end`: `method2_rrf - gescape2 = -4.81`
- `max`: `method2_rrf - gescape2 = -0.29`

이 차이는 중요하다.

- `max`는 거의 비슷하다.
- 그런데 `end`는 `method2_rrf`가 크게 낮다.

이건 `method2_rrf`가 **중간에 좋은 상태에는 가끔 도달하지만, 그 상태를 안정적으로 유지하지 못한다**는 뜻이다. 반대로 `gescape2`는 local trajectory를 더 많이 보존하므로, final iteration까지 무너지지 않는다.

### subset별 차이

`method2_rrf - gescape2` 기준:

- `biology`: `end -1.02`, `max +2.98`
- `earth_science`: `end +2.70`, `max +1.88`
- `economics`: `end -3.82`, `max +0.89`
- `pony`: `end -3.86`, `max -5.32`
- `psychology`: `end -8.67`, `max -0.55`
- `robotics`: `end +8.04`, `max -0.87`
- `stackoverflow`: `end -6.84`, `max -0.28`
- `sustainable_living`: `end -6.40`, `max -1.33`
- `theoremqa_questions`: `end -10.18`, `max -0.20`
- `theoremqa_theorems`: `end -18.00`, `max -0.06`

요약하면:

- `method2_rrf`의 `end` 승리는 10개 subset 중 2개뿐이다.
    - `earth_science`
    - `robotics`
- 반면 `max`는 일부 subset에서 경쟁력이 있다.
    - `biology`, `earth_science`, `economics`

### 해석

이 비교에서 더 중요한 건 `gescape2`가 단순히 weaker method가 아니라는 점이다.

- `gescape2`
    - global signal을 넣되, local beam을 완전히 버리지 않음
    - 그래서 잘못된 explore transition이 생겨도 전체 trajectory가 덜 망가짐
- `method2_rrf`
    - first explore step에서 archive sibling으로 beam을 replace
    - current retrieval 분포가 바뀌고, 이후 iterations가 그 새 beam 위에서 이어짐
    - 따라서 intermediate max는 괜찮아도 end가 쉽게 무너짐

즉 현재 결과는:

- 문제의 핵심이 **"global information을 넣었기 때문"**이 아니라
- **"explore를 state transition으로 너무 강하게 적용해서 trajectory를 끊었기 때문"**
임을 보여준다.

이건 앞의 RRF 분석과도 일관적이다.

- `RRF` 자체는 보정 역할을 함
- 실제 drop은 explore transition에서 발생

그래서 현재 단계의 working conclusion은:

- `gescape2`는 stronger baseline이다.
    - local exploit 구조를 많이 유지하면서도 global escape를 허용하기 때문이다.
- `method2_rrf`는 아이디어 자체는 흥미롭지만, 지금 구현은 너무 공격적이다.
    - first explore rewrite
    - sibling reopen candidate construction
    - beam replace
  이 세 부분 중 적어도 하나는 더 보수적으로 바꿔야 한다.

## 2026-03-15 구현 메모: `method2_mode=expandable_pool`

위 해석에 따라, 다음 실험은 `archive_replace`를 유지한 채 **더 보수적인 explore transition**을 별도 mode로 추가하는 방향으로 구현했다.

구현 목적은 명확하다.

- 기존 `method2_rrf`의 약점은 `RRF` 자체보다도
    - archive sibling reopen
    - beam replace
    - original-query reset
  이 한 묶음의 explore transition이 너무 공격적이라는 데 있었다.
- 따라서 이번 실험에서는 prompt/query semantics는 최대한 normal step과 같게 두고,
  **leaf-trigger 다음 iteration에서 retrieval pool / selector candidate scope만 넓히면 어떤 일이 생기는지**를 보려는 것이다.

### 새 mode의 이름

- `--round6_method2_mode expandable_pool`

기존 `--round6_method2`는 유지하고, mode만 분기한다.

- `archive_replace`
    - 기존 method2 구현
- `expandable_pool`
    - 이번에 추가한 보수적 explore ablation

### 핵심 동작

`expandable_pool` mode에서 leaf-trigger가 발생하면, 다음 iteration 한 번만 explore step으로 동작한다.  
하지만 이 explore step은 기존 `archive_replace`와 달리 아래를 **하지 않는다**.

- `query_pre = original query` reset
- explore 전용 prompt(`agent_executor_v1_icl2_explore`) 사용
- archive sibling candidate만 reopening
- beam replace
- fusion bank를 controller로 사용

대신 아래만 바뀐다.

#### 1. rewrite input은 normal step과 동일

- `query_pre = sample.query`
- prompt도 normal rewrite prompt를 그대로 사용
    - 즉 보통 `agent_executor_v1_icl2`
- input 구성도 normal과 동일
    - `original_query`
    - `previous_rewrite`
    - `gate_descs (= leaf_descs)`

즉 explore라고 해서 prompt semantics를 따로 바꾸지 않는다.

#### 2. gate-desc retrieval pool만 넓힘

normal step에서는 `gate_descs`를 만들 때:

- 현재 beam인 `selected_before` 아래 leaf들

을 retrieval pool로 쓴다.

이번 `expandable_pool` mode의 explore step에서는 대신:

- 현재 prediction tree의 `all expandable paths`
- 그 path들의 descendant leaf union

을 pre-rewrite retrieval pool로 쓴다.

즉 leaf-trigger 다음 iteration에서는

- "현재 beam 아래 local memory"
가 아니라
- "현재 tree에서 아직 확장 가능한 branch state 전체"

를 보고 rewrite를 만든다.

#### 3. branch selector candidate도 all expandable paths로 확장

normal step selector는:

- `selected_before`의 direct child branch만 candidate로 본다.

이번 explore step에서는 대신:

- `sample.get_all_expandable_paths(sample.prediction_tree)`

로 얻은 live expandable state 전체를 candidate로 놓고,
- 기존 `meanscore_global` / `maxscore_global` / `max_hit_global`
  scoring rule로 top-B를 다시 고른다.

즉 이번 mode는 “missed sibling만 reopen”이 아니라,  
**현재 tree 전체의 unresolved branch state를 한 번 다시 global re-ranking**하는 구조다.

#### 4. `sample.update(...)`는 그대로 실행

기존 `archive_replace`는 explore step에서 current live beam을 더 expand하지 않기 위해 `sample.update(...)`를 건너뛰었다.

이번 `expandable_pool`은 그 반대로 간다.

- normal step처럼 먼저 `sample.update(...)`를 실행
- 그 뒤 selector override만
  - direct-child scope가 아니라
  - all-expandable-path scope로 바꿔서
  다시 beam을 고른다

즉 traversal continuity는 baseline에 가깝게 유지한다.

#### 5. controller는 current retrieval only

이번 mode에서는 `RRF bank`를 beam controller로 쓰지 않는다.

- 실제 beam selection:
    - current iteration retrieval only
- fusion bank:
    - diagnostic only

이렇게 한 이유는, 이번 실험의 목적이 “explore scope widening”을 보는 것이기 때문이다.  
controller까지 fusion memory를 계속 쓰면, 성능 변화 원인을 pool widening과 bank controller 중 어디로 돌려야 하는지 다시 모호해진다.

### 코드상 반영된 차이

`src/run_round6.py`에서:

- `round6_method2_mode`를 읽어 `archive_replace` / `expandable_pool`로 분기
- rewrite prep 단계에서:
    - `archive_replace`는 기존 sibling descendant pool 유지
    - `expandable_pool`은 `all_expandable_descendants` pool 사용
- selector 단계에서:
    - `archive_replace`만 beam replace
    - `expandable_pool`은 normal처럼 `sample.update(...)` 후, all-expandable-path candidate를 current retrieval로 rescoring
- metric 단계에서:
    - `archive_replace`만 fusion output을 active metric으로 사용
    - `expandable_pool`은 current retrieval을 official metric으로 사용
    - fusion metrics는 계속 diagnostic으로만 저장

### 로그 필드

분석을 위해 아래 필드를 추가/정리했다.

- `round6_method2_mode`
- `explore_pool_source`
    - 예: `all_expandable_descendants`
- `explore_candidate_scope`
    - 예: `all_expandable_paths`
- `query_pre_reset_to_original`
    - `archive_replace`에서만 `True`
- `fusion_mode_active`
    - `archive_replace`에서만 controller-active
    - `expandable_pool`에서는 빈 값

즉 이후 분석에서는,

- `archive_replace`
    - aggressive state transition
- `expandable_pool`
    - conservative scope widening

을 직접 비교할 수 있게 됐다.
