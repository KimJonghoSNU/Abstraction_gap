# 2026-03-03 Round5 Category Mode 구현 덤프

# 전체 실험에서의 목적
어떤 내용으로 rewrite하면 좋을지 줄 수 있을까? 지금은 abstraction category: theory/entity/example 고정해서 쓰고 있음. 근데 각 branch (=하위 cluster의 요약)은 코퍼스에 어떤 내용이 들어있는지 요약해서 알려주는 역할임. 여기서 뽑은 정보로 대체할 수 있을까?
첫 iter: depth1에서의 abstraction category중 3-5개 선별해서 rewrite, ...

## 0) 목적
- `src/run_round5.py`에서 기존 실험(legacy)을 깨지 않고 유지하면서,
- category selector + category-bound rewrite 실험을 같은 엔트리에서 돌릴 수 있게 확장.

## 1) 이번 구현에서 확정한 설계
- 단일 모드 인자 사용:
    - `--round5_mode {legacy,category}`
- 기본값은 `legacy`.
- `partial_ok`, `fallback_on_parse_fail`은 CLI 인자로 받지 않고 **항상 true 고정**.
- category 모드 기본값:
    - `K=3`
    - open-set category 생성
    - static in-context examples 포함
    - full history bank 사용
    - drift trigger는 leaf-cluster 기반(soft prompt control)

## 2) 코드 변경 요약

### A. `src/hyperparams.py`
- round5 category 관련 인자 추가:
    - `--round5_mode`
    - `--round5_category_k`
    - `--round5_category_generator_prompt_name`
    - `--round5_category_rewrite_prompt_name`
    - `--round5_category_history_scope`
    - `--round5_category_drift_trigger`
- 네이밍 안정화:
    - `legacy` 기본 모드일 때 round5 category 인자들은 경로 이름에서 제거.
    - category 모드에서도 기본값 인자들은 가능한 한 생략되도록 처리.
- 항상 true 고정인 내부 플래그(`round5_category_partial_ok`, `round5_category_fallback_on_parse_fail`)는 경로 네이밍에서 제거.

### B. `src/rewrite_prompts.py`
- 신규 프롬프트 2개 추가:
    1. `round5_category_generator_v1`
        - open-set abstraction category 생성
        - label 규칙(추상/재사용 가능한 snake_case) 강제
        - leaf-cluster 시 history 유지 힌트
    2. `round5_agent_executor_category_v1`
        - 선택된 category 라벨을 `Possible_Answer_Docs` key로 사용하도록 유도
        - query intent / evidence support 중심 rewrite

### C. `src/run_round5.py`
- 모드 분기 추가:
    - `legacy`: 기존 `agent_executor_v1` rewrite 흐름 유지
    - `category`: 아래 2-stage 실행
        1) category generation
        2) category-bound rewrite
- category 모드 핵심 로직:
    - category 파서 `_parse_category_output`
    - label 정규화 `_normalize_category_label`
    - history 포맷/주입 `_format_category_history`
    - leaf-cluster trigger 감지 `_is_leaf_cluster_trigger`
    - rewrite 결과 key를 selected category에 정렬 `_align_docs_to_categories`
- 고정 플래그 반영(항상 true):
    - parse 실패 시 fallback
    - partial output 허용
- 저장 필드 확장:
    - `rewrite_history`: `selected_categories`, `category_prompt`, `category_raw_output`, `leaf_cluster_triggered`
    - `iter_records`: `round5_mode`, `selected_categories`, `rewrite_branch_descs_count`, `leaf_cluster_triggered`
- 기존 누적 leaf pool 동작은 유지:
    - rewrite context / eval retrieval는 cumulative leaf 기준.

### D. `src/bash/round5/run_round5.sh`
- 모드 전환 가능하도록 확장:
    - `ROUND5_MODE` (default: `legacy`)
    - category 모드일 때만 category 관련 인자 주입.

## 3) 모드별 동작 정리

### legacy 모드
- 기존 round5와 동일하게 `agent_executor_v1` 사용.
- 실험 재현성을 위해 기존 rewrite 컨텍스트 사용 방식 최대한 유지.

### category 모드
- Iteration마다:
    1. 현재 query + evidence 컨텍스트로 category generator 실행
    2. 선택된 category를 키 스키마로 사용해 rewrite 실행
    3. rewrite 결과로 retrieval/eval 진행
- category history는 샘플별 누적 bank로 유지되고 다음 iter 프롬프트에 반영.

## 3-1) 실제 실행 순서 (코드 기준 상세)

아래는 `run_round5.py`의 **iteration t**에서 실제 호출 순서다.

1. **누적 leaf pool 업데이트 (t 시작 시점)**
    - 직전 상태의 beam에서 도달한 leaf들을 `cumulative_leaf_indices_by_sample`에 누적.
    - 이 누적 pool은 이후 rewrite-context retrieval과 eval retrieval의 검색 범위가 된다.

2. **rewrite context용 retrieval (query_pre 사용)**
    - `query_pre = sample.query` (없으면 `original_query`).
    - `query_pre`로 누적 leaf pool에서 top-`round5_mrr_pool_k` 검색.
    - 여기서 얻은 top 문서 설명 일부(`rewrite_context_topk`)를 `leaf_descs`로 사용.
    - 즉, **category generator/rewrite가 보는 문서 스니펫은 iteration t에서 query_pre로 새로 검색한 결과**다.

3. **cluster summary 수집 (iteration t 시작 시 beam 상태)**
    - `selected_branches_before = _selected_branch_paths_from_sample(sample)`를 읽는다.
    - 이 경로들의 노드 description을 모아 `branch_descs`를 만든다.
    - 즉, **cluster summary는 iteration t-1의 traversal update 결과로 확정된 branch(보통 depth d)의 summary**다.
    - depth는 트리 구조와 beam 진척에 따라 증가하며, 코드상 고정 depth가 아니라 현재 beam state depth를 그대로 사용한다.

4. **Category Generator 호출 (category 모드일 때만)**
    - 입력:
        - `original_query`, `previous_rewrite`
        - `Retrieved Topic Cluster Summaries` = `branch_descs` (**primary**)
        - `Retrieved Document Evidence Snippets` = `leaf_descs` (**secondary**)
        - `Category History` (누적 bank)
        - `Category Stability Reminder` (leaf-cluster trigger 기반)
        - `domain_route_hint`
    - 출력: `Categories[{label, hint}]` (K개, 기본 3개)
    - parse fail 시 fallback(true), partial 허용(true).

5. **Agent Executor (Query Rewriter) 호출**
    - legacy 모드:
        - `agent_executor_v1` 1회 호출
    - category 모드:
        - `round5_agent_executor_category_v1` 호출
        - `Possible_Answer_Docs` key schema를 selected categories로 강제 주입
    - 공통:
        - rewrite 단계에는 `branch_descs`를 넣지 않음 (`branch_descs=[]`)
        - rewrite key-value를 concat해 `rewrite_blob` 생성
        - `query_post = original_query + rewrite_blob`

6. **eval retrieval (query_post 사용)**
    - `query_post`로 누적 leaf pool에서 `flat_topk` retrieval.
    - 이 결과로 `nDCG@10`, `Recall@K`, `Coverage` 계산.
    - 즉, **iteration t 성능 metric은 query_post 기반 retrieval 결과**다.

7. **traversal update (다음 depth로 확장)**
    - `sample.get_step_prompts()`로 현재 beam의 child slate 생성.
    - LLM 대신 retriever score로 slate 내 ranking 생성 후 `sample.update(...)`.
    - update 결과가 `selected_branches_after`이며, 다음 iteration(t+1)의 `selected_branches_before`가 된다.
    - update 후 새로 도달한 leaf는 누적 leaf pool에 추가.
    - selector mode가 `maxscore_global`/`meanscore_global`이면 추가 override가 한 번 더 수행된다:
        - 후보 branch: `selected_branches_before`의 direct child branch
        - selector local pool: 해당 branch 하위 leaf 합집합(비면 전체 leaf fallback)
        - `query_post` top-K retrieval 점수로 branch score를 계산해 global top-B로 재선택
        - 매핑 실패 시 retriever-slate 결과 유지

정리하면:
- **category generator 입력의 retrieval 결과는 iteration t에서 query_pre로 freshly 검색한 결과**
- **category generator의 cluster summary는 iteration t 시작 시점 beam(=직전 update 결과)에서 가져온 summary**
- **최종 metric은 iteration t rewrite(query_post)로 다시 검색한 결과**

## 4) 검증 결과
- 문법 검증:
    - `python -m py_compile src/run_round5.py src/hyperparams.py src/rewrite_prompts.py` 통과
- 실행 스모크(각 1 sample, 1 iter) 통과:
    - `--round5_mode legacy`
    - `--round5_mode category`
- category 모드 산출물에서 `iter_records.selected_categories` 및 `possible_answer_docs` 매핑 저장 확인.

## 5) 비판적 메모 / 리스크
- 현재 category generator 단계는 별도 캐시를 두지 않아서 대규모 실험 시 토큰/시간 비용 증가 가능.
- category 모드 프롬프트는 key 추상화를 강하게 요구하지만, 모델이 도메인 세부 키로 드리프트할 가능성은 남아 있음.
- fallback을 항상 true로 고정했기 때문에 실패는 줄지만, 잘못된 안전 fallback이 성능 ceiling을 만들 수 있음(향후 ablation 권장).

## 6) 다음 액션 제안
1. category generator cache 추가(비용 절감).
2. category label drift 통계(log analyzer) 추가.
3. legacy vs category를 동일 셋팅으로 5 subset 이상 비교해서 실제 gain/variance 확인.

---

## 2026-03-04 추가 구현: Category 모드에서도 selector score mode 비교 가능

### 왜 추가했는가
- category mode 실험에서도 "rewrite는 동일하게 두고, branch 확장 선택 기준만 바꿨을 때" 성능 차이를 확인하기 위해 selector mode를 통합했다.

### 반영 내용
- `src/run_round5.py`, `src/run_round5_gold.py`
    - `--round5_selector_mode`를 category 모드에서도 동일하게 사용.
    - 동작:
        - `retriever_slate`: 기존 그대로 (`sample.update` 결과 사용)
        - `maxscore_global`: 후보 branch별 top-K leaf 매칭 점수의 max로 global top-B 선택
        - `meanscore_global`: 후보 branch별 top-K leaf 매칭 점수의 mean으로 global top-B 선택
    - 공통 fallback:
        - 후보 없음 / 매칭 없음 / expandable path 매핑 실패 시 retriever-slate 결과 유지
- `iter_records`에 selector 진단 정보 추가:
    - `selector_mode`
    - `selector_pick_reason`
    - `selector_candidate_branch_count`
    - `selector_scored_top`

### 실행 스크립트
- `src/bash/round5/run_round5.sh`
    - 기본적으로 selector 2개(`retriever_slate`, `maxscore_global`)를 for-loop 실행
    - 필요 시 env로 mean 포함 가능:
```bash
ROUND5_SELECTOR_MODES="retriever_slate maxscore_global meanscore_global" \
bash src/bash/round5/run_round5.sh
```

### 실험 해석 시 주의
- category mode 비교에서 selector를 바꿀 때도 rewrite 프롬프트/카테고리 생성 단계는 동일하다.
- 따라서 차이는 branch 확장 단계의 점수 계산식에서 오는 효과로 해석해야 한다.
