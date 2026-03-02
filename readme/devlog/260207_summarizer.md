  ## 2026-02-06 Brainstorm Plan: Corpus-Grounded Dynamic Categories (Safe Explore)

  ### readme/devlog/260206.md 문서 구성안

  - 제목: # 2026-02-06 브레인스토밍: Corpus-grounded Dynamic Category
  - 섹션:
      - 문제정의 (고정 4카테고리 병목)
      - 설계 결정 (Keep+Add, 3-Query Union, Global Backoff)
      - Iteration 알고리즘
      - 실험 계획 (ablation matrix)
      - 리스크/완화 (category drift, cost, noise)
      - TODO 체크리스트

  ### Assumptions / Defaults

  - 기본 정책은 안전성 우선(precision보다 miss 방지 균형).
  - inventory는 subset별 독립 구축.
  - 초기에 prompt_topk=10, rag_topk=32, backoff_k=1, explore_add_k=1로 시작.
  - cross-domain 보강은 query-only 금지, 3-신호 합집합 고정.

## 2026-02-08 Experiment Note: History Prefix ON/OFF (summary fixed ON)

- Source: `results/BRIGHT/ndcg_summary.csv`
- Compared runs:
    - `round3_category_history_none_round3_action_v1_off_summary_on`
    - `round3_category_history_none_round3_action_v1_on_summary_on`
- Controlled factors:
    - summary context: ON (fixed)
    - prompt: `round3_action_v1`
    - anchor local rank: `none`
    - only history prefix ON/OFF changed

### Result Snapshot (nDCG@10 max)

- OFF: biology 60.81, earth_science 54.76, economics 27.55, psychology 41.42, robotics 25.16
- ON: biology 59.54, earth_science 57.35, economics 27.21, psychology 40.50, robotics 25.18
- Average over 5 subsets:
    - OFF: 41.94
    - ON: 41.96
    - Delta (ON - OFF): +0.02 (effectively neutral)

- History prefix with summarized evidence does not produce a consistent gain across subsets.
- Peak iteration tends to shift later in some subsets with history ON (biology/economics/robotics), but this did not translate into consistent max nDCG improvement.

하지만 여전히 summary 그냥 안 쓴 버전 보다는 성능이 낮음.

### Summarizer implementation (what we built)

- 목적: top-k evidence 원문을 그대로 넣지 않고, query-conditioned 핵심만 압축해서 rewrite/history 프롬프트 길이와 노이즈를 줄이기.
- 방식: `src/context_summary.py`에서 문서별 abstractive summary prompt를 만들고(근거 snippet + query), LLM이 짧은 요약(기본 64 words)을 생성.
- 실행 경로: `src/run_round3_1.py`의 `_summarize_rewrite_candidates_with_llm`가 `leaf_descs / branch_descs / router_*_descs`를 배치 요약 후 rewrite/router prompt에 주입.
- 안정화: summary prompt 해시 기반 캐시(`round3_summary`)를 사용하고, 요약 실패 시 extractive fallback을 사용.
- history 연동: iter record에 `context_summaries`를 저장하고, history prefix ON일 때 `src/history_prompts.py`가 doc-id 대신 이 요약을 우선 사용(없으면 doc-id fallback).
- 토글: `--round3_summarized_context on|off`로 제어. `off`면 요약 단계를 건너뛰고 raw evidence + doc-id history fallback을 사용.

## Final Conclusion (baseline comparison, supersedes above interim notes)

- Baseline comparison target (user-specified):
    - `RInTP=-1-NumI=10-S=round3_anchor_local_rank_none_round3_action_v1-FT=1000-PreFRS=branch-RPN=round3_action_v1-RM=concat-RE=1-RCT=5-RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=concat-RRC=leaf`
- Baseline `nDCG@10 max` (biology/earth/economics/psychology/robotics):
    - `61.24 / 57.29 / 30.40 / 44.01 / 27.15` (avg `44.02`)

- `summary=on, history=off`:
    - `60.81 / 54.76 / 27.55 / 41.42 / 25.16` (avg `41.94`)
    - Delta vs baseline avg: `-2.08`

- `summary=on, history=on`:
    - `59.54 / 57.35 / 27.21 / 40.50 / 25.18` (avg `41.96`)
    - Delta vs baseline avg: `-2.06`

### Final takeaways

- 결론 1: `history` prefix를 써도 baseline 대비 성능 이득이 없다.
- 결론 2: LLM 기반 `context summary`를 써도 baseline 대비 성능 이득이 없다.
- 따라서 현재 기준 운영 기본값은 baseline 설정(요약/히스토리 없이)으로 두는 것이 합리적이다.

## Additional conclusion (prompt/action/local-rank comparison)

- 비교 평균 nDCG@10 max (5 subsets):
    - `round3_anchor_local_rank_none + round3_action_v1`: `44.02`
    - `round3_anchor_local_rank_none + round3_agent_executor_v1`: `44.57`
    - `round3_anchor_local_rank_none + round3_free_rewrite_v2`: `43.04`
    - `round3_anchor_local_rank_v2 + round3_action_v1`: `44.59`

- 해석:
    - category 구조를 명시하지 않는 free abstractive rewrite(`43.04`)는 action 계열(`44.02~44.59`)보다 낮다.
    - explore/exploit decision을 모델 단독에 맡기는 것만으로는 큰 이득이 없다 (`44.57` vs `44.02`는 소폭 개선).
    - leaf replacement(v2)는 효과가 있긴 하지만, 현재 설정에서는 개선폭이 작아(최고 `44.59`) 여기서 큰 폭 향상은 어렵다.

## 2026-02-08 Category Assignment v2 (brief)

- 파일: `scripts/document_expansion_role.py`
- 변경 목적:
    - 문서 1개가 쿼리 정답에 기여하는 수단이 하나가 아닐 수 있으므로, category keyword를 multi-label로 확장.
    - chunk 문서 id 정규화 오류(`...숫자_숫자`, `...숫자`)를 줄여 문서 단위 병합 정확도 개선.

- 방법 요약:
    - 2-level category keyword 생성 유지:
        - level1: 역할 타입(Theory/Rule/Method/Evidence/Entity/Resource/Example/Background)
        - level2: 재사용 가능한 추상 카테고리(2~4 words)
    - 문서당 다중 키워드 허용:
        - 출력 필드 `category_level_1_keywords`, `category_level_2_keywords` 추가
        - 역호환을 위해 `category`는 primary level2 유지
    - 프롬프트에 doc id를 본문 snippet에도 명시(` [DOC_ID] ...`)해서 LLM이 문서 단위를 안정적으로 식별하도록 함.
    - 병합 로직:
        - strong pattern: `...<group>_<chunk>.txt`
        - weak pattern: `...<group>.txt` (family evidence 있을 때만 병합)

- 기대 효과:
    - retrieval 신호가 단일 라벨에 눌리지 않고, 한 문서의 복합 역할(예: Theory + Evidence)을 보존.
    - chunk 병합 누락/오병합 감소로 문서 단위 category 부여 품질 개선.
