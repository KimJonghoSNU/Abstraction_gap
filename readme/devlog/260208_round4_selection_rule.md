# 2026-02-08 Round4 Plan (Critical Revision)

## Enrichindex와 차이
- 얘는 document expansion.
- Ours: 문서에 tag된 거. cheaper, no additional encoding stage.



## Context

- 관찰된 실패 패턴:
    - LLM이 생성은 가능하지만, 중요도 선택(summarize/rerank/salience)이 불안정.
- 따라서 Round4는 역할 분리:
    - selection/control: tool (retriever score)
    - generation: LLM

## Locked (already decided)

- Retriever-only로 시작한다 (lexical/BM25 미사용).
- `--round3_category_policy` 방향은 유지한다.
- Round4 baseline 비교는 `summary=off`, `history=off`로 고정해 공정 비교한다.
- `anchor_local_rank`는 baseline 재현을 위해 `none`/`v2`만 운영한다.

## Core algorithm (proposed)

### 1) Category support score

- 입력:
    - `query_t`
    - `evidence_descs` (anchor leaf top-k)
    - `candidate category docs/text`
- 점수:
    - Round4 고정: `support(c) = topm_mean(c_emb · e_emb^T)` only
- 고정 이유:
    - 실제 retrieval은 `query concat`으로 이미 query 신호를 반영함.
    - support 단계는 "카테고리별 근거 지지량" 측정으로 역할을 분리하는 것이 목적임.

### 2) Explore/Exploit decision

- 목표: LLM action text가 아니라 score 규칙으로 모드 결정.
- 실험은 결합하지 않고 `Rule A` / `Rule B`를 독립적으로 수행한다.
- `Rule A (Simple Margin Gate)`:
    - 계산:
        - `s1 = max_c support(c)`
        - `s2 = second_max_c support(c)`
        - `margin = s1 - s2`
    - 결정:
        - `t=0`에서는 `explore` (초기 keep-all)
        - `margin >= tau_margin` 이면 `exploit`
        - 그 외에는 `explore`
    - 해석:
        - 카테고리 우위가 명확할 때만 줄이고, 애매할 때는 유지한다.

- `Rule B (Counterfactual Drop Risk)`:
    - 정의:
        - 현재 카테고리 집합 `S`의 utility를 `U(S)`라 할 때,
        - `risk(c) = (U(S) - U(S\\{c})) / max(|U(S)|, eps)`
    - 결정:
        - `t=0`에서는 `explore` (초기 keep-all)
        - `min_c risk(c) <= tau_drop`이면 해당 `c`를 drop하고 `exploit`
        - 그 외에는 `explore`
    - 해석:
        - "빼도 손실이 거의 없는 카테고리"만 제거하므로 보수적으로 exploit한다.

### 3) Category selection policy

- 현재 논의안:
    - exploit: `worst-one-drop` 고정
    - exploit safeguard: 남는 카테고리 수가 1개 이하가 되면 drop 취소
    - explore: 기존 유지 + 신규 추가 (`keep+add`)는 유지

### 4) Controller state carry-over (exploit lock)

- 문제:
    - `round3_agent_executor_v1` 프롬프트는 매 iteration마다 카테고리 문서를 다시 생성하므로, 이전 exploit 결정이 쉽게 사라질 수 있음.
- 해결:
    - 프롬프트를 바꾸지 않고 controller에서 상태를 유지한다.
    - 이전 iteration decision이 `exploit`이면 다음 rewrite에서 이전 active 카테고리 집합으로 후보를 lock한다.
    - 이전 decision이 `explore`이면 lock을 풀고 새 카테고리 후보를 허용한다.
- lock 세부 규칙:
    - lock source: 이전 `actions != PRUNE` 카테고리
    - 새 rewrite 출력에서 lock source만 필터링
    - lock source가 새 출력에 없으면 이전 문서 텍스트로 fallback해 exploit 연속성 유지
- 기대 효과:
    - "한 번 exploit하면 다음 iteration에서도 집중"이 보장되어 정책 효과를 더 명확히 평가 가능.

## Critical review of current idea

- `topm=10`은 시작 고정값으로 사용하고, 후속 ablation은 `3/5/10`으로 확인.
- support는 `topm(c·e)` only로 고정:
    - query relevance는 retrieval 단계에서 이미 반영된다는 가정을 명시함.
- “가장 별로인 카테고리 1개 제거” 규칙:
    - 직관은 좋지만, low-score가 실제로는 보완 근거일 수도 있어 query drift 유발 가능.
    - drop 규칙은 exploit에서만 사용하고 explore에서는 drop 금지 권장.

## Tau calibration evidence (5 subsets, round3 soft logs)

- 데이터:
    - 대상 subset: `biology`, `psychology`, `economics`, `earth_science`, `robotics`
    - 기준 파일: 각 subset의 `round3_category_support_none_round3_action_v1/.../all_eval_sample_dicts.pkl`
- 전체(5 subset 합산) 통계:
    - `category_support_scores` 유효 레코드: 4,484
    - `margin = top1-top2`: mean=0.01596, median=0.01219, p90=0.03491, p95=0.04387
    - 상대 margin `(top1-top2)/top1`: mean=0.01831, median=0.01413, p90=0.03982
- Rule A에서 `margin>=tau`를 exploit으로 볼 때 예상 exploit 비율 (전체):
    - tau=0.02 -> 29.35%
    - tau=0.03 -> 14.59%
    - tau=0.035 -> 9.92%
    - tau=0.04 -> 6.87%
    - tau=0.05 -> 3.12%
- subset별 exploit 비율 (tau=0.035):
    - biology=10.14%, psychology=11.05%, economics=14.05%, earth_science=7.30%, robotics=7.35%
- 결론:
    - "필요할 때만 exploit" 목표라면 `tau_margin=0.035`가 전역 기본값으로 적절함.
    - 더 보수적으로 가려면 `tau_margin=0.04`를 사용.

## Biology focused analysis (observational, round4 rule_a run)

- 분석 대상:
    - `results/BRIGHT/biology/round4/.../all_eval_sample_dicts.pkl`
    - `results/BRIGHT/biology/round4/.../all_eval_metrics.pkl`
    - 정렬 기준: rewrite decision at iter `t` -> retrieval metric at iter `t+1`
    - metric: anchor `nDCG@10` (main), local `nDCG@10` (reference)
- exploit 빈도:
    - aligned decision events: 927
    - exploit: 56 (6.04%), explore: 871 (93.96%)
- Q1. drop query가 full query보다 항상 좋은가?
    - 결론: 아니오.
    - drop events (`n=90`): mean `Δanchor=-0.9308`, mean `anchor_next=49.40`
    - full events (`n=837`): mean `Δanchor=+2.4914` (bootstrap 포함), mean `anchor_next=61.96`
    - bootstrap 영향 제거(`t>=1`) 후에도 full mean `Δanchor=+0.4342`
- Q2. exploit 시 성능 변화:
    - exploit events (`n=56`): mean `Δanchor=-0.0102` (거의 0, 약한 음수)
    - explore events (`n=871`): mean `Δanchor=+2.2987` (bootstrap 포함), `t>=1` 기준 `+0.3066`
    - 현재 설정에서는 biology에서 exploit/drop의 평균 이득 신호가 약함.
- 해석 주의:
    - 위 수치는 observational 분석이므로 인과 결론이 아님.
    - 따라서 다음 단계는 같은 rewrite 출력에서 `force_full` vs `force_drop_one` A/B 실험으로 확인 필요.

## Biology force_drop_one analysis (exploit-only run)

- 분석 대상:
    - `results/BRIGHT/biology/round4/S=biology-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=5/NumES=1000-MaxBS=2-S=round4_biology_drop_ablation_force_drop_one_round3_agent_executor_v1-FT=1000-GBT=10/PreFRS=branch-RPN=round3_agent_executor_v1-RM=concat-RE=1-RCT=5/RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=concat/RCP=soft-RCSK=2-RCST=10-RCEB=0.1-RRN=rule_a/RST=10-RRAMTau=0.035-RACM=force_drop_one/all_eval_sample_dicts.pkl`
    - 총 sample=103, iter=5 (iter1~4는 exploit step)
- step 단위(`t-1 -> t`) 성능 변화 (`global nDCG@10`):
    - 전체 exploit steps (`n=412`): 상승 33.0%, 동일 50.2%, 하락 16.7%
    - bootstrap 영향이 큰 iter1 제외 (`n=309`): 상승 23.3%, 동일 59.9%, 하락 16.8%, mean Δ=+1.03
- query 단위:
    - exploit step 중 한 번이라도 상승한 sample: 76/103 (73.8%)
    - exploit step 중 한 번이라도 하락한 sample: 55/103 (53.4%)
- 해석:
    - "exploit(drop)으로 성능이 오르는 케이스가 존재하는가?"에 대한 답은 Yes.
    - 다만 step 관점에서는 tie 비율이 높아, exploit 효과가 항상 강하게 나타나지는 않음.
    - 인과 결론(탐색/활용 중 무엇이 더 낫는지)은 `force_full` 대조군과의 직접 비교가 있어야 확정 가능.

## New experiment: anchor top-10 rerank with category+query mean

- 목적:
    - 카테고리 개별 선택보다, 전체 카테고리 신호를 동시에 사용한 통합 점수가 더 유효한지 검증.
- 방법:
    - anchor retrieval 이후 top-10 문서만 대상으로 재정렬.
    - 각 문서 점수는 아래 평균으로 계산:
        - 각 category 문서와의 support similarity 점수들
        - 원래 flat query의 retriever 점수(기본 anchor score)
    - 즉, `combined = mean([score_cat1, score_cat2, ..., score_query])`
- 구현 메모:
    - `src/run_round4.py`에서 hard-coded로 적용.
    - 재정렬 상세 로그는 `query_category_decision_signal.top10_reorder`에 저장.

## Discussion needed (decision required)

- D1. Embedding model 선택
    - Option A (권장): Diver 4B로 먼저 고정 (retrieval과 분포 일치)
    - Option B: `/data4/jaeyoung/models/Qwen3-Embedding-0.6B`로 바로 교체 (저비용)
    - 내 의견: A로 먼저 고정하고, B는 ablation으로 분리.

- D2. Explore/Exploit rule 실험 순서
    - Option A (권장): Rule A 먼저 단독 실험
    - Option B: Rule B 단독 실험
    - 내 의견: A로 먼저 기준선 확보 후 B 비교.

## Implementation plan (minimum)

- `src/run_round4.py`
    - `anchor_local_rank`를 `none|v2`만 허용
    - none/v2 각각 round3_1 baseline path와 동일 동작 유지
    - score-rule 기반 decision 로직 추가(LLM action 의존 축소)
    - exploit decision carry-over를 위한 category lock 추가
    - iteration 로그 확장:
        - `category_support_scores`
        - `decision_mode`
        - `prev_selected_categories`
        - `selected_categories`
        - `category_lock_applied`
        - `category_lock_source`

- `src/hyperparams.py`
    - round4 튜닝 인자 추가:
        - `--round4_support_topm`
        - `--round4_rule_name` (`rule_a` or `rule_b`)
        - `--round4_rule_a_margin_tau`
        - `--round4_rule_b_drop_tau`
        - `--round4_analysis_category_mode` (`default` | `force_full` | `force_drop_one`)

- `src/bash/round4/`
    - round4 전용 실행 스크립트에서 baseline 매트릭스 고정:
        - `none + action_v1`
        - `none + agent_executor_v1`
        - `none + free_rewrite_v2`
        - `v2 + action_v1`

## Evaluation target

- 기준 baseline 평균:
    - `none + action_v1`: 44.02
    - `none + agent_executor_v1`: 44.57
    - `v2 + action_v1`: 44.59
- phase-1 목표:
    - 평균 44.59 재현 또는 초과
    - 5 subset 중 3개 이상 non-negative delta

## 2026-02-10 Plan: Oracle support 기반 selection policy 분석

- 결론 먼저:
    - 가능함. oracle run 결과(`log + all_eval_sample_dicts.pkl + all_eval_metrics.pkl`)를 주면 support score 분포와 decision/action을 매칭해서 real policy 후보를 뽑을 수 있음.

- 왜 이 분석을 하나:
    - 목표는 "support score를 어떻게 쓰면 oracle selection에 가까워지는가"를 정량화하는 것.
    - 단순 margin rule(top1-top2)이 충분한지, 아니면 분포/상태(feature) 결합이 필요한지 확인.

- 입력 아티팩트(실행마다 필요):
    - oracle 결과:
        - `all_eval_sample_dicts.pkl`
        - `all_eval_metrics.pkl`
        - 실행 로그(`Iter t | Anchor nDCG@10=...`)
    - real 결과(비교용):
        - `all_eval_sample_dicts.pkl`
        - `all_eval_metrics.pkl`
        - 실행 로그

- 분석 절차:
    1. metric consistency check
        - 로그 기준 metric(`Anchor nDCG@10`)과 pickle metric을 일치 검증.
        - `nDCG@10`(anchor) 비교
    2. support feature 추출
        - per-iter: `top1`, `top2`, `margin`, `relative_margin`, entropy(추가 예정), `|categories|`.
    3. oracle decision 모사
        - oracle action(explore/exploit)을 label로 두고 threshold/rule을 fitting.
        - exploit precision/recall/F1 + predicted exploit rate를 함께 측정.
    4. real 적용 시뮬레이션
        - fitting된 rule을 real support에 적용.
        - pred_exploit vs pred_explore에서 `Anchor ΔnDCG@10` 비교.
    5. 정책 후보 도출
        - 단일 margin rule이 과-exploit/과-explore를 만들면 2~3 feature rule로 확장.

- 현재 구현 상태:
    - 분석 스크립트: `scripts/analyze_round4_oracle_policy.py`
    - 출력 JSON: `results/analysis/round4_oracle_policy_*.json`
    - ndcg mismatch 검증 로직 포함(Anchor vs Global 분리 확인).

- 주의사항(이번에 확인된 이슈):
    - 기존 일부 oracle 파일은 `query_category_support_scores`가 비어 있음.
    - 이 경우 분석은 "real support + oracle label 매칭" fallback으로 돌아가며, 해석 강도를 낮춰야 함.
    - 그래서 oracle runner에서 support score 저장을 고친 뒤 재실행 결과를 우선 사용.

- 성공 기준:
    - oracle label 모사에서 exploit F1 개선 + predicted exploit rate calibration(oracle exploit rate 근접).
    - real 적용 시 pred_exploit 구간의 평균 `Anchor ΔnDCG@10`가 pred_explore보다 유의하게 높음.
    - 위 조건을 만족하는 단순 규칙(설명 가능한 policy) 1개 이상 확보.

## 2026-02-10 Result: Oracle rerun2 분석 (biology, v1)

- 입력:
    - oracle:
        - `results/BRIGHT/biology/round4_oracle/S=biology-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=5/NumES=1000-MaxBS=2-S=round4_oracle_drop_rerun2_round3_agent_executor_v1-FT=1000-GBT=10/PreFRS=branch-RPN=round3_agent_executor_v1-RM=concat-RE=1-RCT=5/RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=concat/RCP=soft-RCSK=2-RCST=10-RCEB=0.1-RRN=rule_a/RST=10-RRAMTau=0.02-ROM=drop_one_ndcg-ROAT=1000-ROMK=10/all_eval_sample_dicts.pkl`
    - real:
        - `results/BRIGHT/biology/round4/S=biology-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5-RCF=0-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=10/NumES=1000-MaxBS=2-S=round4_rule_a_none_round3_agent_executor_v1-FT=1000-GBT=10/PreFRS=branch-RPN=round3_agent_executor_v1-RM=concat-RE=1-RCT=5/RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=concat/RCP=soft-RCSK=2-RCST=10-RCEB=0-RRN=rule_a/RST=10-RRAMTau=0/all_eval_sample_dicts.pkl`
    - 분석 출력:
        - `results/analysis/round4_oracle_policy_biology_rerun2_vs_rulea_v1.json`
        - `results/analysis/round4_oracle_policy_biology_rerun2_vs_rulea_v1_relative.json`

- 데이터 품질 체크:
    - oracle `iter_records.query_category_support_scores`: non-empty `412/515` (80.0%)
    - 분포: `{0: 103, 1: 11, 2: 51, 3: 96, 4: 254}`
    - 해석: 초기/비활성 step을 제외하면 support score 저장은 정상.

- Oracle label 모사 (단일 threshold):
    - raw margin 기준 best `tau=0.0005033`
        - exploit F1=`0.4731`, precision=`0.3106`, recall=`0.9919`
        - predicted exploit rate=`98.75%`, true exploit rate=`30.92%`
    - relative margin 기준 best `tau=0.0015855`
        - exploit F1=`0.4741` (raw와 사실상 동일)
        - predicted exploit rate=`94.26%`, true exploit rate=`30.92%`

- Real 전이 결과(핵심):
    - raw tau를 real에 적용하면 predicted exploit=`97.51%` (real actual exploit=`6.04%`)
    - relative tau를 real에 적용하면 predicted exploit=`93.17%` (real actual exploit=`6.04%`)
    - 결론: oracle에서 fit한 절대 threshold를 real에 그대로 가져오면 과도한 exploit이 발생함.

- 왜 단일 margin이 실패하는가:
    - oracle 내부에서도 explore/exploit margin 분포가 크게 겹침.
        - explore margin: mean=`0.01616`, p50=`0.01112`, p95=`0.04973`
        - exploit margin: mean=`0.01947`, p50=`0.01563`, p95=`0.05051`
    - exploit 중 `37.9%`가 explore median 이하 margin에 위치.
    - 해석: `top1-top2` 하나만으로 oracle action을 안정적으로 분리하기 어려움.

- metric mismatch 점검:
    - mismatch는 버그라기보다 metric 정의 차이로 확인됨.
    - 예: real iter9
        - anchor(mean from metrics_df)=`61.2271`
        - global(mean from iter_records)=`61.2837`
    - 따라서 앞으로 비교 기준은 `Anchor nDCG@10`으로 고정해서 해석.

- 이번 분석에서 얻은 결정:
    - 단일 margin threshold는 policy로 채택하지 않음.
    - 다음 후보는 "rate-controlled exploit" 계열로 제한:
        - online 분포 기준 quantile gate (상위 q만 exploit)
        - + warmup 이후 적용 (`t>=2`) + 최소 category 수 safeguard 유지

## 2026-02-10 추가 분석: Oracle drop category 빈도 + support 관련성

- 분석 대상:
    - `results/BRIGHT/biology/round4_oracle/S=biology-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm/Llm=Qwen3-4B-Instruct-2507-NumI=5/NumES=1000-MaxBS=2-S=round4_oracle_drop_rerun2_round3_agent_executor_v1-FT=1000-GBT=10/PreFRS=branch-RPN=round3_agent_executor_v1-RM=concat-RE=1-RCT=5/RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=concat/RCP=soft-RCSK=2-RCST=10-RCEB=0.1-RRN=rule_a/RST=10-RRAMTau=0.02-ROM=drop_one_ndcg-ROAT=1000-ROMK=10/all_eval_sample_dicts.pkl`
    - 유효 이벤트(`support>=2`): `401`

- 카테고리별 drop 빈도 (`best_candidate=drop_*`):
    - 전체 drop 선택: `124`
    - `Other=40`, `Theory=32`, `Example=31`, `Entity=21`
    - iteration별 drop 총량: `iter1=47`, `iter2=31`, `iter3=28`, `iter4=18`
    - 관찰: 이 런에서는 `Other`가 가장 자주 제거됨.

- support로 best-drop을 설명할 수 있는지:
    - exploit(drop) 이벤트에서 top-support category가 drop된 비율: `39.52%` (`49/124`)
    - 즉 "`top-support는 항상 keep`" 가설은 oracle 행동과 자주 충돌.
    - 전체 pair 기준 `corr(support, drop_gain)`:
        - `drop_gain = score(drop_c) - score(keep_all)`
        - 상관계수 `-0.0094` (거의 0)
    - tie(`drop_gain=0`) 제외해도 상관계수 `0.0050` (여전히 거의 0)

- 안전 제약 후보 검증 (`never_drop_top1_support`):
    - `regret` 정의(이 문서에서):
        - 각 이벤트에서 oracle이 고를 수 있는 최고 점수와, 제약을 건 정책이 고르는 점수의 차이
        - 식: `regret = max(candidate_scores) - candidate_scores[constrained_choice]`
        - 여기서 `candidate_scores`는 `{keep_all, drop_Theory, ...}`의 oracle `nDCG@10` 값
        - 해석:
            - `regret = 0`: 제약 정책이 oracle 최고 선택과 동일한 점수를 냄
            - `regret > 0`: 제약 때문에 더 좋은 선택을 놓침 (`nDCG@10` 손실)
    - 예시:
        - `keep_all=70.0`, `drop_Theory=74.0`, `drop_Entity=71.0`이고
        - 제약이 `drop_Theory`를 금지해서 `drop_Entity`를 고르면
        - `regret = 74.0 - 71.0 = 3.0`
    - oracle best 대비 regret:
        - mean=`0.9086`, p50=`0.0`, p90=`0.3393`, p95=`5.0499`
    - regret 발생 비율:
        - `regret>0`: `10.72%`
        - `regret>1.0`: `9.48%`
    - 해석: top1-drop 금지는 평균적으로는 큰 변화가 없어 보일 수 있지만, 일부 케이스에서 손실 tail이 큼.

- 이번 추가 분석 결론:
    - support 단일 지표만으로 oracle drop 결정을 재현하기는 어려움.
    - support는 hard rule(절대 금지/절대 drop)보다 soft prior로 쓰는 것이 맞음.
    - 다음 정책은 아래 형태가 더 타당:
        - top1-drop 금지를 기본으로 두되, 예외 해제 조건(반복 실패/추세 악화)을 둔 bounded policy.
        - 또는 support 외 신호 1개 이상(예: 빠른 counterfactual proxy score)을 결합.
