# 2026-03-24 How to improve more

## Question

`round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat`는 iter `3-4`에서 성능이 크게 흔들렸다.

확인하고 싶었던 것은 두 가지다.

- 이 dip이 특정 subset 하나의 문제인가
- 아니면 `ended_reseat` transition 자체의 구조적 문제인가

## Baseline diagnosis

대상 run:

- `MaxBS=10`
- `RMP=reason-embed-qwen3-8b-0928`
- `round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat`

`results/BRIGHT/ndcg_iter_summaryembed.csv` 기준 overall curve:

- `36.02, 35.94, 35.95, 33.72, 31.62, 35.97, 36.19, 36.13, 36.04, 35.91`

즉 실제 shock는 iter `3-4`에 있다.

구조적으로 보면 같은 시점에 다음이 같이 일어난다.

| iter | nDCG mean | ended_beam_count mean | reseat_rate | pool_has_any mean | pool_gold_doc_recall mean |
|---|---:|---:|---:|---:|---:|
| 2 | 35.95 | 0.10 | 0.06 | 0.95 | 0.90 |
| 3 | 33.72 | 3.36 | 0.82 | 0.84 | 0.74 |
| 4 | 31.62 | 8.44 | 0.99 | 0.80 | 0.69 |
| 5 | 35.97 | 7.07 | 1.00 | 0.85 | 0.76 |

핵심 해석:

- iter `3`에서 reseat가 본격적으로 시작된다.
- iter `4`에서는 거의 full-beam reseat 상태가 된다.
- 그 순간 현재 retrieval pool 아래의 gold doc coverage도 같이 줄어든다.

중요한 점은, iter `5` 회복이 full reset은 아니라는 것이다.

- `nDCG`: `31.62 -> 35.97`
- `pool_has_any`: `0.80 -> 0.85`
- `pool_gold_doc_recall`: `0.69 -> 0.76`

즉 iter `4` shock는 실제 retrieval pool 기준으로도 존재하지만, iter `5`부터는 pool 안의 gold doc recoverability가 다시 올라간다.

## What to try

이 진단을 보고 다음 두 방향을 우선순위로 봤다.

1. `full-reseat guard`
    - 한 iter에 거의 beam 전체를 reseat하지 않게 막기
2. `transition-state carryover`
    - reseat 직후 retrieval pool 아래 gold docs가 급격히 사라지지 않게 만들기

## 2026-03-24 update: implemented next try

이번 수정은 `ended_reseat` transition shock를 줄이기 위한 것이다.

- retrieval pool
    - `current beam frontier descendants + cumulative reached leaves`
- query state
    - 일반 step은 새 rewrite로 교체
    - ended-beam transition step은 이전 query state 뒤에 새 rewrite를 append

의도:

- reseat 직후에도 이미 확보한 evidence를 잃지 않게 하기
- leaf-ended transition에서 retrieval direction이 갑자기 끊기지 않게 하기

## Part 1. Reseat depth를 제한해야 한다는 가설은 맞았다

왜 이 분석이 필요했나:

- `frontiercum_qstate`는 `ended_reseat` shock를 크게 줄였지만,
- reseat depth 자체는 여전히 iter `3+`에서 depth `3-4` 수준으로 깊었다.
- 즉 `pool carryover`는 해결했어도 `too-deep reseat`는 그대로 남아 있었다.

이 가설을 더 강하게 만든 신호는 random reseat였다.

- `frontiercum_qstate`에서 random reseat가 score reseat보다 더 좋았다.
- 이건 score reseat가 local score를 과신해서 beam을 너무 깊고 좁은 branch family로 빨리 수축시킬 수 있다는 뜻으로 읽었다.

그래서 `reseat_depth_batch_v1`를 구현했다.

- `ended_reseat` replacement에만 depth control을 건다.
- sample마다 `active reseat depth`를 둔다.
- 현재 active depth의 unseen endpoint를 먼저 보고,
- 같은 depth에서 이미 reseat에 쓴 endpoint는 skip하고,
- active depth가 현재 ended slot을 못 채울 때만 더 깊은 depth로 spill한다.

핵심 결과:

- reseat depth mean이 실제로 내려갔다.
    - non-EMR: `3.44 -> 2.80`
    - EMR: `3.46 -> 2.81`
- early reseat 구간 `iter 3-4`에서도 차이가 크다.
    - non-EMR: `3.28 / 3.54 -> 1.79 / 1.99`
    - EMR: `3.29 / 3.54 -> 1.79 / 1.99`

성능도 같이 봤다.

- `results/BRIGHT/ndcg_end_summaryreason_maxiter8.csv` 기준
- `aops`, `leetcode`, `theoremqa_questions` 제외
- EMR line끼리 비교하면:
    - `frontiercum_qstate_v1 + EMR`: `ndcg_end = 43.62`
    - `reseat_depth_batch_v1 + EMR`: `ndcg_end = 43.68`

즉 gain이 아주 크지는 않지만 방향은 분명하다.

- reseat depth를 실제로 얕게 만들 수 있었고
- matched EMR comparison에서도 end performance가 소폭 올라갔다

결론:

- `depth를 통제해야 한다`는 진단은 맞았다.
- 적어도 현재 line에서는 reseat depth를 줄이는 것이 유효한 control variable이다.

## Part 2. Branch를 다양하게 보여주는 것만으로는 부족했다

왜 이 분석이 필요했나:

- branch desc 자체는 semantic explanation으로 중요하지 않다.
- tree의 역할은 비슷한 내용을 묶어 `이 방향을 계속 볼지 / 다른 방향으로 넘어갈지`를 구조화하는 데 있다.
- 그렇다면 rewrite prompt에도 global top-k 대신 beam-balanced evidence를 넣는 것이 도움이 될 수 있다고 봤다.

그래서 `frontiercum_beampack_v1`를 구현했다.

- base는 그대로 `frontiercum_qstate`다.
- 바꾸는 것은 pre-rewrite evidence selection 하나다.
- 각 beam마다 descendants에서 top-1 evidence를 하나씩 뽑는다.
- prompt는 그대로 mixed `Context Summaries`를 쓴다.
- old/new labeling이나 branch desc semantics는 넣지 않는다.

결과:

- `results/BRIGHT/ndcg_end_summaryreason_maxiter8.csv` 기준
    - `frontiercum_qstate`: `44.13`
    - `frontiercum_beampack_v1`: `44.02`

즉 beam마다 다른 branch evidence를 보여주는 구조 자체는 완전히 틀린 방향은 아니지만, 이것만으로는 성능 bottleneck을 설명하지 못했다.

해석:

- 특정 branch family가 prompt를 독점하는 문제는 일부 줄였을 수 있다.
- 하지만 현재 bottleneck은 단순히 `rewrite evidence diversification`만으로는 풀리지 않는다.
- 더 강한 효과는 여전히 reseat depth control 쪽에서 나왔다.

결론:

- branch를 다양하게 보여주는 것만으로는 충분하지 않았다.
- beam-balanced evidence exposure는 side factor일 수는 있어도 main factor는 아니었다.

## Part 3. Leaf-only baseline과 비교하면, tree search의 의미는 local optima 탈출이다

왜 baseline과 비교해야 하나:

- tree search의 장점이 단순히 structure를 추가하는 데 있는지,
- 아니면 stale한 local direction에서 실제로 빠져나오는 operator를 주는 데 있는지를 구분해서 봐야 한다.

이번 비교는 fair하게 아래처럼 맞췄다.

- EMR line끼리 비교
- `aops`, `leetcode`, `theoremqa_questions` 제외
- `ndcg_max`가 아니라 `ndcg_end` 기준

비교 대상:

- baseline:
    - `baseline3_leaf_only_loop_emr`
    - subset 제외 final-iter mean, 즉 `ndcg_end = 43.10`
- tree-search:
    - `frontiercum_qstate_v1 + EMR`
    - subset 제외 `ndcg_end = 43.62`
- tree-search + depth control:
    - `reseat_depth_batch_v1 + EMR`
    - subset 제외 `ndcg_end = 43.68`

즉 같은 subset 집합, 같은 EMR line에서 보면 tree-search 쪽이 baseline보다 높다.

이 차이는 performance-only로 보면 안 된다. mechanism 차이가 더 중요하다.

이번에는 이를 더 직접적으로 보기 위해 `results/BRIGHT/analysis/round6_vs_baseline3_escape_operator_system_summary.csv`를 만들었다.

- EMR-only
- `aops`, `leetcode`, `theoremqa_questions` 제외
- local branch-like neighborhood는 prefix depth `3` proxy로 정의
- headline surface는 feedback top-10
    - baseline: 현재 step `retrieved_paths[:10]`
    - ours: reseat가 일어난 다음 step의 `pre_hit_paths[:10]`
- metric 설명
    - reseat_uptake_any_feedback@10
       - whether any top-10 feedback path is descendant of ended_beam_reseat_selected_paths_t
    - reseat_uptake_pct_feedback@10
        - fraction of top-10 feedback paths that are descendants of ended_beam_reseat_selected_paths_t

이 기준으로 보면 baseline과 ours의 차이가 분명하다.

- baseline `baseline3_leaf_only_loop_emr`
    - `escape_any_feedback@10_proxy = 19.3%`
    - `escape_pct_feedback@10_proxy = 3.0%`
- `frontiercum_qstate_v1 + EMR`
    - `escape_any_feedback@10_proxy = 79.1%`
    - `escape_pct_feedback@10_proxy = 62.1%`
- `reseat_depth_batch_v1 + EMR`
    - `escape_any_feedback@10_proxy = 76.4%`
    - `escape_pct_feedback@10_proxy = 65.4%`

즉 baseline은 현재 query가 끌고 가는 local branch-like neighborhood 안에서 retrieval이 반복되기 쉽다. 반면 ours는 reseat 이후 next-step feedback top-10 안에 이전 local proxy 바깥의 문서를 대량으로 들여올 수 있다.

strict하게 봐도, ours는 reseat-selected branch에서 온 문서를 next-step top-10에 실제로 주입한다.

- `frontiercum_qstate_v1 + EMR`
    - `reseat_uptake_any_feedback@10 = 8.1%`
    - `reseat_uptake_pct_feedback@10 = 1.6%`
- `reseat_depth_batch_v1 + EMR`
    - `reseat_uptake_any_feedback@10 = 11.3%`
    - `reseat_uptake_pct_feedback@10 = 2.0%`

이 값은 broad escape metric보다 작다. 즉 top-10 전체가 reseat-selected branch로 갈아끼워지는 것은 아니다. 하지만 중요한 점은, baseline과 달리 ours에는 explicit escape / reseat operation이 있고, 그 operator가 next-step feedback에 new region 문서를 실제로 주입할 수 있다는 것이다.

proxy depth를 `2/3/4`로 바꿔도 sign은 안 뒤집힌다.

- baseline escape-any rate:
    - `10.9% / 19.3% / 24.7%`
- `frontiercum_qstate_v1 + EMR`:
    - `53.7% / 79.1% / 80.8%`
- `reseat_depth_batch_v1 + EMR`:
    - `62.8% / 76.4% / 76.9%`

즉 이 차이는 proxy depth를 어떻게 잡아도 유지된다.

utility는 support evidence로만 쓰는 것이 맞다.

- baseline에서 escape event는 rare하고, 그때의 `delta_ndcg10_end` mean도 `-0.44`로 오히려 나쁘다.
- ours는 escape가 common하지만 utility sign은 mix되어 있다.
- 따라서 "escape가 곧바로 성능 상승을 보장한다"고 쓰는 것은 과하다.
- 대신 더 defensible한 claim은, ours가 baseline과 달리 stale한 local direction에서 벗어나는 explicit operator를 가진다는 점이다.

이 해석은 기존 baseline 분석과도 맞는다.

- `results/BRIGHT/analysis/round6_vs_baseline3_offbranch_overall.csv`
    - `baseline3_leaf_only_proxy`
    - `OffBranchPct@K_mean_feedback_macro = 4.16`
    - `OffBranchEventRate_macro = 25.95`
- `results/BRIGHT/analysis/baseline3_derailment/baseline3_derailment_summary.csv`
    - `overall, depth=3`
    - `AnchorQueryRate = 76.59%`
    - `EndMiss@10|FirstAnchor = 26.23%`
    - `NextTop10DriftRate|FirstAnchor = 6.04%`
    - `NextTop100DriftRate|FirstAnchor = 0.19%`

보수적으로 읽으면 baseline도 promising local branch-like state에 도달할 수는 있다. 다만 그 local direction이 충분하지 않을 때, query drift 외에 다른 branch family로 넘어가는 structured jump가 없다.

결론:

- leaf-only baseline과 비교했을 때 tree search의 핵심 이점은 local optima 탈출 operator를 가진다는 점이다.
- `reseat`, `depth batch`는 단순한 structure가 아니라, stale한 local branch state를 벗어나는 explicit 제어다.
