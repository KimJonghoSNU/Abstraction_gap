# 260327: Zero-nDCG Failure by Gold-Prefix Depth

## Purpose

`nDCG@10 = 0`인 iteration들이 정말로 "완전히 틀린 방향으로 간 것"인지, 아니면 gold branch family 안까지는 들어갔는데도 최종 문서 ranking이 실패한 것인지 분리해서 본다.

이번 분석은 LLM evaluator를 쓰지 않고, 저장된 tree / retrieval artifact만으로 본다.

- 대상 run:
  - `round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat_frontiercum_qstate`
  - `RMP=reason-embed-qwen3-8b-0928`
  - `RPN=agent_executor_v1_icl2`
  - `RB=frontiercum_qstate_v1`
- 제외:
  - `random`
  - `emr`
  - `descendant_flat`

## Method

스크립트:

- `scripts/analysis/analyze_round6_zero_ndcg_gold_prefix_depth.py`

실행:

```bash
python -m py_compile scripts/analysis/analyze_round6_zero_ndcg_gold_prefix_depth.py

python scripts/analysis/analyze_round6_zero_ndcg_gold_prefix_depth.py \
    --out_prefix results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth
```

출력:

- `results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth_rows.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth_selected_depth_hist.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth_prehit_depth_hist.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth_active_depth_hist.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth_subset_depth_hist.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth_transition_summary.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_prefix_depth_examples.csv`

정의:

- `gold-prefix depth`
  - candidate path와 gold path 사이의 `longest common prefix length`
- stage
  - `selected`: `selected_branches_after`
  - `prehit`: `pre_hit_paths`
  - `active`: `active_eval_paths`

중요한 주의:

- `selected`는 branch path이고
- `prehit` / `active`는 leaf path라서
- absolute depth를 1:1로 직접 비교하는 것보다, 각 stage의 histogram을 따로 읽는 게 더 중요하다.

## Results

대상 row 수:

- matched subsets: `9`
- `nDCG=0` rows: `2911`

### 1. Selected branch depth distribution

`selected_branches_after` 기준 `best gold-prefix depth`

- depth `0`: `13.57%`
- depth `1`: `32.22%`
- depth `2`: `33.94%`
- depth `3`: `15.25%`
- depth `4`: `4.74%`
- depth `5`: `0.27%`

핵심:

- `depth 0-1`만 합치면 `45.79%`
- `depth 2+`는 `54.21%`

즉 `nDCG=0`의 과반은 branch selection이 최소 depth 2 이상까지는 gold 방향을 따라간다.

### 2. Retrieval evidence depth distribution

`pre_hit_paths` 기준 `best gold-prefix depth`

- depth `0`: `5.91%`
- depth `1`: `8.04%`
- depth `2`: `11.89%`
- depth `3`: `3.92%`
- depth `4`: `42.36%`
- depth `5`: `26.00%`
- depth `6`: `1.89%`

핵심:

- `prehit depth 2+`가 `86.05%`
- `depth 4-5`만 합쳐도 `68.36%`

즉 retrieval evidence 자체는 꽤 자주 gold leaf family 깊은 쪽까지 들어간다.

### 3. Final active ranking depth distribution

`active_eval_paths` 기준 `best gold-prefix depth`

- depth `0`: `4.88%`
- depth `1`: `7.25%`
- depth `2`: `9.38%`
- depth `3`: `1.75%`
- depth `4`: `42.63%`
- depth `5`: `31.43%`
- depth `6`: `2.68%`

핵심:

- `active depth 2+`가 `87.87%`
- `depth 4-5`만 합쳐도 `74.06%`

즉 최종 active ranking도 상당수는 gold branch family 깊은 곳까지 도달해 있다.  
그런데도 `nDCG@10 = 0`이라는 뜻이다.

### 4. Stage transition summary

- `branch_miss`: `13.57%`
- `branch_to_prehit_drop`: `6.39%`
- `depth_preserved`: `80.04%`

즉 zero-ndcg의 대부분은

- branch에서 맞던 방향을 retrieval evidence가 잃어버려서 망한 경우
보다는
- branch / retrieval 모두 같은 gold family 쪽을 계속 보고 있는데도 final ranking이 실패한 경우

가 더 많다.

## Interpretation

이번 결과에서 가장 중요한 문장은 이거다.

- `nDCG=0`이 곧바로 `wrong direction rewrite / wrong branch`를 뜻하지는 않는다.

오히려 현재 evidence는:

- 완전한 branch miss (`selected depth = 0`)는 `13.57%`
- 얕은 방향 실패 (`selected depth = 0 or 1`)도 `45.79%`
- 반대로 과반(`54.21%`)은 `selected depth >= 2`
- retrieval/active 단계에서는 `depth 4-5`가 주류

즉 많은 zero-ndcg case는

- gold 방향 자체를 완전히 놓친 것
보다는
- gold branch family 안까지는 들어갔는데
- top-10 문서 ranking이 실패한 것

에 더 가깝다.

실무적으로는 이렇게 읽는 게 맞다.

- `selected depth 0-1` row:
  - controller / branch choice가 얕게 틀린 failure
- `selected depth 2+`인데도 `nDCG=0`:
  - detail / ranking / document ordering failure 가능성이 더 큼

## Subset differences

`selected_branches_after` 기준 mean gold-prefix depth와 `depth 0-1` 비율:

- `theoremqa_theorems`
  - mean depth `1.35`
  - depth `0-1`: `57.74%`
- `earth_science`
  - mean depth `1.36`
  - depth `0-1`: `53.22%`
- `economics`
  - mean depth `1.40`
  - depth `0-1`: `56.49%`
- `robotics`
  - mean depth `1.56`
  - depth `0-1`: `46.93%`
- `biology`
  - mean depth `1.64`
  - depth `0-1`: `43.20%`
- `psychology`
  - mean depth `1.66`
  - depth `0-1`: `42.40%`
- `sustainable_living`
  - mean depth `1.73`
  - depth `0-1`: `46.93%`
- `pony`
  - mean depth `1.95`
  - depth `0-1`: `38.00%`
- `stackoverflow`
  - mean depth `2.03`
  - depth `0-1`: `32.71%`

해석:

- `theoremqa_theorems`, `earth_science`, `economics`는 zero-ndcg가 상대적으로 더 shallow branch miss 쪽이다.
- `stackoverflow`, `pony`는 zero-ndcg에서도 branch depth는 더 깊게 간다.
  - 즉 이 subset들은 direction miss보다 ranking/detail failure 비중이 더 클 가능성이 높다.

## Takeaway

이번 분석만 놓고 보면, 다음 단계는 LLM evaluator를 곧바로 붙이는 게 아니다.

먼저 이렇게 가는 게 맞다.

1. `selected depth 0-1` bucket
   - controller / branch selection 문제로 본다
2. `selected depth 2+` bucket
   - retrieval/ranking 문제로 본다
3. 특히 `active depth 4-5`인데 `nDCG=0`인 case를 따로 보면
   - "gold branch family 안까지는 갔는데 왜 top-10이 실패하나?"를 더 직접 볼 수 있다

즉 다음 error analysis 축은:

- `wrong direction` 여부를 LLM이 판단하게 하는 것보다
- `gold branch family 안에 들어간 상태에서도 문서를 못 올리는 이유`
를 보는 쪽이 더 정보량이 크다.

## Follow-up: inside the gold family, where does exact gold disappear?

위 depth 분석만으로는 아직 부족했다.  
그래서 `nDCG=0` row 중에서, 이미 **gold leaf neighborhood까지 들어간 row**만 다시 골라 exact gold doc이 어디서 사라지는지 봤다.

스크립트:

- `scripts/analysis/analyze_round6_zero_ndcg_gold_ladder.py`

실행:

```bash
python -m py_compile scripts/analysis/analyze_round6_zero_ndcg_gold_ladder.py

python scripts/analysis/analyze_round6_zero_ndcg_gold_ladder.py \
    --out_prefix results/BRIGHT/analysis/round6_zero_ndcg_gold_ladder
```

출력:

- `results/BRIGHT/analysis/round6_zero_ndcg_gold_ladder_rows.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_ladder_overall_summary.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_ladder_subset_summary.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_ladder_rank_hist.csv`
- `results/BRIGHT/analysis/round6_zero_ndcg_gold_ladder_examples.csv`

정의:

- `family_hit_active_near_leaf`
  - `active_best_gold_prefix_depth >= min_gold_leaf_depth_for_query - 1`
  - 즉 query별 gold leaf depth를 기준으로, gold leaf 바로 위 parent neighborhood까지는 들어간 경우
- exact-gold ladder
  - `prehit100 -> active100 -> active10`

### Overall result

- total `nDCG=0` rows: `2911`
- `family_hit_active_near_leaf`: `2226` rows = `76.47%`
- `family_miss`: `685` rows = `23.53%`

즉 zero-ndcg의 약 `3/4`는 이미 gold leaf neighborhood까지는 들어간다.

### Exact gold ladder inside family-hit rows

family-hit rows 안에서:

- `gold_absent_from_prehit100_and_active100`
  - `253` rows = `11.37%`
- `gold_present_in_prehit100_but_missing_in_active100`
  - `34` rows = `1.53%`
- `gold_recovered_in_active100_after_prehit_miss_but_missing_in_active10`
  - `17` rows = `0.76%`
- `gold_present_in_prehit100_and_active100_but_missing_in_active10`
  - `1922` rows = `86.34%`
- `gold_present_in_active10`
  - `0`

핵심:

- family-hit rows의 대다수는
  - exact gold doc이 `prehit100`에도 있고
  - `active100`에도 있는데
  - `active10`만 실패한다.

즉 주된 failure point는 `family 내부 final top-10 competition`이다.

### Rank summary

family-hit rows에서:

- `gold_in_prehit100`: `87.87%`
- `gold_in_active100`: `87.11%`
- `gold_in_active10`: `0.00%`
- `best_gold_rank_active100` mean: `31.97`
- `best_gold_rank_active100` median: `24`

rank histogram:

- `1-10`: `0.00%`
- `11-20`: `41.00%`
- `21-50`: `41.00%`
- `51-100`: `18.00%`

해석:

- exact gold doc은 아예 없는 게 아니라
- 대부분 이미 rank `11-50` 근처까지 올라와 있다.
- 그런데 top-10의 마지막 압축 단계에서 계속 밀린다.

### Is top-10 filled with wrong docs from the same family?

이것도 같이 봤다.

- `on_family_top10_ratio` mean: `0.0336`
- `on_family_top10_non_gold_ratio` mean: `0.0336`
- `on_family_top100_ratio` mean: `0.0618`
- `on_family_top100_non_gold_ratio` mean: `0.0390`

이 결과는 중요하다.

- top-10이 같은 family의 틀린 문서들로 꽉 차서 지는 그림은 아니다.
- 오히려 exact gold family 문서들은 top-100 안에는 어느 정도 살아 있지만,
- top-10은 대체로 그 family 바깥 문서들한테 밀린다.

즉 문제는

- `same-family wrong sibling docs dominate top10`
보다는
- `gold family docs are present around ranks 11-50, but final ranking still prefers off-family docs`

에 더 가깝다.

### Subset pattern

모든 subset에서 dominant bucket은 같다.

- `gold_present_in_prehit100_and_active100_but_missing_in_active10`

예:

- `biology`: `75.74%`
- `earth_science`: `64.91%`
- `economics`: `59.62%`
- `psychology`: `68.90%`
- `pony`: `78.50%`
- `stackoverflow`: `51.86%`
- `theoremqa_theorems`: `76.99%`

즉 이 failure mode는 특정 subset 하나의 이상치가 아니라, run 전반의 공통 패턴이다.

## Updated takeaway

현재까지의 strongest conclusion은 이거다.

- zero-ndcg의 주 원인은 `wrong direction`이 아니다.
- 많은 실패는 gold family까지는 이미 들어간다.
- exact gold doc도 `prehit100`과 `active100`에는 살아 있다.
- 그런데 `active10`에서 계속 밀린다.

즉 지금 가장 load-bearing한 failure는:

- `branch controller miss`
보다
- `family 내부가 아니라, final rank compression에서 off-family docs를 과대선호하는 문제`

다음 분석/개선은 이 방향이 맞다.

1. `active100` 안에서 gold를 더 위로 올리는 reranking signal이 무엇인지 보기
2. `active10`을 차지하는 off-family docs가 어떤 branch family에서 오는지 보기
3. query rewrite 자체보다, `final ranking preference`와 `branch-conditioned scoring`을 먼저 점검하기

## Short 5-way partition of zero-nDCG rows

용어를 먼저 고정한다.

- `exact gold doc`
  - 정답 문서 그 자체
- `family`
  - 정답 leaf 바로 근처 subtree
  - 구현상으로는:
    - `active_best_gold_prefix_depth >= min_gold_leaf_depth_for_query - 1`
  - 쉽게 말하면:
    - 정답 leaf의 부모 근처까지는 갔는가

예:

- gold path가 `[5, 8, 6, 1, 21]`이면
  - `[5, 8, 6, 1, 22]`, `[5, 8, 6, 1, 23]`은 같은 family다.
- 즉 방향은 거의 맞지만, exact gold leaf는 아니다.
- `prehit100`
  - rewrite 전 `query_pre`로 retrieve한 상위 100 문서다.
- `active100`
  - rewrite 후 `query_post`로 retrieve/rerank한 최종 상위 100 문서다.

`nDCG=0` row만 놓고, 서로 안 겹치게 100%로 나누면 다음 5개다.

| Case | Meaning | Percent of `nDCG=0` rows |
| --- | --- | --- |
| `family miss` | 정답 leaf 근처 subtree까지도 못 감 | `23.53%` |
| `family hit, but exact gold absent in prehit100 and active100` | 방향은 맞는데 exact gold 문서는 아직 못 가져옴 | `8.69%` |
| `family hit, exact gold in prehit100 and active100, but not in active10` | 정답 문서는 후보에 이미 있는데 top-10으로 못 올림 | `66.03%` |
| `family hit, exact gold in prehit100 but missing in active100` | retrieval evidence에는 있었는데 final ranked 100에서 사라짐 | `1.17%` |
| `family hit, exact gold absent in prehit100 but recovered in active100, still not in active10` | prehit에는 없었지만 active100에서 회복, 그래도 top-10 실패 | `0.58%` |

합계:

- `23.53 + 8.69 + 66.03 + 1.17 + 0.58 = 100%`

짧게 읽으면 핵심은 이거다.

- `nDCG=0`의 주 원인은 `family miss`가 아니다.
- 가장 큰 실패는:
  - 정답 근처까지 갔고
  - exact gold 문서도 후보 `100` 안에 있는데
  - `top-10`으로 못 올리는 것
