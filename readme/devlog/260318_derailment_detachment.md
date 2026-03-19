이 문서는 `partial reseat`에 대한 현재 결론만 남긴 요약본이다. 일반적인 detachment / derailment 이론 전체를 정리하는 문서가 아니라, 지금까지 실제 로그 분석으로 확인한 사실과 그 근거를 적는다.

## 핵심 결론

현재 가장 방어 가능한 결론은 아래 두 문장이다.

- `partial reseat`의 immediate drop은 `new reseated branch`가 다음 iter top-10을 바로 점유해서 생기는 것이 아니다.
- `partial reseat`가 `freeze_terminal`보다 최종 성능이 더 좋아질 수 있는 이유는, reseated branch가 beam state에는 남아 있다가 `t+2`부터 retrieval evidence로 늦게 나타나는 `delayed activation` 때문일 가능성이 가장 높다.

조금 더 정확히 쓰면 이렇다.

- `t+1`에서는 retrieval과 rewrite evidence가 여전히 거의 전부 `old ended subtree`에 머문다.
- 동시에, `t`에서 잡았던 gold-aligned branch를 기준으로 보면 `t+1` retrieval은 그 promising region에서 완전히 이탈한다.
- 즉 immediate cost는 `new branch takeover`가 아니라 `promising branch region`에서의 detachment / transition mismatch다.
- 반면 reseated branch 자체는 beam state에 잠깐 살아남고, 그 영향이 `t+2` 이후부터 retrieval에 나타난다.
- 그래서 `partial reseat`는 `t+1`에는 손해를 보지만, 이후에는 `freeze_terminal`보다 좋아질 수 있다.

## 어떤 코드로 이 결론을 만들었는가

### 1. `scripts/analysis/analyze_round6_partial_reseat.py`

이 스크립트는 첫 reseat transition에서 다음 질문을 직접 본다.

- `first reseat` 직후 성능이 왜 떨어지는가?
- 다음 iter top-10과 rewrite evidence는 tree 상 어디에 있는가?

입력 아티팩트:

- round6: `results/BRIGHT/*/round6/**/all_eval_sample_dicts.pkl`
- baseline: `results/BRIGHT/*/**/leaf_iter_metrics.jsonl`

핵심 로직:

- query마다 첫 `ended_beam_reseat` event만 잡는다.
- `ended_beam_count <= 2`이면 `partial`로 분류한다.
- `t+1`의 `active_eval_paths`와 `pre_hit_paths`를 아래 4개 bucket으로 나눈다.
  - `old_ended`: `ended_beam_paths` descendant
  - `old_active`: reseat 직전 살아 있던 active branch descendant
  - `new_reseated`: `ended_beam_reseat_selected_paths` descendant
  - `other`: 위 세 bucket에 속하지 않는 path
- 같은 `(subset, query_idx, iter)`의 baseline `nDCG@10` delta도 같이 본다.

출력 CSV:

- `results/BRIGHT/analysis/round6_partial_reseat_rows.csv`
- `results/BRIGHT/analysis/round6_partial_reseat_group_summary.csv`
- `results/BRIGHT/analysis/round6_partial_reseat_subset_summary.csv`

이 스크립트가 준 핵심 숫자:

- `partial` rows: `n = 530`
- round6 next-step delta:
  - `nDCG@10(t+1) - nDCG@10(t) = -7.85`
- baseline same-step delta on the same queries:
  - `-0.14`
- next top-10 average bucket counts:
  - `old_ended = 9.13`
  - `old_active = 0.04`
  - `new_reseated = 0.00`
  - `other = 0.01`
- next rewrite-evidence pool average bucket counts:
  - `old_ended = 9.13`
  - `old_active = 0.05`
  - `new_reseated = 0.00`
  - `other = 0.01`

이 결과가 뜻하는 바는 명확하다.

- immediate drop은 `new reseated branch`가 다음 iter ranking을 먹어서 생긴 것이 아니다.
- `t+1`의 ranking과 rewrite evidence는 여전히 거의 전부 `old ended subtree`에 남아 있다.

즉 이 단계에서 먼저 버려야 하는 가설은 다음이다.

- `new branch docs immediately enter top-10 and ruin the ranking`

### 2. `scripts/analysis/analyze_round6_derailment.py`

위 스크립트만으로는 `old ended subtree` dominance가 정말 `gold-aligned promising region`을 떠난 것인지까지는 말할 수 없다. 이걸 보기 위해 gold-grounded anchor 분석을 따로 했다.

입력 아티팩트:

- round6: `results/BRIGHT/*/round6/**/all_eval_sample_dicts.pkl`
- baseline: `results/BRIGHT/**/leaf_iter_metrics.jsonl`

핵심 로직:

- 각 iter `t`에서 `selected_branches_after` 중 gold path의 prefix인 branch만 `anchor_branches`로 둔다.
- 즉 `t` 시점에 시스템이 실제로 선택한 branch 중 gold-aligned branch만 anchor로 삼는다.
- `t+1`의 `active_eval_paths`와 `pre_hit_paths`가 이 anchor subtree 안에 얼마나 남아 있는지 본다.
- `on_region_share`가 낮아지면, promising region에서 detachment가 일어난 것이다.

출력 CSV:

- `results/BRIGHT/analysis/round6_derailment_rows.csv`
- `results/BRIGHT/analysis/round6_derailment_summary.csv`
- `results/BRIGHT/analysis/round6_derailment_by_subset.csv`

이 스크립트가 준 핵심 숫자:

- `partial_reseat` gold-anchor transitions: `n = 266`
- `round6_ndcg_delta = -29.97`
- baseline same-step delta: `+0.21`
- `next_top10_on_region_share = 0.00`
- `next_top10_off_region_share = 1.00`
- `next_prehit_on_region_share = 0.00`
- `next_prehit_off_region_share = 1.00`

이 숫자의 해석은 다음과 같다.

- `t+1` ranking이 `new reseated branch`로 이동한 것은 아니다.
- 하지만 그렇다고 promising gold region 안에 남아 있는 것도 아니다.
- 실제로는 `t`에서 잡고 있던 gold-aligned branch region을 잃어버리고, retrieval이 그 밖으로 완전히 벗어난다.

즉 `partial reseat`의 immediate failure는

- `new branch takeover`

가 아니라

- `gold-aligned promising branch에서의 detachment`
- 그리고 그 detachment가 `old ended subtree` dominance 형태로 관측되는 transition mismatch

로 보는 것이 더 정확하다.

### 3. Reseat vs `freeze_terminal` follow-up

위 두 스크립트는 `t+1`의 immediate failure는 잘 보여준다. 하지만 사용자가 실제로 궁금해한 것은 이것이었다.

- 왜 `partial reseat`는 `t+1`에서 망가졌는데도 `freeze_terminal`보다 end 성능이 더 좋을 수 있는가?

이 질문은 `reseat run`과 `freeze_terminal` run을 같은 query, 같은 iter에 맞춰서 봐야 하므로 별도 paired follow-up으로 확인했다.

비교 방식:

- `round6 ... expandable_ended_reseat`에서 query별 첫 `ended_beam_reseat` iter를 찾는다.
- `round6 ... method2_expandable_pool_freeze_terminal`에서 같은 `(subset, query_idx, iter)`를 대응시킨다.
- `t`, `t+1`, `t+2`, `t+3`, final end에서의 `nDCG@10` advantage를 본다.
- 동시에 새 reseated branch가 `selected_branches_before`, `selected_branches_after`, `pre_hit_paths`, `active_eval_paths`에 언제부터 나타나는지 센다.

핵심 숫자:

- `partial reseat` rows: `n = 530`
- reseat minus freeze advantage:
  - at `t`: `+0.04`
  - at `t+1`: `-1.90`
  - at `t+2`: `+1.60`
  - at `t+3`: `+2.04`
  - end: `+1.75`

그리고 새 reseated branch의 시간축은 이렇다.

At `t+1`:

- new reseated branch inside `selected_branches_before`: `1.40`
- new reseated branch inside `selected_branches_after`: `0.40`
- new reseated docs in `pre_hit top-10`: `0.00`
- new reseated docs in `pre_hit top-100`: `0.00`
- new reseated docs in `active_eval top-10`: `0.00`
- new reseated docs in `active_eval top-100`: `0.00`

At `t+2`:

- new reseated branch inside `selected_branches_before`: `0.40`
- new reseated branch inside `selected_branches_after`: `0.08`
- new reseated docs in `pre_hit top-10`: `0.44`
- new reseated docs in `pre_hit top-100`: `10.85`
- new reseated docs in `active_eval top-10`: `0.44`
- new reseated docs in `active_eval top-100`: `10.87`

이 follow-up이 보여주는 것은 단순하다.

- reseated branch는 `t+1`에 retrieval에 바로 반영되지 않는다.
- 하지만 beam state에는 잠깐 남아 있다.
- 그 상태가 `t+2`부터 실제 retrieval evidence로 나타난다.

그래서 현재 가장 그럴듯한 설명은 다음이다.

- immediate cost: `t+1` detachment / transition mismatch
- later benefit: `t+2` 이후 delayed branch activation

## 지금 문장에서 주장해도 되는 것

현재 증거로 방어 가능한 문장:

- `partial reseat`의 immediate drop은 `new branch takeover`가 아니라, gold-aligned promising branch에서 이탈한 transition mismatch다.
- 그럼에도 `partial reseat`가 `freeze_terminal`보다 나중에 좋아질 수 있는 것은, reseated branch가 beam에 보존되었다가 한 step 뒤부터 retrieval에 기여하기 때문이다.

현재 증거로 과한 문장:

- `partial reseat`가 항상 robust하다.
- `new reseated branch`가 즉시 top-10에 들어와 성능을 올린다.
- delayed activation만으로 모든 query-level variance를 설명할 수 있다.

## 한계

- `t+2` 이후 delayed activation은 현재 가장 잘 맞는 설명이지만, 아직 별도 standalone script로 고정해두지는 않았다.
- win / lose query split까지 완전히 설명하는 것은 아니다. 즉 이건 `best-supported current mechanism`이지 닫힌 인과 증명은 아니다.
- 따라서 이 문서는 최종 theory note가 아니라, 현재 로그와 analysis CSV가 지지하는 가장 보수적인 요약으로 보는 것이 맞다.

## Correction: indirect recovery is real, but not the main gain explanation

위의 `delayed activation` 해석은 이제 더 약하게 써야 한다.  
추가 분석 결과, `non-gold-return first reseat` 이후에 나중 iteration에서 gold branch를 다시 밟는 경우는 분명히 있지만, 그 현상이 평균 gain의 주된 원인처럼 보이지는 않는다.

추가한 분석:

- `scripts/analysis/analyze_round6_reseat_return.py`

실행:

```bash
python scripts/analysis/analyze_round6_reseat_return.py \
    --out_prefix results/BRIGHT/analysis/round6_reseat_return_branchstate_v2
```

여기서 새로 본 지표는 두 개다.

- `future_gold_branch_hit`
    - reseat 이후 미래 어느 iteration에서든 `selected_branches_after`가 gold branch를 다시 밟는 경우
- `strict_gold_reacquire`
    - reseat iter `t`의 `selected_branches_after`에는 gold branch가 없었고, 미래 iteration에서 처음으로 gold branch를 다시 밟는 경우

`first reseat + non-gold-return`만 보면(`n=1227`):

- `selected_after_has_gold_t = 27.1%`
- `future_gold_branch_hit = 28.8%`
- `strict_gold_reacquire = 22.0%`
- `future_first_gold_hit_rel_iter = 2.14`

즉 indirect recovery 자체는 존재한다. 하지만 이것이 평균 gain을 설명하는 중심 메커니즘인지가 중요하다.

이를 위해 `future_gold_branch_hit` 유무로 나누면:

### later gold hit 있음 (`n=353`)

- `ndcg_t = 11.33`
- `final_ndcg = 11.84`
- `max_future_ndcg = 14.70`
- `delta_end_from_t = +0.51`
- `delta_max_future_from_t = +3.37`

### later gold hit 없음 (`n=874`)

- `ndcg_t = 39.29`
- `final_ndcg = 41.64`
- `max_future_ndcg = 51.32`
- `delta_end_from_t = +2.34`
- `delta_max_future_from_t = +12.03`

즉 오히려:

- 나중에 gold branch를 다시 밟는 query가 더 큰 gain을 보이는 것이 아니라
- **나중에 gold branch를 다시 안 밟는 쪽이 평균 gain이 더 크다**

`strict_gold_reacquire`만 따로 봐도 같은 방향이다(`n=270`).

- `delta_end_from_t = +0.38`
- `delta_max_future_from_t = +3.23`

이 값 역시 전체 `non-gold-return first reseat` 평균보다 낮다.

### 수정된 해석

따라서 지금은 이렇게 쓰는 게 맞다.

- 맞는 문장:
    - indirect recovery는 존재한다
    - 일부 query는 reseat 이후 나중 iteration에서 gold branch를 다시 밟는다
- 과한 문장:
    - reseat gain의 주된 원인은 delayed gold-branch reacquisition이다

현재 데이터가 더 직접적으로 지지하는 건 이쪽이다.

- direct gold-return retention은 약하다
- indirect recovery도 존재하지만, 평균 gain의 주원인으로 보이지는 않는다
- 따라서 reseat의 이득은 `gold branch를 다시 밟는 것` 그 자체라기보다, 더 넓은 search-space perturbation / alternate-path search 효과에서 나올 가능성이 크다
