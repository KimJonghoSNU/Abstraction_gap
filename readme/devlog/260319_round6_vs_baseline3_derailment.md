이 문서는 baseline과 round6를 같은 gold-grounded derailment 프레임으로 직접 비교한 결과만 남긴 요약본이다. 핵심은 baseline과 round6의 내부 state가 다르더라도, `gold branch를 한 번 밟은 뒤 다음 retrieval과 최종 retrieval을 얼마나 안정적으로 유지하는가`는 retrieval-level에서 직접 비교할 수 있다는 점이다.

## 비교 목적

보고 싶은 질문은 하나다.

- baseline은 tree constraint가 없어서 gold-aligned state를 밟고도 drift할 수 있다고 했는데, round6는 같은 gold-grounded 기준에서 실제로 더 stable한가?

여기서 주의할 점이 있다.

- baseline은 explicit branch state가 없으므로 anchor를 `rewrite_context_paths`로 잡아야 한다.
- round6는 explicit branch state가 있으므로 anchor를 `selected_branches_after`로 잡는 것이 맞다.

즉 anchor의 source field는 다르다. 대신 anchor 이후의 evaluation은 가능한 한 retrieval-level metric으로 맞췄다.

## 어떤 코드로 분석했는가

분석 스크립트:

- [analyze_round6_vs_baseline3_derailment.py](/data4/jongho/lattice/scripts/analysis/analyze_round6_vs_baseline3_derailment.py)

실행:

```bash
python scripts/analysis/analyze_round6_vs_baseline3_derailment.py
```

생성 산출물:

- [baseline3_first_anchor.csv](/data4/jongho/lattice/results/BRIGHT/analysis/round6_vs_baseline3_derailment/baseline3_first_anchor.csv)
- [round6_first_anchor.csv](/data4/jongho/lattice/results/BRIGHT/analysis/round6_vs_baseline3_derailment/round6_first_anchor.csv)
- [derailment_summary_by_model.csv](/data4/jongho/lattice/results/BRIGHT/analysis/round6_vs_baseline3_derailment/derailment_summary_by_model.csv)
- [derailment_compare_delta.csv](/data4/jongho/lattice/results/BRIGHT/analysis/round6_vs_baseline3_derailment/derailment_compare_delta.csv)

## 어떤 논리로 계산했는가

공통점:

- gold doc를 tree path로 복원한다.
- depth `2, 3, 4`에서 gold branch prefix를 만든다.
- 각 query마다 `first gold-grounded anchor`만 사용한다.
- anchor 뒤의 shared metric은 다음과 같다.
  - next retrieval top-10 drift
  - next retrieval top-100 drift
  - final top-10 miss
  - next on-anchor share

baseline anchor:

- `rewrite_context_paths`를 depth별 branch로 투영
- 그중 gold branch와 겹치는 첫 iter를 anchor로 사용

round6 anchor:

- `selected_branches_after`가 덮는 gold branch region을 depth별로 계산
- 그중 첫 iter를 anchor로 사용

round6에는 보조 지표 하나를 더 둔다.

- `NextPreHitTop100DriftRate|FirstAnchor`
- 이유: round6는 rewrite 전 evidence pool인 `pre_hit_paths`가 명시적으로 저장되기 때문이다.

## 현재 핵심 숫자

headline depth는 `3`으로 두었다.

전체 `overall, depth=3`:

- baseline
  - `AnchorQueryRate = 76.59%`
  - `EndMiss@10 | FirstAnchor = 26.23%`
  - `NextTop10DriftRate | FirstAnchor = 6.04%`
  - `NextTop100DriftRate | FirstAnchor = 0.19%`
  - `MeanNextTop10OnAnchorShare = 0.373`
  - `MeanNextTop100OnAnchorShare = 0.169`
- round6
  - `AnchorQueryRate = 99.93%`
  - `EndMiss@10 | FirstAnchor = 42.15%`
  - `NextTop10DriftRate | FirstAnchor = 28.13%`
  - `NextTop100DriftRate | FirstAnchor = 9.26%`
  - `MeanNextTop10OnAnchorShare = 0.295`
  - `MeanNextTop100OnAnchorShare = 0.144`
  - `NextPreHitTop100DriftRate | FirstAnchor = 9.40%`

round6 minus baseline delta:

- `AnchorQueryRate = +23.34`
- `EndMiss@10 | FirstAnchor = +15.93`
- `NextTop10DriftRate = +22.08`
- `NextTop100DriftRate = +9.07`
- `MeanNextTop10OnAnchorShare = -0.079`
- `MeanNextTop100OnAnchorShare = -0.024`
- `MeanEndMinusAnchorNDCG@10 = -4.81`

## 해석

현재 결과는 아주 명확하다.

1. round6는 gold branch를 더 잘 밟는다.

- depth 3 기준으로 round6의 first-anchor rate는 거의 `100%`다.
- baseline은 `76.59%`다.

2. 그런데 `gold branch를 밟은 뒤 stable exploitation`은 오히려 round6가 더 약하다.

- round6의 final miss가 baseline보다 높다.
- round6의 next top-10 drift와 next top-100 drift도 baseline보다 더 크다.
- next on-anchor share도 round6가 더 낮다.

3. 따라서 현재 run 기준으로는 다음 주장을 하면 안 된다.

- `round6 is more stable after a promising gold-aligned hit`
- `tree constraint directly improves post-anchor stability`

현재 숫자는 이걸 지지하지 않는다.

4. 대신 방어 가능한 문장은 다음이다.

- round6는 branch state 덕분에 gold branch를 훨씬 더 자주 밟는다.
- 하지만 그 state를 다음 retrieval과 최종 retrieval까지 안정적으로 exploitation하는 문제는 여전히 남아 있다.
- 즉 round6의 강점은 `finding / touching the right branch state` 쪽이고, 약점은 `post-anchor stability` 쪽이다.

## subset 수준에서 보이는 패턴

전체 경향은 대부분 subset에서도 유지된다. 즉 round6는 anchor rate는 높지만, anchor 이후 drift와 final miss는 baseline보다 큰 경우가 많다.

이건 해석상 중요하다.

- baseline failure를 설명할 때는 `gold-aligned hit 이후에도 drift가 남는다`가 맞다.
- 하지만 round6를 baseline의 direct antidote로 쓰려면, `hit 이후 stability`가 실제로 더 좋아야 한다.
- 현재 run에서는 그 부분이 성립하지 않는다.

## 결론

지금 비교에서 가장 정확한 문장은 이렇다.

- baseline도 gold branch를 전혀 못 찾는 것은 아니지만, hit 이후 drift 때문에 잘못 끝나는 경우가 있다.
- round6는 gold branch를 훨씬 더 자주 밟는다.
- 그러나 current `round6 ... expandable_ended_reseat` run은 gold-grounded first-anchor 이후의 안정성에서는 baseline보다 낫지 않다.
- 따라서 round6의 장점을 주장하려면 `post-anchor stability`가 아니라, `earlier / more frequent gold-branch contact` 또는 다른 metric으로 메시지를 다시 세워야 한다.
