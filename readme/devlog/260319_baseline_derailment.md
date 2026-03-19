이 문서는 `baseline3_leaf_only_loop`에서 정말로 `derailment`라고 부를 만한 현상이 있는지, 그리고 그 현상을 어떤 기준으로 봐야 하는지를 정리한 요약본이다. 핵심은 기존의 generic off-branch proxy만 보는 대신, `gold-aligned branch를 실제로 밟은 뒤에도 나중에 잘못 끝나는가`를 gold-grounded 기준으로 직접 보는 것이다.

## 질문

보고 싶은 질문은 하나다.

- tree constraint가 없는 baseline에서도, 모델이 한 번은 gold branch에 맞는 rewrite context를 잡았는데, 이후 query drift 때문에 그 상태를 제대로 활용하지 못하고 잘못 끝나는가?

이 질문에 답하려면 다음 두 가지를 분리해야 한다.

- `promising-state hit`: 어떤 iter에서 rewrite context가 gold branch와 겹쳤는가
- `derailment`: 그런 hit가 있었는데도 다음 retrieval이나 이후 trajectory에서 그 상태를 유지하지 못하고 최종 top-10 retrieval이 틀렸는가

## 어떤 코드로 분석했는가

분석 스크립트:

- [analyze_baseline3_derailment.py](/data4/jongho/lattice/scripts/analysis/analyze_baseline3_derailment.py)

실행:

```bash
python scripts/analysis/analyze_baseline3_derailment.py \
    --out_dir results/BRIGHT/analysis/baseline3_derailment
```

생성 산출물:

- [baseline3_derailment_rows.csv](/data4/jongho/lattice/results/BRIGHT/analysis/baseline3_derailment/baseline3_derailment_rows.csv)
- [baseline3_derailment_firsthit.csv](/data4/jongho/lattice/results/BRIGHT/analysis/baseline3_derailment/baseline3_derailment_firsthit.csv)
- [baseline3_derailment_summary.csv](/data4/jongho/lattice/results/BRIGHT/analysis/baseline3_derailment/baseline3_derailment_summary.csv)
- [baseline3_derailment_examples.csv](/data4/jongho/lattice/results/BRIGHT/analysis/baseline3_derailment/baseline3_derailment_examples.csv)

## 어떤 논리로 계산했는가

분석 대상 run:

- `baseline3_leaf_only_loop`
- `agent_executor_v1_icl2`
- `PlTau=5.0`
- `RCT=10`
- `RSC=on`

대상 로그:

- `leaf_iter_records.jsonl`
- `leaf_iter_metrics.jsonl`

### 1. Gold branch를 복원한다

각 subset마다 다음을 사용했다.

- `data/BRIGHT/{subset}/examples.jsonl`의 `gold_ids`
- `trees/BRIGHT/{subset}/tree-{TV}.pkl`

트리에서 leaf document의 `doc_id -> tree path`를 복원한 뒤, 각 query의 gold doc들을 gold path로 바꿨다.

### 2. Gold-grounded anchor를 정의한다

각 query와 iter에서 다음을 본다.

- `rewrite_context_paths`
- 이를 depth `2, 3, 4` branch prefix로 투영한 집합
- 같은 depth에서의 gold branch prefix 집합

둘의 교집합이 비어 있지 않으면, 그 iter를 `promising-state hit`로 본다.

이때 `initial_rewrite`도 포함한다. 즉 `iter=-1`도 유효한 anchor다.

### 3. Anchor 뒤에서 실제 drift가 생기는지 본다

anchor iter `t`가 정해지면, `t+1`에서 아래 세 view를 본다.

- next retrieval top-10: `retrieved_paths[:10]`
- next retrieval top-100: `retrieved_paths[:100]`
- next rewrite context: `rewrite_context_paths[:10]`

그리고 이 path들 중 anchor branch 아래에 남아 있는 비율을 계산한다.

- `next_top10_on_anchor_share`
- `next_top100_on_anchor_share`
- `next_ctx_on_anchor_share`

특히 아래 사건을 drift flag로 둔다.

- `next_top10_anymiss = 1`: next top-10이 anchor branch를 하나도 유지하지 못함
- `next_top100_anymiss = 1`: next top-100이 anchor branch를 하나도 유지하지 못함
- `next_ctx_anymiss = 1`: next rewrite context가 anchor branch를 하나도 유지하지 못함

### 4. 최종 실패와 연결한다

각 query의 마지막 iter에서 `retrieved_doc_ids[:10]`에 gold가 없으면 final miss로 둔다.

- `final_top10_hit`
- `final_top10_miss`

그리고 `first anchor` 기준으로 다음을 집계한다.

- gold-aligned hit를 한 query 비율
- 그런 hit 이후에도 최종 top-10 miss로 끝나는 비율
- hit 직후 next top-10 / next top-100 / next context drift 비율

### 5. 기존 off-branch proxy는 control로만 둔다

이 스크립트는 예전 proxy도 같이 계산한다.

- `proxy_next_top10_off_pct`
- `proxy_next_top100_off_pct`

하지만 headline은 아니다. 이 proxy는 context branch 기준의 generic off-branch 양을 보여줄 뿐이고, gold-grounded drift와 동일한 의미는 아니다.

## 현재 핵심 숫자

headline depth는 `3`으로 두었다.

전체 `overall, depth=3`:

- `AnchorQueryRate = 76.59%`
- `AnchorWithNextCount = 1059`
- `EndMiss@10 | FirstAnchor = 26.23%`
- `HitToSuccess@10 | FirstAnchor = 73.77%`
- `NextTop10DriftRate | FirstAnchor = 6.04%`
- `NextTop100DriftRate | FirstAnchor = 0.19%`
- `NextCtxDriftRate | FirstAnchor = 6.04%`
- `EndMiss@10 | FirstAnchor & NextTop100Drift = 100.00%`
- `MeanNextTop10OnAnchorShare | FirstAnchor = 0.373`
- `MeanNextTop100OnAnchorShare | FirstAnchor = 0.169`
- `MeanProxyNextTop100OffPct | FirstAnchor = 52.14%`

깊이를 바꿔도 패턴은 비슷하다.

- depth 2:
  - `AnchorQueryRate = 86.92%`
  - `NextTop100DriftRate = 0.17%`
  - `EndMiss@10 | FirstAnchor = 35.00%`
- depth 4:
  - `AnchorQueryRate = 71.75%`
  - `NextTop100DriftRate = 0.30%`
  - `EndMiss@10 | FirstAnchor = 21.25%`

즉 depth를 어떻게 잡아도, `full next-top100 abandonment` 자체는 드물다. 하지만 `gold-aligned hit 이후에도 최종 실패`는 결코 드물지 않다.

## 해석

현재 데이터가 지지하는 결론은 다음과 같다.

1. baseline은 gold-aligned rewrite state를 전혀 못 잡는 시스템이 아니다.

- depth 3 기준으로 query의 `76.59%`는 적어도 한 번은 gold branch와 겹치는 rewrite context를 만든다.

2. 그런데 그런 hit가 있어도 최종 성공이 보장되지는 않는다.

- depth 3 기준으로 first anchor를 가진 query의 `26.23%`는 최종 top-10 retrieval에서 gold를 놓친다.

3. 다만 강한 형태의 `top-100 전체 detachment`는 흔하지 않다.

- depth 3 기준 `NextTop100DriftRate`는 `0.19%`다.
- 즉 `gold branch를 한 번 잡은 뒤 바로 next top-100 전체가 그 branch를 완전히 떠난다`는 가장 강한 버전의 derailment는 존재하긴 하지만 sparse하다.

4. 더 흔한 failure mode는 `top-10/context drift + later final miss`에 가깝다.

- depth 3 기준 `NextTop10DriftRate = 6.04%`
- depth 3 기준 `NextCtxDriftRate = 6.04%`
- 즉 next step에서 anchor branch가 retrieval top-10과 rewrite context에서 끊기는 경우는 full top-100 abandonment보다 훨씬 자주 보인다.

5. 예전 proxy는 현상을 과장해 보이게 만들 수 있다.

- depth 3 기준 `MeanProxyNextTop100OffPct = 52.14%`지만,
- gold-grounded `NextTop100DriftRate`는 `0.19%`다.

이 차이는 중요하다.

- generic off-branch mass가 많다는 사실만으로 `gold branch를 완전히 잃었다`고 말하면 과장이다.
- baseline failure를 설득력 있게 말하려면, gold-grounded event와 final miss를 같이 묶어야 한다.

## 가장 강한 예시

대표적인 `hit -> drift -> final miss` 예시는 [baseline3_derailment_examples.csv](/data4/jongho/lattice/results/BRIGHT/analysis/baseline3_derailment/baseline3_derailment_examples.csv)에 따로 저장했다.

현재 상위 예시 두 개는 아래와 같다.

- `theoremqa_questions`, `query_idx=128`, `iter_anchor=1`
  - anchor 시점 `nDCG@10 = 31.55`
  - next `nDCG@10 = 0`
  - final `nDCG@10 = 0`
  - next top-100 on-anchor share `= 0`
- `aops`, `query_idx=19`, `iter_anchor=-1`
  - anchor 시점 `nDCG@10 = 30.66`
  - next `nDCG@10 = 0`
  - final `nDCG@10 = 0`
  - next top-100 on-anchor share `= 0`

이런 케이스는 baseline에 실제로 `promising state를 밟았지만 끝까지 유지하지 못한 trajectory`가 존재한다는 qualitative evidence다.

## 지금 주장해도 되는 말

현재 증거로 방어 가능한 문장:

- baseline은 gold-aligned branch를 전혀 못 찾는 것이 아니라, 한 번 찾고도 최종 retrieval을 안정적으로 유지하지 못하는 경우가 있다.
- tree constraint가 없는 baseline의 failure는 `항상 full top-100 abandonment` 형태는 아니고, 더 흔하게는 `next top-10 / rewrite-context drift 이후 final miss` 형태로 나타난다.
- 따라서 baseline derailment를 논문에서 말할 때는 `promising-state hit does not guarantee stable exploitation`라고 쓰는 것이 정확하다.

현재 증거로 과한 문장:

- baseline은 gold branch를 자주 완전히 잃어버린다.
- baseline failure의 대부분은 immediate full top-100 detachment다.
- off-branch proxy가 크다는 사실만으로 gold-grounded derailment가 증명된다.

## 결론

지금 baseline 쪽에서 가장 정확한 서사는 이렇다.

- baseline도 중간에는 gold branch에 맞는 rewrite state를 자주 만든다.
- 하지만 retrieval pool constraint가 없기 때문에, 그 state가 이후 retrieval top-10과 rewrite context에서 안정적으로 유지된다고 볼 수는 없다.
- 가장 강한 형태의 `full top-100 derailment`는 드물지만 실제로 존재하며, 그런 경우는 최종 실패와 강하게 연결된다.
- 전체적으로 더 흔한 문제는 `milder drift after a promising hit`, 즉 promising gold-aligned state를 밟고도 다음 retrieval/context에서 그 상태를 충분히 보존하지 못해 잘못 끝나는 것이다.
