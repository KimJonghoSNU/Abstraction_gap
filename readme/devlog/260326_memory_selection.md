# 2026-03-26 Memory Selection

## Score-only run: do unsupported hypotheses disappear by themselves?

분석 대상:

- `round6_mrr_selector_accum_meanscore_global_expandable_ended_reseat_frontiercum_qstate`
- `RMP=reason-embed-qwen3-8b-0928`
- score reseat only (`RERP=random` 제외)

질문:

- 현재 iteration의 hypothesis들 중 일부가 current top-10 retrieved docs와 잘 맞지 않으면,
  다음 iteration에서 자동으로 drop되거나 다른 방향으로 rewrite되는가?

분석 단위:

- hypothesis = `possible_answer_docs` slot (`Theory`, `Entity`, `Example`, `(Other)`)
- support evidence = 현재 iteration `active_eval_paths[:10]`의 leaf `desc`
- support score = reason-embed max cosine(slot, current top-10 docs)
- 다음 iteration fate = 같은 slot 또는 다른 slot으로의 best semantic match를 보고
  - `retained`
  - `rewritten`
  - `dropped`
  로 분류

실행:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/analysis/analyze_round6_memory_support_drift.py \
    --query_max_length 128 \
    --max_queries_per_subset 20 \
    --sample_seed 0 \
    --out_prefix results/BRIGHT/analysis/round6_memory_support_drift_sample20
```

결과 파일:

- `results/BRIGHT/analysis/round6_memory_support_drift_sample20_slot_rows.csv`
- `results/BRIGHT/analysis/round6_memory_support_drift_sample20_subset_iter_summary.csv`
- `results/BRIGHT/analysis/round6_memory_support_drift_sample20_overall_iter_summary.csv`
- `results/BRIGHT/analysis/round6_memory_support_drift_sample20_examples.csv`

중요한 구현 선택:

- hypothesis 분해는 `query_post` string diff 대신 `possible_answer_docs`를 썼다.
- doc text는 raw corpus가 아니라 retrieved leaf `desc`를 썼다.
- doc embedding은 `trees/BRIGHT/<subset>/node_embs.reasonembed8b.npy`의 precomputed leaf embedding을 그대로 재사용했다.
- full run은 shared GPU 상황에서 너무 비싸서, 현재 결과는 subset당 `20` query sample 기준이다.

핵심 결과:

- 9개 subset, subset당 20 query sample, 총 `6709` slot transition
- support quartile:
  - `q25 = 0.8213`
  - `q50 = 0.8605`
  - `q75 = 0.8864`
- 전체 transition:
  - `retained = 81.3%`
  - `rewritten = 18.7%`
  - `dropped = 0.0%`

가장 중요한 건 lowest-support quartile이다.

- lowest-support quartile:
  - `retained = 78.8%`
  - `rewritten = 21.2%`
  - `dropped = 0.0%`
- lowest-support quartile에서
  - `mean best-next-slot similarity = 0.976`
  - 즉 low-support hypothesis도 대부분 다음 iteration에 매우 비슷한 형태로 살아남는다

iteration별로 보면 rewrite는 초반에 더 많다.

- iter 0:
  - lowest-support slot `rewrite rate = 39.0%`
  - `retain rate = 61.0%`
- iter 1:
  - `rewrite rate = 31.1%`
  - `retain rate = 68.9%`
- iter 2 이후:
  - `rewrite rate = 14% ~ 25%`
  - `retain rate = 75% ~ 86%`

즉 초반에는 일부 수정이 일어나지만, 중후반에는 low-support slot도 대부분 유지된다.

rewrite가 전혀 의미 없는 것은 아니다.

- lowest-support + rewritten:
  - `mean support delta = +0.0164`
  - `56.2%`가 next-step support가 증가
  - `16.0%`는 `+0.05` 이상 의미 있게 증가
- lowest-support + retained:
  - `mean support delta = +0.0016`

즉 low-support hypothesis가 바뀔 때는 support가 좋아지는 경우가 꽤 있다.
하지만 그 비율보다 더 큰 사실은, low-support hypothesis 대부분이 애초에 그대로 유지된다는 점이다.

해석:

- 현재 score-only frontiercum_qstate loop는
  - retrieval support가 약한 hypothesis를 자동으로 drop하는 구조가 아니다.
- 더 정확히 말하면,
  - 일부 early iteration rewrite는 약한 hypothesis를 더 supported한 형태로 바꾸지만,
  - 전체적으로는 weak slot이 다음 iteration에 거의 그대로 carry-over된다.

결론:

- `remove items with no retrieval support`는 현재 loop에서 자연스럽게 emergent하게 일어나는 현상이 아니다.
- 이건 별도 memory-selection 또는 slot-filter rule로 넣어야 한다.

## Question

`run_round6_hypbank.py`를 만들고 나서 다음 질문이 생긴다.

- memory에 무엇을 남길지 결정할 때 `support`, `coverage`, `diversity`, `redundancy`, `failure`를 다 따로 둘 필요가 있는가?
- 아니면 더 단순한 objective로 정리할 수 있는가?
- 현재 구현한 `hypbank` variant는 여전히 의미 있는 중간 단계인가?

## Current judgment

결론부터 말하면, 지금 objective를 너무 많이 두는 것은 좋지 않다.

이유는 두 가지다.

1. 의미가 겹친다.
    - `coverage`, `diversity`, `redundancy`는 사실 모두 "이미 고른 memory set에 비해 새로운 정보를 얼마나 더 주는가"로 묶일 수 있다.
2. `failure`는 점수항이라기보다 제약에 가깝다.
    - 특히 state-local failure는 현재 state에서 이미 약했다고 본 evidence이므로, reward에 부드럽게 섞기보다 hard filter 또는 strong downweight로 다루는 편이 해석이 쉽다.

따라서 다음 단계에서는 objective를 아래처럼 줄이는 것이 맞다.

## Simplified view

memory selection은 두 단계로 나누는 것이 좋다.

### 1. Hard constraints / filters

이 단계에서는 아예 후보를 제거한다.

- current state에서 반복 실패한 hypothesis
- token budget을 초과하는 경우
- support가 너무 낮아 현재 retrieval pool에서 아무 evidence도 못 끌어오는 hypothesis

즉 `failure`는 우선 score term이 아니라 filter로 본다.

### 2. One utility + one marginal gain

남은 후보들 사이에서는 두 가지만 본다.

- `support`
    - 현재 retrieval pool에서 이 item이 실제 evidence를 끌어오는 정도
- `marginal gain`
    - 이미 선택한 memory set에 비해 이 item이 얼마나 새로운 정보를 더 주는가

여기서 `marginal gain` 안에 사실상 아래가 같이 들어간다.

- `coverage`
- `diversity`
- `redundancy`

즉 이 셋을 따로 쓰지 않고, "지금 이미 고른 집합에 비해 얼마나 새롭고 덜 중복되는가"로 묶는 게 더 깔끔하다.

## Recommended objective

현재 단계에서는 아래 정도가 가장 단순하고 해석 가능하다.

- filter:
    - remove state-local failed items
    - remove items with no retrieval support
- greedy selection:
    - first item: highest `support`
    - next items: highest `support + \lambda * marginal_gain`

여기서 `marginal_gain`은 처음부터 복잡하게 하지 말고, 아래 중 하나만 먼저 쓰는 것이 낫다.

1. newly covered supporting docs 수
2. selected hypothesis들과의 embedding dissimilarity
3. slot novelty (`Theory`, `Entity`, `Example`, `Other`)

내 추천 순서는:

1. `new supporting docs`
2. `embedding dissimilarity`
3. `slot novelty`

이유는 현재 system의 목적이 answer-supporting evidence를 더 넓게 유지하는 것이기 때문이다.

## Is current `run_round6_hypbank.py` still meaningful?

의미 있다. 다만 최종 해법이 아니라, 좋은 stage-1 baseline이다.

현재 구현이 검증하는 것은 명확하다.

- reseat에서 전체 rewrite history를 그냥 이어붙이는 것이 문제인가?
- 그렇다면 current retrieval pool에서 support되는 hypothesis만 남기는 것만으로도 도움이 되는가?
- state-local failure memory를 붙이면 같은 state에서 약했던 방향을 피하는 데 도움이 되는가?

즉 현재 구현은 다음 가설을 검증하는 실험이다.

- `stale hypothesis accumulation hurts reseat`

이건 portfolio/diversity를 넣기 전에도 충분히 의미 있다.

왜냐하면 이 실험이 실패하면, 더 복잡한 submodular/bandit memory로 가더라도 gain이 작을 가능성이 높기 때문이다.
반대로 이 실험이 성공하면, 그 다음 단계로 `support-only`를 `support + marginal_gain`으로 확장하는 명확한 이유가 생긴다.

## What the current implementation is actually doing

현재 `src/run_round6_hypbank.py`는 reseat step에서만 다음을 한다.

1. global hypothesis bank에서 후보를 모은다.
2. current state에서 `failed`였던 hypothesis는 우선 제외한다.
3. 각 hypothesis를 단독 query로 현재 retrieval pool에 던져 support를 계산한다.
4. support가 있는 hypothesis top-`3`만 남긴다.
5. failed hypothesis top-`3`는 prompt의 `Avoid Repeating These Weak Directions`에 넣는다.
6. 새 rewrite가 나오면, 기존 active hypothesis와 새 hypothesis를 다시 같이 score해서 top-`3`만 다음 query state로 남긴다.

현재 고정값:

- active keep = top-`3`
- support scoring hits = top-`5`
- failed hint = top-`3`

즉 지금은 `support-only pruning + local failure memory`다.
아직 `portfolio selection`은 아니다.

## Staged roadmap

### Stage 0: current hypbank

목적:

- stale suffix accumulation이 진짜 문제인지 확인

현재 상태:

- 구현 완료

### Stage 1: add marginal gain to hypothesis selection

바꿀 것:

- top-`3`를 support-only로 고르지 않고
- greedy `support + marginal_gain`으로 고르기

최소 구현:

- `marginal_gain = newly covered supporting docs`

이 단계에서 기대하는 것:

- 서로 비슷한 hypothesis 3개가 동시에 살아남는 현상 감소
- retrieval pool 아래 gold docs의 유지력 증가

### Stage 2: apply the same logic to document memory

지금은 hypothesis query state만 관리한다.
다음에는 EMR-style document memory에도 같은 selection logic을 넣을 수 있다.

바꿀 것:

- memory of documents도 support-only 누적이 아니라
- prompt budget 안에서 portfolio로 packing

이 단계에서 기대하는 것:

- hypothesis와 document memory가 같은 선택 원리로 정렬됨

### Stage 3: contextual bandit for long-term retention

bandit은 이 단계에서 넣는 것이 맞다.

arm의 예:

- hypothesis cluster
- state-local memory bucket
- document memory bucket

reward의 예:

- next-step `pool_has_any`
- next-step `Recall@100`
- next-step `nDCG` delta

이 단계의 목적:

- 지금은 약하지만 나중에 다시 useful한 memory를 보존할지 학습

## Recommendation

다음 구현은 `bandit`보다 `Stage 1`이 우선이다.

이유:

- 지금 문제는 long-term exploration보다, prompt 안에 무엇을 함께 남길지의 문제에 더 가깝다.
- 이건 bandit보다 submodular-style greedy selection으로 푸는 것이 더 직접적이다.
- bandit은 reward가 delayed/noisy해서 logging이 더 쌓인 뒤에 들어가는 편이 맞다.

## Reference

- Petcu et al., "Query Decomposition for RAG: Balancing Exploration-Exploitation"  
  https://aclanthology.org/2026.eacl-long.322.pdf

이 논문은 sub-query selection을 budgeted sequential decision으로 본다. 우리 문제는 정확히 같은 setting은 아니지만,
"무엇을 유지할지"를 portfolio selection으로 본다는 점에서 bandit intuition은 유효하다. 다만 현재 단계에서 더 직접적인 formalism은 `submodular-style memory selection`이다.
