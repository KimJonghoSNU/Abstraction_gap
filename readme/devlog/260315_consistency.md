# 2026-03-15 Round5 next retrieval pool 정리

## 질문

`src/bash/round5/run_round5.sh`를 실행했을 때, `src/run_round5.py`에서 다음 retrieval pool과 next branch가 어떻게 정해지는지 정리한다.

특히 아래를 명확히 구분한다.

- `meanscore_global`에서 next branch를 어떻게 고르는지
- 선택된 beam 아래로만 계속 내려가는지
- 어떤 branch의 자식이 전부 leaf일 때 다음 iter에서 retrieval candidate pool이 어떻게 되는지
- 이전 라운드에서 선택되지 못한 branch가 나중에 다시 선택될 수 있는지

---

## 실행 기본값

`src/bash/round5/run_round5.sh` 기본 실행은 아래와 같다.

- `--round5_selector_mode meanscore_global`
- `--max_beam_size 10`
- `--round5_mrr_pool_k 100`
- `--disable_calibration`

즉 기본 round5는 calibration 없이 retriever score 기반으로 branch selector를 돌린다.

---

## 먼저 구분해야 하는 3가지 pool

Round5에서 "retrieval pool"이라고 부를 수 있는 대상이 실제로는 3종류다.

### 1. Traversal slate

현재 beam endpoint 하나마다 그 노드의 직계 자식들로 slate를 만든다.

- branch node에 있으면 그 branch의 자식들이 slate
- 자식이 전부 leaf면 그 leaf들이 slate

즉 beam slot 단위의 immediate candidate set이다.

### 2. Selector local pool

`meanscore_global` / `maxscore_global` / `max_hit_global` branch selector가 쓰는 local leaf pool이다.

- 현재 iteration 시작 시점 beam인 `selected_before` 아래의 leaf들을 모두 모은다
- 현재 beam이 비어 있으면 전체 leaf를 사용한다

이 pool에서 query로 top-`round5_mrr_pool_k` leaf retrieval을 수행한 뒤, 그 hit들이 각 child branch에 얼마나 잘 걸리는지를 보고 다음 branch를 고른다.

### 3. Cumulative pool

rewrite/eval retrieval에 쓰는 누적 leaf pool이다.

- 이전 iteration들에서 한 번이라도 도달한 predicted leaf들의 합집합
- 아직 도달한 leaf가 하나도 없으면 전체 leaf 사용

중요한 점은 이 pool이 branch selector용 hard gate가 아니라, rewrite context와 eval retrieval용 누적 memory라는 점이다.

---

## Iteration 한 번에서 next branch가 정해지는 순서

기본 selector가 `meanscore_global`일 때 per-sample 순서는 아래와 같다.

1. `cumulative_pool`에서 pre-rewrite retrieval을 수행해 rewrite context를 만든다.
2. rewrite로 `query_post`를 만든다.
3. 다시 `cumulative_pool`에서 eval retrieval을 수행한다.
4. 현재 beam으로 traversal slate를 만들고, retriever가 각 slate child를 scoring한다.
5. `sample.update(...)`가 먼저 baseline beam을 만든다.
6. 그다음 `meanscore_global` selector가 현재 beam 이전 상태인 `selected_before`를 기준으로 child branch 후보를 다시 점수화해서 beam을 override한다.

즉 구현은 "selector만 있는 구조"가 아니라:

- baseline retriever-slate beam update
- 그 위에 selector override

이 두 단계가 겹쳐 있다.

---

## `meanscore_global`가 next branch를 고르는 방식

현재 beam path들을 `selected_before`라고 하자.

그러면 후보 branch는 현재 beam의 "직계 child branch"들만이다.

- `selected_before`가 비어 있으면 root의 child branch들
- 비어 있지 않으면 각 selected branch의 direct non-leaf child들

그 다음:

1. `selected_before`의 subtree 아래 leaf 전체를 `selector_local_pool`로 모은다.
2. 이 leaf pool에서 query로 top-100 retrieval을 한다.
3. 각 child branch 후보에 대해, top-100 hit 중 그 branch subtree에 속하는 leaf hit들만 모은다.
4. `meanscore_global`이면 그 hit score 평균을 branch score로 쓴다.
5. branch score 상위 `max_beam_size`개를 다음 beam으로 선택한다.

정리하면, selector 관점에서는 "현재 선택된 beam의 한 단계 아래 child branch들을 global top-B로 재정렬"하는 구조다.

---

## Depth 예시

설명을 위해 아래 트리를 두자.

- Depth 1: `D`
- Depth 2: `D1`, `D2`, `D3`
- Depth 3:
    - `D1` 밑: `D11`, `D12`, `D13`
    - `D2` 밑: `D21`, `D22`
    - `D3` 밑: `D31`

여기서는 이해를 쉽게 하려고 beam size를 2라고 가정한다.

### Iter 0

처음에는 root만 있으므로 `selected_before=[]`에 가깝다.

- candidate child branches = root 아래 top-level branch
- visible tree 기준으로 보면 사실상 `D`

그래서 첫 branch 진입은 `D` 쪽으로 시작한다고 보면 된다.

### Iter 1

현재 beam이 `[D]`라고 하자.

- `selector_local_pool` = `D` 아래 모든 leaf
- candidate child branches = `[D1, D2, D3]`

예를 들어 local retrieval top hit score가 아래처럼 모였다고 하자.

- `D1` subtree leaf hit: `0.91, 0.85`
- `D2` subtree leaf hit: `0.80, 0.79`
- `D3` subtree leaf hit: `0.40`

그러면 `meanscore_global` 평균은:

- `D1`: `0.88`
- `D2`: `0.795`
- `D3`: `0.40`

따라서 다음 beam은 `[D1, D2]`가 된다.

### Iter 2

현재 beam이 `[D1, D2]`라고 하자.

- `selector_local_pool` = `D1` 아래 leaf + `D2` 아래 leaf
- candidate child branches = `[D11, D12, D13, D21, D22]`

중요한 점:

- `D3` subtree leaf는 이제 selector local pool에서 빠진다
- selector는 현재 active beam 아래로 한 단계 더 내려가는 child branch만 본다

예를 들어 mean score가 아래와 같다면:

- `D11`: `0.62`
- `D12`: `0.89`
- `D13`: `0.31`
- `D21`: `0.84`
- `D22`: `0.55`

다음 beam은 `[D12, D21]`가 된다.

---

## 선택된 beam 밑으로만 계속 들어가는가?

selector만 놓고 보면 거의 그렇다.

- 현재 beam의 direct child branch만 후보로 본다
- 현재 beam subtree 아래 leaf만 local pool로 본다

그래서 selector 자체는 "selected beam 아래로 한 단계 더 들어가는 구조"다.

하지만 구현 전체는 hard gate는 아니다.

이유:

- `sample.update(...)`가 먼저 baseline beam을 만든다
- selector가 beam을 다 채우지 못하면 baseline beam으로 빈 자리를 fill한다
- baseline update는 prediction tree 전체에서 이미 생성된 expandable path들을 다시 정렬한다

즉 의도는 subtree descent에 가깝지만, 실제 구현은 baseline beam continuity가 같이 살아 있는 soft gate 구조다.

---

## 어떤 branch의 자식이 전부 leaf일 때

예를 들어 현재 beam 중 하나가 `D1`이고,

- `D1`의 자식이 `L11`, `L12`, `L13`
- 셋 다 leaf

라고 하자.

이 경우를 selector candidate와 retrieval candidate를 구분해서 봐야 한다.

### 이번 iteration의 traversal slate

`D1` slot의 slate는 바로 `[L11, L12, L13]`이다.

즉 이 iteration에서 retriever가 직접 scoring하는 candidate는 `D1`의 leaf 자식들이다.

### scoring 이후

`sample.update(...)`가 `D1` 아래에 `L11`, `L12`, `L13` prediction node를 만든다.

하지만 leaf는 expandable하지 않다.

그래서 다음 beam을 만들 때:

- `D1`은 이미 확장 끝
- `L11`, `L12`, `L13`은 leaf라서 expandable 아님

결국 `D1` slot은 다음 traversal beam에서 사라진다.

### 그 다음 iteration의 retrieval candidate pool

여기서 또 2개를 나눠야 한다.

#### A. 다음 branch selector용 pool

`D1`은 더 이상 non-leaf child candidate를 내지 못한다.

즉 `D1` 쪽은 다음 selector branch 후보를 만들 수 없다.

다만 방금 scoring된 `L11`, `L12`, `L13` 자체는 leaf hit로 남아 있기 때문에, 그 iteration에서의 local evidence로는 사용된다.

#### B. 다음 rewrite/eval용 cumulative pool

`L11`, `L12`, `L13`은 predicted leaf로 남는다.

그래서 다음 iteration 시작 시 `cumulative_pool`에 추가된다.

즉 leaf branch는 traversal frontier에서는 끝나지만, 거기서 얻은 leaf들은 이후 rewrite/eval retrieval memory에는 계속 남는다.

짧게 요약하면:

- leaf-only branch에 들어가면 그 iteration의 slate는 그 leaf들이다
- 한 번 scoring하고 나면 그 branch는 더 이상 traversal beam에 남지 않는다
- 하지만 그 leaf들은 `cumulative_pool`에 축적된다

---

## 이전 라운드에서 선택받지 못한 branch가 다음 라운드에서 다시 선택될 수 있는가?

selector만 보면 직접적으로는 거의 아니다.

- selector는 현재 beam의 child들만 후보로 보기 때문이다

하지만 전체 구현에서는 가능성이 남아 있다.

이유:

- baseline `sample.update(...)`는 prediction tree 전체에서 이미 생성된 expandable path를 다시 정렬한다
- selector가 full beam을 못 채우면 baseline beam path를 fallback으로 다시 채운다

따라서 어떤 branch가 예전에 생성되었다가 beam에서 밀려난 경우:

- selector가 직접 다시 고르지 않더라도
- baseline/fallback 경로를 통해 다음 iteration beam에 재등장할 수 있다

즉 "탈락 branch 부활 가능성"은 닫혀 있지 않다.

---

## 핵심 요약

- `meanscore_global`는 현재 beam의 direct child branch만 후보로 보고, 현재 beam subtree 아래 leaf retrieval hit의 평균 점수로 다음 branch를 고른다.
- selector만 보면 selected beam 아래로만 내려가는 구조다.
- 하지만 실제 구현은 baseline retriever-slate beam update와 fallback이 살아 있어서 완전한 hard gate는 아니다.
- 어떤 branch의 자식이 전부 leaf면, 그 iteration에서는 그 leaf들이 slate가 된다.
- 그 branch는 이후 traversal beam에서는 끝나지만, 그 leaf들은 `cumulative_pool`에 남아서 이후 rewrite/eval retrieval context에 계속 쓰인다.
- 이전 라운드에서 탈락한 branch도, 이미 prediction tree에 생성되어 있었다면 baseline/fallback을 통해 다시 beam에 들어올 수 있다.

---

## 2026-03-15 Round5 fused-memory consistency mode

### 목적

이번 구현의 목적은 `round5`에서 **무엇을 좋은 retrieval로 간주하는지**, 그리고 **다음 iteration rewrite / branch selection이 어떤 evidence를 보고 움직이는지**를 더 일관되게 맞추는 것이다.

기존 `round5`는 아래 세 축이 서로 어긋나 있었다.

- 공식 metric `nDCG@10`: 현재 iteration의 local retrieval 결과
- 다음 rewrite evidence: 현재 `query_pre`로 `cumulative pool`에서 다시 뽑은 local pre-hit
- score-based branch selector: 현재 iteration의 local retrieval hit

즉, 현재 step에서 잘 나온 retrieval을 다음 step이 직접 재사용한다기보다, 매번 local retrieval을 새로 뽑아 rewrite / selector / metric이 서로 다른 기준을 볼 수 있었다.

이번 구현은 `round6 method2`에서 쓰던 `all_iters bank + RRF fusion` 아이디어를 `round5`에 가져와, 특정 실행 모드에서는 다음 세 가지를 같은 memory 축에 맞추는 것을 목표로 한다.

- 공식 `nDCG@10`
- 다음 iteration rewrite evidence
- score-based branch selector input

이렇게 해야 “좋다고 기록한 retrieval”과 “실제로 다음 step decision에 사용한 retrieval” 사이의 consistency를 볼 수 있다.

### 추가한 실행 모드

새 argument:

- `--round5_fused_memory`

이 flag가 켜지면:

- 공식 `nDCG@10`은 current-run local retrieval이 아니라, **completed retrieval bank 전체를 RRF fuse한 결과**가 된다.
- local current-run metric은 `nDCG@10_iter`로 따로 저장한다.
- 다음 iteration rewrite evidence는 현재 local pre-hit이 아니라, **이전 iteration들에서 누적된 bank를 RRF fuse한 top-10 docs**를 사용한다.
- score-based branch selector도 현재 local retrieval 대신, **previous-bank-only fused ranking**을 사용한다.

이 모드는 아래 selector에서만 허용한다.

- `maxscore_global`
- `meanscore_global`
- `max_hit_global`

`retriever_slate`에서는 fail-fast로 막았다. `retriever_slate`까지 fused ranking을 반영하려면 beam update 자체를 다시 정의해야 해서, 현재 patch 범위를 넘어간다고 판단했다.

### 핵심 구현 규칙

#### 1. rewrite evidence

iteration `t`의 rewrite prep에서:

- 항상 기존 local pre-hit retrieval은 수행한다. 이건 diagnostics용으로 남긴다.
- `--round5_fused_memory`가 켜져 있고, previous bank가 비어 있지 않으면:
    - previous bank를 `RRF`로 fuse한다.
    - fused ranking top-10 docs를 `leaf_descs`로 넣는다.
- bank가 비어 있으면 local pre-hit으로 fallback한다.

즉, iter 0은 항상 기존 local path로 시작하고, iter 1부터 fused-memory evidence가 들어간다.

#### 2. selector timing

selector는 **현재 iteration에서 방금 나온 retrieval run을 바로 쓰지 않는다.**

- iter `t` selector는 iter `< t` bank만 본다.
- iter `t` current run은 branch selection이 끝난 뒤 bank에 append된다.
- 따라서 selector가 same-iter current retrieval까지 먹는 구조는 아니다.

이 timing은 `round6 method2`와 맞춘 것이다.

#### 3. 공식 metric

iteration `t`의 공식 metric은:

- current run을 append 후보로 만든 뒤,
- `bank(0..t)`를 fuse해서 계산한다.

즉:

- `nDCG@10`: fused official metric
- `nDCG@10_iter`: current-run local metric

iter 0에서는 bank에 current run 하나만 있으므로 두 값이 같아진다.

### 남긴 로그 / 분석용 필드

`iter_records`에는 아래를 추가로 남긴다.

- `pre_hit_source`
- `pre_hit_local_paths`, `pre_hit_local_doc_ids`
- `active_eval_paths`, `active_eval_doc_ids`
- `current_run_metrics`
- `active_eval_metrics`
- `fusion_mode_active`
- `fusion_bank_size`
- `fusion_bank_source_iters`
- `fusion_bank_runs`
- `selector_source`

의도는 clear하다.

- `local_*`: 이번 iteration local retrieval
- `active_eval_*`: fused official ranking
- `pre_hit_*`: rewrite가 실제로 본 evidence vs local diagnostic evidence

즉 나중에 post-hoc 분석할 때,

- rewrite는 무엇을 봤는지
- selector는 무엇을 봤는지
- metric은 무엇을 기준으로 계산됐는지

를 분리해서 볼 수 있게 했다.

### 구현 파일

- `src/run_round5.py`
    - fused bank helper 추가
    - rewrite evidence source 변경
    - selector source 변경
    - official/local metric 이원화
    - iter_records 확장
- `src/hyperparams.py`
    - `--round5_fused_memory` 추가
- `src/bash/round5/run_round5.sh`
    - `ROUND5_FUSED_MEMORY=1`로 실행할 수 있게 launcher 연결

### 현재 해석

이 모드는 성능 개선 자체보다 먼저, **trajectory를 만드는 memory와 metric memory를 맞춘 consistency 실험 모드**로 보는 게 맞다.

즉 질문은 이런 것이다.

- local-only로 매번 새 retrieval을 보는 것이 좋은가?
- 아니면 이전 retrieval history를 fuse한 memory를 rewrite / selector / metric이 같이 보게 하는 것이 더 안정적인가?

이번 patch는 그 비교를 가능하게 만들기 위한 구현이다.
