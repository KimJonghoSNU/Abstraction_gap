# 2026-03-15 Round5 method summary

## 이 문서에서 보는 것

헷갈리지 않게 아래 3개만 본다.

- `selected_before`: iteration 시작 시점의 active beam endpoint path
- `expandable_paths`: prediction tree 안에서 이미 생성되었고 아직 expand되지 않은 non-leaf path 전체
- `retrieval_pool`: branch selector가 쓰는 local leaf pool. `selected_before` subtree 아래 leaf들의 합집합

중요:

- 이 문서에서 `retrieval_pool`은 `selector local pool`을 뜻한다
- `traversal slate`는 별개다. 이건 현재 beam endpoint의 직계 자식들이다

즉:

- `retrieval_pool` = selector가 branch score를 만들 때 leaf retrieval 하는 대상
- `traversal slate` = 이번 iter에 beam slot에서 직접 scoring하는 immediate child 후보

---

## 코드 기준 짧은 정의

- `selected_before`: `beam_state_paths`의 마지막 path를 읽어서 만듦
- `expandable_paths`: `get_all_expandable_paths()`가 모으는 "미확장 non-leaf path"
- `retrieval_pool`: `_collect_leaf_pool(selected_before, ...)`

따라서 `selected_before`와 `expandable_paths`는 일반적으로 다르다.

- `selected_before`는 현재 beam에 실제로 올라온 path만 본다
- `expandable_paths`는 beam에 없더라도 prediction tree에 이미 생성되어 있고 아직 안 펼친 branch면 포함한다

이 차이 때문에 baseline beam이 예전 branch를 다시 살릴 수 있다.

---

## 예시 트리

hidden root `R`를 포함한 depth 4 tree를 두자. beam size는 2라고 가정한다.

```text
Depth 0: R
Depth 1: D1, D2
Depth 2: D11, D12, D21, D22
Depth 3: D121, D122, D211, D212
Depth 4: leaf docs
```

조금 더 구체적으로 쓰면:

```text
R
|- D1
|  |- D11
|  `- D12
|     |- D121
|     `- D122
`- D2
   |- D21
   |  |- D211
   |  `- D212
   `- D22
```

그리고 마지막 leaf docs는 이렇게 두겠다.

```text
D121 -> l121a, l121b
D122 -> l122a, l122b
D211 -> l211a, l211b
D212 -> l212a, l212b
```

`D11`, `D22`도 같은 식으로 더 내려간다고 생각하면 된다.

---

## Iter 0 시작

- `selected_before = []`
- `expandable_paths = [R]`
- `retrieval_pool = 전체 leaf docs`

이유:

- 시작할 때 beam은 root만 갖고 있다
- root path `()`는 `selected_before`에 넣지 않으므로 빈 리스트가 된다
- `selected_before`가 비면 `_collect_leaf_pool(...)`이 전체 leaf fallback을 쓴다

이번 iter에서 실제로 보는 child:

- `traversal slate`: root child 기준으로 `D1`, `D2`

가정:

- Iter 0 끝에서 beam이 `[D1, D2]`가 되었다고 하자

---

## Iter 1 시작

- `selected_before = [D1, D2]`
- `expandable_paths = [D1, D2]`
- `retrieval_pool = D1 subtree leaf docs + D2 subtree leaf docs`

이 시점에는 사실상 tree 전체 leaf docs가 retrieval pool이다.

이번 iter에서 실제로 보는 child:

- `D1` slot traversal slate = `[D11, D12]`
- `D2` slot traversal slate = `[D21, D22]`

중요:

- `retrieval_pool`은 `[D11, D12, D21, D22]`가 아니다
- selector는 저 branch들의 점수를 만들기 위해, 그 아래 leaf docs까지 내려가서 retrieval 한다

가정:

- Iter 1 끝에서 beam이 `[D12, D21]`가 되었다고 하자

---

## Iter 2 시작

- `selected_before = [D12, D21]`
- `expandable_paths = [D11, D12, D21, D22]`
- `retrieval_pool = D12 subtree leaf docs + D21 subtree leaf docs`

여기서 제일 중요한 포인트:

- `selected_before`는 `[D12, D21]` 두 개뿐이다
- 하지만 `expandable_paths`에는 `[D11, D22]`도 아직 남아 있다
- 이유는 `D11`, `D22`가 prediction tree 안에는 이미 생성되었지만, 아직 expand되지 않은 non-leaf이기 때문이다

이번 iter에서 실제로 보는 child:

- `D12` slot traversal slate = `[D121, D122]`
- `D21` slot traversal slate = `[D211, D212]`

가정:

- Iter 2 끝에서 beam이 `[D121, D211]`가 되었다고 하자

---

## Iter 3 시작

- `selected_before = [D121, D211]`
- `expandable_paths = [D11, D22, D121, D122, D211, D212]`
- `retrieval_pool = D121 subtree leaf docs + D211 subtree leaf docs`

이번 iter에서 실제로 보는 child:

- `D121` slot traversal slate = `[l121a, l121b]`
- `D211` slot traversal slate = `[l211a, l211b]`

여기서 beam이 leaf 직전 branch까지 내려온 상태다.

---

## beam이 leaf까지 도착하면 어떻게 되나

위 Iter 3에서 `D121`과 `D211`을 expand한다고 하자.

그러면 그 iteration 동안은:

- `D121`의 child leaf docs인 `[l121a, l121b]`를 scoring한다
- `D211`의 child leaf docs인 `[l211a, l211b]`를 scoring한다

즉 leaf에 도착한 순간의 immediate candidate는 그 leaf docs다.

여기서 질문이 많았던 포인트를 한 줄로 먼저 쓰면:

- leaf에 도착했을 때 selector local `retrieval_pool`은 **selected parent branch 아래의 모든 leaf docs**다
- **beam size만큼의 leaf만** pool에 넣는 것이 아니다
- 다만 그 full pool에서 실제 retrieval 결과로 쓰는 hit list는 `round5_mrr_pool_k`개 top hit로 잘린다

즉 구분은 아래처럼 해야 한다.

- pool size를 정하는 기준:
    - `selected_before` subtree 아래 leaf 전체
- retrieval hit 수를 자르는 기준:
    - `round5_mrr_pool_k`
- beam 크기를 자르는 기준:
    - `max_beam_size`

따라서 이 셋은 서로 다른 축이다.

하지만 scoring이 끝난 뒤에는:

- leaf docs는 `expandable`이 아니다
- `D121`은 이미 expand 완료라 frontier가 아니다
- `D211`도 마찬가지다

그래서 다음 iteration 시작 시 `expandable_paths`에서는 `D121`, `D211`이 사라진다.

예를 들면 다음 iteration 시작 시:

- `selected_before`는 baseline/update 결과에 따라 `[D122, D22]`처럼 바뀔 수 있다
- `expandable_paths`는 대략 `[D11, D22, D122, D212, ...]`처럼 "남아 있는 미확장 non-leaf"들로 재구성된다
- `retrieval_pool`도 새 `selected_before` 기준으로 다시 잡힌다

즉 leaf에 닿은 branch는 그 자리에서 끝난다.

- 그 branch 아래 leaf docs는 그 iteration에서 scoring된다
- 그 뒤에는 더 내려갈 branch가 없으므로 beam frontier에서는 빠진다
- 대신 그 leaf docs는 predicted leaf로 남아서 `cumulative_pool`에는 축적될 수 있다

---

## `D1`이 막히고 baseline에서 `D2`가 다시 들어오는 3-step 예시

이번에는 일부 branch만 leaf에 닿는 경우를 따로 보자.

```text
R
|- D1
|  |- D11
|  `- D12
|     |- l121
|     `- l122
`- D2
   |- D21
   |  |- D211
   |  `- D212
   `- D22
```

가정:

- beam size는 2
- `D11`, `D22`, `D211`, `D212`는 아직 더 내려갈 수 있는 non-leaf다
- `D12`의 직계 child는 leaf라서, `D12`를 한 번 expand하면 그 branch는 끝난다

### Step 1. leaf 직전 iteration 시작

- `selected_before = [D12, D21]`
- `expandable_paths = [D11, D12, D21, D22]`
- `selector retrieval_pool = leaves(D12 subtree) + leaves(D21 subtree)`
- `traversal slate(D12) = [l121, l122]`
- `traversal slate(D21) = [D211, D212]`

이 iteration에서 중요한 점:

- `D12`는 이번 iter에 leaf child를 scoring한다
- `D21`는 아직 non-leaf child를 scoring한다
- `D11`, `D22`는 beam에는 없지만 여전히 `expandable_paths` 안에 남아 있다

### Step 2. iteration 종료 직후

`sample.update(...)`가 끝나면:

- `D12`는 expand 완료다
- `l121`, `l122`는 predicted leaf가 된다
- leaf는 expandable이 아니므로 `D12` 쪽 frontier는 사라진다
- `D21`가 expand되면서 `D211`, `D212`가 새 expandable path로 생길 수 있다

그래서 baseline 기준 새 `expandable_paths`는 예를 들면:

- `[D11, D22, D211, D212]`

여기서 baseline beam refill은 이 path들을 `path_relevance` 순으로 정렬해서 top-2를 뽑는다.

예를 들어 점수가:

- `D211 = 0.78`
- `D22 = 0.74`
- `D11 = 0.70`
- `D212 = 0.66`

이면 다음 beam은:

- `[D211, D22]`

즉 `D12`가 끝난 자리를 baseline이 `D22` 같은 다른 expandable branch로 채운다.

### Step 3. 다음 iteration 시작

- `selected_before = [D211, D22]`
- `expandable_paths = [D11, D22, D211, D212]`에서 beam top-2만 active
- `selector retrieval_pool = leaves(D211 subtree) + leaves(D22 subtree)`

여기서 핵심:

- 방금 beam에 새로 들어온 `D22`의 descendant leaf들은 **이 다음 iteration부터** selector retrieval pool에 포함된다
- 같은 iteration 안에서 바로 들어오는 것은 아니다
- selector retrieval pool은 항상 현재 `selected_before` 기준으로 다시 계산된다

### `cumulative_pool`은 다르게 움직인다

위 예시에서 `D22`가 beam에 새로 들어왔다고 해서:

- `cumulative_pool`에 `D22 subtree` leaf가 자동으로 바로 들어오지는 않는다

`cumulative_pool`은:

- 지금까지 실제로 도달한 predicted leaf만 누적한다

그래서 위 흐름에서는:

- `l121`, `l122`는 `cumulative_pool`에 들어갈 수 있다
- `D22 subtree` leaf들은 아직 도달하지 않았으므로 `cumulative_pool`에 없다

정리하면:

- baseline으로 beam에 재진입한 branch의 descendant leaf들은 **다음 iter의 selector retrieval_pool**에는 들어간다
- 하지만 **cumulative_pool**에는 그 branch를 실제로 더 내려가서 leaf를 만들기 전까지는 들어가지 않는다

---

## 한 줄씩만 다시 요약

### `selected_before`

- 이번 iter 시작 시 beam에 실제로 올라와 있는 branch path

### `expandable_paths`

- beam 안이든 밖이든 상관없이, prediction tree 안에 이미 생성되었고 아직 expand되지 않은 non-leaf path 전체

### `retrieval_pool`

- `selected_before` subtree 아래 leaf docs의 합집합
- `selected_before`가 비면 전체 leaf docs fallback

### leaf 도착 후

- 그 iteration에서는 leaf docs를 scoring한다
- selector local `retrieval_pool`은 selected parent branch 아래 leaf docs 전체이며, beam size로 자르지 않는다
- 실제 retrieval 결과는 그 full pool에서 `round5_mrr_pool_k` top hit만 사용한다
- 하지만 leaf docs는 expandable이 아니므로 다음 iter beam frontier에서는 사라진다
- 다음 iter beam은 남아 있던 다른 `expandable_paths` 중에서 다시 뽑힌다

---

## 가장 중요한 오해 하나만 정리

일반적으로 retrieval pool은:
- `selected_before` subtree 아래 전체 descendant leaf docs

leaf-only case에서도 같다.

- selected branch의 child가 전부 leaf이면:
    - 그 child leaf docs 전체가 곧 retrieval pool이다
    - 그중 일부만 beam size만큼 고르는 구조가 아니다
    - retrieval은 그 full pool에 대해 수행하고, 반환 hit 수만 `round5_mrr_pool_k`로 제한된다
