# Devlog: Flat Retrieval → Branch-Guided Hierarchical Traversal (Abstraction Gap 실험)

## 0) 목표 (왜 이 실험을 하려는가)

이번 실험은 **abstraction gap**을 “표현(semantic similarity) 문제”가 아니라 **탐색 제약(local expansion) 문제**와 분리해서 진단하기 위한 것이다.

LATTICE류의 hierarchical traversal은 매 스텝에서 “현재 노드의 자식”만 평가하므로, 질문이 **현상(Type1) → 이론/원리(Type2)** 같은 **수직 점프(vertical jump)** 를 요구할 때 초기에 잘못 진입하면 지역 최적에 갇힐 수 있다.

따라서,
1) **leaf/branch 구분 없이 전체 노드를 전역 후보로 펼쳐서(flat)** 먼저 찾고  
2) 그 다음에 **선정된 branch(또는 그 조상)만 타고 들어가는(gated traversal)**  
파이프라인이 실제로 “abstraction gap 완화”로 이어지는지 확인한다.

> 참고: `abstraction_gap.md`의 Type 3분류(Entity/Theory/Analogy)는 “예시”이며, 모든 데이터셋/쿼리에 고정적으로 맞아떨어진다고 가정하지 않는다. 본 실험은 타입을 강제하기보다, **전역-후-국소 탐색이 gap을 구조적으로 완화하는지**를 먼저 본다.

---

## 1) 이 실험이 abstraction gap에 대해 말해줄 수 있는 것 (해석 프레임)

### 케이스 A: Flat 단계가 “gold leaf”는 못 맞추지만 “gold의 ancestor branch”를 자주 맞춘다
- 해석: gap의 본질은 “정답 문서와의 표면 유사도”가 아니라, **정답을 포함하는 상위 개념/토픽(중간 가설, bridge concept)** 을 먼저 잡는 문제.
- 주장: **branch-level hit**이 leaf-level hit의 선행 조건이 될 수 있으며, vertical navigation이 필요하다.

### 케이스 B: Flat → Gated traversal이 전체 retrieval 성능(nDCG/Recall)을 유의미하게 올린다
- 해석: 기존 실패의 상당 부분이 “LLM 판단력 부족”이 아니라 **탐색의 지역성 제약**에서 기인.
- 주장: abstraction gap은 전역 후보 기반의 **초기 진입점 교정(entry-point correction)** 만으로도 완화될 수 있다.

### 케이스 C: Flat 단계가 오히려 성능을 떨어뜨리거나 분산을 키운다
- 해석: “전역 후보를 더 주면 해결”이 아니라, 전역 후보에는 **hard negative / logic poisoning**도 같이 늘어날 수 있음.
- 주장: 이 경우에는 (i) query expansion의 품질, (ii) branch 선택의 보수성, (iii) 노드 표현(설명/요약/구조 신호) 설계가 핵심이며, 이후 단계에서 **aspect-only/구조적 신호**를 넣는 방향이 정당화된다.

---

## 2) 실험 개요 (Pipeline)

### Stage 0: Tree node catalog 만들기 (branch + leaf 전부)
- 입력: `trees/BRIGHT/<subset>/tree-<version>.pkl`
- 출력(파일):
  - `trees/BRIGHT/<subset>/node_catalog.jsonl`
    - 필드 예시: `node_key`, `path`, `depth`, `is_leaf`, `text`(desc), `num_children`, `num_leaves`
  - `trees/BRIGHT/<subset>/node_catalog.meta.json`
    - 인덱싱/버전/해시 등 메타
- 포인트:
  - **전역 검색 대상은 leaf/branch 모두 포함**.
  - leaf가 “문서”이고 branch가 “클러스터”이므로, branch 텍스트는 트리 구축 시 이미 있는 `desc`를 그대로 사용(추후 요약/정제는 ablation으로 분리).

### Stage 1: Query expansion (LLM prompt 기반)
- 목표: retriever가 잡기 어려운 “추상 브릿지” 표현을 query에 보강
- 출력:
  - `results/<dataset>/<subset>/qe_cache.jsonl` (query → expansions)
- 설계:
  - QE 캐시가 있으면 **추가 비용 0** (cache miss만 생성)
  - 사용자 프롬프트(외부 파일 경로) 또는 built-in 프롬프트 이름을 인자로 받아 실행:
    - `--qe_prompt_path <path>` 또는 `--qe_prompt_name bridge_v1|keywords_v1|paraphrase_v1`
  - 1개 expanded query 문자열 또는 N개 후보 쿼리를 생성하고, retriever score를 max/mean으로 집계하는 방식도 가능(옵션).

### Stage 2: Flat retrieval (Diver-Retriever-4B)
- 목표: “전체 노드 집합”에서 상위 K개 노드를 빠르게 뽑는다 (LLM으로 전부 스코어링하지 않음)
- 입력:
  - node_catalog + node_embeddings(사전 계산) + expanded query
  - retriever model/path는 CLI 인자로 전달: `--retriever_model Diver-Retriever-4B` (또는 로컬 경로)
- 출력:
  - query별 top-K 노드 리스트(점수 포함): `results/.../flat_hits.jsonl`
- 주의:
  - 네트워크 제한 환경이므로, 모델은 로컬/캐시 경로 사용을 전제로 하고 **경로/이름을 인자로 받는다**.

### Stage 3: Branch selection (gating set 만들기)
Flat top-K 결과는 leaf/branch가 섞인다. traversal gating을 위해 “들어갈 branch 집합”으로 정규화한다.
- 규칙(초기 단순안):
  1) top-K 중 branch는 그대로 후보
  2) top-K 중 leaf는 **leaf의 ancestor 중 특정 depth의 branch**로 승격
  3) 중복 제거 후 상위 B개 branch만 유지(`--gate_branches_topb`)
- 출력:
  - query별 gated branch root path 목록: `results/.../gates.jsonl`

### Stage 4: Gated hierarchical traversal (LATTICE traversal은 “gate 내부”만)
- 목표: 기존 LATTICE traversal(LLM 기반)을 유지하되, 탐색 공간을 gate로 제한하여
  - gap이 큰 질의에서 **맞는 추상 영역으로 빠르게 진입**하고
  - 불필요한 로컬 확장을 줄인다
- 구현 아이디어:
  - (A) “허용 노드 집합(allowed paths)”을 만들어, 그 외 노드는 `MaskedSemanticNode`처럼 desc를 숨기거나 excluded 처리
  - (B) 또는 “가짜 root”를 만들어 gate root들을 child로 두고 traversal 시작
- 출력:
  - 기존과 동일한 `InferSample` 결과 + 추가로 flat/gate 로그 저장

---

## 3) 사전 계산(offline) 작업: 중간 노드 임베딩 저장

LLM 비용을 줄이기 위해, **전 노드 임베딩을 미리 계산해 저장**한다.

- 입력: `node_catalog.jsonl`
- 출력:
  - `trees/BRIGHT/<subset>/node_embs.npy` (float32, shape: [N, D])
  - `trees/BRIGHT/<subset>/node_index.faiss` (선택: FAISS 사용 시)
  - `trees/BRIGHT/<subset>/node_id_map.json` (row ↔ node_key)
- 옵션:
  - FAISS를 쓰면 top-K가 빨라짐. 우선은 numpy cosine으로도 시작 가능(규모에 따라 결정).

---

## 4) 평가 지표 (기존 + abstraction gap 진단용)

### 기존 LATTICE 지표 (그대로)
- `nDCG@10`, `Recall@10`, `Recall@100`, `Coverage`

### 추가 진단 지표 (이번 실험의 핵심)
- `AncestorHit@K`: flat top-K 중 **gold leaf의 ancestor(임의 depth 포함)** 가 하나라도 있는지
- `GateHit`: gating한 branch들이 gold leaf의 경로와 겹치는지(“정답 서브트리 안에 들어갔는가”)
- `DepthShift`: flat에서 선택되는 노드들의 평균 depth/분산 (gap-heavy query에서 더 얕은(추상) 노드가 잡히는지)
- `Ablation`: (1) no-qe vs qe, (2) leaf-only flatten vs all-nodes flatten, (3) gate 강도(B) 변화

---

## 5) 구현 계획 (코드 단위)

### CLI 파라미터(초안)
- `--flat_then_tree` (bool): 이 모드 활성화
- `--retriever_model` (str): Diver-Retriever-4B 이름/경로
- `--node_catalog_path` (str)
- `--node_emb_path` (str)
- `--node_index_path` (str, optional)
- `--flat_topk` (int): flat retrieval top-K
- `--gate_branches_topb` (int): 최종 gate branch 수
- `--gate_depth` (int/str): leaf를 어느 depth의 ancestor로 승격할지 (또는 “auto”)
- `--qe_prompt_path` (str): query expansion 프롬프트 경로
- `--qe_num_samples` (int): expansion 후보 개수 (선택)

### 파일/스크립트(초안)
- `scripts/export_node_catalog.py`
  - 트리에서 모든 노드를 순회해 `node_catalog.jsonl` 생성
- `scripts/embed_node_catalog.py`
  - `--retriever_model`로 node embeddings 생성 후 저장
- `src/retrievers/diver.py`
  - Diver-Retriever-4B 래퍼(encode/query)
- `src/run.py` (모드 추가)
  - `--flat_then_tree` 활성 시 flat retrieval로 gate 생성 후, **기존 LATTICE traversal 루프(beam=2, iters=20)** 를 gate 내부에서 실행

### 저장/재현성
- 결과 파일에 반드시 포함:
  - 사용한 모델 경로/해시, tree_version, subset, flat_topk, gate_topb, qe_prompt_path, seed

---

## 6) 예상 리스크 / 체크포인트

- **노드 수가 많으면** 임베딩/검색이 느릴 수 있음 → FAISS 도입 여부를 subset별로 결정
- branch `desc` 품질이 낮으면 gate가 흔들릴 수 있음 → 이후 ablation으로 “branch 요약(구조적 신호)”를 추가
- qe가 잘못되면 hard negative로 치우칠 수 있음 → qe 후보 다중 생성 후 retrieval score로 선택하는 옵션 고려

---

## 7) 최소 성공 기준 (Go/No-Go)

- Flat 단계에서 `AncestorHit@K`가 baseline보다 상승하거나,
- Flat→Gate→Traversal이 baseline 대비 `Recall@10` 혹은 `nDCG@10`을 의미 있게 올리거나,
- 성능이 비슷하더라도 **탐색 비용(LLM 호출 수/토큰)** 을 유의미하게 줄이면(= “전역 retriever로 탐색 공간 축소” 효과) 다음 단계(타입/구조 신호)로 진행할 가치가 있다.

---

## 8) 실행 예시 (현 구현 기준)

### (A) 노드 카탈로그 export
```bash
python scripts/export_node_catalog.py \
  --tree_pkl trees/BRIGHT/biology/tree-bottom-up.pkl \
  --out_jsonl trees/BRIGHT/biology/node_catalog.jsonl
```

### (B) 노드 임베딩(offline)
```bash
python scripts/embed_node_catalog.py \
  --node_catalog_jsonl trees/BRIGHT/biology/node_catalog.jsonl \
  --model_path /data4/jaeyoung/models/Diver-Retriever-4B \
  --out_npy trees/BRIGHT/biology/node_embs.diver.npy
```

### (C) Flat→Gate→Traversal 실행 (beam=2, iters=20 유지)
```bash
cd src
python run.py \
  --subset biology --tree_version bottom-up \
  --flat_then_tree \
  --retriever_model_path /data4/jaeyoung/models/Diver-Retriever-4B \
  --node_emb_path ../trees/BRIGHT/biology/node_embs.diver.npy
```

---

## 260113) Query rewriting ↔ document traversal 상호작용(Interaction) 실험 설계

### 문제의식 (왜 “상호작용”이 필요한가)
- 지금까지의 agentic search는 사람이 정의한 레벨(Type/Aspect)을 기준으로 explore/exploit을 수행했고, LATTICE 실험은 “전역 앵커링(flat) → 국소 탐색(traversal)”로 **탐색 지역성(locality)** 병목을 완화할 수 있음을 보려는 것이었다.
- 하지만 현재 파이프라인은 대체로 **query가 고정(QE는 사전에 1회)** 이고, traversal은 그 query로만 내려간다. 이 구조에서는 “문서를 보면서 필요한 abstraction으로 이동한다”가 **정적인 1회 보정**에 머문다.
- abstraction gap이 큰 질의에서는 “어떤 브릿지가 필요한지”가 처음부터 명확하지 않기 때문에, traversal이 관측하는 요약/경로 신호를 query rewriting에 반영하고(업데이트), 다시 그 query로 탐색 정책을 바꾸는(게이팅/확장) **닫힌 루프**가 필요하다.

> 추가 동기: LATTICE의 branch 요약 노드는 “example/entity/theory” 같은 고정 레벨을 직접 맞추는 대신, **corpus가 제공하는 중간 개념(=query별 유효한 abstraction 단위)** 을 rewriting 입력으로 공급한다. 상호작용 설계는 “고정 레벨 설계 부담”을 줄이고, 서브카테고리별 brittleness를 완화하는 방향으로도 해석 가능하다.

### 핵심 가설
1) **Evidence-conditioned rewriting**(현재 경로/요약을 본 뒤 rewrite)이 “추상 브릿지 표현”을 더 잘 만들어낸다.
2) 이 rewrite가 traversal의 child 선택에 반영되면, “올바른 semantic region으로의 이동”이 빨라져 nDCG/Recall이 상승한다.
3) 효과가 없다면 원인은 (i) branch 요약 품질, (ii) rewrite가 drift하여 hard negative를 키움, (iii) traversal 스코어링/슬레이트 구성 병목 중 하나로 좁혀진다.

### 실험 세팅(공통)
- 데이터/트리: BRIGHT, subset별로 동일 세팅 사용
- 비용 고정: **beam size=2, iterations=20 유지**
- 비교 단위: 같은 seed / 같은 LLM / 같은 retriever(Diver)로 ablation
- 캐싱: rewrite/QE는 cache hit이면 0비용(재실행 안정성)

### 변형(Conditions) — “상호작용 강도”를 단계적으로 증가
- **C0 (Baseline-LATTICE)**: original traversal (flat 없음, rewrite 없음)
- **C1 (Flat→Gate, no interaction)**: 현재 flat→gate→traversal (QE는 옵션, traversal query 고정)
- **C2 (Gate-conditioned single rewrite)**:
  - flat→gate 이후, gate로 선택된 상위 branch들의 `desc`(또는 path 요약)를 입력으로 **rewrite 1회** 생성
  - 이후 traversal은 `rewritten_query`를 사용(또는 원쿼리+rewrite 혼합)
  - 목적: “상호작용”을 최소 비용(1회 rewrite)으로 도입했을 때 이득이 있는지 확인
- **C3 (Stagnation-triggered rewrite, per-query budgeted)**:
  - traversal 중 성능/점수 개선이 정체될 때만 rewrite 호출 (예: 연속 3 iter 동안 top leaf score 개선 없음)
  - 질의당 rewrite 호출 상한 `<= 2`로 고정
  - 목적: 비용을 통제하면서도 “필요할 때만 abstraction 업데이트”가 이득인지 확인
- **C4 (Beam-conditioned rewrite + score fusion)**:
  - beam마다 현재 path(ancestor branch desc들) + 최근 top leaf(제목/짧은 요약)로 rewrite를 따로 생성(상한 유지)
  - child scoring을 `original_query`와 `beam_rewrite` 둘 다로 계산해 RRF/가중합으로 합쳐 확장
  - 목적: 상호작용의 최대치(가장 agentic한 형태)가 실제로 추가 이득을 주는지 확인

### 성공/실패를 해석하기 위한 진단 지표(추가)
- **AncestorHit@K / GateHit**: flat이 올바른 서브트리로 “진입”시키는지
- **AncestorHit by depth (final top-10 leaf 기준)**: baseline 대비 “올바른 조상 영역으로 더 자주 들어갔는지”
- **Rewrite dynamics**:
  - rewrite 호출 횟수/질의(비용)
  - rewrite가 만들어낸 키워드가 실제로 다음 iter에서 선택된 branch/leaf에 반영되는지(간단히 토큰 overlap/키워드 hit)
- **Ablation outcome mapping**:
  - C1↑, C2×: “전역 앵커링은 이득이지만, gate 조건 rewrite는 별 효과 없음” → rewrite 입력 설계/프롬프트 병목
  - C2↑, C3≈: “1회 조건부 rewrite면 충분” → 상호작용을 과하게 할 필요 없음(비용 절감 포인트)
  - C3↑, C4↑↑: “진짜 상호작용(beam별)이 추가 이득” → abstraction gap이 질의별/경로별로 달라 업데이트가 필요
  - C1↑, C2↓: rewrite drift로 hard negative 확대 가능 → 원쿼리 혼합/제약 강화 필요

### “abstraction gap”과의 연결(서술 템플릿)
- “Flat→Gate는 올바른 semantic region으로의 **초기 진입**을 교정하고, interaction rewrite는 그 region 내부에서 필요한 abstraction/bridge를 **점진적으로 정제**한다.”
- “따라서 성능 향상이 C1에서만 나타나면 gap의 주원인은 locality이고, C2–C4에서 추가 향상이 나타나면 gap의 주원인은 ‘동적 브릿지 생성(표현 업데이트)’까지 포함한다.”

### 구현 전 체크리스트(실험이 깔끔해지기 위한 최소 요구)
- rewrite cache key 정의: `(original_query, subset, tree_version, path_signature, iter_idx, beam_idx)` 중 필요한 최소만
- rewrite 예산/트리거를 명시적으로 고정(질의당 최대 호출 수)
- 결과 저장: 조건(C0–C4), rewrite 호출 수, 마지막 fused ranking, gate paths를 함께 저장(재분석 가능)

## 260114) Implementation updates (rewrite + logging)

- Rewrite 입력 단순화: **original query + last rewrite(raw)**만 프롬프트로 전달 (current/user query 제거).
- Query composition: 실제 retrieval/traversal query는 **original + current rewrite**로 구성, rewrite는 `replace` 모드 권장.
- Rewrite 컨텍스트 소스 확장:
  - `flat`: flat retrieval top‑K hits
  - `slate`: 현재 traversal slate
  - `fused`: flat leaf + traversal leaf RRF
  - `mixed`: `fused` + `slate`를 다시 RRF로 섞어 top‑K 추출
- Rewrite history 저장: `rewrite_history`를 pkl에 포함하도록 `SAVE_LIST` 추가.
- QE 템플릿 파싱 안정화: JSON braces로 인한 `str.format` 오류에 fallback 추가.
- 로그/결과 저장 경로 단축: `HyperParams.__str__`에서 `MaxBS` 기준으로 subfolder 분리, `save_exp`는 hp 경로에 맞게 mkdirs 처리.

## 260116) Pre-flat rewrite pipeline + logging + run scripts

- Added pre-flat rewrite stage: initial flat retrieval on original query, rewrite from flat context, then re-run flat retrieval for gating/traversal.
    - New flags: `--pre_flat_rewrite`, `--pre_flat_rewrite_source {branch,all}`.
    - When `--pre_flat_rewrite` is set, QE is disabled for that run.
    - Pre-flat rewrite uses the same rewrite prompt/template as traversal rewrite.
- Expanded argument help to clarify QE vs rewrite vs pre-flat rewrite usage.
- Added per-iteration logging of slate node counts by depth (top-k capped by `flat_topk`) in `run.log`.
- New run scripts under `src/bash/` for experiment variants:
    - `run_exp1_qe_iter.sh` (QE -> flat -> traversal with iterative rewrite)
    - `run_exp2_preflat_branch_noiter.sh` (branch-only pre-flat rewrite, no traversal rewrite)
    - `run_exp2_preflat_all_noiter.sh` (all-node pre-flat rewrite, no traversal rewrite)
    - `run_exp3_preflat_branch_iter.sh` (branch-only pre-flat rewrite, traversal rewrite enabled)

## 260118) Branch-seeded traversal from flat retrieval gates

- Gate construction now resolves ancestor/descendant conflicts by score: keep the higher-scoring path when one is a prefix of the other.
- Flat gate scores are carried into traversal and used to seed beam state paths (branch nodes as initial beam states).
- Traversal uses gate-seeded beams only for iter 0; after the first update, gating is cleared and normal traversal continues.
- Difference vs previous version:
    - Before: flat retrieval only constrained traversal via `allowed_prefixes` (slates still started at the root).
    - Now: flat retrieval gates directly initialize the beam to those branch nodes (traversal starts from the retrieved branches).
    - Toggle: `--seed_from_flat_gates` enables the new behavior; omit it to keep the old behavior.
