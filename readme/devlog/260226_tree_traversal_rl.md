# 2026-02-26 Tree Traversal RL 계획

관련 문서:
- `readme/devlog/260223_tree_builder.md`

## 배경
- `scripts/tree_builder/build_category_dag_topdown_algo4_v3.py` 재현 성능 이슈가 있어서, tree 생성 자체 개선은 잠시 보류한다.
- 당분간은 tree를 고정하고, traversal policy를 학습해서 성능을 올린다.

## 범위
- retriever 사용 없이 traversal policy만 학습한다.
- 실행 구조는 `src/run_original.py`의 탐색 흐름(beam search)과 정합되게 설계한다.
- action은 `global-top2`로 고정한다.

## 데이터 정합성 체크 (선행 완료)
- 소스: `data/reason-embed-data-0928/*-formatted.jsonl`의 `pos/neg` 텍스트
- 대상: `data/BRIGHT/documents/*.parquet`의 문서 `content`

결과:
- 텍스트 -> 문서 매핑 exact match: `100%`
  - `pos`: `991,791 / 991,791`
  - `neg`: `7,101,302 / 7,101,302`
- tree leaf coverage:
  - 대부분 subset은 `1.0`
  - 일부 top-down subset(`aops`, `leetcode`, `theoremqa_questions`)만 `~0.9999`
- 결론: 학습 라벨 신뢰도는 충분히 높고, leaf 누락 소수 케이스는 `skip` 처리한다.

## RL 문제 정의

### 1) 상태(State)
- `query`
- 현재 depth
- 현재 expandable path들의 노드 설명(후보 slate)
- 현재까지 선택된 path 히스토리

### 2) 행동(Action)
- 매 step에서 후보 중 점수 상위 2개 path를 선택한다 (`global-top2`).
- 즉, beam size는 `2`로 고정한다.

### 3) 예산(Budget)
- 예산은 `max_depth`로 정의한다.
- depth가 마지막에 도달했을 때 terminal 보상을 계산한다.

### 4) 보상(Reward)
- non-leaf depth 보상 `A_d`:
  - subtree에 `pos`가 있으면 `+1`
  - `pos`는 없고 `neg`가 있으면 `-0.1` (hard negative 반영한 약한 패널티)
  - `pos/neg` 둘 다 없으면 `-1`
- leaf(마지막 depth) 보상 `B`:
  - terminal `nDCG@10` 절대값 사용
- leaf 매핑 누락 케이스:
  - 보상 계산에서 `skip`

의도:
- non-leaf에서는 방향성(branch correctness)을 학습하고,
- leaf에서는 실제 retrieval objective(`nDCG@10`)를 직접 학습한다.

## 학습 절차 (초안)

### Stage 1: Branch Policy 학습
- 대상: non-leaf depth
- 목적: branch 선택 품질 개선
- 방법: preference 기반 학습(DPO 우선 검토)

### Stage 2: Leaf Policy 학습
- 대상: 마지막 depth
- 목적: terminal `nDCG@10` 최적화
- 방법: GRPO 계열 또는 preference-RL

## 구현 시 주의점
- reward 라벨 생성 시 문서 매핑/leaf 포함 여부를 먼저 체크하고, 누락은 반드시 `skip`.
- non-leaf와 leaf의 목적함수가 다르므로 학습 로그도 분리 기록:
  - branch reward 로그 (`A_d`)
  - terminal reward 로그 (`nDCG@10`)
- beam action은 반드시 `global-top2`로 유지한다 (parent-local top2 아님).

---

## 2026-02-26 구현 업데이트 (DPO + GRPO)

이번 턴에서 실제 구현한 코드:
- `scripts/train/build_dataset/build_traversal_dpo_dataset.py`
- `scripts/train/train_branch_policy_dpo.py`
- `scripts/train/train_branch_policy_grpo.py`
- `scripts/train/run_build_traversal_dpo_dataset.sh`
- `scripts/train/run_train_branch_policy_dpo.sh`
- `scripts/train/run_train_branch_policy_grpo.sh`

추가 설치:
- `trl` (기존 사용), `peft`, `bitsandbytes`

### 1) DPO 데이터 생성 (`build_traversal_dpo_dataset.py`)
- 입력: `subset/branch_steps.jsonl`
- 출력: `train_dpo.jsonl`, `eval_dpo.jsonl`
- 학습 포맷: `prompt`, `chosen`, `rejected` (TRL DPOTrainer 표준)
- 프롬프트는 `src/prompts.py`의 `get_traversal_prompt(..., leaf_cluster=False)`를 그대로 사용
- 선호 단위는 `top2 set`:
  - `P = {reward == 1.0}` (gold branch pool)
  - `|P| >= 2`: positive-positive 조합 샘플링
  - `|P| == 1`: chosen은 유일한 gold branch를 반드시 포함
  - rejected는 우선 gold 미포함 조합을 우선 배치
- `--max_pairs_per_step_dpo`로 step당 pair 수 상한을 걸어 tie-heavy step 과대표집을 방지

### 2) DPO 학습 (`train_branch_policy_dpo.py`)
- TRL `DPOTrainer` 기반
- QLoRA 4bit 경로 지원 (`--load_in_4bit`)
- LoRA adapter 학습 (full fine-tune 아님)
- query-level split으로 train/eval 누수 방지 (데이터 빌더 단계)

### 3) GRPO 학습 (`train_branch_policy_grpo.py`)
- TRL `GRPOTrainer` + custom reward function
- 각 step 후보 branch에 대해 subtree 문서 기준 `nDCG@10`를 미리 계산
- 모델 completion(JSON)에서 `ranking` 파싱 -> top2 선택 -> 선택된 2개 branch의 subtree `nDCG@10` 평균을 보상으로 사용
- gold 기준은 any-gold 포함 조건을 유지

중요 제약(현재 구현):
- 현재 GRPO는 **step-level reward 최적화**다.
- 즉, multi-step episode return을 직접 최적화하는 environment rollout 버전은 아직 아님.
- 다음 확장 포인트는 TRL의 `rollout_func`/`environment_factory`로 episode-level return을 붙이는 것.

### 4) 실행 순서
1. DPO 데이터 생성  
   `bash scripts/train/run_build_traversal_dpo_dataset.sh`
2. DPO 학습  
   `bash scripts/train/run_train_branch_policy_dpo.sh`
3. GRPO 학습  
   `bash scripts/train/run_train_branch_policy_grpo.sh`

---

## 2026-03-02 중간 결과 업데이트

### 1) DPO 실험 결과 (baseline1_tree_only, merged)
- 참조 파일: `results/BRIGHT/ndcg_summaryS=baseline1_tree_only.csv`
- 비교 대상:
    - baseline: `Llm=Qwen3-30B-A3B-Instruct-2507`
    - DPO 적용 모델: `Llm=merged`
- 결과(`overall|avg_ndcg_max`):
    - baseline: `39.3775`
    - merged: `32.34125`
    - delta: `-7.03625` (절대값 기준 하락)

subset별 `ndcg_max` delta (merged - baseline):
- biology: `-15.68`
- stackoverflow: `-13.95`
- psychology: `-6.51`
- theoremqa_theorems: `-6.31`
- economics: `-6.16`
- robotics: `-5.57`
- sustainable_living: `-1.68`
- earth_science: `-0.43`

해석(중간 결론):
- 현재 데이터/설정에서는 DPO로 학습한 `merged` 모델이 baseline 대비 일관되게 성능 하락했다.
- 특히 `biology`, `stackoverflow`에서 큰 폭 하락이 발생했다.

### 2) GRPO 진행 상태
- 현재 상태: **본 실험 미완료(실행 안정화 실패)**.
- 관측된 핵심 이슈:
    - `use_vllm=True` 경로에서 communicator 초기화 단계 NCCL 오류로 중단.
    - 환경 특이적으로 socket 초기화 시 address family 관련 경고/에러가 반복됨.
- 확인된 사실:
    - 같은 코드에서 `use_vllm=False` smoke run은 정상 종료됨.
    - 즉, 현재 blocker는 보상 함수/데이터 로직보다 `GRPO + external vLLM communicator` 경로에 가깝다.

### 3) 즉시 액션 아이템
1. `accelerate + zero2` 경로로 GRPO 재실행 안정화 우선.
2. `use_vllm=False` fallback으로 먼저 baseline GRPO 학습/성능 체크(실험 공백 최소화).
3. 실행 안정화 후에만 DPO vs GRPO 공정 비교를 진행.
