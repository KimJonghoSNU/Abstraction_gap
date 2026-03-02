# 2026-02-23 Tree Builder 정리

## 목적
- 기존 content-hierarchical 성향 트리 대신, category-first hierarchy를 만들기 위한 builder를 구현한다.
- `LATTICE` 논문 section 3.3.2 흐름(LLM 기반 계층 키워드 생성 + top-down 구조화)은 유지하되,
  1) category 중심 분류
  2) 최종 저장은 DAG
  를 반영한다.

## 구현 방향
- 입력은 `data/BRIGHT/long_documents/*.parquet` (`id`, `content`)를 사용한다.
- long document 단위에서 5-level category hierarchy를 LLM으로 생성한다.
- 생성 경로를 이용해 DAG를 만든 뒤, split 문서(`data/BRIGHT/documents/*.parquet`)를 leaf로 attach한다.
- 런타임 호환을 위해 DAG에서 single-parent projection tree를 추가 생성한다.
- 분기 상한은 논문 문구(`M ~ 10-20`)를 반영해 `--max_branching`(기본 20)으로 제한한다.

## 구현 내용

### 1) 코드 위치
- Builder: `scripts/tree_builder/build_category_dag_topdown.py`
- 실행 wrapper: `scripts/tree_builder/build_category_dag_topdown.sh`

### 2) LLM 단계
- passage별 JSON 출력 강제 + parse 실패 시 regenerate 재시도.
- level별 길이 제한/정규화:
  - L1: 1-2 words
  - L2: 3-4 words
  - L3: 4-6 words
  - L4: 7-10 words
  - L5: 11-20 words
- alternate path를 허용해서 DAG 후보를 만든다.

### 3) DAG 구성
- 노드: L0(root), L1~L5(category), L6(split leaf).
- leaf는 다중 부모 허용(`--leaf_parent_cap`, 기본 2).
- split leaf attach 시 prefix + 제목 유사도 기반으로 parent 후보를 점수화한다.

### 4) Projection tree
- DAG edge weight 기반으로 child의 primary parent를 1개 선택한다.
- 이렇게 만든 projection은 기존 `SemanticNode` 소비 코드와 호환되는 `tree-*.pkl`로 저장한다.

### 5) 출력 산출물
- `trees/BRIGHT/{subset}/category_dag_topdown_{version}.json`
- `trees/BRIGHT/{subset}/category_dag_edges_topdown_{version}.jsonl`
- `trees/BRIGHT/{subset}/category_leaf_membership_topdown_{version}.jsonl`
- `trees/BRIGHT/{subset}/category_build_report_topdown_{version}.json`
- `trees/BRIGHT/{subset}/category_tree_projection_{version}.pkl`
- `trees/BRIGHT/{subset}/tree-category-topdown-{version}.pkl`
- `trees/BRIGHT/{subset}/category_node_catalog_topdown_{version}.jsonl`


## Critical Note
- 현재 retrieval runtime은 tree traversal 가정이다.
- 그래서 DAG의 다중 부모 정보는 저장/분석에는 반영되지만, 실제 실행은 projection tree를 탄다.
- 즉, native DAG traversal 성능 이득은 아직 반영되지 않았다.
- DAG 자체를 직접 탐색하려면 `run_round4` 계열 탐색 로직(visited/중복 leaf dedup/경로 점수 병합)을 추가 확장해야 한다.





## Following LATTICE top-down
- 원본 유지: `scripts/tree_builder/build_category_dag_topdown.py`는 수정하지 않았다.
- 신규 구현: `scripts/tree_builder/build_category_dag_topdown_algo4.py`를 추가해 논문 section 3.3.2 / Algorithm 4 흐름을 반영했다.

반영된 핵심:
- long doc별 5-level summary 생성 (Figure 8 계열 프롬프트 사용).
- `PartitionQueue` 기반 top-down 재귀 분할.
- `SelectSummaryLevel` 휴리스틱으로 분할 레벨 선택.
- `ClusterLLM`이 topic description + mapping을 반환하고 leaf descendants를 재할당.
- 분할 결과를 category DAG로 저장하고, runtime 호환을 위해 projection tree를 함께 저장.

Figure 8/9와의 관계:
- 현재 프롬프트는 Figure 8/9를 그대로 복붙(verbatim)한 것은 아니고, category-first 목적과 매핑 안정성을 위해 일부 제약을 추가한 변형 버전이다.
- 특히 ClusterLLM 출력은 `keywords` 대신 `summary_ids` 매핑을 직접 받도록 조정했다.

devlog 목적(유지 사항):
- category-first hierarchy 유지.
- 최종 저장은 DAG + projection tree 유지.
- split leaf attach 유지.
- branching factor `M~10-20` 범위는 `--max_branching`으로 유지.

## 2026-02-26 실험 결과 (biology, baseline1 tree-only, 30B)
- 목적: 새 top-down v3 tree가 기존 bottom-up tree 대비 retrieval 성능을 개선하는지 확인.
- 비교 실행:
- top-down v3 로그: `results/BRIGHT/biology/S=biology-TV=category-topdown-algo4-v3/TPV=5-RInTP=-1-NumLC=10-PlTau=5.0/RCF=0.5-LlmApiB=vllm-Llm=Qwen3-30B-A3B/Instruct-2507-NumI=10-NumES=1000-MaxBS=2/S=baseline1_tree_only-FT=200-GBT=10-PreFRS=branch-RM=concat/RE=1-RCT=5-RCS=mixed-RGT=10-RRrfK=60/RRC=leaf-REM=replace-RSC=on/run.log`
- bottom-up v3 로그: `results/BRIGHT/biology/S=biology-TV=bottom-up-TPV=5-RInTP=/1-NumLC=10-PlTau=5.0-RCF=0.5-LlmApiB=vllm/Llm=Qwen3-30B-A3B-Instruct-2507/NumI=10-NumES=1000-MaxBS=2-S=baseline1_tree_only-FT=200/GBT=10-PreFRS=branch-RM=concat-RE=1-RCT=5/RCS=mixed-RGT=10-RRrfK=60-RRC=leaf-REM=replace/RSC=on/run.log`
- 결과 요약:
- top-down v3: `nDCG@10 = 33.12`, `Recall@10 = 34.28`, `Recall@100 = 37.20`, `Failed = 172` (bad_request 다수).
- bottom-up v3: `nDCG@10 = 52.95`, `Recall@10 = 55.88`, `Recall@100 = 72.50`, `Failed = 18`.
- 결론:
- 현재 top-down v3는 기존 bottom-up tree보다 성능이 크게 낮다.
- top-down 쪽은 node desc / keyword가 길어져 traversal prompt가 과도하게 길어지고, vLLM context limit(32768) 초과 에러가 빈번하다.
- 즉, 현 단계 병목은 tree 의미 품질 이전에 prompt length 관리 실패(과도한 keyword/desc 길이)다.
