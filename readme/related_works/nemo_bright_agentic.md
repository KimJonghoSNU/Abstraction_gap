# NVIDIA NeMo Retriever BRIGHT Agentic Pipeline

이 메모는 NVIDIA NeMo Retriever의 BRIGHT submission과 구현을 보고, 우리 구현에 직접 참고할 만한 explore/exploit 디테일만 정리한다.

기준 질문:

1. query를 iteration마다 concat하는가?
2. retrieval pool은 어떻게 정해지는가?
3. rewrite에 쓰는 문서 pool은 무엇인가?
4. 메모리는 어떻게 관리하는가?
5. final candidate pool은 어떻게 유지되고, final selection은 언제 trigger되는가?

Sources:

- submission note: <https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/submissions/bright_agentic.md>
- implementation repo: <https://github.com/NVIDIA/NeMo-Retriever/tree/main/retrieval-bench>

## One-paragraph summary

이 시스템은 tree-constrained retrieval이 아니라, **strong base retriever + query-centric multi-shot retrieval + trajectory-level document union + final reranking** 구조다. query string을 누적 concat하지는 않는다. 대신 original query와 이전 retrieval 결과를 message history에 계속 남기고, LLM이 매 step 새로운 sub-query를 만든다. retrieval은 매번 full corpus dense retrieval을 다시 한다. 초반에 찾은 좋은 문서는 이후 query drift로 다시 안 떠도, retrieval trajectory 전체의 union에 남겨두고 마지막에 `final_results`, selection agent, RRF 중 하나로 최종 top-k를 정한다.

## Why BRIGHT performance seems to improve

내 해석으로는 gain이 세 층에서 온다.

1. **Base retriever가 강하다**
    - `llama-nv-embed-reasoning-3b`
    - synthetic reasoning-intensive queries
    - positive annotation
    - hard negative mining
    - ReasonEmbed / ReasonAug / ReasonRank data 사용
2. **BRIGHT subset마다 task-specific instruction prefix를 쓴다**
    - 예: Biology post / Coding problem / Math theorem retrieval
3. **single-shot retrieval이 아니라 multi-query trajectory를 만들고, 마지막에 trajectory-level fusion을 한다**
    - 여러 sub-query retrieval
    - doc union
    - final_results or selection agent or RRF

즉 submission note를 읽을 때 agent loop만 보면 안 되고, retriever 자체가 이미 reasoning-heavy IR에 맞게 세게 튜닝되어 있다는 점을 같이 봐야 한다.

## Algorithm sketch

### Step 0. Index and retriever setup

- corpus 전체를 dense retriever로 index한다.
- BRIGHT task key에 따라 query prefix를 다르게 붙인다.
- backend retriever internal top-k 기본값은 `500`이다.

### Step 1. Start from the original query

- agent는 original query를 받는다.
- 기본 설정 `user_msg_type="with_results"`라서, agent loop 시작 전에 original query로 한 번 retrieval한다.
- 이 initial retrieval 결과가 user message에 같이 들어간다.
- 따라서 step 0부터 agent는 original query와 initial retrieved docs를 같이 본 상태에서 시작한다.

### Step 2. Agent repeatedly explores with new sub-queries

- agent는 `think`, `retrieve`, `final_results` tool을 쓸 수 있다.
- 각 retrieval step에서 LLM이 새로운 sub-query를 만든다.
- 이전 query string을 자동으로 concat해서 다음 query로 넘기지는 않는다.
- accumulated state는 query string이 아니라 conversation/message history다.

### Step 3. Each retrieval is still full-corpus retrieval

- retrieval call은 local pool이나 branch pool이 아니라 전체 corpus dense retrieval이다.
- query만 바뀌고 scoring pool은 전역 corpus 그대로다.
- retriever는 global top-500을 만든다.
- tool layer는 그 결과 중에서 실제로 agent에게 보여줄 문서를 다시 자른다.

### Step 4. The trajectory keeps documents, not branch states

- retrieval trajectory 동안 본 문서는 dedup된 document memory처럼 유지된다.
- final 단계에서는 retrieval trajectory 전체에서 모은 문서 union을 후보 pool로 쓴다.
- 즉 이 시스템의 positive memory는 branch memory가 아니라 document-union memory다.

### Step 5. Final selection

- 이상적인 경우, main agent가 `final_results` tool을 호출하며 종료한다.
- 그래도 conclude 단계에서 별도로 selection agent와 RRF용 artifact는 계산한다.
- 최종 output source priority는:
    1. `final_results`
    2. `rrf_scores`
    3. `selection_agent`

즉 selection agent가 항상 계산될 수는 있지만, main agent가 정상적으로 `final_results`를 내면 그 결과가 우선이다.

## Explore / exploit details that matter

### What is the query used for retrieval?

- retrieval query는 매 step LLM이 새로 쓰는 sub-query다.
- original query는 message history 안에 계속 남는다.
- 즉 **original query is kept as context**, but **not explicitly concatenated into every new retrieval query**.

중요한 예외:

- optional query rewriting hook이 구현돼 있긴 하다.
- 하지만 config default는 `use_query_rewriting=False`다.
- 즉 기본 pipeline에서는 retrieval 직전에 별도 automatic rewrite를 하지 않는다.

### What document pool is used for "rewrite"?

이 시스템에는 우리처럼 명시적인 separate rewrite module이 없다.

- "rewrite"는 사실상 LLM이 다음 sub-query를 다시 쓰는 행위다.
- 이때 사용하는 evidence pool은 별도 selection된 rewrite pool이 아니라, **conversation history에 들어 있는 retrieved docs와 tool outputs**다.
- 특히 initial retrieval docs와 직전 retrieval docs가 가장 직접적인 단서가 된다.

즉 query rewriting context는:

- original query
- initial retrieved docs
- 이전 step들의 retrieved docs
- think tool outputs

의 합이다.

### What document pool is used for retrieve?

- retrieval scoring pool은 항상 full corpus다.
- local pool restriction이 없다.
- tree descendant pool 같은 것도 없다.

다만 agent가 한 step에서 실제로 읽는 결과 수는 제한된다.

- `RetrieveTool` default `top_k`는 보통 target top-k를 따라 기본 `10`
- `retrieve_with_guarantees`가 seen docs와 excluded docs 수만큼 over-fetch해서
- 최종적으로는 보통 `10`개의 new docs를 agent에게 보여주려고 한다

즉 구현 관점에서 분리하면:

- **retriever-side pool**: full corpus, global top-500
- **LLM-visible per-step pool**: dedup 후 top-10 new docs

## Memory management

문서가 길면 memory 관리가 어려워지는데, 이 구현은 세 가지로 처리한다.

### 1. Message-history memory

- original query와 retrieved docs가 conversation history에 남는다.
- LLM은 이를 읽고 다음 sub-query를 만든다.

### 2. Repeated-doc compression

- 이미 본 doc가 다시 retrieval되면 full text를 다시 길게 넣지 않는다.
- 대신 image/text를 제거하고, "이 문서는 이전에 retrieval되었다"는 note만 남긴다.

즉 repeated docs는 **ID + short note** 수준으로 압축된다.

### 3. Final candidate memory

- retrieval trajectory 전체에서 unique docs를 dedup해서 final candidate pool로 유지한다.
- 이것이 사실상 long-horizon memory 역할을 한다.

selection 단계에서 context가 너무 길어지면 추가 pruning도 있다.

- selection agent가 context window error를 내면
- lowest RRF docs 하위 `1/4`를 버리고 다시 selection을 시도한다

즉 final pool은 무한정 유지하지 않고, overflow 시 low-RRF docs를 버리는 식으로 관리한다.

## Query drift: what they actually do

이 구현은 query drift를 탐색 단계에서 강하게 막지 않는다. 대신 drift가 final output을 망치지 않도록 완화한다.

핵심 완화 장치는 네 가지다.

1. original query가 계속 message history에 남는다
2. 초반에 찾은 좋은 docs가 retrieval trajectory union에 남는다
3. final selection은 original query 기준으로 다시 정렬된다
4. main agent가 실패하면 RRF가 trajectory 전체 retrieval을 fusion한다

즉 이 시스템은 **state recoverability**가 아니라 **document recoverability**를 강하게 잡는다.

## Direct answers to the implementation questions

### Does it keep concatenating the query?

아니다.

- explicit query concat은 없다.
- original query는 context로 유지된다.
- retrieval에 들어가는 string은 매번 새로 만든 sub-query다.

### Is the original query used together with new queries?

직접 string concat되지는 않지만, 간접적으로는 그렇다.

- original query가 message history에 남아 있기 때문이다.
- 또한 final selection 단계는 original query를 직접 다시 사용한다.

### What is the rewrite pool?

- 별도 rewrite pool은 없다.
- retrieved docs와 tool outputs가 message history 안에서 rewrite context 역할을 한다.

### What is the retrieve pool?

- full corpus다.
- local branch pool이나 cumulative restricted pool이 아니다.

### How is memory managed?

- short-term: message history
- repeated-doc compression: already seen docs는 note만 남김
- long-term: retrieval trajectory 전체의 unique doc union
- overflow handling: selection stage에서 low-RRF docs를 버리고 retry

### How is the final candidate pool maintained?

- retrieval trajectory 전체에서 모은 unique docs union

### When is final selection triggered?

- main agent가 `final_results`를 호출하면 interaction은 종료된다
- 그 뒤 conclude 단계에서 selection agent / RRF artifact도 계산될 수 있다
- 최종 output source priority는 `final_results > RRF > selection_agent`

## What is directly useful for us

직접 참고할 만한 건 두 가지다.

1. **query proposal**
    - initial docs와 recent docs를 보고 다음 sub-query를 다시 쓰는 방식
2. **document recoverability**
    - 초반 good docs를 later drift가 있어도 final pool에서 안 잃어버리는 구조

반대로 직접 제공하지 않는 것은 아래다.

- branch-level state return
- tree-constrained retrieval pool control
- archived frontier selection

즉 이 work는 우리 쪽으로 치면:

- `query proposal`에는 강한 reference
- `which archived branch should be reopened?`에는 직접 답이 없음

## References

- BRIGHT submission note: <https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/submissions/bright_agentic.md>
- Agentic pipeline code: <https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/src/retrieval_bench/pipelines/agentic.py>
- Agent loop: <https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/src/retrieval_bench/nemo_agentic/agent.py>
- Prompt: <https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/src/retrieval_bench/nemo_agentic/prompts/02_v1.j2>
- Tool helpers: <https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/src/retrieval_bench/nemo_agentic/tool_helpers.py>
- Backend init / BRIGHT task prefixes: <https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/src/retrieval_bench/pipelines/backends.py>
- BRIGHT instructions: <https://github.com/NVIDIA/NeMo-Retriever/blob/main/retrieval-bench/src/retrieval_bench/prompts/bright_instructions.py>
