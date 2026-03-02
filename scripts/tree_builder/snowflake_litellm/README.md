# Snowflake Custom Provider

LiteLLM을 위한 Snowflake Cortex AI custom provider (SQL 기반)

## 📁 디렉토리 구조

```
litellm_ollama_manager/providers/snowflake/
├── __init__.py           # 패키지 초기화
├── handler.py            # Snowflake Cortex AI 핸들러 (SQL 기반)
├── config.py             # 설정 관리 클래스들
├── utils.py              # 유틸리티 함수들
├── env.example           # 환경변수 설정 예제
└── README.md            # 이 파일
```

## 🚀 핵심 특징

- `snowflake-connector-python` 사용
- `SNOWFLAKE.CORTEX.COMPLETE()` SQL 함수 호출
- 패스워드 인증 기반 (JWT 불필요)

## 🔧 설정 방법

### 1. providers.yaml 설정

```yaml
# ============================================
# Snowflake Cortex AI (SQL 방식)
# ============================================
snowflake:
  accounts:
    - account: your-account-id
      user: your-username
      password: ${SNOWFLAKE_PASSWORD} # .env에서 로드
      warehouse: COMPUTE_WH
      database: SNOWFLAKE_SAMPLE_DATA
      schema: PUBLIC
      role: ACCOUNTADMIN # 선택사항
      enabled: true
      models:
        - name: mistral-large
          cortex_model: mistral-large
          enabled: true
        - name: llama3-8b
          cortex_model: llama3-8b
          enabled: true
```

### 2. .env 파일 설정

```bash
# Snowflake 계정 비밀번호
SNOWFLAKE_PASSWORD=your_password_here
```

### 3. litellm_config.yaml 자동 생성

```bash
# providers.yaml 동기화
lom sync-config

# 생성된 설정 확인
cat litellm_config.yaml
```

자동 생성되는 내용:

```yaml
litellm_settings:
  custom_provider_map:
    - provider: "snowflake-cortex"
      custom_handler: "litellm_ollama_manager.providers.snowflake.handler.snowflake_cortex_llm"

  environment_variables:
    # JSON 형식으로 계정 정보 전달
    SNOWFLAKE_ACCOUNTS: '[{"account":"...","username":"...","password":"...","warehouse":"COMPUTE_WH","database":"...","schema":"..."}]'
    SNOWFLAKE_MODELS: '[{"model_name":"mistral-large","account":"..."}]'

model_list:
  - model_name: mistral-large
    litellm_params:
      model: custom_provider-snowflake-cortex
      custom_llm_provider: snowflake-cortex
```

## 📋 지원 기능

### SQL 함수 사용

```python
# handler.py 내부 동작 (바인딩으로 JSON 파싱 오류 방지)
sql = """
SELECT SNOWFLAKE.CORTEX.COMPLETE(
  %s,
  PARSE_JSON(%s),
  PARSE_JSON(%s)
) as response
"""
params = (model, messages_json, options_json)
```

### 지원 파라미터

- `model`: Cortex 모델명 (mistral-large, llama3-8b 등)
- `messages`: OpenAI 호환 메시지 배열
- `temperature`: 샘플링 온도 (선택)
- `max_tokens`: 최대 토큰 수 (선택)
- `top_p`: Top-p 샘플링 (선택)

### 에러 처리

- `ProgrammingError` → `InvalidRequestError` (SQL 문법 오류)
- `DatabaseError` → `InternalServerError` (DB 오류)
- Trial 종료 감지 → 계정 자동 정지

## 🧪 테스트

### Python 직접 호출

```python
from litellm_ollama_manager.providers.snowflake.handler import SnowflakeCortexLLM

handler = SnowflakeCortexLLM()
response = handler.completion(
    model="mistral-large",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response)
```

### LiteLLM 테스트

```bash
# 헬스 체크
lom health-check

# 직접 테스트
curl -X POST http://localhost:13467/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-large",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

## 📝 주의사항

1. **패스워드 보안**:
   - `.env` 파일에 저장하고 git에 커밋하지 않기
   - 환경변수로 전달: `SNOWFLAKE_PASSWORD`

2. **Warehouse 비용**:
   - SQL 실행 시 warehouse 자동 시작
   - 비용 발생 주의

3. **모델 가용성**:
   - 계정별로 사용 가능한 모델이 다를 수 있음
   - `lom validate-models` 로 확인

4. **스트리밍**:
   - 현재는 pseudo-streaming (완료 후 한번에 반환)
   - 향후 `snowflake.cortex.Complete(stream=True)` 지원 예정

## 🔗 참고 문서

- [Snowflake Cortex COMPLETE Function](https://docs.snowflake.com/en/sql-reference/functions/complete-snowflake-cortex)
- [Snowflake Python Connector](https://docs.snowflake.com/en/developer-guide/python-connector/python-connector)
- [Cortex AI Functions](https://docs.snowflake.com/en/user-guide/snowflake-cortex/aisql)
