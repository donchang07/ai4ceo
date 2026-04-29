# 신규 설치용 DB 초기 생성 가이드

이 가이드는 **기존 데이터가 없는 신규 프로젝트** 기준입니다.

## 1) Supabase 새 프로젝트 생성
- Supabase에서 새 프로젝트를 만듭니다.
- 프로젝트 생성 완료 후 `Project Settings -> API`에서 아래 값을 확인합니다.
  - `SUPABASE_URL`
  - `SUPABASE_ANON_KEY`

## 2) DB 스키마 최초 생성 (처음부터)
- Supabase `SQL Editor`를 엽니다.
- `prompt/10.multi-users/supabase_multi_user_setup.sql` 파일 **전체**를 붙여넣고 실행합니다.
- 이 스크립트는 `sessions/messages/documents` 및 `match_documents` RPC를 생성합니다.

## 3) Auth 설정
- `Authentication -> Providers -> Email`에서 Email 로그인 활성화
- 확인 메일을 사용할 경우 `Confirm email`을 ON
- `Authentication -> URL Configuration` 설정
  - `Site URL`: 배포 앱 URL
  - `Redirect URLs`: 배포 앱 URL, 그리고 `SUPABASE_EMAIL_REDIRECT_URL` 값

## 4) 앱 환경변수
- 로컬 `.env` 또는 Streamlit Cloud Secrets에 다음 키를 설정합니다.
  - `SUPABASE_URL`
  - `SUPABASE_ANON_KEY`
  - `SUPABASE_EMAIL_REDIRECT_URL`

## 5) 앱 실행 및 동작 확인
- 실행: `streamlit run prompt/10.multi-users/multi-users-ref.py`
- 앱 사이드바에서 OpenAI/Anthropic/Gemini API Key 입력
- `회원가입 -> 로그인 -> PDF 업로드 -> 파일 처리 -> 질문`
- `세션저장/세션로드/세션삭제/화면초기화/vectordb` 버튼 동작 확인

## 참고
- 신규 설치에서는 `migration_multi_user_from_multi_session.sql`을 실행하지 않습니다.
- 관리자 즉시 승인 계정은 앱 코드에 이미 반영되어 있습니다.
