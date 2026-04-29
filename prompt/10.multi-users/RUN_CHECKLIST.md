# 멀티유저 멀티세션 RAG 실행 체크리스트

## 1) 로컬 실행
- Python 가상환경 생성/활성화 후 `pip install -r requirements.txt`
- Supabase SQL Editor에서 `supabase_multi_user_setup.sql` 전체 실행
- Supabase Authentication에서 Email Provider 활성화
- 필요 시 Confirm email 켜기/끄기 설정
- 프로젝트 루트 `.env`에 다음 키 설정
  - `SUPABASE_URL`
  - `SUPABASE_ANON_KEY` (권장)
  - `SUPABASE_EMAIL_REDIRECT_URL` (비번 재설정/이메일 확인 링크용)
- 앱 실행: `streamlit run multi-users-ref.py`
- 앱 사이드바에서 OpenAI/Anthropic/Gemini API Key 입력
- 회원가입 페이지(`pages/회원가입.py`)에서 가입 후 로그인 테스트

## 2) Streamlit Cloud 배포
- GitHub에 `prompt/10.multi-users` 폴더 포함 커밋/푸시
- Streamlit Cloud 앱의 Entry point를 `prompt/10.multi-users/multi-users-ref.py`로 지정
- App Settings -> Secrets에 등록
  - `SUPABASE_URL`
  - `SUPABASE_ANON_KEY`
  - `SUPABASE_EMAIL_REDIRECT_URL`
- Supabase Authentication -> URL Configuration
  - Site URL: Streamlit 앱 URL
  - Redirect URLs: Streamlit 앱 URL 및 `SUPABASE_EMAIL_REDIRECT_URL` 값

## 3) 기능 검증 순서
- 회원가입 -> 확인 메일 클릭 -> 로그인
- PDF 업로드 -> 파일 처리 -> 질문/답변 스트리밍 확인
- 세션저장 -> 세션로드 -> 세션삭제 -> 화면초기화 버튼 검증
- `vectordb` 버튼으로 현재 세션 파일명 노출 확인
- 앱 재시작 후 이전 세션/대화 복원 확인

## 4) 주의 사항
- `SUPABASE_SERVICE_ROLE_KEY`는 서버 백엔드 전용이며 공개 앱에서는 사용 지양
- 현재 앱은 `SUPABASE_ANON_KEY` + 사용자 로그인(RLS) 기준으로 동작
- 관리자 즉시 승인 계정은 앱 로직 상 특별 처리
  - ID: `donchang0725@gmail.com`
  - PW: `0000`
