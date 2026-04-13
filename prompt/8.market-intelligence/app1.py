"""
삼성전자 Global Market Intelligence Dashboard
삼성전자 브랜드 가이드라인(Navy & White)을 적용한 실시간 시장 인텔리전스 대시보드
"""

import os
import streamlit as st
import pandas as pd
import json
import re
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from typing import Any

# SSL 인증서 검증 문제 해결을 위한 전역 설정
os.environ["PYTHONHTTPSVERIFY"] = "0"
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
except AttributeError:
    pass

# 환경 변수 로드 (.env 파일을 프로젝트 루트에서 찾기)
current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
env_path = os.path.join(project_root, ".env")

if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

# OpenAI API 키 로드 함수 (ref.py 스타일)
def get_openai_api_key() -> str:
    """OpenAI API 키를 .env 파일에서 명시적으로 로드합니다."""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    env_path = os.path.join(project_root, ".env")
    
    if not os.path.exists(env_path):
        env_path = os.path.join(os.getcwd(), ".env")
    
    if os.path.exists(env_path):
        from dotenv import dotenv_values
        env_vars = dotenv_values(env_path)
        api_key = env_vars.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        return None
    
    return api_key.strip()

# ref.py 유틸: 구분선/취소선 제거
def remove_separators(text: str) -> str:
    """답변에서 구분선(---, ===, ___)과 취소선(~~텍스트~~)을 제거합니다."""
    if not text:
        return text
    text = re.sub(r'~~([^~]+)~~', r'\1', text)
    text = re.sub(r'\n\s*-{3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*={3,}\s*\n', '\n\n', text)
    text = re.sub(r'\n\s*_{3,}\s*\n', '\n\n', text)
    text = re.sub(r'^\s*-{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*={3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*_{3,}\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

# 삼성전자 브랜드 컬러 (Navy & White)
SAMSUNG_NAVY = "#004C97"
SAMSUNG_NAVY_DARK = "#002D5C"
SAMSUNG_NAVY_LIGHT = "#0066CC"
SAMSUNG_WHITE = "#FFFFFF"
SAMSUNG_GRAY = "#F5F5F5"

# 페이지 설정
st.set_page_config(
    page_title="삼성전자 Global Market Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS - 삼성 브랜드 컬러 적용 (ref.py 스타일 참고)
st.markdown(f"""
<style>
    /* 배경 */
    .main {{
        background: linear-gradient(135deg, {SAMSUNG_NAVY_DARK} 0%, {SAMSUNG_NAVY} 100%);
    }}
    .stApp {{
        background: linear-gradient(135deg, {SAMSUNG_NAVY_DARK} 0%, {SAMSUNG_NAVY} 100%);
    }}
    .block-container {{
        background-color: rgba(255, 255, 255, 0.05);
        padding-top: 2rem;
        border-radius: 10px;
    }}
    header {{
        background-color: {SAMSUNG_NAVY_DARK} !important;
    }}
    .stSidebar {{
        background-color: {SAMSUNG_NAVY_DARK} !important;
    }}
    
    /* 모든 텍스트를 흰색으로 */
    h1, h2, h3, h4, h5, h6 {{
        color: {SAMSUNG_WHITE} !important;
        font-weight: bold;
    }}
    h1 {{
        text-align: center;
        padding: 20px 0;
    }}
    
    /* 일반 텍스트 */
    p, span, div, label, .stMarkdown {{
        color: {SAMSUNG_WHITE} !important;
    }}
    
    /* 버튼 */
    .stButton>button {{
        background-color: {SAMSUNG_NAVY} !important;
        color: {SAMSUNG_WHITE} !important;
        border: 2px solid {SAMSUNG_NAVY_LIGHT} !important;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        width: 100%;
        margin: 5px 0;
    }}
    .stButton>button:hover {{
        background-color: {SAMSUNG_NAVY_LIGHT} !important;
        border-color: {SAMSUNG_WHITE} !important;
    }}
    
    /* 테이블 */
    .stDataFrame {{
        background-color: rgba(0, 0, 0, 0.3) !important;
    }}
    table {{
        color: {SAMSUNG_WHITE} !important;
    }}
    thead {{
        background-color: {SAMSUNG_NAVY} !important;
        color: {SAMSUNG_WHITE} !important;
    }}
    tbody {{
        color: {SAMSUNG_WHITE} !important;
    }}
    
    /* Info, Success, Warning, Error 박스 */
    .stInfo, .stSuccess, .stWarning, .stError {{
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: {SAMSUNG_WHITE} !important;
        border-left: 4px solid {SAMSUNG_NAVY_LIGHT} !important;
    }}
    
    /* 입력 필드 */
    .stTextInput>div>div>input {{
        background-color: rgba(0, 0, 0, 0.3) !important;
        color: {SAMSUNG_WHITE} !important;
        border: 1px solid {SAMSUNG_NAVY_LIGHT} !important;
    }}
    
    /* 채팅 메시지 스타일 (ref.py 스타일) */
    .stChatMessage {{
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }}
    .stChatMessage p {{
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        margin: 0.5rem 0 !important;
    }}
    .stChatMessage ul, .stChatMessage ol {{
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        margin: 0.5rem 0 !important;
    }}
    .stChatMessage li {{
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
        margin: 0.3rem 0 !important;
    }}
</style>
""", unsafe_allow_html=True)

# 세션 상태 초기화
if "reviews_data" not in st.session_state:
    st.session_state.reviews_data = pd.DataFrame()
if "search_completed" not in st.session_state:
    st.session_state.search_completed = False
if "analysis_completed" not in st.session_state:
    st.session_state.analysis_completed = False
if "strategic_report" not in st.session_state:
    st.session_state.strategic_report = None
if "research_topic" not in st.session_state:
    st.session_state.research_topic = ""
if "sentiment_summaries" not in st.session_state:
    st.session_state.sentiment_summaries = None

# ============================================================================
# 웹 검색 및 데이터 수집 함수 (gpt-5.2 web_search tool 사용)
# ============================================================================

def search_with_web_search(prompt: str, temperature: float = 0.7) -> str:
    """OpenAI Responses API를 사용하여 web_search를 수행합니다. (ref.py 스타일)"""
    api_key = get_openai_api_key()
    if not api_key:
        return None, "OPENAI_API_KEY가 .env 파일에 설정되어 있지 않습니다. 프로젝트 루트 디렉토리의 .env 파일에 OPENAI_API_KEY를 추가해주세요."
    
    try:
        # OpenAI 클라이언트 초기화
        client = OpenAI(api_key=api_key)
        
        # Responses API (2026년 표준) 호출
        # Built-in web_search 도구를 사용하여 OpenAI 서버에서 검색을 대행함
        response = client.responses.create(
            model="gpt-5.2",
            input=prompt,
            tools=[
                {
                    "type": "web_search",  # OpenAI 내장 검색 도구
                    "search_context_size": "high"  # 상세한 검색 결과 참조
                }
            ]
        )
        
        # 결과 처리
        answer = response.output_text
        return answer, None
        
    except Exception as e:
        return None, f"web_search 검색 중 오류: {e}"

def search_global_media_reviews(research_topic, progress_bar=None, status_text=None):
    """gpt-5.2 web_search를 사용하여 글로벌 미디어 리뷰 검색"""
    try:
        api_key = get_openai_api_key()
        if not api_key:
            return None, "OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다."
        
        if progress_bar:
            progress_bar.progress(0.1)
        if status_text:
            status_text.text("🔍 검색 대상 사이트 확인 중...")
        
        # 검색 대상 사이트를 명시한 상세한 쿼리
        query = f"""다음 주제에 대한 최신 리뷰, 기사, 토론을 검색하고 분석해주세요:

주제: {research_topic}

검색 대상 사이트 (Target Search Sites):
- 글로벌 IT 전문 매체: The Verge, CNET, Wired, TechRadar, Tom's Guide
- 커뮤니티 및 SNS: Reddit (r/Samsung, r/Android), X (Twitter) 트렌드, YouTube Tech Influencers (MKBHD, Mrwhosetheboss 등의 영상 제목 및 댓글 반응)
- 글로벌 지역별 뉴스: Reuters (경제적 파급력), 동남아/유럽 현지 테크 블로그

각 소스에서 다음 정보를 추출하여 JSON 배열 형식으로 정확히 10개의 리뷰를 제공해주세요:

각 리뷰 항목에 포함할 정보:
1. 매체명/소스명 (예: The Verge, CNET, Reddit, YouTube, Twitter 등)
2. 국가/지역 (예: USA, UK, South Korea, Southeast Asia, Europe)
3. 리뷰 날짜 (가능한 경우, 없으면 "Recent")
4. 긍정 요소(Pros) - 혁신성(폴딩 메커니즘), 화면 크기 대비 휴대성, 멀티태스킹 성능 등 (배열 형식)
5. 부정 요소(Cons) - 내구성 우려(주름, 힌지), 가격 저항선, 배터리 효율, 무게/두께 등 (배열 형식)
6. 비교 분석: 기존 갤럭시 폴드 시리즈 및 경쟁사(화웨이 등) 모델과의 차별점
7. 전략적 인사이트: 마케팅 소구점, 차기 모델 수정 제안
8. Website URL: 각 리뷰/기사의 원본 URL

중요: 정확히 10개의 리뷰를 다양한 소스에서 수집하여 JSON 배열 형식으로 반환해주세요. 추가 텍스트 없이 JSON 배열만 반환해주세요.

형식:
[
    {{
        "media": "매체명",
        "country": "국가/지역",
        "date": "리뷰 날짜 또는 Recent",
        "pros": ["장점1", "장점2", "장점3"],
        "cons": ["단점1", "단점2", "단점3"],
        "comparison": "경쟁사 비교 분석",
        "insights": "전략적 인사이트",
        "website_url": "https://example.com/review"
    }},
    ...
]
"""
        
        if progress_bar:
            progress_bar.progress(0.3)
        if status_text:
            status_text.text("🌐 글로벌 미디어 검색 중...")
        
        response_text, error = search_with_web_search(query)
        
        if error:
            if progress_bar:
                progress_bar.progress(1.0)
            if status_text:
                status_text.text("❌ 오류 발생")
            return None, error
        
        if progress_bar:
            progress_bar.progress(0.7)
        if status_text:
            status_text.text("📊 데이터 정리 중...")
        
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("✅ 데이터 수집 완료!")
        
        return response_text, None
        
    except Exception as e:
        if progress_bar:
            progress_bar.progress(1.0)
        if status_text:
            status_text.text("❌ 오류 발생")
        return None, f"검색 중 오류 발생: {str(e)}"

def parse_reviews_from_response(response_text):
    """LLM 응답에서 리뷰 데이터 추출 및 DataFrame 생성"""
    reviews = []
    
    # JSON 형식 추출 시도
    try:
        # JSON 배열 찾기
        json_match = re.search(r'\[.*\]', response_text, re.DOTALL)
        if json_match:
            reviews_data = json.loads(json_match.group())
            for review in reviews_data:
                reviews.append({
                    "매체명": review.get("media", "Unknown"),
                    "국가": review.get("country", "Unknown"),
                    "리뷰 날짜": review.get("date", "Recent"),
                    "긍정 요소": "; ".join(review.get("pros", [])) if isinstance(review.get("pros"), list) else str(review.get("pros", "")),
                    "부정 요소": "; ".join(review.get("cons", [])) if isinstance(review.get("cons"), list) else str(review.get("cons", "")),
                    "비교 분석": review.get("comparison", ""),
                    "전략적 인사이트": review.get("insights", ""),
                    "Website URL": review.get("website_url", "")
                })
    except Exception as e:
        st.warning(f"JSON 파싱 시도 중 오류: {e}")
    
    # JSON 파싱 실패 시 텍스트에서 정보 추출 시도
    if not reviews or len(reviews) < 10:
        # LLM에게 다시 요청하여 JSON 형식으로 변환
        try:
            api_key = get_openai_api_key()
            if api_key:
                client = OpenAI(api_key=api_key)
                llm = ChatOpenAI(model="gpt-4o", temperature=0.3, api_key=api_key)
                
                parse_prompt = f"""다음 텍스트에서 리뷰 정보를 추출하여 JSON 배열 형식으로 변환해주세요. 정확히 10개의 리뷰를 생성해주세요.

텍스트:
{response_text[:4000]}

JSON 배열 형식:
[
    {{
        "media": "매체명",
        "country": "국가/지역",
        "date": "리뷰 날짜 또는 Recent",
        "pros": ["장점1", "장점2"],
        "cons": ["단점1", "단점2"],
        "comparison": "비교 분석",
        "insights": "전략적 인사이트",
        "website_url": "URL"
    }}
]

추가 텍스트 없이 JSON 배열만 반환해주세요."""
                
                parsed_response = llm.invoke([HumanMessage(content=parse_prompt)])
                parsed_text = parsed_response.content if hasattr(parsed_response, "content") else str(parsed_response)
                
                # 다시 JSON 파싱 시도
                json_match = re.search(r'\[.*\]', parsed_text, re.DOTALL)
                if json_match:
                    reviews_data = json.loads(json_match.group())
                    reviews = []
                    for review in reviews_data:
                        reviews.append({
                            "매체명": review.get("media", "Unknown"),
                            "국가": review.get("country", "Unknown"),
                            "리뷰 날짜": review.get("date", "Recent"),
                            "긍정 요소": "; ".join(review.get("pros", [])) if isinstance(review.get("pros"), list) else str(review.get("pros", "")),
                            "부정 요소": "; ".join(review.get("cons", [])) if isinstance(review.get("cons"), list) else str(review.get("cons", "")),
                            "비교 분석": review.get("comparison", ""),
                            "전략적 인사이트": review.get("insights", ""),
                            "Website URL": review.get("website_url", "")
                        })
        except Exception as e:
            st.warning(f"텍스트 파싱 시도 중 오류: {e}")
    
    # 최소 10개가 되도록 보완
    if len(reviews) < 10:
        # 부족한 개수만큼 샘플 데이터 추가 (실제 데이터와 구분되도록)
        needed = 10 - len(reviews)
        for i in range(needed):
            reviews.append({
                "매체명": f"Additional Source {i+1}",
                "국가": "Global",
                "리뷰 날짜": "Recent",
                "긍정 요소": "Data collection in progress",
                "부정 요소": "Data collection in progress",
                "비교 분석": "Data collection in progress",
                "전략적 인사이트": "Data collection in progress",
                "Website URL": ""
            })
    
    df = pd.DataFrame(reviews[:10])  # 정확히 10개만
    
    # 필수 컬럼 확인 및 기본값 설정
    required_columns = ["매체명", "국가", "리뷰 날짜", "긍정 요소", "부정 요소", "비교 분석", "전략적 인사이트", "Website URL"]
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""
    
    return df

# ============================================================================
# 감성 분석 및 시각화 함수
# ============================================================================

def generate_sentiment_summary(df, research_topic):
    """LLM을 사용하여 감성 분석 요약 생성 (출처 명시)"""
    try:
        openai_api_key = get_openai_api_key()
        if not openai_api_key:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            env_path = os.path.join(project_root, ".env")
            
            error_msg = f"""❌ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.

**해결 방법:**
1. 프로젝트 루트 디렉토리(`{project_root}`)에 `.env` 파일이 있는지 확인하세요.
2. `.env` 파일에 다음 형식으로 API 키를 추가하세요:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
3. Streamlit 앱을 재시작하세요.
"""
            st.error(error_msg)
            return None, None, None, None, None
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=openai_api_key
        )
        
        pros_text = " ".join([f"[{row['매체명']}] {row['긍정 요소']}" for _, row in df.iterrows()])
        cons_text = " ".join([f"[{row['매체명']}] {row['부정 요소']}" for _, row in df.iterrows()])
        
        # 내구성 관련 텍스트 추출 (매체명 포함)
        durability_keywords = ["durability", "durable", "fragile", "내구성", "견고", "취약", "crease", "hinge", "screen", "display", "fold", "주름", "힌지"]
        durability_texts = []
        for idx, row in df.iterrows():
            cons = str(row.get("부정 요소", ""))
            pros = str(row.get("긍정 요소", ""))
            media = str(row.get("매체명", "Unknown"))
            combined = (cons + " " + pros).lower()
            if any(keyword.lower() in combined for keyword in durability_keywords):
                durability_texts.append(f"[{media}] {cons} {pros}")
        durability_text = " ".join(durability_texts[:500])
        
        # 가격 관련 텍스트 추출 (매체명 포함)
        price_keywords = ["price", "expensive", "cost", "가격", "비싸", "비용", "affordability", "value", "worth", "저항선"]
        price_texts = []
        for idx, row in df.iterrows():
            cons = str(row.get("부정 요소", ""))
            pros = str(row.get("긍정 요소", ""))
            media = str(row.get("매체명", "Unknown"))
            combined = (cons + " " + pros).lower()
            if any(keyword.lower() in combined for keyword in price_keywords):
                price_texts.append(f"[{media}] {cons} {pros}")
        price_text = " ".join(price_texts[:500])
        
        # 지역별 데이터 준비
        regional_data = {}
        if "국가" in df.columns:
            for country in df["국가"].unique():
                country_df = df[df["국가"] == country]
                if not country_df.empty:
                    regional_data[country] = {
                        "긍정": " ".join([f"[{row['매체명']}] {row['긍정 요소']}" for _, row in country_df.iterrows()][:300]),
                        "부정": " ".join([f"[{row['매체명']}] {row['부정 요소']}" for _, row in country_df.iterrows()][:300])
                    }
        
        # 1. 장점 요약 (출처 명시)
        pros_prompt = f"""다음은 {research_topic}에 대한 글로벌 리뷰에서 언급된 주요 장점들입니다:

{pros_text[:2000]}

위 내용을 바탕으로 3-5개의 주요 장점을 간결하고 명확하게 요약해주세요. 각 장점은 1-2문장으로 설명하되, 구체적인 근거를 포함해주세요. 각 평가마다 출처(매체명)를 명시해주세요. 한국어로 답변해주세요."""
        
        # 2. 단점 요약 (출처 명시)
        cons_prompt = f"""다음은 {research_topic}에 대한 글로벌 리뷰에서 언급된 주요 단점들입니다:

{cons_text[:2000]}

위 내용을 바탕으로 3-5개의 주요 단점을 간결하고 명확하게 요약해주세요. 각 단점은 1-2문장으로 설명하되, 구체적인 근거를 포함해주세요. 각 평가마다 출처(매체명)를 명시해주세요. 한국어로 답변해주세요."""
        
        # 3. 내구성 요약 (출처 명시)
        durability_prompt = f"""다음은 {research_topic}에 대한 리뷰에서 내구성(durability, crease, hinge, screen 등)과 관련된 언급들입니다:

{durability_text[:1500] if durability_text else "내구성 관련 언급이 충분하지 않습니다."}

위 내용을 바탕으로 내구성에 대한 주요 우려사항과 긍정적 평가를 요약해주세요. 각 평가마다 출처(매체명)를 명시해주세요. 한국어로 답변해주세요."""
        
        # 4. 가격 요약 (출처 명시)
        price_prompt = f"""다음은 {research_topic}에 대한 리뷰에서 가격(price, expensive, cost, value 등)과 관련된 언급들입니다:

{price_text[:1500] if price_text else "가격 관련 언급이 충분하지 않습니다."}

위 내용을 바탕으로 가격에 대한 주요 의견과 평가를 요약해주세요. 각 평가마다 출처(매체명)를 명시해주세요. 한국어로 답변해주세요."""
        
        # 5. 지역별 반응 요약 (출처 명시)
        regional_prompt = f"""다음은 {research_topic}에 대한 지역별 반응 데이터입니다:

{json.dumps(regional_data, ensure_ascii=False, indent=2)[:2000]}

위 데이터를 바탕으로 지역별로 반응이 어떻게 다른지 요약해주세요. 각 지역의 특징적인 반응과 차이점을 명확히 설명해주세요. 각 평가마다 출처(매체명)를 명시해주세요. 한국어로 답변해주세요."""
        
        # 모든 요약 생성
        try:
            pros_summary = llm.invoke([HumanMessage(content=pros_prompt)]).content
        except Exception as e:
            pros_summary = None
            st.warning(f"장점 요약 생성 실패: {str(e)}")
        
        try:
            cons_summary = llm.invoke([HumanMessage(content=cons_prompt)]).content
        except Exception as e:
            cons_summary = None
            st.warning(f"단점 요약 생성 실패: {str(e)}")
        
        try:
            durability_summary = llm.invoke([HumanMessage(content=durability_prompt)]).content
        except Exception as e:
            durability_summary = None
            st.warning(f"내구성 요약 생성 실패: {str(e)}")
        
        try:
            price_summary = llm.invoke([HumanMessage(content=price_prompt)]).content
        except Exception as e:
            price_summary = None
            st.warning(f"가격 요약 생성 실패: {str(e)}")
        
        try:
            regional_summary = llm.invoke([HumanMessage(content=regional_prompt)]).content
        except Exception as e:
            regional_summary = None
            st.warning(f"지역별 요약 생성 실패: {str(e)}")
        
        return pros_summary, cons_summary, durability_summary, price_summary, regional_summary
        
    except Exception as e:
        error_msg = f"감성 분석 생성 중 오류 발생: {str(e)}"
        st.error(f"❌ {error_msg}")
        return f"오류: {error_msg}", None, None, None, None

# ============================================================================
# 전략 보고서 생성 함수
# ============================================================================

def generate_strategic_report(df, research_topic):
    """임원용 전략 보고서 자동 생성 (ref.py 스타일 적용)"""
    try:
        openai_api_key = get_openai_api_key()
        if not openai_api_key:
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
            env_path = os.path.join(project_root, ".env")
            
            error_msg = f"""❌ OPENAI_API_KEY가 .env 파일에 설정되지 않았습니다.

**해결 방법:**
1. 프로젝트 루트 디렉토리(`{project_root}`)에 `.env` 파일이 있는지 확인하세요.
2. `.env` 파일에 다음 형식으로 API 키를 추가하세요:
   ```
   OPENAI_API_KEY=sk-your-actual-api-key-here
   ```
3. Streamlit 앱을 재시작하세요.
"""
            return error_msg
        
        llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.7,
            api_key=openai_api_key
        )
        
        # 데이터 요약
        summary = f"""
        총 리뷰 수: {len(df)}
        국가별 분포: {df['국가'].value_counts().to_dict()}
        """
        
        pros_text = " ".join([f"[{row['매체명']}] {row['긍정 요소']}" for _, row in df.iterrows()])
        cons_text = " ".join([f"[{row['매체명']}] {row['부정 요소']}" for _, row in df.iterrows()])
        comparison_text = " ".join([f"[{row['매체명']}] {row['비교 분석']}" for _, row in df.iterrows()])
        insights_text = " ".join([f"[{row['매체명']}] {row['전략적 인사이트']}" for _, row in df.iterrows()])
        
        # 지역별 상세 분석 추가
        regional_summary = ""
        if "국가" in df.columns:
            for country in df["국가"].unique():
                country_df = df[df["국가"] == country]
                regional_summary += f"\n{country}: 리뷰 {len(country_df)}개"
        
        prompt = f"""당신은 삼성전자 DX 부문의 수석 전략 분석가입니다. 
다음은 {research_topic}에 대한 글로벌 런칭 초기 반응 데이터입니다:

=== 데이터 요약 ===
{summary}
{regional_summary}

=== 주요 장점 언급 ===
{pros_text[:2000]}

=== 주요 단점 언급 ===
{cons_text[:2000]}

=== 비교 분석 ===
{comparison_text[:1500]}

=== 전략적 인사이트 ===
{insights_text[:1500]}

=== 요청사항 ===
DX 부문 부사장급 이상 임원에게 보고할 '글로벌 런칭 초기 반응 보고서'를 다음 형식으로 작성해주세요.
각 섹션은 구체적이고 실행 가능한 인사이트를 포함해야 합니다.
답변 형식은 ref.py 스타일을 따라야 합니다:
- 답변은 반드시 제목과 본문으로 구분하여 작성하세요
- 제목(# H1)은 질문의 핵심을 짧고 명확하게 요약한 한 문장으로 작성하세요
- 제목 다음에 빈 줄을 하나 두고 본문을 작성하세요
- 본문은 ## (H2)와 ### (H3) 헤딩을 사용하여 구조화하세요
- 본문은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요
- 답변 중간에 구분선(---, ===, ___)을 사용하지 마세요
- 취소선(~~텍스트~~)을 사용하지 마세요

# {research_topic} 글로벌 런칭 초기 반응 보고서

## 1. 시장의 열광 포인트
(3-5개 주요 포인트를 구체적으로 제시하며, 각 포인트에 대해 데이터 기반 근거를 포함)

## 2. 즉시 대응이 필요한 비판적 여론
(3-5개 주요 이슈를 우선순위별로 제시하고, 각 이슈에 대한 즉시 대응 방안 포함)

## 3. 경쟁사 대비 우위 요소
(3-5개 차별화 포인트를 명확히 제시하며, 시장에서의 경쟁력 강화 방안 포함)

## 4. 마케팅팀 전달사항: 핵심 광고 카피 3개
(각 카피는 1-2문장으로 구성하며, 감성적이고 설득력 있게 작성. 타겟 고객층별로 차별화)

## 5. DX 부문 임원 Follow-up 이슈 3개
(임원이 즉시 검토하고 대응해야 할 3가지 핵심 이슈를 우선순위별로 제시)

보고서는 전문적이고 실행 가능한 인사이트를 포함해야 하며, 데이터 기반의 객관적인 분석을 제공해야 합니다."""
        
        response = llm.invoke([HumanMessage(content=prompt)])
        cleaned = remove_separators(response.content if hasattr(response, "content") else str(response))
        return cleaned
        
    except Exception as e:
        return f"보고서 생성 중 오류 발생: {str(e)}"

# ============================================================================
# 메인 대시보드 UI
# ============================================================================

def main():
    # 제목
    st.markdown(f"<h1 style='color: {SAMSUNG_WHITE}; text-align: center; padding: 20px 0;'>삼성전자 Global Market Intelligence</h1>", unsafe_allow_html=True)
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 마켓 리서치 제목 입력 필드
        st.subheader("🔍 마켓 리서치 제목")
        research_topic = st.text_input(
            "마켓 리서치 제목을 입력하세요",
            value=st.session_state.research_topic,
            placeholder="예: Samsung Galaxy Trifold, Galaxy S24 Ultra, etc.",
            key="research_topic_input"
        )
        
        # 제목이 변경되면 세션 상태 업데이트
        if research_topic and research_topic != st.session_state.research_topic:
            st.session_state.research_topic = research_topic
            # 제목이 변경되면 기존 데이터 초기화
            if st.session_state.search_completed:
                st.session_state.reviews_data = pd.DataFrame()
                st.session_state.search_completed = False
                st.session_state.analysis_completed = False
                st.session_state.strategic_report = None
                st.session_state.sentiment_summaries = None
        
        st.markdown("---")
        
        # 기능 버튼들
        st.subheader("📊 기능")
        
        # 1. 검색과 데이터 수집 버튼
        if st.button("🔍 검색과 데이터 수집", type="primary"):
            if not research_topic or research_topic.strip() == "":
                st.error("❌ 마켓 리서치 제목을 입력해주세요.")
            else:
                # Progress bar 및 상태 텍스트 생성
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                response_text, error = search_global_media_reviews(research_topic, progress_bar, status_text)
                
                if error:
                    st.error(error)
                else:
                    df = parse_reviews_from_response(response_text)
                    st.session_state.reviews_data = df
                    st.session_state.search_completed = True
                    st.success(f"✅ {len(df)}개 리뷰 수집 완료!")
                    # Progress bar 및 상태 텍스트 제거
                    progress_bar.empty()
                    status_text.empty()
                    st.rerun()
        
        # 2. 감성 분석 버튼
        if st.session_state.search_completed and not st.session_state.analysis_completed:
            if st.button("📊 감성 분석", type="primary"):
                with st.spinner("감성 분석 및 요약 생성 중... (약 20초 소요)"):
                    df = st.session_state.reviews_data
                    summaries = generate_sentiment_summary(df, st.session_state.research_topic)
                    st.session_state.sentiment_summaries = summaries
                    st.session_state.analysis_completed = True
                    st.rerun()
        
        # 3. 임원보고서 버튼
        if st.session_state.analysis_completed and not st.session_state.strategic_report:
            if st.button("📋 임원보고서", type="primary"):
                if not st.session_state.reviews_data.empty:
                    with st.spinner("임원용 전략 보고서 생성 중... (약 30초 소요)"):
                        df = st.session_state.reviews_data
                        report = generate_strategic_report(df, st.session_state.research_topic)
                        st.session_state.strategic_report = report
                        st.rerun()
        
        # 데이터 초기화
        st.markdown("---")
        if st.button("🔄 데이터 초기화"):
            st.session_state.reviews_data = pd.DataFrame()
            st.session_state.search_completed = False
            st.session_state.analysis_completed = False
            st.session_state.strategic_report = None
            st.session_state.sentiment_summaries = None
            st.rerun()
    
    # 메인 컨텐츠
    if st.session_state.reviews_data.empty:
        st.info("👈 사이드바에서 '검색과 데이터 수집' 버튼을 클릭하여 데이터를 수집하세요.")
    else:
        df = st.session_state.reviews_data
        
        # ====================================================================
        # Step 1: 리뷰 데이터 테이블
        # ====================================================================
        st.markdown("---")
        st.subheader("📰 글로벌 미디어 리뷰 데이터")
        
        display_columns = ["매체명", "국가", "리뷰 날짜", "긍정 요소", "부정 요소", "비교 분석", "전략적 인사이트", "Website URL"]
        st.dataframe(
            df[display_columns],
            hide_index=True,
            width='stretch'
        )
        
        st.markdown("---")
        
        # ====================================================================
        # Step 2: 감성 분석 및 시각화
        # ====================================================================
        if st.session_state.analysis_completed:
            st.subheader("📈 감성 분석 및 인사이트")
            
            # 텍스트 요약 표시
            if st.session_state.sentiment_summaries:
                pros_summary, cons_summary, durability_summary, price_summary, regional_summary = st.session_state.sentiment_summaries
                
                # 모든 요약이 None인지 확인
                all_none = all(s is None for s in [pros_summary, cons_summary, durability_summary, price_summary, regional_summary])
                
                if all_none:
                    st.warning("⚠️ 감성 분석 요약이 생성되지 않았습니다. OPENAI_API_KEY가 .env 파일에 올바르게 설정되어 있는지 확인해주세요.")
                else:
                    # 장점 요약
                    if pros_summary and not pros_summary.startswith("오류:"):
                        st.markdown("### ✅ 주요 장점 요약")
                        st.markdown(pros_summary)
                        st.markdown("---")
                    elif pros_summary and pros_summary.startswith("오류:"):
                        st.error(pros_summary)
                        st.markdown("---")
                    
                    # 단점 요약
                    if cons_summary:
                        st.markdown("### ❌ 주요 단점 요약")
                        st.markdown(cons_summary)
                        st.markdown("---")
                    
                    # 내구성 및 가격 요약
                    summary_col1, summary_col2 = st.columns(2)
                    
                    with summary_col1:
                        if price_summary:
                            st.markdown("#### 💰 가격 관련 언급 요약")
                            st.markdown(price_summary)
                    
                    with summary_col2:
                        if durability_summary:
                            st.markdown("#### 🛡️ 내구성 관련 언급 요약")
                            st.markdown(durability_summary)
                    
                    st.markdown("---")
                    
                    # 지역별 반응 요약
                    if regional_summary:
                        st.markdown("### 🌍 지역별 반응 차이 요약")
                        st.markdown(regional_summary)
                        st.markdown("---")
            else:
                st.warning("⚠️ 감성 분석 요약 데이터가 없습니다. '감성 분석' 버튼을 다시 클릭해주세요.")
            
            st.markdown("---")
            
            # ====================================================================
            # Step 3: 임원용 전략 보고서 (메인 화면)
            # ====================================================================
            if st.session_state.strategic_report:
                st.subheader("📋 임원용 전략 보고서")
                
                # 보고서 내용 표시
                st.markdown(st.session_state.strategic_report)
                
                st.markdown("---")
                
                # 보고서 다운로드 버튼
                report_text = st.session_state.strategic_report
                topic_safe = st.session_state.get("research_topic", "product").replace(" ", "_")
                st.download_button(
                    label="📥 보고서 다운로드 (TXT)",
                    data=report_text,
                    file_name=f"{topic_safe}_strategic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )
        else:
            st.info("👈 사이드바에서 '감성 분석' 버튼을 클릭하여 분석을 시작하세요.")

if __name__ == "__main__":
    main()
