"""
특정 주제에 대한 딥 리서치 시스템
- Agent 1: 자료 수집 (인터넷 + Arxiv)
- Agent 2: 반대 의견 및 다른 접근 방식 찾기
- Agent 3: 양쪽 결과 종합, 장단점 논의, 결론 및 향후 전망
"""

import os
import streamlit as st
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import ArxivLoader
from langchain_core.messages import HumanMessage, SystemMessage

# Perplexity import
try:
    from langchain_perplexity import ChatPerplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
    st.warning("langchain-perplexity가 설치되지 않았습니다. pip install langchain-perplexity 로 설치해주세요.")

# 환경 변수 로드
load_dotenv()

# ==================== 모델 초기화 함수 ====================
def get_llm_model(model_name: str, temperature: float = 0.3):
    """선택된 모델 이름에 따라 LLM 인스턴스를 반환합니다."""
    if model_name == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    elif model_name == "gpt-5":
        # gpt-5는 아직 출시되지 않았으므로 gpt-4o로 매핑
        return ChatOpenAI(model="gpt-5", temperature=temperature)
    elif model_name == "gemini-2.5-pro":
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_api_key:
            return ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=temperature,
                google_api_key=google_api_key
            )
        else:
            return ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=temperature)
    elif model_name == "claude-4-sonnet":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            return ChatAnthropic(
                model="claude-4-sonnet",
                temperature=temperature,
                anthropic_api_key=anthropic_api_key
            )
        else:
            return ChatAnthropic(model="claude-4-sonnet", temperature=temperature)
    else:
        # 기본값으로 gpt-4o 사용
        return ChatOpenAI(model="gpt-4o", temperature=temperature)


# ==================== State 정의 ====================
class ResearchState(TypedDict):
    topic: str
    agent1_research: str  # Agent 1의 연구 결과
    agent2_counter_research: str  # Agent 2의 반대 의견 연구 결과
    agent3_final_report: str  # Agent 3의 최종 리포트
    research_history: List[str]


# ==================== 도구 정의 ====================
def search_internet_tool(query: str) -> str:
    """인터넷 검색 도구 (Perplexity sonar-pro 사용)"""
    try:
        if not PERPLEXITY_AVAILABLE:
            return "Perplexity API를 사용할 수 없습니다. langchain-perplexity를 설치해주세요."
        
        # Perplexity API 키 확인
        perplexity_api_key = os.getenv("PERPLEXITY_API_KEY") or os.getenv("PPLX_API_KEY")
        if not perplexity_api_key:
            return "PERPLEXITY_API_KEY 또는 PPLX_API_KEY가 설정되지 않았습니다."
        
        # Perplexity 모델 초기화
        perplexity_llm = ChatPerplexity(
            api_key=perplexity_api_key,
            model="sonar-pro"
        )
        
        # 검색 프롬프트 구성
        search_prompt = f"""최신 정보를 바탕으로 다음 질문에 대해 상세하고 정확한 답변을 한국어로 제공해주세요.

질문: {query}

요구사항:
- 최신 정보와 사실을 바탕으로 답변하세요
- 출처와 참고 URL을 포함하세요
- 상세하고 구체적인 정보를 제공하세요
- 한국어로 답변하세요
"""
        
        # Perplexity로 검색 및 답변 생성
        response = perplexity_llm.invoke(search_prompt)
        
        if response and hasattr(response, 'content') and response.content:
            return response.content
        else:
            return f"Perplexity에서 빈 응답을 받았습니다. 쿼리: {query}"
            
    except Exception as e:
        return f"인터넷 검색 중 오류 발생: {str(e)}"


def search_arxiv_tool(query: str, max_docs: int = 3) -> str:
    """Arxiv 논문 검색 도구"""
    try:
        loader = ArxivLoader(
            query=query,
            load_max_docs=max_docs,
            load_all_available_meta=True
        )
        docs = loader.load()
        
        if not docs:
            return "Arxiv에서 관련 논문을 찾을 수 없습니다."
        
        # 논문 정보 포맷팅
        formatted_results = []
        for doc in docs:
            metadata = doc.metadata
            title = metadata.get("Title", "제목 없음")
            authors = metadata.get("Authors", "저자 정보 없음")
            summary = metadata.get("Summary", doc.page_content[:500])
            published = metadata.get("Published", "날짜 정보 없음")
            
            formatted_results.append(
                f"제목: {title}\n"
                f"저자: {authors}\n"
                f"발행일: {published}\n"
                f"요약: {summary[:800]}...\n"
                f"전체 내용: {doc.page_content[:1000]}...\n"
                f"{'='*80}\n"
            )
        
        return "\n".join(formatted_results)
    except Exception as e:
        return f"Arxiv 검색 중 오류 발생: {str(e)}"


# ==================== Agent 1: 자료 수집 Agent ====================
def agent1_research(state: ResearchState, progress_bar=None, status_text=None) -> Dict[str, Any]:
    """Agent 1: 인터넷과 Arxiv에서 자료 수집 및 종합"""
    st.info("🔍 Agent 1: 자료 수집 중...")
    
    topic = state["topic"]
    # 세션 상태에서 선택된 모델 가져오기
    model_name = st.session_state.get("selected_model", "gpt-4o")
    llm = get_llm_model(model_name, temperature=0.3)
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.1, text="인터넷 검색 중...")
    if status_text:
        status_text.text("🔍 Agent 1: 인터넷 검색 중...")
    
    # 인터넷 검색
    internet_query = f"{topic}에 대한 최신 정보와 뉴스"
    internet_results = search_internet_tool(internet_query)
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.2, text="Arxiv 논문 검색 중...")
    if status_text:
        status_text.text("📚 Agent 1: Arxiv 논문 검색 중...")
    
    # Arxiv 검색
    arxiv_query = topic
    arxiv_results = search_arxiv_tool(arxiv_query, max_docs=3)
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.3, text="정보 종합 중...")
    if status_text:
        status_text.text("📝 Agent 1: 정보 종합 중...")
    
    # 종합 프롬프트
    synthesis_prompt = f"""당신은 전문 리서처입니다. 주어진 주제에 대해 인터넷과 학술 논문에서 수집한 정보를 종합하여 상세한 연구 보고서를 작성해주세요.

주제: {topic}

인터넷 검색 결과:
{internet_results}

Arxiv 논문 검색 결과:
{arxiv_results}

요구사항:
1. 수집한 정보를 체계적으로 정리하세요
2. 주요 발견사항과 핵심 내용을 강조하세요
3. 출처를 명확히 표시하세요
4. 최소 1000자 이상의 상세한 보고서를 작성하세요
5. 한국어로 작성하세요

보고서 형식:
# {topic} 연구 보고서

## 1. 개요
## 2. 주요 발견사항
## 3. 상세 분석
## 4. 출처 및 참고문헌
"""
    
    response = llm.invoke(synthesis_prompt)
    research_result = response.content
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.33, text="Agent 1 완료!")
    if status_text:
        status_text.text("✅ Agent 1: 자료 수집 완료!")
    
    current_history = state.get("research_history", [])
    
    return {
        "topic": state.get("topic", ""),
        "agent1_research": research_result,
        "agent2_counter_research": state.get("agent2_counter_research", ""),
        "agent3_final_report": state.get("agent3_final_report", ""),
        "research_history": current_history + [f"Agent 1 완료: {datetime.now().strftime('%H:%M:%S')}"]
    }


# ==================== Agent 2: 반대 의견 검색 Agent ====================
def agent2_counter_research(state: ResearchState, progress_bar=None, status_text=None) -> Dict[str, Any]:
    """Agent 2: 반대 의견 및 다른 접근 방식 찾기"""
    st.info("🔄 Agent 2: 반대 의견 및 다른 접근 방식 검색 중...")
    
    topic = state["topic"]
    agent1_result = state.get("agent1_research", "")
    # 세션 상태에서 선택된 모델 가져오기
    model_name = st.session_state.get("selected_model", "gpt-4o")
    llm = get_llm_model(model_name, temperature=0.3)
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.4, text="반대 의견 검색 쿼리 생성 중...")
    if status_text:
        status_text.text("💭 Agent 2: 반대 의견 검색 쿼리 생성 중...")
    
    # 반대 의견 검색 쿼리 생성
    counter_query_prompt = f"""다음 주제와 연구 결과를 바탕으로 반대 의견이나 다른 접근 방식을 찾기 위한 검색 쿼리를 생성해주세요.

주제: {topic}
Agent 1의 연구 결과 요약: {agent1_result[:1000]}

반대 의견, 비판적 시각, 대안적 접근 방식, 다른 철학적 관점을 찾기 위한 검색 쿼리 3개를 생성해주세요.
각 쿼리는 한 줄로 작성하고, 번호를 매겨주세요.
"""
    
    counter_queries_response = llm.invoke(counter_query_prompt)
    counter_queries = counter_queries_response.content
    
    # 각 쿼리로 검색
    all_counter_results = []
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.5, text="반대 의견 인터넷 검색 중...")
    if status_text:
        status_text.text("🌐 Agent 2: 반대 의견 인터넷 검색 중...")
    
    # 인터넷에서 반대 의견 검색
    query_count = 0
    for query_line in counter_queries.split("\n")[:3]:
        if query_line.strip() and not query_line.strip().startswith("#"):
            query = query_line.strip().lstrip("1234567890. -")
            if query:
                internet_results = search_internet_tool(f"{query} {topic}")
                all_counter_results.append(f"검색 쿼리: {query}\n결과:\n{internet_results}\n")
                query_count += 1
                # 프로그레스 바 업데이트
                if progress_bar:
                    progress_bar.progress(0.5 + (query_count * 0.05), text=f"반대 의견 검색 중... ({query_count}/3)")
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.65, text="대안적 접근 방식 논문 검색 중...")
    if status_text:
        status_text.text("📚 Agent 2: 대안적 접근 방식 논문 검색 중...")
    
    # Arxiv에서 대안적 접근 방식 검색
    alternative_arxiv_query = f"alternative approach {topic} criticism debate"
    arxiv_results = search_arxiv_tool(alternative_arxiv_query, max_docs=2)
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.7, text="반대 의견 정보 종합 중...")
    if status_text:
        status_text.text("📝 Agent 2: 반대 의견 정보 종합 중...")
    
    # 종합 프롬프트
    counter_synthesis_prompt = f"""당신은 비판적 분석가입니다. 주어진 주제에 대한 반대 의견, 비판적 시각, 대안적 접근 방식을 찾아 종합적인 보고서를 작성해주세요.

주제: {topic}

Agent 1의 연구 결과:
{agent1_result[:2000]}

반대 의견 검색 결과:
{chr(10).join(all_counter_results)}

대안적 접근 방식 논문:
{arxiv_results}

요구사항:
1. Agent 1의 주장에 대한 반대 의견을 찾아 정리하세요
2. 다른 철학적 접근 방식이나 대안적 관점을 제시하세요
3. 비판적 시각과 한계점을 분석하세요
4. 출처를 명확히 표시하세요
5. 최소 1000자 이상의 상세한 보고서를 작성하세요
6. 한국어로 작성하세요

보고서 형식:
# {topic} 반대 의견 및 대안적 접근 방식 분석

## 1. 반대 의견 요약
## 2. 비판적 시각
## 3. 대안적 접근 방식
## 4. 한계점 및 이슈
## 5. 출처 및 참고문헌
"""
    
    response = llm.invoke(counter_synthesis_prompt)
    counter_result = response.content
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.66, text="Agent 2 완료!")
    if status_text:
        status_text.text("✅ Agent 2: 반대 의견 분석 완료!")
    
    current_history = state.get("research_history", [])
    
    return {
        "topic": state.get("topic", ""),
        "agent1_research": state.get("agent1_research", ""),
        "agent2_counter_research": counter_result,
        "agent3_final_report": state.get("agent3_final_report", ""),
        "research_history": current_history + [f"Agent 2 완료: {datetime.now().strftime('%H:%M:%S')}"]
    }


# ==================== Agent 3: 종합 분석 및 리포트 생성 ====================
def agent3_final_report(state: ResearchState, progress_bar=None, status_text=None) -> Dict[str, Any]:
    """Agent 3: 양쪽 결과 종합, 장단점 논의, 결론 및 향후 전망"""
    st.info("📊 Agent 3: 최종 리포트 작성 중...")
    
    topic = state["topic"]
    agent1_result = state.get("agent1_research", "")
    agent2_result = state.get("agent2_counter_research", "")
    # 세션 상태에서 선택된 모델 가져오기
    model_name = st.session_state.get("selected_model", "gpt-4o")
    llm = get_llm_model(model_name, temperature=0.5)
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.8, text="최종 리포트 작성 중...")
    if status_text:
        status_text.text("📊 Agent 3: 최종 리포트 작성 중...")
    
    final_report_prompt = f"""당신은 전문 분석가입니다. 두 가지 상반된 연구 결과를 종합하여 균형잡힌 최종 리포트를 작성해주세요.

주제: {topic}

Agent 1의 연구 결과 (주요 의견):
{agent1_result}

Agent 2의 연구 결과 (반대 의견 및 대안적 접근):
{agent2_result}

요구사항:
1. 양쪽의 주장을 공정하게 소개하세요
2. 각 접근 방식의 장점과 단점을 명확히 분석하세요
3. 주요 이슈와 쟁점을 정리하세요
4. 자신만의 결론을 제시하세요
5. 향후 전망과 영향에 대해 논의하세요
6. 추가로 리서치가 필요한 영역을 제안하세요
7. 최소 2000자 이상의 상세한 리포트를 작성하세요
8. 한국어로 작성하세요

리포트 형식:
# {topic} 종합 분석 리포트

## 1. 서론
## 2. 주요 의견 소개
### 2.1 Agent 1의 주요 주장
### 2.2 Agent 2의 반대 의견 및 대안
## 3. 장단점 분석
### 3.1 주요 의견의 장점과 단점
### 3.2 반대 의견의 장점과 단점
## 4. 주요 이슈 및 쟁점
## 5. 종합 분석 및 결론
## 6. 향후 전망 및 영향
## 7. 추가 리서치 제안
## 8. 참고문헌
"""
    
    response = llm.invoke(final_report_prompt)
    final_report = response.content
    
    # 프로그레스 바 업데이트
    if progress_bar:
        progress_bar.progress(0.95, text="최종 리포트 완성 중...")
    if status_text:
        status_text.text("✨ Agent 3: 최종 리포트 완성 중...")
    
    current_history = state.get("research_history", [])
    
    return {
        "topic": state.get("topic", ""),
        "agent1_research": state.get("agent1_research", ""),
        "agent2_counter_research": state.get("agent2_counter_research", ""),
        "agent3_final_report": final_report,
        "research_history": current_history + [f"Agent 3 완료: {datetime.now().strftime('%H:%M:%S')}"]
    }


# ==================== Streamlit UI ====================
def main():
    """메인 애플리케이션"""
    
    # 페이지 설정
    st.set_page_config(
        page_title="딥 리서치 시스템",
        page_icon="🔍",
        layout="wide"
    )
    
    # CSS 스타일 (ref.py 참고)
    st.markdown("""
    <style>
    h1 {
        font-size: 1.4rem !important;
        font-weight: 600 !important;
        color: #ff69b4 !important;
    }
    h2 {
        font-size: 1.2rem !important;
        font-weight: 600 !important;
        color: #ffd700 !important;
    }
    h3 {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: #1f77b4 !important;
    }
    .stButton > button {
        background-color: #ff69b4 !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: #ff1493 !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 제목
    st.markdown("""
    <div style="text-align: center; margin-top: -4rem; margin-bottom: 0.5rem;">
        <h1 style="font-size: 2.5rem; font-weight: bold; margin: 0;">
            <span style="color: #1f77b4;">딥</span> 
            <span style="color: #ffffff; font-size: 0.7em;">리서치</span> 
            <span style="color: #ffd700;">시스템</span>
        </h1>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("3개의 전문 Agent가 협력하여 주제에 대한 심층 리서치를 수행합니다.")
    
    # 세션 상태 초기화
    if "research_state" not in st.session_state:
        st.session_state.research_state = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gpt-4o"
    
    # 사이드바
    with st.sidebar:
        st.markdown('<h2 style="color: #1f77b4;">설정</h2>', unsafe_allow_html=True)
        
        # 모델 선택
        st.markdown("### 🤖 모델 선택")
        selected_model = st.selectbox(
            "사용할 LLM 모델을 선택하세요:",
            ["gpt-4o", "gpt-5", "gemini-2.5-pro", "claude-4-sonnet"],
            index=0,
            key="model_selector"
        )
        st.session_state.selected_model = selected_model
        
        # 모델별 설명
        model_info = {
            "gpt-4o": "OpenAI의 멀티모달 모델",
            "gpt-5": "OpenAI 최신 모델 (gpt-4o로 매핑)",
            "gemini-2.5-pro": "Google의 고급 모델",
            "claude-4-sonnet": "Anthropic의 고성능 모델"
        }
        st.caption(f"📌 {model_info.get(selected_model, '')}")
        
        st.markdown("---")
        
        # 주제 입력 (사이드바)
        st.markdown("### 📝 리서치 주제")
        topic = st.text_area(
            "리서치 주제를 입력하세요:",
            height=100,
            placeholder="예: 대규모 언어 모델(LLM)의 미래와 영향",
            help="구체적이고 명확한 주제를 입력하세요",
            key="topic_input"
        )
        
        # 리서치 시작 버튼
        if st.button("🚀 리서치 시작", type="primary", use_container_width=True):
            if not topic.strip():
                st.error("❌ 주제를 입력해주세요.")
                st.rerun()
            
            # API 키 확인 (내부적으로만 사용)
            pplx_key = bool(os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY"))
            openai_key = bool(os.getenv("OPENAI_API_KEY"))
            
            if not pplx_key or not openai_key:
                st.error("❌ 필요한 API 키가 설정되지 않았습니다.")
                st.rerun()
            
            # 이전 리서치 내용 및 채팅 기록 초기화
            st.session_state.research_state = None
            st.session_state.chat_history = []
            
            # 워크플로우 실행
            try:
                # 프로그레스 바 생성
                progress_bar = st.progress(0, text="리서치 시작...")
                status_text = st.empty()
                
                # Agent 1 실행
                initial_state = {
                    "topic": topic,
                    "agent1_research": "",
                    "agent2_counter_research": "",
                    "agent3_final_report": "",
                    "research_history": []
                }
                
                # Agent 1
                state = agent1_research(initial_state, progress_bar, status_text)
                
                # Agent 2
                state = agent2_counter_research(state, progress_bar, status_text)
                
                # Agent 3
                state = agent3_final_report(state, progress_bar, status_text)
                
                # 프로그레스 바 완료
                progress_bar.progress(1.0, text="리서치 완료!")
                status_text.text("✅ 리서치가 완료되었습니다!")
                
                st.session_state.research_state = state
                
                # 프로그레스 바 제거 (선택사항)
                import time
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                
                st.success("✅ 리서치 완료!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 오류 발생: {str(e)}")
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'status_text' in locals():
                    status_text.empty()
        
        st.markdown("---")
        st.markdown("### Agent 설명")
        st.info("""
        **Agent 1**: 자료 수집
        - 인터넷 검색
        - Arxiv 논문 검색
        - 정보 종합
        
        **Agent 2**: 반대 의견
        - 반대 의견 검색
        - 대안적 접근 방식
        - 비판적 분석
        
        **Agent 3**: 종합 리포트
        - 양쪽 결과 종합
        - 장단점 분석
        - 결론 및 전망
        """)
    
    # 메인 컨텐츠
    # 현재 주제 표시
    if st.session_state.get("research_state"):
        current_topic = st.session_state.research_state.get("topic", "")
        if current_topic:
            st.markdown(f"### 📌 현재 리서치 주제: {current_topic}")
            st.markdown("---")
    
    # 결과 표시
    if st.session_state.research_state:
        state = st.session_state.research_state
        
        # 탭으로 결과 표시
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 최종 리포트",
            "🔍 Agent 1 결과",
            "🔄 Agent 2 결과",
            "📈 진행 상황"
        ])
        
        with tab1:
            st.markdown("## 최종 리포트")
            if state.get("agent3_final_report"):
                st.markdown(state["agent3_final_report"])
                
                # 다운로드 버튼
                st.download_button(
                    label="📥 리포트 다운로드",
                    data=state["agent3_final_report"],
                    file_name=f"research_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
            else:
                st.info("최종 리포트가 아직 생성되지 않았습니다.")
        
        with tab2:
            st.markdown("## Agent 1: 자료 수집 결과")
            if state.get("agent1_research"):
                st.markdown(state["agent1_research"])
            else:
                st.info("Agent 1의 결과가 아직 없습니다.")
        
        with tab3:
            st.markdown("## Agent 2: 반대 의견 및 대안 분석")
            if state.get("agent2_counter_research"):
                st.markdown(state["agent2_counter_research"])
            else:
                st.info("Agent 2의 결과가 아직 없습니다.")
        
        with tab4:
            st.markdown("## 진행 상황")
            if state.get("research_history"):
                for history in state["research_history"]:
                    st.text(history)
            else:
                st.info("진행 기록이 없습니다.")
    
    # 대화 내용 표시 (ref.py 스타일)
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💬 리포트 토론")
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # 사용자 입력 영역 (ref.py처럼 화면 하단에 고정)
    if prompt := st.chat_input("리포트에 대해 질문하거나 토론하세요"):
        # 사용자 메시지 추가
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        if not st.session_state.research_state:
            with st.chat_message("assistant"):
                st.write("먼저 리서치를 시작해주세요.")
            st.session_state.chat_history.append({"role": "assistant", "content": "먼저 리서치를 시작해주세요."})
        else:
            # AI 응답 생성
            with st.chat_message("assistant"):
                with st.spinner("답변 생성 중..."):
                    try:
                        # 선택된 모델 사용
                        model_name = st.session_state.get("selected_model", "gpt-4o")
                        llm = get_llm_model(model_name, temperature=0.7)
                        
                        # 컨텍스트 구성
                        context = f"""
리서치 주제: {st.session_state.research_state['topic']}

Agent 1의 연구 결과:
{st.session_state.research_state.get('agent1_research', '')[:1000]}

Agent 2의 반대 의견:
{st.session_state.research_state.get('agent2_counter_research', '')[:1000]}

최종 리포트:
{st.session_state.research_state.get('agent3_final_report', '')[:2000]}

사용자 질문: {prompt}

위 리서치 결과를 바탕으로 사용자의 질문에 대해 상세하고 균형잡힌 답변을 제공해주세요.
"""
                        
                        response = llm.invoke(context)
                        answer = response.content
                        
                        st.write(answer)
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                    except Exception as e:
                        error_msg = f"오류가 발생했습니다: {str(e)}"
                        st.write(error_msg)
                        st.session_state.chat_history.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()

