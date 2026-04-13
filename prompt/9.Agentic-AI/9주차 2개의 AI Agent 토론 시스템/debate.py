"""
도구를 활용한 2개 에이전트 토론 프로그램
Streamlit 기반 UI로 구현
"""

import streamlit as st
from typing import Callable, List
from dotenv import load_dotenv
import os
import tempfile

# API KEY 로드
load_dotenv()

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
try:
    from langchain_perplexity import ChatPerplexity
    PERPLEXITY_AVAILABLE = True
except ImportError:
    PERPLEXITY_AVAILABLE = False
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_react_agent
from langchain import hub
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader


# ==================== 클래스 정의 ====================

class DialogueAgent:
    """기본 대화 에이전트 클래스"""
    
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: BaseChatModel,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = ""
        self.reset()

    def reset(self):
        """대화 내역을 초기화합니다."""
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """메시지를 생성하고 반환합니다."""
        message = self.model.invoke(
            [
                self.system_message,
                HumanMessage(content="\n".join([self.prefix] + self.message_history)),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """다른 에이전트의 메시지를 받습니다."""
        self.message_history.append(f"{name}: {message}")


class DialogueAgentWithTools(DialogueAgent):
    """도구를 사용할 수 있는 대화 에이전트 클래스"""
    
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: BaseChatModel,
        tools,
    ) -> None:
        super().__init__(name, system_message, model)
        self.tools = tools
        # 모델 타입 확인
        self.is_gemini = isinstance(model, ChatGoogleGenerativeAI)

    def send(self) -> str:
        """도구를 사용하여 메시지를 생성하고 반환합니다."""
        # Gemini 모델인 경우 create_react_agent 사용
        if self.is_gemini:
            prompt = hub.pull("hwchase17/react")
            agent = create_react_agent(self.model, self.tools, prompt)
        else:
            # OpenAI, Anthropic 등은 create_openai_tools_agent 사용
            prompt = hub.pull("hwchase17/openai-functions-agent")
            agent = create_openai_tools_agent(self.model, self.tools, prompt)
        
        # 파싱 오류 처리 함수
        def handle_parsing_error(error: Exception) -> str:
            """파싱 오류 발생 시 LLM 출력을 그대로 반환"""
            error_str = str(error)
            # 오류 메시지에서 실제 LLM 출력 추출 시도
            if "Could not parse LLM output:" in error_str:
                # 오류 메시지에서 실제 출력 부분 추출
                try:
                    start_idx = error_str.find("`") + 1
                    end_idx = error_str.rfind("`")
                    if start_idx > 0 and end_idx > start_idx:
                        return error_str[start_idx:end_idx].strip()
                except:
                    pass
            # 추출 실패 시 기본 메시지 반환
            return "파싱 오류가 발생했지만, 에이전트가 계속 진행합니다."
        
        agent_executor = AgentExecutor(
            agent=agent, 
            tools=self.tools, 
            verbose=False,
            handle_parsing_errors=handle_parsing_error
        )
        
        try:
            result = agent_executor.invoke(
                {
                    "input": "\n".join(
                        [self.system_message.content]
                        + [self.prefix]
                        + self.message_history
                    )
                }
            )
            message = AIMessage(content=result["output"])
            return message.content
        except Exception as e:
            # 예상치 못한 오류 발생 시 오류 메시지 반환
            return f"에이전트 실행 중 오류가 발생했습니다: {str(e)}"


class DialogueSimulator:
    """다중 에이전트 대화 시뮬레이터"""
    
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        """모든 에이전트를 초기화합니다."""
        for agent in self.agents:
            agent.reset()
        self._step = 0

    def inject(self, name: str, message: str):
        """대화를 시작합니다."""
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def step(self) -> tuple[str, str]:
        """다음 발언자를 선택하고 메시지를 생성합니다."""
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]
        message = speaker.send()
        
        for receiver in self.agents:
            receiver.receive(speaker.name, message)
        
        self._step += 1
        return speaker.name, message


# ==================== 유틸리티 함수 ====================

def create_model(model_name: str, temperature: float = 0.7) -> BaseChatModel:
    """모델 이름에 따라 적절한 LangChain 모델 인스턴스를 생성합니다."""
    if model_name == "gpt-5.1":
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)
    elif model_name == "gemini-3-pro-preview":
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if not google_api_key:
            st.error("GOOGLE_API_KEY가 환경변수에 설정되어 있지 않습니다.")
            st.stop()
        return ChatGoogleGenerativeAI(
            model="gemini-3-pro-preview", 
            temperature=temperature,
            google_api_key=google_api_key
        )
    elif model_name == "claude-sonnet-4-5":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if not anthropic_api_key:
            st.error("ANTHROPIC_API_KEY가 환경변수에 설정되어 있지 않습니다.")
            st.stop()
        return ChatAnthropic(
            model="claude-sonnet-4-5", 
            temperature=temperature,
            anthropic_api_key=anthropic_api_key
        )
    else:
        # 기본값: gpt-5.1
        return ChatOpenAI(model="gpt-5.1", temperature=temperature)


def create_perplexity_search_tool():
    """Perplexity를 사용한 인터넷 검색 도구를 생성합니다."""
    from langchain_core.tools import tool
    
    @tool
    def perplexity_search(query: str) -> str:
        """인터넷에서 최신 정보를 검색합니다. 최신 뉴스, 실시간 정보, 현재 상황 등을 검색할 때 사용하세요."""
        try:
            if not PERPLEXITY_AVAILABLE:
                return "Perplexity API를 사용할 수 없습니다. langchain-perplexity를 설치해주세요."
            
            perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
            if not perplexity_api_key:
                return "PERPLEXITY_API_KEY가 .env 파일에 설정되지 않았습니다."
            
            # Perplexity LLM 생성
            llm = ChatPerplexity(
                api_key=perplexity_api_key,
                model="sonar-pro"
            )
            
            # 검색 쿼리 실행
            response = llm.invoke(f"다음 질문에 대해 최신 정보를 바탕으로 자세하고 정확하게 답변해주세요. 한국어로 답변해주세요: {query}")
            if hasattr(response, 'content'):
                return response.content
            return str(response)
        except Exception as e:
            return f"인터넷 검색 중 오류가 발생했습니다: {str(e)}"
    
    return perplexity_search


def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    """다음 발언자를 순환적으로 선택합니다."""
    idx = step % len(agents)
    return idx


def generate_system_message(name: str, description: str, topic: str, other_name: str) -> str:
    """에이전트의 시스템 메시지를 생성합니다."""
    conversation_description = f"""Here is the topic of conversation: {topic}
The participants are: {name}, {other_name}"""
    
    return f"""{conversation_description}
    
Your name is {name}.

Your description is as follows: {description}

Your goal is to persuade your conversation partner of your point of view.

CRITICAL RULES FOR YOUR SPEECH:
- DO NOT include your name ({name}) at the beginning of your message.
- DO NOT say "{name}:" or any prefix with your name.
- DO NOT mention your partner's name ({other_name}) in your speech.
- Just speak your argument directly without any name prefix.
- Start your message immediately with your argument.
- DO NOT restate something that has already been said in the past.
- If you have nothing new to say, keep your response very brief or indicate you have finished.

DO look up information with your tool to refute your partner's claims.
DO cite your sources naturally, as if you are speaking in a real conversation.

IMPORTANT: When you use tools to search for information, speak naturally like a human would. 
- DO NOT say "Perplexity 검색을 통해" or "도구를 사용하여" or mention the tool name explicitly.
- Instead, say things like "최근 연구에 따르면", "보도에 따르면", "조사 결과", "참고 자료에 의하면" etc.

DO NOT fabricate fake citations.
DO NOT cite any source that you did not look up.

DO NOT restate something that has already been said in the past.
DO NOT add anything else.

Stop speaking the moment you finish speaking from your perspective.

Answer in KOREAN.
"""


def create_retriever_tool_from_file(file_path: str, description: str):
    """파일 경로로부터 retriever 도구를 생성합니다."""
    try:
        # PDF 파일만 지원
        if file_path.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        else:
            st.error(f"지원하지 않는 파일 형식입니다: {file_path}")
            return None
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = loader.load_and_split(text_splitter)
        vector = FAISS.from_documents(docs, OpenAIEmbeddings())
        retriever = vector.as_retriever(search_kwargs={"k": 5})
        
        tool = create_retriever_tool(
            retriever,
            name="document_search",
            description=description
        )
        return tool
    except Exception as e:
        st.error(f"파일 로드 오류 ({file_path}): {str(e)}")
        return None


def process_uploaded_file(uploaded_file) -> str:
    """업로드된 파일을 임시 파일로 저장하고 경로를 반환합니다."""
    # 임시 파일 생성
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def generate_debate_summary(
    model: BaseChatModel,
    topic: str,
    agent1_name: str,
    agent2_name: str,
    debate_history: List[tuple[str, str]]
) -> str:
    """토론 내용을 요약합니다."""
    debate_text = "\n\n".join([
        f"{name}: {message}" 
        for name, message in debate_history 
        if name != "Moderator"
    ])
    
    prompt = f"""다음 토론의 내용을 요약해주세요.

토론 주제: {topic}
참가자: {agent1_name}, {agent2_name}

토론 내용:
{debate_text}

위 토론에서 {agent1_name}과 {agent2_name}의 주요 논점과 주장을 각각 요약해주세요.
각 참가자의 핵심 주장, 근거, 반박 내용을 명확하게 정리해주세요.
한국어로 답변해주세요."""

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    except Exception as e:
        return f"요약 생성 중 오류가 발생했습니다: {str(e)}"


def generate_moderator_introduction(
    model: BaseChatModel,
    topic: str,
    agent1_name: str,
    agent2_name: str
) -> str:
    """사회자가 토론 주제를 간결하게 소개하고 토론을 시작합니다."""
    prompt = f"""당신은 토론의 사회자입니다. 토론을 간결하고 직접적으로 시작해야 합니다.

토론 주제: {topic}
참가자: {agent1_name} (찬성측), {agent2_name} (반대측)

다음 형식으로 간결하게 토론을 시작해주세요:
1. 토론 주제를 한 문장으로 제시
2. 참가자 소개 (한 줄)
3. "{agent1_name} (찬성측)부터 발언을 시작하겠습니다."라고 바로 안내

장황한 설명 없이, 간결하고 직접적으로 토론을 시작해주세요. 3-4문장 이내로 작성해주세요.
한국어로 답변해주세요."""

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    except Exception as e:
        return f"인트로 생성 중 오류가 발생했습니다: {str(e)}"


def generate_moderator_opinion(
    model: BaseChatModel,
    topic: str,
    agent1_name: str,
    agent2_name: str,
    summary: str,
    debate_history: List[tuple[str, str]]
) -> str:
    """사회자의 의견을 생성합니다."""
    debate_text = "\n\n".join([
        f"{name}: {message}" 
        for name, message in debate_history 
        if name != "Moderator"
    ])
    
    prompt = f"""당신은 토론의 사회자입니다. 토론이 끝난 후 양측의 논리를 듣고 자신의 생각을 제시해야 합니다.

토론 주제: {topic}
참가자: {agent1_name}, {agent2_name}

토론 요약:
{summary}

전체 토론 내용:
{debate_text}

위 토론을 듣고, 사회자로서 다음을 포함하여 자신의 의견을 제시해주세요:
1. 양측의 논리 중 어떤 부분이 더 설득력 있었는지
2. 토론 주제에 대한 사회자 자신의 생각과 판단
3. 추가로 고려해야 할 사항이나 관점

객관적이고 균형잡힌 시각으로, 하지만 명확한 의견을 제시해주세요.
한국어로 답변해주세요."""

    try:
        response = model.invoke([HumanMessage(content=prompt)])
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    except Exception as e:
        return f"의견 생성 중 오류가 발생했습니다: {str(e)}"


def check_message_similarity(message: str, previous_messages: List[str], threshold: float = 0.8) -> bool:
    """메시지가 이전 메시지와 너무 유사한지 확인합니다."""
    # 간단한 유사도 체크: 같은 단어가 많이 포함되어 있는지 확인
    message_words = set(message.lower().split())
    for prev_msg in previous_messages[-3:]:  # 최근 3개 메시지만 확인
        prev_words = set(prev_msg.lower().split())
        if len(message_words) > 0 and len(prev_words) > 0:
            similarity = len(message_words & prev_words) / len(message_words | prev_words)
            if similarity > threshold:
                return True
    return False


# ==================== Streamlit UI ====================

def main():
    st.set_page_config(
        page_title="에이전트 토론 시스템",
        page_icon="💬",
        layout="wide"
    )
    
    # 세션 상태 초기화
    if "debate_history" not in st.session_state:
        st.session_state.debate_history = []
    if "debate_started" not in st.session_state:
        st.session_state.debate_started = False
    if "temp_files" not in st.session_state:
        st.session_state.temp_files = []
    
    # 화려한 제목 (배경 black)
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: #000000; 
                border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.3);'>
        <h1 style='font-size: 3em; font-weight: bold; margin: 0; padding: 20px; 
                   text-shadow: 3px 3px 6px rgba(255,255,255,0.3);
                   background: linear-gradient(90deg, #FFD700, #FF6B6B, #4ECDC4, #45B7D1);
                   -webkit-background-clip: text;
                   -webkit-text-fill-color: transparent;
                   background-clip: text;'>
            🤖 2개의 AI Agent 토론 시스템 💬
        </h1>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 모델 선택
        model_options = ["gpt-5.1", "gemini-3-pro-preview", "claude-sonnet-4-5"]
        model_name = st.selectbox(
            "모델 선택",
            model_options,
            index=0
        )
        
        # 최대 반복 횟수
        max_iters = st.slider(
            "최대 토론횟수",
            min_value=2,
            max_value=20,
            value=6,
            step=1
        )
        
        # 도구 선택
        tool_type = st.radio(
            "도구 유형",
            ["인터넷 검색", "문서검색", "도구 없음"],
            index=0
        )
        
        # 인터넷 검색 선택 시 Perplexity API 키 확인
        if tool_type == "인터넷 검색":
            perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
            if not perplexity_api_key:
                st.error("⚠️ PERPLEXITY_API_KEY가 .env 파일에 설정되지 않았습니다.")
            else:
                st.success("✅ Perplexity API 키 설정됨")
        
        st.markdown("---")
        
        # 다시 시작하기 버튼
        if st.button("🔄 다시 시작하기", use_container_width=True, type="secondary"):
            st.session_state.debate_history = []
            st.session_state.debate_started = False
            # 임시 파일 정리
            for temp_file in st.session_state.temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
            st.session_state.temp_files = []
            # 위젯 값 초기화
            if "agent1_name" in st.session_state:
                del st.session_state.agent1_name
            if "agent2_name" in st.session_state:
                del st.session_state.agent2_name
            if "agent1_desc" in st.session_state:
                del st.session_state.agent1_desc
            if "agent2_desc" in st.session_state:
                del st.session_state.agent2_desc
            st.rerun()
    
    # 메인 영역
    st.subheader("📋 토론 설정")
    
    # 주제 입력 (전체 너비)
    topic = st.text_area(
        "토론주제: ",
        placeholder="예: 인공지능이 인간의 일자리를 대체할 것인가?",
        height=100
    )
    
    # 에이전트 설정 (2열로 나누기)
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 에이전트 1 설정
        st.markdown("### 에이전트1 설정")
        agent1_name = st.text_input(
            "에이전트 1 이름",
            value=st.session_state.get("agent1_name", "찬성측"),
            key="agent1_name"
        )
        agent1_description = st.text_area(
            "에이전트 1 입장설명 ",
            value=st.session_state.get("agent1_desc", ""),
            placeholder="예: 인공지능은 인간의 일자리를 대체하지만, 새로운 일자리를 창출할 것입니다.",
            height=100,
            key="agent1_desc"
        )
        
        # 에이전트 1 도구 설정
        agent1_file = None
        agent1_file_path = None
        if tool_type == "문서검색":
            agent1_file = st.file_uploader(
                "에이전트 1 PDF 문서 선택",
                type=["pdf"],
                key="agent1_file"
            )
            if agent1_file:
                agent1_file_path = process_uploaded_file(agent1_file)
                st.session_state.temp_files.append(agent1_file_path)
                st.success(f"파일 업로드 완료: {agent1_file.name}")
    
    with col2:
        # 에이전트 2 설정
        st.markdown("### 에이전트2 설정")
        agent2_name = st.text_input(
            "에이전트 2 이름",
            value=st.session_state.get("agent2_name", "반대측"),
            key="agent2_name"
        )
        agent2_description = st.text_area(
            "에이전트 2 입장설명 ",
            value=st.session_state.get("agent2_desc", ""),
            placeholder="예: 인공지능은 많은 일자리를 대체하고, 사회적 불평등을 심화시킬 것입니다.",
            height=100,
            key="agent2_desc"
        )
        
        # 에이전트 2 도구 설정
        agent2_file = None
        agent2_file_path = None
        if tool_type == "문서검색":
            agent2_file = st.file_uploader(
                "에이전트 2 PDF 문서 선택",
                type=["pdf"],
                key="agent2_file"
            )
            if agent2_file:
                agent2_file_path = process_uploaded_file(agent2_file)
                st.session_state.temp_files.append(agent2_file_path)
                st.success(f"파일 업로드 완료: {agent2_file.name}")
    
    st.markdown("---")
    
    # 토론 시작 버튼
    if st.button("🚀 토론 시작", type="primary", use_container_width=True):
        # 입력 검증
        if not topic:
            st.error("토론 주제를 입력해주세요.")
            return
        
        if not agent1_name or not agent2_name:
            st.error("에이전트 이름을 입력해주세요.")
            return
        
        if not agent1_description or not agent2_description:
            st.error("에이전트 입장 설명을 입력해주세요.")
            return
        
        if tool_type == "문서검색":
            if not agent1_file or not agent2_file:
                st.error("문서 파일을 업로드해주세요.")
                return
        
        if tool_type == "인터넷 검색":
            perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
            if not perplexity_api_key:
                st.error("인터넷 검색을 사용하려면 PERPLEXITY_API_KEY가 .env 파일에 설정되어야 합니다.")
                return
        
        # 토론 실행
        st.session_state.debate_started = True
        st.session_state.debate_history = []
        
        with st.spinner("토론을 준비하고 있습니다..."):
            try:
                # 모델 초기화
                model = create_model(model_name, temperature=0.7)
                
                # 도구 설정
                agent1_tools = []
                agent2_tools = []
                
                if tool_type == "인터넷 검색":
                    # Perplexity 검색 도구 생성
                    search_tool = create_perplexity_search_tool()
                    agent1_tools = [search_tool]
                    agent2_tools = [search_tool]
                elif tool_type == "문서검색":
                    agent1_tool = create_retriever_tool_from_file(
                        agent1_file_path,
                        f"이 문서는 {agent1_name}의 입장을 지지하는 정보를 담고 있습니다."
                    )
                    agent2_tool = create_retriever_tool_from_file(
                        agent2_file_path,
                        f"이 문서는 {agent2_name}의 입장을 지지하는 정보를 담고 있습니다."
                    )
                    if agent1_tool:
                        agent1_tools = [agent1_tool]
                    if agent2_tool:
                        agent2_tools = [agent2_tool]
                
                # 시스템 메시지 생성
                system_message1 = SystemMessage(
                    content=generate_system_message(
                        agent1_name, agent1_description, topic, agent2_name
                    )
                )
                system_message2 = SystemMessage(
                    content=generate_system_message(
                        agent2_name, agent2_description, topic, agent1_name
                    )
                )
                
                # 에이전트 생성
                if agent1_tools or agent2_tools:
                    agent1 = DialogueAgentWithTools(
                        name=agent1_name,
                        system_message=system_message1,
                        model=model,
                        tools=agent1_tools if agent1_tools else []
                    )
                    agent2 = DialogueAgentWithTools(
                        name=agent2_name,
                        system_message=system_message2,
                        model=model,
                        tools=agent2_tools if agent2_tools else []
                    )
                else:
                    agent1 = DialogueAgent(
                        name=agent1_name,
                        system_message=system_message1,
                        model=model
                    )
                    agent2 = DialogueAgent(
                        name=agent2_name,
                        system_message=system_message2,
                        model=model
                    )
                
                # 시뮬레이터 생성
                simulator = DialogueSimulator(
                    agents=[agent1, agent2],
                    selection_function=select_next_speaker
                )
                simulator.reset()
                
                # 토론 결과 표시 영역
                st.markdown("---")
                st.subheader("💬 토론진행")
                
                # Moderator 상세 설명 생성
                with st.spinner("사회자가 토론 주제를 설명하는 중..."):
                    moderator_intro = generate_moderator_introduction(
                        model=model,
                        topic=topic,
                        agent1_name=agent1_name,
                        agent2_name=agent2_name
                    )
                
                # Moderator 메시지 표시
                with st.chat_message("assistant"):
                    st.markdown(f"**Moderator**: {moderator_intro}")
                
                # 토론 히스토리 초기화 및 moderator 인트로 추가
                debate_history = []
                debate_history.append(("Moderator", moderator_intro))
                st.session_state.debate_history = debate_history
                
                # 시뮬레이터에 moderator 인트로 주입
                simulator.inject("Moderator", moderator_intro)
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                previous_messages = []
                consecutive_short = 0
                
                for n in range(max_iters):
                    status_text.text(f"토론 진행 중... ({n+1}/{max_iters})")
                    progress_bar.progress((n+1) / max_iters)
                    
                    try:
                        name, message = simulator.step()
                        
                        # 메시지에서 이름 prefix 제거 (중복 방지)
                        cleaned_message = message.strip()
                        if cleaned_message.startswith(f"{name}:"):
                            cleaned_message = cleaned_message[len(f"{name}:"):].strip()
                        if cleaned_message.startswith(f"{name}:"):
                            cleaned_message = cleaned_message[len(f"{name}:"):].strip()
                        
                        # 중복 체크
                        if check_message_similarity(cleaned_message, previous_messages):
                            st.warning(f"{name}의 발언이 이전 발언과 유사합니다. 토론을 중단합니다.")
                            break
                        
                        # 짧은 메시지 체크 (할 말이 없을 때)
                        if len(cleaned_message.split()) < 10:
                            consecutive_short += 1
                            if consecutive_short >= 2:
                                st.info(f"{name}이(가) 더 이상 할 말이 없다고 판단되어 토론을 중단합니다.")
                                break
                        else:
                            consecutive_short = 0
                        
                        debate_history.append((name, cleaned_message))
                        previous_messages.append(cleaned_message)
                        st.session_state.debate_history = debate_history
                        
                        # 메시지 표시
                        with st.chat_message("user" if name == agent1_name else "assistant"):
                            st.markdown(f"**{name}**: {cleaned_message}")
                        
                    except Exception as e:
                        st.error(f"토론 중 오류 발생: {str(e)}")
                        break
                
                progress_bar.empty()
                status_text.empty()
                
                st.success("토론이 완료되었습니다!")
                
                # 사회자 요약 및 의견 생성
                st.markdown("---")
                st.subheader("📊 사회자의 정리요약및 개인생각 표시")
                
                with st.spinner("사회자가 토론을 요약하고 의견을 제시하는 중..."):
                    # 토론 요약 생성
                    summary = generate_debate_summary(
                        model=model,
                        topic=topic,
                        agent1_name=agent1_name,
                        agent2_name=agent2_name,
                        debate_history=debate_history
                    )
                    
                    # 사회자 의견 생성
                    moderator_opinion = generate_moderator_opinion(
                        model=model,
                        topic=topic,
                        agent1_name=agent1_name,
                        agent2_name=agent2_name,
                        summary=summary,
                        debate_history=debate_history
                    )
                
                # 요약 표시
                with st.expander("📝 토론 요약", expanded=True):
                    st.markdown(summary)
                
                # 사회자 의견 표시
                with st.chat_message("assistant"):
                    st.markdown(f"**Moderator (사회자 의견)**:\n\n{moderator_opinion}")
                
                # 토론 요약 다운로드 (요약과 의견 포함)
                full_debate_text = "\n\n".join([f"**{name}**: {msg}" for name, msg in debate_history])
                full_text_with_summary = f"""{full_debate_text}

---

## 토론 요약

{summary}

---

## 사회자 의견

{moderator_opinion}
"""
                
                st.download_button(
                    label="📥 토론 내용 다운로드 (요약 및 의견 포함)",
                    data=full_text_with_summary,
                    file_name="debate_result.txt",
                    mime="text/plain"
                )
                
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                st.exception(e)


if __name__ == "__main__":
    main()

