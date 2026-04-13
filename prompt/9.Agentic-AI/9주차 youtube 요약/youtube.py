import os
import streamlit as st
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import glob

# 페이지 설정 (먼저 실행)
st.set_page_config(
    page_title="Youtube Q&A Chatbot",
    page_icon="🎥",
    layout="wide"
)

# 필수 패키지 확인
try:
    import yt_dlp
except ImportError:
    st.error("⚠️ yt_dlp 패키지가 설치되지 않았습니다. 다음 명령어로 설치해주세요: `pip install yt_dlp`")
    st.stop()

try:
    import pydub
except ImportError:
    st.error("⚠️ pydub 패키지가 설치되지 않았습니다. 다음 명령어로 설치해주세요: `pip install pydub`")
    st.stop()

try:
    import librosa
except ImportError:
    st.error("⚠️ librosa 패키지가 설치되지 않았습니다. 다음 명령어로 설치해주세요: `pip install librosa`")
    st.stop()

from langchain_community.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from pydub import AudioSegment
from openai import OpenAI

# 환경 변수 로드
load_dotenv()

# ==================== 모델 생성 함수 ====================
def create_model(model_name: str, temperature: float = 0) -> ChatOpenAI | ChatGoogleGenerativeAI | ChatAnthropic:
    """모델 이름에 따라 적절한 LangChain 모델 인스턴스를 생성합니다."""
    if model_name == "gpt-4o":
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    elif model_name == "gpt-5":
        # gpt-5는 아직 출시되지 않았으므로 gpt-4o로 매핑
        return ChatOpenAI(model="gpt-4o", temperature=temperature)
    elif model_name == "gemini-2.5-pro":
        google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if google_api_key:
            return ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp", 
                temperature=temperature,
                google_api_key=google_api_key
            )
        else:
            return ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp", temperature=temperature)
    elif model_name == "claude-4-sonnet":
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            return ChatAnthropic(
                model="claude-3-5-sonnet-20241022", 
                temperature=temperature,
                anthropic_api_key=anthropic_api_key
            )
        else:
            return ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=temperature)
    else:
        # 기본값으로 gpt-4o 사용
        return ChatOpenAI(model="gpt-4o", temperature=temperature)

# ==================== 병렬 처리 함수들 ====================
def split_audio_file(audio_path, chunk_length_minutes=10, output_dir=None):
    """
    오디오 파일을 여러 청크로 분할합니다.
    
    Args:
        audio_path: 오디오 파일 경로
        chunk_length_minutes: 각 청크의 길이 (분)
        output_dir: 출력 디렉토리
    
    Returns:
        분할된 오디오 파일 경로 리스트
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(audio_path), "chunks")
    os.makedirs(output_dir, exist_ok=True)
    
    # 오디오 로드
    audio = AudioSegment.from_file(audio_path)
    
    # 청크 길이를 밀리초로 변환
    chunk_length_ms = chunk_length_minutes * 60 * 1000
    
    # 파일명에서 확장자 제거
    base_name = Path(audio_path).stem
    
    chunk_paths = []
    total_length = len(audio)
    chunk_index = 0
    
    for start_ms in range(0, total_length, chunk_length_ms):
        end_ms = min(start_ms + chunk_length_ms, total_length)
        chunk = audio[start_ms:end_ms]
        
        chunk_path = os.path.join(output_dir, f"{base_name}_chunk_{chunk_index:03d}.mp3")
        chunk.export(chunk_path, format="mp3")
        chunk_paths.append(chunk_path)
        chunk_index += 1
    
    return chunk_paths

def transcribe_audio_chunk(chunk_path, openai_client):
    """
    오디오 청크를 텍스트로 변환합니다.
    
    Args:
        chunk_path: 오디오 청크 파일 경로
        openai_client: OpenAI 클라이언트
    
    Returns:
        변환된 텍스트
    """
    try:
        with open(chunk_path, "rb") as audio_file:
            transcript = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="ko"  # 한국어로 지정
            )
        return transcript.text
    except Exception as e:
        st.warning(f"청크 {chunk_path} 처리 중 오류 발생: {str(e)}")
        return ""

def process_youtube_parallel(youtube_url, save_dir, max_workers=4, chunk_length_minutes=10, progress_bar=None, status_text=None):
    """
    YouTube 동영상을 병렬로 처리합니다.
    
    Args:
        youtube_url: YouTube URL
        save_dir: 오디오 저장 디렉토리
        max_workers: 병렬 처리 워커 수
        chunk_length_minutes: 오디오 청크 길이 (분)
        progress_bar: 진행 상황 표시용 progress bar
        status_text: 상태 텍스트 표시용
    
    Returns:
        변환된 텍스트 리스트
    """
    # 1. YouTube 오디오 다운로드
    if status_text:
        status_text.text("오디오 다운로드 중...")
    if progress_bar:
        progress_bar.progress(5)
    
    urls = [youtube_url]
    loader = GenericLoader(
        YoutubeAudioLoader(urls, save_dir),
        OpenAIWhisperParser()
    )
    
    # 오디오 파일 다운로드
    audio_files = glob.glob(os.path.join(save_dir, "*.m4a")) + \
                  glob.glob(os.path.join(save_dir, "*.mp3")) + \
                  glob.glob(os.path.join(save_dir, "*.mp4")) + \
                  glob.glob(os.path.join(save_dir, "*.wav"))
    
    # 기존 파일이 없으면 다운로드
    if not audio_files:
        try:
            loader.load()
            audio_files = glob.glob(os.path.join(save_dir, "*.m4a")) + \
                          glob.glob(os.path.join(save_dir, "*.mp3")) + \
                          glob.glob(os.path.join(save_dir, "*.mp4")) + \
                          glob.glob(os.path.join(save_dir, "*.wav"))
        except Exception as e:
            try:
                docs = loader.load()
                if docs:
                    return [doc.page_content for doc in docs]
            except:
                pass
            return []
    
    if not audio_files:
        return []
    
    # 가장 최근 파일 사용
    audio_file = max(audio_files, key=os.path.getctime)
    
    if status_text:
        status_text.text(f"오디오 파일 분할 중... ({len(audio_files)}개 파일 발견)")
    if progress_bar:
        progress_bar.progress(15)
    
    # 2. 오디오 파일을 청크로 분할
    chunk_dir = os.path.join(save_dir, "chunks")
    chunk_paths = split_audio_file(audio_file, chunk_length_minutes, chunk_dir)
    
    if status_text:
        status_text.text(f"병렬 처리 시작... ({len(chunk_paths)}개 청크, {max_workers}개 워커)")
    if progress_bar:
        progress_bar.progress(20)
    
    # 3. 병렬로 텍스트 변환
    openai_client = OpenAI()
    transcripts = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 모든 작업 제출
        future_to_chunk = {
            executor.submit(transcribe_audio_chunk, chunk_path, openai_client): (i, chunk_path) 
            for i, chunk_path in enumerate(chunk_paths)
        }
        
        # 진행 상황 표시를 위한 카운터
        completed = 0
        total = len(chunk_paths)
        
        # 결과 수집 (순서 유지를 위해 딕셔너리 사용)
        results = {}
        
        for future in as_completed(future_to_chunk):
            i, chunk_path = future_to_chunk[future]
            try:
                transcript = future.result()
                if transcript:
                    results[i] = transcript
                completed += 1
                
                # 진행 상황 업데이트
                if status_text:
                    status_text.text(f"처리 중... ({completed}/{total} 완료)")
                if progress_bar:
                    progress_bar.progress(20 + int(70 * completed / total))
                    
            except Exception as e:
                st.warning(f"청크 처리 실패: {chunk_path}, 오류: {str(e)}")
                completed += 1
    
    # 순서대로 정렬하여 반환
    transcripts = [results[i] for i in sorted(results.keys())]
    
    return transcripts

# 초기 상태 설정
if "conversation_memory" not in st.session_state:
    st.session_state.conversation_memory = []

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "youtube_url" not in st.session_state:
    st.session_state.youtube_url = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summary" not in st.session_state:
    st.session_state.summary = None

if "processing" not in st.session_state:
    st.session_state.processing = False

if "reset_url" not in st.session_state:
    st.session_state.reset_url = False

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "gpt-4o"

# CSS 스타일
st.markdown("""
<style>
/* 헤딩 스타일 */
h1 {
    font-size: 1.4rem !important;
    font-weight: 600 !important;
    color: #ff69b4 !important; /* 분홍색 */
}
h2 {
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    color: #ffd700 !important; /* 노랑색 */
}
h3 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #1f77b4 !important; /* 청색 */
}
h4 {
    font-size: 1.1rem !important;
    font-weight: 600 !important;
}
h5 {
    font-size: 1rem !important;
    font-weight: 600 !important;
}
h6 {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 채팅 메시지 스타일 */
.stChatMessage {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
}

/* 답변 내용 스타일 */
.stChatMessage p {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

/* 리스트 스타일 */
.stChatMessage ul, .stChatMessage ol {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
}

.stChatMessage li {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.3rem 0 !important;
}

/* 강조 텍스트 스타일 */
.stChatMessage strong, .stChatMessage b {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
}

/* 인용문 스타일 */
.stChatMessage blockquote {
    font-size: 0.95rem !important;
    line-height: 1.5 !important;
    margin: 0.5rem 0 !important;
    padding-left: 1rem !important;
    border-left: 3px solid #e0e0e0 !important;
}

/* 코드 스타일 */
.stChatMessage code {
    font-size: 0.9rem !important;
    background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important;
    border-radius: 3px !important;
}

/* 전체 텍스트 일관성 */
.stChatMessage * {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}

/* 버튼 스타일 */
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
        <span style="color: #1f77b4;">Youtube</span> 
        <span style="color: #ffffff; font-size: 0.7em;">Q&A</span> 
        <span style="color: #ffd700;">Chatbot</span>
    </h1>
</div>
""", unsafe_allow_html=True)

st.markdown("YouTube 동영상 URL을 입력하고 내용에 관해 질문해보세요!")

# 사이드바 설정
with st.sidebar:
    st.markdown('<h2 style="color: #1f77b4;">설정</h2>', unsafe_allow_html=True)
    
    # 모델 선택
    model_option = st.selectbox(
        "모델 선택",
        ["gpt-4o", "gpt-5", "gemini-2.5-pro", "claude-4-sonnet"],
        index=0
    )
    st.session_state.selected_model = model_option
    
    st.markdown('<h3 style="color: #ffd700;">YouTube URL</h3>', unsafe_allow_html=True)
    # reset_url이 True면 빈 문자열, 아니면 기존 URL 또는 빈 문자열
    url_value = "" if st.session_state.reset_url else (st.session_state.youtube_url or "")
    youtube_url_input = st.text_input("YouTube URL을 입력하세요", value=url_value, key="youtube_url_input")
    
    # URL 입력 시 reset_url 플래그 해제
    if youtube_url_input and st.session_state.reset_url:
        st.session_state.reset_url = False
    
    if youtube_url_input:
        # 이미 처리 중이거나 같은 URL이 처리된 경우 버튼 비활성화
        is_same_url = st.session_state.youtube_url == youtube_url_input
        is_processing = st.session_state.processing
        
        if is_processing:
            st.warning("동영상 처리 중입니다. 잠시만 기다려주세요...")
            process_button = st.button("동영상 처리하기", disabled=True)
        elif is_same_url and st.session_state.retriever is not None:
            st.info("이미 처리된 동영상입니다. 다시 처리하려면 다른 URL을 입력하거나 '다시 시작하기'를 클릭하세요.")
            process_button = st.button("동영상 처리하기", disabled=True)
        else:
            process_button = st.button("동영상 처리하기")
        
        if process_button and not is_processing:
            st.session_state.processing = True
            try:
                # 오디오 저장 디렉토리 생성
                save_dir = "./youtube_audios/"
                os.makedirs(save_dir, exist_ok=True)
                
                # 병렬 처리 방식
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("1/4: YouTube 오디오 다운로드 중...")
                progress_bar.progress(10)
                
                # 병렬 처리로 텍스트 변환
                transcripts = process_youtube_parallel(
                    youtube_url_input, 
                    save_dir, 
                    max_workers=4,
                    chunk_length_minutes=10,
                    progress_bar=progress_bar,
                    status_text=status_text
                )
                
                if not transcripts:
                    st.session_state.processing = False
                    st.error("동영상을 처리할 수 없습니다. URL을 확인해주세요.")
                else:
                    status_text.text("2/4: 텍스트 결합 중...")
                    progress_bar.progress(50)
                    
                    # 텍스트 결합
                    text = " ".join(transcripts)
                    
                    # 공통 처리 로직
                    status_text.text("3/4: 벡터 스토어 생성 중...")
                    progress_bar.progress(70)
                    
                    # 텍스트 분할
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1500,
                        chunk_overlap=150
                    )
                    splits = text_splitter.split_text(text)
                    
                    # 벡터 스토어 생성
                    embeddings = OpenAIEmbeddings()
                    vectordb = FAISS.from_texts(splits, embeddings)
                    st.session_state.vectorstore = vectordb
                    
                    # 검색기 생성
                    st.session_state.retriever = vectordb.as_retriever(
                        search_type="similarity",
                        search_kwargs={"k": 5}
                    )
                    
                    # YouTube URL 저장
                    st.session_state.youtube_url = youtube_url_input
                    
                    # 요약 생성
                    status_text.text("4/4: 요약 생성 중...")
                    progress_bar.progress(90)
                    
                    summary_prompt = f"""
다음은 YouTube 동영상의 자막 내용입니다. 이 내용을 요약해주세요.

{text[:5000]}

위 내용을 바탕으로 다음 형식으로 요약해주세요:
- 주요 주제
- 핵심 내용
- 중요한 포인트

요약은 한국어로 작성하고, 구조화된 형식으로 작성해주세요.
"""
                    
                    # 모델에 따라 LLM 선택
                    llm = create_model(model_option, temperature=0)
                    
                    summary_response = llm.invoke(summary_prompt)
                    st.session_state.summary = summary_response.content
                    
                    progress_bar.progress(100)
                    status_text.text("처리 완료!")
                    
                    st.session_state.processing = False
                    st.success("동영상 처리가 완료되었습니다!")
                    st.rerun()
                        
            except Exception as e:
                st.session_state.processing = False
                st.error(f"동영상 처리 중 오류가 발생했습니다: {str(e)}")
                st.error("YouTube URL을 확인하거나 잠시 후 다시 시도해주세요.")
    
    # 현재 처리된 YouTube URL 표시
    if st.session_state.youtube_url:
        st.markdown('<h3 style="color: #ffd700;">처리된 동영상</h3>', unsafe_allow_html=True)
        st.write(f"URL: {st.session_state.youtube_url}")
    
    # 다시 시작하기 버튼
    if st.button("다시 시작하기"):
        # 모든 세션 상태 초기화
        st.session_state.conversation_memory = []
        st.session_state.retriever = None
        st.session_state.vectorstore = None
        st.session_state.youtube_url = None
        st.session_state.chat_history = []
        st.session_state.summary = None
        st.session_state.processing = False
        st.session_state.reset_url = True  # URL 입력 필드 초기화 플래그
        st.rerun()
    
    # 시스템 상태 표시
    if st.session_state.youtube_url:
        st.subheader("📊 시스템 상태")
        st.info(f"처리된 동영상: 1개")
        st.info(f"대화 기록 수: {len(st.session_state.chat_history)}")

# 요약 표시
if st.session_state.summary and not st.session_state.chat_history:
    st.markdown('<h2 style="color: #1f77b4;">📝 동영상 요약</h2>', unsafe_allow_html=True)
    st.markdown(st.session_state.summary)
    st.markdown("---")

# 대화 내용 표시
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 사용자 입력 영역 (맨 아래)
if prompt := st.chat_input("질문을 입력하세요"):
    # 사용자 메시지 추가
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.write(prompt)
    
    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            st.write("먼저 YouTube URL을 입력하고 동영상을 처리해주세요.")
        st.session_state.chat_history.append({"role": "assistant", "content": "먼저 YouTube URL을 입력하고 동영상을 처리해주세요."})
    else:
        with st.spinner("답변을 생성 중입니다..."):
            try:
                # RAG 검색
                retrieved_docs = st.session_state.retriever.invoke(prompt)
                
                if not retrieved_docs:
                    response = f"죄송합니다. '{prompt}'에 대한 관련 내용을 찾을 수 없습니다."
                else:
                    # 상위 3개 문서만 사용
                    top_docs = retrieved_docs[:3]
                    
                    # 컨텍스트 구성
                    context_text = ""
                    max_context_length = 8000
                    current_length = 0
                    
                    for i, doc in enumerate(top_docs):
                        doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                        if current_length + len(doc_text) > max_context_length:
                            st.warning(f"토큰 제한으로 인해 문서 {i+1}개만 사용합니다.")
                            break
                        context_text += doc_text
                        current_length += len(doc_text)
                    
                    # 과거 대화 맥락 구성
                    conversation_context = ""
                    if st.session_state.conversation_memory:
                        conversation_context = "\n\n=== 이전 대화 맥락 ===\n"
                        recent_conversations = st.session_state.conversation_memory[-50:]
                        for conv in recent_conversations:
                            conversation_context += f"{conv}\n"
                        conversation_context += "=== 대화 맥락 끝 ===\n"
                    
                    # 시스템 프롬프트 구성
                    system_prompt = f"""
질문: {prompt}

관련 문서:
{context_text}{conversation_context}

위 문서 내용과 이전 대화 맥락을 모두 고려하여 질문에 답변해주세요.
이전 대화에서 언급된 내용이 있다면 그것을 참조하여 더 정확하고 맥락적인 답변을 제공하세요.

답변 형식:
- 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
- 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
- 답변이 길거나 복잡한 경우 여러 헤딩을 사용하여 가독성을 높이세요
- 답변은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
- "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
- 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
- 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
"""
                    
                    # 모델에 따라 LLM 선택
                    llm = create_model(st.session_state.selected_model, temperature=0)
                    
                    # LLM으로 답변 생성
                    response = llm.invoke(system_prompt).content
                
                # 답변 표시
                with st.chat_message("assistant"):
                    st.write(response)
                
                # 대화 기록에 추가
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                
                # 대화 맥락 메모리에 추가
                st.session_state.conversation_memory.append(f"사용자: {prompt}")
                st.session_state.conversation_memory.append(f"AI: {response}")
                if len(st.session_state.conversation_memory) > 100:
                    st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
                
            except Exception as e:
                with st.chat_message("assistant"):
                    st.write(f"오류가 발생했습니다: {str(e)}")
                st.session_state.chat_history.append({"role": "assistant", "content": f"오류가 발생했습니다: {str(e)}"})

