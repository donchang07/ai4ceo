# 1주차: AI 개요 및 현황 (AI Overview & Landscape)

---

## 학습 목표 (Learning Objectives)

이번 주 수업을 마치면 다음을 할 수 있습니다:
- AI, 머신러닝, 딥러닝의 개념 차이를 설명할 수 있다.
- AI 발전의 역사적 흐름을 이해한다.
- 2024~2026 글로벌 AI 트렌드를 파악한다.
- 생성형 AI 도구를 실제로 체험하고 업무 적용 가능성을 탐색한다.

---

## 1. 인공지능이란 무엇인가?

### 1.1 핵심 개념 정리

```
인공지능 (AI, Artificial Intelligence)
└── 머신러닝 (ML, Machine Learning)
    └── 딥러닝 (DL, Deep Learning)
        └── 생성형 AI (Generative AI)
```

| 개념 | 정의 | 예시 |
|:-----|:-----|:-----|
| **인공지능 (AI)** | 인간의 지능을 모방하는 컴퓨터 시스템 | 체스 AI, 번역기, 추천 시스템 |
| **머신러닝 (ML)** | 데이터로부터 패턴을 학습하는 AI 기법 | 스팸 필터, 신용 평가 모델 |
| **딥러닝 (DL)** | 다층 신경망을 활용한 ML의 한 종류 | 이미지 인식, 음성 인식 |
| **생성형 AI** | 텍스트, 이미지, 영상 등을 생성하는 AI | ChatGPT, DALL-E, Sora |

> 💡 **CEO를 위한 한 줄 요약**: AI는 "대량의 데이터로 훈련된, 특정 작업을 자동화하는 소프트웨어"입니다.

---

### 1.2 AI의 발전 역사

| 연도 | 사건 |
|:-----|:-----|
| 1950 | 앨런 튜링, "기계가 생각할 수 있는가?" 논문 발표 |
| 1956 | 다트머스 회의에서 "인공지능" 용어 최초 사용 |
| 1986 | 역전파 알고리즘(Backpropagation) 발표 – 신경망 학습 가능 |
| 1997 | IBM Deep Blue, 체스 세계 챔피언 카스파로프 격파 |
| 2012 | AlexNet – 딥러닝으로 이미지 인식 혁신 (ImageNet 우승) |
| 2016 | AlphaGo, 이세돌 9단에 4:1 승리 |
| 2017 | Transformer 아키텍처 발표 ("Attention is All You Need") |
| 2020 | GPT-3 발표 – 175억 개 파라미터 대규모 언어 모델 |
| 2022 | ChatGPT 출시 – 5일 만에 100만 사용자 돌파 |
| 2023 | GPT-4, Claude, Gemini 등 멀티모달 AI 경쟁 심화 |
| 2024 | AI 에이전트, 멀티모달 AI 실용화 가속 |
| 2025 | 추론 특화 AI (o1, DeepSeek R1 등), AI 코딩 에이전트 대중화 |

---

## 2. 2025~2026 글로벌 AI 트렌드

### 2.1 주요 트렌드

1. **생성형 AI의 기업 도입 가속화**
   - 기업의 77%가 AI 파일럿 또는 도입 단계 (McKinsey, 2025)
   - Microsoft Copilot, Google Workspace AI 등 업무용 AI 보편화

2. **AI 에이전트 (Agentic AI)**
   - 단순 질의응답을 넘어 스스로 계획하고 실행하는 AI
   - 예: 이메일 작성 → 전송 → 일정 예약까지 자동화

3. **소형 언어 모델 (Small Language Models, SLM)**
   - 기업 내부에서 실행 가능한 경량 AI 모델
   - Microsoft Phi, Meta LLaMA 등 온디바이스 AI

4. **멀티모달 AI**
   - 텍스트 + 이미지 + 음성 + 영상을 동시에 처리
   - GPT-4o, Gemini 1.5 Pro 등

5. **AI 규제 및 거버넌스**
   - EU AI Act 시행 (2025~)
   - 각국 AI 안전 법안 및 가이드라인 강화

---

### 2.2 한국 AI 현황

- **정부**: 디지털-AI 강국 실현을 위한 AI 컴퓨팅 센터 구축, AI 바우처 지원
- **기업**: 삼성, LG, SK, 현대 등 주요 대기업의 AI 내재화 경쟁
- **스타트업**: 뤼튼, 업스테이지, 모레, 리벨리온 등 AI 유니콘 성장
- **인재**: AI 인력 부족 (2027년까지 약 10만 명 부족 전망)

---

## 3. 실습: 생성형 AI 체험

### 3.1 추천 도구

| 도구 | 특징 | URL |
|:-----|:-----|:----|
| **ChatGPT** | OpenAI의 대화형 AI, 가장 널리 사용 | chat.openai.com |
| **Claude** | Anthropic의 AI, 긴 문서 처리 강점 | claude.ai |
| **Gemini** | Google의 AI, Google 서비스 연동 | gemini.google.com |
| **뤼튼** | 한국어 특화, 무료 사용 가능 | wrtn.ai |
| **Microsoft Copilot** | Microsoft 365 연동 | copilot.microsoft.com |

### 3.2 실습 과제

다음 프롬프트를 각 도구에서 실행하고 결과를 비교해 보세요:

```
프롬프트 1: "우리 회사 [업종]에서 AI를 활용할 수 있는 3가지 방법을 제안해줘"

프롬프트 2: "AI 도입 시 주요 리스크와 대응 방안을 경영진 보고서 형식으로 작성해줘"

프롬프트 3: "다음 회의록을 요약하고 주요 액션 아이템을 뽑아줘: [회의록 내용 붙여넣기]"
```

---

## 4. 핵심 정리

> ✅ **이번 주 핵심 메시지**
>
> 1. AI는 도구입니다 – 잘 사용하는 조직이 경쟁 우위를 가집니다.
> 2. 생성형 AI는 이미 실무에서 활용 가능한 수준입니다.
> 3. CEO의 역할은 AI 기술자가 아니라 **AI 전략가**입니다.

---

## 5. 다음 주 예고

**2주차: 머신러닝 기초**  
- 데이터가 왜 중요한가?
- 머신러닝은 어떻게 학습하는가?
- 좋은 AI 모델의 조건

---

## 📎 참고자료

- [AI for Everyone - Andrew Ng](https://www.coursera.org/learn/ai-for-everyone)
- [McKinsey Global AI Survey 2025](https://www.mckinsey.com/capabilities/quantumblack/our-insights/the-state-of-ai)
- [과학기술정보통신부 AI 정책](https://www.msit.go.kr)
