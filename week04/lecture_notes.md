# 4주차: AI 비즈니스 전략 (AI Business Strategy)

---

## 학습 목표 (Learning Objectives)

이번 주 수업을 마치면 다음을 할 수 있습니다:
- AI 성숙도 모델을 활용하여 조직의 현재 수준을 진단할 수 있다.
- AI 투자 의사결정 프레임워크(Build/Buy/Partner)를 적용할 수 있다.
- AI 프로젝트의 ROI를 계산하고 경영진에게 설명할 수 있다.
- 글로벌 AI 선도 기업의 전략을 분석하고 시사점을 도출할 수 있다.

---

## 1. AI 성숙도 모델 (AI Maturity Model)

### 1.1 5단계 AI 성숙도

```
Level 1: 인식 (Aware)       - AI가 무엇인지 이해, 탐색 단계
Level 2: 실험 (Experimental) - PoC(개념 증명) 진행, 소규모 파일럿
Level 3: 운영 (Operational)  - 특정 영역 AI 실제 운영
Level 4: 체계화 (Systematic) - 다양한 영역에 AI 적용, 데이터 플랫폼 구축
Level 5: 혁신 (Transformational) - AI가 핵심 경쟁력, 비즈니스 모델 혁신
```

### 1.2 자가 진단 체크리스트

**데이터 역량**
- [ ] 주요 업무 데이터가 디지털화되어 있다
- [ ] 데이터를 중앙에서 통합 관리하고 있다
- [ ] 데이터 품질 관리 프로세스가 있다
- [ ] 데이터 거버넌스 정책이 수립되어 있다

**기술 역량**
- [ ] 전담 데이터/AI 조직이 있다
- [ ] 클라우드 인프라를 활용하고 있다
- [ ] AI 모델을 운영 환경에 배포한 경험이 있다
- [ ] MLOps 또는 AI 운영 체계가 있다

**비즈니스 역량**
- [ ] 경영진의 AI에 대한 이해도가 높다
- [ ] AI 전략 로드맵이 수립되어 있다
- [ ] AI 프로젝트 예산이 별도로 책정되어 있다
- [ ] AI 교육 및 변화관리 프로그램이 있다

**점수 해석**: 0~3개 = Level 1, 4~6개 = Level 2, 7~9개 = Level 3, 10~12개 = Level 4+

---

## 2. Build vs Buy vs Partner

### 2.1 의사결정 프레임워크

```
AI 솔루션 필요
        ↓
  핵심 경쟁력인가?
   YES         NO
    ↓            ↓
 Build       상용 솔루션이 있나?
          YES            NO
           ↓              ↓
          Buy          Partner
                       (공동개발/아웃소싱)
```

### 2.2 각 옵션 비교

| 항목 | Build (자체 개발) | Buy (구매) | Partner (파트너십) |
|:-----|:----------------|:----------|:-----------------|
| **비용** | 높음 (초기 투자 大) | 중간 (구독료/라이선스) | 중간 |
| **기간** | 길다 (6개월~2년) | 짧다 (즉시~3개월) | 중간 |
| **차별화** | 높음 | 낮음 | 중간 |
| **유지보수** | 자체 부담 | 벤더 의존 | 공동 부담 |
| **데이터 보안** | 완전 통제 | 제한적 | 계약에 따라 다름 |
| **적합 케이스** | 핵심 IP, 고유 데이터 | 범용 기능, 빠른 도입 | 기술 부족, 협력 시너지 |

### 2.3 주요 AI 솔루션 예시

**Buy (상용 SaaS AI)**:
- Microsoft 365 Copilot (업무 생산성)
- Salesforce Einstein (CRM/영업)
- SAP AI (ERP/SCM)
- UiPath (RPA + AI)

**Partner (클라우드 AI 플랫폼)**:
- AWS SageMaker
- Google Vertex AI
- Microsoft Azure AI
- 카카오 클라우드, NAVER CLOVA

---

## 3. AI 투자 ROI 계산

### 3.1 ROI 계산 공식

```
ROI (%) = (AI 도입 효익 - AI 도입 비용) / AI 도입 비용 × 100
```

### 3.2 효익 항목 분류

**직접 효익 (정량화 용이)**:
- 인건비 절감: 자동화로 인한 FTE(Full-time Equivalent) 감소
- 오류 감소: 재작업 비용, 클레임 비용 절감
- 처리 속도 향상: 처리 시간 단축으로 인한 기회 비용 감소
- 매출 증대: 추천 시스템, 개인화로 인한 매출 향상

**간접 효익 (정성적)**:
- 고객 만족도 향상
- 직원 업무 만족도 증가
- 브랜드 이미지 향상
- 신규 비즈니스 모델 창출 기회

### 3.3 비용 항목

| 구분 | 세부 항목 |
|:-----|:---------|
| **초기 비용** | 소프트웨어 구입/개발, 인프라 구축, 데이터 준비 |
| **운영 비용** | 클라우드 비용, 라이선스료, 유지보수 |
| **인력 비용** | AI 인력 채용/교육, 변화관리 |
| **기회 비용** | 도입 기간 중 기존 업무 방해 |

### 3.4 실제 계산 예시

**사례: 고객 서비스 챗봇 도입**

| 항목 | 금액 |
|:-----|-----:|
| 챗봇 구축 비용 | 5,000만원 |
| 연간 운영 비용 | 1,200만원 |
| **총 1년 비용** | **6,200만원** |
| 콜센터 인력 절감 (3명 × 4,000만원) | 12,000만원 |
| 야간/주말 서비스 확대로 매출 증가 | 2,000만원 |
| **총 1년 효익** | **14,000만원** |
| **1년 ROI** | **126%** |

---

## 4. 글로벌 AI 선도 기업 전략

### 4.1 Big Tech AI 전략 비교

| 기업 | AI 전략 키워드 | 핵심 제품/서비스 |
|:-----|:-------------|:--------------|
| **Microsoft** | AI Copilot for Everyone | GitHub Copilot, M365 Copilot, Azure OpenAI |
| **Google** | AI-native Search & Workspace | Gemini, Vertex AI, Google Workspace AI |
| **Amazon** | AI-powered Commerce & Cloud | Alexa, AWS AI/ML, Amazon Q |
| **Meta** | Open AI for Social | LLaMA (오픈소스), Meta AI |
| **Apple** | Privacy-first On-device AI | Apple Intelligence, Siri |

### 4.2 산업별 AI 선도 기업 사례

| 산업 | 기업 | AI 활용 사례 |
|:-----|:-----|:-----------|
| 유통 | Amazon | 수요 예측, 로봇 물류, 개인화 추천 |
| 금융 | JPMorgan | 계약서 검토(COIN), 사기 탐지 |
| 제조 | Siemens | 예지 정비, 디지털 트윈 |
| 의료 | Mayo Clinic | 진단 보조, 환자 예후 예측 |
| 자동차 | Tesla | 자율주행 데이터 학습 플랫폼 |

---

## 5. 핵심 정리

> ✅ **이번 주 핵심 메시지**
>
> 1. AI 도입은 **비즈니스 목표**에서 시작해야 합니다 (기술 → 비즈니스 아님).
> 2. Build/Buy/Partner 결정은 **핵심 경쟁력 여부**로 판단하세요.
> 3. AI ROI는 측정 가능합니다 – **사전에 KPI를 설정**하고 추적하세요.

---

## 6. 다음 주 예고

**5주차: 산업별 AI 활용 사례**  
- 우리 산업에서 AI를 어떻게 활용하고 있는가?
- 경쟁사는 어떤 AI를 도입했는가?
- AI로 새로운 비즈니스를 만들 수 있는가?

---

## 📎 참고자료

- [McKinsey AI Value Creation Report](https://www.mckinsey.com/capabilities/quantumblack)
- [Gartner AI Hype Cycle 2025](https://www.gartner.com/en/articles/what-s-new-in-artificial-intelligence-from-the-2023-gartner-hype-cycle)
- [Harvard Business Review: AI Strategy](https://hbr.org/topic/subject/ai-and-machine-learning)
