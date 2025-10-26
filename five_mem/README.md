
# 🏗️ Architecture Overview
## 1. 핵심 맥락 항목 정의 (Core Context Items Definition)
- EventEntity: Base entity with timestamp fields
- Message: Individual conversation messages
- Episode: Core episode structure with context, topics, goals, and boundaries
- EpisodeAfterBoundary: Extended episode with insights
## 2. 사건의 경계선 나누기 (Event Boundary Detection)
- EventBoundaryTurn: Boundary detection results
- EventMessages: Message collections forming events
- EventBoundary: Boundary markers
- EventCandidate: Candidate events for processing
- Builder classes: State updaters and event processors
## 3. 에피소드 청킹 (Episode Chunking)
- EpisodeCandidate: Episodes with message history and boundary info
- Insight: Analysis results (constraints, decisions, followups)
- Processing modules: State updaters and boundary builders
## 4. 의미 기억 구축 (Semantic Memory Construction)
- SemanticMemoryEntry: General knowledge entries
- SemanticMemoryOperation: Memory operations (add/update/delete)
- SessionMessageHistory: Message history for analysis
- Builder classes: Entity creators

# 🎨 Visual Features
- Color-coded sections for easy navigation
- Clear inheritance relationships showing how classes extend each other
- Data flow connections showing how information moves between sections
- Processing pipelines showing the builder and updater workflows
- Legend explaining the color scheme and relationship types

The diagram captures the complete workflow from raw messages through boundary detection, episode creation, and semantic memory construction, making it easy to understand the system's architecture and data flow.


```
사건의 경계선
## 🧠 설명

### 1. **경험의 연속성**

우리는 시간을 따라 끊임없이 감각·사고·감정을 경험하지만,  
뇌는 이를 **하나의 연속적인 스트림(stream)** 으로 저장하지 않습니다.

→ 대신, “맥락의 변화”를 감지할 때마다 **‘이전’과 ‘다음’을 구분**합니다.
경계는 “지금까지와는 다른 상황이 시작됨”을 알리는 **인지적 신호**입니다.  
다음 변화가 감지되면 경계가 생깁니다.

| 유형       | 예시                          |     |
| -------- | --------------------------- | --- |
| 🏞 환경 변화 | 장소 이동, 조명 변화, 다른 인물 등장      |     |
| 💭 감정 변화 | 놀람, 공포, 분노, 안도감 등 급격한 감정 변동 |     |
| 🎯 목표 변화 | 새로운 의도, 작업, 주의 대상의 전환       |     |
| 🗣 화제 전환 | 대화 주제 변경, 새로운 사건 이야기 시작     |     |

### 3. **경계 감지 후의 처리**

경계가 감지되면 뇌는 다음 두 단계를 수행합니다.

1. **기존 에피소드 인코딩(Encoding)**  
    → 해마(hippocampus)가 지금까지의 사건을 하나의 완결된 단위로 저장.  
    → “이전 장면의 끝”이 됨.
    
2. **새로운 에피소드 시작(New Encoding)**  
    → 현재 맥락 정보를 새 시점으로 초기화하여 “새로운 이야기”로 인식.  
    → “다음 장면의 시작”이 됨.
```