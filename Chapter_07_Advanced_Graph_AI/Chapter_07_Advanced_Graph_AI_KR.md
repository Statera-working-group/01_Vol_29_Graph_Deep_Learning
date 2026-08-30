**Volume 29. Graph Deep Learning**

# Chapter 07. Advanced Graph AI

## 07.01. Heterogeneous GNN

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

이종 그래프 신경망(Heterogeneous Graph Neural Networks)은 여러 유형의 노드(node)와 관계(relationship)를 포함하는 그래프를 처리할 수 있도록 기존 그래프 신경망(GNN)을 확장한 구조이다. 모든 정점(vertex)과 엣지(edge)가 동일한 의미를 가진다고 가정하는 대신, 이종 그래프 신경망(Heterogeneous GNN)은 서로 다른 엔터티 유형(entity type)과 관계 유형(relation type)을 명시적으로 표현한다. 이러한 구조는 실제 시스템이 서로 다른 역할, 속성, 행동을 가진 객체들의 상호작용으로 구성될 때 특히 중요하다.

이종 그래프(heterogeneous graph)는 노드와 엣지에 유형 매핑(type mapping)이 연결된 그래프로 설명할 수 있다. 노드는 사람(person), 제품(product), 문서(document), 센서(sensor), 로봇(robot), 도로 구간(road segment), 조직(organization) 등을 나타낼 수 있으며, 엣지는 구매(purchases), 인용(cites), 감지(detects), 제어(controls), 통신(communicates with), 소속(belongs to) 등의 관계를 나타낼 수 있다. 따라서 생성되는 토폴로지(topology)는 연결 구조와 의미 구조를 동시에 표현한다.

이러한 특성은 동종 그래프(homogeneous graph)와 근본적으로 다르다. 기존 그래프 합성곱 신경망(GCN)이나 그래프 어텐션 네트워크(GAT)에서는 일반적으로 이웃 노드(neighboring node)를 공유 변환(shared transformation)을 통해 처리한다. 반면 이종 그래프에서는 '구매(purchases)' 관계를 통해 전달되는 메시지가 '리뷰(reviews)' 또는 '제조(manufactured-by)' 관계를 통해 전달되는 메시지와 완전히 다른 의미를 가질 수 있다.

유용한 형식적 표현에서는 노드 유형 매핑(node-type mapping) φ(v)와 엣지 유형 매핑(edge-type mapping) ψ(e)를 도입한다. 이러한 매핑은 모든 노드에 노드 유형을, 모든 엣지에 관계 유형을 연결한다. 출발 노드 유형(source-node type), 관계 유형(relation type), 목적 노드 유형(destination-node type)의 조합은 사용자--구매--제품(User--buys--Product) 또는 센서--관측--객체(Sensor--observes--Object)와 같은 의미적 관계 패턴(semantic relation pattern)을 형성한다.

따라서 이종 그래프 신경망(Heterogeneous GNN)의 핵심 과제는 유형 인식 메시지 전달(type-aware message passing)이다. 모든 입력 정보에 하나의 변환 행렬(transformation matrix)을 적용하는 대신, 모델은 서로 다른 관계마다 다른 변환을 학습할 수 있다. 관계 r을 통해 생성되는 메시지는 개념적으로 mᵣ = fᵣ(h_source)로 표현할 수 있으며, 여기서 fᵣ은 관계별 변환(relation-specific transformation), h_source는 이웃 출발 노드의 표현이다.

관계별 메시지(relation-specific message)가 생성된 이후에는 이를 목적 노드 표현(destination representation)으로 집계해야 한다. 집계(aggregation)는 두 단계로 수행될 수 있는데, 먼저 각각의 관계 유형 내부에서 정보를 결합하고 이후 여러 관계의 정보를 다시 통합한다. 합(sum), 평균(mean), 어텐션(attention), 학습형 게이팅(learned gating) 등을 사용할 수 있으며, 이를 통해 네트워크는 어떤 이웃 노드뿐 아니라 어떤 관계 유형이 중요한지도 학습한다.

어텐션 메커니즘(attention mechanism)은 서로 다른 관계가 동일한 중요도를 가지지 않는 이종 그래프에서 특히 유용하다. 예를 들어 연구자의 연구 주제를 예측할 때 저자 관계(authorship relation)가 기관 관계(institutional relationship)보다 더 중요한 정보를 제공할 수 있다. 이종 어텐션(heterogeneous attention)은 이웃 수준(neighbor level), 관계 수준(relation level), 또는 두 수준 모두에서 중요도를 학습할 수 있다.

관계형 그래프 합성곱 신경망(Relational Graph Convolutional Network, R-GCN)과 같은 모델은 이러한 접근의 중요한 기반을 제공한다. R-GCN은 이웃 정보를 집계할 때 관계 의존적 변환(relation-dependent transformation)을 적용하기 때문에 지식 그래프(knowledge graph)와 다중 관계 네트워크(multi-relational network)에 적합하다. 관계 유형이 많아질 경우 발생하는 과도한 매개변수를 줄이기 위해 기저 분해(basis decomposition)나 블록 분해(block decomposition)를 사용할 수 있다.

이종 그래프 어텐션 네트워크(Heterogeneous Graph Attention Network)는 서로 다른 의미 구조(semantic structure)에 대한 계층적 어텐션(hierarchical attention)을 학습함으로써 이러한 원리를 확장한다. 모든 엣지 유형을 독립적으로 처리하는 대신, 모델은 서로 다른 엔터티 유형을 연결하는 의미 있는 경로를 구성할 수 있다. 이러한 경로는 메타 경로(meta-path)라고 하며 저자--논문--저자(Author--Paper--Author), 사용자--아이템--카테고리(User--Item--Category) 등의 관계 패턴을 표현한다.

메타 경로(meta-path)는 그래프 토폴로지(graph topology)와 의미 추론(semantic reasoning)을 연결하는 중요한 역할을 한다. 직접적인 연결은 두 엔터티가 상호작용한다는 사실을 나타내지만, 메타 경로는 유형화된 엔터티와 관계의 연속을 통해 두 엔터티가 어떻게 관련되는지를 표현한다. 따라서 서로 다른 메타 경로는 서로 다른 의미적 관점(semantic perspective)을 나타내며, 어텐션은 특정 예측 작업에서 어떤 관점이 가장 유용한지를 결정할 수 있다.

최근의 이종 그래프 신경망 구조는 수작업으로 정의된 메타 경로에 대한 의존성을 줄이는 방향으로 발전하고 있다. 예를 들어 이종 그래프 트랜스포머(Heterogeneous Graph Transformer)는 트랜스포머(Transformer)와 유사한 구조 안에서 유형 인식 투영(type-aware projection), 관계 의존적 어텐션(relation-dependent attention), 관계별 메시지 변환(relation-specific message transformation)을 사용한다. 이를 통해 노드와 엣지 유형 정보를 유지하면서 유용한 상호작용 패턴을 직접 학습할 수 있다.

서로 다른 노드 유형은 호환되지 않는 원시 특징(raw feature)을 가질 수 있기 때문에 표현 공간(representation space) 자체도 신중하게 설계해야 한다. 문서는 텍스트 임베딩(text embedding)을 포함할 수 있고, 사용자는 인구통계 또는 행동 특성을 가질 수 있으며, 센서는 수치 측정값을 제공할 수 있다. 유형별 인코더(type-specific encoder)는 이러한 입력을 호환 가능한 잠재 공간(latent space)으로 투영한 후 관계형 메시지 전달을 통해 문맥적 표현(contextual representation)으로 통합할 수 있다.

학습 목적(training objective)은 그래프 작업의 목적에 따라 달라진다. 노드 분류(node classification)는 하나 이상의 노드 유형에 대한 범주를 예측할 수 있으며, 링크 예측(link prediction)은 엔터티 사이에 특정 유형의 관계가 존재할 가능성을 추정한다. 엣지 분류(edge classification), 추천(recommendation), 그래프 완성(graph completion), 이상 탐지(anomaly detection), 표현 학습(representation learning) 역시 노드와 관계 유형의 의미적 제약을 유지하면서 구성할 수 있다.

네거티브 샘플링(negative sampling)은 이종 그래프의 링크 예측에서 특히 중요하다. 임의의 노드를 무작위로 연결하면 많은 노드 유형 조합 자체가 구조적으로 불가능하기 때문에 의미 없는 네거티브 샘플이 생성될 수 있다. 따라서 효과적인 학습에서는 그래프 스키마(graph schema)를 고려하여 예측 대상 관계에서 허용되는 출발 및 목적 노드 유형으로부터 후보 네거티브 엣지를 샘플링해야 한다.

이종 그래프는 확장성(scalability) 측면에서도 새로운 문제를 발생시킨다. 실제 지식 그래프, 추천 시스템, 통신 네트워크, 산업 시스템은 수백만 또는 수십억 개의 유형화된 엣지를 포함할 수 있다. 따라서 이웃 샘플링(neighbor sampling)은 그래프 크기뿐 아니라 관계 분포도 고려해야 한다. 빈도가 높은 관계를 지나치게 샘플링하면 드물지만 중요한 관계가 억제될 수 있고, 균일 샘플링은 덜 중요한 이웃에 계산 자원을 낭비할 수 있다.

또 다른 문제는 매개변수 증가(parameter growth)이다. 모든 노드와 관계 유형에 독립적인 신경망 변환을 할당하면 스키마가 커질수록 계산 및 메모리 비용이 증가한다. 매개변수 공유(parameter sharing), 관계 임베딩(relation embedding), 저랭크 분해(low-rank decomposition), 그룹 변환(grouped transformation), 공유 어텐션(shared attention) 등을 통해 이러한 비용을 줄일 수 있다. 핵심은 의미적 전문화와 관계 간 통계적 공유 사이에서 균형을 찾는 것이다.

이종 그래프 신경망은 지식 자체가 본질적으로 다중 관계 구조를 가지기 때문에 지식 그래프(knowledge graph)에 특히 자연스럽게 적용된다. 엔터티는 서로 다른 개념적 클래스에 속하고 술어(predicate)는 엔터티 사이의 서로 다른 관계를 정의한다. 그래프 임베딩(graph embedding)과 유형 인식 메시지 전달을 결합하면 로컬 엔터티 속성, 이웃 정보, 관계 의미를 통합하여 엔터티 분류, 링크 예측, 그래프 완성, 신경망 추론(neural reasoning)을 수행할 수 있다.

추천 시스템(recommendation system) 역시 중요한 적용 분야이다. 사용자(user), 제품(product), 브랜드(brand), 카테고리(category), 리뷰(review), 검색어(query), 세션(session)을 서로 다른 노드 유형으로 표현할 수 있다. 조회(viewing), 구매(purchasing), 평가(rating), 검색(searching), 소속(belonging-to) 등의 상호작용은 서로 다른 엣지 유형이 된다. 이종 그래프 신경망은 이러한 정보를 하나의 사용자--아이템 행렬로 압축하지 않고 관계별 의미를 유지하면서 통합할 수 있다.

로보틱스 및 공간 AI(Robotics and Spatial AI)에서는 로봇(robot), 센서(sensor), 객체(object), 장소(place), 지도 요소(map element), 작업(task), 인간 운영자(human operator)를 하나의 관계 구조 안에서 표현할 수 있다. 관계는 공간적 근접성(spatial proximity), 가시성(visibility), 통신(communication), 소유 관계(ownership), 이동 연결성(navigation connectivity), 제어 의존성(control dependency) 등을 나타낼 수 있으며, 유형 인식 메시지 전달을 통해 기하학적, 의미적, 운영적 정보를 통합할 수 있다.

다중 로봇 또는 인프라 시스템에서는 그래프가 로봇, 충전소(charging station), 엘리베이터(elevator), 문(door), 도로 구간(road segment), 교통 신호(traffic signal), 서버(server), 작업(task)까지 포함하도록 확장될 수 있다. 이종 그래프 추론(heterogeneous graph reasoning)은 각 객체의 기능적 역할을 유지하면서 서로 다른 운영 엔터티 사이에 정보를 전달하여 협업(coordination), 할당(allocation), 예측(prediction), 모니터링(monitoring), 공유 상황 인식(shared situational understanding)을 지원한다.

시간적 변화(temporal evolution)는 여기에 또 하나의 차원을 추가한다. 이종 그래프는 엔터티가 생성되거나 사라지고, 이동하거나 통신하며, 상태를 변경함에 따라 지속적으로 변화할 수 있다. 이종 그래프 신경망이 주로 엔터티와 관계 유형의 다양성을 처리한다면, 시간 모델링(temporal modeling)을 결합함으로써 이종 시간 그래프(heterogeneous temporal graph)를 구성할 수 있다. 이는 다음의 주요 고급 그래프 AI 주제인 시간 그래프 신경망(Temporal GNN)으로 자연스럽게 연결된다.

이종 그래프 신경망(Heterogeneous GNN)의 더 넓은 의미는 그래프 AI(Graph AI)를 단순한 일반적 연결 구조의 학습에서 구조화된 엔터티와 관계 시스템의 학습으로 확장한다는 데 있다. 이 표현은 단순히 누가 누구와 연결되어 있는지를 나타내는 것이 아니라, 각각의 엔터티가 무엇이고 각 연결이 무엇을 의미하며 서로 다른 의미적 상호작용이 전체 시스템에 어떻게 영향을 미치는지를 함께 표현한다. 이러한 특성으로 인해 이종 그래프 신경망은 고급 관계형 지능(relational intelligence)을 구현하는 핵심 구조가 된다.

## 07.02. Temporal GNN

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

시간 그래프 신경망(Temporal Graph Neural Networks)은 그래프 학습(graph learning)을 정적인 관계 구조(static relational structure)에서 노드(node), 엣지(edge), 특징(feature), 상호작용(interaction)이 시간에 따라 변화하는 시스템으로 확장한다. 기존 그래프 신경망(GNN)은 그래프의 토폴로지(topology)를 하나의 고정된 상태처럼 처리하지만, 시간 그래프 신경망(Temporal GNN)은 관계 구조와 시간적 변화(temporal evolution)를 동시에 모델링한다. 이러한 능력은 통신 네트워크, 교통 시스템, 금융 거래, 사회적 상호작용, 로보틱스와 같은 동적 환경에서 필수적이다.

시간 그래프(temporal graph)는 일련의 그래프 스냅샷(graph snapshot) 또는 타임스탬프(timestamp)가 지정된 연속적인 이벤트 스트림(event stream)으로 표현할 수 있다. 스냅샷 기반 표현에서는 그래프를 G₁, G₂, \..., Gₜ와 같은 이산적인 시간 구간으로 분할하며, 각각의 스냅샷은 특정 기간 동안의 시스템 상태를 나타낸다. 반면 연속 시간 표현(continuous-time representation)은 엣지 생성, 메시지 전송, 거래, 센서 관측, 노드 상태 갱신 등의 개별 이벤트를 정확한 발생 시간과 함께 기록한다.

정적 그래프(static graph)와 시간 그래프(temporal graph)의 차이는 이웃 정보(neighborhood information)의 의미를 변화시킨다. 정적 그래프에서는 연결된 두 노드를 단순히 이웃으로 처리하지만, 시간 그래프에서는 두 노드의 상호작용이 몇 초, 몇 시간 또는 몇 달의 간격을 두고 발생했을 수 있다. 따라서 이웃 노드의 중요성은 해당 노드의 특징과 관계 유형뿐만 아니라 상호작용이 언제 발생했으며 이후 시스템이 어떻게 변화했는지에 따라서도 결정된다.

따라서 시간 그래프 신경망(Temporal GNN)은 구조적 메시지 전달(structural message passing)과 시간을 표현하는 메커니즘을 결합한다. 노드 j에서 노드 i로 전달되는 메시지는 개념적으로 출발 노드 표현(source representation) hⱼ, 엣지 정보(edge information) eᵢⱼ, 경과 시간(elapsed time) Δt의 시간 인코딩(temporal encoding)에 의존할 수 있다. 결과 메시지는 mᵢⱼ(t)=f(hⱼ,eᵢⱼ,Δt)로 표현할 수 있으며, 이를 통해 최근 상호작용과 오래된 관측을 구별할 수 있다.

시간 정보(time information)는 여러 방법으로 인코딩할 수 있다. 간단한 방법에서는 정규화된 타임스탬프(normalized timestamp)나 경과 시간 값을 사용하고, 보다 표현력이 높은 모델에서는 사인파 인코딩(sinusoidal encoding), 학습 가능한 시간 임베딩(learnable time embedding), 연속 시간 함수(continuous-time function)를 사용한다. 이러한 표현은 시간적 거리를 신경망이 처리할 수 있는 특징으로 변환하여 주기적 행동, 최신성 효과(recency effect), 장기 의존성(long-term dependency), 불규칙한 시간 패턴을 학습할 수 있도록 한다.

시간 그래프 신경망의 주요 계열 중 하나는 이산 그래프 스냅샷(discrete graph snapshot)을 사용한다. 먼저 그래프 신경망(GNN)이 각 스냅샷에서 구조적 표현(structural representation)을 추출하고, 이후 순환 신경망(RNN), 게이트 순환 유닛(GRU), 장단기 메모리(LSTM), 트랜스포머(Transformer)와 같은 순차 모델(sequential model)이 스냅샷 사이의 변화를 포착한다. 이 구조는 공간적 또는 관계적 추론과 시간적 추론을 분리하며, 데이터가 초, 분, 시간, 일과 같은 일정한 간격으로 제공될 때 특히 유용하다.

또 다른 계열은 연속적인 시간 이벤트(continuous temporal event)를 직접 처리한다. 완전한 그래프 스냅샷을 반복적으로 재구성하는 대신 상호작용이 발생할 때마다 노드 표현을 갱신한다. 이벤트 기반 처리(event-driven processing)는 의미 있는 변화에 계산을 집중하기 때문에 희소한 동적 시스템(sparse dynamic system)에 효율적이다. 또한 여러 이벤트를 거친 스냅샷으로 묶을 때 손실될 수 있는 정확한 시간 순서(temporal ordering)를 보존한다.

시간 그래프 네트워크(Temporal Graph Networks)는 메모리(memory)를 동적 그래프 학습(dynamic graph learning)의 핵심 구성요소로 도입한다. 각 노드는 관련된 과거 상호작용을 요약하는 메모리 상태(memory state)를 유지할 수 있다. 새로운 이벤트가 발생하면 상호작용하는 엔터티(entity), 타임스탬프, 현재 메모리로부터 메시지를 생성하고, 갱신 함수(update function)가 메모리를 수정하며, 임베딩 모듈(embedding module)이 저장된 이력과 현재 그래프 이웃을 결합하여 시간 의존적 노드 표현(time-dependent node representation)을 생성한다.

메모리(memory)는 현재 특징(current feature)과 축적된 이력(accumulated history)을 구분하는 중요한 역할을 한다. 두 노드가 현재 시점에서 동일한 관측 특징을 가지고 있더라도 서로 매우 다른 상호작용 이력을 가질 수 있다. 따라서 내부 시간 상태(internal temporal state)는 크게 다를 수 있다. 이를 통해 시간 그래프 신경망은 미래 행동이 현재 구성뿐 아니라 현재 상태에 도달하기까지의 이벤트 순서에도 영향을 받는 경로 의존적 시스템(path-dependent system)을 표현할 수 있다.

어텐션 메커니즘(attention mechanism)은 시간적 추론(temporal reasoning)을 위한 또 하나의 강력한 방법이다. 시간 어텐션(temporal attention)은 과거 이웃의 내용, 관계, 타임스탬프에 따라 서로 다른 중요도를 부여할 수 있다. 단기 행동이 중요할 때는 최근 상호작용에 높은 어텐션을 부여할 수 있고, 지속적인 관계를 나타내는 경우에는 오래된 상호작용도 중요하게 유지할 수 있다. 모델은 고정된 시간 감쇠 규칙(temporal decay rule)에만 의존하지 않고 이러한 의존성을 직접 학습한다.

시간 그래프 어텐션 네트워크(Temporal Graph Attention Networks)와 시간 그래프 네트워크(Temporal Graph Networks)와 같은 모델은 어텐션, 메모리, 메시지 전달, 시간 인코딩을 어떻게 결합할 수 있는지를 보여준다. 다른 접근법에서는 그래프 합성곱(graph convolution)의 매개변수를 시간에 따라 변화시키거나 순환 메커니즘(recurrent mechanism)을 사용하여 노드 상태를 갱신한다. 구조적 차이는 있지만 공통적인 목표는 그래프 구조와 시간적 문맥(temporal context)에 동시에 의존하는 표현 hᵢ(t)를 학습하는 것이다.

시간 링크 예측(temporal link prediction)은 이 분야에서 가장 중요한 작업 중 하나이다. 두 노드가 일반적으로 연결되어 있는지를 판단하는 대신 특정 상호작용이 미래의 특정 시점에 발생할 가능성을 예측한다. 미래 거래, 통신 이벤트, 사용자-아이템 상호작용, 차량 간 조우, 로봇-자원 관계 등의 예측이 이에 해당한다. 학습 과정에서 미래 정보가 유출되는 것을 방지하기 위해 시간 순서(temporal ordering)를 주의 깊게 보존해야 한다.

동적 노드 분류(dynamic node classification)는 시간에 따라 변화하는 레이블(label)이나 상태(state)를 예측한다. 기계를 나타내는 노드는 정상 작동 상태에서 성능 저하 상태로 변화할 수 있으며, 도로 구간은 원활한 상태에서 혼잡 상태로 변화할 수 있다. 시간 노드 임베딩(temporal node embedding)은 현재 관측과 과거 문맥을 모두 반영하므로 상태 추정(state estimation), 이상 탐지(anomaly detection), 위험 예측(risk prediction), 행동 예측(behavior forecasting)에 활용할 수 있다.

시간 그래프 학습(temporal graph learning)은 고유한 데이터 분할(data splitting) 요구사항도 가진다. 엣지를 무작위로 학습 세트와 테스트 세트로 분할하면 미래 이벤트가 과거를 예측하는 표현에 간접적으로 영향을 줄 수 있기 때문에 비현실적인 평가가 이루어질 수 있다. 따라서 시간 데이터셋은 일반적으로 시간 순서에 따라 분할하며, 모델은 이전 상호작용으로 학습하고 이후 이벤트로 평가한다. 이는 미래 관측을 사용할 수 없는 실제 배포 환경을 더욱 정확하게 반영한다.

네거티브 샘플링(negative sampling) 역시 시간을 고려해야 한다. 특정 시점에서 연결되지 않은 노드 쌍이 이후에는 연결될 수 있기 때문에 현재의 네거티브 샘플이 실제로는 유효한 미래 상호작용을 나타낼 수 있다. 따라서 시간 네거티브 샘플링(temporal negative sampling)에서는 관측 구간(observation window)과 예측 범위(prediction horizon)를 신중하게 정의해야 한다. 현재 존재하지 않는 모든 엣지를 영구적인 네거티브로 처리하는 것이 아니라 관련 기간 안에서 발생하지 않는 이벤트를 구별하는 것이 목적이다.

동적 그래프가 지속적으로 새로운 이벤트를 생성하기 때문에 확장성(scalability)은 특히 어려운 문제가 된다. 모든 상호작용이 발생할 때마다 전체 그래프의 임베딩을 다시 계산하는 것은 현실적으로 지나치게 많은 계산 비용을 요구한다. 증분 갱신(incremental update), 시간 이웃 샘플링(temporal neighbor sampling), 메모리 캐싱(memory caching), 미니배치 이벤트 처리(mini-batch event processing), 제한된 과거 이웃(bounded historical neighborhood)을 사용하면 이러한 비용을 줄일 수 있다.

시간 그래프 신경망은 교통 및 스마트 인프라(transportation and smart infrastructure)에 매우 적합하다. 도로, 교차로, 차량, 교통 신호, 교통 시설은 상태가 지속적으로 변화하는 그래프를 형성한다. 교통 밀도, 이동 시간, 사고, 연결 상태는 시간에 따라 변화한다. 시간 그래프 모델은 공간 관계와 과거 관측을 결합하여 교통 예측(traffic forecasting), 혼잡 예측(congestion prediction), 인프라 모니터링(infrastructure monitoring), 동적 경로 설정(dynamic routing)을 지원할 수 있다.

로보틱스 및 공간 AI(Robotics and Spatial AI)에서 시간 그래프는 로봇이 이동하고 상호작용함에 따라 변화하는 환경을 표현하는 자연스러운 방법을 제공한다. 노드는 로봇, 객체, 위치, 센서, 작업, 인프라를 나타낼 수 있으며, 타임스탬프가 지정된 엣지는 관측, 통신, 근접성, 내비게이션, 작업 할당을 나타낸다. 따라서 그래프는 환경에 대한 정적인 표현이 아니라 지속적으로 변화하는 관계형 메모리(relational memory)가 된다.

다중 로봇 시스템(multi-robot system)은 특히 중요한 사례이다. 로봇은 통신 범위에 들어오거나 벗어날 수 있고, 작업이 재할당될 수 있으며, 충전소가 점유되고 이동 가능성(traversability) 조건이 변화할 수 있다. 시간 그래프 신경망은 과거 문맥을 유지하면서 이러한 변화를 관계 구조 전체에 전달하여 협업(coordination), 자원 할당(resource allocation), 혼잡 예측, 공유 상황 인식(shared situational awareness), 플릿 수준 의사결정 지원(fleet-level decision support)을 가능하게 한다.

시간 그래프(temporal graph)는 이종 그래프(heterogeneous graph)와 결합할 수도 있다. 이종 시간 그래프(heterogeneous temporal graph)에서는 엔터티 유형과 관계 유형을 유지하면서 상호작용이 시간에 따라 변화한다. 따라서 로봇, 센서, 객체, 충전소, 지도 영역, 작업은 서로 다른 의미적 정체성(semantic identity)을 유지하면서 타임스탬프가 지정된 관계에 참여할 수 있다. 이는 유형 인식 관계형 지능(type-aware relational intelligence)과 동적 상태 모델링(dynamic state modeling)을 결합한다.

시간 그래프 신경망(Temporal GNN)과 일반적인 순차 모델(sequence model) 사이에는 중요한 개념적 차이가 존재한다. 장단기 메모리(LSTM)나 트랜스포머(Transformer)는 벡터 시퀀스가 시간에 따라 어떻게 변화하는지를 모델링할 수 있지만, 여러 상호작용 엔터티 사이에서 변화하는 관계를 본질적으로 표현하지는 않는다. 시간 그래프 신경망은 두 차원을 명시적으로 모델링한다. 그래프 구조는 누가 누구와 상호작용하는지를 표현하고, 시간 메커니즘은 상호작용이 언제 발생하며 그 영향이 어떻게 지속되는지를 표현한다.

이러한 결합으로 인해 시간 그래프 신경망(Temporal GNN)은 고립된 관측이 아니라 변화하는 관계를 이해하는 것이 지능의 핵심이 되는 시스템에 적합하다. 시간 그래프 신경망은 그래프를 정적인 관계 지도(static relational map)에서 상호작용하는 엔터티와 그 이력을 표현하는 동적 계산 표현(dynamic computational representation)으로 변화시킨다. 고급 그래프 AI(Advanced Graph AI)에서 이는 구조적 이해를 넘어 지속적으로 변화하는 시스템에 대한 예측(prediction), 적응(adaptation), 추론(reasoning)으로 발전하는 데 필요한 시간적 차원을 제공한다.

## 07.03. Scalable GNN

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

확장 가능한 그래프 신경망(Scalable Graph Neural Networks)은 수백만 또는 수십억 개의 노드(node)와 엣지(edge)를 포함하는 데이터셋에 그래프 학습(graph learning)을 적용할 때 발생하는 문제를 해결한다. 기존 그래프 신경망(GNN)은 이웃 노드의 정보를 반복적으로 집계하는 방식으로 동작하며 작은 그래프에서는 효과적이지만, 그래프 크기와 연결성이 증가하면 계산 비용이 급격히 증가한다. 따라서 확장성(scalability)을 확보하려면 핵심 그래프 구조를 유지하면서 데이터 접근, 메시지 전달(message passing), 메모리 관리, 학습, 분산 실행을 함께 재설계해야 한다.

확장성 문제의 근본적인 원인은 이웃 확장(neighborhood expansion)에서 발생한다. 다층 그래프 신경망에서 각 노드는 자신의 이웃으로부터 정보를 집계하고, 다시 그 이웃들은 자신의 이웃에 의존하기 때문에 네트워크 깊이가 증가하면서 수용 영역(receptive field)이 빠르게 커진다. 평균 노드 차수가 d이고 모델이 L개의 계층을 가진다면 잠재적으로 dᴸ에 가까운 노드에 접근할 수 있으며, 이러한 현상을 일반적으로 이웃 폭발(neighborhood explosion)이라고 한다.

전체 배치 학습(full-batch training)은 모든 최적화 단계에서 전체 그래프를 처리한다. 이 방식은 정확한 이웃 집계(neighborhood aggregation)를 제공하며 비교적 작은 그래프에서는 실용적이지만, 대규모 인접 구조(adjacency structure), 노드 특징(node feature), 중간 임베딩(intermediate embedding), 그래디언트(gradient)가 GPU 메모리 용량을 초과할 수 있다. 그래프가 호스트 메모리에 저장되더라도 CPU와 GPU 사이의 반복적인 데이터 전송이 주요 성능 병목이 될 수 있다.

이웃 샘플링(neighbor sampling)은 가장 널리 사용되는 해결 방법 중 하나이다. 대상 노드의 모든 이웃을 수집하는 대신 각 GNN 계층에서 제한된 수의 이웃만 샘플링한다. 그래프세이지(GraphSAGE)는 미니배치 학습(mini-batch learning)을 위해 제한된 계산 이웃(computational neighborhood)을 구성하는 이러한 접근을 대중화하였다. 샘플링을 사용하면 잠재적으로 거대한 수용 영역을 메모리와 계산 요구량을 제어할 수 있는 관리 가능한 계산 그래프로 변환할 수 있다.

계층별 샘플링(layer-wise sampling)은 각각의 대상 노드에서 독립적으로 이웃을 확장하는 대신 각 계층에서 노드 또는 엣지를 선택하는 또 다른 전략이다. 서로 다른 대상 노드가 동일한 중간 노드를 공유하는 경우가 많기 때문에 중복 계산을 줄일 수 있다. 중요도 샘플링(importance sampling)을 기반으로 하는 방법은 구조적 또는 통계적으로 중요한 노드를 우선적으로 선택하여 계산 효율성과 근사 정확도 사이의 균형을 개선할 수 있다.

서브그래프 샘플링(subgraph sampling)은 계산 단위를 개별 대상 노드의 이웃에서 샘플링된 그래프 영역으로 변경한다. 각각의 학습 반복에서 하나의 서브그래프(subgraph)를 선택하고 그 영역 내부에서 일반적인 메시지 전달을 수행한다. 그래프세인트(GraphSAINT)와 같은 방법은 노드, 엣지 또는 랜덤 워크 서브그래프(random-walk subgraph)를 샘플링하며, 클러스터 기반 방법은 원래 그래프를 밀접하게 연결된 영역으로 분할하여 미니배치로 처리한다.

그래프 클러스터링(graph clustering)은 원래 네트워크가 강한 커뮤니티 구조(community structure)를 가지고 있을 때 특히 효과적이다. 클러스터-GCN(Cluster-GCN)은 중요한 엣지가 개별 클러스터 내부에 최대한 유지되도록 그래프를 분할하여 각 학습 배치 외부와의 통신을 줄인다. 연결성을 개선하기 위해 여러 클러스터를 결합할 수도 있다. 핵심 목표는 동시에 처리해야 하는 그래프 데이터의 양을 제한하면서 유용한 로컬 그래프 구조를 유지하는 것이다.

샘플링은 원래 이웃의 일부만 각각의 갱신에 사용하기 때문에 근사 오차(approximation error)를 발생시킨다. 작은 샘플은 속도와 메모리 효율성을 향상시키지만 중요한 관계 정보를 누락할 수 있으며, 큰 샘플은 더 많은 계산 비용을 사용하는 대신 전체 이웃 계산에 가까워진다. 따라서 샘플링 전략은 정확도, 분산(variance), 계산량, 메모리, 그래프 연결성 사이의 균형을 결정하는 중요한 모델 설계 요소가 된다.

확장성은 단순히 노드의 수에 의해서만 결정되지 않는다. 엣지 밀도(edge density), 차수 분포(degree distribution), 특징 차원(feature dimensionality), 관계 다양성(relation diversity), 그래프 동역학(graph dynamics)은 계산 요구량을 크게 변화시킬 수 있다. 허브 노드(hub node)를 포함하는 그래프는 특히 어려운데, 소수의 고도로 연결된 정점이 거대한 이웃을 생성할 수 있기 때문이다. 차수 인식 샘플링(degree-aware sampling)과 고차수 노드의 특수 처리를 통해 이러한 허브가 계산을 지배하는 것을 방지할 수 있다.

대규모 GNN 시스템은 효율적인 그래프 저장(graph storage)에도 크게 의존한다. 압축 희소 행(compressed sparse row)과 같은 희소 인접 표현(sparse adjacency representation)을 사용하면 거대한 밀집 인접 행렬(dense adjacency matrix)을 저장하지 않아도 된다. 노드 특징은 CPU 메모리에 유지하면서 자주 접근하는 임베딩을 GPU에 캐싱할 수 있다. 특징 분할(feature partitioning), 메모리 매핑(memory mapping), 압축(compression), 비동기 데이터 로딩(asynchronous data loading)을 사용하면 학습 파이프라인을 통한 그래프 정보 이동 비용을 추가로 줄일 수 있다.

그래프 샘플링은 불규칙하고 메모리 집약적인 경우가 많은 반면 신경망 변환은 GPU 계산에 적합하기 때문에 CPU-GPU 조정(CPU-GPU coordination)이 매우 중요하다. CPU가 모든 미니배치를 구성할 때까지 GPU가 기다려야 한다면 가속기 활용률(accelerator utilization)이 크게 감소한다. 따라서 고성능 시스템은 파이프라이닝(pipelining)과 비동기 처리(asynchronous processing)를 통해 샘플링, 특징 수집(feature gathering), 메모리 전송, GPU 실행을 서로 중첩하여 처리한다.

그래프가 단일 머신의 처리 용량을 초과하면 분산 그래프 학습(distributed graph training)이 필요하다. 그래프를 여러 작업자(worker)로 분할하고 각각의 작업자는 노드, 엣지, 특징의 일부를 저장한다. 로컬 메시지 전달(local message passing)은 효율적으로 수행할 수 있지만 파티션 경계(partition boundary)를 통과하는 엣지는 머신 사이의 통신을 필요로 한다. 따라서 효과적인 그래프 분할은 계산 부하의 균형을 유지하면서 파티션 간 통신량을 최소화하는 것을 목표로 한다.

분산 실행(distributed execution)은 계산(computation)과 통신(communication) 사이의 새로운 균형 문제를 발생시킨다. GPU나 머신 수를 증가시키면 처리 능력이 향상되지만 그래프의 의존성 때문에 작업자 사이에서 노드 특징, 임베딩 또는 그래디언트를 교환해야 한다. 잘못된 그래프 분할은 통신 비용이 신경망 계산 자체보다 커지게 만들 수 있다. 따라서 확장성은 토폴로지 인식 분할(topology-aware partitioning), 캐싱, 통신 스케줄링, 효율적인 집합 통신 연산(collective operation)에 크게 의존한다.

데이터 병렬 처리(data parallelism)는 여러 작업자에 GNN 모델을 복제하고 각각의 장치에 서로 다른 그래프 미니배치를 할당할 수 있다. 최적화 과정에서는 기존의 분산 딥러닝과 유사하게 그래디언트를 동기화한다. 그러나 그래프 작업은 서로 다른 미니배치가 동일한 원격 노드에 접근할 수 있기 때문에 불규칙한 데이터 의존성을 추가로 가진다. 따라서 분산 샘플링(distributed sampling)과 특징 검색(feature retrieval)을 그래디언트 동기화와 함께 조정해야 한다.

그래프 분할(graph partitioning)은 지역성(locality)에 따라 그래프 영역을 할당함으로써 모델 실행을 지원할 수도 있다. 자주 상호작용하는 노드는 가능하면 동일한 작업자에 배치하여 원격 특징 요청(remote feature request)을 줄이는 것이 바람직하다. 균형 잡힌 파티션은 노드 수, 엣지 수, 계산 복잡도, 메모리 요구량을 동시에 고려해야 한다. 이종 그래프(heterogeneous graph)의 경우 노드 및 관계 유형의 분포까지 유지해야 할 수 있다.

추론(inference)은 학습과는 다른 확장성 문제를 가진다. 대규모 운영 그래프(production graph)는 노드 특징과 엣지가 지속적으로 변화하는 동시에 연속적인 예측 요청을 받을 수 있다. 비교적 안정적인 그래프에서는 임베딩을 사전 계산(precomputation)하여 지연시간을 줄일 수 있지만, 동적 시스템에서는 증분 갱신(incremental update)이 필요할 수 있다. 미니배치 추론, 임베딩 캐시, 분산 서빙(distributed serving), 선택적 재계산(selective recomputation)을 통해 최신성, 처리량, 지연시간 사이의 균형을 맞출 수 있다.

분리형 GNN 구조(decoupled GNN architecture)는 확장성을 확보하는 또 다른 방법을 제공한다. 일부 방법은 그래프 전파(graph propagation)와 신경망 특징 변환(neural feature transformation)을 분리하여 계산 비용이 높은 이웃 집계를 사전 계산하거나 단순화한다. 다른 구조에서는 메시지 전달 계층의 수를 줄이거나 반복적인 전파를 캐시된 표현(cached representation)으로 대체한다. 이러한 설계는 긴밀하게 결합된 그래프 계산의 일부를 희생하는 대신 계산 효율성을 크게 향상시킨다.

대규모 그래프는 최적화(optimization) 측면에서도 어려움을 발생시킨다. 샘플링은 그래디언트 분산(gradient variance)을 증가시킬 수 있고, 파티션은 구조적 편향(structural bias)을 발생시킬 수 있으며, 드문 노드나 관계는 충분한 갱신을 받지 못할 수 있다. 따라서 평가는 예측 정확도뿐만 아니라 학습 처리량(training throughput), GPU 활용률, 메모리 사용량, 통신량, 수렴 속도(convergence speed), 추론 지연시간, 서로 다른 노드 차수 범위에서의 성능까지 측정해야 한다.

시간 그래프(temporal graph)와 이종 그래프(heterogeneous graph)는 확장성 문제를 더욱 어렵게 만든다. 시간 그래프 신경망(Temporal GNN)은 과거 이웃을 검색하고 변화하는 메모리를 유지해야 하며, 이종 그래프 신경망(Heterogeneous GNN)은 유형 인식 샘플링(type-aware sampling)과 관계별 계산(relation-specific computation)을 필요로 한다. 따라서 확장 가능한 구조는 예측에 가장 중요한 정보를 유지하면서 시간적 이력과 관계별 이웃을 동시에 제한해야 할 수 있다.

추천 시스템(recommendation system)에서 확장 가능한 GNN은 사용자, 제품, 상호작용, 카테고리, 행동 이벤트를 포함하는 거대한 그래프를 처리할 수 있다. 샘플링을 통해 각 사용자의 과거 상호작용 처리량을 제한하고, 분산 임베딩 테이블(distributed embedding table)을 통해 대규모 엔터티 표현을 저장할 수 있다. 효율적인 검색과 캐싱을 사용하면 전체 상호작용 그래프를 하나의 가속기에 적재하지 않고도 관계형 추천 모델을 운영할 수 있다.

로보틱스, 교통, 스마트 인프라(robotics, transportation, smart infrastructure)는 또 다른 형태의 확장성 문제를 제공한다. 그래프는 로봇, 지도 영역, 도로, 센서, 작업, 시설, 누적된 과거 관측을 포함하면서 계속 커질 수 있다. 모든 제어 주기(control cycle)마다 모든 관계를 처리할 필요는 없다. 로컬 서브그래프(local subgraph), 계층적 표현(hierarchical representation), 시간 윈도우(temporal window), 선택적 메시지 전달(selective message passing)을 통해 현재 운영 상황과 관련된 엔터티에 계산을 집중할 수 있다.

플릿 규모 로보틱스(fleet-scale robotics)에서는 이러한 원리가 특히 중요하다. 개별 로봇은 작은 로컬 그래프(local graph)를 기반으로 추론하고, 엣지 또는 온프레미스 시스템(edge or on-premise system)은 더 큰 사이트 또는 플릿 그래프(site or fleet graph)의 정보를 통합할 수 있다. 계층적 그래프 처리(hierarchical graph processing)는 모든 로봇이 모든 관측 정보를 다른 모든 로봇과 교환하는 것을 방지한다. 대신 관계 정보는 적절한 공간적, 시간적, 조직적 규모에서 집계될 수 있다.

따라서 확장 가능한 GNN(Scalable GNN)의 설계는 하나의 알고리즘이 아니라 샘플링(sampling), 분할(partitioning), 희소 저장(sparse storage), 캐싱(caching), 분산 계산(distributed computation), 효율적인 메모리 이동(memory movement), 아키텍처 선택(architecture selection)을 결합하는 시스템 수준의 접근이다. 핵심 목표는 계산량의 증가를 제어하면서 유용한 관계형 추론(relational reasoning)을 유지하는 것이다. 이러한 관점은 고급 그래프 AI(Advanced Graph AI)를 이후의 샘플링 방법, 분산 학습, GPU 최적화, 실시간 그래프 처리(real-time graph processing)와 같은 엔지니어링 및 배포 주제로 자연스럽게 연결한다.

## 07.04. Graph Transformers

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

그래프 트랜스포머(Graph Transformers)는 어텐션 기반 표현 학습(attention-based representation learning)과 명시적인 관계 구조(explicit relational structure)를 결합하여 트랜스포머(Transformer) 아키텍처를 그래프 구조 데이터(graph-structured data)로 확장한다. 일반적인 트랜스포머는 순서가 있는 토큰 시퀀스(token sequence)를 처리하지만, 그래프는 불규칙한 이웃 구조, 임의의 연결성, 보편적인 노드 순서가 없는 특성을 가진다. 따라서 그래프 트랜스포머는 순열 인식 그래프 처리(permutation-aware graph processing)를 유지하면서 노드, 엣지, 토폴로지, 구조적 관계를 어텐션에 어떻게 반영할 것인지를 결정해야 한다.

핵심 아이디어는 기존의 이웃 집계(neighborhood aggregation)를 셀프 어텐션(self-attention)으로 대체하거나 보완하는 것이다. 전통적인 그래프 신경망(GNN)에서는 노드가 주로 직접 연결된 이웃으로부터 메시지를 전달받는다. 그래프 트랜스포머에서는 어텐션을 통해 어떤 노드들이 서로 관련되어 있으며 그 정보가 어느 정도의 강도로 상호작용해야 하는지를 결정할 수 있다. 이를 통해 사전에 정의된 집계 함수에만 의존하지 않고 관계적 의존성(relational dependency)을 유연하게 학습할 수 있다.

일반적인 셀프 어텐션(self-attention)은 쿼리(query), 키(key), 값(value) 사이의 관계를 계산한다. 노드 i의 표현으로부터 쿼리 qᵢ를 생성하고, 다른 노드들은 키 kⱼ와 값 vⱼ를 생성한다. 어텐션 점수(attention score)는 qᵢ와 kⱼ 사이의 적합성을 측정하며, 정규화된 점수를 통해 값이 어느 정도 집계될지를 결정한다. 그래프 트랜스포머는 이 원리를 유지하면서 그래프 토폴로지, 엣지 특징, 위치 정보 또는 구조 인코딩(structural encoding)을 사용하여 어텐션을 수정한다.

근본적인 문제는 그래프에는 텍스트에서 사용할 수 있는 자연스러운 위치 순서(positional ordering)가 존재하지 않는다는 것이다. 언어 모델에서는 토큰 위치를 통해 첫 번째 단어와 열 번째 단어를 구별할 수 있지만, 그래프에서 노드 인덱스(node index)는 일반적으로 임의적이다. 따라서 그래프 트랜스포머는 절대적인 시퀀스 위치에 의존하기보다 그래프 토폴로지를 기준으로 노드가 어디에 위치하는지를 설명하는 구조 또는 위치 인코딩(structural or positional encoding)을 필요로 한다.

라플라시안 위치 인코딩(Laplacian positional encoding)은 이러한 문제를 해결하는 하나의 접근법이다. 그래프 라플라시안(graph Laplacian)에서 계산된 고유벡터(eigenvector)는 그래프의 전역적인 구조 패턴을 반영하는 좌표를 제공한다. 이러한 벡터를 노드 특징에 더하거나 연결하여 트랜스포머에 그래프 위치 정보를 제공할 수 있다. 그러나 대규모 그래프에 적용할 때는 고유벡터의 모호성(eigenvector ambiguity), 계산 비용, 확장성(scalability)을 함께 고려해야 한다.

다른 구조 인코딩(structural encoding) 방법에서는 최단 경로 거리(shortest-path distance), 랜덤 워크 통계(random-walk statistics), 노드 차수(node degree), 중심성(centrality), 로컬 서브그래프 특성(local subgraph property)을 사용한다. 최단 경로 인코딩은 한 홉 떨어진 노드와 여러 홉 떨어진 노드를 구별할 수 있게 한다. 차수 정보는 로컬 연결성을 나타내고 랜덤 워크 특징은 확산 행동(diffusion behavior)을 포착한다. 이러한 신호는 일반적인 트랜스포머 어텐션만으로는 얻기 어려운 구조적 문맥(structural context)을 제공한다.

엣지 정보(edge information) 역시 중요한 구성요소이다. 기존 트랜스포머에서는 토큰 사이의 관계를 주로 토큰 표현과 위치로부터 추론하지만, 그래프는 거리, 관계 유형, 결합 유형, 통신 품질, 방향, 용량 등의 도메인별 속성을 설명하는 명시적인 엣지 특징(edge feature)을 제공할 수 있다. 그래프 트랜스포머는 이러한 엣지 특징을 어텐션 점수나 값 변환(value transformation)에 직접 반영할 수 있다.

어텐션은 로컬 그래프 이웃(local graph neighborhood)으로 제한하거나 그래프 전체에 전역적으로 적용할 수 있다. 로컬 어텐션(local attention)은 각 노드가 인접하거나 가까운 노드에만 어텐션을 적용하기 때문에 메시지 전달과 유사하며 계산 비용을 줄이고 그래프의 지역성을 유지한다. 전역 어텐션(global attention)은 모든 노드가 다른 모든 노드와 상호작용할 수 있게 하여 기존의 얕은 GNN이 표현하기 어려운 장거리 의존성(long-range dependency)을 포착할 수 있다.

그러나 전역 어텐션(global attention)은 중요한 확장성 문제를 발생시킨다. N개의 노드에 대한 일반적인 셀프 어텐션은 모든 노드가 다른 모든 노드에 어텐션할 수 있기 때문에 대략 이차 복잡도(quadratic complexity)를 가진다. 따라서 대규모 그래프에서 제한 없는 전역 어텐션은 지나치게 높은 계산 비용을 요구할 수 있다. 이를 제어하기 위해 희소 어텐션(sparse attention), 이웃 어텐션, 계층적 처리(hierarchical processing), 그래프 분할, 샘플링, 근사 어텐션(approximate attention) 등이 사용된다.

장거리 관계(long-range relationship)를 모델링할 수 있다는 점은 그래프 트랜스포머를 사용하는 주요 이유 중 하나이다. 기존 메시지 전달 GNN은 그래프의 먼 영역까지 정보를 전달하려면 여러 계층을 거쳐야 한다. 깊은 전파(deep propagation)는 정보가 서로 구별되지 않게 되는 과도한 평활화(over-smoothing) 또는 좁은 그래프 병목을 통해 정보가 압축되는 과도한 압축(over-squashing)을 발생시킬 수 있다. 어텐션은 멀리 떨어져 있지만 관련성이 높은 노드 사이에 보다 직접적인 정보 경로를 형성할 수 있다.

그래포머(Graphormer)는 구조 정보를 트랜스포머 어텐션에 직접 통합하는 대표적인 설계를 보여준다. 공간 인코딩(spatial encoding)은 그래프 거리를 표현하고, 중심성 인코딩(centrality encoding)은 노드의 중요도 또는 차수 관련 특성을 나타내며, 엣지 인코딩(edge encoding)은 그래프 경로상의 관계 정보를 제공한다. 이는 성공적인 그래프 트랜스포머가 일반적인 트랜스포머를 순서 없는 노드 집합에 단순히 적용하는 것 이상의 구조적 설계를 필요로 한다는 점을 보여준다.

그래프GPS(GraphGPS)는 로컬 메시지 전달(local message passing)과 전역 어텐션(global attention)을 결합하는 또 다른 중요한 아키텍처 방향을 보여준다. 로컬 GNN 구성요소는 이웃 구조를 효율적으로 포착하고, 트랜스포머 구성요소는 더 넓은 그래프 수준의 의존성을 모델링한다. 이러한 하이브리드 설계(hybrid design)는 로컬 관계형 귀납 편향(local relational inductive bias)과 전역 정보 교환(global information exchange)이 서로를 완전히 대체하는 것이 아니라 상호 보완적인 장점을 제공한다는 점을 반영한다.

그래프 트랜스포머는 어텐션을 노드 유형과 관계 유형에 의존하도록 구성하여 이종 그래프(heterogeneous graph)를 지원할 수도 있다. 유형별 쿼리, 키, 값 투영(type-specific query, key, value projection)을 사용하면 서로 다른 엔터티 범주가 어텐션 과정에 서로 다른 방식으로 참여할 수 있다. 관계 의존적 매개변수(relation-dependent parameter)를 사용하면 출발 노드와 목적 노드 유형 사이의 상호작용을 추가로 조정할 수 있다. 따라서 이종 그래프 트랜스포머(Heterogeneous Graph Transformer)는 의미적 유형 정보와 유연한 어텐션 기반 관계 학습을 결합한다.

시간 정보(temporal information)도 그래프 어텐션에 통합할 수 있다. 시간 그래프 트랜스포머(Temporal Graph Transformer)는 타임스탬프(timestamp), 경과 시간(elapsed time), 이벤트 순서(event order), 과거 상호작용 패턴을 구조 정보와 함께 인코딩할 수 있다. 이에 따라 어텐션은 어떤 노드들이 관계되어 있는지뿐만 아니라 해당 상호작용이 언제 발생했는지에도 의존하게 된다. 이는 동적인 관계 시스템을 모델링하기 위해 그래프 트랜스포머와 시간 그래프 신경망(Temporal GNN)을 연결한다.

그래프 수준 예측(graph-level prediction)을 수행하려면 많은 노드 표현을 하나의 통합된 표현으로 변환해야 한다. 풀링(pooling), 특수 그래프 토큰(special graph token), 어텐션 기반 판독(attention-based readout), 계층적 집계(hierarchical aggregation)를 사용하여 트랜스포머 처리 이후의 노드 정보를 요약할 수 있다. 생성된 그래프 임베딩(graph embedding)은 전체 분자, 장면, 네트워크 또는 관계 시스템을 대상으로 하는 분류, 회귀, 생성, 유사도 추정 등의 작업에 사용할 수 있다.

노드 수준에서 그래프 트랜스포머는 분류(classification), 상태 추정(state estimation), 이상 탐지(anomaly detection), 표현 학습(representation learning)을 수행할 수 있다. 엣지 수준에서는 링크 예측(link prediction), 관계 분류(relation classification), 그래프 완성(graph completion)을 지원할 수 있다. 또한 유연한 어텐션 메커니즘을 통해 여러 엔터티 사이의 관계를 엄격한 로컬 전파 방식이 아니라 공동으로 고려해야 하는 그래프 생성(graph generation)과 추론 작업에도 활용할 수 있다.

분자 및 과학 그래프(molecular and scientific graph)는 원자나 구성요소 사이의 상호작용이 로컬 결합과 더 넓은 구조적 문맥 모두에 의존할 수 있기 때문에 주요 적용 분야이다. 지식 그래프(knowledge graph)는 관계 인식 어텐션(relation-aware attention)을 이용하여 엔터티와 술어(predicate)에 걸친 추론을 수행할 수 있다. 추천 시스템은 사용자, 아이템, 행동, 문맥 관계를 통합할 수 있으며, 교통 그래프는 도로, 교차로, 차량, 인프라 사이의 상호작용을 모델링할 수 있다.

로보틱스 및 공간 AI(Robotics and Spatial AI)에서는 객체, 로봇, 위치, 센서, 작업, 지도 요소를 서로 상호작용하는 엔터티로 표현할 수 있다. 로컬 엣지는 물리적 근접성(physical proximity), 가시성(visibility), 내비게이션 연결성(navigation connectivity)을 표현할 수 있고, 보다 넓은 범위의 어텐션은 작업 수준 또는 장면 수준의 의존성을 포착할 수 있다. 이러한 결합은 즉각적인 공간 관계와 멀리 떨어진 문맥 정보가 동시에 의사결정에 영향을 미치는 경우에 유용하다.

다중 로봇 시스템(multi-robot system)은 로컬 추론과 전역 추론을 결합하는 가치가 특히 잘 드러나는 사례이다. 각각의 로봇은 주로 주변 장애물, 자원, 다른 로봇과 상호작용하지만, 플릿 수준의 협업(fleet-level coordination)은 멀리 떨어진 작업 할당, 혼잡, 충전 가능성, 공유 목표에 의존할 수 있다. 계층적 또는 희소 그래프 트랜스포머(hierarchical or sparse Graph Transformer)는 효율적인 로컬 추론을 유지하면서 더 큰 운영 그래프에 걸쳐 필요한 정보만 선택적으로 교환할 수 있다.

그래프 트랜스포머의 학습에는 구조적 편향(structural bias), 계산 복잡도(computational complexity), 일반화(generalization)를 신중하게 고려해야 한다. 지나치게 강한 구조 인코딩은 어텐션의 유연성을 제한할 수 있고, 반대로 그래프 정보가 부족하면 모델이 일반적인 집합 트랜스포머(set Transformer)처럼 동작할 수 있다. 효과적인 아키텍처는 학습된 어텐션과 그래프 특화 귀납 편향(graph-specific inductive bias) 사이의 균형을 유지하여 토폴로지가 학습을 안내하면서도 완전히 결정하지 않도록 한다.

확장성(scalability)은 여전히 핵심적인 엔지니어링 제약 중 하나이다. 효율적인 그래프 트랜스포머는 점차 희소 어텐션, 샘플링, 클러스터링, 분산 계산(distributed computation), 메모리 효율적인 커널(memory-efficient kernel), 로컬-전역 아키텍처(local-global architecture)를 결합하는 방향으로 발전하고 있다. 대규모 관계 시스템에서는 학습이나 추론의 모든 단계에서 모든 노드 사이에 제한 없는 전대전 어텐션(all-to-all attention)을 적용할 수 없기 때문에 이러한 기술은 확장 가능한 GNN(Scalable GNN)의 원리와 직접 연결된다.

그래프 트랜스포머(Graph Transformers)는 그래프 학습(graph learning)과 파운데이션 모델 아키텍처(foundation-model architecture)가 서로 융합되는 더 넓은 흐름을 나타낸다. 그래프의 관계 구조를 핵심 정보원으로 유지하면서 유연한 어텐션, 심층 표현 학습(deep representation learning), 대규모 사전학습(large-scale pretraining)의 가능성을 그래프 영역으로 가져온다. 따라서 그래프 트랜스포머는 전문화된 GNN 아키텍처에서 그래프 파운데이션 모델(Graph Foundation Models)로 발전하는 중요한 연결 지점이며, 여러 데이터셋, 도메인, 작업에 걸쳐 전이 가능한 그래프 표현(transferable graph representation)을 학습하기 위한 기반을 제공한다.

## 07.05. Foundation Models for Graphs

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

그래프 파운데이션 모델(Foundation Models for Graphs)은 언어, 비전, 멀티모달 데이터에서 발전한 파운데이션 모델(foundation-model) 패러다임을 그래프 구조 정보(graph-structured information)로 확장한다. 각각의 데이터셋과 작업마다 별도의 그래프 신경망(Graph Neural Network)을 학습하는 대신, 크고 다양한 그래프 집합을 대상으로 폭넓은 능력을 가진 그래프 모델을 사전학습(pretraining)하는 것이 목표이다. 이렇게 학습된 표현은 훨씬 적은 작업별 학습만으로도 다운스트림 작업(downstream task)에 전이(transfer), 적응(adaptation), 프롬프팅(prompting)될 수 있다.

이러한 접근의 동기는 기존 그래프 학습(graph learning)의 한계에서 비롯된다. 많은 GNN은 하나의 그래프, 하나의 도메인, 하나의 예측 목표를 위해 학습되기 때문에 특정 데이터셋에서 획득한 지식을 다른 환경에서 재사용하기 어렵다. 그래프 파운데이션 모델(Graph Foundation Model)은 연결성(connectivity), 커뮤니티(community), 모티프(motif), 계층 구조(hierarchy), 상호작용(interaction), 구조적 역할(structural role) 등 서로 다른 그래프에서 반복적으로 나타나는 패턴을 학습하여 재사용 가능한 관계 지식(relational knowledge)을 획득하고자 한다.

그래프는 도메인마다 구조가 크게 다르기 때문에 일반적인 토큰 시퀀스보다 더욱 어려운 파운데이션 모델 문제를 제시한다. 분자 그래프(molecular graph)는 원자와 결합으로 구성되고, 지식 그래프(knowledge graph)는 엔터티와 술어(predicate)로 구성되며, 로보틱스 그래프(robotics graph)는 로봇, 객체, 장소, 센서, 작업을 포함할 수 있다. 따라서 범용 모델은 노드 의미, 엣지 의미, 특징 공간(feature space), 토폴로지(topology), 규모, 예측 목표의 차이를 모두 수용할 수 있어야 한다.

그래프 사전학습(graph pretraining)은 이러한 전이 가능한 표현을 구축하는 핵심 메커니즘이다. 대규모 그래프 집합이나 하나의 거대한 네트워크는 완전한 수작업 레이블 없이도 자기지도학습(self-supervised learning)을 위한 신호를 제공할 수 있다. 모델은 사전학습 과정에서 유용한 구조적 및 의미적 표현을 학습하고, 이후 다운스트림 적응을 통해 이러한 표현을 분류, 링크 예측, 그래프 예측, 생성, 추론, 제어 관련 응용에 특화할 수 있다.

마스킹 그래프 모델링(masked graph modeling)은 중요한 사전학습 전략 중 하나이다. 선택된 노드 특징, 엣지 속성 또는 구조적 요소를 숨긴 후 모델이 주변 문맥으로부터 누락된 정보를 복원하도록 학습한다. 이는 마스킹 언어 모델링(masked language modeling)과 유사하지만 단순한 순차 문맥이 아니라 관계적 문맥(relational context)을 필요로 한다. 성공적인 복원 학습은 엔터티 속성과 그래프 구조 사이의 의존성을 모델이 학습하도록 유도한다.

대조 학습(contrastive learning)은 또 다른 주요 접근법이다. 동일한 노드, 서브그래프 또는 전체 그래프에서 생성된 서로 다른 증강 뷰(augmented view)가 유사한 표현을 생성하도록 하고, 관련되지 않은 예제는 표현 공간에서 서로 분리되도록 학습한다. 그래프 증강(graph augmentation)은 특징 마스킹, 엣지 제거, 서브그래프 샘플링, 토폴로지 변경 등을 사용할 수 있지만 그래프의 의미적 정체성이 우연히 손상되지 않도록 신중하게 설계해야 한다.

예측 목적(predictive objective)을 이용하여 누락되거나 미래의 관계 정보를 추론하도록 그래프 모델을 학습할 수도 있다. 모델은 엣지의 존재 여부, 관계 유형, 이웃 노드의 속성을 예측하거나 서브그래프를 복원하고 그래프의 미래 변화를 예측할 수 있다. 이러한 학습 목표는 단순히 노드 특징을 암기하는 대신 관계적 규칙성(relational regularity)을 포착하도록 유도하며, 그래프 사전학습을 링크 예측(link prediction) 및 시간 그래프 학습(temporal graph learning)과 자연스럽게 연결한다.

그래프 트랜스포머(Graph Transformers)는 어텐션(attention)을 통해 유연한 상호작용과 대규모 사전학습을 지원할 수 있기 때문에 그래프 파운데이션 모델(Graph Foundation Model)의 중요한 후보 아키텍처이다. 구조 인코딩(structural encoding), 엣지 표현(edge representation), 로컬 메시지 전달(local message passing), 전역 어텐션(global attention)을 트랜스포머 아키텍처에 통합할 수 있다. 이러한 결합은 표현력 높은 관계 모델링과 대규모 트랜스포머 기반 모델에서 발전한 최적화 및 확장 기술을 연결한다.

주요 과제 중 하나는 서로 다른 그래프 도메인에서 사용할 수 있는 공통 표현 인터페이스(common representation interface)를 개발하는 것이다. 텍스트 모델은 재사용 가능한 토큰 어휘(token vocabulary)를 사용하지만 그래프의 노드는 서로 완전히 다른 개념을 나타낼 수 있다. 따라서 그래프 파운데이션 모델은 도메인별 특징 인코더(domain-specific feature encoder)와 공유 그래프 백본(shared graph backbone)을 결합할 수 있다. 인코더는 원시 노드와 엣지 정보를 공통 잠재 표현(common latent representation)으로 변환하고, 이후 전이 가능한 그래프 계층이 이를 처리한다.

구조 정보(structural information)는 도메인에 독립적인 지식의 또 다른 원천이 될 수 있다. 노드 차수(node degree), 연결성, 최단 경로 거리(shortest-path distance), 모티프, 커뮤니티, 중심성(centrality), 확산 패턴(diffusion pattern)과 같은 개념은 노드 특징이 서로 다르더라도 다양한 그래프 유형에서 의미를 가진다. 이러한 구조적 속성을 기반으로 사전학습하면 서로 다른 의미적 어휘를 가진 도메인 사이에서도 전이할 수 있는 일반적인 관계 원리(relational principle)를 학습하는 데 도움이 된다.

이종 그래프(heterogeneous graph)는 이러한 문제를 더욱 복잡하게 만들지만 동시에 더욱 강력한 표현 능력을 제공한다. 파운데이션 모델은 모든 그래프가 동일한 스키마(schema)를 사용한다고 가정하지 않고 여러 노드 유형과 관계 유형을 처리해야 할 수 있다. 유형 임베딩(type embedding), 관계 인코더(relation encoder), 의미 어댑터(semantic adapter), 스키마 인식 어텐션(schema-aware attention)을 통해 유연한 인터페이스를 제공할 수 있다. 목표는 의미 있는 차이를 유지하면서 서로 다른 그래프 스키마 사이에서 공통 관계 패턴을 공유하는 것이다.

시간 그래프(temporal graph)는 관계 변화의 전이 가능한 패턴까지 학습해야 한다는 요구사항을 추가한다. 그래프 파운데이션 모델은 정적인 연결성뿐 아니라 상호작용이 시간에 따라 생성되고, 사라지고, 강화되고, 약화되거나 전파되는 방식을 학습할 수 있다. 시간 이벤트(temporal event)를 기반으로 한 사전학습은 예측(forecasting), 이상 탐지(anomaly detection), 동적 링크 예측(dynamic link prediction), 변화하는 상태 추정(evolving state estimation) 등 과거의 관계 문맥이 중요한 작업을 지원할 수 있다.

규모(scale) 역시 핵심 요구사항이다. 파운데이션 모델은 일반적으로 대규모 데이터셋과 높은 모델 용량에서 이점을 얻지만, 그래프 계산은 모델 크기를 증가시키기 이전부터 많은 계산 비용을 요구할 수 있다. 이웃 폭발(neighborhood explosion), 불규칙한 메모리 접근, 전역 어텐션 복잡도, 분산 그래프 통신은 중요한 문제가 된다. 따라서 샘플링, 분할(partitioning), 희소 어텐션(sparse attention), 효율적인 그래프 저장, 캐싱, 분산 학습(distributed training)이 실용적인 대규모 사전학습의 기반이 된다.

전이 학습(transfer learning)은 사전학습이 실제로 유용한 파운데이션 모델을 만들어냈는지를 결정한다. 사전학습된 그래프 인코더(graph encoder)는 레이블이 있는 다운스트림 데이터로 미세조정(fine-tuning)할 수 있으며, 매개변수 효율적 접근(parameter-efficient approach)은 어댑터(adapter), 선택된 계층 또는 작은 작업별 모듈만 갱신할 수 있다. 다운스트림 데이터가 제한적인 경우 강력한 사전학습 표현은 전체 그래프 모델을 처음부터 학습하는 것보다 필요한 지도 데이터의 양을 줄이고 적응 성능을 향상시킬 수 있다.

프롬프트 기반 그래프 학습(prompt-based graph learning)은 프롬프팅(prompting) 패러다임을 그래프 모델로 확장하려는 접근이다. 각각의 작업을 위해 네트워크를 다시 설계하고 전체를 재학습하는 대신 추가 특징, 가상 노드(virtual node), 구조적 패턴, 학습 가능한 프롬프트 벡터(prompt vector), 텍스트 지시(textual instruction)를 통해 작업 문맥을 제공할 수 있다. 그러나 그래프에는 자연어 토큰과 같은 범용 인터페이스가 존재하지 않기 때문에 그래프에서 프롬프팅이 정확히 무엇을 의미하는지는 여전히 발전 중이다.

제로샷 및 퓨샷 그래프 학습(zero-shot and few-shot graph learning)은 그래프 파운데이션 모델의 중요한 장기 목표이다. 퓨샷 환경에서는 소수의 레이블 예제만 사용하여 새로운 그래프 작업에 모델을 적응시키고, 제로샷 학습에서는 전이 가능한 의미 및 구조 지식을 이용하여 이전에 보지 못한 작업이나 범주를 처리하고자 한다. 이를 안정적으로 구현하려면 그래프 표현, 작업 설명, 그리고 필요에 따라 다른 모달리티 사이의 정렬(alignment)이 필요하다.

대규모 언어 모델(Large Language Model)은 이러한 정렬을 위한 중요한 경로를 제공한다. 실제 그래프의 많은 노드와 엣지에는 설명, 레이블, 문서, 지시, 메타데이터 등의 텍스트 정보가 연결되어 있다. 그래프 인코더와 대규모 언어 모델을 결합하면 구조적 관계와 의미 추론(semantic reasoning)을 연결할 수 있다. 그래프는 명시적인 관계적 기반(relational grounding)을 제공하고, 언어 모델은 광범위한 개념 지식과 지시 수행 능력(instruction-following capability)을 제공한다.

그래프-언어 모델(graph-language model)은 여러 방식으로 정보를 교환할 수 있다. 그래프 임베딩을 언어 모델의 표현 공간으로 투영하거나, 그래프 구조를 구조화된 텍스트 설명으로 변환하거나, 특수한 교차 어텐션 모듈(cross-attention module)을 통해 그래프와 언어 표현을 연결할 수 있다. 여기서 핵심 설계 과제는 모든 관계 구조를 길고 모호할 수 있는 텍스트 시퀀스로 단순 변환하지 않고 그래프 토폴로지 자체를 보존하는 것이다.

멀티모달 그래프 파운데이션 모델(Multimodal Graph Foundation Models)은 이러한 개념을 더욱 확장한다. 노드는 이미지, 텍스트, 센서 측정값, 기하학 정보, 비디오 또는 다른 모달리티를 포함할 수 있으며, 그래프 엣지는 이들 사이의 관계를 나타낸다. 통합 모델은 모달리티별 인코더(modality-specific encoder)와 관계형 백본(relational backbone)을 결합하여 의미적 콘텐츠와 구조적 문맥이 서로 영향을 주도록 할 수 있다. 이는 특히 체화 시스템(embodied system)과 물리적 시스템(physical system)에 중요하다.

로보틱스 및 공간 AI(Robotics and Spatial AI)에서 그래프 파운데이션 모델은 로봇, 객체, 장소, 지도, 센서, 작업, 상호작용 전반에서 재사용 가능한 표현을 학습할 수 있다. 서로 다른 환경은 서로 다른 그래프 인스턴스를 생성하지만 근접성(proximity), 도달 가능성(reachability), 가시성(visibility), 소유 관계(ownership), 작업 의존성(task dependency), 자원 경쟁(resource competition)과 같은 반복적인 관계 개념은 공유될 수 있다. 따라서 다양한 환경에 걸친 사전학습은 새로운 배포 환경에 재사용할 수 있는 관계형 사전 지식(relational prior)을 제공할 수 있다.

플릿 로보틱스(fleet robotics)는 이러한 개념을 개별 환경에서 집단 경험(collective experience)으로 확장한다. 여러 로봇이 생성한 그래프에는 내비게이션, 혼잡, 충전, 작업 할당, 고장, 인간 상호작용과 관련된 반복적인 패턴이 포함될 수 있다. 충분히 확장 가능한 그래프 모델은 축적된 관계 경험으로부터 학습하고 새로운 로봇이나 사이트로 지식을 전이할 수 있으며, 이를 통해 그래프 파운데이션 모델과 집단 관계형 지능(collective relational intelligence)을 연결할 수 있다.

평가(evaluation)는 하나의 벤치마크에서 얻은 성능만을 측정해서는 안 된다. 진정한 파운데이션 모델은 서로 다른 데이터셋, 그래프 크기, 스키마, 도메인, 작업 유형에 걸친 전이 능력을 보여주어야 한다. 사전학습이 샘플 효율성(sample efficiency)을 향상시키는지, 분포 변화(distribution shift)에서도 표현이 유용한지, 새로운 노드 또는 관계 유형을 수용할 수 있는지, 적응을 위해 전체 재학습이 필요한지 아니면 제한된 매개변수 갱신만으로 가능한지를 평가해야 한다.

그래프 파운데이션 모델(Graph Foundation Models)은 아직 하나의 표준화된 아키텍처로 확립된 분야가 아니라 계속 발전하고 있는 연구 방향이다. 이를 구현하려면 GNN, 그래프 트랜스포머(Graph Transformers), 이종 및 시간 그래프 학습, 자기지도학습, 확장 가능한 학습(scalable training), 멀티모달 모델, 언어 모델의 발전을 결합해야 한다. 핵심 목표는 일관되다. 개별 작업에 특화된 그래프 모델에서 벗어나 다양한 그래프 구조 환경에서 학습하고, 전이하고, 적응하며, 추론할 수 있는 재사용 가능한 관계형 모델(reusable relational model)로 발전하는 것이다.

## 07.06. Multi Modal Graphs

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

멀티모달 그래프(Multi-Modal Graphs)는 노드(node), 엣지(edge), 그래프 수준 문맥(graph-level context)이 하나의 특징 공간(feature space)이 아니라 여러 모달리티(modality)의 정보를 포함할 수 있도록 그래프 학습(graph learning)을 확장한다. 텍스트, 이미지, 비디오, 오디오, 기하학 정보(geometry), 센서 신호(sensor signal), 구조화된 속성(structured attribute)이 하나의 관계형 표현(relational representation) 안에 공존할 수 있다. 그래프는 엔터티 사이의 명시적 관계를 제공하고, 모달리티별 데이터는 해당 엔터티가 무엇을 포함하고 관측하며 전달하는지를 설명한다.

핵심적인 동기는 많은 실제 시스템을 하나의 모달리티만으로는 충분히 이해할 수 없다는 점이다. 로봇은 RGB 이미지, 깊이(depth), 라이다(LiDAR), 언어 레이블(language label), 물리적 측정값을 통해 하나의 객체를 관측할 수 있으며, 사회적 엔터티(social entity)는 텍스트, 이미지, 프로필, 상호작용 이력을 포함할 수 있다. 이러한 신호를 독립적으로 표현하면 관계적 문맥(relational context)을 잃을 수 있지만, 멀티모달 그래프는 이종 콘텐츠와 관계를 하나의 통합된 계산 구조로 구성한다.

유용한 개념적 표현에서는 각각의 노드 v에 모달리티별 관측(modality-dependent observation) xᵥ¹, xᵥ², \..., xᵥᴹ을 연결한다. 모든 노드가 반드시 모든 모달리티를 포함하는 것은 아니며 특징 차원도 서로 호환되지 않을 수 있다. 따라서 모달리티별 인코더(modality-specific encoder)를 사용하여 원시 입력을 잠재 표현(latent representation)으로 변환한다. 예를 들어 이미지에는 비전 인코더(vision encoder), 텍스트에는 언어 인코더(language encoder), 오디오, 기하학, 센서 측정에는 각각 특화된 신경망을 사용할 수 있다.

모달리티별 인코딩 이후 생성된 표현은 의미 있는 상호작용이 가능한 공간으로 정렬(alignment)되어야 한다. 간단한 방법은 임베딩(embedding)을 연결하거나 더하는 것이며, 보다 정교한 시스템에서는 학습된 투영(learned projection), 게이팅(gating), 어텐션(attention), 교차 어텐션(cross-attention)을 사용한다. 목표는 단순히 서로 다른 모달리티를 동일한 크기의 벡터로 만드는 것이 아니라 의미적으로 관련된 관측이 그래프 추론에 함께 참여할 수 있는 호환 가능한 표현을 생성하는 것이다.

융합(fusion)은 아키텍처의 여러 단계에서 수행될 수 있다. 초기 융합(early fusion)은 그래프 메시지 전달(message passing) 이전에 모달리티 특징을 결합하여 GNN이 이미 통합된 노드 정보를 전파하도록 한다. 중간 융합(intermediate fusion)은 각각의 모달리티가 특화된 표현을 유지하면서 그래프 계층을 통해 정보를 교환하도록 한다. 후기 융합(late fusion)은 모달리티를 독립적으로 처리한 뒤 출력에 가까운 단계에서 예측이나 임베딩을 결합하여 높은 모듈성을 제공하지만 초기 단계의 상호작용은 상대적으로 약하다.

그래프 구조(graph structure) 자체도 중요한 융합 메커니즘(fusion mechanism)으로 작동한다. 모든 모달리티를 다른 모든 모달리티와 전역적으로 결합하는 대신, 관계를 통해 어떤 정보가 서로 상호작용해야 하는지를 지정한다. 객체와 연결된 이미지 특징은 해당 객체의 텍스트 설명과 정보를 교환할 수 있고, 센서 측정값은 공간적 또는 물리적 엣지를 통해 전파될 수 있다. 따라서 토폴로지(topology)는 의미 있는 관계 구조에 따라 멀티모달 통합을 제한하고 조직화한다.

교차 모달 어텐션(cross-modal attention)은 어떤 모달리티가 다른 모달리티에 어느 정도 영향을 미쳐야 하는지를 결정하는 유연한 메커니즘을 제공한다. 한 모달리티에서 생성된 쿼리(query)가 다른 모달리티의 키(key)와 값(value)에 어텐션할 수 있으며, 이를 통해 텍스트가 관련된 시각 영역을 검색하거나 센서 표현이 대응되는 기하학적 엔터티를 강조할 수 있다. 그래프 토폴로지와 결합하면 명시적 관계에 따라 어텐션을 제한하거나 편향시켜 구조 인식 교차 모달 추론(structure-aware cross-modal reasoning)을 수행할 수 있다.

멀티모달 그래프는 이종 그래프(heterogeneous graph)와 밀접하게 관련되어 있지만 두 개념이 동일한 것은 아니다. 이종 그래프는 서로 다른 노드 및 관계 유형을 강조하는 반면, 멀티모달 그래프는 엔터티와 관계에 연결된 서로 다른 정보 모달리티를 강조한다. 하나의 시스템은 이종 그래프이면서 동시에 멀티모달 그래프일 수 있다. 로봇 노드, 객체 노드, 위치 노드는 서로 다른 의미 유형을 가지면서 동시에 비전, 언어, 기하학, 센서 정보를 포함할 수 있다.

누락 모달리티(missing modality)는 중요한 실용적 문제이다. 어떤 노드는 이미지를 가지고 있지만 텍스트가 없을 수 있고, 센서가 일시적으로 고장날 수도 있으며, 과거 그래프 데이터에는 일부 측정값만 존재할 수 있다. 강건한 모델(robust model)은 모든 모달리티가 항상 존재한다고 가정해서는 안 된다. 마스킹(masking), 모달리티 드롭아웃(modality dropout), 학습 가능한 누락 모달리티 임베딩, 복원 목적(reconstruction objective), 교차 모달 예측(cross-modal prediction)을 사용하여 일부 입력이 없어도 유용한 표현을 유지할 수 있다.

모달리티 불균형(modality imbalance) 역시 어려운 문제를 발생시킨다. 고차원의 시각 임베딩이 작은 수치 특징을 압도하거나 풍부한 텍스트 정보가 희소한 센서 관측보다 학습을 지배할 수 있다. 정규화(normalization), 모달리티별 투영, 어텐션, 게이팅, 손실 균형(loss balancing), 제어된 샘플링(controlled sampling)을 통해 모델이 가장 쉽거나 통계적으로 지배적인 하나의 모달리티에만 의존하는 것을 방지할 수 있다.

자기지도학습(self-supervised learning)은 멀티모달 그래프가 자연스럽게 교차 모달 지도 신호(cross-modal supervision)를 제공하기 때문에 특히 중요하다. 모델은 마스킹된 노드 속성을 예측하거나, 누락된 모달리티를 복원하거나, 두 관측이 동일한 엔터티에 대응하는지를 판단하거나, 일치하는 그래프 문맥과 일치하지 않는 그래프 문맥을 대조할 수 있다. 이러한 목적을 통해 작업별 지도학습을 적용하기 전에 대규모 비레이블 관계 데이터를 활용하여 표현을 학습할 수 있다.

대조 학습(contrastive learning)은 서로 다른 모달리티의 표현을 정렬하는 데 사용할 수 있다. 객체 이미지와 해당 객체의 텍스트 설명은 잠재 공간(latent space)에서 서로 가까운 위치에 배치되도록 학습하고, 관련되지 않은 이미지-텍스트 쌍은 서로 멀어지도록 학습할 수 있다. 그래프 구조는 직접적인 모달리티 쌍을 넘어 연결된 엔터티, 공유 이웃(shared neighborhood), 대응되는 서브그래프를 통해 의미 있는 추가 양성 신호(positive signal)를 제공한다.

그래프 트랜스포머(Graph Transformers)는 어텐션이 엔터티와 모달리티 표현 전반에서 작동할 수 있기 때문에 멀티모달 그래프를 위한 자연스러운 아키텍처를 제공한다. 노드 토큰(node token), 모달리티 토큰(modality token), 엣지 특징, 구조 인코딩(structural encoding)을 함께 처리하면서 어텐션을 통해 관련된 상호작용을 결정할 수 있다. 로컬 그래프 어텐션(local graph attention)은 관계적 지역성을 유지하고, 선택적 전역 어텐션(selective global attention)은 멀리 떨어진 엔터티나 모달리티 사이의 더 넓은 의존성을 포착할 수 있다.

대규모 언어 모델(Large Language Models)은 멀티모달 그래프의 텍스트 구성요소에 대한 의미 추론(semantic reasoning)을 제공할 수 있다. 노드 설명, 작업 지시, 객체 레이블, 문서, 메타데이터를 언어 모델로 인코딩하여 그래프 표현과 연결할 수 있다. 반대로 그래프 임베딩은 대규모 언어 모델에 관계적 기반(relational grounding)을 제공하여 긴 텍스트 설명만으로 다시 구성해야 했던 명시적인 연결 관계를 직접 활용할 수 있도록 한다.

비전-언어-그래프 통합(vision-language-graph integration)은 이러한 관계를 더욱 확장한다. 비전 인코더는 객체, 영역, 장면, 기하학적 특징을 식별하고, 언어 인코더는 설명과 지시를 표현하며, 그래프는 이들 사이의 관계를 정의한다. 결과적으로 시스템은 어떤 객체가 다른 객체 옆에 있는지, 어떤 영역에 도달할 수 있는지, 어떤 엔터티가 언어로 정의된 작업 조건을 만족하는지와 같은 관계적 문제를 추론할 수 있다.

시간 정보(temporal information) 역시 또 하나의 모달리티 또는 구조적 차원으로 포함될 수 있다. 비디오 프레임, 센서 스트림, 이벤트 이력, 변화하는 관계는 시간에 따라 변화하는 관측을 생성한다. 시간 모델링과 멀티모달 그래프 추론을 결합하면 서로 다른 센서가 현재 무엇을 관측하는지뿐만 아니라 이전 관측 과정에서 엔터티와 관계가 어떻게 변화했는지도 표현할 수 있다.

멀티모달 그래프는 과학 및 생의학 응용(scientific and biomedical applications)에서 특히 중요하다. 분자 구조는 그래프 연결성과 화학적 기술자(chemical descriptor), 기하학적 좌표, 실험 측정값, 이미지, 텍스트 지식을 결합할 수 있다. 생의학 그래프는 환자, 유전자, 단백질, 질병, 이미지, 임상 텍스트를 연결할 수 있다. 관계형 융합(relational fusion)을 사용하면 각각의 모달리티를 독립적으로 분석할 때 발견하기 어려운 의존성을 파악할 수 있다.

추천 및 지식 시스템(recommendation and knowledge systems)도 멀티모달 그래프 학습의 이점을 활용할 수 있다. 제품은 이미지, 텍스트 설명, 카테고리, 리뷰, 상호작용 이력을 포함할 수 있으며, 사용자는 행동 신호와 선호도를 제공한다. 지식 엔터티는 텍스트 설명, 이미지, 구조화된 속성, 관계 링크를 함께 포함할 수 있다. 그래프 학습은 콘텐츠 유사성(content similarity)과 상호작용 구조가 예측에 공동으로 기여하도록 한다.

로보틱스 및 공간 AI(Robotics and Spatial AI)는 멀티모달 그래프가 중요한 이유를 가장 명확하게 보여주는 분야 중 하나이다. 로봇은 카메라, 라이다(LiDAR), 깊이 센서(depth sensor), 레이더(radar), 관성 측정 장치(IMU), 지도, 언어 지시, 내부 상태 측정값을 통해 세계를 인식한다. 이러한 신호들은 동일한 물리적 엔터티와 위치를 참조한다. 그래프는 센서 관측을 객체, 장소, 로봇 상태, 작업, 내비게이션 관계와 명시적으로 연결할 수 있다.

따라서 공간 장면 그래프(spatial scene graph)는 환경에 대한 멀티모달 메모리(multimodal memory)가 될 수 있다. 하나의 객체 노드는 시각 임베딩(visual embedding), 의미 레이블(semantic label), 3차원 위치(3D position), 추정 속도, 불확실성(uncertainty), 과거 관측을 포함할 수 있다. 엣지는 근접성, 가시성, 지지 관계(support), 포함 관계(containment), 도달 가능성, 움직임 관계를 나타낼 수 있다. 메시지 전달은 물리적 장면의 관계 구조를 유지하면서 이러한 신호를 통합한다.

다중 로봇 시스템(multi-robot systems)은 멀티모달 그래프를 여러 에이전트(agent)로 확장한다. 서로 다른 로봇은 서로 다른 센서, 관측 시점(viewpoint), 연산 능력, 동일 환경에 대한 서로 다른 관측을 가질 수 있다. 각각의 로컬 그래프(local graph)는 공유 객체, 위치, 지도 또는 작업을 통해 정렬될 수 있다. 모든 원시 센서 스트림을 전송하는 대신 선택된 그래프 엔터티와 임베딩을 교환함으로써 보다 효율적인 집단 인지(collective perception)와 상황 인식(situational awareness)을 구현할 수 있다.

그래프 크기와 모달리티 크기가 동시에 증가하면 확장성(scalability)이 어려워진다. 이미지, 비디오, 포인트 클라우드(point cloud), 언어 임베딩은 일반적인 노드 속성보다 훨씬 많은 메모리를 요구할 수 있다. 따라서 실용적인 시스템에서는 모달리티 압축(modality compression), 선택적 인코딩(selective encoding), 계층적 그래프(hierarchical graph), 희소 어텐션(sparse attention), 샘플링, 캐싱, 분산 처리(distributed processing)가 필요하다. 계산 비용이 높은 원시 모달리티는 한 번 인코딩한 뒤 반복적인 그래프 계산에서는 압축된 특징으로 표현할 수 있다.

평가(evaluation)는 최종 예측 정확도만을 측정해서는 안 된다. 멀티모달 그래프 모델은 누락된 모달리티, 노이즈가 있는 센서, 서로 충돌하는 관측, 이전에 보지 못한 그래프 구조, 분포 변화(distribution shift) 환경에서도 평가되어야 한다. 교차 모달 정렬 품질(cross-modal alignment quality), 강건성(robustness), 보정(calibration), 계산 비용, 각 모달리티의 기여도가 중요하다. 제거 실험(ablation study)을 통해 그래프가 실제로 여러 모달리티를 통합하는지 또는 하나의 지배적인 정보원에만 의존하는지를 확인할 수 있다.

멀티모달 그래프(Multi-Modal Graphs)는 궁극적으로 서로 보완적인 두 가지 형태의 지능을 결합한다. 멀티모달 표현 학습(multimodal representation learning)은 서로 다른 정보 채널이 전달하는 콘텐츠가 무엇인지를 설명하고, 그래프 학습(graph learning)은 엔터티 사이에 어떤 관계가 존재하는지를 설명한다. 두 기술의 통합은 무엇이 관측되었는지와 그 관측들이 구조적으로 어떻게 연결되어 있는지를 함께 표현한다. 이러한 특성으로 인해 멀티모달 그래프는 고급 그래프 AI(Advanced Graph AI)의 자연스러운 완성 단계이자 관계형(relational), 멀티모달(multimodal), 물리적 기반 지능(physically grounded intelligence)을 구현하기 위한 중요한 기반이 된다.
