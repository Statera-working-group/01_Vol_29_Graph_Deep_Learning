**Volume 29. Graph Deep Learning**

# Chapter 02. Graph Representation

## 02.00. Overview

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

그래프 표현(Graph Representation)은 개체(Entity), 관계, 맥락 정보를 그래프 학습 알고리즘이 처리할 수 있는 구조화된 형태로 변환하는 과정입니다. 고정된 길이의 벡터, 시퀀스(Sequence), 또는 규칙적인 격자로 배열된 전통적인 데이터와 달리, 그래프 데이터는 사물들의 모음을 그들 간의 연결과 함께 기술합니다. 따라서 그래프는 속성 정보와 관계적 구조를 모두 제공하여, 학습 시스템이 개별 개체가 무엇인지와 그것들이 어떻게 상호작용하는지를 추론할 수 있게 합니다.

그래프는 흔히 G=(V,E)로 표현되며, 여기서 V는 노드(Node)들의 집합을, E는 엣지(Edge)들의 집합을 나타냅니다. 실용적인 머신러닝 시스템에서 이러한 구조적 정의는 보통 노드 특성(Feature), 엣지 특성, 그래프 수준의 속성, 레이블(Label), 타임스탬프(Timestamp), 유형 정보로 확장됩니다. 그 결과로 나오는 표현은 위상(Topology)과 데이터를 결합하여, 추상적인 수학적 그래프를 그래프 신경망(Graph Neural Network)과 다른 그래프 학습 방법에 적합한 입력 구조로 변환합니다.

노드 특성은 개별 개체와 관련된 속성을 기술합니다. 사람을 나타내는 노드는 인구통계학적 또는 행동적 속성을 포함할 수 있는 반면, 문서를 나타내는 노드는 텍스트 임베딩(Embedding)이나 주제 정보를 포함할 수 있습니다. 로보틱스에서 노드는 사물, 위치, 센서, 로봇, 또는 랜드마크(Landmark)를 나타낼 수 있습니다. 그 특성은 위치, 속도, 의미론적 클래스, 신뢰도, 물리적 상태, 또는 학습된 잠재 표현(Latent Representation)을 부호화할 수 있어, 그래프 모델이 단순한 연결성을 넘어 개체에 대해 추론할 수 있게 합니다.

엣지 특성은 노드들 간의 관계에 대한 정보를 제공합니다. 엣지는 소셜 네트워크에서의 친구관계, 문서 네트워크에서의 인용, 인프라에서의 연결성, 사물들 간의 공간적 근접성, 또는 로봇들 간의 통신을 나타낼 수 있습니다. 관계는 또한 거리, 방향, 용량, 상호작용 빈도, 신뢰도, 또는 상대적 자세(Pose)와 같은 속성을 지닐 수 있습니다. 결과적으로, 엣지는 단순한 연결이 아니라 개체들이 서로에게 어떻게 영향을 미치거나 제약을 가하는지를 나타내는 정보가 풍부한 표현이 될 수 있습니다.

그래프 수준 특성은 개별 노드나 엣지가 아니라 전체 그래프에 적용되는 속성을 기술합니다. 이러한 정보에는 환경 조건, 시스템 상태, 분자 속성, 전역 레이블, 운영 모드, 또는 맥락적 메타데이터(Metadata)가 포함될 수 있습니다. 그래프 수준 표현은 예측 목표가 분자 속성 예측, 장면 분류, 인프라 평가, 또는 전체 상호작용 네트워크의 분류에서처럼 완전한 구조와 관련될 때 특히 중요합니다.

그래프 표현은 또한 이질적인 구조를 수용해야 합니다. 많은 실제 시스템은 단일한 노드 및 엣지 유형이 아니라 여러 범주의 개체와 관계를 포함합니다. 지식 그래프(Knowledge Graph)는 사람, 조직, 제품, 장소, 개념을 여러 의미론적 관계를 통해 연결할 수 있습니다. 로봇 장면 그래프(Scene Graph)는 마찬가지로 로봇, 사물, 방, 랜드마크, 작업을 연결할 수 있습니다. 이질적인 표현은 학습 알고리즘이 서로 다른 의미론적 관계를 적절하게 다룰 수 있도록 이러한 구별을 보존합니다.

또 다른 중요한 차원은 시간입니다. 정적 그래프는 노드, 엣지, 그리고 그 속성들이 분석 중에 고정된 채로 유지된다고 가정하지만, 많은 실용적인 네트워크는 지속적으로 진화합니다. 사회적 상호작용은 나타났다가 사라지고, 교통 네트워크는 상태가 변화하며, 금융 거래는 순차적으로 발생하고, 로봇은 변화하는 환경을 통과해 이동합니다. 따라서 동적 그래프 표현은 모델이 관계적 구조뿐만 아니라 그 구조가 어떻게 진화하는지도 학습할 수 있도록 타임스탬프, 시간적 사건, 스냅샷(Snapshot), 또는 지속적으로 갱신되는 특성을 포함합니다.

표현의 선택은 그래프 학습 모델이 회복할 수 있는 정보에 직접적으로 영향을 미칩니다. 동일한 연결성을 가진 두 그래프라도 그 노드와 엣지 속성이 다르면 완전히 다른 시스템을 기술할 수 있습니다. 반대로, 유사한 개체 속성이라도 그 관계적 위상이 변화하면 다른 행동을 만들어낼 수 있습니다. 따라서 효과적인 그래프 표현은 학습 목표를 개선하지 않으면서 계산 복잡도만 증가시키는 불필요한 정보를 피하면서, 특성의 의미와 구조적 관계를 모두 신중하게 보존할 것을 필요로 합니다.

그래프 데이터는 결국 학습과 추론에 적합한 계산적 구조로 변환되어야 합니다. 일반적인 구현은 엣지 리스트(List), 인접(Adjacency) 구조, 희소 행렬(Sparse Matrix), 또는 색인화된 텐서(Tensor)를 통해 연결성을 저장하는 반면, 속성은 노드, 엣지, 그래프 수준의 특성 텐서로 조직화됩니다. 그래프 데이터 파이프라인(Pipeline)은 원시 기록을 파싱(Parsing)하고, 식별자를 할당하고, 엣지를 구성하고, 특성을 생성하고, 속성을 정규화하고, 데이터셋(Dataset)을 분할하고, 그래프를 배치(Batch)로 묶고, 가속화된 처리를 위한 구조를 준비하는 작업을 수행합니다.

표현 설계는 또한 학습 목표에 크게 의존합니다. 노드 분류는 개별 노드의 속성을 예측하는 데 유용한 정보를 보존하는 표현을 필요로 하는 반면, 링크 예측(Link Prediction)은 관계와 이웃 구조를 강조합니다. 그래프 분류는 지역적 정보를 전역적 표현으로 효과적으로 집계할 것을 필요로 합니다. 이상 탐지(Anomaly Detection), 추천, 지식 그래프 완성, 공간적 추론, 그래프 생성을 포함한 다른 작업들은 위상, 속성, 레이블, 샘플링 전략에 서로 다른 요구사항을 부과합니다.

실제 세계의 데이터셋들은 그래프 표현의 다양성을 보여줍니다. 인용 네트워크는 논문을 노드로, 인용을 엣지로 모델링하고, 소셜 네트워크는 사용자와 그들의 상호작용을 나타내며, 지식 그래프는 의미론적 관계를 통해 연결된 개체를 부호화합니다. 이러한 범주들은 동일한 수학적 그래프 추상화가 공통된 그래프 학습 기법군을 뒷받침하면서도 근본적으로 다른 영역을 어떻게 나타낼 수 있는지를 이해하는 중요한 사례들을 형성합니다.

그래프 표현은 그래프 이론과 그래프 딥러닝(Deep Learning) 사이의 교량을 제공합니다. 고전적인 그래프 이론은 노드, 엣지, 인접, 차수(Degree), 연결성, 그래프 구조와 같은 개념을 확립하는 반면, 표현 엔지니어링은 그러한 구조에 기계가 읽을 수 있는 특성과 의미를 부여합니다. 이러한 표현이 확립되면, 그래프 합성곱(Convolution), 어텐션(Attention), 메시지 전달(Message Passing), 풀링(Pooling), 임베딩 방법이 지역적인 관계 정보를 이후의 예측과 추론을 위한 학습된 표현으로 변환할 수 있습니다.

이러한 진행 과정은 더 넓은 학습 순서에서 그래프 표현이 그래프 이론의 기초와 이후의 그래프 신경망 아키텍처(Architecture) 사이에 위치하는 이유를 설명합니다. 이 장은 GCN, GAT, 메시지 전달, 풀링, 그리고 현대적인 GNN 아키텍처로 나아가기 전에 노드 특성, 엣지 특성, 그래프 수준 특성, 이질적 그래프, 동적 그래프, 그래프 데이터 파이프라인, 대표적인 데이터셋을 확립합니다. 따라서 그래프 표현은 이후의 그래프 딥러닝 방법들이 작동하는 데이터 모델링의 기초입니다.

## 02.01. Node Features

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

노드 특징(Node Features)은 그래프(Graph)의 개별 정점(Vertex)에 연결된 속성(Attribute)으로, 그래프 위상(Graph Topology)을 보완하는 설명 정보를 제공한다. 엣지(Edge)가 어떤 개체(Entity)들이 서로 관계를 가지는지를 나타낸다면, 노드 특징은 각각의 개체가 무엇인지 또는 현재 어떤 상태에 있는지를 설명한다. 따라서 그래프 딥러닝(Graph Deep Learning)에서는 각 노드를 연결 구조 내의 위치뿐 아니라 수치형(Numerical), 범주형(Categorical), 의미적(Semantic), 또는 학습된(Learned) 속성의 집합으로 표현한다.

그래프 (G=(V,E))에서 노드 (v_i \\in V)는 특징 벡터(Feature Vector) (x_i)와 연결될 수 있다. 이러한 벡터들을 모으면 노드 특징 행렬(Node Feature Matrix) (X)가 만들어지며, 각 행(Row)은 하나의 노드에 대응하고 각 열(Column)은 하나의 특징 차원(Feature Dimension)을 나타낸다. 이 행렬은 그래프 학습 모델(Graph Learning Model)이 처리하는 초기 정보를 제공하며, 그래프 구조(Graph Structure)는 서로 다른 행의 정보가 관계를 통해 어떻게 상호작용하는지를 결정한다.

노드 특징은 표현 대상 개체에서 직접 측정할 수 있는 속성으로부터 생성될 수 있다. 소셜 네트워크(Social Network)에서 사용자 노드(User Node)는 프로필 특성(Profile Characteristics), 활동 통계(Activity Statistics), 관심사(Interests), 행동 지표(Behavioral Indicators)를 포함할 수 있다. 인용 네트워크(Citation Network)에서 논문 노드(Publication Node)는 키워드(Keywords), 문서 통계(Document Statistics), 주제 설명자(Topic Descriptors), 또는 문서 내용에서 생성된 텍스트 임베딩(Text Embedding)을 포함할 수 있다. 그래프 표현(Graph Representation)은 이러한 속성과 친구 관계나 인용과 같은 연결을 결합한다.

서로 다른 도메인(Domain)은 자연스럽게 서로 다른 특징 의미론(Feature Semantics)을 생성한다. 물리적 위치(Physical Location)를 나타내는 노드는 좌표(Coordinates), 범주(Category), 접근성(Accessibility), 점유 상태(Occupancy), 환경 조건(Environmental Conditions)을 포함할 수 있다. 로봇 노드(Robot Node)는 자세(Pose), 속도(Velocity), 운용 상태(Operating State), 배터리 상태(Battery Status), 센서 관측(Sensor Observations), 작업 정보(Task Information)를 포함할 수 있다. 생물학적 그래프(Biological Graph)에서는 노드 특징이 분자적(Molecular), 유전적(Genetic), 기능적(Functional) 속성을 나타낼 수 있다. 그래프 학습(Graph Learning)은 이러한 다양한 개체 설명을 관계와 함께 처리할 수 있는 공통 프레임워크(Common Framework)를 제공한다.

노드 특징은 연속형(Continuous), 이산형(Discrete), 범주형(Categorical), 이진형(Binary), 또는 다차원형(Multidimensional)이 될 수 있다. 위치(Position), 온도(Temperature), 거리(Distance), 속도(Velocity)와 같은 연속적인 값은 적절한 스케일링(Scaling)을 거친 후 직접 수치로 표현할 수 있다. 범주형 속성은 일반적으로 인코딩 방식(Encoding Scheme)이 필요하며, 이진 특징은 특정 특성의 존재 여부를 나타낼 수 있다. 텍스트(Text), 이미지(Image), 센서 신호(Sensor Signal)와 같은 복잡한 관측 정보는 그래프 모델에 입력되기 전에 압축된 특징 벡터(Feature Vector)로 변환되는 경우가 많다.

범주형 노드 정보(Categorical Node Information)는 가능한 범주의 수가 관리 가능한 수준일 경우 원-핫 인코딩(One-Hot Encoding)을 사용하여 표현할 수 있다. 각 범주에는 별도의 특징 차원(Feature Dimension)이 할당되고, 노드는 자신의 범주에 해당하는 차원을 활성화한다. 매우 큰 범주 공간(Categorical Space)에서는 식별자(Identifier)나 범주를 밀집 벡터(Dense Vector)로 매핑하는 학습 임베딩(Learned Embedding)이 더 효율적인 경우가 많다. 이러한 임베딩은 서로 독립적인 원-핫 차원만으로는 자연스럽게 표현하기 어려운 유사성(Similarity)을 학습할 수 있다.

일부 그래프에서는 유용한 노드 속성(Node Attribute)을 사용할 수 없거나 정보가 불완전할 수 있다. 이 경우 구조적 정보(Structural Information)가 초기 표현(Initial Representation)에 기여할 수 있다. 노드 차수(Node Degree), 이웃 통계(Neighborhood Statistics), 중심성 척도(Centrality Measures), 군집 특성(Clustering Characteristics), 위치 인코딩(Positional Encoding)과 같은 값은 그래프 내에서 노드가 담당하는 역할에 대한 신호를 제공한다. 이러한 구조적 특징(Structural Features)은 노드 자체의 고유 속성이 부족하지만 위치와 연결 패턴이 학습 작업에 중요한 정보를 포함하는 경우 특히 유용하다.

노드 특징은 다른 머신러닝 모델(Machine Learning Model)을 통해 생성될 수도 있다. 노드와 연결된 텍스트는 언어 인코더(Language Encoder)를 통해 변환될 수 있고, 이미지는 시각 인코더(Visual Encoder)를 이용하여 표현될 수 있으며, 센서 관측은 학습된 잠재 벡터(Learned Latent Vector)로 압축될 수 있다. 이를 통해 그래프 모델은 각 노드가 복잡한 정보를 요약하면서 그래프 엣지가 인코딩된 개체 사이의 관계를 지정하는 멀티모달 표현(Multimodal Representation)을 처리할 수 있다.

서로 다른 노드 속성은 호환되지 않는 스케일(Scale)과 통계적 분포(Statistical Distribution)를 가질 수 있기 때문에 특징 전처리(Feature Preprocessing)가 중요하다. 큰 수치 범위는 최적화(Optimization) 과정에서 작은 값을 압도할 수 있으며, 누락되거나 잡음이 많은 값은 표현 품질(Representation Quality)을 저하시킬 수 있다. 따라서 정규화(Normalization), 표준화(Standardization), 범주형 인코딩(Categorical Encoding), 결측값 처리(Missing-Value Handling), 차원 축소(Dimensionality Reduction), 특징 선택(Feature Selection)은 그래프 학습 아키텍처(Graph Learning Architecture)에 노드 특징을 제공하기 전에 수행되는 중요한 그래프 데이터 파이프라인(Graph Data Pipeline)의 일부가 될 수 있다.

노드 특징은 응용 분야에 따라 정적(Static)이거나 동적(Dynamic)일 수 있다. 영구적인 객체 범주(Object Category)는 그래프의 수명 동안 변하지 않을 수 있지만, 위치(Position), 속도(Velocity), 상호작용 상태(Interaction State), 센서 측정값(Sensor Measurements), 운용 조건(Operational Conditions)은 지속적으로 변화할 수 있다. 동적 그래프(Dynamic Graph)에서는 노드 (v_i)의 표현을 (x_i(t))로 나타낼 수 있으며, 이를 통해 동일한 개체가 자신의 정체성(Identity)을 유지하면서 관측 상태(Observable State) 또는 잠재 상태(Latent State)를 시간에 따라 변화시킬 수 있다.

노드 특징(Node Features)의 의미는 노드 레이블(Node Labels)과 구분해야 한다. 특징은 모델에 입력(Input)으로 제공되는 정보를 의미하는 반면, 레이블은 일반적으로 지도 학습(Supervised Learning)에서 사용되는 예측 대상(Prediction Target)을 나타낸다. 예를 들어 문서의 속성은 노드 특징을 구성하고, 해당 문서의 연구 분야 범주(Research Category)는 목표 레이블(Target Label)이 될 수 있다. 이러한 구분을 유지하는 것은 정보 누출(Information Leakage)을 방지하고 올바른 학습(Training), 검증(Validation), 테스트(Test) 절차를 구성하는 데 필수적이다.

그래프 신경망(Graph Neural Network, GNN)은 초기 노드 특징(Initial Node Features)을 점진적으로 학습된 노드 표현(Learned Node Representation)으로 변환한다. 메시지 패싱(Message Passing) 과정에서 각 노드는 이웃 노드(Neighboring Nodes)에서 생성된 정보를 받아들이고, 이러한 관계적 문맥(Relational Context)을 자신의 표현과 결합한다. 여러 계층(Layer)을 반복하면 점점 더 넓은 범위의 이웃 정보를 인코딩할 수 있다. 결과적으로 생성된 은닉 표현(Hidden Representation)은 노드의 원래 속성과 주변 그래프에서 수집된 구조적 정보를 함께 반영한다.

이러한 변환은 그래프 딥러닝(Graph Deep Learning)에서 노드 특징이 중요한 핵심 이유 중 하나이다. 초기 속성이 유사한 두 노드라도 서로 다른 관계적 문맥(Relational Context)에 위치하면 서로 다른 표현을 획득할 수 있으며, 반대로 처음에는 서로 다른 노드라도 주변 이웃이 유사한 정보를 제공한다면 비슷한 표현으로 변화할 수 있다. 따라서 그래프 학습은 개체의 의미가 주변의 개체와 관계에 부분적으로 의존하도록 함으로써 독립적인 특징 처리(Independent Feature Processing)를 넘어선다.

노드 특징의 품질은 노드 분류(Node Classification), 링크 예측(Link Prediction), 추천(Recommendation), 이상 탐지(Anomaly Detection), 군집화(Clustering), 그래프 분류(Graph Classification), 관계 추론(Relational Reasoning)과 같은 다운스트림 작업(Downstream Tasks)의 성능에 직접적인 영향을 미친다. 정보성이 높은 특징은 유용한 의미적 신호(Semantic Signal)를 제공하며, 그래프 위상(Graph Topology)은 이웃 집계(Neighborhood Aggregation)를 통해 이러한 신호를 더욱 풍부하게 만들 수 있다. 반대로 잘못 설계된 특징은 불필요한 변동을 추가하거나 그래프 연결만으로 복원할 수 없는 중요한 차이를 숨길 수 있다.

그래프 표현(Graph Representation)에서 노드 특징은 엣지 특징(Edge Features), 그래프 수준 특징(Graph-Level Features), 이종 노드 및 관계 유형(Heterogeneous Node and Relationship Types), 시간 정보(Temporal Information), 그래프 데이터 파이프라인(Graph Data Pipeline)을 포함하는 더 넓은 데이터 모델(Data Model)의 한 계층을 구성한다. 이러한 구성은 이후 그래프 합성곱(Graph Convolution), 어텐션(Attention), 메시지 패싱(Message Passing), 풀링(Pooling), 임베딩(Embedding) 기법을 적용하기 위한 데이터를 준비한다. 따라서 노드 표현(Node Representation)은 현실 세계의 개체를 그래프 딥러닝 시스템이 학습할 수 있는 계산 가능한 객체(Computational Object)로 변환하는 기본 메커니즘이다.

## 02.02. Edge Features

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

엣지 특징(Edge Features)은 그래프(Graph)에서 노드(Node) 쌍을 연결하는 관계(Relationship)의 속성을 설명한다. 노드 특징(Node Features)이 개별 개체(Entity)의 특성을 나타낸다면, 엣지 특징은 두 개체가 어떻게 관련되고, 상호작용하며, 서로 영향을 주는지를 설명한다. 엣지 특징은 연결(Connection)에 추가 정보를 부여하여 그래프 위상(Graph Topology)을 풍부하게 만들고, 단순한 노드 간 연결만으로는 동일하게 보이는 관계들을 그래프 학습 모델(Graph Learning Model)이 서로 구분할 수 있도록 한다.

그래프 (G=(V,E))에서 엣지 (e_{ij}\\in E)는 노드 (v_i)와 노드 (v_j)를 연결한다. 이 엣지는 해당 관계의 속성을 포함하는 특징 벡터(Feature Vector) (a_{ij})와 연결될 수 있다. 이러한 벡터들을 모으면 각 행(Row)이 하나의 엣지에 대응하고 각 열(Column)이 하나의 관계 속성을 나타내는 엣지 특징 행렬(Edge Feature Matrix)이 만들어진다. 이러한 표현을 통해 그래프 학습 알고리즘(Graph Learning Algorithm)은 연결성과 관계 속성을 함께 처리할 수 있다.

엣지 특징은 응용 도메인(Application Domain)에 따라 매우 다양한 정보를 표현할 수 있다. 소셜 네트워크(Social Network)에서는 상호작용 빈도(Interaction Frequency), 관계 강도(Relationship Strength), 통신 지속 시간(Communication Duration), 사용자 간 유사도(Similarity)를 포함할 수 있다. 인용 네트워크(Citation Network)에서는 인용 유형(Citation Type), 출판 시점 차이(Publication Time Difference), 문맥 정보(Contextual Information)를 포함할 수 있다. 교통 네트워크(Transportation Network)에서는 도로 길이(Road Length), 이동 시간(Travel Time), 용량(Capacity), 제한 속도(Speed Limit), 혼잡도(Congestion), 이동 방향(Direction of Movement)을 나타낼 수 있다.

물리적 및 공간적 시스템(Physical and Spatial Systems)은 정보성이 높은 엣지 속성(Edge Attributes)에 크게 의존하는 경우가 많다. 객체 사이의 관계에는 상대 위치(Relative Position), 방향(Orientation), 거리(Distance), 가시성(Visibility), 충돌 위험(Collision Risk), 공간적 제약(Spatial Constraints)이 포함될 수 있다. 로보틱스(Robotics)에서는 로봇, 랜드마크(Landmark), 센서(Sensor), 객체(Object), 위치(Location) 사이의 엣지가 상대 자세(Relative Pose), 통신 품질(Communication Quality), 근접성(Proximity), 주행 가능성(Traversability), 상호작용 상태(Interaction State)를 인코딩할 수 있다. 이러한 속성은 단순한 연결성만으로는 충분히 표현할 수 없는 물리적 관계를 그래프 모델이 나타낼 수 있도록 한다.

엣지 특징은 수치형(Numerical), 범주형(Categorical), 이진형(Binary), 벡터형(Vector-Valued), 또는 학습형(Learned)으로 구성될 수 있다. 수치형 속성에는 거리(Distance), 비용(Cost), 지연 시간(Latency), 확률(Probability), 용량(Capacity), 상호작용 빈도(Interaction Frequency) 등이 포함된다. 범주형 특징은 관계 유형(Relationship Type)을 식별할 수 있으며, 이진형 속성은 특정 조건의 존재 여부를 나타낼 수 있다. 더욱 복잡한 관계는 여러 속성을 하나의 통합된 엣지 표현(Edge Representation)으로 결합한 다차원 벡터(Multidimensional Vector)를 통해 표현할 수 있다.

엣지의 기본적인 속성 중 하나는 가중치(Weight)이다. 가중 그래프(Weighted Graph)는 엣지에 수치 값 (w_{ij})를 할당하며, 이는 유사도(Similarity), 중요도(Importance), 거리(Distance), 확률(Probability), 연결 강도(Connection Strength) 등을 나타낼 수 있다. 엣지 가중치는 노드 사이에서 정보가 얼마나 강하게 전달되는지를 결정함으로써 그래프 계산(Graph Computation)에 직접 영향을 줄 수 있다. 그러나 일반적인 엣지 특징 벡터(Edge Feature Vector)는 하나의 관계에 대한 여러 특성을 동시에 표현할 수 있으므로 단일 가중치보다 더 포괄적인 개념이다.

방향(Direction)은 엣지 표현(Edge Representation)의 또 다른 중요한 구성 요소이다. 무방향 그래프(Undirected Graph)에서는 (v_i)와 (v_j) 사이의 관계를 대칭적으로 처리하지만, 방향 그래프(Directed Graph)에서는 (e_{ij})와 (e_{ji})를 구분한다. 이러한 차이는 인용(Citation), 정보 흐름(Information Flow), 의존성(Dependency), 교통(Transportation), 명령 구조(Command Structure)와 같은 관계에서 필수적이다. 방향은 방향성 엣지(Directed Edge)를 통해 구조적으로 표현할 수 있으며, 각 방향의 상호작용 특성을 설명하는 특징을 추가할 수도 있다.

엣지 특징은 의미적 관계 유형(Semantic Relationship Type)을 인코딩할 수도 있다. 이종 그래프(Heterogeneous Graph)에서는 서로 다른 엣지가 근무 관계(Works-For), 위치 관계(Located-In), 소유 관계(Owns), 통신 관계(Communicates-With), 연결 관계(Connected-To), 의존 관계(Depends-On)와 같은 관계를 나타낼 수 있다. 동일한 유형의 노드를 연결하더라도 이러한 관계들은 근본적으로 서로 다른 의미를 가진다. 명시적인 관계 유형을 사용하면 이종 그래프 모델(Heterogeneous Graph Model)이 각 연결의 의미론(Semantics)에 따라 서로 다른 변환(Transformation)이나 메시지 패싱(Message Passing) 규칙을 적용할 수 있다.

시간에 따라 관계가 변화하는 경우에는 시간 정보(Temporal Information)가 특히 중요하다. 상호작용 엣지(Interaction Edge)는 타임스탬프(Timestamp), 지속 시간(Duration), 생성 시간(Creation Time), 만료 시간(Expiration Time), 순서 인덱스(Sequence Index)를 포함할 수 있다. 동적 그래프 시스템(Dynamic Graph System)은 이러한 정보를 사용하여 관계가 언제 존재했으며 그 속성이 어떻게 변화했는지를 판단할 수 있다. 따라서 엣지는 시간 의존적 관계(Time-Dependent Relationship) (e_{ij}(t))로 표현될 수 있으며, 특징 (a_{ij}(t))는 서로 다른 시점에서 변화하는 상태를 설명한다.

일부 엣지 특징은 직접 관측(Direct Observation)을 통해 얻어지지만, 다른 특징들은 노드 속성(Node Attributes)이나 외부 측정값(External Measurements)으로부터 유도될 수 있다. 예를 들어 노드 특징으로 저장된 지리 좌표(Geographic Coordinates)를 이용하여 연결된 노드 사이의 거리를 계산할 수 있다. 상대 속도(Relative Velocity)는 개별 노드의 운동 상태(Motion State)로부터 계산할 수 있으며, 유사도 점수(Similarity Score)는 노드 임베딩(Node Embedding)으로부터 생성할 수 있다. 이렇게 유도된 엣지 특징(Derived Edge Features)은 중요한 쌍별 관계(Pairwise Relationship)를 명시적으로 표현하여 그래프 모델이 이를 간접적으로 추론해야 하는 부담을 줄여준다.

노드 특징과 마찬가지로 엣지 속성도 학습 전에 전처리(Preprocessing)가 필요한 경우가 많다. 연속형 값(Continuous Values)은 정규화(Normalization) 또는 표준화(Standardization)가 필요할 수 있으며, 범주형 관계(Categorical Relationship)는 원-핫 인코딩(One-Hot Encoding)이나 학습 임베딩(Learned Embedding)을 사용할 수 있다. 누락된 엣지 값(Missing Edge Values) 역시 일관된 방식으로 처리해야 한다. 부적절한 스케일링이나 인코딩은 메시지 패싱과 최적화(Optimization) 과정에서 서로 다른 상호작용의 상대적 중요성을 왜곡할 수 있으므로 특징 변환(Feature Transformation)은 관계의 의미를 보존해야 한다.

엣지 특징은 메시지 패싱 신경망(Message Passing Neural Network, MPNN)에서 직접적인 역할을 수행한다. 모델은 이웃 노드의 표현만을 사용하여 메시지를 생성하는 대신 출발 노드(Source Node), 목적 노드(Destination Node), 그리고 이들을 연결하는 엣지의 정보를 함께 이용할 수 있다. 개념적으로 메시지는 (m_{ij}=\\phi(h_i,h_j,a_{ij}))와 같이 표현할 수 있으며, 여기서 (h_i)와 (h_j)는 노드 표현(Node Representation), (a_{ij})는 엣지 속성이다. 함수 (\\phi)는 관계 정보가 노드 사이의 정보 전달을 어떻게 변화시켜야 하는지를 학습한다.

이러한 엣지 인식 메시지 패싱(Edge-Aware Message Passing)은 훨씬 풍부한 관계 모델링(Relational Modeling)을 가능하게 한다. 동일한 이웃 노드라도 거리, 관계 유형, 신뢰도(Confidence), 방향, 상호작용 강도에 따라 서로 다른 방식으로 영향을 줄 수 있다. 가까운 객체는 멀리 있는 객체보다 로봇에 더 강한 영향을 줄 수 있으며, 고용량 통신 링크(High-Capacity Communication Link)는 신뢰성이 낮은 연결과 다른 의미를 가질 수 있다. 따라서 엣지 특징은 관계적 증거(Relational Evidence)가 어떻게 해석되고 전달되는지를 제어하는 문맥 정보(Contextual Information)의 역할을 한다.

엣지 특징의 중요성은 그래프 학습 작업(Graph Learning Task)에 따라 달라진다. 링크 예측(Link Prediction)은 관계의 존재 여부나 향후 변화를 추정하기 위해 엣지 특징을 사용할 수 있다. 추천 시스템(Recommendation System)은 상호작용 유형과 빈도를 표현할 수 있으며, 이상 탐지(Anomaly Detection)는 비정상적인 관계 패턴을 식별할 수 있다. 교통(Transportation), 분자(Molecular), 지식(Knowledge), 소셜(Social), 로보틱스(Robotics) 그래프에서는 연결된 개체의 정체성만큼 상호작용의 특성이 중요하기 때문에 엣지 속성에 크게 의존하는 경우가 많다.

전체 그래프 표현(Graph Representation) 구조에서 엣지 특징은 노드 특징(Node Features)과 그래프 수준 특징(Graph-Level Features)을 보완한다. 노드 특징은 개별 개체가 무엇인지를 설명하고, 엣지 특징은 이러한 개체들이 어떻게 관계를 맺는지를 설명하며, 그래프 수준 특징은 이들을 포함하는 전체적인 문맥(Global Context)을 나타낸다. 이종 유형(Heterogeneous Types), 시간 정보(Temporal Information), 그래프 위상(Graph Topology)과 함께 이러한 표현들은 이후의 그래프 합성곱(Graph Convolution), 어텐션(Attention), 메시지 패싱(Message Passing), 임베딩(Embedding), 추론(Reasoning) 기법에 필요한 구조화된 입력(Structured Input)을 제공한다.

## 02.03. Graph Level Features

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

그래프 수준 특징(Graph-Level Features)은 개별 노드(Node)나 엣지(Edge)가 아니라 전체 그래프(Graph)에 속하는 속성을 설명한다. 노드 특징(Node Features)이 개체(Entity)를 특성화하고 엣지 특징(Edge Features)이 관계(Relationship)를 특성화한다면, 그래프 수준 특징은 모델링되는 전체 구조에 대한 전역 문맥(Global Context)을 제공한다. 특히 여러 그래프가 각각 독립적인 샘플(Sample)로 취급되고 학습 목표가 각 그래프 전체의 특성에 의존하는 경우 중요하다.

그래프 (G=(V,E))에서 그래프 수준 정보(Graph-Level Information)는 전체 그래프와 연결된 특징 벡터(Feature Vector) (g)로 표현할 수 있다. 각 차원은 전역 측정값(Global Measurements), 메타데이터(Metadata), 환경 조건(Environmental Conditions), 운용 상태(Operating States), 기타 문맥 변수(Contextual Variables)를 인코딩할 수 있다. 그래프 학습 시스템(Graph Learning System)은 (g)를 노드 특징 행렬(Node Feature Matrix) (X) 및 엣지 속성(Edge Attributes)과 결합하여 지역적 개체, 관계, 전역 문맥을 동시에 포착하는 표현을 구성할 수 있다.

그래프 수준 특징(Graph-Level Features)은 신경망(Neural Network)이 학습한 그래프 수준 표현(Graph-Level Representation)과 구분해야 한다. 그래프 수준 특징은 일반적으로 입력 정보(Input Information)로 제공되는 반면, 학습된 그래프 표현(Learned Graph Representation) 또는 그래프 임베딩(Graph Embedding)은 노드와 엣지의 정보를 집계하여 생성된다. 두 가지를 결합할 수도 있으며, 외부에서 제공된 전역 속성은 알려진 문맥을 설명하고 학습된 임베딩은 그래프 자체에서 발견된 구조적·의미적 패턴을 요약한다.

많은 그래프 수준 특징은 그래프가 존재하는 환경이나 시스템을 설명하는 메타데이터(Metadata)에서 생성된다. 교통 그래프(Transportation Graph)는 날씨(Weather), 교통 시간대(Traffic Period), 지역 조건(Regional Conditions), 운용 모드(Operating Mode)를 포함할 수 있다. 로봇 장면 그래프(Robotic Scene Graph)는 임무 유형(Mission Type), 환경 범주(Environment Category), 플랫폼 상태(Platform State), 시나리오 정보(Scenario Information)를 포함할 수 있다. 이러한 변수들은 여러 노드와 엣지의 해석에 동시에 영향을 주므로 전역 그래프 수준에서 표현하는 것이 자연스럽다.

전역 구조 통계(Global Structural Statistics)도 그래프 수준 특징으로 사용할 수 있다. 예를 들어 노드 및 엣지 수(Number of Nodes and Edges), 그래프 밀도(Graph Density), 평균 차수(Average Degree), 차수 분포 통계(Degree Distribution Statistics), 연결 요소 수(Connected-Component Counts), 군집 계수(Clustering Coefficients), 지름(Diameter), 경로 길이 통계(Path-Length Statistics), 스펙트럼 속성(Spectral Properties) 등이 있다. 이러한 값은 예측 문제와 직접적으로 관련될 경우 원시 연결성(Raw Connectivity)만으로 효율적으로 복원하기 어려운 위상적 특성을 요약한다.

구조적 특징(Structural Features)의 유용성은 응용 분야(Application)에 따라 달라지므로 항상 유용하다고 가정해서는 안 된다. 노드 수(Node Count)는 특정 작업에서는 매우 중요한 정보가 될 수 있지만 다른 작업에서는 무관하거나 오히려 잘못된 신호를 제공할 수 있다. 마찬가지로 그래프 밀도나 평균 차수는 의도하지 않게 데이터셋 고유의 편향(Dataset-Specific Bias)을 드러낼 수 있다. 따라서 그래프 수준 특징 공학(Graph-Level Feature Engineering)에서는 전역 통계가 의미 있는 도메인 정보(Domain Information)를 나타내는지, 아니면 단순히 데이터 수집과 전처리 과정의 인공적 특성(Artifact)을 반영하는지를 판단해야 한다.

그래프 수준 특징은 수치형(Numerical), 범주형(Categorical), 이진형(Binary), 시간형(Temporal), 또는 다차원형(Multidimensional)이 될 수 있다. 연속형 값(Continuous Quantity)은 물리적 측정값이나 전역 통계를 나타낼 수 있고, 범주형 변수는 환경이나 운용 모드를 식별할 수 있으며, 이진 값은 전역적인 조건의 존재 여부를 나타낼 수 있다. 복잡한 그래프 문맥(Graph Context)은 밀집 벡터(Dense Vector)로 인코딩하여 여러 전역 속성을 하나의 통합된 수치 표현으로 모델에 제공할 수도 있다.

전체 그래프가 특정 시간 구간이나 시스템 상태에 대응하는 경우 시간 문맥(Temporal Context)도 그래프 수준 특징이 될 수 있다. 예를 들어 연속된 그래프들은 변화하는 네트워크의 스냅샷(Snapshot) (G_{t_1},G_{t_2},\...,G_{t_n})을 나타낼 수 있다. 각 스냅샷은 시간(Time), 환경 조건(Environmental Conditions), 작업 부하(Workload), 시나리오 상태(Scenario State)를 설명하는 전역 변수를 포함할 수 있다. 이를 통해 시간 그래프 모델(Temporal Graph Model)은 지역적 상호작용에 의해 발생한 구조 변화와 전역 조건에 의해 발생한 변화를 구분할 수 있다.

그래프 수준 레이블(Graph-Level Labels)은 그래프 수준 특징과 관련되어 있지만 개념적으로는 서로 다르다. 특징은 모델이 사용할 수 있는 입력 정보를 제공하는 반면, 레이블(Label)은 지도 학습(Supervised Learning) 과정에서 예측해야 하는 목표(Target)를 나타낸다. 분자 그래프(Molecular Graph)는 전역 실험 조건(Global Experimental Conditions)을 특징으로 포함하면서 분자 독성(Molecular Toxicity)을 예측 목표로 사용할 수 있다. 마찬가지로 시스템 그래프(System Graph)는 운용 조건을 입력으로 포함하고 고장 상태(Failure State) 또는 성능 등급(Performance Class)을 예측 레이블로 사용할 수 있다.

그래프 수준 학습(Graph-Level Learning)의 핵심 연산 가운데 하나는 리드아웃(Readout) 또는 풀링(Pooling)이며, 이는 가변적인 수의 노드 표현(Node Representations)을 고정 차원의 그래프 표현(Fixed-Dimensional Graph Representation)으로 변환한다. 일반적인 방법은 합(Sum), 평균(Mean), 최댓값(Maximum)과 같은 연산으로 노드 정보를 집계하며, 보다 발전된 방법에서는 어텐션 기반 집계(Attention-Based Aggregation)나 계층적 집계(Hierarchical Aggregation)를 학습한다. 생성된 벡터는 그래프 분류(Graph Classification), 회귀(Regression), 검색(Retrieval), 비교(Comparison), 다운스트림 추론(Downstream Reasoning)에 사용할 수 있다.

단순한 풀링 방법(Simple Pooling Methods)은 서로 다른 가정을 내포한다. 합 풀링(Sum Pooling)은 노드 수가 증가할수록 기여도가 누적되므로 그래프 크기(Graph Size)와 관련된 정보를 보존한다. 평균 풀링(Mean Pooling)은 노드들의 평균적인 특성을 강조하고 그래프 크기에 대한 직접적인 의존성을 줄인다. 최대 풀링(Max Pooling)은 그래프 내 어느 위치에서든 나타나는 강한 특징 활성화(Feature Activation)를 유지한다. 따라서 집계 방법(Aggregation Method)의 선택은 학습된 그래프 표현에서 어떤 전역 속성이 강조되는지에 영향을 준다.

집계 방법 자체를 고정하지 않고 학습할 경우 더욱 표현력이 높은 그래프 수준 표현을 얻을 수 있다. 어텐션 메커니즘(Attention Mechanism)은 현재 작업에 따라 서로 다른 노드에 다른 중요도를 부여할 수 있으며, 계층적 풀링(Hierarchical Pooling)은 전역 임베딩(Global Embedding)을 생성하기 전에 노드들을 점진적으로 상위 수준 구조로 그룹화할 수 있다. 이러한 방법을 통해 모델은 모든 노드를 동일하게 취급하는 대신 중요한 서브그래프(Subgraph), 커뮤니티(Community), 기능적 구성 요소(Functional Component), 공간 영역(Spatial Region)을 식별할 수 있다.

외부에서 제공된 그래프 수준 특징은 예측 전에 학습된 그래프 임베딩(Learned Graph Embedding)과 융합(Fusion)할 수 있다. 개념적으로 풀링된 표현 (h_G)는 전역 입력 벡터(Global Input Vector) (g)와 결합되어 ([h_G \\Vert g])와 같은 공동 표현(Joint Representation)을 생성할 수 있다. 이후 예측 네트워크(Prediction Network)는 노드와 엣지로부터 학습된 관계 정보와 전체 그래프를 설명하는 문맥 정보를 동시에 활용할 수 있다. 이는 전역 조건이 지역 패턴(Local Pattern)의 해석 방식에 영향을 미치는 경우 특히 유용하다.

그래프 수준 속성의 전처리(Preprocessing)는 노드 및 엣지 특징을 준비하는 원칙과 유사하다. 수치형 변수는 정규화(Normalization) 또는 표준화(Standardization)가 필요할 수 있고, 범주형 변수는 원-핫 인코딩(One-Hot Encoding)이나 학습 임베딩(Learned Embedding)을 사용할 수 있으며, 결측값(Missing Values)은 일관된 방식으로 처리해야 한다. 특히 전역 특징이 예측 레이블과 높은 상관관계를 갖는 정보를 포함할 경우 부적절한 특징 구성으로 정보 누출(Information Leakage)이 발생하여 평가 결과가 왜곡될 수 있으므로 주의해야 한다.

그래프 수준 특징과 표현은 전체 그래프가 예측의 기본 단위가 되는 응용 분야를 지원한다. 분자 그래프는 화학적 특성(Chemical Properties)에 따라 분류할 수 있고, 상호작용 그래프(Interaction Graph)는 행동 패턴(Behavioral Patterns)에 따라 분석할 수 있으며, 인프라 네트워크(Infrastructure Network)는 운용 상태(Operational State)에 따라 평가할 수 있다. 로보틱스(Robotics)와 공간 인공지능(Spatial AI)에서는 전체 장면 또는 시스템 그래프를 이용하여 환경 인식(Environment Recognition), 상황 평가(Situation Assessment), 임무 수준 추론(Mission-Level Reasoning), 서로 다른 운용 구성 간 비교를 수행할 수 있다.

전체 그래프 표현(Graph Representation) 프레임워크에서 그래프 수준 특징은 정보의 계층 구조(Hierarchy of Information)를 완성한다. 노드 특징(Node Features)은 개별 개체를 설명하고, 엣지 특징(Edge Features)은 개체 사이의 관계를 설명하며, 그래프 수준 특징(Graph-Level Features)은 전체 구조가 공유하는 전역 환경(Global Environment), 상태(State), 속성(Properties)을 설명한다. 이종 유형(Heterogeneous Types), 동적 정보(Dynamic Information), 그래프 위상(Graph Topology)과 함께 이러한 구성 요소는 이후 그래프 합성곱 신경망(Graph Convolutional Network, GCN), 그래프 어텐션 네트워크(Graph Attention Network, GAT), 메시지 패싱(Message Passing), 풀링(Pooling), 임베딩(Embedding), 그래프 추론(Graph Reasoning) 방법을 위한 풍부한 그래프 입력(Rich Graph Input)을 구성한다.

## 02.04. Heterogeneous Graphs

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

이종 그래프(Heterogeneous Graphs)는 하나의 그래프 구조(Graph Structure) 안에 여러 유형의 노드(Node), 엣지(Edge), 또는 이 둘을 모두 포함하는 시스템을 표현한다. 노드와 관계를 일반적으로 동일한 의미적 범주(Semantic Category)에 속하는 것으로 처리하는 동종 그래프(Homogeneous Graph)와 달리, 이종 그래프는 서로 다른 종류의 개체(Entity)와 상호작용(Interaction) 사이의 차이를 보존한다. 따라서 서로 다른 역할, 속성, 관계 의미를 가진 구성 요소로 이루어진 복잡한 현실 세계 시스템을 모델링하는 데 적합하다.

이종 그래프는 (G=(V,E))와 함께 노드 유형(Node Type) 및 엣지 유형(Edge Type)을 정의하는 매핑(Mapping)을 사용하여 설명할 수 있다. 노드 유형 함수(Node-Type Function)는 각 노드를 의미적 범주에 할당하고, 엣지 유형 함수(Edge-Type Function)는 각 관계의 의미를 식별한다. 따라서 두 노드는 서로 다른 특징 공간(Feature Space)을 가질 수 있으며, 유사한 개체를 연결하는 두 엣지 역시 근본적으로 다른 상호작용을 나타낼 수 있다. 그러므로 유형 정보(Type Information)는 그래프 표현(Graph Representation)의 명시적인 구성 요소가 된다.

사람(Person), 조직(Organization), 제품(Product), 장소(Place), 문서(Document)를 포함하는 지식 그래프(Knowledge Graph)를 생각할 수 있다. 사람은 조직에서 근무하거나, 제품을 구매하거나, 장소를 방문하거나, 문서를 작성할 수 있다. 각각의 관계는 서로 다른 의미를 표현하므로 상호 교환 가능한 연결로 해석할 수 없다. 이종 표현(Heterogeneous Representation)은 이러한 차이를 유지하여 그래프 학습 모델(Graph Learning Model)이 개체의 정체성(Entity Identity)과 관계 의미론(Relationship Semantics)을 함께 추론할 수 있도록 한다.

노드 유형(Node Types)은 서로 다른 특징 스키마(Feature Schema)를 가지는 경우가 많다. 사람 노드는 인구통계학적 또는 행동적 속성을 포함할 수 있고, 조직 노드는 산업 분야와 규모 정보를 포함할 수 있으며, 제품 노드는 범주, 가격, 기술적 특성을 포함할 수 있다. 이러한 특징은 의미와 차원이 서로 다르기 때문에 이종 그래프 모델(Heterogeneous Graph Model)은 관계 학습(Relational Learning)을 시작하기 전에 서로 다른 원시 속성을 호환 가능한 잠재 표현(Latent Representation)으로 변환하는 유형별 인코더(Type-Specific Encoder)를 사용하는 경우가 많다.

엣지 이질성(Edge Heterogeneity)은 또 다른 수준의 복잡성을 추가한다. 근무 관계(Works-For), 소유 관계(Owns), 위치 관계(Located-In), 통신 관계(Communicates-With), 구매 관계(Purchases), 의존 관계(Depends-On)와 같은 관계는 서로 다른 노드 유형의 조합을 연결할 수 있다. 각 관계는 강도(Strength), 빈도(Frequency), 타임스탬프(Timestamp), 신뢰도(Confidence), 거리(Distance), 방향(Direction)과 같은 고유한 엣지 특징(Edge Features)을 포함할 수도 있다. 이러한 관계 유형을 보존하면 의미적으로 서로 다른 상호작용이 하나의 일반적인 인접 구조(Adjacency Structure)로 축소되는 것을 방지할 수 있다.

이종 그래프를 이해하기 위한 유용한 추상화는 그래프 스키마(Graph Schema) 또는 네트워크 스키마(Network Schema)이다. 스키마는 어떤 노드 유형이 존재하고 어떤 관계 유형이 그 사이에서 허용되는지를 설명한다. 개별 개체를 표현하는 대신 개체의 범주와 이들 사이에 허용되는 연결을 표현한다. 이러한 상위 수준 설명(Higher-Level Description)은 그래프의 의미적 조직(Semantic Organization)을 정의하고, 서로 다른 종류의 개체들이 어떻게 상호작용할 수 있는지에 대한 정보를 그래프 학습 모델에 제공한다.

유형이 지정된 관계(Typed Relationship)는 흔히 ((\\text{source type},\\text{relation type},\\text{destination type}))과 같은 트리플릿(Triplet)으로 표현된다. 예를 들어 ((Person, works_for, Organization))과 ((Organization, produces, Product))는 서로 다른 의미적 채널(Semantic Channel)을 나타낸다. 그래프 학습 시스템은 이러한 관계 유형마다 별도의 매개변수(Parameter) 또는 변환 함수(Transformation Function)를 연결하여 모든 엣지를 동일하게 처리하는 대신 각 연결의 의미에 따라 정보를 처리할 수 있다.

따라서 이종 그래프의 메시지 패싱(Message Passing)은 유형 정보(Type Information)를 고려해야 한다. 센서(Sensor)에서 로봇(Robot)으로 전달되는 메시지는 다른 로봇, 객체(Object), 위치(Location)에서 전달되는 메시지와 서로 다른 변환을 요구할 수 있다. 모델은 먼저 출발 노드 유형(Source Node Type)과 관계 유형(Relation Type)에 따라 정보를 변환하고, 서로 다른 관계를 통해 도착한 메시지를 집계(Aggregation)한 다음, 목적 노드(Destination Node)의 의미적 역할에 따라 해당 노드 표현을 갱신할 수 있다.

관계별 처리(Relation-Specific Processing)는 개체들이 어떻게 연결되는지에 따라 동일한 노드 특징도 서로 다른 의미를 가질 수 있기 때문에 표현력(Expressive Power)을 향상시킨다. 위치 관계(Located-In)를 통해 전달된 정보는 소유 관계(Owns)나 통신 관계(Communicates-With)를 통해 전달된 정보와 동일한 방식으로 해석해서는 안 될 수 있다. 이종 그래프 학습(Heterogeneous Graph Learning)은 이러한 의미적 차이를 명시적으로 모델링하고 특정 예측 또는 추론 작업에서 어떤 관계 유형이 가장 유용한지를 학습할 수 있다.

이종 그래프는 동일한 노드 쌍 사이에 여러 관계(Multiple Relations)를 포함할 수도 있다. 두 사용자는 동시에 서로 통신하고, 협업하며, 동일한 조직에 소속될 수 있다. 로봇과 위치 사이에도 현재 점유(Current Occupancy), 이동 기록(Navigation History), 접근 가능성(Accessibility), 작업 할당(Task Assignment)을 설명하는 여러 관계가 존재할 수 있다. 이러한 다중 관계 구조(Multi-Relational Structure)는 모든 연결을 하나의 엣지 유형으로 축소할 경우 손실되는 여러 차원의 상호작용을 보존한다.

시간 정보(Temporal Information)를 추가하면 이종 그래프 표현을 더욱 확장할 수 있다. 서로 다른 노드 및 엣지 유형은 시간에 따라 나타나거나 사라질 수 있으며 속성이 변화할 수도 있다. 로봇은 위치 사이를 이동할 수 있고, 객체의 소유권은 변경될 수 있으며, 통신 링크(Communication Link)의 품질이나 가용성도 달라질 수 있다. 따라서 이종 그래프와 동적 그래프(Dynamic Graph)의 개념을 결합하면 의미적 구조와 관계 상태(Relational State)가 모두 지속적으로 변화하는 시스템을 표현할 수 있다.

로보틱스(Robotics)와 피지컬 AI(Physical AI) 시스템은 이종 그래프의 자연스러운 응용 사례를 제공한다. 노드는 로봇(Robot), 사람(Human), 객체(Object), 방(Room), 랜드마크(Landmark), 센서(Sensor), 작업(Task), 인프라 구성 요소(Infrastructure Component)를 나타낼 수 있다. 관계는 관측(Observes), 위치(Located-In), 연결(Connected-To), 할당(Assigned-To), 통신(Communicates-With), 근접(Near), 운반(Carries), 상호작용(Interacts-With)을 설명할 수 있다. 이러한 표현은 공간적, 의미적, 운용적, 관계적 정보를 장면 이해(Scene Understanding)와 다중 에이전트 추론(Multi-Agent Reasoning)에 적합한 하나의 공통 구조로 통합한다.

지식 그래프(Knowledge Graph)는 서로 다른 개체 유형과 관계 유형을 조직하는 것이 기본 목적이기 때문에 또 하나의 중요한 응용 분야이다. 추천 시스템(Recommendation System) 역시 사용자, 제품, 범주, 상호작용을 하나의 네트워크로 표현할 수 있다. 이외에도 생의학 네트워크(Biomedical Network), 학술 그래프(Academic Graph), 금융 시스템(Financial System), 사이버보안 네트워크(Cybersecurity Network), 교통 시스템(Transportation System), 산업 환경(Industrial Environment) 등 여러 종류의 개체가 서로 다른 관계에 참여하는 분야에서 활용할 수 있다.

이종 그래프 특징(Heterogeneous Graph Features)을 설계하려면 세심한 전처리(Preprocessing)가 필요하다. 서로 다른 노드 유형은 각각 별도의 정규화(Normalization), 범주형 인코딩(Categorical Encoding), 텍스트 또는 이미지 인코더(Text or Image Encoder), 결측값 처리 전략(Missing-Value Strategy)을 필요로 할 수 있다. 관계 유형에도 개별적인 특징 변환(Feature Transformation)이 필요할 수 있다. 목표는 모든 원시 데이터를 동일한 의미로 강제 변환하는 것이 아니라, 중요한 차이를 유지하면서 유형별 정보를 그래프 학습 과정에서 의미 있게 상호작용할 수 있는 표현으로 매핑하는 것이다.

이종 그래프 학습의 출력(Output)은 노드, 엣지, 관계 또는 전체 그래프를 대상으로 할 수 있다. 모델은 개체 분류(Entity Classification), 누락된 유형별 링크 예측(Typed Link Prediction), 상호작용 추천(Interaction Recommendation), 비정상 관계 탐지(Anomalous Relationship Detection), 다단계 의미 관계 추론(Multi-Step Semantic Relation Inference)을 수행할 수 있다. 그래프가 개체 및 관계 유형을 명시적으로 보존하기 때문에 특정 관계에 어떤 범주의 개체가 참여할 수 있는지와 같은 의미 있는 제약 조건(Constraint)을 예측 과정에 반영할 수 있다.

그래프 표현(Graph Representation)에서 이종 그래프는 명시적인 의미 유형화(Semantic Typing)를 추가함으로써 노드 특징(Node Features), 엣지 특징(Edge Features), 그래프 수준 특징(Graph-Level Features)을 확장한다. 이를 통해 그래프는 단순한 정점(Vertex)과 연결(Connection)의 일반적인 네트워크에서 여러 종류의 개체와 상호작용을 포함하는 구조화된 관계 모델(Structured Relational Model)로 발전한다. 이러한 기반은 현대 그래프 딥러닝(Graph Deep Learning)에서 사용되는 이종 메시지 패싱(Heterogeneous Message Passing), 어텐션(Attention), 관계별 변환(Relation-Specific Transformation), 임베딩(Embedding), 고급 그래프 추론(Advanced Graph Reasoning)을 위한 데이터를 준비한다.

## 02.05. Dynamic Graphs

![](images/image6.png){width="7.268055555555556in" height="7.268055555555556in"}

동적 그래프(Dynamic Graphs)는 구조(Structure), 속성(Attributes), 또는 이 두 가지가 시간에 따라 변화하는 시스템을 표현한다. 정적 그래프(Static Graph)에서는 분석 과정에서 노드 집합(Node Set), 엣지 집합(Edge Set), 그리고 관련 특징(Features)이 고정되어 있다고 가정한다. 반면 동적 그래프는 개체(Entity)가 나타나거나 사라지고, 관계(Relationship)가 생성되거나 제거되며, 기반 시스템이 변화함에 따라 노드 또는 엣지 속성이 지속적으로 변할 수 있는 진화하는 상태(Evolving State)를 모델링한다.

동적 그래프는 개념적으로 (G(t)=(V(t),E(t)))와 같이 표현할 수 있으며, 여기서 그래프는 시간(Time) (t)에 따라 달라진다. 노드 특징(Node Features)은 (x_i(t)), 엣지 특징(Edge Features)은 (a_{ij}(t))와 같이 표현할 수 있다. 이러한 형식은 시간을 그래프 표현(Graph Representation)의 명시적인 구성 요소로 만들며, 하나의 그래프 시스템이 변화하는 연결성(Connectivity), 진화하는 개체 상태(Entity States), 시간 의존적 관계(Time-Dependent Relationships)를 표현할 수 있도록 한다.

동적 그래프는 일반적으로 이산 스냅샷(Discrete Snapshots) 또는 연속적인 이벤트 스트림(Continuous Event Streams)을 이용하여 표현한다. 스냅샷 표현(Snapshot Representation)에서는 변화하는 시스템을 (G_{t_1},G_{t_2},\...,G_{t_n})과 같은 그래프로 분할하며, 각각은 특정 시간 구간의 상태를 설명한다. 이 방식은 동적 시스템을 정적 그래프의 시퀀스(Sequence)로 표현하며, 관측값이 일정한 간격으로 생성되거나 시간 해상도(Temporal Resolution)를 이산화할 수 있을 때 편리하다.

이벤트 기반 표현(Event-Based Representation)은 모든 시간 단계에서 전체 그래프를 다시 구성하는 대신 개별적인 변화만을 기록한다. 이벤트(Event)는 노드가 등장하거나, 엣지가 생성되거나, 상호작용이 발생하거나, 특정 속성이 변경되었음을 나타낼 수 있다. 각각의 이벤트에는 타임스탬프(Timestamp)와 관련 특징이 연결된다. 이 방식은 세밀한 시간 정보(Fine-Grained Temporal Information)를 보존할 수 있으며 상호작용이 비동기적(Asynchronous)이거나 불규칙한 시간 간격으로 발생하는 시스템에 유용하다.

노드 동역학(Node Dynamics)은 개체와 해당 개체의 상태가 어떻게 변화하는지를 설명한다. 사용자는 소셜 플랫폼(Social Platform)에 가입하거나 탈퇴할 수 있고, 차량은 교통 네트워크(Transportation Network)에 진입하거나 이탈할 수 있으며, 로봇은 위치 사이를 이동하면서 배터리(Battery), 속도(Velocity), 작업(Task), 센서 상태(Sensor State)가 변화할 수 있다. 동적 노드 특징(Dynamic Node Features)은 개체의 정체성(Identity)을 유지하면서 표현이 변화하도록 하여 지속적으로 존재하는 개체와 변화하는 상태를 구분할 수 있도록 한다.

엣지 동역학(Edge Dynamics)은 개체 사이의 관계가 변화하는 방식을 설명한다. 통신 링크(Communication Link)가 생성되거나 사라질 수 있고, 계정 사이에서 거래(Transaction)가 발생할 수 있으며, 도로가 혼잡해지거나 로봇 사이에 일시적인 협력 관계(Coordination Relationship)가 형성될 수 있다. 또한 기본적인 연결 자체가 유지되는 경우에도 거리(Distance), 상호작용 강도(Interaction Strength), 통신 품질(Communication Quality), 신뢰도(Confidence), 상대 자세(Relative Pose)와 같은 엣지 속성은 시간에 따라 변화할 수 있다.

시간 자체도 여러 가지 방식으로 인코딩(Encoding)할 수 있다. 그래프 표현에는 절대 타임스탬프(Absolute Timestamp), 상대 시간 차이(Relative Time Difference), 이벤트 순서(Event Ordering), 지속 시간(Duration), 주기 정보(Periodic Information), 학습된 시간 임베딩(Learned Temporal Embedding)이 포함될 수 있다. 특히 상대 시간(Relative Time)은 상호작용의 중요성이 얼마나 최근에 발생했는지에 따라 달라지는 경우가 많기 때문에 유용하다. 최근 이벤트는 훨씬 이전에 관측된 유사한 상호작용보다 현재 상태를 판단하는 데 더 강한 증거를 제공할 수 있다.

동적 그래프는 구조적 의존성(Structural Dependency)과 시간적 의존성(Temporal Dependency)이라는 두 가지 상호 보완적인 정보 차원을 모델이 포착하도록 요구한다. 구조적 의존성은 특정 시점에서 연결된 개체들이 서로 어떻게 영향을 주는지를 설명하며, 시간적 의존성은 이전 그래프 상태가 이후 상태에 어떻게 영향을 미치는지를 설명한다. 따라서 효과적인 동적 그래프 학습(Dynamic Graph Learning)은 노드와 엣지를 통한 관계 추론(Relational Reasoning)과 시간에 따라 정보를 유지하고 갱신하는 메커니즘을 결합한다.

시간적 메시지 패싱(Temporal Message Passing)은 상호작용이 언제 발생했는지를 고려함으로써 기존 그래프 메시지 패싱(Graph Message Passing)을 확장한다. 노드 (v_i)에서 (v_j)로 전달되는 메시지는 현재 노드 표현(Current Node Representations), 엣지 속성, 시간 정보를 함께 사용할 수 있다. 개념적으로 시간적 메시지는 (m_{ij}(t)=\\phi(h_i(t),h_j(t),a_{ij}(t),\\Delta t))와 같이 표현할 수 있으며, 여기서 (\\Delta t)는 해당 상호작용과 관련된 시간 정보를 설명한다.

메모리 메커니즘(Memory Mechanisms)은 현재 그래프만으로는 확인할 수 없는 과거 정보(Historical Information)를 유지하는 데 도움을 줄 수 있다. 노드의 현재 표현은 현재 이웃(Current Neighbors)뿐만 아니라 이전 상호작용과 과거 상태에 의존할 수 있다. 순환 신경망(Recurrent Networks), 시간적 어텐션(Temporal Attention), 상태 공간 메커니즘(State-Space Mechanisms), 학습된 메모리 모듈(Learned Memory Modules)은 이벤트가 발생할 때마다 표현을 갱신하여 그래프 모델이 과거의 관계 활동(Relational Activity)을 압축된 형태로 유지할 수 있도록 한다.

동적 그래프 학습은 지속적인 구조(Persistent Structure)와 일시적인 구조(Temporary Structure)의 차이도 고려해야 한다. 일부 관계는 장기간 안정적으로 유지되는 반면, 다른 관계는 매우 짧은 시간 동안만 존재한다. 도로 연결(Road Connection)은 구조적으로 영구적일 수 있지만 교통 상태(Traffic State)는 빠르게 변화할 수 있으며, 이동 로봇 사이의 통신 링크는 서로 통신 범위 안에 있을 때만 나타날 수 있다. 구조적 지속성(Structural Persistence)과 동적 상태(Dynamic State)를 구분하면 더욱 의미 있는 표현을 구성할 수 있다.

이질성(Heterogeneity)과 동적 특성(Dynamics)은 실제 시스템에서 함께 나타나는 경우가 많다. 로봇 환경(Robotic Environment)은 로봇, 사람, 객체(Object), 센서(Sensor), 방(Room), 작업(Task)을 포함할 수 있으며 이들의 속성과 관계는 시간에 따라 변화한다. 이종 동적 그래프(Heterogeneous Dynamic Graph)는 의미적 유형 정보(Semantic Type Information)와 시간적 변화(Temporal Evolution)를 모두 보존하여 어떤 종류의 개체가 상호작용하는지뿐만 아니라 언제, 어떤 상태에서, 어떤 관계 유형을 통해 상호작용하는지도 구분할 수 있도록 한다.

동적 그래프는 물리적 환경이 본질적으로 시간 의존적(Time-Dependent)이기 때문에 로보틱스(Robotics)와 피지컬 AI(Physical AI)에서 특히 중요하다. 로봇은 이동하고, 객체는 위치를 변경하며, 사람은 장면(Scene)에 들어오거나 나가고, 작업은 재할당되며, 통신 조건은 변화하고, 센서 관측(Sensor Observations)은 지속적으로 갱신된다. 동적 그래프는 이러한 변화하는 세계(Evolving World)를 구조적으로 표현하고 예측(Prediction), 계획(Planning), 협력(Coordination), 상황 이해(Situation Understanding)를 지원할 수 있다.

다른 주요 응용 분야에는 소셜 네트워크(Social Networks), 추천 시스템(Recommendation Systems), 금융 거래 네트워크(Financial Transaction Networks), 사이버보안(Cybersecurity), 교통(Transportation), 통신 시스템(Communication Systems), 과학적 상호작용 네트워크(Scientific Interaction Networks)가 있다. 이러한 분야에서는 이벤트와 관계의 시간적 순서(Temporal Ordering)가 중요한 정보를 포함한다. 동적 그래프 모델은 미래 링크 예측(Future Link Prediction), 변화하는 노드 분류(Evolving Node Classification), 이상 탐지(Anomaly Detection), 상호작용 예측(Interaction Forecasting), 시간적 그래프 수준 예측(Temporal Graph-Level Prediction)과 같은 작업을 지원할 수 있다.

동적 그래프 데이터를 준비하려면 신중한 동기화(Synchronization)와 시간적 분할(Temporal Splitting)이 필요하다. 이벤트는 일관된 순서로 정렬되어야 하며, 필요한 경우 타임스탬프를 정규화하고 관측값을 작업에 적합한 시간 해상도에 맞추어야 한다. 미래 행동을 예측하는 경우 학습(Training), 검증(Validation), 테스트(Test) 데이터셋은 일반적으로 시간 순서를 유지해야 한다. 미래 이벤트를 학습 데이터에 무작위로 혼합하면 시간 정보 누출(Temporal Information Leakage)이 발생하여 비현실적인 평가 결과를 만들 수 있기 때문이다.

그래프 표현(Graph Representation)에서 동적 그래프는 노드 특징(Node Features), 엣지 특징(Edge Features), 그래프 수준 특징(Graph-Level Features), 이종 구조(Heterogeneous Structures)에 시간(Time)이라는 차원을 추가한다. 이를 통해 그래프는 관계에 대한 고정된 설명에서 개체, 상호작용, 상태가 지속적으로 변화하는 진화형 표현(Evolving Representation)으로 확장된다. 이러한 기반은 시간적 메시지 패싱(Temporal Message Passing), 동적 임베딩(Dynamic Embeddings), 메모리 기반 그래프 모델(Memory-Based Graph Models), 예측(Forecasting), 그리고 무엇이 연결되어 있는지를 넘어 관계적 세계(Relational World)가 시간에 따라 어떻게 변화하는지를 이해해야 하는 이후의 그래프 추론(Graph Reasoning) 방법을 가능하게 한다.

## 02.06. Graph Data Pipelines [w/Code]

![](images/image7.png){width="7.268055555555556in" height="7.268055555555556in"}

그래프 데이터 파이프라인(Graph Data Pipeline)은 원시 관계형 데이터(Raw Relational Data)를 그래프 학습 모델(Graph Learning Model)이 안정적으로 사용할 수 있는 그래프 구조(Graph Structure)로 변환하는 일련의 처리 과정이다. 현실 세계의 데이터는 학습 준비가 완료된 그래프 형태로 제공되는 경우가 드물다. 대신 데이터베이스(Database), 테이블(Table), 로그(Log), 센서 스트림(Sensor Stream), 문서(Document), 이벤트 기록(Event Record) 등에 분산되어 있을 수 있다. 파이프라인은 이러한 데이터를 노드(Node), 엣지(Edge), 특징(Feature), 레이블(Label), 그래프 수준 정보(Graph-Level Information)로 변환하면서 원래의 의미를 보존한다.

첫 번째 단계는 데이터 획득 및 수집(Data Acquisition and Ingestion)이다. 원시 정보는 관계형 데이터베이스(Relational Database), CSV 파일, API, 트랜잭션 로그(Transaction Log), 지식 베이스(Knowledge Base), 시뮬레이션 환경(Simulation Environment), 센서(Sensor), 분산 시스템(Distributed System) 등에서 가져올 수 있다. 이 단계의 목적은 그래프 정의(Graph Definition)에 필요한 개체(Entity)와 상호작용(Interaction)을 수집하는 것이다. 타임스탬프(Timestamp), 식별자(Identifier), 데이터 출처(Source Information), 측정 신뢰도(Measurement Confidence)와 같은 메타데이터(Metadata)도 이후 그래프 구성이나 검증에 사용될 수 있다면 함께 유지해야 한다.

개체 식별(Entity Identification)은 어떤 레코드(Record)를 노드로 변환할 것인지 결정하고 여러 데이터 소스에서 해당 개체의 정체성을 어떻게 유지할 것인지를 정의한다. 하나의 물리적 또는 논리적 개체가 여러 식별자로 나타날 수 있으며, 서로 다른 개체가 우연히 유사한 이름이나 속성을 가질 수도 있다. 따라서 개체 해석(Entity Resolution), 중복 제거(Deduplication), 식별자 매핑(Identifier Mapping), 일관성 검사(Consistency Checking)가 중요한 처리 과정이 된다. 안정적인 노드 식별자는 전체 파이프라인에서 관계와 특징이 올바른 개체에 연결되도록 한다.

그래프 구성(Graph Construction)은 식별된 개체와 상호작용을 그래프 위상(Graph Topology)으로 변환한다. 개체로부터 노드를 생성하고 관측되거나 정의된 관계로부터 엣지를 생성한다. 엣지는 거래(Transaction)나 인용(Citation)과 같은 명시적 기록에서 생성될 수도 있고, 공간적 근접성(Spatial Proximity), 유사도 임계값(Similarity Threshold), 통신 범위(Communication Range), 시간적 상호작용(Temporal Interaction)과 같은 규칙으로부터 유도될 수도 있다. 그래프 구성 정책은 학습 모델이 어떤 관계적 의존성(Relational Dependency)을 관측할 수 있는지를 결정하기 때문에 중요하다.

특징 공학(Feature Engineering)은 원시 속성(Raw Attribute)을 기계가 처리할 수 있는 노드 특징(Node Features), 엣지 특징(Edge Features), 그래프 수준 특징(Graph-Level Features)으로 변환한다. 수치 측정값은 정규화(Normalization)할 수 있고, 범주형 변수(Categorical Variable)는 인코딩(Encoding)하며, 텍스트는 임베딩(Embedding)으로 변환할 수 있다. 이미지나 센서 관측은 특화된 인코더(Specialized Encoder)를 통해 처리할 수 있다. 상대 거리(Relative Distance), 상호작용 빈도(Interaction Frequency), 중심성(Centrality), 시간 간격(Temporal Interval)과 같은 파생 특징(Derived Features)도 계산할 수 있다. 목적은 유용한 도메인 정보를 보존하면서 그래프 계산에 적합한 표현을 만드는 것이다.

이종 그래프(Heterogeneous Graph)를 처리하는 경우 그래프 파이프라인은 서로 다른 특징 스키마(Feature Schema)를 수용해야 한다. 각각의 노드 유형(Node Type)은 자체적인 전처리(Preprocessing)와 인코더(Encoder)를 필요로 할 수 있으며, 서로 다른 관계 유형(Relation Type)은 고유한 엣지 속성을 포함할 수 있다. 로봇(Robot), 센서(Sensor), 작업(Task), 위치(Location)가 반드시 동일한 원시 특징 표현을 공유할 필요는 없다. 유형별 처리(Type-Specific Processing)는 의미적 차이를 유지하면서 이러한 서로 다른 입력을 호환 가능한 잠재 공간(Latent Space)으로 매핑하여 이후의 이종 메시지 패싱(Heterogeneous Message Passing)에 사용할 수 있도록 한다.

동적 그래프 파이프라인(Dynamic Graph Pipeline)은 추가적인 시간적 요구사항(Temporal Requirements)을 가진다. 이벤트(Event)는 시간순으로 정렬되어야 하고, 타임스탬프는 일관된 방식으로 표현되어야 하며, 노드, 엣지, 속성의 변화를 올바르게 재구성해야 한다. 학습 방법에 따라 파이프라인은 이산 그래프 스냅샷(Discrete Graph Snapshot)을 생성하거나 이벤트를 연속적인 시간 스트림(Continuous Temporal Stream)으로 유지할 수 있다. 미래 정보가 이전 상태를 표현하기 위한 데이터에 잘못 영향을 주어서는 안 되므로 시간적 순서(Temporal Ordering)가 특히 중요하다.

데이터 정제 및 검증(Data Cleaning and Validation)은 구성된 그래프가 내부적으로 일관성을 유지하도록 한다. 파이프라인은 중복 엣지(Duplicated Edge), 잘못된 노드 참조(Invalid Node Reference), 누락된 특징(Missing Feature), 불가능한 타임스탬프(Impossible Timestamp), 일관되지 않은 관계 유형(Inconsistent Relationship Type), 비정상적인 수치 값(Abnormal Numerical Value)을 탐지할 수 있다. 자기 루프(Self-Loop)와 고립 노드(Isolated Node)는 자동으로 제거하기보다는 사용하려는 그래프 모델의 목적에 따라 처리해야 한다. 검증 과정에서는 그래프 변환이 다운스트림 학습 목표(Downstream Learning Objective)에 필요한 의미를 보존하는지 확인해야 한다.

그래프를 구성한 이후에는 일반적으로 엣지 리스트(Edge List), 인접 구조(Adjacency Structure), 희소 행렬(Sparse Matrix), 인덱스 텐서(Indexed Tensor)와 같은 계산 표현(Computational Representation)으로 변환한다. 노드 특징은 행렬 (X)에 저장하고, 연결성은 엣지 인덱스(Edge Index) 또는 인접 표현(Adjacency Representation)에 저장하며, 엣지 특징은 대응하는 속성 텐서(Attribute Tensor)에 저장할 수 있다. 실제 그래프에서는 가능한 모든 노드 쌍에 비해 실제 엣지 수가 훨씬 적은 경우가 많으므로 희소 표현(Sparse Representation)이 특히 중요하다.

데이터셋 분할(Dataset Splitting)은 관계적 의존성 때문에 기존의 무작위 분할(Random Splitting)이 문제가 될 수 있어 그래프 파이프라인에서 매우 중요한 과정이다. 노드 분류(Node Classification)는 노드 수준 마스크(Node-Level Mask)를 사용할 수 있고, 그래프 분류(Graph Classification)는 전체 그래프를 서로 다른 데이터 부분집합으로 나눌 수 있으며, 링크 예측(Link Prediction)은 관측 엣지와 예측 대상 엣지를 신중하게 분리해야 한다. 동적 그래프는 일반적으로 학습 데이터가 검증 및 테스트 기간보다 앞서도록 시간순 분할(Chronological Splitting)을 적용하여 미래 정보에 비현실적으로 접근하는 것을 방지해야 한다.

정보 누출(Information Leakage)은 그래프 데이터셋에서 특히 주의해야 한다. 학습 집합(Training Set)의 노드가 검증 또는 테스트 노드와 연결되어 있을 수 있으며, 전처리 연산이 전체 데이터셋에서 계산된 통계를 의도하지 않게 사용할 수도 있다. 링크 예측에서는 목표 엣지(Target Edge)가 메시지 패싱(Message Passing)에 노출될 수 있으며, 시간적 작업에서는 미래 이벤트가 유출될 수 있다. 따라서 올바른 파이프라인은 일반적인 데이터셋 분할만으로 충분하다고 가정하지 않고 각 학습 및 평가 단계에서 어떤 정보를 사용할 수 있는지를 명시적으로 정의해야 한다.

대규모 그래프(Large Graph)는 모든 노드와 엣지를 동시에 처리할 경우 메모리 또는 계산 한계를 초과할 수 있으므로 샘플링(Sampling)이 필요한 경우가 많다. 이웃 샘플링(Neighborhood Sampling)은 목표 노드 주변의 일부 이웃을 선택하고, 서브그래프 샘플링(Subgraph Sampling)은 학습을 위해 더 작은 관계 영역을 추출한다. 그래프 수준 데이터셋에서는 여러 독립적인 그래프를 배치(Batch)로 구성할 수 있다. 샘플링 전략은 계산 비용을 제어하면서 학습에 필요한 충분한 지역적·구조적 정보를 보존하고 체계적인 샘플링 편향(Sampling Bias)을 방지해야 한다.

그래프 데이터의 배칭(Batching)은 그래프마다 노드와 엣지 수가 서로 다를 수 있기 때문에 고정 크기 벡터나 이미지의 배칭과 차이가 있다. 여러 그래프를 서로 연결되지 않은 하나의 큰 그래프로 결합하면서 각 노드가 어느 원래 그래프에 속하는지를 나타내는 인덱스(Index)를 유지할 수 있다. 노드 중심 작업(Node-Centered Task)에서는 샘플링된 이웃을 미니배치(Mini-Batch)로 구성할 수 있다. 효율적인 배칭을 사용하면 모든 그래프가 동일한 크기를 가질 필요 없이 가속기(Accelerator)에서 병렬 처리를 수행할 수 있다.

그래프 파이프라인에는 재현성(Reproducibility)과 추적 가능성(Traceability)도 필요하다. 원시 레코드에서 학습 그래프로 변환되는 과정은 명시적인 구성(Configuration), 가능한 경우 결정론적 매핑(Deterministic Mapping), 데이터셋 버전(Dataset Version), 문서화된 전처리 규칙(Preprocessing Rules)을 통해 관리해야 한다. 특징 정의(Feature Definition), 그래프 구성 임계값(Graph Construction Threshold), 데이터 분할 정책(Split Policy), 정규화 통계(Normalization Statistics)를 데이터셋과 함께 보존해야 한다. 이를 통해 실험을 반복할 수 있으며 전처리 과정의 조용한 변경이 모델 아키텍처의 성능 향상으로 잘못 해석되는 것을 방지할 수 있다.

로보틱스(Robotics)와 피지컬 AI(Physical AI)에서는 그래프 데이터 파이프라인이 인지(Perception)와 관계 학습(Relational Learning)을 연결할 수 있다. 센서 관측(Sensor Observation)을 통해 탐지된 객체(Detected Object), 로봇, 사람(Human), 랜드마크(Landmark), 주행 가능 영역(Traversable Region), 의미적 위치(Semantic Location)를 먼저 생성할 수 있다. 이러한 개체는 노드가 되고 공간적(Spatial), 통신(Communication), 작업(Task), 상호작용(Interaction) 관계는 엣지가 된다. 새로운 관측이 들어오면 그래프를 동적으로 갱신하여 물리적 세계의 원시 관측과 상위 수준 그래프 추론(Graph Reasoning) 사이에 구조화된 인터페이스를 만들 수 있다.

그래프 데이터 파이프라인의 최종 출력은 단순한 그래프 파일(Graph File)이 아니라 작업에 필요한 위상(Topology), 노드 특징(Node Features), 엣지 특징(Edge Features), 그래프 수준 문맥(Graph-Level Context), 유형 정보(Type Information), 시간 정보(Temporal Information), 레이블(Label), 데이터셋 분할(Dataset Partition)을 포함하는 학습 준비 표현(Learning-Ready Representation)이다. 이러한 구성 요소는 이후 그래프 합성곱 신경망(Graph Convolutional Network, GCN), 그래프 어텐션 네트워크(Graph Attention Network, GAT), 메시지 패싱(Message Passing), 풀링(Pooling), 임베딩(Embedding), 추론(Reasoning) 모델의 입력이 된다. 따라서 잘 설계된 그래프 데이터 파이프라인은 원시 관계형 데이터와 그래프 딥러닝(Graph Deep Learning)을 연결하는 실질적인 운영적 가교(Operational Bridge)를 형성한다.

## 02.07. Graph Datasets

![](images/image8.png){width="7.268055555555556in" height="7.268055555555556in"}

그래프 데이터셋(Graph Datasets)은 그래프 학습 알고리즘(Graph Learning Algorithms)을 개발하고 평가하기 위한 개체(Entity), 관계(Relationship), 속성(Attribute), 레이블(Label)의 구조화된 집합을 제공한다. 독립적인 샘플(Independent Samples)로 구성되는 일반적인 데이터셋과 달리, 그래프 데이터셋은 엣지(Edge)를 통해 관측값 사이의 의존 관계를 보존한다. 이러한 구성은 모델이 어떤 관계 패턴(Relational Patterns)을 학습할 수 있는지를 결정하며 노드 분류(Node Classification), 링크 예측(Link Prediction), 그래프 분류(Graph Classification), 임베딩(Embedding), 추천(Recommendation), 관계 추론(Relational Reasoning)과 같은 작업에 큰 영향을 준다.

그래프 데이터셋은 일반적으로 그래프 위상(Graph Topology)과 함께 노드(Node), 엣지(Edge), 그리고 경우에 따라 그래프 수준 정보(Graph-Level Information)를 포함한다. 노드는 개체를 나타내고, 엣지는 관계를 인코딩하며, 특징(Features)은 이들과 연결된 측정 가능하거나 학습된 속성을 제공한다. 학습 목표에 따라 레이블은 노드, 엣지 또는 전체 그래프에 연결될 수 있다. 따라서 데이터셋 준비는 단순히 샘플을 수집하는 것뿐만 아니라 개체와 관계를 계산 가능한 형태로 어떻게 표현할 것인지 정의하는 과정도 포함한다.

인용 네트워크(Citation Networks)는 그래프 머신러닝(Graph Machine Learning)에서 널리 사용되는 그래프 데이터셋의 한 유형이다. 일반적으로 논문(Publication)을 노드로 표현하고 논문 사이의 인용(Citation)을 엣지로 구성한다. 노드 특징(Node Features)은 키워드(Keywords), 단어 가방 표현(Bag-of-Words Representation), 주제 정보(Topic Information), 학습된 텍스트 임베딩(Learned Text Embedding)을 통해 문서 내용을 인코딩할 수 있다. 레이블은 주로 연구 분야(Research Area)나 문서 범주(Document Category)를 나타내므로 인용 네트워크는 노드 분류와 표현 학습(Representation Learning)을 연구하는 데 특히 적합하다.

인용 관계(Citation Relationship)는 개별 문서의 내용만으로 얻을 수 있는 것 이상의 정보를 제공한다. 관련 주제를 다루는 논문은 서로 유사하거나 관련된 문헌을 인용하는 경우가 많아 의미 있는 학술 구조(Academic Structure)를 가진 이웃(Neighborhood)을 형성한다. 따라서 그래프 학습 모델은 노드 표현(Node Representation)을 구성할 때 문서 특징과 인용 연결성(Citation Connectivity)을 함께 사용할 수 있다. 이러한 특성으로 인해 인용 네트워크는 독립적인 특징 기반 분류(Feature-Based Classification)를 넘어 관계적 문맥(Relational Context)이 어떻게 예측 성능을 향상할 수 있는지를 보여주는 데 유용하다.

인용 네트워크는 링크 예측(Link Prediction)과 과학 네트워크 분석(Scientific Network Analysis)에도 활용할 수 있다. 모델은 한 논문이 다른 논문을 인용할 가능성을 추정하거나, 관련 연구 커뮤니티(Research Communities)를 식별하거나, 구조적·의미적 유사성을 반영하는 임베딩을 학습할 수 있다. 또한 인용은 일반적으로 새로운 연구가 이전에 공개된 연구를 참조하기 때문에 인용 방향(Citation Direction)과 출판 시간(Publication Time)은 중요한 정보를 제공하며, 그래프에 방향성(Directed)과 시간적 특성(Temporal Characteristics)을 동시에 부여할 수 있다.

소셜 네트워크(Social Networks)는 사람(Person), 계정(Account), 조직(Organization), 커뮤니티(Community) 또는 기타 사회적 개체를 노드로 표현하고 이들을 상호작용이나 관계를 통해 연결한다. 엣지는 친구 관계(Friendship), 팔로우(Following), 통신(Communication), 협업(Collaboration), 멤버십(Membership) 등의 관계를 나타낼 수 있다. 노드 특징은 프로필 속성(Profile Attributes), 행동 통계(Behavioral Statistics), 관심사(Interests), 활동 패턴(Activity Patterns), 학습된 표현(Learned Representations)을 포함할 수 있으며, 엣지 특징은 상호작용 빈도, 강도, 방향, 지속 시간 또는 유형을 설명할 수 있다.

소셜 그래프 데이터셋(Social Graph Datasets)은 연결성(Connectivity)이 개별 속성에서 직접 확인할 수 없는 정보를 어떻게 드러낼 수 있는지를 보여준다. 커뮤니티는 밀집된 상호작용 패턴(Dense Interaction Patterns)에서 나타날 수 있고, 영향력이 높은 노드(Influential Nodes)는 구조적으로 중요한 위치를 차지할 수 있으며, 유사한 사용자들은 특징적인 이웃 구조를 형성할 수 있다. 이러한 특성으로 인해 소셜 네트워크는 노드 분류, 커뮤니티 탐지(Community Detection), 추천, 영향력 분석(Influence Analysis), 링크 예측, 이상 탐지(Anomaly Detection), 그래프 표현 학습(Graph Representation Learning)에 활용할 수 있다.

소셜 네트워크는 구성 방식에 따라 동종 그래프(Homogeneous Graph) 또는 이종 그래프(Heterogeneous Graph)가 될 수 있다. 단순한 친구 관계 네트워크는 하나의 사용자 유형과 하나의 관계 유형만 포함할 수 있지만, 더 풍부한 플랫폼 그래프는 사용자, 게시물(Post), 그룹(Group), 주제(Topic), 조직을 여러 관계를 통해 연결할 수 있다. 또한 사회적 연결, 통신 패턴, 관심사, 활동 수준은 시간에 따라 변화하기 때문에 소셜 상호작용은 본질적으로 동적이며 시간 그래프 표현(Temporal Graph Representation)이 특히 중요할 수 있다.

소셜 네트워크 데이터셋을 사용할 때는 개인정보 보호(Privacy), 샘플링(Sampling), 편향(Bias)을 중요하게 고려해야 한다. 관측된 네트워크는 더 큰 사회 시스템의 일부만을 나타낼 수 있으며, 데이터 수집 정책(Data Collection Policies)이 연결 패턴에 큰 영향을 줄 수 있다. 누락된 상호작용, 비활성 계정(Inactive Accounts), 플랫폼별 행동(Platform-Specific Behavior), 인구통계학적 또는 활동 수준의 불균형은 학습된 표현에 영향을 미칠 수 있다. 따라서 그래프 구조를 실제 사회적 관계에 대한 완전하거나 편향되지 않은 설명으로 자동 해석해서는 안 된다.

지식 그래프(Knowledge Graphs)는 개체와 의미적 관계(Semantic Relationships)를 구조화된 관계 형태로 표현한다. 노드는 사람, 조직, 장소(Place), 제품(Product), 사건(Event), 개념(Concept), 문서 또는 기타 식별 가능한 개체를 나타낼 수 있다. 엣지는 위치 관계(Located-In), 근무 관계(Works-For), 생성 관계(Created-By), 부분 관계(Part-Of), 연관 관계(Related-To), 소유 관계(Owns)와 같은 유형화된 관계(Typed Relations)를 표현한다. 이러한 다중 관계 구조(Multi-Relational Structure)는 지식 그래프를 이종 그래프 표현(Heterogeneous Graph Representation)의 대표적인 사례로 만든다.

지식 그래프에서 일반적으로 사용되는 표현은 ((head, relation, tail)) 형태의 트리플릿(Triplet)이다. 예를 들어 사람(Person)은 근무 관계(Works-For)를 통해 조직(Organization)에 연결될 수 있고, 조직은 위치 관계(Located-In)를 통해 특정 장소(Location)에 연결될 수 있다. 여기에서 관계 자체는 단순한 일반적 연결(Generic Connection)이 아니라 명시적인 의미(Semantic Meaning)를 가진다. 따라서 지식 그래프 학습(Knowledge Graph Learning)은 이러한 구조화된 사실을 포착하기 위해 개체 표현(Entity Representation)과 관계 표현(Relation Representation)을 모두 모델링해야 한다.

지식 그래프 데이터셋은 기존 사실을 기반으로 누락된 관계를 추론하는 지식 그래프 완성(Knowledge Graph Completion)에 자주 사용된다. 트리플릿을 구성하는 요소 중 두 개가 주어지면 모델은 누락된 개체 또는 관계를 예측할 수 있다. 다른 주요 작업으로는 개체 분류(Entity Classification), 관계 예측(Relation Prediction), 의미 검색(Semantic Search), 질의응답(Question Answering), 추천, 다중 홉 추론(Multi-Hop Reasoning)이 있다. 학습된 임베딩은 유용한 관계 패턴을 유지하면서 개체와 관계를 연속 공간(Continuous Space)에 표현할 수 있다.

인용 네트워크(Citation Networks), 소셜 네트워크(Social Networks), 지식 그래프(Knowledge Graphs)는 각각 중심이 되는 그래프 의미론(Graph Semantics)이 다르다. 인용 네트워크는 문서 사이의 관계를 강조하고, 소셜 네트워크는 사회적 개체 사이의 상호작용을 강조하며, 지식 그래프는 이종 개체 사이의 유형화된 의미적 사실(Typed Semantic Facts)을 강조한다. 그러나 세 가지 모두 개체를 그 자체의 속성뿐만 아니라 주변의 관계적 문맥을 통해 함께 해석해야 한다는 그래프 학습의 핵심 원리를 공유한다.

이러한 데이터셋들은 그래프 표현의 복잡성이 증가하는 과정도 보여준다. 기본적인 인용 그래프는 문서 특징과 인용 엣지를 사용할 수 있지만, 소셜 그래프는 여러 상호작용, 시간적 행동(Temporal Behavior), 더욱 풍부한 속성을 포함할 수 있다. 지식 그래프는 여러 개체 및 관계 유형을 명시적으로 도입하며 관계별 추론(Relation-Specific Reasoning)을 요구하는 경우가 많다. 이러한 차이를 이해하면 학습 문제에 동종(Homogeneous), 이종(Heterogeneous), 정적(Static), 동적(Dynamic), 방향성(Directed), 가중(Weighted), 다중 관계(Multi-Relational) 그래프 표현 가운데 어떤 방식이 필요한지를 결정하는 데 도움이 된다.

데이터셋 분할(Dataset Splitting)과 평가(Evaluation)는 목표로 하는 그래프 작업(Graph Task)을 반영해야 한다. 노드 분류는 일반적으로 레이블이 있는 노드를 학습(Training), 검증(Validation), 테스트(Test) 부분집합으로 분리하며, 링크 예측은 모델이 복원하거나 예측해야 하는 관계를 분리한다. 지식 그래프 완성에서는 일반적으로 선택된 트리플릿을 평가용으로 제외한다. 시간적 데이터셋(Temporal Dataset)에서 미래 상호작용을 예측하는 경우에는 시간 순서를 유지하여 이후 관측 정보가 학습 데이터에 누출되는 것을 방지해야 한다.

따라서 그래프 데이터셋(Graph Datasets)은 단순한 예제들의 집합을 넘어 그래프 학습이 이루어지는 관계적 환경(Relational Environment)을 정의한다. 인용 네트워크는 문서 연결성(Document Connectivity)을 보여주고, 소셜 네트워크는 상호작용 구조(Interaction Structure)를 드러내며, 지식 그래프는 명시적인 의미 관계(Semantic Relations)를 인코딩한다. 이들 데이터셋 유형은 함께 그래프 표현(Graph Representation)을 이해하기 위한 대표적인 기반을 제공하며, 이후 그래프 합성곱 신경망(Graph Convolutional Network, GCN), 그래프 어텐션 네트워크(Graph Attention Network, GAT), 메시지 패싱(Message Passing), 그래프 임베딩(Graph Embeddings), 이종 그래프 학습(Heterogeneous Learning), 고급 관계 추론(Advanced Relational Reasoning)으로 발전하기 위한 토대를 제공한다.
