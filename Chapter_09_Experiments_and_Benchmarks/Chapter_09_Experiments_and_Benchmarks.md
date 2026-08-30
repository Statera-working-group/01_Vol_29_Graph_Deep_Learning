**Volume 29. Graph Deep Learning**


# Chapter 09. Experiments and Benchmarks

##  

## 09.01. Datasets

![](images/image1.png){width="7.268055555555556in" height="7.268055555555556in"}

Graph datasets provide the experimental foundation for evaluating graph learning models under controlled and comparable conditions. Unlike conventional tabular datasets, they encode entities as nodes and relationships as edges while often attaching attributes, labels, timestamps, or relation types. Dataset selection therefore determines not only the amount of training data but also the topology, sparsity, homophily, scale, and structural complexity that a graph model must learn.

A useful graph benchmark should expose both feature information and relational structure so that experiments can measure how effectively a model combines them. Important characteristics include the number of nodes and edges, feature dimensionality, class distribution, average degree, connected components, graph density, and label availability. These properties strongly influence memory consumption, convergence behavior, neighborhood aggregation, sampling strategies, and the interpretation of final performance.

Cora is one of the most widely used citation-network datasets for introductory graph neural network experiments. Scientific publications are represented as nodes, while citation relationships form edges between papers. Each paper is associated with a feature representation derived from its textual content and a research-category label. This combination makes Cora particularly suitable for studying semi-supervised node classification using both document features and citation structure.

The relatively modest scale of Cora makes it useful for validating implementations of GCN, GAT, GraphSAGE, and related message-passing architectures before moving to larger benchmarks. Researchers can rapidly test propagation depth, hidden dimensions, dropout, normalization, and learning-rate settings without requiring large computational resources. However, strong results on Cora alone should not be interpreted as evidence that a model will remain effective on large, heterogeneous, or dynamically changing graphs.

PubMed provides a larger citation-network environment and is commonly used to evaluate whether graph learning methods remain effective beyond small academic benchmarks. Nodes correspond to biomedical publications, citation links define graph connectivity, and node attributes summarize document information. The prediction problem typically involves assigning publications to predefined categories, allowing models to exploit both semantic features and relationships among scientific documents.

Because PubMed contains substantially more nodes and edges than smaller citation datasets such as Cora, it provides a useful intermediate step between prototype experiments and large-scale graph learning. Models that perform well on Cora may encounter different optimization, memory, and propagation behavior when applied to PubMed. Comparing results across both datasets therefore helps reveal whether improvements arise from robust architectural principles or from characteristics specific to a small graph.

The Open Graph Benchmark, commonly abbreviated as OGB, extends graph evaluation toward larger and more standardized experimental settings. Rather than representing a single graph, OGB provides a collection of benchmark datasets covering node prediction, link prediction, and graph-level prediction. Its standardized dataset processing, evaluation metrics, and predefined data splits are designed to reduce inconsistencies that can otherwise make results from different graph learning studies difficult to compare.

OGB is especially valuable for testing scalability because several of its datasets contain graphs that are substantially larger and structurally more complex than traditional citation benchmarks. Experiments may require neighborhood sampling, mini-batch processing, sparse operations, efficient data loading, or distributed computation. Consequently, OGB connects algorithmic evaluation with engineering concerns such as GPU memory utilization, training throughput, preprocessing cost, and inference scalability.

Another important contribution of OGB is the use of predefined training, validation, and test splits that often reflect realistic prediction conditions. Depending on the benchmark, splits may account for temporal ordering, molecular structure, domain-specific relationships, or other constraints. This discourages arbitrary experimental partitioning and makes comparisons between architectures more meaningful. Evaluation should therefore preserve the official split and metric definitions whenever benchmark comparability is a primary objective.

Custom datasets become necessary when standard benchmarks cannot represent the relationships, attributes, constraints, or operating conditions of a target application. A custom graph may represent robots and spatial regions, road infrastructure, industrial equipment, financial transactions, biological interactions, communication networks, or knowledge entities. Designing such datasets requires explicit definitions of what constitutes a node, an edge, a feature, a label, and the prediction target.

Constructing a custom dataset begins with converting raw domain information into a consistent graph schema. Node identifiers must remain stable, edge direction and relation semantics must be defined, and numerical or categorical attributes must be transformed into usable features. Missing values, duplicated relationships, isolated nodes, self-loops, temporal records, and inconsistent labels require deliberate handling because preprocessing decisions can alter the topology that the learning algorithm ultimately observes.

Data splitting is particularly important for custom graph datasets because random splitting can introduce information leakage through graph connectivity. A node in the training set may be directly connected to test nodes, future edges may accidentally appear during training, or entities representing the same physical object may occur across multiple partitions. Depending on the application, experiments may therefore require temporal, spatial, entity-disjoint, inductive, or topology-aware splits rather than conventional random sampling.

Custom datasets should also be documented with the same rigor expected from public benchmarks. Dataset versions, graph construction rules, feature transformations, label definitions, filtering criteria, split generation, preprocessing software, and basic graph statistics should be recorded. Without this information, improvements attributed to a model may actually result from undocumented changes in data preparation. Dataset versioning therefore becomes an essential component of reproducible graph experimentation.

Cora, PubMed, OGB, and custom datasets together form a practical progression of experimental difficulty within graph deep learning. Cora supports rapid verification and educational experimentation, PubMed introduces greater scale, OGB enables standardized and more demanding benchmarking, and custom datasets test whether graph learning techniques transfer to real application structures. Using multiple dataset levels prevents evaluation from becoming overly dependent on the assumptions of a single benchmark family.

A strong experimental workflow should therefore treat the dataset as part of the benchmark specification rather than merely as input to a model. Graph statistics, preprocessing, split policy, task definition, evaluation metric, and computational requirements should be established before comparing architectures. This dataset-centered discipline provides the foundation for the subsequent benchmark methodology, reproducibility, hyperparameter tuning, and model comparison stages defined in the experimental structure of Graph Deep Learning.

그래프 데이터셋(Graph Datasets)은 통제되고 비교 가능한 조건에서 그래프 학습 모델(Graph Learning Models)을 평가하기 위한 실험적 기반을 제공한다. 일반적인 테이블형 데이터셋(Tabular Datasets)과 달리 그래프 데이터셋은 개체(Entity)를 노드(Node)로, 관계(Relationship)를 엣지(Edge)로 표현하며, 여기에 속성(Attribute), 레이블(Label), 타임스탬프(Timestamp), 관계 유형(Relation Type) 등이 추가될 수 있다. 따라서 데이터셋 선택은 학습 데이터의 양뿐만 아니라 모델이 학습해야 하는 토폴로지(Topology), 희소성(Sparsity), 동질성(Homophily), 규모(Scale), 구조적 복잡성(Structural Complexity)을 결정한다.

유용한 그래프 벤치마크(Graph Benchmark)는 모델이 특징 정보(Feature Information)와 관계 구조(Relational Structure)를 얼마나 효과적으로 결합하는지 평가할 수 있도록 두 요소를 모두 포함해야 한다. 중요한 특성에는 노드와 엣지의 수, 특징 차원(Feature Dimensionality), 클래스 분포(Class Distribution), 평균 차수(Average Degree), 연결 요소(Connected Components), 그래프 밀도(Graph Density), 레이블 가용성(Label Availability) 등이 있다. 이러한 특성은 메모리 사용량, 수렴 특성, 이웃 집계(Neighborhood Aggregation), 샘플링 전략(Sampling Strategy), 최종 성능 해석에 큰 영향을 준다.

코라(Cora)는 그래프 신경망(Graph Neural Network)의 기초적인 실험에서 가장 널리 사용되는 인용 네트워크 데이터셋(Citation Network Dataset) 중 하나이다. 과학 논문은 노드(Node)로 표현되고, 논문 사이의 인용 관계는 엣지(Edge)를 구성한다. 각 논문에는 텍스트 내용에서 추출된 특징 표현(Feature Representation)과 연구 분야를 나타내는 레이블(Label)이 연결되어 있다. 이러한 구성은 문서 특징과 인용 구조를 함께 이용하는 준지도 노드 분류(Semi-Supervised Node Classification)를 연구하는 데 특히 적합하다.

코라(Cora)는 비교적 작은 규모이므로 대규모 벤치마크로 이동하기 전에 그래프 합성곱 신경망(GCN), 그래프 어텐션 네트워크(GAT), 그래프세이지(GraphSAGE)와 같은 메시지 패싱(Message Passing) 구조의 구현을 검증하는 데 유용하다. 연구자는 대규모 계산 자원 없이도 전파 깊이(Propagation Depth), 은닉 차원(Hidden Dimension), 드롭아웃(Dropout), 정규화(Normalization), 학습률(Learning Rate)을 빠르게 실험할 수 있다. 그러나 Cora에서 높은 성능을 얻었다고 해서 대규모·이종·동적 그래프에서도 동일한 성능을 보장한다고 해석해서는 안 된다.

펍메드(PubMed)는 더 큰 규모의 인용 네트워크 환경을 제공하며, 그래프 학습 방법이 소규모 학술 벤치마크를 넘어 효과적으로 작동하는지를 평가하는 데 일반적으로 사용된다. 노드는 생의학 분야의 논문을 나타내고, 인용 링크(Citation Link)는 그래프 연결 구조를 정의하며, 노드 속성(Node Attribute)은 문서 정보를 요약한다. 일반적인 예측 문제는 논문을 사전에 정의된 범주에 할당하는 것으로, 모델은 의미적 특징(Semantic Feature)과 과학 문서 사이의 관계를 동시에 활용할 수 있다.

펍메드(PubMed)는 Cora와 같은 소규모 인용 데이터셋보다 훨씬 많은 노드와 엣지를 포함하기 때문에 프로토타입 실험(Prototype Experiment)과 대규모 그래프 학습(Large-Scale Graph Learning) 사이의 유용한 중간 단계가 된다. Cora에서 우수한 모델도 PubMed에서는 서로 다른 최적화, 메모리 사용, 정보 전파 특성을 경험할 수 있다. 따라서 두 데이터셋의 결과를 비교하면 성능 향상이 일반적인 아키텍처 원리에서 발생한 것인지, 특정 소규모 그래프의 특성에 의존한 것인지를 판단하는 데 도움이 된다.

오픈 그래프 벤치마크(Open Graph Benchmark), 즉 OGB는 그래프 평가를 보다 크고 표준화된 실험 환경으로 확장한다. OGB는 하나의 그래프만을 의미하는 것이 아니라 노드 예측(Node Prediction), 링크 예측(Link Prediction), 그래프 수준 예측(Graph-Level Prediction)을 포함하는 여러 벤치마크 데이터셋의 집합을 제공한다. 표준화된 데이터 처리, 평가 지표(Evaluation Metric), 사전 정의된 데이터 분할(Predefined Data Split)은 서로 다른 그래프 학습 연구 결과를 비교할 때 발생할 수 있는 불일치를 줄이도록 설계되어 있다.

OGB는 전통적인 인용 벤치마크보다 훨씬 크고 구조적으로 복잡한 그래프를 포함하기 때문에 확장성(Scalability)을 평가하는 데 특히 유용하다. 실험 과정에서는 이웃 샘플링(Neighborhood Sampling), 미니배치 처리(Mini-Batch Processing), 희소 연산(Sparse Operation), 효율적인 데이터 로딩(Data Loading), 분산 연산(Distributed Computing)이 필요할 수 있다. 따라서 OGB는 알고리즘 평가를 GPU 메모리 활용, 학습 처리량(Training Throughput), 전처리 비용, 추론 확장성(Inference Scalability)과 같은 엔지니어링 문제와 연결한다.

OGB의 또 다른 중요한 특징은 현실적인 예측 조건을 반영할 수 있도록 사전에 정의된 학습·검증·테스트 분할(Training, Validation, Test Split)을 제공한다는 점이다. 벤치마크에 따라 시간 순서(Temporal Ordering), 분자 구조(Molecular Structure), 도메인별 관계 또는 기타 제약 조건이 분할 과정에 반영될 수 있다. 이는 임의적인 실험 데이터 분할을 줄이고 아키텍처 간 비교를 더욱 의미 있게 만든다. 따라서 벤치마크 비교 가능성이 중요한 경우에는 공식 데이터 분할과 평가 지표 정의를 유지하는 것이 바람직하다.

사용자 정의 데이터셋(Custom Dataset)은 표준 벤치마크가 실제 대상 응용 분야의 관계, 속성, 제약 조건 또는 운영 환경을 충분히 표현하지 못할 때 필요하다. 사용자 정의 그래프는 로봇과 공간 영역, 도로 인프라, 산업 장비, 금융 거래, 생물학적 상호작용, 통신 네트워크 또는 지식 개체(Knowledge Entity)를 표현할 수 있다. 이러한 데이터셋을 설계하려면 무엇을 노드, 엣지, 특징, 레이블 그리고 예측 대상(Prediction Target)으로 정의할 것인지 명확하게 결정해야 한다.

사용자 정의 데이터셋(Custom Dataset)의 구축은 원시 도메인 정보(Raw Domain Information)를 일관된 그래프 스키마(Graph Schema)로 변환하는 과정에서 시작된다. 노드 식별자(Node Identifier)는 안정적으로 유지되어야 하며, 엣지 방향(Edge Direction)과 관계 의미(Relation Semantics)가 명확하게 정의되어야 한다. 또한 수치형 또는 범주형 속성은 모델이 사용할 수 있는 특징으로 변환되어야 한다. 결측값, 중복 관계, 고립 노드(Isolated Node), 셀프 루프(Self-Loop), 시간 기록, 불일치 레이블 등은 최종 그래프 토폴로지에 영향을 줄 수 있으므로 신중하게 처리해야 한다.

사용자 정의 그래프 데이터셋에서는 데이터 분할(Data Splitting)이 특히 중요하다. 단순한 무작위 분할(Random Split)은 그래프 연결 구조를 통해 정보 누출(Information Leakage)을 발생시킬 수 있기 때문이다. 학습 세트의 노드가 테스트 노드와 직접 연결되거나 미래 시점의 엣지가 학습 과정에 포함될 수 있으며, 동일한 물리적 개체를 나타내는 데이터가 여러 분할 영역에 동시에 존재할 수도 있다. 따라서 응용 분야에 따라 시간 기반(Temporal), 공간 기반(Spatial), 개체 분리형(Entity-Disjoint), 귀납형(Inductive), 토폴로지 인식형(Topology-Aware) 분할이 필요할 수 있다.

사용자 정의 데이터셋(Custom Dataset) 역시 공개 벤치마크와 동일한 수준의 체계적인 문서화가 필요하다. 데이터셋 버전(Dataset Version), 그래프 구축 규칙(Graph Construction Rule), 특징 변환(Feature Transformation), 레이블 정의, 필터링 기준, 데이터 분할 생성 방법, 전처리 소프트웨어, 기본 그래프 통계 등을 기록해야 한다. 이러한 정보가 없다면 모델의 개선으로 보이는 결과가 실제로는 문서화되지 않은 데이터 준비 과정의 변화에서 발생했을 가능성이 있다. 따라서 데이터셋 버전 관리(Dataset Versioning)는 재현 가능한 그래프 실험(Reproducible Graph Experiment)의 핵심 요소가 된다.

코라(Cora), 펍메드(PubMed), OGB, 사용자 정의 데이터셋(Custom Dataset)은 그래프 딥러닝(Graph Deep Learning) 실험의 난이도를 단계적으로 확장하는 실용적인 흐름을 형성한다. Cora는 빠른 구현 검증과 교육용 실험에 적합하고, PubMed는 더 큰 규모의 그래프를 제공하며, OGB는 표준화되고 더욱 까다로운 벤치마킹을 가능하게 한다. 사용자 정의 데이터셋은 그래프 학습 기술이 실제 응용 분야의 구조로 전이될 수 있는지를 검증한다. 여러 수준의 데이터셋을 함께 사용하면 하나의 벤치마크 계열이 가진 특정 가정에 평가가 지나치게 의존하는 것을 방지할 수 있다.

따라서 강건한 실험 워크플로(Experimental Workflow)는 데이터셋을 단순히 모델에 입력되는 데이터로 취급하는 것이 아니라 벤치마크 명세(Benchmark Specification)의 일부로 다루어야 한다. 아키텍처를 비교하기 전에 그래프 통계, 전처리(Preprocessing), 데이터 분할 정책(Split Policy), 태스크 정의(Task Definition), 평가 지표, 계산 자원 요구사항을 명확히 설정해야 한다. 이러한 데이터셋 중심의 실험 원칙은 이후의 벤치마크 방법론(Benchmark Methodology), 재현성(Reproducibility), 하이퍼파라미터 튜닝(Hyperparameter Tuning), 모델 비교(Model Comparison)를 수행하기 위한 기반을 제공한다.

##  

## 09.02. Benchmark Methodology

![](images/image2.png){width="7.268055555555556in" height="7.268055555555556in"}

A benchmark methodology for graph deep learning defines a controlled experimental framework for determining whether one model actually performs better than another. Because graph models depend simultaneously on node features, connectivity, sampling, propagation depth, and dataset structure, meaningful benchmarking requires more than reporting a final accuracy value. The complete experimental configuration must be treated as part of the benchmark itself.

The first principle is experimental consistency. Models being compared should use the same dataset version, preprocessing pipeline, training-validation-test split, evaluation protocol, and hardware conditions whenever possible. Changing several factors simultaneously makes it difficult to identify why performance changed. A benchmark should therefore isolate the model or technique being investigated while keeping unrelated experimental variables controlled.

Dataset characterization should precede model training. Researchers should record the number of nodes and edges, feature dimensions, label distribution, average degree, graph density, connected components, and other relevant structural statistics. These properties provide essential context because identical GNN architectures can behave very differently on sparse, dense, homophilic, heterophilic, small, or extremely large graphs.

Data splitting must reflect the intended learning scenario. Transductive evaluation allows the model to observe the overall graph structure while labels for validation and test nodes remain hidden during training. Inductive evaluation instead tests whether the model can generalize to previously unseen nodes, subgraphs, or graphs. Temporal applications may require chronological splits so that future information cannot influence predictions made from historical data.

Information leakage deserves special attention because relationships between samples make graph datasets fundamentally different from independent tabular examples. Features, edges, labels, preprocessing statistics, or neighborhood information from validation and test entities can unintentionally influence training. Benchmark methodology should explicitly describe which graph information is available during each stage and ensure that the resulting evaluation corresponds to the intended deployment condition.

Evaluation metrics must match the graph learning task. Node classification commonly uses accuracy, precision, recall, F1 score, or ROC-AUC depending on class balance and application requirements. Link prediction may use ROC-AUC, average precision, Hits@K, or ranking-based metrics, while graph classification can employ conventional classification measures. No single metric captures every aspect of performance, so metric selection should reflect the practical objective.

Training protocols should also be standardized. Optimizer type, learning rate, weight decay, batch size, number of epochs, early-stopping rules, hidden dimensions, propagation layers, dropout, neighborhood sampling, and learning-rate schedules can substantially affect results. A fair comparison should either use equivalent configurations or apply a clearly defined tuning procedure to every model rather than extensively optimizing only the preferred architecture.

Graph neural networks are sensitive to random initialization, data sampling, mini-batch construction, and stochastic optimization. Reporting the result of a single training run can therefore provide a misleading impression of model quality. Reliable benchmarks normally repeat experiments using multiple random seeds and report aggregate statistics such as the mean and standard deviation, making both expected performance and experimental variability visible.

Baseline selection determines whether an experimental improvement is meaningful. Comparisons should include simple feature-only methods when appropriate, established graph architectures such as GCN, GAT, or GraphSAGE, and stronger modern baselines relevant to the task. A complex architecture that only marginally outperforms a simple baseline may offer limited practical value if it requires substantially more memory, training time, or engineering complexity.

Ablation studies complement direct model comparisons by identifying which components actually produce an observed improvement. Individual mechanisms such as attention, residual connections, normalization, positional encoding, sampling strategies, or auxiliary objectives can be removed or replaced while the remaining system stays unchanged. This separates genuine architectural contributions from improvements caused by additional parameters or altered training procedures.

Performance benchmarking should extend beyond predictive quality when models are intended for practical deployment. Training time, inference latency, peak GPU memory, CPU memory, preprocessing cost, parameter count, throughput, and storage requirements may be measured alongside accuracy. For large graphs, scalability as the number of nodes and edges increases can become as important as small improvements in predictive performance.

Computational comparisons require careful interpretation because runtime depends on hardware and software environments. GPU model, CPU configuration, memory capacity, framework version, CUDA stack, sparse kernels, data-loader settings, and mixed-precision configuration can influence measured speed. Hardware-sensitive measurements should therefore be reported together with the environment in which they were obtained rather than presented as universally valid characteristics.

Scalability experiments examine how a graph learning system behaves as graph size or computational workload increases. Researchers can vary node count, edge count, neighborhood size, batch size, or sampling fan-out and observe changes in memory consumption, training throughput, and inference latency. Such experiments reveal practical limits that may remain invisible when evaluation is restricted to compact datasets such as Cora or PubMed.

Benchmarking across multiple datasets provides stronger evidence of generalization than optimizing for one dataset. Small citation networks support rapid iteration, larger standardized benchmarks such as OGB test scalability and controlled comparison, and custom datasets examine domain transfer. Consistent improvements across different graph structures provide stronger evidence that an architectural contribution is broadly useful rather than specialized to one benchmark.

The final benchmark record should connect dataset configuration, preprocessing, split policy, model architecture, hyperparameters, random seeds, evaluation metrics, computational environment, and measured results into one traceable experiment definition. This methodology establishes the foundation for reproducibility, hyperparameter tuning, and model comparison in the remaining experimental workflow, allowing reported improvements to be independently interpreted and verified.

그래프 딥러닝(Graph Deep Learning)의 벤치마크 방법론(Benchmark Methodology)은 한 모델이 다른 모델보다 실제로 더 우수한지를 판단하기 위한 통제된 실험 프레임워크(Experimental Framework)를 정의한다. 그래프 모델은 노드 특징(Node Features), 연결 구조(Connectivity), 샘플링(Sampling), 전파 깊이(Propagation Depth), 데이터셋 구조(Dataset Structure)에 동시에 영향을 받기 때문에 의미 있는 벤치마킹은 단순히 최종 정확도만 보고하는 것 이상을 요구한다. 전체 실험 구성을 벤치마크 자체의 일부로 다루어야 한다.

첫 번째 원칙은 실험 일관성(Experimental Consistency)이다. 비교되는 모델들은 가능한 한 동일한 데이터셋 버전(Dataset Version), 전처리 파이프라인(Preprocessing Pipeline), 학습·검증·테스트 분할(Training-Validation-Test Split), 평가 프로토콜(Evaluation Protocol), 하드웨어 조건(Hardware Conditions)을 사용해야 한다. 여러 요소를 동시에 변경하면 성능 변화의 원인을 파악하기 어렵다. 따라서 벤치마크에서는 관련 없는 실험 변수를 통제하면서 조사하려는 모델이나 기법을 독립적으로 평가해야 한다.

데이터셋 특성화(Dataset Characterization)는 모델 학습보다 먼저 수행되어야 한다. 연구자는 노드와 엣지의 수, 특징 차원(Feature Dimensions), 레이블 분포(Label Distribution), 평균 차수(Average Degree), 그래프 밀도(Graph Density), 연결 요소(Connected Components) 및 기타 관련 구조적 통계를 기록해야 한다. 동일한 그래프 신경망(GNN) 아키텍처도 희소(Sparse), 밀집(Dense), 동질적(Homophilic), 이질적(Heterophilic), 소규모 또는 초대규모 그래프에서 매우 다르게 동작할 수 있기 때문에 이러한 특성은 실험 결과를 해석하는 데 필수적인 배경 정보를 제공한다.

데이터 분할(Data Splitting)은 의도한 학습 시나리오(Learning Scenario)를 반영해야 한다. 변환적 평가(Transductive Evaluation)에서는 검증 및 테스트 노드의 레이블을 학습 중 숨긴 상태에서 모델이 전체 그래프 구조를 관찰할 수 있다. 반면 귀납적 평가(Inductive Evaluation)는 모델이 이전에 보지 못한 노드, 서브그래프(Subgraph), 그래프에 일반화할 수 있는지를 평가한다. 시간적 응용(Temporal Application)에서는 미래 정보가 과거 데이터를 이용한 예측에 영향을 주지 않도록 시간 순서 기반 분할(Chronological Split)이 필요할 수 있다.

정보 누출(Information Leakage)은 그래프 데이터셋의 샘플들이 서로 독립적인 일반 테이블 데이터와 근본적으로 다르기 때문에 특별한 주의가 필요하다. 검증 및 테스트 개체의 특징, 엣지, 레이블, 전처리 통계 또는 이웃 정보(Neighborhood Information)가 의도하지 않게 학습 과정에 영향을 줄 수 있다. 따라서 벤치마크 방법론은 각 단계에서 어떤 그래프 정보가 사용 가능한지를 명확히 설명하고, 최종 평가가 실제 의도한 배포 조건(Deployment Condition)에 대응하도록 해야 한다.

평가 지표(Evaluation Metrics)는 그래프 학습 태스크(Graph Learning Task)에 적합해야 한다. 노드 분류(Node Classification)에서는 클래스 균형과 응용 요구사항에 따라 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1 점수(F1 Score), ROC-AUC 등을 사용할 수 있다. 링크 예측(Link Prediction)에서는 ROC-AUC, 평균 정밀도(Average Precision), Hits@K 또는 순위 기반 지표(Ranking-Based Metrics)를 사용할 수 있으며, 그래프 분류(Graph Classification)에서는 일반적인 분류 지표를 적용할 수 있다. 하나의 지표만으로 모든 성능 특성을 표현할 수 없으므로 실제 목적에 맞게 지표를 선택해야 한다.

학습 프로토콜(Training Protocol) 역시 표준화되어야 한다. 옵티마이저(Optimizer) 종류, 학습률(Learning Rate), 가중치 감쇠(Weight Decay), 배치 크기(Batch Size), 에포크(Epoch) 수, 조기 종료(Early Stopping) 규칙, 은닉 차원(Hidden Dimensions), 전파 계층(Propagation Layers), 드롭아웃(Dropout), 이웃 샘플링(Neighborhood Sampling), 학습률 스케줄(Learning-Rate Schedule)은 결과에 상당한 영향을 줄 수 있다. 공정한 비교를 위해서는 동일하거나 동등한 설정을 적용하거나, 모든 모델에 동일하게 정의된 튜닝 절차(Tuning Procedure)를 적용해야 한다.

그래프 신경망(Graph Neural Networks)은 무작위 초기화(Random Initialization), 데이터 샘플링(Data Sampling), 미니배치 구성(Mini-Batch Construction), 확률적 최적화(Stochastic Optimization)에 민감하다. 따라서 한 번의 학습 결과만 보고하면 모델 품질에 대한 잘못된 인상을 줄 수 있다. 신뢰할 수 있는 벤치마크는 일반적으로 여러 랜덤 시드(Random Seeds)를 사용하여 실험을 반복하고 평균(Mean)과 표준편차(Standard Deviation) 등의 통계값을 보고함으로써 기대 성능과 실험 변동성을 함께 보여준다.

베이스라인 선택(Baseline Selection)은 실험에서 관찰된 성능 향상이 실제로 의미가 있는지를 결정한다. 필요한 경우 단순한 특징 기반 방법(Feature-Only Method)을 포함하고, GCN, GAT, GraphSAGE와 같은 기존 그래프 아키텍처 및 해당 태스크에 적합한 강력한 최신 베이스라인(Modern Baseline)을 함께 비교해야 한다. 복잡한 아키텍처가 단순한 베이스라인보다 약간 높은 성능만 제공하면서 훨씬 많은 메모리, 학습 시간, 엔지니어링 복잡성을 요구한다면 실제 활용 가치는 제한적일 수 있다.

절제 연구(Ablation Study)는 직접적인 모델 비교를 보완하여 어떤 구성 요소가 실제 성능 향상을 만들어내는지를 파악한다. 어텐션(Attention), 잔차 연결(Residual Connection), 정규화(Normalization), 위치 인코딩(Positional Encoding), 샘플링 전략(Sampling Strategy), 보조 목적 함수(Auxiliary Objective)와 같은 개별 메커니즘을 제거하거나 대체하면서 나머지 시스템은 동일하게 유지할 수 있다. 이를 통해 실제 아키텍처의 기여와 단순히 파라미터 증가 또는 학습 절차 변경으로 발생한 개선을 구분할 수 있다.

실제 시스템 배포를 목적으로 하는 모델의 성능 벤치마킹(Performance Benchmarking)은 예측 품질(Predictive Quality) 이상으로 확장되어야 한다. 정확도와 함께 학습 시간(Training Time), 추론 지연시간(Inference Latency), 최대 GPU 메모리(Peak GPU Memory), CPU 메모리, 전처리 비용(Preprocessing Cost), 파라미터 수(Parameter Count), 처리량(Throughput), 저장 공간 요구사항(Storage Requirements)을 측정할 수 있다. 대규모 그래프에서는 노드와 엣지가 증가할 때의 확장성(Scalability)이 작은 예측 성능 향상만큼 중요해질 수 있다.

연산 성능 비교(Computational Comparison)는 실행 시간이 하드웨어 및 소프트웨어 환경에 영향을 받기 때문에 신중하게 해석해야 한다. GPU 모델, CPU 구성, 메모리 용량, 프레임워크 버전(Framework Version), CUDA 스택(CUDA Stack), 희소 연산 커널(Sparse Kernel), 데이터 로더(Data Loader) 설정, 혼합 정밀도(Mixed Precision) 구성 등이 측정된 속도에 영향을 줄 수 있다. 따라서 하드웨어 의존적인 측정값은 보편적인 특성으로 제시하기보다 해당 결과가 측정된 실행 환경과 함께 보고해야 한다.

확장성 실험(Scalability Experiment)은 그래프 크기 또는 계산 부하가 증가할 때 그래프 학습 시스템의 동작이 어떻게 변화하는지를 평가한다. 연구자는 노드 수, 엣지 수, 이웃 크기(Neighborhood Size), 배치 크기, 샘플링 팬아웃(Sampling Fan-Out)을 변화시키면서 메모리 사용량, 학습 처리량(Training Throughput), 추론 지연시간의 변화를 관찰할 수 있다. 이러한 실험은 Cora나 PubMed와 같은 비교적 작은 데이터셋만 사용한 평가에서는 드러나지 않는 시스템의 실질적인 한계를 보여준다.

여러 데이터셋을 이용한 벤치마킹은 하나의 데이터셋만을 대상으로 최적화하는 것보다 모델의 일반화(Generalization)에 대한 강력한 근거를 제공한다. 소규모 인용 네트워크(Citation Network)는 빠른 실험 반복을 지원하고, OGB와 같은 대규모 표준 벤치마크는 확장성과 통제된 비교를 평가하며, 사용자 정의 데이터셋(Custom Dataset)은 도메인 전이(Domain Transfer)를 검증한다. 서로 다른 그래프 구조에서 일관된 성능 향상이 나타난다면 해당 아키텍처의 기여가 특정 벤치마크에 국한되지 않고 폭넓게 활용될 수 있다는 더욱 강한 근거가 된다.

최종 벤치마크 기록(Benchmark Record)은 데이터셋 구성, 전처리, 데이터 분할 정책(Split Policy), 모델 아키텍처(Model Architecture), 하이퍼파라미터(Hyperparameters), 랜덤 시드, 평가 지표, 계산 환경(Computational Environment), 측정 결과를 하나의 추적 가능한 실험 정의(Traceable Experiment Definition)로 연결해야 한다. 이러한 방법론은 이후 실험 과정의 재현성(Reproducibility), 하이퍼파라미터 튜닝(Hyperparameter Tuning), 모델 비교(Model Comparison)를 위한 기반을 제공하며, 보고된 성능 향상을 독립적으로 해석하고 검증할 수 있도록 한다.

##  

## 09.03. Reproducibility [w/Code]

![](images/image3.png){width="7.268055555555556in" height="7.268055555555556in"}

Reproducibility in graph deep learning means that an experiment can be repeated under clearly documented conditions and produce results that are statistically consistent with the original findings. This is especially important for graph neural networks because performance may vary with random initialization, graph partitioning, neighborhood sampling, stochastic optimization, and hardware-dependent computation. Reproducibility therefore concerns the complete experimental process rather than only preserving model code.

A reproducible experiment begins with deterministic control of randomness wherever practical. Python, NumPy, and the deep learning framework may each maintain independent random number generators, while graph libraries can introduce additional randomness through sampling or data loading. A common implementation pattern is to define one seed value and propagate it consistently across these components so that initialization, shuffling, sampling, and other stochastic operations can be reconstructed.

In Python-based graph experiments, seed initialization can conceptually be implemented with random.seed(seed), numpy.random.seed(seed), torch.manual_seed(seed), and torch.cuda.manual_seed_all(seed). When CUDA is used, deterministic algorithm settings may further reduce nondeterministic behavior. Complete determinism is not always guaranteed or desirable, however, because some optimized GPU kernels trade deterministic execution for higher performance and may behave differently across hardware or software versions.

Reproducibility should not be confused with reporting only one deterministic run. A fixed seed is useful for debugging and exact experiment reconstruction, but scientific conclusions should account for stochastic variability. Graph models should normally be trained with multiple predefined seeds, after which evaluation metrics can be summarized using the mean, standard deviation, confidence intervals, or other suitable statistics. This separates reproducible execution from statistically reliable evaluation.

Dataset reproducibility requires precise control of both source data and graph construction. The dataset name alone is insufficient because different library versions may apply different preprocessing, filtering, feature normalization, edge handling, or split generation. Experiments should therefore preserve the dataset version, download source or identifier, preprocessing configuration, graph schema, and relevant statistics such as node count, edge count, feature dimensions, and class distribution.

Training, validation, and test splits must also be reproducible. If predefined benchmark splits exist, they should be preserved rather than regenerated without documentation. For custom datasets, split indices or the exact algorithm and seed used to create them should be stored. This is particularly important for graph learning because small changes in node or edge partitions can alter neighborhood information and produce evaluation conditions that are no longer directly comparable.

The preprocessing pipeline should be treated as executable experimental configuration. Operations such as feature normalization, self-loop insertion, edge deduplication, conversion to undirected graphs, isolated-node handling, negative-edge sampling, and graph augmentation can substantially change the effective dataset. These transformations should be explicitly recorded and applied in a fixed order so that another execution reconstructs the same graph presented to the model.

Model configuration must be preserved independently from source code. Hidden dimensions, number of message-passing layers, aggregation functions, attention heads, activation functions, normalization, residual connections, dropout, and initialization rules should be stored as configuration parameters. This makes it possible to reproduce an architecture without relying on undocumented defaults that may change when the implementation or graph-learning framework is updated.

The same principle applies to training configuration. Optimizer, learning rate, weight decay, batch size, epoch limit, early-stopping patience, learning-rate scheduler, gradient clipping, sampling fan-out, loss function, and checkpoint policy should be recorded. Configuration files or structured experiment dictionaries are preferable to manually modifying constants inside source code because they create a traceable connection between a reported result and the exact settings that produced it.

Software environments are another major source of reproducibility failures. Python, PyTorch, CUDA, GPU drivers, graph libraries such as PyTorch Geometric or DGL, and supporting packages can change numerical behavior or API semantics between releases. Environment information should therefore be captured using dependency files, environment specifications, containers, or equivalent mechanisms so that the software stack associated with an experiment can later be reconstructed.

Hardware information should also accompany benchmark results when computational measurements are reported. GPU architecture, GPU memory, CPU, system memory, precision mode, and accelerator configuration can influence runtime, memory consumption, and sometimes numerical behavior. Training time or inference latency without hardware context is difficult to reproduce meaningfully. Reproducibility therefore includes both algorithmic configuration and the computational environment in which measurements were obtained.

Experiment tracking provides a practical mechanism for connecting all of these elements. Each run can be assigned a unique identifier associated with dataset version, split, model configuration, training parameters, seed, software environment, hardware information, metrics, and checkpoints. Logging systems can additionally preserve learning curves and resource statistics, allowing researchers to reconstruct not merely the final score but the sequence of events that produced it.

Model checkpoints should be saved together with sufficient metadata to reload and evaluate them independently. A checkpoint may include model parameters, optimizer state, scheduler state, current epoch, random seed, configuration identifier, and best validation score. Saving only the neural network weights may be adequate for inference, but it is insufficient when the objective is to resume training or reconstruct the precise state of an interrupted experiment.

Reproducibility verification should be performed as an explicit experiment rather than assumed from documentation. A completed run can be repeated from a clean environment using only the stored code, dataset instructions, configuration, and dependencies. The resulting metrics should then be compared with the recorded values within an expected tolerance. This process exposes hidden dependencies, undocumented preprocessing steps, local files, and accidental assumptions before results are published or shared.

A practical reproducibility package ultimately connects code, data, configuration, environment, seeds, checkpoints, logs, and evaluation results into one traceable experimental artifact. Within the Graph Deep Learning experimental workflow, this foundation makes subsequent hyperparameter tuning and model comparison credible because every candidate can be reconstructed under consistent conditions. Reproducibility transforms a successful individual run into evidence that can be independently tested, compared, extended, and trusted.

그래프 딥러닝(Graph Deep Learning)에서 재현성(Reproducibility)이란 명확하게 문서화된 조건에서 실험을 반복했을 때 원래의 결과와 통계적으로 일관된 결과를 얻을 수 있음을 의미한다. 그래프 신경망(Graph Neural Networks)은 무작위 초기화(Random Initialization), 그래프 분할(Graph Partitioning), 이웃 샘플링(Neighborhood Sampling), 확률적 최적화(Stochastic Optimization), 하드웨어 의존적 연산에 따라 성능이 달라질 수 있기 때문에 특히 중요하다. 따라서 재현성은 단순한 모델 코드 보존이 아니라 전체 실험 과정의 재구성을 의미한다.

재현 가능한 실험(Reproducible Experiment)은 가능한 범위에서 무작위성(Randomness)을 결정론적으로 제어하는 것에서 시작한다. Python, NumPy, 딥러닝 프레임워크(Deep Learning Framework)는 각각 독립적인 난수 생성기(Random Number Generator)를 사용할 수 있으며, 그래프 라이브러리(Graph Library)는 샘플링이나 데이터 로딩 과정에서 추가적인 무작위성을 발생시킬 수 있다. 일반적인 구현 방식은 하나의 시드 값(Seed Value)을 정의하고 이를 모든 구성 요소에 일관되게 적용하여 초기화, 셔플링, 샘플링 등의 확률적 연산을 재구성할 수 있도록 하는 것이다.

Python 기반 그래프 실험에서는 시드 초기화(Seed Initialization)를 개념적으로 random.seed(seed), numpy.random.seed(seed), torch.manual_seed(seed), torch.cuda.manual_seed_all(seed)와 같은 방식으로 구현할 수 있다. CUDA를 사용하는 경우 결정론적 알고리즘 설정(Deterministic Algorithm Settings)을 통해 비결정적 동작을 추가로 줄일 수 있다. 그러나 일부 최적화된 GPU 커널(GPU Kernel)은 결정론적 실행보다 높은 성능을 우선하며 하드웨어나 소프트웨어 버전에 따라 다르게 동작할 수 있으므로 완전한 결정론(Complete Determinism)이 항상 보장되거나 바람직한 것은 아니다.

재현성(Reproducibility)을 하나의 결정론적 실행 결과만 보고하는 것과 혼동해서는 안 된다. 고정 시드(Fixed Seed)는 디버깅(Debugging)과 정확한 실험 재구성에는 유용하지만 과학적 결론은 확률적 변동성(Stochastic Variability)을 고려해야 한다. 일반적으로 그래프 모델은 사전에 정의된 여러 랜덤 시드(Random Seeds)를 사용하여 반복 학습한 후 평균(Mean), 표준편차(Standard Deviation), 신뢰구간(Confidence Interval) 또는 적절한 통계량으로 평가 결과를 요약해야 한다. 이를 통해 재현 가능한 실행과 통계적으로 신뢰할 수 있는 평가를 구분할 수 있다.

데이터셋 재현성(Dataset Reproducibility)은 원본 데이터(Source Data)와 그래프 구축(Graph Construction)을 모두 정확하게 제어해야 한다. 데이터셋 이름만 기록하는 것으로는 충분하지 않으며, 라이브러리 버전에 따라 전처리(Preprocessing), 필터링(Filtering), 특징 정규화(Feature Normalization), 엣지 처리(Edge Handling), 데이터 분할 생성 방식이 달라질 수 있다. 따라서 데이터셋 버전, 다운로드 소스 또는 식별자, 전처리 구성, 그래프 스키마(Graph Schema), 노드 수, 엣지 수, 특징 차원, 클래스 분포 등의 주요 통계를 함께 보존해야 한다.

학습·검증·테스트 분할(Training-Validation-Test Split) 역시 재현 가능해야 한다. 사전에 정의된 벤치마크 분할(Predefined Benchmark Split)이 존재한다면 별도의 기록 없이 새롭게 생성하기보다는 기존 분할을 유지해야 한다. 사용자 정의 데이터셋(Custom Dataset)의 경우에는 분할 인덱스(Split Indices)를 저장하거나 분할 생성에 사용된 정확한 알고리즘과 시드를 기록해야 한다. 그래프 학습에서는 노드 또는 엣지 분할의 작은 변화도 이웃 정보를 변경하여 직접 비교하기 어려운 평가 조건을 만들 수 있기 때문에 특히 중요하다.

전처리 파이프라인(Preprocessing Pipeline)은 실행 가능한 실험 구성(Executable Experimental Configuration)의 일부로 취급해야 한다. 특징 정규화, 셀프 루프(Self-Loop) 삽입, 엣지 중복 제거(Edge Deduplication), 무방향 그래프(Undirected Graph) 변환, 고립 노드(Isolated Node) 처리, 음성 엣지 샘플링(Negative-Edge Sampling), 그래프 증강(Graph Augmentation) 등의 연산은 실제 모델에 입력되는 데이터셋을 크게 변화시킬 수 있다. 따라서 이러한 변환 과정과 적용 순서를 명확하게 기록하여 동일한 그래프를 다시 구성할 수 있도록 해야 한다.

모델 구성(Model Configuration)은 소스 코드(Source Code)와 독립적으로 보존되어야 한다. 은닉 차원(Hidden Dimensions), 메시지 패싱 계층(Message-Passing Layers)의 수, 집계 함수(Aggregation Functions), 어텐션 헤드(Attention Heads), 활성화 함수(Activation Functions), 정규화(Normalization), 잔차 연결(Residual Connections), 드롭아웃(Dropout), 초기화 규칙(Initialization Rules)을 구성 파라미터(Configuration Parameters)로 저장해야 한다. 이를 통해 구현이나 그래프 학습 프레임워크가 변경되더라도 문서화되지 않은 기본 설정(Default Settings)에 의존하지 않고 아키텍처를 재현할 수 있다.

동일한 원칙은 학습 구성(Training Configuration)에도 적용된다. 옵티마이저(Optimizer), 학습률(Learning Rate), 가중치 감쇠(Weight Decay), 배치 크기(Batch Size), 최대 에포크(Epoch), 조기 종료 인내값(Early-Stopping Patience), 학습률 스케줄러(Learning-Rate Scheduler), 그래디언트 클리핑(Gradient Clipping), 샘플링 팬아웃(Sampling Fan-Out), 손실 함수(Loss Function), 체크포인트 정책(Checkpoint Policy)을 기록해야 한다. 소스 코드 내부의 상수를 직접 수정하는 방식보다 구성 파일(Configuration File)이나 구조화된 실험 딕셔너리(Experiment Dictionary)를 사용하면 결과와 해당 결과를 생성한 설정 사이의 추적성을 확보할 수 있다.

소프트웨어 환경(Software Environment)은 재현성 실패의 또 다른 주요 원인이 된다. Python, PyTorch, CUDA, GPU 드라이버(GPU Driver), PyTorch Geometric이나 DGL과 같은 그래프 라이브러리 및 관련 패키지는 버전에 따라 수치적 동작이나 API 의미가 변경될 수 있다. 따라서 의존성 파일(Dependency File), 환경 명세(Environment Specification), 컨테이너(Container) 또는 이에 상응하는 방법으로 환경 정보를 저장하여 실험에 사용된 소프트웨어 스택(Software Stack)을 이후에도 재구성할 수 있도록 해야 한다.

연산 성능을 보고하는 경우에는 하드웨어 정보(Hardware Information)도 벤치마크 결과와 함께 기록해야 한다. GPU 아키텍처, GPU 메모리, CPU, 시스템 메모리(System Memory), 정밀도 모드(Precision Mode), 가속기 구성(Accelerator Configuration)은 실행 시간, 메모리 소비량, 경우에 따라 수치적 동작에도 영향을 미칠 수 있다. 하드웨어 정보가 없는 학습 시간이나 추론 지연시간(Inference Latency)은 의미 있게 재현하기 어렵다. 따라서 재현성은 알고리즘 구성뿐만 아니라 측정이 수행된 계산 환경(Computational Environment)까지 포함한다.

실험 추적(Experiment Tracking)은 이러한 모든 요소를 연결하는 실용적인 방법을 제공한다. 각 실행(Run)에 고유 식별자(Unique Identifier)를 부여하고 데이터셋 버전, 데이터 분할, 모델 구성, 학습 파라미터, 시드, 소프트웨어 환경, 하드웨어 정보, 평가 지표, 체크포인트(Checkpoint)를 연결할 수 있다. 로깅 시스템(Logging System)을 이용하면 학습 곡선(Learning Curve)과 자원 사용 통계까지 보존할 수 있어 최종 점수뿐만 아니라 해당 결과가 생성된 전체 과정을 재구성할 수 있다.

모델 체크포인트(Model Checkpoint)는 독립적으로 다시 불러오고 평가할 수 있도록 충분한 메타데이터(Metadata)와 함께 저장해야 한다. 체크포인트에는 모델 파라미터, 옵티마이저 상태(Optimizer State), 스케줄러 상태(Scheduler State), 현재 에포크, 랜덤 시드, 구성 식별자(Configuration Identifier), 최고 검증 점수(Best Validation Score) 등을 포함할 수 있다. 신경망 가중치만 저장하는 것은 추론(Inference)에는 충분할 수 있지만 중단된 학습을 재개하거나 정확한 실험 상태를 복원하려는 경우에는 충분하지 않다.

재현성 검증(Reproducibility Verification)은 문서화만으로 충족되었다고 가정하지 말고 명시적인 실험으로 수행해야 한다. 완료된 실행을 저장된 코드, 데이터셋 지침, 구성 정보, 의존성만 사용하여 깨끗한 환경(Clean Environment)에서 다시 수행할 수 있다. 이후 재실행된 평가 지표를 기존 기록과 예상 허용 오차(Expected Tolerance) 범위에서 비교해야 한다. 이 과정은 결과를 공개하거나 공유하기 전에 숨겨진 의존성(Hidden Dependency), 문서화되지 않은 전처리 과정, 로컬 파일, 우발적인 가정을 발견하는 데 도움이 된다.

실용적인 재현성 패키지(Reproducibility Package)는 최종적으로 코드(Code), 데이터(Data), 구성(Configuration), 환경(Environment), 시드(Seeds), 체크포인트(Checkpoints), 로그(Logs), 평가 결과(Evaluation Results)를 하나의 추적 가능한 실험 산출물(Traceable Experimental Artifact)로 연결한다. 그래프 딥러닝(Graph Deep Learning)의 실험 워크플로에서 이러한 기반은 이후의 하이퍼파라미터 튜닝(Hyperparameter Tuning)과 모델 비교(Model Comparison)를 신뢰할 수 있도록 한다. 재현성은 한 번 성공한 개별 실행을 독립적으로 검증하고 비교하며 확장하고 신뢰할 수 있는 실험적 증거로 전환한다.

##  

## 09.04. Hyperparameter Tuning [w/Code]

![](images/image4.png){width="7.268055555555556in" height="7.268055555555556in"}

Hyperparameter tuning in graph deep learning is the systematic process of selecting training and architectural settings that maximize validation performance while preserving generalization to unseen data. Unlike ordinary neural networks, GNNs are influenced not only by optimization parameters but also by message-passing depth, neighborhood structure, aggregation behavior, and graph sampling. Tuning therefore connects model architecture, graph topology, and training dynamics within one experimental process.

Hyperparameters should be distinguished from parameters learned through gradient descent. Model weights and biases are optimized during training, whereas learning rate, hidden dimension, number of GNN layers, dropout rate, weight decay, attention heads, batch size, and neighborhood sampling settings are selected externally. Their values define the conditions under which learning occurs and can substantially change both predictive quality and computational requirements.

The learning rate is usually one of the most influential optimization hyperparameters. A value that is too large may produce unstable updates or prevent convergence, while an excessively small value can make training unnecessarily slow or trap optimization in weak solutions. Instead of testing only closely spaced linear values, experiments commonly explore learning rates across logarithmic scales, such as orders of magnitude between relatively small and large candidate values.

Weight decay controls regularization by discouraging excessively large parameter values and can improve generalization when graph models overfit labeled nodes. Dropout provides another important regularization mechanism by randomly suppressing intermediate activations during training. These parameters interact with model capacity, dataset size, label availability, and graph structure, so their optimal values should be determined jointly rather than assumed to transfer unchanged across datasets.

The number of message-passing layers determines how far information can propagate through graph neighborhoods. Increasing depth expands the receptive field, allowing a node representation to incorporate information from progressively distant nodes. However, deeper GNNs can suffer from over-smoothing, optimization difficulties, or excessive information mixing. Layer count should therefore be tuned according to graph topology and the relational distance required by the prediction task.

Hidden dimensionality controls representation capacity and directly affects parameter count, memory consumption, and computational cost. Small embeddings may fail to encode sufficient information, whereas unnecessarily large dimensions can increase overfitting and resource usage without meaningful performance improvement. A useful tuning strategy evaluates progressively larger dimensions while monitoring validation quality together with memory consumption and training throughput.

Architecture-specific hyperparameters must also be included. GAT models may require tuning the number of attention heads, GraphSAGE may depend on the selected aggregation function and sampling fan-out, and graph transformers can introduce positional encodings, attention dimensions, and structural biases. Consequently, a universal search space is rarely sufficient; the tuning configuration should contain common parameters together with model-specific choices.

Neighborhood sampling introduces another tuning dimension for large graphs. Sampling fan-out determines how many neighbors are collected at each message-passing layer, influencing both the information available to the model and the computational workload. Large fan-outs approximate broader neighborhoods but increase memory and processing cost, while aggressive sampling improves scalability at the risk of discarding useful relational information.

Grid search evaluates predefined combinations of hyperparameter values and is straightforward to understand and reproduce. It is effective when the number of parameters and candidate values is small, but its computational cost grows rapidly with search-space dimensionality. For example, five hyperparameters with five candidate values each already require thousands of combinations, making exhaustive evaluation inefficient for expensive graph neural network training.

Random search samples configurations from predefined distributions instead of evaluating every combination. This approach often explores large search spaces more efficiently because only a subset of hyperparameters may strongly affect performance. In code, parameter distributions can be defined for learning rate, weight decay, hidden size, dropout, layer count, and sampling settings, after which each trial receives a sampled configuration and an independent experiment identifier.

More advanced optimization methods can use results from previous trials to decide which configurations should be evaluated next. Bayesian optimization estimates promising regions of the search space, while early-stopping and pruning strategies terminate poorly performing trials before they consume the full training budget. These approaches are particularly valuable when individual GNN experiments require substantial GPU memory, long preprocessing, or expensive neighborhood sampling.

The validation set should guide hyperparameter selection, while the test set must remain isolated until the final configuration has been chosen. Repeatedly selecting configurations according to test performance effectively turns the test set into part of the optimization process and produces optimistic estimates of generalization. A rigorous workflow searches on training data, ranks configurations using validation results, and evaluates the selected model on the test set only afterward.

A practical tuning loop can be expressed as a configuration generator followed by repeated train-and-evaluate operations. Each trial creates a model from its sampled hyperparameters, trains using the fixed benchmark protocol, measures validation metrics, and records computational statistics. The best configuration is selected according to a predefined objective such as validation accuracy, F1, ROC-AUC, or a combined performance-efficiency criterion.

Tuning results should be analyzed rather than reduced to only the winning configuration. Recording every trial enables researchers to examine parameter sensitivity, interactions, unstable regions, and performance-cost tradeoffs. Learning rate may dominate optimization behavior, for example, while hidden dimensions beyond a certain threshold may provide negligible gains. Such patterns reveal how the architecture behaves and can guide future experiments more efficiently.

Fair model comparison requires comparable tuning effort. Giving one architecture hundreds of trials while evaluating another with default parameters can create an artificial performance advantage unrelated to architectural quality. Search spaces do not need to be identical, because different models expose different parameters, but computational budgets, validation criteria, stopping rules, and optimization procedures should be reasonably equivalent.

The final tuning record should preserve the search space, sampling method, trial budget, random seeds, validation objective, early-stopping policy, complete trial history, selected configuration, and associated computational cost. Combined with the reproducibility framework, this creates a traceable path from candidate hyperparameters to the final benchmark model and provides the methodological foundation for reliable model comparison in the final stage of Graph Deep Learning experiments.

그래프 딥러닝(Graph Deep Learning)에서 하이퍼파라미터 튜닝(Hyperparameter Tuning)은 보지 못한 데이터(Unseen Data)에 대한 일반화(Generalization)를 유지하면서 검증 성능(Validation Performance)을 최대화할 수 있는 학습 및 아키텍처 설정을 체계적으로 선택하는 과정이다. 일반적인 신경망과 달리 그래프 신경망(GNN)은 최적화 파라미터뿐만 아니라 메시지 패싱 깊이(Message-Passing Depth), 이웃 구조(Neighborhood Structure), 집계 방식(Aggregation Behavior), 그래프 샘플링(Graph Sampling)의 영향을 받는다. 따라서 튜닝은 모델 아키텍처, 그래프 토폴로지(Graph Topology), 학습 동역학(Training Dynamics)을 하나의 실험 과정 안에서 연결한다.

하이퍼파라미터(Hyperparameters)는 경사하강법(Gradient Descent)을 통해 학습되는 파라미터와 구분해야 한다. 모델 가중치(Model Weights)와 편향(Biases)은 학습 과정에서 최적화되지만, 학습률(Learning Rate), 은닉 차원(Hidden Dimension), GNN 계층 수, 드롭아웃 비율(Dropout Rate), 가중치 감쇠(Weight Decay), 어텐션 헤드(Attention Heads), 배치 크기(Batch Size), 이웃 샘플링 설정(Neighborhood Sampling Settings)은 외부에서 선택된다. 이러한 값은 학습이 이루어지는 조건을 정의하며 예측 품질과 계산 자원 요구량 모두에 큰 영향을 줄 수 있다.

학습률(Learning Rate)은 일반적으로 가장 영향력이 큰 최적화 하이퍼파라미터 중 하나이다. 값이 너무 크면 업데이트가 불안정해지거나 수렴하지 않을 수 있으며, 지나치게 작으면 학습 속도가 불필요하게 느려지거나 좋지 않은 해에 머물 수 있다. 따라서 서로 가까운 선형 값만 시험하기보다 비교적 작은 값에서 큰 값까지 여러 자릿수 범위를 포함하는 로그 스케일(Logarithmic Scale)로 학습률을 탐색하는 방법이 일반적으로 사용된다.

가중치 감쇠(Weight Decay)는 지나치게 큰 파라미터 값을 억제하여 정규화(Regularization)를 수행하며, 그래프 모델이 레이블이 있는 노드에 과적합(Overfitting)될 때 일반화 성능을 향상시킬 수 있다. 드롭아웃(Dropout)은 학습 과정에서 중간 활성값(Intermediate Activations)을 무작위로 제거하는 또 다른 중요한 정규화 방법이다. 이러한 파라미터는 모델 용량(Model Capacity), 데이터셋 크기, 레이블 가용성(Label Availability), 그래프 구조와 상호작용하므로 서로 독립적으로 가정하기보다 함께 최적값을 탐색해야 한다.

메시지 패싱 계층(Message-Passing Layers)의 수는 정보가 그래프 이웃을 통해 얼마나 멀리 전파될 수 있는지를 결정한다. 깊이가 증가하면 수용 영역(Receptive Field)이 확장되어 노드 표현(Node Representation)이 점점 더 먼 노드의 정보를 포함할 수 있다. 그러나 깊은 GNN은 과도한 평활화(Over-Smoothing), 최적화 어려움, 지나친 정보 혼합(Information Mixing)을 경험할 수 있다. 따라서 계층 수는 그래프 토폴로지와 예측 태스크에 필요한 관계적 거리(Relational Distance)를 고려하여 튜닝해야 한다.

은닉 차원(Hidden Dimensionality)은 표현 용량(Representation Capacity)을 결정하며 파라미터 수(Parameter Count), 메모리 소비량(Memory Consumption), 계산 비용(Computational Cost)에 직접적인 영향을 준다. 너무 작은 임베딩(Embedding)은 충분한 정보를 표현하지 못할 수 있지만, 불필요하게 큰 차원은 의미 있는 성능 향상 없이 과적합과 자원 사용량을 증가시킬 수 있다. 따라서 점진적으로 더 큰 차원을 평가하면서 검증 성능과 함께 메모리 사용량 및 학습 처리량(Training Throughput)을 관찰하는 것이 유용하다.

아키텍처별 하이퍼파라미터(Architecture-Specific Hyperparameters)도 튜닝 대상에 포함해야 한다. GAT 모델은 어텐션 헤드 수(Number of Attention Heads)를 조정해야 할 수 있고, GraphSAGE는 선택된 집계 함수(Aggregation Function)와 샘플링 팬아웃(Sampling Fan-Out)에 영향을 받을 수 있다. 그래프 트랜스포머(Graph Transformer)는 위치 인코딩(Positional Encoding), 어텐션 차원(Attention Dimension), 구조적 편향(Structural Bias) 등을 추가할 수 있다. 따라서 하나의 보편적인 탐색 공간(Search Space)을 모든 모델에 동일하게 적용하기보다는 공통 파라미터와 모델별 설정을 함께 구성해야 한다.

이웃 샘플링(Neighborhood Sampling)은 대규모 그래프에서 또 하나의 중요한 튜닝 차원을 제공한다. 샘플링 팬아웃(Sampling Fan-Out)은 각 메시지 패싱 계층에서 수집하는 이웃의 수를 결정하며 모델이 활용할 수 있는 정보와 계산 부하 모두에 영향을 준다. 큰 팬아웃은 더 넓은 이웃을 근사할 수 있지만 메모리와 연산 비용을 증가시키며, 공격적인 샘플링(Aggressive Sampling)은 확장성을 향상시키는 대신 유용한 관계 정보를 제거할 위험이 있다.

그리드 탐색(Grid Search)은 사전에 정의된 하이퍼파라미터 값의 조합을 평가하는 방법으로 이해하기 쉽고 재현하기도 용이하다. 파라미터와 후보 값의 수가 적을 때 효과적이지만 탐색 공간의 차원이 증가하면 계산 비용이 급격히 증가한다. 예를 들어 5개의 하이퍼파라미터가 각각 5개의 후보 값을 가진다면 이미 수천 개의 조합을 평가해야 하므로 학습 비용이 높은 그래프 신경망에서는 모든 조합을 평가하는 방식이 비효율적일 수 있다.

무작위 탐색(Random Search)은 모든 조합을 평가하는 대신 사전에 정의된 분포에서 설정을 샘플링한다. 일부 하이퍼파라미터만이 성능에 강한 영향을 미칠 수 있기 때문에 대규모 탐색 공간을 보다 효율적으로 탐색할 수 있다. 코드에서는 학습률, 가중치 감쇠, 은닉 크기, 드롭아웃, 계층 수, 샘플링 설정 등에 대한 파라미터 분포(Parameter Distribution)를 정의하고 각 시행(Trial)에 샘플링된 구성과 독립적인 실험 식별자(Experiment Identifier)를 할당할 수 있다.

보다 발전된 최적화 방법은 이전 시행(Trial)의 결과를 이용하여 다음에 평가할 설정을 결정할 수 있다. 베이지안 최적화(Bayesian Optimization)는 탐색 공간에서 유망한 영역을 추정하며, 조기 종료(Early Stopping)와 가지치기 전략(Pruning Strategy)은 성능이 낮은 시행이 전체 학습 예산을 소비하기 전에 중단시킬 수 있다. 이러한 방법은 개별 GNN 실험에 많은 GPU 메모리, 긴 전처리 시간 또는 비용이 높은 이웃 샘플링이 필요한 경우 특히 유용하다.

검증 세트(Validation Set)는 하이퍼파라미터 선택의 기준으로 사용해야 하며, 테스트 세트(Test Set)는 최종 설정이 결정될 때까지 격리되어야 한다. 테스트 성능을 반복적으로 확인하면서 설정을 선택하면 사실상 테스트 세트가 최적화 과정의 일부가 되어 일반화 성능을 지나치게 낙관적으로 평가하게 된다. 엄격한 워크플로에서는 학습 데이터(Training Data)를 이용하여 후보 모델을 학습하고 검증 결과를 기준으로 설정을 선택한 다음, 선택된 모델만 최종적으로 테스트 세트에서 평가한다.

실용적인 튜닝 루프(Tuning Loop)는 구성 생성기(Configuration Generator)와 반복적인 학습 및 평가(Train-and-Evaluate) 과정으로 표현할 수 있다. 각 시행은 샘플링된 하이퍼파라미터를 이용하여 모델을 생성하고 고정된 벤치마크 프로토콜(Benchmark Protocol)에 따라 학습한 뒤 검증 지표와 계산 성능 통계를 측정한다. 최종적으로 검증 정확도, F1, ROC-AUC 또는 성능과 효율성을 결합한 기준과 같이 사전에 정의된 목적 함수(Objective)에 따라 최적 구성을 선택한다.

튜닝 결과는 단순히 가장 좋은 설정 하나만 남기는 것이 아니라 전체적으로 분석해야 한다. 모든 시행을 기록하면 파라미터 민감도(Parameter Sensitivity), 파라미터 간 상호작용, 불안정한 영역, 성능과 비용 사이의 절충 관계(Performance-Cost Tradeoff)를 분석할 수 있다. 예를 들어 학습률이 최적화 동작을 지배하는 반면, 은닉 차원은 특정 크기 이상에서 거의 성능 향상을 제공하지 않을 수 있다. 이러한 패턴은 아키텍처의 동작 특성을 이해하고 이후 실험을 더욱 효율적으로 설계하는 데 활용할 수 있다.

공정한 모델 비교(Fair Model Comparison)를 위해서는 비교 가능한 수준의 튜닝 노력(Tuning Effort)이 필요하다. 한 아키텍처에는 수백 번의 시행을 수행하면서 다른 모델은 기본 파라미터(Default Parameters)만 사용한다면 아키텍처의 품질과 무관한 인위적인 성능 차이가 발생할 수 있다. 모델마다 서로 다른 파라미터를 가지므로 탐색 공간 자체가 동일할 필요는 없지만 계산 예산(Computational Budget), 검증 기준, 중단 규칙(Stopping Rules), 최적화 절차는 합리적으로 동등해야 한다.

최종 튜닝 기록(Tuning Record)은 탐색 공간(Search Space), 샘플링 방법(Sampling Method), 시행 예산(Trial Budget), 랜덤 시드(Random Seeds), 검증 목적 함수(Validation Objective), 조기 종료 정책(Early-Stopping Policy), 전체 시행 기록(Trial History), 선택된 구성(Selected Configuration), 관련 계산 비용을 보존해야 한다. 이를 재현성 프레임워크(Reproducibility Framework)와 결합하면 후보 하이퍼파라미터에서 최종 벤치마크 모델까지 이어지는 추적 가능한 경로를 구축할 수 있으며, 그래프 딥러닝 실험의 마지막 단계인 신뢰할 수 있는 모델 비교(Model Comparison)를 위한 방법론적 기반을 제공한다.

##  

## 09.05. Model Comparison

![](images/image5.png){width="7.268055555555556in" height="7.268055555555556in"}

Model comparison in graph deep learning is the final experimental stage in which competing architectures are evaluated under a common benchmark protocol to determine their relative strengths, limitations, and practical suitability. A meaningful comparison extends beyond identifying the model with the highest score. It examines predictive quality, robustness, computational efficiency, scalability, architectural complexity, and consistency across datasets and experimental conditions.

Fair comparison begins by controlling the experimental environment. Models should use the same dataset version, preprocessing pipeline, training-validation-test split, evaluation metrics, and hardware configuration whenever these conditions can reasonably be shared. Model-specific requirements may differ, but unrelated experimental variables should remain fixed so that observed performance differences can be attributed primarily to the architectures or learning strategies being compared.

Graph neural networks should be compared against appropriate baselines representing different levels of complexity. A feature-only model can reveal how much predictive information exists without graph structure, while established architectures such as GCN, GAT, and GraphSAGE provide reference points for message-passing methods. More advanced architectures should then demonstrate measurable benefits over these baselines rather than merely outperforming an artificially weak reference model.

Predictive performance must be evaluated using metrics appropriate to the task. Node classification may emphasize accuracy, F1 score, or ROC-AUC, while link prediction can require average precision, Hits@K, or ranking metrics. Graph-level prediction may use classification or regression measures depending on the target. When class imbalance or asymmetric errors are important, relying on a single aggregate accuracy value can hide meaningful differences between competing models.

Model comparison should incorporate repeated experiments because stochastic training can produce substantial variation. Each architecture can be evaluated across multiple random seeds using the same seed set, after which results are summarized with statistics such as mean and standard deviation. A model whose average performance is slightly higher but highly unstable may be less attractive than one that consistently produces strong results across independent runs.

Statistical uncertainty should be considered when performance differences are small. A fraction of a percentage point between two models may not represent a meaningful improvement if run-to-run variation is of similar magnitude. Confidence intervals, paired comparisons, or suitable statistical tests can help determine whether observed differences are sufficiently consistent to support stronger conclusions rather than being consequences of random experimental variation.

Hyperparameter tuning must be integrated fairly into the comparison process. Each model should receive a reasonably comparable tuning budget and should be selected according to the same validation principle. Search spaces may differ because GCN, GAT, GraphSAGE, and graph transformers expose different architectural parameters, but the comparison should avoid extensively optimizing one model while leaving competitors near arbitrary default configurations.

Accuracy alone is insufficient when models are intended for real systems. Training time, inference latency, parameter count, peak GPU memory, CPU memory, throughput, preprocessing requirements, and storage cost provide additional dimensions of comparison. A model that improves accuracy slightly while requiring several times more memory or computation may be inappropriate for large-scale, real-time, edge, or resource-constrained graph applications.

Efficiency should therefore be interpreted as a performance-cost tradeoff rather than as an isolated measurement. Two models with similar predictive quality may differ substantially in computational requirements. Results can be examined in terms of accuracy per parameter, throughput at a target accuracy, latency under a fixed resource budget, or memory required for a specified graph size. Such analysis helps identify models that lie near a practical performance-efficiency frontier.

Scalability comparison evaluates how competing architectures behave as graphs become larger or denser. Increasing node count, edge count, neighborhood size, batch size, or sampling fan-out can expose memory and throughput limitations that are invisible on small citation datasets. Architectures that appear similar on Cora or PubMed may diverge significantly when evaluated on larger OGB benchmarks or application-specific graphs requiring neighborhood sampling and sparse computation.

Robustness provides another important comparison dimension. Models can be evaluated under feature noise, missing edges, perturbed topology, class imbalance, incomplete labels, or distribution shifts when these conditions are relevant to the target application. A graph model that achieves the best result on a clean benchmark but degrades sharply under moderate structural disturbance may be less useful than a slightly less accurate model with stable behavior.

Ablation results should accompany comparisons when a proposed model introduces several new components. Removing attention modules, residual connections, normalization, positional encodings, specialized aggregators, or auxiliary objectives can reveal how much each component contributes. Without such analysis, it is difficult to determine whether improvement comes from the central architectural idea, additional parameters, stronger regularization, or a more favorable training configuration.

Cross-dataset comparison provides evidence about generalization across graph structures. A useful evaluation may progress from small citation networks such as Cora, through larger datasets such as PubMed and standardized OGB benchmarks, to custom domain-specific datasets. A model that consistently performs well across graphs with different scales and structural characteristics provides stronger evidence of general usefulness than one optimized for a single benchmark.

Results should be presented in a form that makes tradeoffs visible rather than reducing every experiment to one ranking. Comparison tables can combine predictive metrics with parameter count, training time, inference latency, memory consumption, and variability across runs. Learning curves and scaling measurements can further show convergence behavior and computational growth, helping readers understand why one model may be preferable under different deployment constraints.

The final model should therefore be selected according to the objective of the application rather than automatically choosing the architecture with the highest benchmark score. Research experiments may prioritize predictive improvement, large-scale systems may emphasize scalability and throughput, and real-time applications may prioritize latency and memory efficiency. Model comparison becomes a multi-objective decision process in which performance, reliability, complexity, and resource requirements must be balanced.

A complete comparison record connects datasets, benchmark methodology, reproducibility controls, hyperparameter tuning, repeated runs, evaluation metrics, computational measurements, robustness tests, and final conclusions. Within the Graph Deep Learning structure, this closes the experimental workflow by transforming isolated model scores into evidence that can be reproduced, interpreted, and compared fairly, providing a defensible basis for selecting architectures for research and real-world deployment.

그래프 딥러닝(Graph Deep Learning)에서 모델 비교(Model Comparison)는 경쟁하는 아키텍처를 공통 벤치마크 프로토콜(Benchmark Protocol) 아래에서 평가하여 상대적인 강점, 한계, 실용적 적합성을 판단하는 최종 실험 단계이다. 의미 있는 비교는 가장 높은 점수를 얻은 모델을 찾는 것에 그치지 않는다. 예측 품질(Predictive Quality), 강건성(Robustness), 계산 효율성(Computational Efficiency), 확장성(Scalability), 아키텍처 복잡성(Architectural Complexity), 그리고 다양한 데이터셋과 실험 조건에서의 일관성을 함께 평가해야 한다.

공정한 비교(Fair Comparison)는 실험 환경(Experimental Environment)을 통제하는 것에서 시작한다. 가능한 경우 모델들은 동일한 데이터셋 버전(Dataset Version), 전처리 파이프라인(Preprocessing Pipeline), 학습·검증·테스트 분할(Training-Validation-Test Split), 평가 지표(Evaluation Metrics), 하드웨어 구성(Hardware Configuration)을 사용해야 한다. 모델별 요구사항은 다를 수 있지만 관련 없는 실험 변수는 고정하여 관찰된 성능 차이가 주로 비교 대상 아키텍처나 학습 전략에서 발생하도록 해야 한다.

그래프 신경망(Graph Neural Networks)은 서로 다른 복잡성 수준을 나타내는 적절한 베이스라인(Baselines)과 비교해야 한다. 특징만 사용하는 모델(Feature-Only Model)은 그래프 구조를 사용하지 않고도 어느 정도의 예측 정보가 존재하는지를 보여주며, GCN, GAT, GraphSAGE와 같은 기존 아키텍처는 메시지 패싱(Message Passing) 방법의 기준점을 제공한다. 이후 고급 아키텍처는 의도적으로 약하게 설정된 기준 모델이 아니라 이러한 적절한 베이스라인보다 측정 가능한 이점을 보여주어야 한다.

예측 성능(Predictive Performance)은 태스크에 적합한 지표를 사용하여 평가해야 한다. 노드 분류(Node Classification)는 정확도(Accuracy), F1 점수(F1 Score), ROC-AUC 등을 중요하게 사용할 수 있으며, 링크 예측(Link Prediction)은 평균 정밀도(Average Precision), Hits@K 또는 순위 기반 지표(Ranking Metrics)가 필요할 수 있다. 그래프 수준 예측(Graph-Level Prediction)은 목표에 따라 분류 또는 회귀 지표를 사용할 수 있다. 클래스 불균형(Class Imbalance)이나 비대칭 오류(Asymmetric Errors)가 중요한 경우 하나의 전체 정확도만으로는 모델 사이의 의미 있는 차이를 발견하기 어렵다.

모델 비교에는 확률적 학습(Stochastic Training)에서 발생하는 상당한 변동을 고려하기 위해 반복 실험(Repeated Experiments)이 포함되어야 한다. 각 아키텍처를 동일한 랜덤 시드 집합(Random Seed Set)을 사용하여 여러 번 평가한 후 평균(Mean)과 표준편차(Standard Deviation)와 같은 통계값으로 결과를 요약할 수 있다. 평균 성능은 약간 높지만 실행마다 결과가 크게 달라지는 모델보다 독립적인 여러 실행에서 지속적으로 우수한 결과를 제공하는 모델이 더 적합할 수 있다.

성능 차이가 작은 경우에는 통계적 불확실성(Statistical Uncertainty)을 고려해야 한다. 두 모델 사이의 성능 차이가 몇 분의 1 퍼센트에 불과하고 실행 간 변동도 비슷한 수준이라면 그 차이가 실제로 의미 있는 개선을 나타낸다고 보기 어렵다. 신뢰구간(Confidence Intervals), 대응 비교(Paired Comparisons), 적절한 통계 검정(Statistical Tests)을 사용하면 관찰된 차이가 무작위 실험 변동의 결과인지 아니면 더 강한 결론을 뒷받침할 정도로 일관된 차이인지 판단하는 데 도움이 된다.

하이퍼파라미터 튜닝(Hyperparameter Tuning)은 모델 비교 과정에 공정하게 통합되어야 한다. 각 모델에는 합리적으로 비교 가능한 튜닝 예산(Tuning Budget)을 제공하고 동일한 검증 원칙(Validation Principle)에 따라 모델을 선택해야 한다. GCN, GAT, GraphSAGE, 그래프 트랜스포머(Graph Transformer)는 서로 다른 아키텍처 파라미터를 가지므로 탐색 공간(Search Space)이 동일할 필요는 없지만, 특정 모델만 집중적으로 최적화하고 다른 모델은 임의의 기본 설정(Default Configuration)으로 평가하는 방식은 피해야 한다.

실제 시스템(Real Systems)에 적용할 모델을 평가할 때 정확도만으로는 충분하지 않다. 학습 시간(Training Time), 추론 지연시간(Inference Latency), 파라미터 수(Parameter Count), 최대 GPU 메모리(Peak GPU Memory), CPU 메모리, 처리량(Throughput), 전처리 요구사항(Preprocessing Requirements), 저장 비용(Storage Cost)도 중요한 비교 요소이다. 정확도가 조금 향상되더라도 메모리나 계산량이 몇 배 증가한다면 대규모, 실시간(Real-Time), 엣지(Edge), 자원 제약형(Resource-Constrained) 그래프 응용에는 적합하지 않을 수 있다.

따라서 효율성(Efficiency)은 독립적인 측정값이 아니라 성능과 비용의 절충 관계(Performance-Cost Tradeoff)로 해석해야 한다. 예측 품질이 비슷한 두 모델도 계산 자원 요구량에서는 큰 차이를 보일 수 있다. 파라미터당 정확도(Accuracy per Parameter), 목표 정확도에서의 처리량, 고정된 자원 예산(Resource Budget)에서의 지연시간, 특정 그래프 크기를 처리하기 위해 필요한 메모리 등의 관점에서 결과를 분석할 수 있다. 이러한 분석은 실제적인 성능-효율성 경계(Performance-Efficiency Frontier)에 가까운 모델을 식별하는 데 도움이 된다.

확장성 비교(Scalability Comparison)는 그래프가 더 크거나 밀집될 때 경쟁 아키텍처가 어떻게 동작하는지를 평가한다. 노드 수, 엣지 수, 이웃 크기(Neighborhood Size), 배치 크기(Batch Size), 샘플링 팬아웃(Sampling Fan-Out)을 증가시키면 소규모 인용 데이터셋에서는 나타나지 않았던 메모리와 처리량의 한계를 발견할 수 있다. Cora나 PubMed에서는 비슷하게 보이는 아키텍처도 더 큰 OGB 벤치마크나 이웃 샘플링 및 희소 연산(Sparse Computation)이 필요한 응용 그래프에서는 상당한 차이를 보일 수 있다.

강건성(Robustness)은 모델 비교에서 또 다른 중요한 평가 차원을 제공한다. 대상 응용에 관련성이 있다면 특징 잡음(Feature Noise), 누락된 엣지(Missing Edges), 교란된 토폴로지(Perturbed Topology), 클래스 불균형, 불완전한 레이블(Incomplete Labels), 분포 변화(Distribution Shift) 조건에서 모델을 평가할 수 있다. 깨끗한 벤치마크에서는 가장 높은 성능을 보이지만 작은 구조적 교란에도 성능이 급격히 감소하는 모델보다 약간 낮은 정확도를 가지더라도 안정적인 모델이 실제 환경에서는 더 유용할 수 있다.

제안된 모델이 여러 새로운 구성 요소를 도입한다면 절제 연구(Ablation Study)의 결과를 모델 비교와 함께 제시해야 한다. 어텐션 모듈(Attention Module), 잔차 연결(Residual Connection), 정규화(Normalization), 위치 인코딩(Positional Encoding), 특수 집계기(Specialized Aggregator), 보조 목적 함수(Auxiliary Objective)를 제거하면서 각각의 기여도를 평가할 수 있다. 이러한 분석이 없다면 성능 향상이 핵심 아키텍처 아이디어에서 발생했는지, 추가 파라미터나 더 강한 정규화 또는 유리한 학습 설정에서 발생했는지를 판단하기 어렵다.

교차 데이터셋 비교(Cross-Dataset Comparison)는 서로 다른 그래프 구조에서의 일반화(Generalization)에 대한 근거를 제공한다. 평가 과정은 Cora와 같은 소규모 인용 네트워크에서 시작하여 PubMed와 같은 더 큰 데이터셋, 표준화된 OGB 벤치마크, 그리고 사용자 정의 도메인 데이터셋(Custom Domain-Specific Dataset)으로 확장할 수 있다. 서로 다른 규모와 구조적 특성을 가진 그래프에서 지속적으로 좋은 성능을 보이는 모델은 하나의 벤치마크에 최적화된 모델보다 일반적인 활용 가능성에 대한 더 강력한 근거를 제공한다.

결과는 모든 실험을 하나의 순위로 단순화하기보다 모델 사이의 절충 관계가 명확하게 드러나는 형태로 제시해야 한다. 비교표(Comparison Table)는 예측 지표와 함께 파라미터 수, 학습 시간, 추론 지연시간, 메모리 소비량, 반복 실행에서의 변동성을 함께 표현할 수 있다. 학습 곡선(Learning Curves)과 확장성 측정(Scaling Measurements)을 추가하면 수렴 특성과 계산량 증가를 확인할 수 있어 서로 다른 배포 제약(Deployment Constraints)에서 어떤 모델이 더 적합한지를 이해하는 데 도움이 된다.

따라서 최종 모델(Final Model)은 가장 높은 벤치마크 점수를 기록한 아키텍처를 자동으로 선택하는 것이 아니라 실제 응용의 목적(Objective)에 따라 선택해야 한다. 연구 실험에서는 예측 성능 향상을 우선할 수 있고, 대규모 시스템은 확장성과 처리량을 중요하게 고려할 수 있으며, 실시간 응용은 지연시간과 메모리 효율성(Memory Efficiency)을 우선할 수 있다. 따라서 모델 비교는 성능, 신뢰성(Reliability), 복잡성, 자원 요구사항을 균형 있게 고려하는 다목적 의사결정 과정(Multi-Objective Decision Process)이 된다.

완전한 모델 비교 기록(Complete Comparison Record)은 데이터셋, 벤치마크 방법론(Benchmark Methodology), 재현성 통제(Reproducibility Controls), 하이퍼파라미터 튜닝, 반복 실행, 평가 지표, 계산 성능 측정, 강건성 시험(Robustness Tests), 최종 결론을 하나의 실험 체계로 연결한다. 그래프 딥러닝(Graph Deep Learning)의 전체 구조에서 이 단계는 개별적인 모델 점수를 재현하고 해석하며 공정하게 비교할 수 있는 실험적 증거로 전환함으로써 전체 실험 워크플로를 완성하고, 연구 및 실제 배포(Real-World Deployment)를 위한 아키텍처를 선택할 수 있는 타당한 근거를 제공한다.
