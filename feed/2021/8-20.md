
## 2021-8-20

### [[2107.14072] What Does TERRA-REF's High Resolution, Multi Sensor Plant Sensing Public Domain Data Offer the Computer Vision Community?](http://arxiv.org/abs/2107.14072)


  A core objective of the TERRA-REF project was to generate an open-access
reference dataset for the evaluation of sensing technologies to study plants
under field conditions. The TERRA-REF program deployed a suite of
high-resolution, cutting edge technology sensors on a gantry system with the
aim of scanning 1 hectare (10$^4$) at around 1 mm$^2$ spatial resolution
multiple times per week. The system contains co-located sensors including a
stereo-pair RGB camera, a thermal imager, a laser scanner to capture 3D
structure, and two hyperspectral cameras covering wavelengths of 300-2500nm.
This sensor data is provided alongside over sixty types of traditional plant
phenotype measurements that can be used to train new machine learning models.
Associated weather and environmental measurements, information about agronomic
management and experimental design, and the genomic sequences of hundreds of
plant varieties have been collected and are available alongside the sensor and
plant phenotype data.
Over the course of four years and ten growing seasons, the TERRA-REF system
generated over 1 PB of sensor data and almost 45 million files. The subset that
has been released to the public domain accounts for two seasons and about half
of the total data volume. This provides an unprecedented opportunity for
investigations far beyond the core biological scope of the project.
The focus of this paper is to provide the Computer Vision and Machine
Learning communities an overview of the available data and some potential
applications of this one of a kind data.

    

### [[2108.08523] Routing in Small Satellite Networks: A GNN-based Learning Approach](http://arxiv.org/abs/2108.08523)


  Small satellite networks (SSNs), which are constructed by large number of
small satellites in low earth orbits (LEO), are considered as promising ways to
provide ubiquitous Internet access. To handle stochastic Internet traffic,
on-board routing is necessary in SSNs. However, large-scale, high dynamic SSN
topologies and limited resources make on-board routing in SSNs face great
challenges. To address this issue, we turn to graph neural network (GNN), a
deep learning network inherently designed for graph data, motivated by the fact
that SSNs can be naturally modeled as graphs. By exploiting GNN's topology
extraction capabilities, we propose a GNN-based learning routing approach (GLR)
to achieve near-optimal on-board routing with low complexity. We design
high-order and low-order feature extractor and cross process to deal with high
dynamic topologies of SSNs, even those topologies that have never been seen in
training. Simulation results demonstrate that GLR results in a significant
reduction in routing computation cost while achieves near-optimal routing
performance in SSNs with different scales compared with typical existing
satellite routing algorithms.

    

### [[2108.08700] 5G System Security Analysis](http://arxiv.org/abs/2108.08700)


  Fifth generation mobile networks (5G) are currently being deployed by mobile
operators around the globe. 5G acts as an enabler for various use cases and
also improves the security and privacy over 4G and previous network
generations. However, as recent security research has revealed, the standard
still has security weaknesses that may be exploitable by attackers. In
addition, the migration from 4G to 5G systems is taking place by first
deploying 5G solutions in a non-standalone (NSA) manner where the first step of
the 5G deployment is restricted to the new radio aspects of 5G, while the
control of the user equipment is still based on 4G protocols, i.e. the core
network is still the legacy 4G evolved packet core (EPC) network. As a result,
many security vulnerabilities of 4G networks are still present in current 5G
deployments. This paper presents a systematic risk analysis of standalone and
non-standalone 5G networks. We first describe an overview of the 5G system
specification and the new security features of 5G compared to 4G. Then, we
define possible threats according to the STRIDE threat classification model and
derive a risk matrix based on the likelihood and impact of 12 threat scenarios
that affect the radio access and the network core. Finally, we discuss possible
mitigations and security controls. Our analysis is generic and does not account
for the specifics of particular 5G network vendors or operators. Further work
is required to understand the security vulnerabilities and risks of specific 5G
implementations and deployments.

    

### [[2108.08842] Communication-Efficient Federated Learning via Robust Distributed Mean Estimation](http://arxiv.org/abs/2108.08842)


  Federated learning commonly relies on algorithms such as distributed
(mini-batch) SGD, where multiple clients compute their gradients and send them
to a central coordinator for averaging and updating the model. To optimize the
transmission time and the scalability of the training process, clients often
use lossy compression to reduce the message sizes. DRIVE is a recent state of
the art algorithm that compresses gradients using one bit per coordinate (with
some lower-order overhead). In this technical report, we generalize DRIVE to
support any bandwidth constraint as well as extend it to support heterogeneous
client resources and make it robust to packet loss.

    

### [[2103.04092] RAN Slicing Performance Trade-offs: Timing versus Throughput Requirements](http://arxiv.org/abs/2103.04092)


  The coexistence of diverse services with heterogeneous requirements is a
fundamental feature of 5G. This necessitates efficient radio access network
(RAN) slicing, defined as sharing of the wireless resources among diverse
services while guaranteeing their respective throughput, timing, and/or
reliability requirements. In this paper, we investigate RAN slicing for an
uplink scenario in the form of multiple access schemes for two user types: (1)
broadband users with throughput requirements and (2) intermittently active
users with timing requirements, expressed as either latency-reliability (LR) or
Peak Age of Information (PAoI). Broadband users transmit data continuously,
hence, are allocated non-overlapping parts of the spectrum. We evaluate the
trade-offs between the achievable throughput of a broadband user and the timing
requirements of an intermittent user under Orthogonal Multiple Access (OMA) and
Non-Orthogonal Multiple Access (NOMA), considering capture. Our analysis shows
that NOMA, in combination with packet-level coding, is a superior strategy in
most cases for both LR and PAoI, achieving a similar LR with only slight 2%
decrease in throughput with respect to the upper bound in performance. However,
there are extreme cases where OMA achieves a slightly greater throughput than
NOMA at the expense of an increased PAoI.

    

### [[2104.00075] Ultra-Reliable Indoor Millimeter Wave Communications using Multiple Artificial Intelligence-Powered Intelligent Surfaces](http://arxiv.org/abs/2104.00075)


  In this paper, a novel framework for guaranteeing ultra-reliable millimeter
wave (mmW) communications using multiple artificial intelligence (AI)-enabled
reconfigurable intelligent surfaces (RISs) is proposed. The use of multiple
AI-powered RISs allows changing the propagation direction of the signals
transmitted from a mmW access point (AP) thereby improving coverage
particularly for non-line-of-sight (NLoS) areas. However, due to the
possibility of highly stochastic blockage over mmW links, designing an
intelligent controller to jointly optimize the mmW AP beam and RIS phase shifts
is a daunting task. In this regard, first, a parametric risk-sensitive episodic
return is proposed to maximize the expected bit rate and mitigate the risk of
mmW link blockage. Then, a closed-form approximation of the policy gradient of
the risk-sensitive episodic return is analytically derived. Next, the problem
of joint beamforming for mmW AP and phase shift control for mmW RISs is modeled
as an identical payoff stochastic game within a cooperative multi-agent
environment, in which the agents are the mmW AP and the RISs. Two centralized
and distributed controllers are proposed to control the policies of the mmW AP
and RISs. To directly find an optimal solution, the parametric functional-form
policies for these controllers are modeled using deep recurrent neural networks
(RNNs). Simulation results show that the error between policies of the optimal
and the RNN-based controllers is less than 1.5%. Moreover, the variance of the
achievable rates resulting from the deep RNN-based controllers is 60% less than
the variance of the risk-averse baseline.

    

### [[2104.07557] Decentralized Federated Learning for UAV Networks: Architecture, Challenges, and Opportunities](http://arxiv.org/abs/2104.07557)


  Unmanned aerial vehicles (UAVs), or say drones, are envisioned to support
extensive applications in next-generation wireless networks in both civil and
military fields. Empowering UAVs networks intelligence by artificial
intelligence (AI) especially machine learning (ML) techniques is inevitable and
appealing to enable the aforementioned applications. To solve the problems of
traditional cloud-centric ML for UAV networks such as privacy concern,
unacceptable latency, and resource burden, a distributed ML technique,
\textit(i.e.), federated learning (FL), has been recently proposed to enable
multiple UAVs to collaboratively train ML model without letting out raw data.
However, almost all existing FL paradigms are still centralized, \textit{i.e.},
a central entity is in charge of ML model aggregation and fusion over the whole
network, which could result in the issue of a single point of failure and are
inappropriate to UAV networks with both unreliable nodes and links. Thus
motivated, in this article, we propose a novel architecture called DFL-UN
(\underline{D}ecentralized \underline{F}ederated \underline{L}earning for
\underline{U}AV \underline{N}etworks), which enables FL within UAV networks
without a central entity. We also conduct a preliminary simulation study to
validate the feasibility and effectiveness of the DFL-UN architecture. Finally,
we discuss the main challenges and potential research directions in the DFL-UN.

    

### [[2105.15035] Machine Learning for Security in Vehicular Networks: A Comprehensive Survey](http://arxiv.org/abs/2105.15035)


  Machine Learning (ML) has emerged as an attractive and viable technique to
provide effective solutions for a wide range of application domains. An
important application domain is vehicular networks wherein ML-based approaches
are found to be very useful to address various problems. The use of wireless
communication between vehicular nodes and/or infrastructure makes it vulnerable
to different types of attacks. In this regard, ML and its variants are gaining
popularity to detect attacks and deal with different kinds of security issues
in vehicular communication. In this paper, we present a comprehensive survey of
ML-based techniques for different security issues in vehicular networks. We
first briefly introduce the basics of vehicular networks and different types of
communications. Apart from the traditional vehicular networks, we also consider
modern vehicular network architectures. We propose a taxonomy of security
attacks in vehicular networks and discuss various security challenges and
requirements. We classify the ML techniques developed in the literature
according to their use in vehicular network applications. We explain the
solution approaches and working principles of these ML techniques in addressing
various security challenges and provide insightful discussion. The limitations
and challenges in using ML-based methods in vehicular networks are discussed.
Finally, we present observations and lessons learned before we conclude our
work.

    

### [[2108.08292] GSVMA: A Genetic-Support Vector Machine-Anova method for CAD diagnosis based on Z-Alizadeh Sani dataset](http://arxiv.org/abs/2108.08292)


  Coronary heart disease (CAD) is one of the crucial reasons for cardiovascular
mortality in middle-aged people worldwide. The most typical tool is angiography
for diagnosing CAD. The challenges of CAD diagnosis using angiography are
costly and have side effects. One of the alternative solutions is the use of
machine learning-based patterns for CAD diagnosis. Hence, this paper provides a
new hybrid machine learning model called Genetic Support Vector Machine and
Analysis of Variance (GSVMA). The ANOVA is known as the kernel function for
SVM. The proposed model is performed based on the Z-Alizadeh Sani dataset. A
genetic optimization algorithm is used to select crucial features. In addition,
SVM with Anova, Linear SVM, and LibSVM with radial basis function methods were
applied to classify the dataset. As a result, the GSVMA hybrid method performs
better than other methods. This proposed method has the highest accuracy of
89.45% through a 10-fold cross-validation technique with 35 selected features
on the Z-Alizadeh Sani dataset. Therefore, the genetic optimization algorithm
is very effective for improving accuracy. The computer-aided GSVMA method can
be helped clinicians with CAD diagnosis.

    

### [[2108.08295] AIRCHITECT: Learning Custom Architecture Design and Mapping Space](http://arxiv.org/abs/2108.08295)


  Design space exploration is an important but costly step involved in the
design/deployment of custom architectures to squeeze out maximum possible
performance and energy efficiency. Conventionally, optimizations require
iterative sampling of the design space using simulation or heuristic tools. In
this paper we investigate the possibility of learning the optimization task
using machine learning and hence using the learnt model to predict optimal
parameters for the design and mapping space of custom architectures, bypassing
any exploration step. We use three case studies involving the optimal array
design, SRAM buffer sizing, mapping, and schedule determination for
systolic-array-based custom architecture design and mapping space. Within the
purview of these case studies, we show that it is possible to capture the
design space and train a model to "generalize" prediction the optimal design
and mapping parameters when queried with workload and design constraints. We
perform systematic design-aware and statistical analysis of the optimization
space for our case studies and highlight the patterns in the design space. We
formulate the architecture design and mapping as a machine learning problem
that allows us to leverage existing ML models for training and inference. We
design and train a custom network architecture called AIRCHITECT, which is
capable of learning the architecture design space with as high as 94.3% test
accuracy and predicting optimal configurations which achieve on average
(GeoMean) of 99.9% the best possible performance on a test dataset with $10^5$
GEMM workloads.

    

### [[2108.08296] Deep Contrastive Learning for Multi-View Network Embedding](http://arxiv.org/abs/2108.08296)


  Multi-view network embedding aims at projecting nodes in the network to
low-dimensional vectors, while preserving their multiple relations and
attribute information. Contrastive learning-based methods have preliminarily
shown promising performance in this task. However, most contrastive
learning-based methods mostly rely on high-quality graph embedding and explore
less on the relationships between different graph views. To deal with these
deficiencies, we design a novel node-to-node Contrastive learning framework for
Multi-view network Embedding (CREME), which mainly contains two contrastive
objectives: Multi-view fusion InfoMax and Inter-view InfoMin. The former
objective distills information from embeddings generated from different graph
views, while the latter distinguishes different graph views better to capture
the complementary information between them. Specifically, we first apply a view
encoder to generate each graph view representation and utilize a multi-view
aggregator to fuse these representations. Then, we unify the two contrastive
objectives into one learning objective for training. Extensive experiments on
three real-world datasets show that CREME outperforms existing methods
consistently.

    

### [[2108.08298] TFRD: A Benchmark Dataset for Research on Temperature Field Reconstruction of Heat-Source Systems](http://arxiv.org/abs/2108.08298)


  Heat management plays an important role in engineering. Temperature field
reconstruction of heat source systems (TFR-HSS) with limited monitoring
tensors, performs an essential role in heat management. However, prior methods
with common interpolations usually cannot provide accurate reconstruction. In
addition, there exists no public dataset for widely research of reconstruction
methods to further boost the field reconstruction in engineering. To overcome
this problem, this work construct a specific dataset, namely TFRD, for TFR-HSS
task with commonly used methods, including the interpolation methods and the
surrogate model based methods, as baselines to advance the research over
temperature field reconstruction. First, the TFR-HSS task is mathematically
modelled from real-world engineering problem and three types of numerically
modellings have been constructed to transform the problem into discrete mapping
forms. Besides, this work selects four typical reconstruction problem with
different heat source information and boundary conditions and generate the
standard samples as training and testing samples for further research. Finally,
a comprehensive review of the prior methods for TFR-HSS task as well as recent
widely used deep learning methods is given and we provide a performance
analysis of typical methods on TFRD, which can be served as the baseline
results on this benchmark.

    

### [[2108.08301] Identifying Illicit Drug Dealers on Instagram with Large-scale Multimodal Data Fusion](http://arxiv.org/abs/2108.08301)


  Illicit drug trafficking via social media sites such as Instagram has become
a severe problem, thus drawing a great deal of attention from law enforcement
and public health agencies. How to identify illicit drug dealers from social
media data has remained a technical challenge due to the following reasons. On
the one hand, the available data are limited because of privacy concerns with
crawling social media sites; on the other hand, the diversity of drug dealing
patterns makes it difficult to reliably distinguish drug dealers from common
drug users. Unlike existing methods that focus on posting-based detection, we
propose to tackle the problem of illicit drug dealer identification by
constructing a large-scale multimodal dataset named Identifying Drug Dealers on
Instagram (IDDIG). Totally nearly 4,000 user accounts, of which over 1,400 are
drug dealers, have been collected from Instagram with multiple data sources
including post comments, post images, homepage bio, and homepage images. We
then design a quadruple-based multimodal fusion method to combine the multiple
data sources associated with each user account for drug dealer identification.
Experimental results on the constructed IDDIG dataset demonstrate the
effectiveness of the proposed method in identifying drug dealers (almost 95%
accuracy). Moreover, we have developed a hashtag-based community detection
technique for discovering evolving patterns, especially those related to
geography and drug types.

    

### [[2108.08305] Temporal Kernel Consistency for Blind Video Super-Resolution](http://arxiv.org/abs/2108.08305)


  Deep learning-based blind super-resolution (SR) methods have recently
achieved unprecedented performance in upscaling frames with unknown
degradation. These models are able to accurately estimate the unknown
downscaling kernel from a given low-resolution (LR) image in order to leverage
the kernel during restoration. Although these approaches have largely been
successful, they are predominantly image-based and therefore do not exploit the
temporal properties of the kernels across multiple video frames. In this paper,
we investigated the temporal properties of the kernels and highlighted its
importance in the task of blind video super-resolution. Specifically, we
measured the kernel temporal consistency of real-world videos and illustrated
how the estimated kernels might change per frame in videos of varying
dynamicity of the scene and its objects. With this new insight, we revisited
previous popular video SR approaches, and showed that previous assumptions of
using a fixed kernel throughout the restoration process can lead to visual
artifacts when upscaling real-world videos. In order to counteract this, we
tailored existing single-image and video SR techniques to leverage kernel
consistency during both kernel estimation and video upscaling processes.
Extensive experiments on synthetic and real-world videos show substantial
restoration gains quantitatively and qualitatively, achieving the new
state-of-the-art in blind video SR and underlining the potential of exploiting
kernel temporal consistency.

    

### [[2108.08307] Multivariate and Propagation Graph Attention Network for Spatial-Temporal Prediction with Outdoor Cellular Traffic](http://arxiv.org/abs/2108.08307)


  Spatial-temporal prediction is a critical problem for intelligent
transportation, which is helpful for tasks such as traffic control and accident
prevention. Previous studies rely on large-scale traffic data collected from
sensors. However, it is unlikely to deploy sensors in all regions due to the
device and maintenance costs. This paper addresses the problem via outdoor
cellular traffic distilled from over two billion records per day in a telecom
company, because outdoor cellular traffic induced by user mobility is highly
related to transportation traffic. We study road intersections in urban and aim
to predict future outdoor cellular traffic of all intersections given historic
outdoor cellular traffic. Furthermore, We propose a new model for multivariate
spatial-temporal prediction, mainly consisting of two extending graph attention
networks (GAT). First GAT is used to explore correlations among multivariate
cellular traffic. Another GAT leverages the attention mechanism into graph
propagation to increase the efficiency of capturing spatial dependency.
Experiments show that the proposed model significantly outperforms the
state-of-the-art methods on our dataset.

    

### [[2108.08350] Data-driven Modeling for Distribution Grids Under Partial Observability](http://arxiv.org/abs/2108.08350)


  Accurately modeling power distribution grids is crucial for designing
effective monitoring and decision making algorithms. This paper addresses the
partial observability issue of data-driven distribution modeling in order to
improve the accuracy of line parameter estimation. Inspired by the sparse
changes in residential loads, we advocate to regularize the group sparsity of
the unobservable injections in a bi-linear estimation problem. The alternating
minimization scheme of guaranteed convergence is proposed to take advantage of
convex subproblems with efficient solutions. Numerical results using real-world
load data on the single-phase equivalent of the IEEE 123-bus test case have
demonstrated the accuracy improvements of the proposed solution over existing
work for both parameter estimation and voltage modeling.

    

### [[2108.08368] Computing Steiner Trees using Graph Neural Networks](http://arxiv.org/abs/2108.08368)


  Graph neural networks have been successful in many learning problems and
real-world applications. A recent line of research explores the power of graph
neural networks to solve combinatorial and graph algorithmic problems such as
subgraph isomorphism, detecting cliques, and the traveling salesman problem.
However, many NP-complete problems are as of yet unexplored using this method.
In this paper, we tackle the Steiner Tree Problem. We employ four learning
frameworks to compute low cost Steiner trees: feed-forward neural networks,
graph neural networks, graph convolutional networks, and a graph attention
model. We use these frameworks in two fundamentally different ways: 1) to train
the models to learn the actual Steiner tree nodes, 2) to train the model to
learn good Steiner point candidates to be connected to the constructed tree
using a shortest path in a greedy fashion. We illustrate the robustness of our
heuristics on several random graph generation models as well as the SteinLib
data library. Our finding suggests that the out-of-the-box application of GNN
methods does worse than the classic 2-approximation method. However, when
combined with a greedy shortest path construction, it even does slightly better
than the 2-approximation algorithm. This result sheds light on the fundamental
capabilities and limitations of graph learning techniques on classical
NP-complete problems.

    

### [[2108.08375] Contributions of Transformer Attention Heads in Multi- and Cross-lingual Tasks](http://arxiv.org/abs/2108.08375)


  This paper studies the relative importance of attention heads in
Transformer-based models to aid their interpretability in cross-lingual and
multi-lingual tasks. Prior research has found that only a few attention heads
are important in each mono-lingual Natural Language Processing (NLP) task and
pruning the remaining heads leads to comparable or improved performance of the
model. However, the impact of pruning attention heads is not yet clear in
cross-lingual and multi-lingual tasks. Through extensive experiments, we show
that (1) pruning a number of attention heads in a multi-lingual
Transformer-based model has, in general, positive effects on its performance in
cross-lingual and multi-lingual tasks and (2) the attention heads to be pruned
can be ranked using gradients and identified with a few trial experiments. Our
experiments focus on sequence labeling tasks, with potential applicability on
other cross-lingual and multi-lingual tasks. For comprehensiveness, we examine
two pre-trained multi-lingual models, namely multi-lingual BERT (mBERT) and
XLM-R, on three tasks across 9 languages each. We also discuss the validity of
our findings and their extensibility to truly resource-scarce languages and
other task settings.

    

### [[2108.08394] Learning to Detect: A Data-driven Approach for Network Intrusion Detection](http://arxiv.org/abs/2108.08394)


  With massive data being generated daily and the ever-increasing
interconnectivity of the world's Internet infrastructures, a machine learning
based intrusion detection system (IDS) has become a vital component to protect
our economic and national security. In this paper, we perform a comprehensive
study on NSL-KDD, a network traffic dataset, by visualizing patterns and
employing different learning-based models to detect cyber attacks. Unlike
previous shallow learning and deep learning models that use the single learning
model approach for intrusion detection, we adopt a hierarchy strategy, in which
the intrusion and normal behavior are classified firstly, and then the specific
types of attacks are classified. We demonstrate the advantage of the
unsupervised representation learning model in binary intrusion detection tasks.
Besides, we alleviate the data imbalance problem with SVM-SMOTE oversampling
technique in 4-class classification and further demonstrate the effectiveness
and the drawback of the oversampling mechanism with a deep neural network as a
base model.

    

### [[2108.08404] Federated Variational Learning for Anomaly Detection in Multivariate Time Series](http://arxiv.org/abs/2108.08404)


  Anomaly detection has been a challenging task given high-dimensional
multivariate time series data generated by networked sensors and actuators in
Cyber-Physical Systems (CPS). Besides the highly nonlinear, complex, and
dynamic natures of such time series, the lack of labeled data impedes data
exploitation in a supervised manner and thus prevents an accurate detection of
abnormal phenomenons. On the other hand, the collected data at the edge of the
network is often privacy sensitive and large in quantity, which may hinder the
centralized training at the main server. To tackle these issues, we propose an
unsupervised time series anomaly detection framework in a federated fashion to
continuously monitor the behaviors of interconnected devices within a network
and alerts for abnormal incidents so that countermeasures can be taken before
undesired consequences occur. To be specific, we leave the training data
distributed at the edge to learn a shared Variational Autoencoder (VAE) based
on Convolutional Gated Recurrent Unit (ConvGRU) model, which jointly captures
feature and temporal dependencies in the multivariate time series data for
representation learning and downstream anomaly detection tasks. Experiments on
three real-world networked sensor datasets illustrate the advantage of our
approach over other state-of-the-art models. We also conduct extensive
experiments to demonstrate the effectiveness of our detection framework under
non-federated and federated settings in terms of overall performance and
detection latency.

    

### [[2108.08411] FeelsGoodMan: Inferring Semantics of Twitch Neologisms](http://arxiv.org/abs/2108.08411)


  Twitch chats pose a unique problem in natural language understanding due to a
large presence of neologisms, specifically emotes. There are a total of 8.06
million emotes, over 400k of which were used in the week studied. There is
virtually no information on the meaning or sentiment of emotes, and with a
constant influx of new emotes and drift in their frequencies, it becomes
impossible to maintain an updated manually-labeled dataset. Our paper makes a
two fold contribution. First we establish a new baseline for sentiment analysis
on Twitch data, outperforming the previous supervised benchmark by 7.9% points.
Secondly, we introduce a simple but powerful unsupervised framework based on
word embeddings and k-NN to enrich existing models with out-of-vocabulary
knowledge. This framework allows us to auto-generate a pseudo-dictionary of
emotes and we show that we can nearly match the supervised benchmark above even
when injecting such emote knowledge into sentiment classifiers trained on
extraneous datasets such as movie reviews or Twitter.

    

### [[2108.08421] Exploiting Multi-Object Relationships for Detecting Adversarial Attacks in Complex Scenes](http://arxiv.org/abs/2108.08421)


  Vision systems that deploy Deep Neural Networks (DNNs) are known to be
vulnerable to adversarial examples. Recent research has shown that checking the
intrinsic consistencies in the input data is a promising way to detect
adversarial attacks (e.g., by checking the object co-occurrence relationships
in complex scenes). However, existing approaches are tied to specific models
and do not offer generalizability. Motivated by the observation that language
descriptions of natural scene images have already captured the object
co-occurrence relationships that can be learned by a language model, we develop
a novel approach to perform context consistency checks using such language
models. The distinguishing aspect of our approach is that it is independent of
the deployed object detector and yet offers very high accuracy in terms of
detecting adversarial examples in practical scenes with multiple objects.

    

### [[2108.08426] Self-Supervised Video Representation Learning with Meta-Contrastive Network](http://arxiv.org/abs/2108.08426)


  Self-supervised learning has been successfully applied to pre-train video
representations, which aims at efficient adaptation from pre-training domain to
downstream tasks. Existing approaches merely leverage contrastive loss to learn
instance-level discrimination. However, lack of category information will lead
to hard-positive problem that constrains the generalization ability of this
kind of methods. We find that the multi-task process of meta learning can
provide a solution to this problem. In this paper, we propose a
Meta-Contrastive Network (MCN), which combines the contrastive learning and
meta learning, to enhance the learning ability of existing self-supervised
approaches. Our method contains two training stages based on model-agnostic
meta learning (MAML), each of which consists of a contrastive branch and a meta
branch. Extensive evaluations demonstrate the effectiveness of our method. For
two downstream tasks, i.e., video action recognition and video retrieval, MCN
outperforms state-of-the-art approaches on UCF101 and HMDB51 datasets. To be
more specific, with R(2+1)D backbone, MCN achieves Top-1 accuracies of 84.8%
and 54.5% for video action recognition, as well as 52.5% and 23.7% for video
retrieval.

    

### [[2108.08435] Fair and Consistent Federated Learning](http://arxiv.org/abs/2108.08435)


  Federated learning (FL) has gain growing interests for its capability of
learning from distributed data sources collectively without the need of
accessing the raw data samples across different sources. So far FL research has
mostly focused on improving the performance, how the algorithmic disparity will
be impacted for the model learned from FL and the impact of algorithmic
disparity on the utility inconsistency are largely unexplored. In this paper,
we propose an FL framework to jointly consider performance consistency and
algorithmic fairness across different local clients (data sources). We derive
our framework from a constrained multi-objective optimization perspective, in
which we learn a model satisfying fairness constraints on all clients with
consistent performance. Specifically, we treat the algorithm prediction loss at
each local client as an objective and maximize the worst-performing client with
fairness constraints through optimizing a surrogate maximum function with all
objectives involved. A gradient-based procedure is employed to achieve the
Pareto optimality of this optimization problem. Theoretical analysis is
provided to prove that our method can converge to a Pareto solution that
achieves the min-max performance with fairness constraints on all clients.
Comprehensive experiments on synthetic and real-world datasets demonstrate the
superiority that our approach over baselines and its effectiveness in achieving
both fairness and consistency across all local clients.

    

### [[2108.08448] Prior Is All You Need to Improve the Robustness and Safety for the First Time Deployment of Meta RL](http://arxiv.org/abs/2108.08448)


  The field of Meta Reinforcement Learning (Meta-RL) has seen substantial
advancements recently. In particular, off-policy methods were developed to
improve the data efficiency of Meta-RL techniques. \textit{Probabilistic
embeddings for actor-critic RL} (PEARL) is currently one of the leading
approaches for multi-MDP adaptation problems. A major drawback of many existing
Meta-RL methods, including PEARL, is that they do not explicitly consider the
safety of the prior policy when it is exposed to a new task for the very first
time. This is very important for some real-world applications, including field
robots and Autonomous Vehicles (AVs). In this paper, we develop the PEARL PLUS
(PEARL$^+$) algorithm, which optimizes the policy for both prior safety and
posterior adaptation. Building on top of PEARL, our proposed PEARL$^+$
algorithm introduces a prior regularization term in the reward function and a
new Q-network for recovering the state-action value with prior context
assumption, to improve the robustness and safety of the trained network
exposing to a new task for the first time. The performance of the PEARL$^+$
method is demonstrated by solving three safety-critical decision-making
problems related to robots and AVs, including two MuJoCo benchmark problems.
From the simulation experiments, we show that the safety of the prior policy is
significantly improved compared to that of the original PEARL method.

    

### [[2108.08454] Improving Human Decision-Making with Machine Learning](http://arxiv.org/abs/2108.08454)


  A key aspect of human intelligence is their ability to convey their knowledge
to others in succinct forms. However, despite their predictive power, current
machine learning models are largely blackboxes, making it difficult for humans
to extract useful insights. Focusing on sequential decision-making, we design a
novel machine learning algorithm that conveys its insights to humans in the
form of interpretable "tips". Our algorithm selects the tip that best bridges
the gap in performance between human users and the optimal policy. We evaluate
our approach through a series of randomized controlled user studies where
participants manage a virtual kitchen. Our experiments show that the tips
generated by our algorithm can significantly improve human performance relative
to intuitive baselines. In addition, we discuss a number of empirical insights
that can help inform the design of algorithms intended for human-AI
collaboration. For instance, we find evidence that participants do not simply
blindly follow our tips; instead, they combine them with their own experience
to discover additional strategies for improving performance.

    

### [[2108.08456] Blockchain Phishing Scam Detection via Multi-channel Graph Classification](http://arxiv.org/abs/2108.08456)


  With the popularity of blockchain technology, the financial security issues
of blockchain transaction networks have become increasingly serious. Phishing
scam detection methods will protect possible victims and build a healthier
blockchain ecosystem. Usually, the existing works define phishing scam
detection as a node classification task by learning the potential features of
users through graph embedding methods such as random walk or graph neural
network (GNN). However, these detection methods are suffered from high
complexity due to the large scale of the blockchain transaction network,
ignoring temporal information of the transaction. Addressing this problem, we
defined the transaction pattern graphs for users and transformed the phishing
scam detection into a graph classification task. To extract richer information
from the input graph, we proposed a multi-channel graph classification model
(MCGC) with multiple feature extraction channels for GNN. The transaction
pattern graphs and MCGC are more able to detect potential phishing scammers by
extracting the transaction pattern features of the target users. Extensive
experiments on seven benchmark and Ethereum datasets demonstrate that the
proposed MCGC can not only achieve state-of-the-art performance in the graph
classification task but also achieve effective phishing scam detection based on
the target users' transaction pattern graphs.

    

### [[2108.08467] Medical Image Segmentation using 3D Convolutional Neural Networks: A Review](http://arxiv.org/abs/2108.08467)


  Computer-aided medical image analysis plays a significant role in assisting
medical practitioners for expert clinical diagnosis and deciding the optimal
treatment plan. At present, convolutional neural networks (CNN) are the
preferred choice for medical image analysis. In addition, with the rapid
advancements in three-dimensional (3D) imaging systems and the availability of
excellent hardware and software support to process large volumes of data, 3D
deep learning methods are gaining popularity in medical image analysis. Here,
we present an extensive review of the recently evolved 3D deep learning methods
in medical image segmentation. Furthermore, the research gaps and future
directions in 3D medical image segmentation are discussed.

    

### [[2108.08468] QUEACO: Borrowing Treasures from Weakly-labeled Behavior Data for Query Attribute Value Extraction](http://arxiv.org/abs/2108.08468)


  We study the problem of query attribute value extraction, which aims to
identify named entities from user queries as diverse surface form attribute
values and afterward transform them into formally canonical forms. Such a
problem consists of two phases: {named entity recognition (NER)} and {attribute
value normalization (AVN)}. However, existing works only focus on the NER phase
but neglect equally important AVN. To bridge this gap, this paper proposes a
unified query attribute value extraction system in e-commerce search named
QUEACO, which involves both two phases. Moreover, by leveraging large-scale
weakly-labeled behavior data, we further improve the extraction performance
with less supervision cost. Specifically, for the NER phase, QUEACO adopts a
novel teacher-student network, where a teacher network that is trained on the
strongly-labeled data generates pseudo-labels to refine the weakly-labeled data
for training a student network. Meanwhile, the teacher network can be
dynamically adapted by the feedback of the student's performance on
strongly-labeled data to maximally denoise the noisy supervisions from the weak
labels. For the AVN phase, we also leverage the weakly-labeled
query-to-attribute behavior data to normalize surface form attribute values
from queries into canonical forms from products. Extensive experiments on a
real-world large-scale E-commerce dataset demonstrate the effectiveness of
QUEACO.

    

### [[2108.08473] Classification of Diabetic Retinopathy Severity in Fundus Images with DenseNet121 and ResNet50](http://arxiv.org/abs/2108.08473)


  In this work, deep learning algorithms are used to classify fundus images in
terms of diabetic retinopathy severity. Six different combinations of two model
architectures, the Dense Convolutional Network-121 and the Residual Neural
Network-50 and three image types, RGB, Green, and High Contrast, were tested to
find the highest performing combination. We achieved an average validation loss
of 0.17 and a max validation accuracy of 85 percent. By testing out multiple
combinations, certain combinations of parameters performed better than others,
though minimal variance was found overall. Green filtration was shown to
perform the poorest, while amplified contrast appeared to have a negligible
effect in comparison to RGB analysis. ResNet50 proved to be less of a robust
model as opposed to DenseNet121.

    

### [[2108.08474] Trends in Neural Architecture Search: Towards the Acceleration of Search](http://arxiv.org/abs/2108.08474)


  In modern deep learning research, finding optimal (or near optimal) neural
network models is one of major research directions and it is widely studied in
many applications. In this paper, the main research trends of neural
architecture search (NAS) are classified as neuro-evolutionary algorithms,
reinforcement learning based algorithms, and one-shot architecture search
approaches. Furthermore, each research trend is introduced and finally all the
major three trends are compared. Lastly, the future research directions of NAS
research trends are discussed.

    

### [[2108.08477] Image2Lego: Customized LEGO Set Generation from Images](http://arxiv.org/abs/2108.08477)


  Although LEGO sets have entertained generations of children and adults, the
challenge of designing customized builds matching the complexity of real-world
or imagined scenes remains too great for the average enthusiast. In order to
make this feat possible, we implement a system that generates a LEGO brick
model from 2D images. We design a novel solution to this problem that uses an
octree-structured autoencoder trained on 3D voxelized models to obtain a
feasible latent representation for model reconstruction, and a separate network
trained to predict this latent representation from 2D images. LEGO models are
obtained by algorithmic conversion of the 3D voxelized model to bricks. We
demonstrate first-of-its-kind conversion of photographs to 3D LEGO models. An
octree architecture enables the flexibility to produce multiple resolutions to
best fit a user's creative vision or design needs. In order to demonstrate the
broad applicability of our system, we generate step-by-step building
instructions and animations for LEGO models of objects and human faces.
Finally, we test these automatically generated LEGO sets by constructing
physical builds using real LEGO bricks.

    

### [[2108.08481] Neural Operator: Learning Maps Between Function Spaces](http://arxiv.org/abs/2108.08481)


  The classical development of neural networks has primarily focused on
learning mappings between finite dimensional Euclidean spaces or finite sets.
We propose a generalization of neural networks tailored to learn operators
mapping between infinite dimensional function spaces. We formulate the
approximation of operators by composition of a class of linear integral
operators and nonlinear activation functions, so that the composed operator can
approximate complex nonlinear operators. Furthermore, we introduce four classes
of operator parameterizations: graph-based operators, low-rank operators,
multipole graph-based operators, and Fourier operators and describe efficient
algorithms for computing with each one. The proposed neural operators are
resolution-invariant: they share the same network parameters between different
discretizations of the underlying function spaces and can be used for zero-shot
super-resolutions. Numerically, the proposed models show superior performance
compared to existing machine learning based methodologies on Burgers' equation,
Darcy flow, and the Navier-Stokes equation, while being several order of
magnitude faster compared to conventional PDE solvers.

    

### [[2108.08483] A Multi-input Multi-output Transformer-based Hybrid Neural Network for Multi-class Privacy Disclosure Detection](http://arxiv.org/abs/2108.08483)


  The concern regarding users' data privacy has risen to its highest level due
to the massive increase in communication platforms, social networking sites,
and greater users' participation in online public discourse. An increasing
number of people exchange private information via emails, text messages, and
social media without being aware of the risks and implications. Researchers in
the field of Natural Language Processing (NLP) have concentrated on creating
tools and strategies to identify, categorize, and sanitize private information
in text data since a substantial amount of data is exchanged in textual form.
However, most of the detection methods solely rely on the existence of
pre-identified keywords in the text and disregard the inference of the
underlying meaning of the utterance in a specific context. Hence, in some
situations, these tools and algorithms fail to detect disclosure, or the
produced results are miss-classified. In this paper, we propose a multi-input,
multi-output hybrid neural network which utilizes transfer-learning,
linguistics, and metadata to learn the hidden patterns. Our goal is to better
classify disclosure/non-disclosure content in terms of the context of
situation. We trained and evaluated our model on a human-annotated ground truth
dataset, containing a total of 5,400 tweets. The results show that the proposed
model was able to identify privacy disclosure through tweets with an accuracy
of 77.4% while classifying the information type of those tweets with an
impressive accuracy of 99%, by jointly learning for two separate tasks.

    

### [[2108.08485] Language Model Augmented Relevance Score](http://arxiv.org/abs/2108.08485)


  Although automated metrics are commonly used to evaluate NLG systems, they
often correlate poorly with human judgements. Newer metrics such as BERTScore
have addressed many weaknesses in prior metrics such as BLEU and ROUGE, which
rely on n-gram matching. These newer methods, however, are still limited in
that they do not consider the generation context, so they cannot properly
reward generated text that is correct but deviates from the given reference.
In this paper, we propose Language Model Augmented Relevance Score (MARS), a
new context-aware metric for NLG evaluation. MARS leverages off-the-shelf
language models, guided by reinforcement learning, to create augmented
references that consider both the generation context and available human
references, which are then used as additional references to score generated
text. Compared with seven existing metrics in three common NLG tasks, MARS not
only achieves higher correlation with human reference judgements, but also
differentiates well-formed candidates from adversarial samples to a larger
degree.

    

### [[2108.08487] Amplitude-Phase Recombination: Rethinking Robustness of Convolutional Neural Networks in Frequency Domain](http://arxiv.org/abs/2108.08487)


  Recently, the generalization behavior of Convolutional Neural Networks (CNN)
is gradually transparent through explanation techniques with the frequency
components decomposition. However, the importance of the phase spectrum of the
image for a robust vision system is still ignored. In this paper, we notice
that the CNN tends to converge at the local optimum which is closely related to
the high-frequency components of the training images, while the amplitude
spectrum is easily disturbed such as noises or common corruptions. In contrast,
more empirical studies found that humans rely on more phase components to
achieve robust recognition. This observation leads to more explanations of the
CNN's generalization behaviors in both robustness to common perturbations and
out-of-distribution detection, and motivates a new perspective on data
augmentation designed by re-combing the phase spectrum of the current image and
the amplitude spectrum of the distracter image. That is, the generated samples
force the CNN to pay more attention to the structured information from phase
components and keep robust to the variation of the amplitude. Experiments on
several image datasets indicate that the proposed method achieves
state-of-the-art performances on multiple generalizations and calibration
tasks, including adaptability for common corruptions and surface variations,
out-of-distribution detection, and adversarial attack.

    

### [[2108.08500] Inverse design optimization framework via a two-step deep learning approach: application to a wind turbine airfoil](http://arxiv.org/abs/2108.08500)


  Though inverse approach is computationally efficient in aerodynamic design as
the desired target performance distribution is specified, it has some
significant limitations that prevent full efficiency from being achieved.
First, the iterative procedure should be repeated whenever the specified target
distribution changes. Target distribution optimization can be performed to
clarify the ambiguity in specifying this distribution, but several additional
problems arise in this process such as loss of the representation capacity due
to parameterization of the distribution, excessive constraints for a realistic
distribution, inaccuracy of quantities of interest due to theoretical/empirical
predictions, and the impossibility of explicitly imposing geometric
constraints. To deal with these issues, a novel inverse design optimization
framework with a two-step deep learning approach is proposed. A variational
autoencoder and multi-layer perceptron are used to generate a realistic target
distribution and predict the quantities of interest and shape parameters from
the generated distribution, respectively. Then, target distribution
optimization is performed as the inverse design optimization. The proposed
framework applies active learning and transfer learning techniques to improve
accuracy and efficiency. Finally, the framework is validated through
aerodynamic shape optimizations of the airfoil of a wind turbine blade, where
inverse design is actively being applied. The results of the optimizations show
that this framework is sufficiently accurate, efficient, and flexible to be
applied to other inverse design engineering applications.

    

### [[2108.08504] Understanding and Mitigating Annotation Bias in Facial Expression Recognition](http://arxiv.org/abs/2108.08504)


  The performance of a computer vision model depends on the size and quality of
its training data. Recent studies have unveiled previously-unknown composition
biases in common image datasets which then lead to skewed model outputs, and
have proposed methods to mitigate these biases. However, most existing works
assume that human-generated annotations can be considered gold-standard and
unbiased. In this paper, we reveal that this assumption can be problematic, and
that special care should be taken to prevent models from learning such
annotation biases. We focus on facial expression recognition and compare the
label biases between lab-controlled and in-the-wild datasets. We demonstrate
that many expression datasets contain significant annotation biases between
genders, especially when it comes to the happy and angry expressions, and that
traditional methods cannot fully mitigate such biases in trained models. To
remove expression annotation bias, we propose an AU-Calibrated Facial
Expression Recognition (AUC-FER) framework that utilizes facial action units
(AUs) and incorporates the triplet loss into the objective function.
Experimental results suggest that the proposed method is more effective in
removing expression annotation bias than existing techniques.

    

### [[2108.08536] A Unified Objective for Novel Class Discovery](http://arxiv.org/abs/2108.08536)


  In this paper, we study the problem of Novel Class Discovery (NCD). NCD aims
at inferring novel object categories in an unlabeled set by leveraging from
prior knowledge of a labeled set containing different, but related classes.
Existing approaches tackle this problem by considering multiple objective
functions, usually involving specialized loss terms for the labeled and the
unlabeled samples respectively, and often requiring auxiliary regularization
terms. In this paper, we depart from this traditional scheme and introduce a
UNified Objective function (UNO) for discovering novel classes, with the
explicit purpose of favoring synergy between supervised and unsupervised
learning. Using a multi-view self-labeling strategy, we generate pseudo-labels
that can be treated homogeneously with ground truth labels. This leads to a
single classification objective operating on both known and unknown classes.
Despite its simplicity, UNO outperforms the state of the art by a significant
margin on several benchmarks (~+10% on CIFAR-100 and +8% on ImageNet). The
project page is available at: \url{this https URL}.

    

### [[2108.08542] Learning System Parameters from Turing Patterns](http://arxiv.org/abs/2108.08542)


  The Turing mechanism describes the emergence of spatial patterns due to
spontaneous symmetry breaking in reaction-diffusion processes and underlies
many developmental processes. Identifying Turing mechanisms in biological
systems defines a challenging problem. This paper introduces an approach to the
prediction of Turing parameter values from observed Turing patterns. The
parameter values correspond to a parametrized system of reaction-diffusion
equations that generate Turing patterns as steady state. The Gierer-Meinhardt
model with four parameters is chosen as a case study. A novel invariant pattern
representation based on resistance distance histograms is employed, along with
Wasserstein kernels, in order to cope with the highly variable arrangement of
local pattern structure that depends on the initial conditions which are
assumed to be unknown. This enables to compute physically plausible distances
between patterns, to compute clusters of patterns and, above all, model
parameter prediction: for small training sets, classical state-of-the-art
methods including operator-valued kernels outperform neural networks that are
applied to raw pattern data, whereas for large training sets the latter are
more accurate. Excellent predictions are obtained for single parameter values
and reasonably accurate results for jointly predicting all parameter values.

    

### [[2108.08557] DECA: Deep viewpoint-Equivariant human pose estimation using Capsule Autoencoders](http://arxiv.org/abs/2108.08557)


  Human Pose Estimation (HPE) aims at retrieving the 3D position of human
joints from images or videos. We show that current 3D HPE methods suffer a lack
of viewpoint equivariance, namely they tend to fail or perform poorly when
dealing with viewpoints unseen at training time. Deep learning methods often
rely on either scale-invariant, translation-invariant, or rotation-invariant
operations, such as max-pooling. However, the adoption of such procedures does
not necessarily improve viewpoint generalization, rather leading to more
data-dependent methods. To tackle this issue, we propose a novel capsule
autoencoder network with fast Variational Bayes capsule routing, named DECA. By
modeling each joint as a capsule entity, combined with the routing algorithm,
our approach can preserve the joints' hierarchical and geometrical structure in
the feature space, independently from the viewpoint. By achieving viewpoint
equivariance, we drastically reduce the network data dependency at training
time, resulting in an improved ability to generalize for unseen viewpoints. In
the experimental validation, we outperform other methods on depth images from
both seen and unseen viewpoints, both top-view, and front-view. In the RGB
domain, the same network gives state-of-the-art results on the challenging
viewpoint transfer task, also establishing a new framework for top-view HPE.
The code can be found at this https URL.

    

### [[2108.08560] Pruning in the Face of Adversaries](http://arxiv.org/abs/2108.08560)


  The vulnerability of deep neural networks against adversarial examples -
inputs with small imperceptible perturbations - has gained a lot of attention
in the research community recently. Simultaneously, the number of parameters of
state-of-the-art deep learning models has been growing massively, with
implications on the memory and computational resources required to train and
deploy such models. One approach to control the size of neural networks is
retrospectively reducing the number of parameters, so-called neural network
pruning. Available research on the impact of neural network pruning on the
adversarial robustness is fragmentary and often does not adhere to established
principles of robustness evaluation. We close this gap by evaluating the
robustness of pruned models against L-0, L-2 and L-infinity attacks for a wide
range of attack strengths, several architectures, data sets, pruning methods,
and compression rates. Our results confirm that neural network pruning and
adversarial robustness are not mutually exclusive. Instead, sweet spots can be
found that are favorable in terms of model size and adversarial robustness.
Furthermore, we extend our analysis to situations that incorporate additional
assumptions on the adversarial scenario and show that depending on the
situation, different strategies are optimal.

    

### [[2108.08562] Concurrent Discrimination and Alignment for Self-Supervised Feature Learning](http://arxiv.org/abs/2108.08562)


  Existing self-supervised learning methods learn representation by means of
pretext tasks which are either (1) discriminating that explicitly specify which
features should be separated or (2) aligning that precisely indicate which
features should be closed together, but ignore the fact how to jointly and
principally define which features to be repelled and which ones to be
attracted. In this work, we combine the positive aspects of the discriminating
and aligning methods, and design a hybrid method that addresses the above
issue. Our method explicitly specifies the repulsion and attraction mechanism
respectively by discriminative predictive task and concurrently maximizing
mutual information between paired views sharing redundant information. We
qualitatively and quantitatively show that our proposed model learns better
features that are more effective for the diverse downstream tasks ranging from
classification to semantic segmentation. Our experiments on nine established
benchmarks show that the proposed model consistently outperforms the existing
state-of-the-art results of self-supervised and transfer learning protocol.

    

### [[2108.08577] Towards More Efficient Federated Learning with Better Optimization Objects](http://arxiv.org/abs/2108.08577)


  Federated Learning (FL) is a privacy-protected machine learning paradigm that
allows model to be trained directly at the edge without uploading data. One of
the biggest challenges faced by FL in practical applications is the
heterogeneity of edge node data, which will slow down the convergence speed and
degrade the performance of the model. For the above problems, a representative
solution is to add additional constraints in the local training, such as
FedProx, FedCurv and FedCL. However, the above algorithms still have room for
improvement. We propose to use the aggregation of all models obtained in the
past as new constraint target to further improve the performance of such
algorithms. Experiments in various settings demonstrate that our method
significantly improves the convergence speed and performance of the model.

    

### [[2108.08605] Using Multilevel Circulant Matrix Approximate to Speed Up Kernel Logistic Regression](http://arxiv.org/abs/2108.08605)


  Kernel logistic regression (KLR) is a classical nonlinear classifier in
statistical machine learning. Newton method with quadratic convergence rate can
solve KLR problem more effectively than the gradient method. However, an
obvious limitation of Newton method for training large-scale problems is the
$O(n^{3})$ time complexity and $O(n^{2})$ space complexity, where $n$ is the
number of training instances. In this paper, we employ the multilevel circulant
matrix (MCM) approximate kernel matrix to save in storage space and accelerate
the solution of the KLR. Combined with the characteristics of MCM and our
ingenious design, we propose an MCM approximate Newton iterative method. We
first simplify the Newton direction according to the semi-positivity of the
kernel matrix and then perform a two-step approximation of the Newton direction
by using MCM. Our method reduces the time complexity of each iteration to $O(n
\log n)$ by using the multidimensional fast Fourier transform (mFFT). In
addition, the space complexity can be reduced to $O(n)$ due to the built-in
periodicity of MCM. Experimental results on some large-scale binary and
multi-classification problems show that our method makes KLR scalable for
large-scale problems, with less memory consumption, and converges to test
accuracy without sacrifice in a shorter time.

    

### [[2108.08612] Settling the Variance of Multi-Agent Policy Gradients](http://arxiv.org/abs/2108.08612)


  Policy gradient (PG) methods are popular reinforcement learning (RL) methods
where a baseline is often applied to reduce the variance of gradient estimates.
In multi-agent RL (MARL), although the PG theorem can be naturally extended,
the effectiveness of multi-agent PG (MAPG) methods degrades as the variance of
gradient estimates increases rapidly with the number of agents. In this paper,
we offer a rigorous analysis of MAPG methods by, firstly, quantifying the
contributions of the number of agents and agents' explorations to the variance
of MAPG estimators. Based on this analysis, we derive the optimal baseline (OB)
that achieves the minimal variance. In comparison to the OB, we measure the
excess variance of existing MARL algorithms such as vanilla MAPG and COMA.
Considering using deep neural networks, we also propose a surrogate version of
OB, which can be seamlessly plugged into any existing PG methods in MARL. On
benchmarks of Multi-Agent MuJoCo and StarCraft challenges, our OB technique
effectively stabilises training and improves the performance of multi-agent PPO
and COMA algorithms by a significant margin.

    

### [[2108.08617] Spatially-Adaptive Image Restoration using Distortion-Guided Networks](http://arxiv.org/abs/2108.08617)


  We present a general learning-based solution for restoring images suffering
from spatially-varying degradations. Prior approaches are typically
degradation-specific and employ the same processing across different images and
different pixels within. However, we hypothesize that such spatially rigid
processing is suboptimal for simultaneously restoring the degraded pixels as
well as reconstructing the clean regions of the image. To overcome this
limitation, we propose SPAIR, a network design that harnesses
distortion-localization information and dynamically adjusts computation to
difficult regions in the image. SPAIR comprises of two components, (1) a
localization network that identifies degraded pixels, and (2) a restoration
network that exploits knowledge from the localization network in filter and
feature domain to selectively and adaptively restore degraded pixels. Our key
idea is to exploit the non-uniformity of heavy degradations in spatial-domain
and suitably embed this knowledge within distortion-guided modules performing
sparse normalization, feature extraction and attention. Our architecture is
agnostic to physical formation model and generalizes across several types of
spatially-varying degradations. We demonstrate the efficacy of SPAIR
individually on four restoration tasks-removal of rain-streaks, raindrops,
shadows and motion blur. Extensive qualitative and quantitative comparisons
with prior art on 11 benchmark datasets demonstrate that our
degradation-agnostic network design offers significant performance gains over
state-of-the-art degradation-specific architectures. Code available at
this https URL.

    

### [[2108.08627] An Innovative Attack Modelling and Attack Detection Approach for a Waiting Time-based Adaptive Traffic Signal Controller](http://arxiv.org/abs/2108.08627)


  An adaptive traffic signal controller (ATSC) combined with a connected
vehicle (CV) concept uses real-time vehicle trajectory data to regulate green
time and has the ability to reduce intersection waiting time significantly and
thereby improve travel time in a signalized corridor. However, the CV-based
ATSC increases the size of the surface vulnerable to potential cyber-attack,
allowing an attacker to generate disastrous traffic congestion in a roadway
network. An attacker can congest a route by generating fake vehicles by
maintaining traffic and car-following rules at a slow rate so that the signal
timing and phase change without having any abrupt changes in number of
vehicles. Because of the adaptive nature of ATSC, it is a challenge to model
this kind of attack and also to develop a strategy for detection. This paper
introduces an innovative "slow poisoning" cyberattack for a waiting time based
ATSC algorithm and a corresponding detection strategy. Thus, the objectives of
this paper are to: (i) develop a "slow poisoning" attack generation strategy
for an ATSC, and (ii) develop a prediction-based "slow poisoning" attack
detection strategy using a recurrent neural network -- i.e., long short-term
memory model. We have generated a "slow poisoning" attack modeling strategy
using a microscopic traffic simulator -- Simulation of Urban Mobility (SUMO) --
and used generated data from the simulation to develop both the attack model
and detection model. Our analyses revealed that the attack strategy is
effective in creating a congestion in an approach and detection strategy is
able to flag the attack.

    

### [[2108.08628] A Reinforcement Learning Approach for GNSS Spoofing Attack Detection of Autonomous Vehicles](http://arxiv.org/abs/2108.08628)


  A resilient and robust positioning, navigation, and timing (PNT) system is a
necessity for the navigation of autonomous vehicles (AVs). Global Navigation
Satelite System (GNSS) provides satellite-based PNT services. However, a
spoofer can temper an authentic GNSS signal and could transmit wrong position
information to an AV. Therefore, a GNSS must have the capability of real-time
detection and feedback-correction of spoofing attacks related to PNT receivers,
whereby it will help the end-user (autonomous vehicle in this case) to navigate
safely if it falls into any compromises. This paper aims to develop a deep
reinforcement learning (RL)-based turn-by-turn spoofing attack detection using
low-cost in-vehicle sensor data. We have utilized Honda Driving Dataset to
create attack and non-attack datasets, develop a deep RL model, and evaluate
the performance of the RL-based attack detection model. We find that the
accuracy of the RL model ranges from 99.99% to 100%, and the recall value is
100%. However, the precision ranges from 93.44% to 100%, and the f1 score
ranges from 96.61% to 100%. Overall, the analyses reveal that the RL model is
effective in turn-by-turn spoofing attack detection.

    

### [[2108.08631] Determinant-free fermionic wave function using feed-forward neural network](http://arxiv.org/abs/2108.08631)


  We propose a general framework for finding the ground state of many-body
fermionic systems by using feed-forward neural networks. The anticommutation
relation for fermions is usually implemented to a variational wave function by
the Slater determinant (or Pfaffian), which is a computational bottleneck
because of the numerical cost of $O(N^3)$ for $N$ particles. We bypass this
bottleneck by explicitly calculating the sign changes associated with particle
exchanges in real space and using fully connected neural networks for
optimizing the rest parts of the wave function. This reduces the computational
cost to $O(N^2)$ or less. We show that the accuracy of the approximation can be
improved by optimizing the "variance" of the energy simultaneously with the
energy itself. We also find that a reweighting method in Monte Carlo sampling
can stabilize the calculation. These improvements can be applied to other
approaches based on variational Monte Carlo methods. Moreover, we show that the
accuracy can be further improved by using the symmetry of the system, the
representative states, and an additional neural network implementing a
generalized Gutzwiller-Jastrow factor. We demonstrate the efficiency of the
method by applying it to a two-dimensional Hubbard model.

    

### [[2108.08635] A Sensor Fusion-based GNSS Spoofing Attack Detection Framework for Autonomous Vehicles](http://arxiv.org/abs/2108.08635)


  This paper presents a sensor fusion based Global Navigation Satellite System
(GNSS) spoofing attack detection framework for autonomous vehicles (AV) that
consists of two concurrent strategies: (i) detection of vehicle state using
predicted location shift -- i.e., distance traveled between two consecutive
timestamps -- and monitoring of vehicle motion state -- i.e., standstill/ in
motion; and (ii) detection and classification of turns (i.e., left or right).
Data from multiple low-cost in-vehicle sensors (i.e., accelerometer, steering
angle sensor, speed sensor, and GNSS) are fused and fed into a recurrent neural
network model, which is a long short-term memory (LSTM) network for predicting
the location shift, i.e., the distance that an AV travels between two
consecutive timestamps. This location shift is then compared with the
GNSS-based location shift to detect an attack. We have then combined k-Nearest
Neighbors (k-NN) and Dynamic Time Warping (DTW) algorithms to detect and
classify left and right turns using data from the steering angle sensor. To
prove the efficacy of the sensor fusion-based attack detection framework,
attack datasets are created for four unique and sophisticated spoofing
attacks-turn-by-turn, overshoot, wrong turn, and stop, using the publicly
available real-world Honda Research Institute Driving Dataset (HDD). Our
analysis reveals that the sensor fusion-based detection framework successfully
detects all four types of spoofing attacks within the required computational
latency threshold.

    

### [[2108.08643] Batch Curation for Unsupervised Contrastive Representation Learning](http://arxiv.org/abs/2108.08643)


  The state-of-the-art unsupervised contrastive visual representation learning
methods that have emerged recently (SimCLR, MoCo, SwAV) all make use of data
augmentations in order to construct a pretext task of instant discrimination
consisting of similar and dissimilar pairs of images. Similar pairs are
constructed by randomly extracting patches from the same image and applying
several other transformations such as color jittering or blurring, while
transformed patches from different image instances in a given batch are
regarded as dissimilar pairs. We argue that this approach can result similar
pairs that are \textit{semantically} dissimilar. In this work, we address this
problem by introducing a \textit{batch curation} scheme that selects batches
during the training process that are more inline with the underlying
contrastive objective. We provide insights into what constitutes beneficial
similar and dissimilar pairs as well as validate \textit{batch curation} on
CIFAR10 by integrating it in the SimCLR model.

    

### [[2108.08647] Multi-Center Federated Learning](http://arxiv.org/abs/2108.08647)


  Federated learning (FL) can protect data privacy in distributed learning
since it merely collects local gradients from users without access to their
data. However, FL is fragile in the presence of heterogeneity that is commonly
encountered in practical settings, e.g., non-IID data over different users.
Existing FL approaches usually update a single global model to capture the
shared knowledge of all users by aggregating their gradients, regardless of the
discrepancy between their data distributions. By comparison, a mixture of
multiple global models could capture the heterogeneity across various users if
assigning the users to different global models (i.e., centers) in FL. To this
end, we propose a novel multi-center aggregation mechanism . It learns multiple
global models from data, and simultaneously derives the optimal matching
between users and centers. We then formulate it as a bi-level optimization
problem that can be efficiently solved by a stochastic expectation maximization
(EM) algorithm. Experiments on multiple benchmark datasets of FL show that our
method outperforms several popular FL competitors. The source code are open
source on Github.

    

### [[2108.08655] Global Convergence of the ODE Limit for Online Actor-Critic Algorithms in Reinforcement Learning](http://arxiv.org/abs/2108.08655)


  Actor-critic algorithms are widely used in reinforcement learning, but are
challenging to mathematically analyze due to the online arrival of non-i.i.d.
data samples. The distribution of the data samples dynamically changes as the
model is updated, introducing a complex feedback loop between the data
distribution and the reinforcement learning algorithm. We prove that, under a
time rescaling, the online actor-critic algorithm with tabular parametrization
converges to an ordinary differential equations (ODEs) as the number of updates
becomes large. The proof first establishes the geometric ergodicity of the data
samples under a fixed actor policy. Then, using a Poisson equation, we prove
that the fluctuations of the data samples around a dynamic probability measure,
which is a function of the evolving actor model, vanish as the number of
updates become large. Once the ODE limit has been derived, we study its
convergence properties using a two time-scale analysis which asymptotically
de-couples the critic ODE from the actor ODE. The convergence of the critic to
the solution of the Bellman equation and the actor to the optimal policy are
proven. In addition, a convergence rate to this global minimum is also
established. Our convergence analysis holds under specific choices for the
learning rates and exploration rates in the actor-critic algorithm, which could
provide guidance for the implementation of actor-critic algorithms in practice.

    

### [[2108.08659] Residual Tensor Train: a Flexible and Efficient Approach for Learning Multiple Multilinear Correlations](http://arxiv.org/abs/2108.08659)


  Tensor Train (TT) approach has been successfully applied in the modelling of
the multilinear interaction of features. Nevertheless, the existing models lack
flexibility and generalizability, as they only model a single type of
high-order correlation. In practice, multiple multilinear correlations may
exist within the features. In this paper, we present a novel Residual Tensor
Train (ResTT) which integrates the merits of TT and residual structure to
capture the multilinear feature correlations, from low to higher orders, within
the same model. In particular, we prove that the fully-connected layer in
neural networks and the Volterra series can be taken as special cases of ResTT.
Furthermore, we derive the rule for weight initialization that stabilizes the
training of ResTT based on a mean-field analysis. We prove that such a rule is
much more relaxed than that of TT, which means ResTT can easily address the
vanishing and exploding gradient problem that exists in the current TT models.
Numerical experiments demonstrate that ResTT outperforms the state-of-the-art
tensor network approaches, and is competitive with the benchmark deep learning
models on MNIST and Fashion-MNIST datasets.

    

### [[2108.08670] On Accelerating Distributed Convex Optimizations](http://arxiv.org/abs/2108.08670)


  This paper studies a distributed multi-agent convex optimization problem. The
system comprises multiple agents in this problem, each with a set of local data
points and an associated local cost function. The agents are connected to a
server, and there is no inter-agent communication. The agents' goal is to learn
a parameter vector that optimizes the aggregate of their local costs without
revealing their local data points. In principle, the agents can solve this
problem by collaborating with the server using the traditional distributed
gradient-descent method. However, when the aggregate cost is ill-conditioned,
the gradient-descent method (i) requires a large number of iterations to
converge, and (ii) is highly unstable against process noise. We propose an
iterative pre-conditioning technique to mitigate the deleterious effects of the
cost function's conditioning on the convergence rate of distributed
gradient-descent. Unlike the conventional pre-conditioning techniques, the
pre-conditioner matrix in our proposed technique updates iteratively to
facilitate implementation on the distributed network. In the distributed
setting, we provably show that the proposed algorithm converges linearly with
an improved rate of convergence than the traditional and adaptive
gradient-descent methods. Additionally, for the special case when the minimizer
of the aggregate cost is unique, our algorithm converges superlinearly. We
demonstrate our algorithm's superior performance compared to prominent
distributed algorithms for solving real logistic regression problems and
emulating neural network training via a noisy quadratic model, thereby
signifying the proposed algorithm's efficiency for distributively solving
non-convex optimization. Moreover, we empirically show that the proposed
algorithm results in faster training without compromising the generalization
performance.

    

### [[2108.08677] Order Optimal One-Shot Federated Learning for non-Convex Loss Functions](http://arxiv.org/abs/2108.08677)


  We consider the problem of federated learning in a one-shot setting in which
there are $m$ machines, each observing $n$ samples function from an unknown
distribution on non-convex loss functions. Let $F:[-1,1]^d\to\mathbb{R}$ be the
expected loss function with respect to this unknown distribution. The goal is
to find an estimate of the minimizer of $F$. Based on its observations, each
machine generates a signal of bounded length $B$ and sends it to a server. The
sever collects signals of all machines and outputs an estimate of the minimizer
of $F$. We propose a distributed learning algorithm, called Multi-Resolution
Estimator for Non-Convex loss function (MRE-NC), whose expected error is
bounded by $\max\big(1/\sqrt{n}(mB)^{1/d}, 1/\sqrt{mn}\big)$, up to
polylogarithmic factors. We also provide a matching lower bound on the
performance of any algorithm, showing that MRE-NC is order optimal in terms of
$n$ and $m$. Experiments on synthetic and real data show the effectiveness of
MRE-NC in distributed learning of model's parameters for non-convex loss
functions.

    

### [[2108.08687] Clustering dynamics on graphs: from spectral clustering to mean shift through Fokker-Planck interpolation](http://arxiv.org/abs/2108.08687)


  In this work we build a unifying framework to interpolate between
density-driven and geometry-based algorithms for data clustering, and
specifically, to connect the mean shift algorithm with spectral clustering at
discrete and continuum levels. We seek this connection through the introduction
of Fokker-Planck equations on data graphs. Besides introducing new forms of
mean shift algorithms on graphs, we provide new theoretical insights on the
behavior of the family of diffusion maps in the large sample limit as well as
provide new connections between diffusion maps and mean shift dynamics on a
fixed graph. Several numerical examples illustrate our theoretical findings and
highlight the benefits of interpolating density-driven and geometry-based
clustering algorithms.

    

### [[2108.08689] Analyze and Design Network Architectures by Recursion Formulas](http://arxiv.org/abs/2108.08689)


  The effectiveness of shortcut/skip-connection has been widely verified, which
inspires massive explorations on neural architecture design. This work attempts
to find an effective way to design new network architectures. It is discovered
that the main difference between network architectures can be reflected in
their recursion formulas. Based on this, a methodology is proposed to design
novel network architectures from the perspective of mathematical formulas.
Afterwards, a case study is provided to generate an improved architecture based
on ResNet. Furthermore, the new architecture is compared with ResNet and then
tested on ResNet-based networks. Massive experiments are conducted on CIFAR and
ImageNet, which witnesses the significant performance improvements provided by
the architecture.

    

### [[2108.08704] IT2CFNN: An Interval Type-2 Correlation-Aware Fuzzy Neural Network to Construct Non-Separable Fuzzy Rules with Uncertain and Adaptive Shapes for Nonlinear Function Approximation](http://arxiv.org/abs/2108.08704)


  In this paper, a new interval type-2 fuzzy neural network able to construct
non-separable fuzzy rules with adaptive shapes is introduced. To reflect the
uncertainty, the shape of fuzzy sets considered to be uncertain. Therefore, a
new form of interval type-2 fuzzy sets based on a general Gaussian model able
to construct different shapes (including triangular, bell-shaped, trapezoidal)
is proposed. To consider the interactions among input variables, input vectors
are transformed to new feature spaces with uncorrelated variables proper for
defining each fuzzy rule. Next, the new features are fed to a fuzzification
layer using proposed interval type-2 fuzzy sets with adaptive shape.
Consequently, interval type-2 non-separable fuzzy rules with proper shapes,
considering the local interactions of variables and the uncertainty are formed.
For type reduction the contribution of the upper and lower firing strengths of
each fuzzy rule are adaptively selected separately. To train different
parameters of the network, the Levenberg-Marquadt optimization method is
utilized. The performance of the proposed method is investigated on clean and
noisy datasets to show the ability to consider the uncertainty. Moreover, the
proposed paradigm, is successfully applied to real-world time-series
predictions, regression problems, and nonlinear system identification.
According to the experimental results, the performance of our proposed model
outperforms other methods with a more parsimonious structure.

    

### [[2108.08706] Attribute-based Explanations of Non-Linear Embeddings of High-Dimensional Data](http://arxiv.org/abs/2108.08706)


  Embeddings of high-dimensional data are widely used to explore data, to
verify analysis results, and to communicate information. Their explanation, in
particular with respect to the input attributes, is often difficult. With
linear projects like PCA the axes can still be annotated meaningfully. With
non-linear projections this is no longer possible and alternative strategies
such as attribute-based color coding are required. In this paper, we review
existing augmentation techniques and discuss their limitations. We present the
Non-Linear Embeddings Surveyor (NoLiES) that combines a novel augmentation
strategy for projected data (rangesets) with interactive analysis in a small
multiples setting. Rangesets use a set-based visualization approach for binned
attribute values that enable the user to quickly observe structure and detect
outliers. We detail the link between algebraic topology and rangesets and
demonstrate the utility of NoLiES in case studies with various challenges
(complex attribute value distribution, many attributes, many data points) and a
real-world application to understand latent features of matrix completion in
thermodynamics.

    

### [[2108.08708] Czech News Dataset for Semanic Textual Similarity](http://arxiv.org/abs/2108.08708)


  This paper describes a novel dataset consisting of sentences with semantic
similarity annotations. The data originate from the journalistic domain in the
Czech language. We describe the process of collecting and annotating the data
in detail. The dataset contains 138,556 human annotations divided into train
and test sets. In total, 485 journalism students participated in the creation
process. To increase the reliability of the test set, we compute the annotation
as an average of 9 individual annotations. We evaluate the quality of the
dataset by measuring inter and intra annotation annotators' agreements. Beside
agreement numbers, we provide detailed statistics of the collected dataset. We
conclude our paper with a baseline experiment of building a system for
predicting the semantic similarity of sentences. Due to the massive number of
training annotations (116 956), the model can perform significantly better than
an average annotator (0,92 versus 0,86 of Person's correlation coefficients).

    

### [[2108.08709] Neural density estimation and uncertainty quantification for laser induced breakdown spectroscopy spectra](http://arxiv.org/abs/2108.08709)


  Constructing probability densities for inference in high-dimensional spectral
data is often intractable. In this work, we use normalizing flows on structured
spectral latent spaces to estimate such densities, enabling downstream
inference tasks. In addition, we evaluate a method for uncertainty
quantification when predicting unobserved state vectors associated with each
spectrum. We demonstrate the capability of this approach on laser-induced
breakdown spectroscopy data collected by the ChemCam instrument on the Mars
rover Curiosity. Using our approach, we are able to generate realistic spectral
samples and to accurately predict state vectors with associated well-calibrated
uncertainties. We anticipate that this methodology will enable efficient
probabilistic modeling of spectral data, leading to potential advances in
several areas, including out-of-distribution detection and sensitivity
analysis.

    

### [[2108.08712] Teaching Uncertainty Quantification in Machine Learning through Use Cases](http://arxiv.org/abs/2108.08712)


  Uncertainty in machine learning is not generally taught as general knowledge
in Machine Learning course curricula. In this paper we propose a short
curriculum for a course about uncertainty in machine learning, and complement
the course with a selection of use cases, aimed to trigger discussion and let
students play with the concepts of uncertainty in a programming setting. Our
use cases cover the concept of output uncertainty, Bayesian neural networks and
weight distributions, sources of uncertainty, and out of distribution
detection. We expect that this curriculum and set of use cases motivates the
community to adopt these important concepts into courses for safety in AI.

    

### [[2108.08721] Improving Semi-Supervised Learning for Remaining Useful Lifetime Estimation Through Self-Supervision](http://arxiv.org/abs/2108.08721)


  RUL estimation suffers from a server data imbalance where data from machines
near their end of life is rare. Additionally, the data produced by a machine
can only be labeled after the machine failed. Semi-Supervised Learning (SSL)
can incorporate the unlabeled data produced by machines that did not yet fail.
Previous work on SSL evaluated their approaches under unrealistic conditions
where the data near failure was still available. Even so, only moderate
improvements were made. This paper proposes a novel SSL approach based on
self-supervised pre-training. The method can outperform two competing
approaches from the literature and a supervised baseline under realistic
conditions on the NASA C-MAPSS dataset. Nevertheless, we observe degraded
performance in some circumstances and discuss possible causes.

    

### [[2108.08723] Feature-weighted Stacking for Nonseasonal Time Series Forecasts: A Case Study of the COVID-19 Epidemic Curves](http://arxiv.org/abs/2108.08723)


  We investigate ensembling techniques in forecasting and examine their
potential for use in nonseasonal time-series similar to those in the early days
of the COVID-19 pandemic. Developing improved forecast methods is essential as
they provide data-driven decisions to organisations and decision-makers during
critical phases. We propose using late data fusion, using a stacked ensemble of
two forecasting models and two meta-features that prove their predictive power
during a preliminary forecasting stage. The final ensembles include a Prophet
and long short term memory (LSTM) neural network as base models. The base
models are combined by a multilayer perceptron (MLP), taking into account
meta-features that indicate the highest correlation with each base model's
forecast accuracy. We further show that the inclusion of meta-features
generally improves the ensemble's forecast accuracy across two forecast
horizons of seven and fourteen days. This research reinforces previous work and
demonstrates the value of combining traditional statistical models with deep
learning models to produce more accurate forecast models for time-series across
domains.

    

### [[2108.08728] Counterfactual Attention Learning for Fine-Grained Visual Categorization and Re-identification](http://arxiv.org/abs/2108.08728)


  Attention mechanism has demonstrated great potential in fine-grained visual
recognition tasks. In this paper, we present a counterfactual attention
learning method to learn more effective attention based on causal inference.
Unlike most existing methods that learn visual attention based on conventional
likelihood, we propose to learn the attention with counterfactual causality,
which provides a tool to measure the attention quality and a powerful
supervisory signal to guide the learning process. Specifically, we analyze the
effect of the learned visual attention on network prediction through
counterfactual intervention and maximize the effect to encourage the network to
learn more useful attention for fine-grained image recognition. Empirically, we
evaluate our method on a wide range of fine-grained recognition tasks where
attention plays a crucial role, including fine-grained image categorization,
person re-identification, and vehicle re-identification. The consistent
improvement on all benchmarks demonstrates the effectiveness of our method.
Code is available at this https URL


### [[2108.08735] SiReN: Sign-Aware Recommendation Using Graph Neural Networks](http://arxiv.org/abs/2108.08735)


  In recent years, many recommender systems using network embedding (NE) such
as graph neural networks (GNNs) have been extensively studied in the sense of
improving recommendation accuracy. However, such attempts have focused mostly
on utilizing only the information of positive user-item interactions with high
ratings. Thus, there is a challenge on how to make use of low rating scores for
representing users' preferences since low ratings can be still informative in
designing NE-based recommender systems. In this study, we present SiReN, a new
sign-aware recommender system based on GNN models. Specifically, SiReN has
three key components: 1) constructing a signed bipartite graph for more
precisely representing users' preferences, which is split into two
edge-disjoint graphs with positive and negative edges each, 2) generating two
embeddings for the partitioned graphs with positive and negative edges via a
GNN model and a multi-layer perceptron (MLP), respectively, and then using an
attention model to obtain the final embeddings, and 3) establishing a
sign-aware Bayesian personalized ranking (BPR) loss function in the process of
optimization. Through comprehensive experiments, we empirically demonstrate
that SiReN consistently outperforms state-of-the-art NE-aided recommendation
methods.

    

### [[2108.08752] A Framework for an Assessment of the Kernel-target Alignment in Tree Ensemble Kernel Learning](http://arxiv.org/abs/2108.08752)


  Kernels ensuing from tree ensembles such as random forest (RF) or gradient
boosted trees (GBT), when used for kernel learning, have been shown to be
competitive to their respective tree ensembles (particularly in higher
dimensional scenarios). On the other hand, it has been also shown that
performance of the kernel algorithms depends on the degree of the kernel-target
alignment. However, the kernel-target alignment for kernel learning based on
the tree ensembles has not been investigated and filling this gap is the main
goal of our work.
Using the eigenanalysis of the kernel matrix, we demonstrate that for
continuous targets good performance of the tree-based kernel learning is
associated with strong kernel-target alignment. Moreover, we show that well
performing tree ensemble based kernels are characterized by strong target
aligned components that are expressed through scalar products between the
eigenvectors of the kernel matrix and the target. This suggests that when tree
ensemble based kernel learning is successful, relevant information for the
supervised problem is concentrated near lower dimensional manifold spanned by
the target aligned components. Persistence of the strong target aligned
components in tree ensemble based kernels is further supported by sensitivity
analysis via landmark learning. In addition to a comprehensive simulation
study, we also provide experimental results from several real life data sets
that are in line with the simulations.

    

### [[2108.08754] Temporal Graph Network Embedding with Causal Anonymous Walks Representations](http://arxiv.org/abs/2108.08754)


  Many tasks in graph machine learning, such as link prediction and node
classification, are typically solved by using representation learning, in which
each node or edge in the network is encoded via an embedding. Though there
exists a lot of network embeddings for static graphs, the task becomes much
more complicated when the dynamic (i.e. temporal) network is analyzed. In this
paper, we propose a novel approach for dynamic network representation learning
based on Temporal Graph Network by using a highly custom message generating
function by extracting Causal Anonymous Walks. For evaluation, we provide a
benchmark pipeline for the evaluation of temporal network embeddings. This work
provides the first comprehensive comparison framework for temporal network
representation learning in every available setting for graph machine learning
problems involving node classification and link prediction. The proposed model
outperforms state-of-the-art baseline models. The work also justifies the
difference between them based on evaluation in various transductive/inductive
edge/node classification tasks. In addition, we show the applicability and
superior performance of our model in the real-world downstream graph machine
learning task provided by one of the top European banks, involving credit
scoring based on transaction data.

    

### [[2108.08760] Efficient remedies for outlier detection with variational autoencoders](http://arxiv.org/abs/2108.08760)


  Deep networks often make confident, yet incorrect, predictions when tested
with outlier data that is far removed from their training distributions.
Likelihoods computed by deep generative models are a candidate metric for
outlier detection with unlabeled data. Yet, previous studies have shown that
such likelihoods are unreliable and can be easily biased by simple
transformations to input data. Here, we examine outlier detection with
variational autoencoders (VAEs), among the simplest class of deep generative
models. First, we show that a theoretically-grounded correction readily
ameliorates a key bias with VAE likelihood estimates. The bias correction is
model-free, sample-specific, and accurately computed with the Bernoulli and
continuous Bernoulli visible distributions. Second, we show that a well-known
preprocessing technique, contrast normalization, extends the effectiveness of
bias correction to natural image datasets. Third, we show that the variance of
the likelihoods computed over an ensemble of VAEs also enables robust outlier
detection. We perform a comprehensive evaluation of our remedies with nine
(grayscale and natural) image datasets, and demonstrate significant advantages,
in terms of both speed and accuracy, over four other state-of-the-art methods.
Our lightweight remedies are biologically inspired and may serve to achieve
efficient outlier detection with many types of deep generative models.

    

### [[2108.08762] Dynamic Difficulty Adjustment in Virtual Reality Exergames through Experience-driven Procedural Content Generation](http://arxiv.org/abs/2108.08762)


  Virtual Reality (VR) games that feature physical activities have been shown
to increase players' motivation to do physical exercise. However, for such
exercises to have a positive healthcare effect, they have to be repeated
several times a week. To maintain player motivation over longer periods of
time, games often employ Dynamic Difficulty Adjustment (DDA) to adapt the
game's challenge according to the player's capabilities. For exercise games,
this is mostly done by tuning specific in-game parameters like the speed of
objects. In this work, we propose to use experience-driven Procedural Content
Generation for DDA in VR exercise games by procedurally generating levels that
match the player's current capabilities. Not only finetuning specific
parameters but creating completely new levels has the potential to decrease
repetition over longer time periods and allows for the simultaneous adaptation
of the cognitive and physical challenge of the exergame. As a proof-of-concept,
we implement an initial prototype in which the player must traverse a maze that
includes several exercise rooms, whereby the generation of the maze is realized
by a neural network. Passing those exercise rooms requires the player to
perform physical activities. To match the player's capabilities, we use Deep
Reinforcement Learning to adjust the structure of the maze and to decide which
exercise rooms to include in the maze. We evaluate our prototype in an
exploratory user study utilizing both biodata and subjective questionnaires.

    

### [[2108.08765] Provably Efficient Generative Adversarial Imitation Learning for Online and Offline Setting with Linear Function Approximation](http://arxiv.org/abs/2108.08765)


  In generative adversarial imitation learning (GAIL), the agent aims to learn
a policy from an expert demonstration so that its performance cannot be
discriminated from the expert policy on a certain predefined reward set. In
this paper, we study GAIL in both online and offline settings with linear
function approximation, where both the transition and reward function are
linear in the feature maps. Besides the expert demonstration, in the online
setting the agent can interact with the environment, while in the offline
setting the agent only accesses an additional dataset collected by a prior. For
online GAIL, we propose an optimistic generative adversarial policy
optimization algorithm (OGAP) and prove that OGAP achieves
$\widetilde{\mathcal{O}}(H^2 d^{3/2}K^{1/2}+KH^{3/2}dN_1^{-1/2})$ regret. Here
$N_1$ represents the number of trajectories of the expert demonstration, $d$ is
the feature dimension, and $K$ is the number of episodes.
For offline GAIL, we propose a pessimistic generative adversarial policy
optimization algorithm (PGAP). For an arbitrary additional dataset, we obtain
the optimality gap of PGAP, achieving the minimax lower bound in the
utilization of the additional dataset. Assuming sufficient coverage on the
additional dataset, we show that PGAP achieves
$\widetilde{\mathcal{O}}(H^{2}dK^{-1/2}
+H^2d^{3/2}N_2^{-1/2}+H^{3/2}dN_1^{-1/2} \ )$ optimality gap. Here $N_2$
represents the number of trajectories of the additional dataset with sufficient
coverage.

    

### [[2108.08767] Threshold Phenomena in Learning Halfspaces with Massart Noise](http://arxiv.org/abs/2108.08767)


  We study the problem of PAC learning halfspaces on $\mathbb{R}^d$ with
Massart noise under Gaussian marginals. In the Massart noise model, an
adversary is allowed to flip the label of each point $\mathbf{x}$ with
probability $\eta(\mathbf{x}) \leq \eta$, for some parameter $\eta \in
[0,1/2]$.
The goal of the learner is to output a hypothesis with missclassification
error $\mathrm{opt} + \epsilon$, where $\mathrm{opt}$ is the error of the
target halfspace. Prior work studied this problem assuming that the target
halfspace is homogeneous and that the parameter $\eta$ is strictly smaller than
$1/2$. We explore how the complexity of the problem changes when either of
these assumptions is removed, establishing the following threshold phenomena:
For $\eta = 1/2$, we prove a lower bound of $d^{\Omega (\log(1/\epsilon))}$
on the complexity of any Statistical Query (SQ) algorithm for the problem,
which holds even for homogeneous halfspaces. On the positive side, we give a
new learning algorithm for arbitrary halfspaces in this regime with sample
complexity and running time $O_\epsilon(1) \, d^{O(\log(1/\epsilon))}$.
For $\eta <1/2$, we establish a lower bound of $d^{\Omega(\log(1/\gamma))}$
on the SQ complexity of the problem, where $\gamma = \max\{\epsilon,
\min\{\mathbf{Pr}[f(\mathbf{x}) = 1], \mathbf{Pr}[f(\mathbf{x}) = -1]\} \}$ and
$f$ is the target halfspace. In particular, this implies an SQ lower bound of
$d^{\Omega (\log(1/\epsilon) )}$ for learning arbitrary Massart halfspaces
(even for small constant $\eta$). We complement this lower bound with a new
learning algorithm for this regime with sample complexity and runtime
$d^{O_{\eta}(\log(1/\gamma))} \mathrm{poly}(1/\epsilon)$.
Taken together, our results qualitatively characterize the complexity of
learning halfspaces in the Massart model.

    

### [[2108.08768] Client Selection Approach in Support of Clustered Federated Learning over Wireless Edge Networks](http://arxiv.org/abs/2108.08768)


  Clustered Federated Multitask Learning (CFL) was introduced as an efficient
scheme to obtain reliable specialized models when data is imbalanced and
distributed in a non-i.i.d. (non-independent and identically distributed)
fashion amongst clients. While a similarity measure metric, like the cosine
similarity, can be used to endow groups of the client with a specialized model,
this process can be arduous as the server should involve all clients in each of
the federated learning rounds. Therefore, it is imperative that a subset of
clients is selected periodically due to the limited bandwidth and latency
constraints at the network edge. To this end, this paper proposes a new client
selection algorithm that aims to accelerate the convergence rate for obtaining
specialized machine learning models that achieve high test accuracies for all
client groups. Specifically, we introduce a client selection approach that
leverages the devices' heterogeneity to schedule the clients based on their
round latency and exploits the bandwidth reuse for clients that consume more
time to update the model. Then, the server performs model averaging and
clusters the clients based on predefined thresholds. When a specific cluster
reaches a stationary point, the proposed algorithm uses a greedy scheduling
algorithm for that group by selecting the clients with less latency to update
the model. Extensive experiments show that the proposed approach lowers the
training time and accelerates the convergence rate by up to 50% while imbuing
each client with a specialized model that is fit for its local data
distribution.

    

### [[2108.08770] Learning-to-learn non-convex piecewise-Lipschitz functions](http://arxiv.org/abs/2108.08770)


  We analyze the meta-learning of the initialization and step-size of learning
algorithms for piecewise-Lipschitz functions, a non-convex setting with
applications to both machine learning and algorithms. Starting from recent
regret bounds for the exponential forecaster on losses with dispersed
discontinuities, we generalize them to be initialization-dependent and then use
this result to propose a practical meta-learning procedure that learns both the
initialization and the step-size of the algorithm from multiple online learning
tasks. Asymptotically, we guarantee that the average regret across tasks scales
with a natural notion of task-similarity that measures the amount of overlap
between near-optimal regions of different tasks. Finally, we instantiate the
method and its guarantee in two important settings: robust meta-learning and
multi-task data-driven algorithm design.

    

### [[2108.08775] MobileCaps: A Lightweight Model for Screening and Severity Analysis of COVID-19 Chest X-Ray Images](http://arxiv.org/abs/2108.08775)


  The world is going through a challenging phase due to the disastrous effect
caused by the COVID-19 pandemic on the healthcare system and the economy. The
rate of spreading, post-COVID-19 symptoms, and the occurrence of new strands of
COVID-19 have put the healthcare systems in disruption across the globe. Due to
this, the task of accurately screening COVID-19 cases has become of utmost
priority. Since the virus infects the respiratory system, Chest X-Ray is an
imaging modality that is adopted extensively for the initial screening. We have
performed a comprehensive study that uses CXR images to identify COVID-19 cases
and realized the necessity of having a more generalizable model. We utilize
MobileNetV2 architecture as the feature extractor and integrate it into Capsule
Networks to construct a fully automated and lightweight model termed as
MobileCaps. MobileCaps is trained and evaluated on the publicly available
dataset with the model ensembling and Bayesian optimization strategies to
efficiently classify CXR images of patients with COVID-19 from non-COVID-19
pneumonia and healthy cases. The proposed model is further evaluated on two
additional RT-PCR confirmed datasets to demonstrate the generalizability. We
also introduce MobileCaps-S and leverage it for performing severity assessment
of CXR images of COVID-19 based on the Radiographic Assessment of Lung Edema
(RALE) scoring technique. Our classification model achieved an overall recall
of 91.60, 94.60, 92.20, and a precision of 98.50, 88.21, 92.62 for COVID-19,
non-COVID-19 pneumonia, and healthy cases, respectively. Further, the severity
assessment model attained an R$^2$ coefficient of 70.51. Owing to the fact that
the proposed models have fewer trainable parameters than the state-of-the-art
models reported in the literature, we believe our models will go a long way in
aiding healthcare systems in the battle against the pandemic.

    

### [[2108.08780] Optimally Efficient Sequential Calibration of Binary Classifiers to Minimize Classification Error](http://arxiv.org/abs/2108.08780)


  In this work, we aim to calibrate the score outputs of an estimator for the
binary classification problem by finding an 'optimal' mapping to class
probabilities, where the 'optimal' mapping is in the sense that minimizes the
classification error (or equivalently, maximizes the accuracy). We show that
for the given target variables and the score outputs of an estimator, an
'optimal' soft mapping, which monotonically maps the score values to
probabilities, is a hard mapping that maps the score values to $0$ and $1$. We
show that for class weighted (where the accuracy for one class is more
important) and sample weighted (where the samples' accurate classifications are
not equally important) errors, or even general linear losses; this hard mapping
characteristic is preserved. We propose a sequential recursive merger approach,
which produces an 'optimal' hard mapping (for the observed samples so far)
sequentially with each incoming new sample. Our approach has a logarithmic in
sample size time complexity, which is optimally efficient.

    

### [[2108.08790] Simple is better: Making Decision Trees faster using random sampling](http://arxiv.org/abs/2108.08790)


  In recent years, gradient boosted decision trees have become popular in
building robust machine learning models on big data. The primary technique that
has enabled these algorithms success has been distributing the computation
while building the decision trees. A distributed decision tree building, in
turn, has been enabled by building quantiles of the big datasets and choosing
the candidate split points from these quantile sets. In XGBoost, for instance,
a sophisticated quantile building algorithm is employed to identify the
candidate split points for the decision trees. This method is often projected
to yield better results when the computation is distributed. In this paper, we
dispel the notion that these methods provide more accurate and scalable methods
for building decision trees in a distributed manner. In a significant
contribution, we show theoretically and empirically that choosing the split
points uniformly at random provides the same or even better performance in
terms of accuracy and computational efficiency. Hence, a simple random
selection of points suffices for decision tree building compared to more
sophisticated methods.

    

### [[2108.08800] EqGNN: Equalized Node Opportunity in Graphs](http://arxiv.org/abs/2108.08800)


  Graph neural networks (GNNs), has been widely used for supervised learning
tasks in graphs reaching state-of-the-art results. However, little work was
dedicated to creating unbiased GNNs, i.e., where the classification is
uncorrelated with sensitive attributes, such as race or gender. Some ignore the
sensitive attributes or optimize for the criteria of statistical parity for
fairness. However, it has been shown that neither approaches ensure fairness,
but rather cripple the utility of the prediction task. In this work, we present
a GNN framework that allows optimizing representations for the notion of
Equalized Odds fairness criteria. The architecture is composed of three
components: (1) a GNN classifier predicting the utility class, (2) a sampler
learning the distribution of the sensitive attributes of the nodes given their
labels. It generates samples fed into a (3) discriminator that discriminates
between true and sampled sensitive attributes using a novel "permutation loss"
function. Using these components, we train a model to neglect information
regarding the sensitive attribute only with respect to its label. To the best
of our knowledge, we are the first to optimize GNNs for the equalized odds
criteria. We evaluate our classifier over several graph datasets and sensitive
attributes and show our algorithm reaches state-of-the-art results.

    

### [[2108.08802] Lifelong Computing](http://arxiv.org/abs/2108.08802)


  Computing systems form the backbone of many aspects of our life, hence they
are becoming as vital as water, electricity, and road infrastructures for our
society. Yet, engineering long running computing systems that achieve their
goals in ever-changing environments pose significant challenges. Currently, we
can build computing systems that adjust or learn over time to match changes
that were anticipated. However, dealing with unanticipated changes, such as
anomalies, novelties, new goals or constraints, requires system evolution,
which remains in essence a human-driven activity. Given the growing complexity
of computing systems and the vast amount of highly complex data to process,
this approach will eventually become unmanageable. To break through the status
quo, we put forward a new paradigm for the design and operation of computing
systems that we coin "lifelong computing." The paradigm starts from
computing-learning systems that integrate computing/service modules and
learning modules. Computing warehouses offer such computing elements together
with data sheets and usage guides. When detecting anomalies, novelties, new
goals or constraints, a lifelong computing system activates an evolutionary
self-learning engine that runs online experiments to determine how the
computing-learning system needs to evolve to deal with the changes, thereby
changing its architecture and integrating new computing elements from computing
warehouses as needed. Depending on the domain at hand, some activities of
lifelong computing systems can be supported by humans. We motivate the need for
lifelong computing with a future fish farming scenario, outline a blueprint
architecture for lifelong computing systems, and highlight key research
challenges to realise the vision of lifelong computing.

    

### [[2108.08809] Surrogate Assisted Strategies (The Parameterisation of an Infectious Disease Agent-Based Model)](http://arxiv.org/abs/2108.08809)


  Parameter calibration is a significant challenge in agent-based modelling and
simulation (ABMS). An agent-based model's (ABM) complexity grows as the number
of parameters required to be calibrated increases. This parameter expansion
leads to the ABMS equivalent of the \say{curse of dimensionality}. In
particular, infeasible computational requirements searching an infinite
parameter space. We propose a more comprehensive and adaptive ABMS Framework
that can effectively swap out parameterisation strategies and surrogate models
to parameterise an infectious disease ABM. This framework allows us to evaluate
different strategy-surrogate combinations' performance in accuracy and
efficiency (speedup). We show that we achieve better than parity in accuracy
across the surrogate assisted sampling strategies and the baselines. Also, we
identify that the Metric Stochastic Response Surface strategy combined with the
Support Vector Machine surrogate is the best overall in getting closest to the
true synthetic parameters. Also, we show that DYnamic COOrdindate Search Using
Response Surface Models with XGBoost as a surrogate attains in combination the
highest probability of approximating a cumulative synthetic daily infection
data distribution and achieves the most significant speedup with regards to our
analysis. Lastly, we show in a real-world setting that DYCORS XGBoost and MSRS
SVM can approximate the real world cumulative daily infection distribution with
$97.12$\% and $96.75$\% similarity respectively.

    

### [[2108.08810] Do Vision Transformers See Like Convolutional Neural Networks?](http://arxiv.org/abs/2108.08810)


  Convolutional neural networks (CNNs) have so far been the de-facto model for
visual data. Recent work has shown that (Vision) Transformer models (ViT) can
achieve comparable or even superior performance on image classification tasks.
This raises a central question: how are Vision Transformers solving these
tasks? Are they acting like convolutional networks, or learning entirely
different visual representations? Analyzing the internal representation
structure of ViTs and CNNs on image classification benchmarks, we find striking
differences between the two architectures, such as ViT having more uniform
representations across all layers. We explore how these differences arise,
finding crucial roles played by self-attention, which enables early aggregation
of global information, and ViT residual connections, which strongly propagate
features from lower to higher layers. We study the ramifications for spatial
localization, demonstrating ViTs successfully preserve input spatial
information, with noticeable effects from different classification methods.
Finally, we study the effect of (pretraining) dataset scale on intermediate
features and transfer learning, and conclude with a discussion on connections
to new architectures such as the MLP-Mixer.

    

### [[2108.08812] Provable Benefits of Actor-Critic Methods for Offline Reinforcement Learning](http://arxiv.org/abs/2108.08812)


  Actor-critic methods are widely used in offline reinforcement learning
practice, but are not so well-understood theoretically. We propose a new
offline actor-critic algorithm that naturally incorporates the pessimism
principle, leading to several key advantages compared to the state of the art.
The algorithm can operate when the Bellman evaluation operator is closed with
respect to the action value function of the actor's policies; this is a more
general setting than the low-rank MDP model. Despite the added generality, the
procedure is computationally tractable as it involves the solution of a
sequence of second-order programs. We prove an upper bound on the suboptimality
gap of the policy returned by the procedure that depends on the data coverage
of any arbitrary, possibly data dependent comparator policy. The achievable
guarantee is complemented with a minimax lower bound that is matching up to
logarithmic factors.

    

### [[2108.08818] Discriminating modelling approaches for Point in Time Economic Scenario Generation](http://arxiv.org/abs/2108.08818)


  We introduce the notion of Point in Time Economic Scenario Generation (PiT
ESG) with a clear mathematical problem formulation to unify and compare
economic scenario generation approaches conditional on forward looking market
data. Such PiT ESGs should provide quicker and more flexible reactions to
sudden economic changes than traditional ESGs calibrated solely to long periods
of historical data. We specifically take as economic variable the S&P500 Index
with the VIX Index as forward looking market data to compare the nonparametric
filtered historical simulation, GARCH model with joint likelihood estimation
(parametric), Restricted Boltzmann Machine and the conditional Variational
Autoencoder (Generative Networks) for their suitability as PiT ESG. Our
evaluation consists of statistical tests for model fit and benchmarking the out
of sample forecasting quality with a strategy backtest using model output as
stop loss criterion. We find that both Generative Networks outperform the
nonparametric and classic parametric model in our tests, but that the CVAE
seems to be particularly well suited for our purposes: yielding more robust
performance and being computationally lighter.

    

### [[2108.08839] PoinTr: Diverse Point Cloud Completion with Geometry-Aware Transformers](http://arxiv.org/abs/2108.08839)


  Point clouds captured in real-world applications are often incomplete due to
the limited sensor resolution, single viewpoint, and occlusion. Therefore,
recovering the complete point clouds from partial ones becomes an indispensable
task in many practical applications. In this paper, we present a new method
that reformulates point cloud completion as a set-to-set translation problem
and design a new model, called PoinTr that adopts a transformer encoder-decoder
architecture for point cloud completion. By representing the point cloud as a
set of unordered groups of points with position embeddings, we convert the
point cloud to a sequence of point proxies and employ the transformers for
point cloud generation. To facilitate transformers to better leverage the
inductive bias about 3D geometric structures of point clouds, we further devise
a geometry-aware block that models the local geometric relationships
explicitly. The migration of transformers enables our model to better learn
structural knowledge and preserve detailed information for point cloud
completion. Furthermore, we propose two more challenging benchmarks with more
diverse incomplete point clouds that can better reflect the real-world
scenarios to promote future research. Experimental results show that our method
outperforms state-of-the-art methods by a large margin on both the new
benchmarks and the existing ones. Code is available at
this https URL


### [[2108.08843] Learning Equilibria in Matching Markets from Bandit Feedback](http://arxiv.org/abs/2108.08843)


  Large-scale, two-sided matching platforms must find market outcomes that
align with user preferences while simultaneously learning these preferences
from data. However, since preferences are inherently uncertain during learning,
the classical notion of stability (Gale and Shapley, 1962; Shapley and Shubik,
1971) is unattainable in these settings. To bridge this gap, we develop a
framework and algorithms for learning stable market outcomes under uncertainty.
Our primary setting is matching with transferable utilities, where the platform
both matches agents and sets monetary transfers between them. We design an
incentive-aware learning objective that captures the distance of a market
outcome from equilibrium. Using this objective, we analyze the complexity of
learning as a function of preference structure, casting learning as a
stochastic multi-armed bandit problem. Algorithmically, we show that "optimism
in the face of uncertainty," the principle underlying many bandit algorithms,
applies to a primal-dual formulation of matching with transfers and leads to
near-optimal regret bounds. Our work takes a first step toward elucidating when
and how stable matchings arise in large, data-driven marketplaces.

    

### [[1805.05052] Machine Learning: The Basics](http://arxiv.org/abs/1805.05052)


  Machine learning (ML) has become a commodity in our every-day lives. We
routinely ask ML empowered smartphones to suggest lovely food places or to
guide us through a strange place. ML methods have also become standard tools in
many fields of science and engineering. A plethora of ML applications transform
human lives at unprecedented pace and scale. This book portrays ML as the
combination of three basic components: data, model and loss. ML methods combine
these three components within computationally efficient implementations of the
basic scientific principle "trial and error". This principle consists of the
continuous adaptation of a hypothesis about a phenomenon that generates data.
ML methods use a hypothesis to compute predictions for future events. We
believe that thinking about ML as combinations of three components given by
data, model, and loss helps to navigate the steadily growing offer for
ready-to-use ML methods. Our three-component picture of ML allows a unified
treatment of a wide range of concepts and techniques which seem quite unrelated
at first sight. The regularization effect of early stopping in iterative
methods is due to the shrinking of the effective hypothesis space.
Privacy-preserving ML is obtained by particular choices for the features of
data points. Explainable ML methods are characterized by particular choices for
the hypothesis space. To make good use of ML tools it is instrumental to
understand its underlying principles at different levels of detail. On a lower
level, this tutorial helps ML engineers to choose suitable methods for the
application at hand. The book also offers a higher-level view on the
implementation of ML methods which is typically required to manage a team of ML
engineers and data scientists.

    

### [[1910.00618] Omnipush: accurate, diverse, real-world dataset of pushing dynamics with RGB-D video](http://arxiv.org/abs/1910.00618)


  Pushing is a fundamental robotic skill. Existing work has shown how to
exploit models of pushing to achieve a variety of tasks, including grasping
under uncertainty, in-hand manipulation and clearing clutter. Such models,
however, are approximate, which limits their applicability. Learning-based
methods can reason directly from raw sensory data with accuracy, and have the
potential to generalize to a wider diversity of scenarios. However, developing
and testing such methods requires rich-enough datasets. In this paper we
introduce Omnipush, a dataset with high variety of planar pushing behavior. In
particular, we provide 250 pushes for each of 250 objects, all recorded with
RGB-D and a high precision tracking system. The objects are constructed so as
to systematically explore key factors that affect pushing -- the shape of the
object and its mass distribution -- which have not been broadly explored in
previous datasets, and allow to study generalization in model learning.
Omnipush includes a benchmark for meta-learning dynamic models, which requires
algorithms that make good predictions and estimate their own uncertainty. We
also provide an RGB video prediction benchmark and propose other relevant tasks
that can be suited with this dataset.
Data and code are available at
\url{this https URL}.

    

### [[2003.04390] Meta-Baseline: Exploring Simple Meta-Learning for Few-Shot Learning](http://arxiv.org/abs/2003.04390)


  Meta-learning has been the most common framework for few-shot learning in
recent years. It learns the model from collections of few-shot classification
tasks, which is believed to have a key advantage of making the training
objective consistent with the testing objective. However, some recent works
report that by training for whole-classification, i.e. classification on the
whole label-set, it can get comparable or even better embedding than many
meta-learning algorithms. The edge between these two lines of works has yet
been underexplored, and the effectiveness of meta-learning in few-shot learning
remains unclear. In this paper, we explore a simple process: meta-learning over
a whole-classification pre-trained model on its evaluation metric. We observe
this simple method achieves competitive performance to state-of-the-art methods
on standard benchmarks. Our further analysis shed some light on understanding
the trade-offs between the meta-learning objective and the whole-classification
objective in few-shot learning.

    

### [[2004.11722] Counterfactual Learning of Stochastic Policies withContinuous Actions: from Models to Offline Evaluation](http://arxiv.org/abs/2004.11722)


  Counterfactual reasoning from logged data has become increasingly important
for many applications such as web advertising or healthcare. In this paper, we
address the problem of learning stochastic policies with continuous actions
from the viewpoint of counterfactual risk minimization (CRM). While the CRM
framework is appealing and well studied for discrete actions, the continuous
action case raises new challenges about modelization, optimization, and~offline
model selection with real data which turns out to be particularly challenging.
Our paper contributes to these three aspects of the CRM estimation pipeline.
First, we introduce a modelling strategy based on a joint kernel embedding of
contexts and actions, which overcomes the shortcomings of previous
discretization approaches. Second, we empirically show that the optimization
aspect of counterfactual learning is important, and we demonstrate the benefits
of proximal point algorithms and differentiable estimators. Finally, we propose
an evaluation protocol for offline policies in real-world logged systems, which
is challenging since policies cannot be replayed on test data, and we release a
new large-scale dataset along with multiple synthetic, yet realistic,
evaluation setups.

    

### [[2004.12571] Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning](http://arxiv.org/abs/2004.12571)


  As a decentralized model training method, federated learning is designed to
integrate the isolated data islands and protect data privacy. Recent studies,
however, have demonstrated that the Generative Adversarial Network (GAN) based
attacks can be used in federated learning to learn the distribution of the
victim's private dataset and accordingly reconstruct human-distinguishable
images. In this paper, we exploit defenses against GAN-based attacks in
federated learning, and propose a framework, Anti-GAN, to prevent attackers
from learning the real distribution of the victim's data. The core idea of
Anti-GAN is to corrupt the visual features of the victim's private training
images, such that the images restored by the attacker are indistinguishable to
human eyes. Specifically, in Anti-GAN, the victim first projects the personal
dataset onto a GAN's generator, then mixes the fake images generated by the
generator with the real images to obtain the training dataset, which will be
fed into the federated model for training. We redesign the structure of the
victim's GAN to encourage it to learn the classification features (instead of
the visual features) of the real images. We further introduce an unsupervised
task to the GAN model for obfuscating the visual features of the generated
images. The experiments demonstrate that Anti-GAN can effectively prevent the
attacker from learning the distribution of the private images, meanwhile
causing little harm to the accuracy of the federated model.

    

### [[2005.00568] Adversarial domain adaptation to reduce sample bias of a high energy physics classifier](http://arxiv.org/abs/2005.00568)


  We apply adversarial domain adaptation in unsupervised setting to reduce
sample bias in a supervised high energy physics events classifier training. We
make use of a neural network containing event and domain classifier with a
gradient reversal layer to simultaneously enable signal versus background
events classification on the one hand, while on the other hand minimising the
difference in response of the network to background samples originating from
different MC models via adversarial domain classification loss. We show the
successful bias removal on the example of simulated events at the LHC with
$t\bar{t}H$ signal versus $t\bar{t}b\bar{b}$ background classification and
discuss implications and limitations of the method

    

### [[2005.09235] On the Theoretical Properties of the Exchange Algorithm](http://arxiv.org/abs/2005.09235)


  The exchange algorithm is one of the most popular extensions of the
Metropolis--Hastings algorithm to sample from doubly-intractable distributions.
However, the theoretical exploration of the exchange algorithm is very limited.
For example, natural questions like `Does exchange algorithm converge at a
geometric rate?' or `Does the exchange algorithm admit a Central Limit
Theorem?' have not been answered yet. In this paper, we study the theoretical
properties of the exchange algorithm, in terms of asymptotic variance and
convergence speed. We compare the exchange algorithm with the original
Metropolis--Hastings algorithm and provide both necessary and sufficient
conditions for the geometric ergodicity of the exchange algorithm. Moreover, we
prove that our results can be applied to various practical applications such as
location models, Gaussian models, Poisson models, and a large class of
exponential families, which includes most of the practical applications of the
exchange algorithm. A central limit theorem for the exchange algorithm is also
established. Our results justify the theoretical usefulness of the exchange
algorithm.

    

### [[2006.07054] Learning TSP Requires Rethinking Generalization](http://arxiv.org/abs/2006.07054)


  End-to-end training of neural network solvers for combinatorial optimization
problems such as the Travelling Salesman Problem is intractable and inefficient
beyond a few hundreds of nodes. While state-of-the-art Machine Learning
approaches perform closely to classical solvers when trained on trivially small
sizes, they are unable to generalize the learnt policy to larger instances of
practical scales. Towards leveraging transfer learning to solve large-scale
TSPs, this paper identifies inductive biases, model architectures and learning
algorithms that promote generalization to instances larger than those seen in
training. Our controlled experiments provide the first principled investigation
into such zero-shot generalization, revealing that extrapolating beyond
training data requires rethinking the neural combinatorial optimization
pipeline, from network layers and learning paradigms to evaluation protocols.

    

### [[2006.10461] Auxiliary-task learning for geographic data with autoregressive embeddings](http://arxiv.org/abs/2006.10461)


  Machine learning is gaining popularity in a broad range of areas working with
geographic data, such as ecology or atmospheric sciences. Here, data often
exhibit spatial effects, which can be difficult to learn for neural networks.
In this study, we propose SXL, a method for embedding information on the
autoregressive nature of spatial data directly into the learning process using
auxiliary tasks. We utilize the local Moran's I, a popular measure of local
spatial autocorrelation, to "nudge" the model to learn the direction and
magnitude of local spatial effects, complementing the learning of the primary
task. We further introduce a novel expansion of Moran's I to multiple
resolutions, thus capturing spatial interactions over longer and shorter
distances simultaneously. The novel multi-resolution Moran's I can be
constructed easily and as a multi-dimensional tensor offers seamless
integration into existing machine learning frameworks. Throughout a range of
experiments using real-world data, we highlight how our method consistently
improves the training of neural networks in unsupervised and supervised
learning tasks. In generative spatial modeling experiments, we propose a novel
loss for auxiliary task GANs utilizing task uncertainty weights. Our proposed
method outperforms domain-specific spatial interpolation benchmarks,
highlighting its potential for downstream applications. This study bridges
expertise from geographic information science and machine learning, showing how
this integration of disciplines can help to address domain-specific challenges.
The code for our experiments is available on Github:
this https URL.

    

### [[2006.12245] Enhancing Few-Shot Image Classification with Unlabelled Examples](http://arxiv.org/abs/2006.12245)


  We develop a transductive meta-learning method that uses unlabelled instances
to improve few-shot image classification performance. Our approach combines a
regularized Mahalanobis-distance-based soft k-means clustering procedure with a
modified state of the art neural adaptive feature extractor to achieve improved
test-time classification accuracy using unlabelled data. We evaluate our method
on transductive few-shot learning tasks, in which the goal is to jointly
predict labels for query (test) examples given a set of support (training)
examples. We achieve state-of-the-art performance on the Meta-Dataset,
mini-ImageNet and tiered-ImageNet benchmarks.

    

### [[2007.03193] Policies for elementary links in a quantum network](http://arxiv.org/abs/2007.03193)


  Distributing entanglement over long distances is one of the central tasks in
quantum networks. An important problem, especially for near-term quantum
networks, is to develop optimal entanglement distribution protocols that take
into account the limitations of current and near-term hardware, such as quantum
memories with limited coherence time. We address this problem by initiating the
study of quantum network protocols for entanglement distribution using the
theory of decision processes, such that optimal protocols (referred to as
policies in the context of decision processes) can be found using dynamic
programming or reinforcement learning algorithms. As a first step, in this work
we focus exclusively on the elementary link level. We start by defining a
quantum decision process for elementary links, along with figures of merit for
evaluating policies. We then provide two algorithms for determining policies,
one of which we prove to be optimal (with respect to fidelity and success
probability) among all policies. Then we show that the previously-studied
memory-cutoff protocol can be phrased as a policy within our decision process
framework, allowing us to obtain several new fundamental results about it. The
conceptual developments and results of this work pave the way for the
systematic study of the fundamental limitations of near-term quantum networks,
and the requirements for physically realizing them.

    

### [[2007.04921] Graph Neural Network Based Coarse-Grained Mapping Prediction](http://arxiv.org/abs/2007.04921)


  The selection of coarse-grained (CG) mapping operators is a critical step for
CG molecular dynamics (MD) simulation. It is still an open question about what
is optimal for this choice and there is a need for theory. The current
state-of-the art method is mapping operators manually selected by experts. In
this work, we demonstrate an automated approach by viewing this problem as
supervised learning where we seek to reproduce the mapping operators produced
by experts. We present a graph neural network based CG mapping predictor called
DEEP SUPERVISED GRAPH PARTITIONING MODEL(DSGPM) that treats mapping operators
as a graph segmentation problem. DSGPM is trained on a novel dataset,
Human-annotated Mappings (HAM), consisting of 1,206 molecules with expert
annotated mapping operators. HAM can be used to facilitate further research in
this area. Our model uses a novel metric learning objective to produce
high-quality atomic features that are used in spectral clustering. The results
show that the DSGPM outperforms state-of-the-art methods in the field of graph
segmentation. Finally, we find that predicted CG mapping operators indeed
result in good CG MD models when used in simulation.

    

### [[2007.10253] Quantum algorithms for escaping from saddle points](http://arxiv.org/abs/2007.10253)


  We initiate the study of quantum algorithms for escaping from saddle points
with provable guarantee. Given a function $f\colon\mathbb{R}^{n}\to\mathbb{R}$,
our quantum algorithm outputs an $\epsilon$-approximate second-order stationary
point using $\tilde{O}(\log^{2} (n)/\epsilon^{1.75})$ queries to the quantum
evaluation oracle (i.e., the zeroth-order oracle). Compared to the classical
state-of-the-art algorithm by Jin et al. with $\tilde{O}(\log^{6}
(n)/\epsilon^{1.75})$ queries to the gradient oracle (i.e., the first-order
oracle), our quantum algorithm is polynomially better in terms of $\log n$ and
matches its complexity in terms of $1/\epsilon$. Technically, our main
contribution is the idea of replacing the classical perturbations in gradient
descent methods by simulating quantum wave equations, which constitutes the
improvement in the quantum query complexity with $\log n$ factors for escaping
from saddle points. We also show how to use a quantum gradient computation
algorithm due to Jordan to replace the classical gradient queries by quantum
evaluation queries with the same complexity. Finally, we also perform numerical
experiments that support our theoretical findings.

    

### [[2009.10623] Tailoring: encoding inductive biases by optimizing unsupervised objectives at prediction time](http://arxiv.org/abs/2009.10623)


  From CNNs to attention mechanisms, encoding inductive biases into neural
networks has been a fruitful source of improvement in machine learning. Adding
auxiliary losses to the main objective function is a general way of encoding
biases that can help networks learn better representations. However, since
auxiliary losses are minimized only on training data, they suffer from the same
generalization gap as regular task losses. Moreover, by adding a term to the
loss function, the model optimizes a different objective than the one we care
about. In this work we address both problems: first, we take inspiration from
\textit{transductive learning} and note that after receiving an input but
before making a prediction, we can fine-tune our networks on any unsupervised
loss. We call this process {\em tailoring}, because we customize the model to
each input to ensure our prediction satisfies the inductive bias. Second, we
formulate {\em meta-tailoring}, a nested optimization similar to that in
meta-learning, and train our models to perform well on the task objective after
adapting them using an unsupervised loss. The advantages of tailoring and
meta-tailoring are discussed theoretically and demonstrated empirically on a
diverse set of examples.

    

### [[2010.02542] Astraea: Grammar-based Fairness Testing](http://arxiv.org/abs/2010.02542)


  Software often produces biased outputs. In particular, machine learning (ML)
based software are known to produce erroneous predictions when processing
discriminatory inputs. Such unfair program behavior can be caused by societal
bias. In the last few years, Amazon, Microsoft and Google have provided
software services that produce unfair outputs, mostly due to societal bias
(e.g. gender or race). In such events, developers are saddled with the task of
conducting fairness testing. Fairness testing is challenging; developers are
tasked with generating discriminatory inputs that reveal and explain biases.
We propose a grammar-based fairness testing approach (called ASTRAEA) which
leverages context-free grammars to generate discriminatory inputs that reveal
fairness violations in software systems. Using probabilistic grammars, ASTRAEA
also provides fault diagnosis by isolating the cause of observed software bias.
ASTRAEA's diagnoses facilitate the improvement of ML fairness.
ASTRAEA was evaluated on 18 software systems that provide three major natural
language processing (NLP) services. In our evaluation, ASTRAEA generated
fairness violations with a rate of ~18%. ASTRAEA generated over 573K
discriminatory test cases and found over 102K fairness violations. Furthermore,
ASTRAEA improves software fairness by ~76%, via model-retraining.

    

### [[2010.15245] A marine radioisotope gamma-ray spectrum analysis method based on Monte Carlo simulation and MLP neural network](http://arxiv.org/abs/2010.15245)


  The monitoring of Cs-137 in seawater using scintillation detector relies on
the spectrum analysis method to extract the Cs-137 concentration. And when in
poor statistic situation, the calculation result of the traditional net peak
area (NPA) method has a large uncertainty. We present a machine learning based
method to better analyze the gamma-ray spectrum with low Cs-137 concentration.
We apply multilayer perceptron (MLP) to analyze the 662 keV full energy peak of
Cs-137 in the seawater spectrum. And the MLP can be trained with a few measured
background spectrums by combining the simulated Cs-137 signal with measured
background spectrums. Thus, it can save the time of preparing and measuring the
standard samples for generating the training dataset. To validate the MLP-based
method, we use Geant4 and background gamma-ray spectrums measured by a seaborne
monitoring device to generate an independent test dataset to test the result by
our method and the traditional NPA method. We find that the MLP-based method
achieves a root mean squared error of 0.159, 2.3 times lower than that of the
traditional net peak area method, indicating the MLP-based method improves the
precision of Cs-137 concentration calculation

    

### [[2011.03687] When Optimizing $f$-divergence is Robust with Label Noise](http://arxiv.org/abs/2011.03687)


  We show when maximizing a properly defined $f$-divergence measure with
respect to a classifier's predictions and the supervised labels is robust with
label noise. Leveraging its variational form, we derive a nice decoupling
property for a family of $f$-divergence measures when label noise presents,
where the divergence is shown to be a linear combination of the variational
difference defined on the clean distribution and a bias term introduced due to
the noise. The above derivation helps us analyze the robustness of different
$f$-divergence functions. With established robustness, this family of
$f$-divergence functions arises as useful metrics for the problem of learning
with noisy labels, which do not require the specification of the labels' noise
rate. When they are possibly not robust, we propose fixes to make them so. In
addition to the analytical results, we present thorough experimental evidence.
Our code is available at
this https URL.

    

### [[2011.11857] Augmented Lagrangian Adversarial Attacks](http://arxiv.org/abs/2011.11857)


  Adversarial attack algorithms are dominated by penalty methods, which are
slow in practice, or more efficient distance-customized methods, which are
heavily tailored to the properties of the distance considered. We propose a
white-box attack algorithm to generate minimally perturbed adversarial examples
based on Augmented Lagrangian principles. We bring several algorithmic
modifications, which have a crucial effect on performance. Our attack enjoys
the generality of penalty methods and the computational efficiency of
distance-customized algorithms, and can be readily used for a wide set of
distances. We compare our attack to state-of-the-art methods on three datasets
and several models, and consistently obtain competitive performances with
similar or lower computational complexity.

    

### [[2011.12815] Learning Multiscale Convolutional Dictionaries for Image Reconstruction](http://arxiv.org/abs/2011.12815)


  Convolutional neural networks (CNNs) have been tremendously successful in
solving imaging inverse problems. To understand their success, an effective
strategy is to construct simpler and mathematically more tractable
convolutional sparse coding (CSC) models that share essential ingredients with
CNNs. Existing CSC methods, however, underperform leading CNNs in challenging
inverse problems. We hypothesize that the performance gap may be attributed in
part to how they process images at different spatial scales: While many CNNs
use multiscale feature representations, existing CSC models mostly rely on
single-scale dictionaries. To close the performance gap, we thus propose a
multiscale convolutional dictionary structure. The proposed dictionary
structure is derived from the U-Net, arguably the most versatile and widely
used CNN for image-to-image learning problems. We show that incorporating the
proposed multiscale dictionary in an otherwise standard CSC framework yields
performance competitive with state-of-the-art CNNs across a range of
challenging inverse problems including CT and MRI reconstruction. Our work thus
demonstrates the effectiveness and scalability of the multiscale CSC approach
in solving challenging inverse problems.

    

### [[2011.13006] A Simulated Annealing Algorithm for Joint Stratification and Sample Allocation Designs](http://arxiv.org/abs/2011.13006)


  This study combines simulated annealing with delta evaluation to solve the
joint stratification and sample allocation problem. In this problem, atomic
strata are partitioned into mutually exclusive and collectively exhaustive
strata. Each stratification is a solution, the quality of which is measured by
its cost. The Bell number of possible solutions is enormous for even a moderate
number of atomic strata and an additional layer of complexity is added with the
evaluation time of each solution. Many larger scale combinatorial optimisation
problems cannot be solved to optimality because the search for an optimum
solution requires a prohibitive amount of computation time; a number of local
search heuristic algorithms have been designed for this problem but these can
become trapped in local minima preventing any further improvements. We add to
the existing suite of local search algorithms a simulated annealing algorithm
that allows for an escape from local minima and uses delta evaluation to
exploit the similarity between consecutive solutions and thereby reduce the
evaluation time. We compare the simulated annealing algorithm with two recent
algorithms. In both cases the SAA attains a solution of comparable quality in
considerably less computation time.

    

### [[2012.06985] Contrastive Learning for Label-Efficient Semantic Segmentation](http://arxiv.org/abs/2012.06985)


  Collecting labeled data for the task of semantic segmentation is expensive
and time-consuming, as it requires dense pixel-level annotations. While recent
Convolutional Neural Network (CNN) based semantic segmentation approaches have
achieved impressive results by using large amounts of labeled training data,
their performance drops significantly as the amount of labeled data decreases.
This happens because deep CNNs trained with the de facto cross-entropy loss can
easily overfit to small amounts of labeled data. To address this issue, we
propose a simple and effective contrastive learning-based training strategy in
which we first pretrain the network using a pixel-wise, label-based contrastive
loss, and then fine-tune it using the cross-entropy loss. This approach
increases intra-class compactness and inter-class separability, thereby
resulting in a better pixel classifier. We demonstrate the effectiveness of the
proposed training strategy using the Cityscapes and PASCAL VOC 2012
segmentation datasets. Our results show that pretraining with the proposed
contrastive loss results in large performance gains (more than 20% absolute
improvement in some settings) when the amount of labeled data is limited. In
many settings, the proposed contrastive pretraining strategy, which does not
use any additional data, is able to match or outperform the widely-used
ImageNet pretraining strategy that uses more than a million additional labeled
images.

    

### [[2101.00562] Few-shot Image Classification: Just Use a Library of Pre-trained Feature Extractors and a Simple Classifier](http://arxiv.org/abs/2101.00562)


  Recent papers have suggested that transfer learning can outperform
sophisticated meta-learning methods for few-shot image classification. We take
this hypothesis to its logical conclusion, and suggest the use of an ensemble
of high-quality, pre-trained feature extractors for few-shot image
classification. We show experimentally that a library of pre-trained feature
extractors combined with a simple feed-forward network learned with an
L2-regularizer can be an excellent option for solving cross-domain few-shot
image classification. Our experimental results suggest that this simpler
sample-efficient approach far outperforms several well-established
meta-learning algorithms on a variety of few-shot tasks.

    

### [[2102.07861] Low Curvature Activations Reduce Overfitting in Adversarial Training](http://arxiv.org/abs/2102.07861)


  Adversarial training is one of the most effective defenses against
adversarial attacks. Previous works suggest that overfitting is a dominant
phenomenon in adversarial training leading to a large generalization gap
between test and train accuracy in neural networks. In this work, we show that
the observed generalization gap is closely related to the choice of the
activation function. In particular, we show that using activation functions
with low (exact or approximate) curvature values has a regularization effect
that significantly reduces both the standard and robust generalization gaps in
adversarial training. We observe this effect for both differentiable/smooth
activations such as SiLU as well as non-differentiable/non-smooth activations
such as LeakyReLU. In the latter case, the "approximate" curvature of the
activation is low. Finally, we show that for activation functions with low
curvature, the double descent phenomenon for adversarially trained models does
not occur.

    

### [[2102.08446] Smoothed Analysis with Adaptive Adversaries](http://arxiv.org/abs/2102.08446)


  We prove novel algorithmic guarantees for several online problems in the
smoothed analysis model. In this model, at each time an adversary chooses an
input distribution with density function bounded above by $\tfrac{1}{\sigma}$
times that of the uniform distribution; nature then samples an input from this
distribution. Crucially, our results hold for {\em adaptive} adversaries that
can choose an input distribution based on the decisions of the algorithm and
the realizations of the inputs in the previous time steps.
This paper presents a general technique for proving smoothed algorithmic
guarantees against adaptive adversaries, in effect reducing the setting of
adaptive adversaries to the simpler case of oblivious adversaries. We apply
this technique to prove strong smoothed guarantees for three problems:
-Online learning: We consider the online prediction problem, where instances
are generated from an adaptive sequence of $\sigma$-smooth distributions and
the hypothesis class has VC dimension $d$. We bound the regret by
$\tilde{O}\big(\sqrt{T d\ln(1/\sigma)} + d\sqrt{\ln(T/\sigma)}\big)$. This
answers open questions of [RST11,Hag18].
-Online discrepancy minimization: We consider the online Komlós problem,
where the input is generated from an adaptive sequence of $\sigma$-smooth and
isotropic distributions on the $\ell_2$ unit ball. We bound the $\ell_\infty$
norm of the discrepancy vector by $\tilde{O}\big(\ln^2\!\big(
\frac{nT}{\sigma}\big) \big)$.
-Dispersion in online optimization: We consider online optimization of
piecewise Lipschitz functions where functions with $\ell$ discontinuities are
chosen by a smoothed adaptive adversary and show that the resulting sequence is
$\big( {\sigma}/{\sqrt{T\ell}}, \tilde O\big(\sqrt{T\ell}
\big)\big)$-dispersed. This matches the parameters of [BDV18] for oblivious
adversaries, up to log factors.

    

### [[2103.12424] BossNAS: Exploring Hybrid CNN-transformers with Block-wisely Self-supervised Neural Architecture Search](http://arxiv.org/abs/2103.12424)


  A myriad of recent breakthroughs in hand-crafted neural architectures for
visual recognition have highlighted the urgent need to explore hybrid
architectures consisting of diversified building blocks. Meanwhile, neural
architecture search methods are surging with an expectation to reduce human
efforts. However, whether NAS methods can efficiently and effectively handle
diversified search spaces with disparate candidates (e.g. CNNs and
transformers) is still an open question. In this work, we present Block-wisely
Self-supervised Neural Architecture Search (BossNAS), an unsupervised NAS
method that addresses the problem of inaccurate architecture rating caused by
large weight-sharing space and biased supervision in previous methods. More
specifically, we factorize the search space into blocks and utilize a novel
self-supervised training scheme, named ensemble bootstrapping, to train each
block separately before searching them as a whole towards the population
center. Additionally, we present HyTra search space, a fabric-like hybrid
CNN-transformer search space with searchable down-sampling positions. On this
challenging search space, our searched model, BossNet-T, achieves up to 82.5%
accuracy on ImageNet, surpassing EfficientNet by 2.4% with comparable compute
time. Moreover, our method achieves superior architecture rating accuracy with
0.78 and 0.76 Spearman correlation on the canonical MBConv search space with
ImageNet and on NATS-Bench size search space with CIFAR-100, respectively,
surpassing state-of-the-art NAS methods. Code:
this https URL


### [[2103.14005] Contrasting Contrastive Self-Supervised Representation Learning Pipelines](http://arxiv.org/abs/2103.14005)


  In the past few years, we have witnessed remarkable breakthroughs in
self-supervised representation learning. Despite the success and adoption of
representations learned through this paradigm, much is yet to be understood
about how different training methods and datasets influence performance on
downstream tasks. In this paper, we analyze contrastive approaches as one of
the most successful and popular variants of self-supervised representation
learning. We perform this analysis from the perspective of the training
algorithms, pre-training datasets and end tasks. We examine over 700 training
experiments including 30 encoders, 4 pre-training datasets and 20 diverse
downstream tasks. Our experiments address various questions regarding the
performance of self-supervised models compared to their supervised
counterparts, current benchmarks used for evaluation, and the effect of the
pre-training data on end task performance. Our Visual Representation Benchmark
(ViRB) is available at: this https URL.

    

### [[2104.04162] eGAN: Unsupervised approach to class imbalance using transfer learning](http://arxiv.org/abs/2104.04162)


  Class imbalance is an inherent problem in many machine learning
classification tasks. This often leads to trained models that are unusable for
any practical purpose. In this study we explore an unsupervised approach to
address these imbalances by leveraging transfer learning from pre-trained image
classification models to encoder-based Generative Adversarial Network (eGAN).
To the best of our knowledge, this is the first work to tackle this problem
using GAN without needing to augment with synthesized fake images.
In the proposed approach we use the discriminator network to output a
negative or positive score. We classify as minority, test samples with negative
scores and as majority those with positive scores. Our approach eliminates
epistemic uncertainty in model predictions, as the P(minority) + P(majority)
need not sum up to 1. The impact of transfer learning and combinations of
different pre-trained image classification models at the generator and
discriminator is also explored. Best result of 0.69 F1-score was obtained on
CIFAR-10 classification task with imbalance ratio of 1:2500.
Our approach also provides a mechanism of thresholding the specificity or
sensitivity of our machine learning system. Keywords: Class imbalance, Transfer
Learning, GAN, nash equilibrium

    

### [[2104.06405] BARF: Bundle-Adjusting Neural Radiance Fields](http://arxiv.org/abs/2104.06405)


  Neural Radiance Fields (NeRF) have recently gained a surge of interest within
the computer vision community for its power to synthesize photorealistic novel
views of real-world scenes. One limitation of NeRF, however, is its requirement
of accurate camera poses to learn the scene representations. In this paper, we
propose Bundle-Adjusting Neural Radiance Fields (BARF) for training NeRF from
imperfect (or even unknown) camera poses -- the joint problem of learning
neural 3D representations and registering camera frames. We establish a
theoretical connection to classical image alignment and show that
coarse-to-fine registration is also applicable to NeRF. Furthermore, we show
that naïvely applying positional encoding in NeRF has a negative impact on
registration with a synthesis-based objective. Experiments on synthetic and
real-world data show that BARF can effectively optimize the neural scene
representations and resolve large camera pose misalignment at the same time.
This enables view synthesis and localization of video sequences from unknown
camera poses, opening up new avenues for visual localization systems (e.g.
SLAM) and potential applications for dense 3D mapping and reconstruction.

    

### [[2104.09493] Bayesian Uncertainty and Expected Gradient Length - Regression: Two Sides Of The Same Coin?](http://arxiv.org/abs/2104.09493)


  Active learning algorithms select a subset of data for annotation to maximize
the model performance on a budget. One such algorithm is Expected Gradient
Length, which as the name suggests uses the approximate gradient induced per
example in the sampling process. While Expected Gradient Length has been
successfully used for classification and regression, the formulation for
regression remains intuitively driven. Hence, our theoretical contribution
involves deriving this formulation, thereby supporting the experimental
evidence. Subsequently, we show that expected gradient length in regression is
equivalent to Bayesian uncertainty. If certain assumptions are infeasible, our
algorithmic contribution (EGL++) approximates the effect of ensembles with a
single deterministic network. Instead of computing multiple possible inferences
per input, we leverage previously annotated samples to quantify the probability
of previous labels being the true label. Such an approach allows us to extend
expected gradient length to a new task: human pose estimation. We perform
experimental validation on two human pose datasets (MPII and LSP/LSPET),
highlighting the interpretability and competitiveness of EGL++ with different
active learning algorithms for human pose estimation.

    

### [[2105.11816] Public Transportation Demand Analysis: A Case Study of Metropolitan Lagos](http://arxiv.org/abs/2105.11816)


  Modelling, simulation, and forecasting offer a means of facilitating better
planning and decision-making. These quantitative approaches can add value
beyond traditional methods that do not rely on data and are particularly
relevant for public transportation. Lagos is experiencing rapid urbanization
and currently has a population of just under 15 million. Both long waiting
times and uncertain travel times has driven many people to acquire their own
vehicle or use alternative modes of transport. This has significantly increased
the number of vehicles on the roads leading to even more traffic and greater
traffic congestion. This paper investigates urban travel demand in Lagos and
explores passenger dynamics in time and space. Using individual commuter trip
data from tickets purchased from the Lagos State Bus Rapid Transit (BRT), the
demand patterns through the hours of the day, days of the week and bus stations
are analysed. This study aims to quantify demand from actual passenger trips
and estimate the impact that dynamic scheduling could have on passenger waiting
times. Station segmentation is provided to cluster stations by their demand
characteristics in order to tailor specific bus schedules. Intra-day public
transportation demand in Lagos BRT is analysed and predictions are compared.
Simulations using fixed and dynamic bus scheduling demonstrate that the average
waiting time could be reduced by as much as 80%. The load curves, insights and
the approach developed will be useful for informing policymaking in Lagos and
similar African cities facing the challenges of rapid urbanization.

    

### [[2106.09701] Always Be Dreaming: A New Approach for Data-Free Class-Incremental Learning](http://arxiv.org/abs/2106.09701)


  Modern computer vision applications suffer from catastrophic forgetting when
incrementally learning new concepts over time. The most successful approaches
to alleviate this forgetting require extensive replay of previously seen data,
which is problematic when memory constraints or data legality concerns exist.
In this work, we consider the high-impact problem of Data-Free
Class-Incremental Learning (DFCIL), where an incremental learning agent must
learn new concepts over time without storing generators or training data from
past tasks. One approach for DFCIL is to replay synthetic images produced by
inverting a frozen copy of the learner's classification model, but we show this
approach fails for common class-incremental benchmarks when using standard
distillation strategies. We diagnose the cause of this failure and propose a
novel incremental distillation strategy for DFCIL, contributing a modified
cross-entropy training and importance-weighted feature distillation, and show
that our method results in up to a 25.1% increase in final task accuracy
(absolute difference) compared to SOTA DFCIL methods for common
class-incremental benchmarks. Our method even outperforms several standard
replay based methods which store a coreset of images.

    

### [[2107.00761] On the Bike Spreading Problem](http://arxiv.org/abs/2107.00761)


  A free-floating bike-sharing system (FFBSS) is a dockless rental system where
an individual can borrow a bike and returns it anywhere, within the service
area. To improve the rental service, available bikes should be distributed over
the entire service area: a customer leaving from any position is then more
likely to find a near bike and then to use the service. Moreover, spreading
bikes among the entire service area increases urban spatial equity since the
benefits of FFBSS are not a prerogative of just a few zones. For guaranteeing
such distribution, the FFBSS operator can use vans to manually relocate bikes,
but it incurs high economic and environmental costs. We propose a novel
approach that exploits the existing bike flows generated by customers to
distribute bikes. More specifically, by envisioning the problem as an Influence
Maximization problem, we show that it is possible to position batches of bikes
on a small number of zones, and then the daily use of FFBSS will efficiently
spread these bikes on a large area. We show that detecting these zones is
NP-complete, but there exists a simple and efficient $1-1/e$ approximation
algorithm; our approach is then evaluated on a dataset of rides from the
free-floating bike-sharing system of the city of Padova.

    

### [[2108.08497] Monarch: A Durable Polymorphic Memory For Data Intensive Applications](http://arxiv.org/abs/2108.08497)


  3D die stacking has often been proposed to build large-scale DRAM-based
caches. Unfortunately, the power and performance overheads of DRAM limit the
efficiency of high-bandwidth memories. Also, DRAM is facing serious scalability
challenges that make alternative technologies more appealing. This paper
examines Monarch, a resistive 3D stacked memory based on a novel reconfigurable
crosspoint array called XAM. The XAM array is capable of switching between
random access and content-addressable modes, which enables Monarch (i) to
better utilize the in-package bandwidth and (ii) to satisfy both the random
access memory and associative search requirements of various applications.
Moreover, the Monarch controller ensures a given target lifetime for the
resistive stack. Our simulation results on a set of parallel memory-intensive
applications indicate that Monarch outperforms an ideal DRAM caching by 1.21x
on average. For in-memory hash table and string matching workloads, Monarch
improves performance up to 12x over the conventional high bandwidth memories.

    

### [[2108.08672] A Survey on Domain-Specific Memory Architectures](http://arxiv.org/abs/2108.08672)


  The never-ending demand for high performance and energy efficiency is pushing
designers towards an increasing level of heterogeneity and specialization in
modern computing systems. In such systems, creating efficient memory
architectures is one of the major opportunities for optimizing modern workloads
(e.g., computer vision, machine learning, graph analytics, etc.) that are
extremely data-driven. However, designers demand proper design methods to
tackle the increasing design complexity and address several new challenges,
like the security and privacy of the data to be elaborated. This paper
overviews the current trend for the design of domain-specific memory
architectures. Domain-specific architectures are tailored for the given
application domain, with the introduction of hardware accelerators and custom
memory modules while maintaining a certain level of flexibility. We describe
the major components, the common challenges, and the state-of-the-art design
methodologies for building domain-specific memory architectures. We also
discuss the most relevant research projects, providing a classification based
on our main topics.

    

### [[1010.4059] Multiplierless Modules for Forward and Backward Integer Wavelet Transform](http://arxiv.org/abs/1010.4059)


  This article is about the architecture of a lossless wavelet filter bank with
reprogrammable logic. It is based on second generation of wavelets with a
reduced of number of operations. A new basic structure for parallel
architecture and modules to forward and backward integer discrete wavelet
transform is proposed.

    

### [[2108.08395] What Distributed Systems Say: A Study of Seven Spark Application Logs](http://arxiv.org/abs/2108.08395)


  Execution logs are a crucial medium as they record runtime information of
software systems. Although extensive logs are helpful to provide valuable
details to identify the root cause in postmortem analysis in case of a failure,
this may also incur performance overhead and storage cost. Therefore, in this
research, we present the result of our experimental study on seven Spark
benchmarks to illustrate the impact of different logging verbosity levels on
the execution time and storage cost of distributed software systems. We also
evaluate the log effectiveness and the information gain values, and study the
changes in performance and the generated logs for each benchmark with various
types of distributed system failures. Our research draws insightful findings
for developers and practitioners on how to set up and utilize their distributed
systems to benefit from the execution logs.

    

### [[2108.08441] Chaos Engineering For Understanding Consensus Algorithms Performance in Permissioned Blockchains](http://arxiv.org/abs/2108.08441)


  A critical component of any blockchain or distributed ledger technology (DLT)
platform is the consensus algorithm. Blockchain consensus algorithms are the
primary vehicle for the nodes within a blockchain network to reach an
agreement. In recent years, many blockchain consensus algorithms have been
proposed mainly for private and permissioned blockchain networks. However, the
performance of these algorithms and their reliability in hostile environments
or the presence of byzantine and other network failures are not well
understood. In addition, the testing and validation of blockchain applications
come with many technical challenges. In this paper, we apply chaos engineering
and testing to understand the performance of consensus algorithms in the
presence of different loads, byzantine failure and other communication failure
scenarios. We apply chaos engineering to evaluate the performance of three
different consensus algorithms (PBFT, Clique, Raft) and their respective
blockchain platforms. We measure the blockchain network's throughput, latency,
and success rate while executing chaos and load tests. We develop lightweight
blockchain applications to execute our test in a semi-production environment.
Our results show that using chaos engineering helps understand how different
consensus algorithms perform in a hostile or unreliable environment and the
limitations of blockchain platforms. Our work demonstrates the benefits of
using chaos engineering in testing complex distributed systems such as
blockchain networks.

    

### [[2108.08541] Byzantine Cluster-Sending in Expected Constant Communication](http://arxiv.org/abs/2108.08541)


  Traditional resilient systems operate on fully-replicated fault-tolerant
clusters, which limits their scalability and performance. One way to make the
step towards resilient high-performance systems that can deal with huge
workloads, is by enabling independent fault-tolerant clusters to efficiently
communicate and cooperate with each other, as this also enables the usage of
high-performance techniques such as sharding and parallel processing. Recently,
such inter-cluster communication was formalized as the Byzantine
cluster-sending problem, and worst-case optimal protocols have been proposed
that solve this problem. Unfortunately, these protocols have an all-case linear
complexity in the size of the clusters involved.
In this paper, we propose probabilistic cluster-sending techniques that can
reliably send messages from one Byzantine fault-tolerant cluster to another
with only an expected constant message complexity, this independent of the size
of the clusters involved. Depending on the robustness of the clusters involved,
our techniques require only two-to-four message round-trips. Furthermore, our
protocols can support worst-case linear communication between clusters, which
is optimal, and deal with asynchronous and unreliable communication. As such,
our work provides a strong foundation for the further development of resilient
high-performance systems.

    

### [[2108.08545] A Nested Cross Decomposition Algorithm for Power System Capacity Expansion with Multiscale Uncertainties](http://arxiv.org/abs/2108.08545)


  Modern electric power systems have witnessed rapidly increasing penetration
of renewable energy, storage, electrical vehicles and various demand response
resources. The electric infrastructure planning is thus facing more challenges
due to the variability and uncertainties arising from the diverse new
resources. This study aims to develop a multistage and multiscale stochastic
mixed integer programming (MM-SMIP) model to capture both the
coarse-temporal-scale uncertainties, such as investment cost and long-run
demand stochasticity, and fine-temporal-scale uncertainties, such as hourly
renewable energy output and electricity demand uncertainties, for the power
system capacity expansion problem. To be applied to a real power system, the
resulting model will lead to extremely large-scale mixed integer programming
problems, which suffer not only the well-known curse of dimensionality, but
also computational difficulties with a vast number of integer variables at each
stage. In addressing such challenges associated with the MM-SMIP model, we
propose a nested cross decomposition algorithm that consists of two layers of
decomposition, that is, the Dantzig-Wolfe decomposition and L-shaped
decomposition. The algorithm exhibits promising computational performance under
our numerical study, and is especially amenable to parallel computing, which
will also be demonstrated through the computational results.

    

### [[2108.08656] Max-min Fairness Based Faucet Design for Blockchains](http://arxiv.org/abs/2108.08656)


  In order to have transactions executed and recorded on blockchains such as
the Ethereum Mainnet, fees expressed in crypto-currency units of the blockchain
must be paid. One can buy crypto-currency called Ether of the Ethereum
blockchain from exchanges and pay for the transaction fees. In the case of test
networks (such as Rinkeby) or scientific research blockchains (such as
Bloxberg), free crypto-currency, Ether, is distributed to users via faucets.
Since transaction slots on the blocks, storage and smart contract executions
are consuming blockchain resources, Ethers are distributed by fixed small
amounts to users. Users may have different amount of Ether requirements; some
small amounts and some large amounts during different times. As a result,
rather than allowing the user to get a fixed small amount of Ether, a more
general distribution mechanism that allows a user to demand and claim arbitrary
amounts of Ether, while satisfying fairness among users, is needed. For this
end, Max-min Fairness based schemes have been used in centralized settings. Our
work contributes a Max-min Fairness based algorithm and its Solidity smart
contract implementation that requires low transaction costs independent of the
number of users. This is important on the Ethereum blockchain, since a smart
contract execution with transaction costs depending on the number of users
would mean block gas limit exhaustion problem will eventually be met, making
the smart contract ineffective. We report tests which confirm that the low
transaction cost aims have been achieved by our algorithm.

    

### [[2108.08685] On the Future of Cloud Engineering](http://arxiv.org/abs/2108.08685)


  Ever since the commercial offerings of the Cloud started appearing in 2006,
the landscape of cloud computing has been undergoing remarkable changes with
the emergence of many different types of service offerings, developer
productivity enhancement tools, and new application classes as well as the
manifestation of cloud functionality closer to the user at the edge. The notion
of utility computing, however, has remained constant throughout its evolution,
which means that cloud users always seek to save costs of leasing cloud
resources while maximizing their use. On the other hand, cloud providers try to
maximize their profits while assuring service-level objectives of the
cloud-hosted applications and keeping operational costs low. All these outcomes
require systematic and sound cloud engineering principles. The aim of this
paper is to highlight the importance of cloud engineering, survey the landscape
of best practices in cloud engineering and its evolution, discuss many of the
existing cloud engineering advances, and identify both the inherent technical
challenges and research opportunities for the future of cloud computing in
general and cloud engineering in particular.

    

### [[2108.08758] Parallel Quasi-concave set optimization: A new frontier that scales without needing submodularity](http://arxiv.org/abs/2108.08758)


  Classes of set functions along with a choice of ground set are a bedrock to
determine and develop corresponding variants of greedy algorithms to obtain
efficient solutions for combinatorial optimization problems. The class of
approximate constrained submodular optimization has seen huge advances at the
intersection of good computational efficiency, versatility and approximation
guarantees while exact solutions for unconstrained submodular optimization are
NP-hard. What is an alternative to situations when submodularity does not hold?
Can efficient and globally exact solutions be obtained? We introduce one such
new frontier: The class of quasi-concave set functions induced as a dual class
to monotone linkage functions. We provide a parallel algorithm with a time
complexity over $n$ processors of $\mathcal{O}(n^2g)
+\mathcal{O}(\log{\log{n}})$ where $n$ is the cardinality of the ground set and
$g$ is the complexity to compute the monotone linkage function that induces a
corresponding quasi-concave set function via a duality. The complexity reduces
to $\mathcal{O}(gn\log(n))$ on $n^2$ processors and to $\mathcal{O}(gn)$ on
$n^3$ processors. Our algorithm provides a globally optimal solution to a
maxi-min problem as opposed to submodular optimization which is approximate. We
show a potential for widespread applications via an example of diverse feature
subset selection with exact global maxi-min guarantees upon showing that a
statistical dependency measure called distance correlation can be used to
induce a quasi-concave set function.

    

### [[2108.08796] Towards an Automatic Proof of Lamport's Paxos](http://arxiv.org/abs/2108.08796)


  Lamport's celebrated Paxos consensus protocol is generally viewed as a
complex hard-to-understand algorithm. Notwithstanding its complexity, in this
paper, we take a step towards automatically proving the safety of Paxos by
taking advantage of three structural features in its specification: spatial
regularity in its unordered domains, temporal regularity in its totally-ordered
domain, and its hierarchical composition. By carefully integrating these
structural features in IC3PO, a novel model checking algorithm, we were able to
infer an inductive invariant that identically matches the human-written one
previously derived with significant manual effort using interactive theorem
proving. While various attempts have been made to verify different versions of
Paxos, to the best of our knowledge, this is the first demonstration of an
automatically-inferred inductive invariant for Lamport's original Paxos
specification. We note that these structural features are not specific to Paxos
and that IC3PO can serve as an automatic general-purpose protocol verification
tool.

    

### [[2108.08821] Performance comparison of CFD-DEM solver MFiX-Exa, on GPUs and CPUs](http://arxiv.org/abs/2108.08821)


  We present computational performance comparisons of gas-solid simulations
performed on current CPU and GPU architectures using MFiX Exa, a CFD-DEM solver
that leverages hybrid CPU+GPU parallelism. A representative fluidized bed
simulation with varying particle numbers from 2 to 67 million is used to
compare serial and parallel performance. A single GPU was observed to be about
10 times faster compared to a single CPU core. The use of 3 GPUs on a single
compute node was observed to be 4x faster than using all 64 CPU cores. We also
observed that using an error controlled adaptive time stepping scheme for
particle advance provided a consistent 4x speed-up on both CPUs and GPUs. Weak
scaling results indicate superior parallel efficiencies when using GPUs
compared to CPUs for the problem sizes studied in this work.

    

### [[1712.00285] Locally-Iterative Distributed (Delta + 1)-Coloring below Szegedy-Vishwanathan Barrier, and Applications to Self-Stabilization and to Restricted-Bandwidth Models](http://arxiv.org/abs/1712.00285)


  We consider graph coloring and related problems in the distributed
message-passing model. {Locally-iterative algorithms} are especially important
in this setting. These are algorithms in which each vertex decides about its
next color only as a function of the current colors in its 1-hop neighborhood.
In STOC'93 Szegedy and Vishwanathan showed that any locally-iterative
(Delta+1)-coloring algorithm requires Omega(Delta log Delta + log^* n) rounds,
unless there is "a very special type of coloring that can be very efficiently
reduced" \cite{SV93}. In this paper we obtain this special type of coloring.
Specifically, we devise a locally-iterative (Delta+1)-coloring algorithm with
running time O(Delta + log^* n), i.e., {below} Szegedy-Vishwanathan barrier.
This demonstrates that this barrier is not an inherent limitation for
locally-iterative algorithms. As a result, we also achieve significant
improvements for dynamic, self-stabilizing and bandwidth-restricted settings:
- We obtain self-stabilizing distributed algorithms for
(Delta+1)-vertex-coloring, (2Delta-1)-edge-coloring, maximal independent set
and maximal matching with O(Delta+log^* n) time. This significantly improves
previously-known results that have O(n) or larger running times \cite{GK10}.
- We devise a (2Delta-1)-edge-coloring algorithm in the CONGEST model with
O(Delta + log^* n) time and in the Bit-Round model with O(Delta + log n) time.
Previously-known algorithms had superlinear dependency on Delta for
(2Delta-1)-edge-coloring in these models.
- We obtain an arbdefective coloring algorithm with running time O(\sqrt
Delta + log^* n). We employ it in order to compute proper colorings that
improve the recent state-of-the-art bounds of Barenboim from PODC'15 \cite{B15}
and Fraigniaud et al. from FOCS'16 \cite{FHK16} by polylogarithmic factors.
- Our algorithms are applicable to the SET-LOCAL model of \cite{HKMS15}.

    

### [[2106.13972] Exploring Spatial Indexing for Accelerated Feature Retrieval in HPC](http://arxiv.org/abs/2106.13972)


  Despite the critical role that range queries play in analysis and
visualization for HPC applications, there has been no comprehensive analysis of
indices that are designed to accelerate range queries and the extent to which
they are viable in an HPC setting. In this state of the practice paper we
present the first such evaluation, examining 20 open-source C and C++ libraries
that support range queries. Contributions of this paper include answering the
following questions: which of the implementations are viable in an HPC setting,
how do these libraries compare in terms of build time, query time, memory
usage, and scalability, what are other trade-offs between these
implementations, is there a single overall best solution, and when does a brute
force solution offer the best performance? We also share key insights learned
during this process that can assist both HPC application scientists and spatial
index developers.

    

### [[2108.04202] FliT: A Library for Simple and Efficient Persistent Algorithms](http://arxiv.org/abs/2108.04202)


  Non-volatile random access memory (NVRAM) offers byte-addressable persistence
at speeds comparable to DRAM. However, with caches remaining volatile,
automatic cache evictions can reorder updates to memory, potentially leaving
persistent memory in an inconsistent state upon a system crash. Flush and fence
instructions can be used to force ordering among updates, but are expensive.
This has motivated significant work studying how to write correct and efficient
persistent programs for NVRAM.
In this paper, we present FliT, a C++ library that facilitates writing
efficient persistent code. Using the library's default mode makes any
linearizable data structure durable with minimal changes to the code. FliT
avoids many redundant flush instructions by using a novel algorithm to track
dirty cache lines. The FliT library also allows for extra optimizations, but
achieves good performance even in its default setting.
To describe the FliT library's capabilities and guarantees, we define a
persistent programming interface, called the P-V Interface, which FliT
implements. The P-V Interface captures the expected behavior of code in which
some instructions' effects are persisted and some are not. We show that the
interface captures the desired semantics of many practical algorithms in the
literature.
We apply the FliT library to four different persistent data structures, and
show that across several workloads, persistence implementations, and data
structure sizes, the FliT library always improves operation throughput, by at
least $2.1\times$ over a naive implementation in all but one workload.

    

### [[2108.08297] Fact-Tree Reasoning for N-ary Question Answering over Knowledge Graphs](http://arxiv.org/abs/2108.08297)


  In the question answering(QA) task, multi-hop reasoning framework has been
extensively studied in recent years to perform more efficient and interpretable
answer reasoning on the Knowledge Graph(KG). However, multi-hop reasoning is
inapplicable for answering n-ary fact questions due to its linear reasoning
nature. We discover that there are two feasible improvements: 1) upgrade the
basic reasoning unit from entity or relation to fact; and 2) upgrade the
reasoning structure from chain to tree. Based on these, we propose a novel
fact-tree reasoning framework, through transforming the question into a fact
tree and performing iterative fact reasoning on it to predict the correct
answer. Through a comprehensive evaluation on the n-ary fact KGQA dataset
introduced by this work, we demonstrate that the proposed fact-tree reasoning
framework has the desired advantage of high answer prediction accuracy. In
addition, we also evaluate the fact-tree reasoning framework on two binary KGQA
datasets and show that our approach also has a strong reasoning ability
compared with several excellent baselines. This work has direct implications
for exploring complex reasoning scenarios and provides a preliminary baseline
approach.

    

### [[2108.08339] End-to-End License Plate Recognition Pipeline for Real-time Low Resource Video Based Applications](http://arxiv.org/abs/2108.08339)


  Automatic License Plate Recognition systems aim to provide an end-to-end
solution towards detecting, localizing, and recognizing license plate
characters from vehicles appearing in video frames. However, deploying such
systems in the real world requires real-time performance in low-resource
environments. In our paper, we propose a novel two-stage detection pipeline
paired with Vision API that aims to provide real-time inference speed along
with consistently accurate detection and recognition performance. We used a
haar-cascade classifier as a filter on top of our backbone MobileNet SSDv2
detection model. This reduces inference time by only focusing on high
confidence detections and using them for recognition. We also impose a temporal
frame separation strategy to identify multiple vehicle license plates in the
same clip. Furthermore, there are no publicly available Bangla license plate
datasets, for which we created an image dataset and a video dataset containing
license plates in the wild. We trained our models on the image dataset and
achieved an AP(0.5) score of 86% and tested our pipeline on the video dataset
and observed reasonable detection and recognition performance (82.7% detection
rate, and 60.8% OCR F1 score) with real-time processing speed (27.2 frames per
second).

    

### [[2108.08344] The Multi-Modal Video Reasoning and Analyzing Competition](http://arxiv.org/abs/2108.08344)


  In this paper, we introduce the Multi-Modal Video Reasoning and Analyzing
Competition (MMVRAC) workshop in conjunction with ICCV 2021. This competition
is composed of four different tracks, namely, video question answering,
skeleton-based action recognition, fisheye video-based action recognition, and
person re-identification, which are based on two datasets: SUTD-TrafficQA and
UAV-Human. We summarize the top-performing methods submitted by the
participants in this competition and show their results achieved in the
competition.

    

### [[2108.08423] Second-Order Specifications and Quantifier Elimination for Consistent Query Answering in Databases](http://arxiv.org/abs/2108.08423)


  Consistent answers to a query from a possibly inconsistent database are
answers that are simultaneously retrieved from every possible repair of the
database. Repairs are consistent instances that minimally differ from the
original inconsistent instance. It has been shown before that database repairs
can be specified as the stable models of a disjunctive logic program. In this
paper we show how to use the repair programs to transform the problem of
consistent query answering into a problem of reasoning w.r.t. a theory written
in second-order predicate logic. It also investigated how a first-order theory
can be obtained instead by applying second-order quantifier elimination
techniques.

    

### [[2108.08443] Semantic Reinforced Attention Learning for Visual Place Recognition](http://arxiv.org/abs/2108.08443)


  Large-scale visual place recognition (VPR) is inherently challenging because
not all visual cues in the image are beneficial to the task. In order to
highlight the task-relevant visual cues in the feature embedding, the existing
attention mechanisms are either based on artificial rules or trained in a
thorough data-driven manner. To fill the gap between the two types, we propose
a novel Semantic Reinforced Attention Learning Network (SRALNet), in which the
inferred attention can benefit from both semantic priors and data-driven
fine-tuning. The contribution lies in two-folds. (1) To suppress misleading
local features, an interpretable local weighting scheme is proposed based on
hierarchical feature distribution. (2) By exploiting the interpretability of
the local weighting scheme, a semantic constrained initialization is proposed
so that the local attention can be reinforced by semantic priors. Experiments
demonstrate that our method outperforms state-of-the-art techniques on
city-scale VPR benchmark datasets.

    

### [[2108.08470] ChMusic: A Traditional Chinese Music Dataset for Evaluation of Instrument Recognition](http://arxiv.org/abs/2108.08470)


  Musical instruments recognition is a widely used application for music
information retrieval. As most of previous musical instruments recognition
dataset focus on western musical instruments, it is difficult for researcher to
study and evaluate the area of traditional Chinese musical instrument
recognition. This paper propose a traditional Chinese music dataset for
training model and performance evaluation, named ChMusic. This dataset is free
and publicly available, 11 traditional Chinese musical instruments and 55
traditional Chinese music excerpts are recorded in this dataset. Then an
evaluation standard is proposed based on ChMusic dataset. With this standard,
researchers can compare their results following the same rule, and results from
different researchers will become comparable.

    

### [[2108.08476] Proceedings of the 1st International Workshop on Adaptive Cyber Defense](http://arxiv.org/abs/2108.08476)


  The 1st International Workshop on Adaptive Cyber Defense was held as part of
the 2021 International Joint Conference on Artificial Intelligence. This
workshop was organized to share research that explores unique applications of
Artificial Intelligence (AI) and Machine Learning (ML) as foundational
capabilities for the pursuit of adaptive cyber defense. The cyber domain cannot
currently be reliably and effectively defended without extensive reliance on
human experts. Skilled cyber defenders are in short supply and often cannot
respond fast enough to cyber threats.
Building on recent advances in AI and ML the Cyber defense research community
has been motivated to develop new dynamic and sustainable defenses through the
adoption of AI and ML techniques to both cyber and non-cyber settings. Bridging
critical gaps between AI and Cyber researchers and practitioners can accelerate
efforts to create semi-autonomous cyber defenses that can learn to recognize
and respond to cyber attacks or discover and mitigate weaknesses in cooperation
with other cyber operation systems and human experts. Furthermore, these
defenses are expected to be adaptive and able to evolve over time to thwart
changes in attacker behavior, changes in the system health and readiness, and
natural shifts in user behavior over time.
The Workshop (held on August 19th and 20th 2021 in Montreal-themed virtual
reality) was comprised of technical presentations and a panel discussion
focused on open problems and potential research solutions. Workshop submissions
were peer reviewed by a panel of domain experts with a proceedings consisting
of 10 technical articles exploring challenging problems of critical importance
to national and global security. Participation in this workshop offered new
opportunities to stimulate research and innovation in the emerging domain of
adaptive and autonomous cyber defense.

    

### [[2108.08502] A relaxed technical assumption for posterior sampling-based reinforcement learning for control of unknown linear systems](http://arxiv.org/abs/2108.08502)


  We revisit the Thompson sampling algorithm to control an unknown linear
quadratic (LQ) system recently proposed by Ouyang et al (arXiv:1709.04047). The
regret bound of the algorithm was derived under a technical assumption on the
induced norm of the closed loop system. In this technical note, we show that by
making a minor modification in the algorithm (in particular, ensuring that an
episode does not end too soon), this technical assumption on the induced norm
can be replaced by a milder assumption in terms of the spectral radius of the
closed loop system. The modified algorithm has the same Bayesian regret of
$\tilde{\mathcal{O}}(\sqrt{T})$, where $T$ is the time-horizon and the
$\tilde{\mathcal{O}}(\cdot)$ notation hides logarithmic terms in~$T$.

    

### [[2108.08570] Monitoring weeder robots and anticipating their functioning by using advanced topological data analysis](http://arxiv.org/abs/2108.08570)


  The present paper aims at analyzing the topological content of the complex
trajectories that weeder-autonomous robots follow in operation. We will prove
that the topological descriptors of these trajectories are affected by the
robot environment as well as by the robot state, with respect to maintenance
operations. Topological Data Analysis will be used for extracting the
trajectory descriptors, based on homology persistence. Then, appropriate
metrics will be applied in order to compare that topological representation of
the trajectories, for classifying them or for making efficient pattern
recognition.

    

### [[2108.08603] Forgetting Formulas and Signature Elements in Epistemic States](http://arxiv.org/abs/2108.08603)


  Delgrande's knowledge level account of forgetting provides a general approach
to forgetting syntax elements from sets of formulas with links to many other
forgetting operations, in particular, to Boole's variable elimination. On the
other hand, marginalisation of epistemic states is a specific approach to
actively reduce signatures in more complex semantic frameworks, also aiming at
forgetting atoms that is very well known from probability theory. In this
paper, we bring these two perspectives of forgetting together by showing that
marginalisation can be considered as an extension of Delgrande's approach to
the level of epistemic states. More precisely, we generalize Delgrande's axioms
of forgetting to forgetting in epistemic states, and show that marginalisation
is the most specific and informative forgetting operator that satisfies these
axioms. Moreover, we elaborate suitable phrasings of Delgrande's concept of
forgetting for formulas by transferring the basic ideas of the axioms to
forgetting formulas from epistemic states. However, here we show that this
results in trivial approaches to forgetting formulas. This finding supports the
claim that forgetting syntax elements is essentially different from belief
contraction, as e.g. axiomatized in the AGM belief change framework.

    

### [[2108.08606] Prof. Schönhage's Mysterious Machines](http://arxiv.org/abs/2108.08606)


  We give a simple Schönhage Storage Modification Machine that simulates one
iteration of the Rule 110 cellular automaton. This provides an alternative
construction to Schönhage's original proof of the Turing completeness of the
eponymous machines.

    

### [[2108.08615] Probability Estimation of Uncertain Process Traces](http://arxiv.org/abs/2108.08615)


  Process mining is a scientific discipline that analyzes event data, often
collected in databases called event logs. Recently, uncertain event logs have
become of interest, which contain non-deterministic and stochastic event
attributes that may represent many possible real-life scenarios. In this paper,
we present a method to reliably estimate the probability of each of such
scenarios, allowing their analysis. Experiments show that the probabilities
calculated with our method closely match the true chances of occurrence of
specific outcomes, enabling more trustworthy analyses on uncertain data.

    

### [[2108.08739] Neural Predictive Control for the Optimization of Smart Grid Flexibility Schedules](http://arxiv.org/abs/2108.08739)


  Model predictive control (MPC) is a method to formulate the optimal
scheduling problem for grid flexibilities in a mathematical manner. The
resulting time-constrained optimization problem can be re-solved in each
optimization time step using classical optimization methods such as Second
Order Cone Programming (SOCP) or Interior Point Methods (IPOPT). When applying
MPC in a rolling horizon scheme, the impact of uncertainty in forecasts on the
optimal schedule is reduced. While MPC methods promise accurate results for
time-constrained grid optimization they are inherently limited by the
calculation time needed for large and complex power system models. Learning the
optimal control behaviour using function approximation offers the possibility
to determine near-optimal control actions with short calculation time. A Neural
Predictive Control (NPC) scheme is proposed to learn optimal control policies
for linear and nonlinear power systems through imitation. It is demonstrated
that this procedure can find near-optimal solutions, while reducing the
calculation time by an order of magnitude. The learned controllers are
validated using a benchmark smart grid.

    

### [[2108.08815] Click to Move: Controlling Video Generation with Sparse Motion](http://arxiv.org/abs/2108.08815)


  This paper introduces Click to Move (C2M), a novel framework for video
generation where the user can control the motion of the synthesized video
through mouse clicks specifying simple object trajectories of the key objects
in the scene. Our model receives as input an initial frame, its corresponding
segmentation map and the sparse motion vectors encoding the input provided by
the user. It outputs a plausible video sequence starting from the given frame
and with a motion that is consistent with user input. Notably, our proposed
deep architecture incorporates a Graph Convolution Network (GCN) modelling the
movements of all the objects in the scene in a holistic manner and effectively
combining the sparse user motion information and image features. Experimental
results show that C2M outperforms existing methods on two publicly available
datasets, thus demonstrating the effectiveness of our GCN framework at
modelling object interactions. The source code is publicly available at
this https URL.

    

### [[1812.01825] Cooperative Multi-Agent Policy Gradients with Sub-optimal Demonstration](http://arxiv.org/abs/1812.01825)


  Many reality tasks such as robot coordination can be naturally modelled as
multi-agent cooperative system where the rewards are sparse. This paper focuses
on learning decentralized policies for such tasks using sub-optimal
demonstration. To learn the multi-agent cooperation effectively and tackle the
sub-optimality of demonstration, a self-improving learning method is proposed:
On the one hand, the centralized state-action values are initialized by the
demonstration and updated by the learned decentralized policy to improve the
sub-optimality. On the other hand, the Nash Equilibrium are found by the
current state-action value and are used as a guide to learn the policy. The
proposed method is evaluated on the combat RTS games which requires a high
level of multi-agent cooperation. Extensive experimental results on various
combat scenarios demonstrate that the proposed method can learn multi-agent
cooperation effectively. It significantly outperforms many state-of-the-art
demonstration based approaches.

    

### [[2004.03744] e-SNLI-VE: Corrected Visual-Textual Entailment with Natural Language Explanations](http://arxiv.org/abs/2004.03744)


  The recently proposed SNLI-VE corpus for recognising visual-textual
entailment is a large, real-world dataset for fine-grained multimodal
reasoning. However, the automatic way in which SNLI-VE has been assembled (via
combining parts of two related datasets) gives rise to a large number of errors
in the labels of this corpus. In this paper, we first present a data collection
effort to correct the class with the highest error rate in SNLI-VE. Secondly,
we re-evaluate an existing model on the corrected corpus, which we call
SNLI-VE-2.0, and provide a quantitative comparison with its performance on the
non-corrected corpus. Thirdly, we introduce e-SNLI-VE, which appends
human-written natural language explanations to SNLI-VE-2.0. Finally, we train
models that learn from these explanations at training time, and output such
explanations at testing time.

    

### [[2009.08644] Efficient Reinforcement Learning Development with RLzoo](http://arxiv.org/abs/2009.08644)


  Many researchers and developers are exploring for adopting Deep Reinforcement
Learning (DRL) techniques in their applications. They however often find such
an adoption challenging. Existing DRL libraries provide poor support for
prototyping DRL agents (i.e., models), customising the agents, and comparing
the performance of DRL agents. As a result, the developers often report low
efficiency in developing DRL agents. In this paper, we introduce RLzoo, a new
DRL library that aims to make the development of DRL agents efficient. RLzoo
provides developers with (i) high-level yet flexible APIs for prototyping DRL
agents, and further customising the agents for best performance, (ii) a model
zoo where users can import a wide range of DRL agents and easily compare their
performance, and (iii) an algorithm that can automatically construct DRL agents
with custom components (which are critical to improve agent's performance in
custom applications). Evaluation results show that RLzoo can effectively reduce
the development cost of DRL agents, while achieving comparable performance with
existing DRL libraries.

    

### [[2012.12111] MOCCA: Multi-Layer One-Class ClassificAtion for Anomaly Detection](http://arxiv.org/abs/2012.12111)


  Anomalies are ubiquitous in all scientific fields and can express an
unexpected event due to incomplete knowledge about the data distribution or an
unknown process that suddenly comes into play and distorts observations. Due to
such events' rarity, to train deep learning models on the Anomaly Detection
(AD) task, scientists only rely on "normal" data, i.e., non-anomalous samples.
Thus, letting the neural network infer the distribution beneath the input data.
In such a context, we propose a novel framework, named Multi-layer One-Class
ClassificAtion (MOCCA),to train and test deep learning models on the AD task.
Specifically, we applied it to autoencoders. A key novelty in our work stems
from the explicit optimization of intermediate representations for the AD task.
Indeed, differently from commonly used approaches that consider a neural
network as a single computational block, i.e., using the output of the last
layer only, MOCCA explicitly leverages the multi-layer structure of deep
architectures. Each layer's feature space is optimized for AD during training,
while in the test phase, the deep representations extracted from the trained
layers are combined to detect anomalies. With MOCCA, we split the training
process into two steps. First, the autoencoder is trained on the reconstruction
task only. Then, we only retain the encoder tasked with minimizing the L_2
distance between the output representation and a reference point, the
anomaly-free training data centroid, at each considered layer. Subsequently, we
combine the deep features extracted at the various trained layers of the
encoder model to detect anomalies at inference time. To assess the performance
of the models trained with MOCCA, we conduct extensive experiments on publicly
available datasets. We show that our proposed method reaches comparable or
superior performance to state-of-the-art approaches available in the
literature.

    

### [[2102.02558] Evolutionary Multitask Optimization: a Methodological Overview, Challenges and Future Research Directions](http://arxiv.org/abs/2102.02558)


  In this work we consider multitasking in the context of solving multiple
optimization problems simultaneously by conducting a single search process. The
principal goal when dealing with this scenario is to dynamically exploit the
existing complementarities among the problems (tasks) being optimized, helping
each other through the exchange of valuable knowledge. Additionally, the
emerging paradigm of Evolutionary Multitasking tackles multitask optimization
scenarios by using as inspiration concepts drawn from Evolutionary Computation.
The main purpose of this survey is to collect, organize and critically examine
the abundant literature published so far in Evolutionary Multitasking, with an
emphasis on the methodological patterns followed when designing new algorithmic
proposals in this area (namely, multifactorial optimization and
multipopulation-based multitasking). We complement our critical analysis with
an identification of challenges that remain open to date, along with promising
research directions that can stimulate future efforts in this topic. Our
discussions held throughout this manuscript are offered to the audience as a
reference of the general trajectory followed by the community working in this
field in recent times, as well as a self-contained entry point for newcomers
and researchers interested to join this exciting research avenue.

    

### [[2102.08777] An asymptotic analysis of probabilistic logic programming, with implications for expressing projective families of distributions](http://arxiv.org/abs/2102.08777)


  Probabilistic logic programming is a major part of statistical relational
artificial intelligence, where approaches from logic and probability are
brought together to reason about and learn from relational domains in a setting
of uncertainty. However, the behaviour of statistical relational
representations across variable domain sizes is complex, and scaling inference
and learning to large domains remains a significant challenge. In recent years,
connections have emerged between domain size dependence, lifted inference and
learning from sampled subpopulations. The asymptotic behaviour of statistical
relational representations has come under scrutiny, and projectivity was
investigated as the strongest form of domain-size dependence, in which query
marginals are completely independent of the domain size.
In this contribution we show that every probabilistic logic program under the
distribution semantics is asymptotically equivalent to an acyclic probabilistic
logic program consisting only of determinate clauses over probabilistic facts.
We conclude that every probabilistic logic program inducing a projective family
of distributions is in fact everywhere equivalent to a program from this
fragment, and we investigate the consequences for the projective families of
distributions expressible by probabilistic logic programs.
To facilitate the application of classical results from finite model theory,
we introduce the abstract distribution semantics, defined as an arbitrary
logical theory over probabilistic facts. This bridges the gap to the
distribution semantics underlying probabilistic logic programming. In this
representation, determinate logic programs correspond to quantifier-free
theories, making asymptotic quantifier elimination results available for the
setting of probabilistic logic programming.
This paper is under consideration for acceptance in TPLP.

    

### [[2105.08625] Stylized Story Generation with Style-Guided Planning](http://arxiv.org/abs/2105.08625)


  Current storytelling systems focus more ongenerating stories with coherent
plots regard-less of the narration style, which is impor-tant for controllable
text generation. There-fore, we propose a new task, stylized story gen-eration,
namely generating stories with speci-fied style given a leading context. To
tacklethe problem, we propose a novel generationmodel that first plans the
stylized keywordsand then generates the whole story with theguidance of the
keywords. Besides, we pro-pose two automatic metrics to evaluate theconsistency
between the generated story andthe specified style. Experiments
demonstratesthat our model can controllably generateemo-tion-driven
orevent-driven stories based onthe ROCStories dataset (Mostafazadeh et
al.,2016). Our study presents insights for stylizedstory generation in further
research.

    

### [[2010.15525] Self-Learning Threshold-Based Load Balancing](http://arxiv.org/abs/2010.15525)


  We consider a large-scale service system where incoming tasks have to be
instantaneously dispatched to one out of many parallel server pools. The
user-perceived performance degrades with the number of concurrent tasks and the
dispatcher aims at maximizing the overall quality-of-service by balancing the
load through a simple threshold policy. We demonstrate that such a policy is
optimal on the fluid and diffusion scales, while only involving a small
communication overhead, which is crucial for large-scale deployments. In order
to set the threshold optimally, it is important, however, to learn the load of
the system, which may be unknown. For that purpose, we design a control rule
for tuning the threshold in an online manner. We derive conditions which
guarantee that this adaptive threshold settles at the optimal value, along with
estimates for the time until this happens. In addition, we provide numerical
experiments which support the theoretical results and further indicate that our
policy copes effectively with time-varying demand patterns.

    

### [[2105.01677] A fluid simulation system based on the MPS method](http://arxiv.org/abs/2105.01677)


  Fluid flow simulation is a highly active area with applications in a wide
range of engineering problems and interactive systems. Meshless methods like
the Moving Particle Semi-implicit (MPS) are a great alternative to deal
efficiently with large deformations and free-surface flow. However, mesh-based
approaches can achieve higher numerical precision than particle-based
techniques with a performance cost. This paper presents a numerically stable
and parallelized system that benefits from advances in the literature and
parallel computing to obtain an adaptable MPS method. The proposed technique
can simulate liquids using different approaches, such as two ways to calculate
the particles' pressure, turbulent flow, and multiphase interaction. The method
is evaluated under traditional test cases presenting comparable results to
recent techniques. This work integrates the previously mentioned advances into
a single solution, which can switch on improvements, such as better momentum
conservation and less spurious pressure oscillations, through a graphical
interface. The code is entirely open-source under the GPLv3 free software
license. The GPU-accelerated code reached speedups ranging from 3 to 43 times,
depending on the total number of particles. The simulation runs at one fps for
a case with approximately 200,000 particles. Code:
this https URL


### [[2108.08464] Svar: A Tiny C++ Header Brings Unified Interface for Multiple programming Languages](http://arxiv.org/abs/2108.08464)


  There are numerous types of programming languages developed in the last
decades, and most of them provide interface to call C++ or C for high
efficiency implementation. The motivation of Svar is to design an efficient,
light-weighted and general middle-ware for multiple languages, meanwhile,
brings the dynamism features from script language to C++ in a straightforward
way. Firstly, a Svar class with JSON like data structure is designed to hold
everything exists in C++, including basic values, functions or user defined
classes and objects. Secondly, arguments are auto cast to and from Svar
efficiently with compile time pointers, references and shared\_ptr detection.
Thirdly, classes and functions are binded with string names to support
reflection, this means all functions and classes in a shared library can be
exported to a Svar object, which also calls a Svar module. The Svar modules can
be accessed by different languages and this paper demonstrates how to import
and use a Svar module in Python and Node.js. Moreover, the Svar modules or even
a python module can also be imported by C++ at runtime, which makes C++ easier
to compile and use since headers are not required anymore. We compare the
performance of Svar with two state-of-the-art binding tool for Python and
Node.js, and the result demonstrates that Svar is efficient, elegant and
general. The core of this project is one single tiny modern C++ header with
less than 5000 lines code without extra dependency. To help developers using
Svar, all the source codes related are public available on
this http URL, including documentations and benchmarks.

    

### [[2108.08683] MESH: A Memory-Efficient Safe Heap for C/C++](http://arxiv.org/abs/2108.08683)


  While memory corruption bugs stemming from the use of unsafe programming
languages are an old and well-researched problem, the resulting vulnerabilities
still dominate real-world exploitation today. Various mitigations have been
proposed to alleviate the problem, mainly in the form of language dialects,
static program analysis, and code or binary instrumentation. Solutions like
AdressSanitizer (ASan) and Softbound/CETS have proven that the latter approach
is very promising, being able to achieve memory safety without requiring manual
source code adaptions, albeit suffering substantial performance and memory
overheads. While performance overhead can be seen as a flexible constraint,
extensive memory overheads can be prohibitive for the use of such solutions in
memory-constrained environments. To address this problem, we propose MESH, a
highly memory-efficient safe heap for C/C++. With its constant, very small
memory overhead (configurable up to 2 MB on x86-64) and constant complexity for
pointer access checking, MESH offers efficient, byte-precise spatial and
temporal memory safety for memory-constrained scenarios. Without jeopardizing
the security of safe heap objects, MESH is fully compatible with existing code
and uninstrumented libraries, making it practical to use in heterogeneous
environments. We show the feasibility of our approach with a full LLVM-based
prototype supporting both major architectures, i.e., x86-64 and ARM64, in a
Linux runtime environment. Our prototype evaluation shows that, compared to
ASan and Softbound/CETS, MESH can achieve huge memory savings while preserving
similar execution performance.

    

### [[2108.08724] Programming-By-Example by Programming-By-Example: Synthesis of Looping Programs](http://arxiv.org/abs/2108.08724)


  Program synthesis has seen many new applications in recent years, in large
part thanks to the introduction of SyGuS. However, no existing SyGuS solvers
have support for synthesizing recursive functions. We introduce an multi-phase
algorithm for the synthesis of recursive ``looplike'' programs in SyGuS for
programming-by-example. We solve constraints individually and treat them as
``unrolled`` examples of how a recursive program would behave, and solve for
the generalized recursive solution. Our approach is modular and supports any
SyGuS Solver.

    

### [[2106.06205] Time Warps, from Algebra to Algorithms](http://arxiv.org/abs/2106.06205)


  Graded modalities have been proposed in recent work on programming languages
as a general framework for refining type systems with intensional properties.
In particular, continuous endomaps of the discrete time scale, or time warps,
can be used to quantify the growth of information in the course of program
execution. Time warps form a complete residuated lattice, with the residuals
playing an important role in potential programming applications. In this paper,
we study the algebraic structure of time warps, and prove that their equational
theory is decidable, a necessary condition for their use in real-world
compilers. We also describe how our universal-algebraic proof technique lends
itself to a constraint-based implementation, establishing a new link between
universal algebra and verification technology.

    