
## 2021-8-2

### [[2107.14494] Feasibility of GNSS-free Localization: A TDoA-based Approach Using LoRaWAN](http://arxiv.org/abs/2107.14494)


  LoRaWAN has garnered tremendous attention owing to the low power consumption
of end nodes, long range, high resistance to multipath, low cost, and use of
license-free sub-GHz bands. Consequently, LoRaWAN is gradually replacing Wi-Fi
and Bluetooth in sundry IoT applications including utility metering, smart
cities, and localization. Localization, in particular, has already witnessed a
surge of alternatives to Global Navigation Satellite System (GNSS), based on
Wi-Fi, Bluetooth, Ultra Wide Band, 5G, etc. in indoor and low power domains due
to the poor indoor coverage and high power consumption of GNSS. With the need
for localization only shooting up with dense IoT deployments, LoRaWAN is seen
as a promising solution in this context. Indeed, many attempts employing
various techniques such as Time of Arrival (ToA), Time Difference of Arrival
(TDoA), and Received Signal Strength Index (RSSI) have been made to achieve
localization using LoRaWAN. However, a significant drawback in this scenario is
the lack of extensive data on path loss and signal propagation modeling,
particularly in Indian cityscapes. Another demerit is the use of GNSS at some
stage primarily for time synchronization of gateways. In this work, we attempt
to nullify these two disadvantages of LoRaWAN based localization. The first
part of this work presents experimental data of LoRaWAN transmissions inside a
typical city building to study signal propagation and path loss. The latter
part proposes a standalone GNSS-free localization approach using LoRaWAN that
is achieved by applying a collaborative, TDoA-based methodology. An additional
stationary node is introduced into the network to allow the synchronization of
gateways without GNSS. Finally, the distribution of localization error in a
triangle of gateways and the effect of timing resolution, time-on-air, and duty
cycle constraints on it are investigated.

    

### [[2107.14502] Collaboration in the Sky: A Distributed Framework for Task Offloading and Resource Allocation in Multi-Access Edge Computing](http://arxiv.org/abs/2107.14502)


  Recently, unmanned aerial vehicles (UAVs) assisted multi-access edge
computing (MEC) systems emerged as a promising solution for providing
computation services to mobile users outside of terrestrial infrastructure
coverage. As each UAV operates independently, however, it is challenging to
meet the computation demands of the mobile users due to the limited computing
capacity at the UAV's MEC server as well as the UAV's energy constraint.
Therefore, collaboration among UAVs is needed. In this paper, a collaborative
multi-UAV-assisted MEC system integrated with a MEC-enabled terrestrial base
station (BS) is proposed. Then, the problem of minimizing the total latency
experienced by the mobile users in the proposed system is studied by optimizing
the offloading decision as well as the allocation of communication and
computing resources while satisfying the energy constraints of both mobile
users and UAVs. The proposed problem is shown to be a non-convex, mixed-integer
nonlinear problem (MINLP) that is intractable. Therefore, the formulated
problem is decomposed into three subproblems: i) users tasks offloading
decision problem, ii) communication resource allocation problem and iii)
UAV-assisted MEC decision problem. Then, the Lagrangian relaxation and
alternating direction method of multipliers (ADMM) methods are applied to solve
the decomposed problems, alternatively. Simulation results show that the
proposed approach reduces the average latency by up to 40.7\% and 4.3\%
compared to the greedy and exhaustive search methods.

    

### [[2107.14540] A Hierarchical and Modular Radio Resource Management Architecture for 5G and Beyond](http://arxiv.org/abs/2107.14540)


  The evolution of mobile wireless systems into Heterogeneous Networks, along
with the introduction of the 5th Generation (5G) systems, significantly
increased the complexity of radio resource management. The current mobile
networks consist of a multitude of spectrum bands, use cases, system features,
licensing schemes, radio technologies, and network layers. Additionally, the
traffic demand is uneven in terms of spatial and temporal domains, calling for
a dynamic approach to radio resource allocation. To cope with these
complexities, a generic and adaptive scheme is required for the efficient
operation of Heterogeneous Networks. This article proposes to use a
hierarchical and modular framework as an approach to cover the mentioned
challenges and to generalize this scheme to different network layers. The
proposed management solution is based on three main components: specialized
solutions for individual requirements, exposed to the coordination layer,
through abstraction middleware. In this approach, new items can be added as
plugins.

    

### [[2107.14661] Feasibility Analysis of Fifth-generation (5G) Mobile Networks for Transmission of Medical Imaging Data](http://arxiv.org/abs/2107.14661)


  Next to higher data rates and lower latency, the upcoming fifth-generation
mobile network standard will introduce a new service ecosystem. Concepts such
as multi-access edge computing or network slicing will enable tailoring service
level requirements to specific use-cases. In medical imaging, researchers and
clinicians are currently working towards higher portability of scanners. This
includes i) small scanners to be wheeled inside the hospital to the bedside and
ii) conventional scanners provided via trucks to remote areas. Both use-cases
introduce the need for mobile networks adhering to high safety standards and
providing high data rates. These requirements could be met by fifth-generation
mobile networks. In this work, we analyze the feasibility of transferring
medical imaging data using the current state of development of fifth-generation
mobile networks (3GPP Release 15). We demonstrate the potential of reaching 100
Mbit/s upload rates using already available consumer-grade hardware.
Furthermore, we show an effective average data throughput of 50 Mbit/s when
transferring medical images using out-of-the-box open-source software based on
the Digital Imaging and Communications in Medicine (DICOM) standard. During
transmissions, we sample the radio frequency bands to analyse the
characteristics of the mobile radio network. Additionally, we discuss the
potential of new features such as network slicing that will be introduced in
forthcoming releases.

    

### [[2107.14710] Advanced techniques for adaptive video streaming in SDNs](http://arxiv.org/abs/2107.14710)


  This chapter briefky describes a study on software defined networks and the
tools necessary for video transmission on this type of networks. Among the
aspects presented is the methodology used to establish the video transmission
and the procedure to evaluate the quality obtained. A video transmission
experiment is presented, which consists on the evaluation of adaptive
transmission of video streams using adaptive techniques as MPEG/DASH and
scalable coding. The experiments were carried out over a software defined
network topology in the MININET Platform

    

### [[2107.14756] Unveiling the potential of Graph Neural Networks for robust Intrusion Detection](http://arxiv.org/abs/2107.14756)


  The last few years have seen an increasing wave of attacks with serious
economic and privacy damages, which evinces the need for accurate Network
Intrusion Detection Systems (NIDS). Recent works propose the use of Machine
Learning (ML) techniques for building such systems (e.g., decision trees,
neural networks). However, existing ML-based NIDS are barely robust to common
adversarial attacks, which limits their applicability to real networks. A
fundamental problem of these solutions is that they treat and classify flows
independently. In contrast, in this paper we argue the importance of focusing
on the structural patterns of attacks, by capturing not only the individual
flow features, but also the relations between different flows (e.g., the
source/destination hosts they share). To this end, we use a graph
representation that keeps flow records and their relationships, and propose a
novel Graph Neural Network (GNN) model tailored to process and learn from such
graph-structured information. In our evaluation, we first show that the
proposed GNN model achieves state-of-the-art results in the well-known
CIC-IDS2017 dataset. Moreover, we assess the robustness of our solution under
two common adversarial attacks, that intentionally modify the packet size and
inter-arrival times to avoid detection. The results show that our model is able
to maintain the same level of accuracy as in previous experiments, while
state-of-the-art ML techniques degrade up to 50% their accuracy (F1-score)
under these attacks. This unprecedented level of robustness is mainly induced
by the capability of our GNN model to learn flow patterns of attacks structured
as graphs.

    

### [[2107.14772] Decentralized Power Allocation for MIMO-NOMA Vehicular Edge Computing Based on Deep Reinforcement Learning](http://arxiv.org/abs/2107.14772)


  Vehicular edge computing (VEC) is envisioned as a promising approach to
process the explosive computation tasks of vehicular user (VU). In the VEC
system, each VU allocates power to process partial tasks through offloading and
the remaining tasks through local execution. During the offloading, each VU
adopts the multi-input multi-out and non-orthogonal multiple access (MIMO-NOMA)
channel to improve the channel spectrum efficiency and capacity. However, the
channel condition is uncertain due to the channel interference among VUs caused
by the MIMO-NOMA channel and the time-varying path-loss caused by the mobility
of each VU. In addition, the task arrival of each VU is stochastic in the real
world. The stochastic task arrival and uncertain channel condition affect
greatly on the power consumption and latency of tasks for each VU. It is
critical to design an optimal power allocation scheme considering the
stochastic task arrival and channel variation to optimize the long-term reward
including the power consumption and latency in the MIMO-NOMA VEC. Different
from the traditional centralized deep reinforcement learning (DRL)-based
scheme, this paper constructs a decentralized DRL framework to formulate the
power allocation optimization problem, where the local observations are
selected as the state. The deep deterministic policy gradient (DDPG) algorithm
is adopted to learn the optimal power allocation scheme based on the
decentralized DRL framework. Simulation results demonstrate that our proposed
power allocation scheme outperforms the existing schemes.

    

### [[2103.05091] Learning Connectivity for Data Distribution in Robot Teams](http://arxiv.org/abs/2103.05091)


  Many algorithms for control of multi-robot teams operate under the assumption
that low-latency, global state information necessary to coordinate agent
actions can readily be disseminated among the team. However, in harsh
environments with no existing communication infrastructure, robots must form
ad-hoc networks, forcing the team to operate in a distributed fashion. To
overcome this challenge, we propose a task-agnostic, decentralized, low-latency
method for data distribution in ad-hoc networks using Graph Neural Networks
(GNN). Our approach enables multi-agent algorithms based on global state
information to function by ensuring it is available at each robot. To do this,
agents glean information about the topology of the network from packet
transmissions and feed it to a GNN running locally which instructs the agent
when and where to transmit the latest state information. We train the
distributed GNN communication policies via reinforcement learning using the
average Age of Information as the reward function and show that it improves
training stability compared to task-specific reward functions. Our approach
performs favorably compared to industry-standard methods for data distribution
such as random flooding and round robin. We also show that the trained policies
generalize to larger teams of both static and mobile agents.

    

### [[2103.16329] E-GraphSAGE: A Graph Neural Network based Intrusion Detection System](http://arxiv.org/abs/2103.16329)


  This paper presents a new Network Intrusion Detection System (NIDS) based on
Graph Neural Networks (GNNs). GNNs are a relatively new sub-field of deep
neural networks, which can leverage the inherent structure of graph-based data.
Training and evaluation data for NIDSs are typically represented as flow
records, which can naturally be represented in a graph format. This establishes
the potential and motivation for exploring GNNs for network intrusion
detection, which is the focus of this paper. Current approaches to graph
representation learning can only consider topological information and/or node
features, but not edge features. This is a key limitation for the use of
current GNN models for network intrusion detection, since critical flow
information for the detection of anomalous or malicious traffic, e.g. flow
size, flow duration, etc., is represented as edge features in a graph
representation. In this paper, we propose E-GraphSAGE, a first GNN approach
which overcomes this limitation and which allows capturing the edge features of
a graph, in addition to node features and topological information. We present a
novel NIDS based on E-GraphSAGE, and our extensive experimental evaluation on
six recent NIDS benchmark datasets shows that it outperforms the
state-of-the-art in regards to key classification metrics in four out of six
cases, and closely matches it in the other two cases. Our research and initial
basic system demonstrates the potential of GNNs for network intrusion
detection, and provides motivation for further research.

    

### [[2106.07160] Learning Intrusion Prevention Policies through Optimal Stopping](http://arxiv.org/abs/2106.07160)


  We study automated intrusion prevention using reinforcement learning. In a
novel approach, we formulate the problem of intrusion prevention as an optimal
stopping problem. This formulation allows us insight into the structure of the
optimal policies, which turn out to be threshold based. Since the computation
of the optimal defender policy using dynamic programming is not feasible for
practical cases, we approximate the optimal policy through reinforcement
learning in a simulation environment. To define the dynamics of the simulation,
we emulate the target infrastructure and collect measurements. Our evaluations
show that the learned policies are close to optimal and that they indeed can be
expressed using thresholds.

    

### [[2107.14235] EEG multipurpose eye blink detector using convolutional neural network](http://arxiv.org/abs/2107.14235)


  The electrical signal emitted by the eyes movement produces a very strong
artifact on EEG signaldue to its close proximity to the sensors and abundance
of occurrence. In the context of detectingeye blink artifacts in EEG waveforms
for further removal and signal purification, multiple strategieswhere proposed
in the literature. Most commonly applied methods require the use of a large
numberof electrodes, complex equipment for sampling and processing data. The
goal of this work is to createa reliable and user independent algorithm for
detecting and removing eye blink in EEG signals usingCNN (convolutional neural
network). For training and validation, three sets of public EEG data wereused.
All three sets contain samples obtained while the recruited subjects performed
assigned tasksthat included blink voluntarily in specific moments, watch a
video and read an article. The modelused in this study was able to have an
embracing understanding of all the features that distinguish atrivial EEG
signal from a signal contaminated with eye blink artifacts without being
overfitted byspecific features that only occurred in the situations when the
signals were registered.

    

### [[2107.14257] Modeling and Optimizing Laser-Induced Graphene](http://arxiv.org/abs/2107.14257)


  A lot of technological advances depend on next-generation materials, such as
graphene, which enables a raft of new applications, for example better
electronics. Manufacturing such materials is often difficult; in particular,
producing graphene at scale is an open problem. We provide a series of datasets
that describe the optimization of the production of laser-induced graphene, an
established manufacturing method that has shown great promise. We pose three
challenges based on the datasets we provide -- modeling the behavior of
laser-induced graphene production with respect to parameters of the production
process, transferring models and knowledge between different precursor
materials, and optimizing the outcome of the transformation over the space of
possible production parameters. We present illustrative results, along with the
code used to generate them, as a starting point for interested users. The data
we provide represents an important real-world application of machine learning;
to the best of our knowledge, no similar datasets are available.

    

### [[2107.14261] Quantifying Uncertainty for Machine Learning Based Diagnostic](http://arxiv.org/abs/2107.14261)


  Virtual Diagnostic (VD) is a deep learning tool that can be used to predict a
diagnostic output. VDs are especially useful in systems where measuring the
output is invasive, limited, costly or runs the risk of damaging the output.
Given a prediction, it is necessary to relay how reliable that prediction is.
This is known as 'uncertainty quantification' of a prediction. In this paper,
we use ensemble methods and quantile regression neural networks to explore
different ways of creating and analyzing prediction's uncertainty on
experimental data from the Linac Coherent Light Source at SLAC. We aim to
accurately and confidently predict the current profile or longitudinal phase
space images of the electron beam. The ability to make informed decisions under
uncertainty is crucial for reliable deployment of deep learning tools on
safety-critical systems as particle accelerators.

    

### [[2107.14263] Batch Active Learning at Scale](http://arxiv.org/abs/2107.14263)


  The ability to train complex and highly effective models often requires an
abundance of training data, which can easily become a bottleneck in cost, time,
and computational resources. Batch active learning, which adaptively issues
batched queries to a labeling oracle, is a common approach for addressing this
problem. The practical benefits of batch sampling come with the downside of
less adaptivity and the risk of sampling redundant examples within a batch -- a
risk that grows with the batch size. In this work, we analyze an efficient
active learning algorithm, which focuses on the large batch setting. In
particular, we show that our sampling method, which combines notions of
uncertainty and diversity, easily scales to batch sizes (100K-1M) several
orders of magnitude larger than used in previous studies and provides
significant improvements in model training efficiency compared to recent
baselines. Finally, we provide an initial theoretical analysis, proving label
complexity guarantees for a related sampling method, which we show is
approximately equivalent to our sampling method in specific settings.

    

### [[2107.14280] Deciphering Cryptic Behavior in Bimetallic Transition Metal Complexes with Machine Learning](http://arxiv.org/abs/2107.14280)


  The rational tailoring of transition metal complexes is necessary to address
outstanding challenges in energy utilization and storage. Heterobimetallic
transition metal complexes that exhibit metal-metal bonding in stacked "double
decker" ligand structures are an emerging, attractive platform for catalysis,
but their properties are challenging to predict prior to laborious synthetic
efforts. We demonstrate an alternative, data-driven approach to uncovering
structure-property relationships for rational bimetallic complex design. We
tailor graph-based representations of the metal-local environment for these
heterobimetallic complexes for use in training of multiple linear regression
and kernel ridge regression (KRR) models. Focusing on oxidation potentials, we
obtain a set of 28 experimentally characterized complexes to develop a multiple
linear regression model. On this training set, we achieve good accuracy (mean
absolute error, MAE, of 0.25 V) and preserve transferability to unseen
experimental data with a new ligand structure. We trained a KRR model on a
subset of 330 structurally characterized heterobimetallics to predict the
degree of metal-metal bonding. This KRR model predicts relative metal-metal
bond lengths in the test set to within 5%, and analysis of key features reveals
the fundamental atomic contributions (e.g., the valence electron configuration)
that most strongly influence the behavior of complexes. Our work provides
guidance for rational bimetallic design, suggesting that properties including
the formal shortness ratio should be transferable from one period to another.

    

### [[2107.14293] Self-supervised Transformer for Multivariate Clinical Time-Series with Missing Values](http://arxiv.org/abs/2107.14293)


  Multivariate time-series (MVTS) data are frequently observed in critical care
settings and are typically characterized by excessive missingness and irregular
time intervals. Existing approaches for learning representations in this domain
handle such issues by either aggregation or imputation of values, which in-turn
suppresses the fine-grained information and adds undesirable noise/overhead
into the machine learning model. To tackle this challenge, we propose STraTS
(Self-supervised Transformer for TimeSeries) model which bypasses these
pitfalls by treating time-series as a set of observation triplets instead of
using the traditional dense matrix representation. It employs a novel
Continuous Value Embedding (CVE) technique to encode continuous time and
variable values without the need for discretization. It is composed of a
Transformer component with Multi-head attention layers which enables it to
learn contextual triplet embeddings while avoiding problems of recurrence and
vanishing gradients that occur in recurrent architectures. Many healthcare
datasets also suffer from the limited availability of labeled data. Our model
utilizes self-supervision by leveraging unlabeled data to learn better
representations by performing time-series forecasting as a self-supervision
task. Experiments on real-world multivariate clinical time-series benchmark
datasets show that STraTS shows better prediction performance than
state-of-the-art methods for mortality prediction, especially when labeled data
is limited. Finally, we also present an interpretable version of STraTS which
can identify important measurements in the time-series data.

    

### [[2107.14309] Distributed Identification of Contracting and/or Monotone Network Dynamics](http://arxiv.org/abs/2107.14309)


  This paper proposes methods for identification of large-scale networked
systems with guarantees that the resulting model will be contracting -- a
strong form of nonlinear stability -- and/or monotone, i.e. order relations
between states are preserved. The main challenges that we address are:
simultaneously searching for model parameters and a certificate of stability,
and scalability to networks with hundreds or thousands of nodes. We propose a
model set that admits convex constraints for stability and monotonicity, and
has a separable structure that allows distributed identification via the
alternating directions method of multipliers (ADMM). The performance and
scalability of the approach is illustrated on a variety of linear and
non-linear case studies, including a nonlinear traffic network with a
200-dimensional state space.

    

### [[2107.14316] Survey of Recent Multi-Agent Reinforcement Learning Algorithms Utilizing Centralized Training](http://arxiv.org/abs/2107.14316)


  Much work has been dedicated to the exploration of Multi-Agent Reinforcement
Learning (MARL) paradigms implementing a centralized learning with
decentralized execution (CLDE) approach to achieve human-like collaboration in
cooperative tasks. Here, we discuss variations of centralized training and
describe a recent survey of algorithmic approaches. The goal is to explore how
different implementations of information sharing mechanism in centralized
learning may give rise to distinct group coordinated behaviors in multi-agent
systems performing cooperative tasks.

    

### [[2107.14317] Temporal Dependencies in Feature Importance for Time Series Predictions](http://arxiv.org/abs/2107.14317)


  Explanation methods applied to sequential models for multivariate time series
prediction are receiving more attention in machine learning literature. While
current methods perform well at providing instance-wise explanations, they
struggle to efficiently and accurately make attributions over long periods of
time and with complex feature interactions. We propose WinIT, a framework for
evaluating feature importance in time series prediction settings by quantifying
the shift in predictive distribution over multiple instances in a windowed
setting. Comprehensive empirical evidence shows our method improves on the
previous state-of-the-art, FIT, by capturing temporal dependencies in feature
importance. We also demonstrate how the solution improves the appropriate
attribution of features within time steps, which existing interpretability
methods often fail to do. We compare with baselines on simulated and real-world
clinical data. WinIT achieves 2.47x better performance than FIT and other
feature importance methods on real-world clinical MIMIC-mortality task. The
code for this work is available at this https URL.

    

### [[2107.14324] Deep Networks Provably Classify Data on Curves](http://arxiv.org/abs/2107.14324)


  Data with low-dimensional nonlinear structure are ubiquitous in engineering
and scientific problems. We study a model problem with such structure -- a
binary classification task that uses a deep fully-connected neural network to
classify data drawn from two disjoint smooth curves on the unit sphere. Aside
from mild regularity conditions, we place no restrictions on the configuration
of the curves. We prove that when (i) the network depth is large relative to
certain geometric properties that set the difficulty of the problem and (ii)
the network width and number of samples is polynomial in the depth,
randomly-initialized gradient descent quickly learns to correctly classify all
points on the two curves with high probability. To our knowledge, this is the
first generalization guarantee for deep networks with nonlinear data that
depends only on intrinsic data properties. Our analysis proceeds by a reduction
to dynamics in the neural tangent kernel (NTK) regime, where the network depth
plays the role of a fitting resource in solving the classification problem. In
particular, via fine-grained control of the decay properties of the NTK, we
demonstrate that when the network is sufficiently deep, the NTK can be locally
approximated by a translationally invariant operator on the manifolds and
stably inverted over smooth functions, which guarantees convergence and
generalization.

    

### [[2107.14330] Developing Open Source Educational Resources for Machine Learning and Data Science](http://arxiv.org/abs/2107.14330)


  Education should not be a privilege but a common good. It should be openly
accessible to everyone, with as few barriers as possible; even more so for key
technologies such as Machine Learning (ML) and Data Science (DS). Open
Educational Resources (OER) are a crucial factor for greater educational
equity. In this paper, we describe the specific requirements for OER in ML and
DS and argue that it is especially important for these fields to make source
files publicly available, leading to Open Source Educational Resources (OSER).
We present our view on the collaborative development of OSER, the challenges
this poses, and first steps towards their solutions. We outline how OSER can be
used for blended learning scenarios and share our experiences in university
education. Finally, we discuss additional challenges such as credit assignment
or granting certificates.

    

### [[2107.14344] Towards robust vision by multi-task learning on monkey visual cortex](http://arxiv.org/abs/2107.14344)


  Deep neural networks set the state-of-the-art across many tasks in computer
vision, but their generalization ability to image distortions is surprisingly
fragile. In contrast, the mammalian visual system is robust to a wide range of
perturbations. Recent work suggests that this generalization ability can be
explained by useful inductive biases encoded in the representations of visual
stimuli throughout the visual cortex. Here, we successfully leveraged these
inductive biases with a multi-task learning approach: we jointly trained a deep
network to perform image classification and to predict neural activity in
macaque primary visual cortex (V1). We measured the out-of-distribution
generalization abilities of our network by testing its robustness to image
distortions. We found that co-training on monkey V1 data leads to increased
robustness despite the absence of those distortions during training.
Additionally, we showed that our network's robustness is very close to that of
an Oracle network where parts of the architecture are directly trained on noisy
images. Our results also demonstrated that the network's representations become
more brain-like as their robustness improves. Using a novel constrained
reconstruction analysis, we investigated what makes our brain-regularized
network more robust. We found that our co-trained network is more sensitive to
content than noise when compared to a Baseline network that we trained for
image classification alone. Using DeepGaze-predicted saliency maps for ImageNet
images, we found that our monkey co-trained network tends to be more sensitive
to salient regions in a scene, reminiscent of existing theories on the role of
V1 in the detection of object borders and bottom-up saliency. Overall, our work
expands the promising research avenue of transferring inductive biases from the
brain, and provides a novel analysis of the effects of our transfer.

    

### [[2107.14345] Modeling User Empathy Elicited by a Robot Storyteller](http://arxiv.org/abs/2107.14345)


  Virtual and robotic agents capable of perceiving human empathy have the
potential to participate in engaging and meaningful human-machine interactions
that support human well-being. Prior research in computational empathy has
focused on designing empathic agents that use verbal and nonverbal behaviors to
simulate empathy and attempt to elicit empathic responses from humans. The
challenge of developing agents with the ability to automatically perceive
elicited empathy in humans remains largely unexplored. Our paper presents the
first approach to modeling user empathy elicited during interactions with a
robotic agent. We collected a new dataset from the novel interaction context of
participants listening to a robot storyteller (46 participants, 6.9 hours of
video). After each storytelling interaction, participants answered a
questionnaire that assessed their level of elicited empathy during the
interaction with the robot. We conducted experiments with 8 classical machine
learning models and 2 deep learning models (long short-term memory networks and
temporal convolutional networks) to detect empathy by leveraging patterns in
participants' visual behaviors while they were listening to the robot
storyteller. Our highest-performing approach, based on XGBoost, achieved an
accuracy of 69% and AUC of 72% when detecting empathy in videos. We contribute
insights regarding modeling approaches and visual features for automated
empathy detection. Our research informs and motivates future development of
empathy perception models that can be leveraged by virtual and robotic agents
during human-machine interactions.

    

### [[2107.14362] MLMOD Package: Machine Learning Methods for Data-Driven Modeling in LAMMPS](http://arxiv.org/abs/2107.14362)


  We discuss a software package for incorporating into simulations data-driven
models trained using machine learning methods. These can be used for (i)
modeling dynamics and time-step integration, (ii) modeling interactions between
system components, and (iii) computing quantities of interest characterizing
system state. The package allows for use of machine learning methods with
general model classes including Neural Networks, Gaussian Process Regression,
Kernel Models, and other approaches. We discuss in this whitepaper our
prototype C++ package, aims, and example usage.

    

### [[2107.14367] OpenSync: An opensource platform for synchronizing multiple measures in neuroscience experiments](http://arxiv.org/abs/2107.14367)


  Background: The human mind is multimodal. Yet most behavioral studies rely on
century-old measures such as task accuracy and latency. To create a better
understanding of human behavior and brain functionality, we should introduce
other measures and analyze behavior from various aspects. However, it is
technically complex and costly to design and implement the experiments that
record multiple measures. To address this issue, a platform that allows
synchronizing multiple measures from human behavior is needed. Method: This
paper introduces an opensource platform named OpenSync, which can be used to
synchronize multiple measures in neuroscience experiments. This platform helps
to automatically integrate, synchronize and record physiological measures
(e.g., electroencephalogram (EEG), galvanic skin response (GSR), eye-tracking,
body motion, etc.), user input response (e.g., from mouse, keyboard, joystick,
etc.), and task-related information (stimulus markers). In this paper, we
explain the structure and details of OpenSync, provide two case studies in
PsychoPy and Unity. Comparison with existing tools: Unlike proprietary systems
(e.g., iMotions), OpenSync is free and it can be used inside any opensource
experiment design software (e.g., PsychoPy, OpenSesame, Unity, etc.,
this https URL and
this https URL). Results: Our experimental
results show that the OpenSync platform is able to synchronize multiple
measures with microsecond resolution.

    

### [[2107.14368] Deep Quantized Representation for Enhanced Reconstruction](http://arxiv.org/abs/2107.14368)


  While machine learning approaches have shown remarkable performance in
biomedical image analysis, most of these methods rely on high-quality and
accurate imaging data. However, collecting such data requires intensive and
careful manual effort. One of the major challenges in imaging the Shoot Apical
Meristem (SAM) of Arabidopsis thaliana, is that the deeper slices in the
z-stack suffer from different perpetual quality-related problems like poor
contrast and blurring. These quality-related issues often lead to the disposal
of the painstakingly collected data with little to no control on quality while
collecting the data. Therefore, it becomes necessary to employ and design
techniques that can enhance the images to make them more suitable for further
analysis. In this paper, we propose a data-driven Deep Quantized Latent
Representation (DQLR) methodology for high-quality image reconstruction in the
Shoot Apical Meristem (SAM) of Arabidopsis thaliana. Our proposed framework
utilizes multiple consecutive slices in the z-stack to learn a low dimensional
latent space, quantize it and subsequently perform reconstruction using the
quantized representation to obtain sharper images. Experiments on a publicly
available dataset validate our methodology showing promising results.

    

### [[2107.14370] Otimizacao de pesos e funcoes de ativacao de redes neurais aplicadas na previsao de series temporais](http://arxiv.org/abs/2107.14370)


  Neural Networks have been applied for time series prediction with good
experimental results that indicate the high capacity to approximate functions
with good precision. Most neural models used in these applications use
activation functions with fixed parameters. However, it is known that the
choice of activation function strongly influences the complexity and
performance of the neural network and that a limited number of activation
functions have been used. In this work, we propose the use of a family of free
parameter asymmetric activation functions for neural networks and show that
this family of defined activation functions satisfies the requirements of the
universal approximation theorem. A methodology for the global optimization of
this family of activation functions with free parameter and the weights of the
connections between the processing units of the neural network is used. The
central idea of the proposed methodology is to simultaneously optimize the
weights and the activation function used in a multilayer perceptron network
(MLP), through an approach that combines the advantages of simulated annealing,
tabu search and a local learning algorithm, with the purpose of improving
performance in the adjustment and forecasting of time series. We chose two
learning algorithms: backpropagation with the term momentum (BPM) and
LevenbergMarquardt (LM).

    

### [[2107.14372] Using transfer learning to study burned area dynamics: A case study of refugee settlements in West Nile, Northern Uganda](http://arxiv.org/abs/2107.14372)


  With the global refugee crisis at a historic high, there is a growing need to
assess the impact of refugee settlements on their hosting countries and
surrounding environments. Because fires are an important land management
practice in smallholder agriculture in sub-Saharan Africa, burned area (BA)
mappings can help provide information about the impacts of land management
practices on local environments. However, a lack of BA ground-truth data in
much of sub-Saharan Africa limits the use of highly scalable deep learning (DL)
techniques for such BA mappings. In this work, we propose a scalable transfer
learning approach to study BA dynamics in areas with little to no ground-truth
data such as the West Nile region in Northern Uganda. We train a deep learning
model on BA ground-truth data in Portugal and propose the application of that
model on refugee-hosting districts in West Nile between 2015 and 2020. By
comparing the district-level BA dynamic with the wider West Nile region, we aim
to add understanding of the land management impacts of refugee settlements on
their surrounding environments.

    

### [[2107.14385] Random vector functional link neural network based ensemble deep learning for short-term load forecasting](http://arxiv.org/abs/2107.14385)


  Electricity load forecasting is crucial for the power systems' planning and
maintenance. However, its un-stationary and non-linear characteristics impose
significant difficulties in anticipating future demand. This paper proposes a
novel ensemble deep Random Vector Functional Link (edRVFL) network for
electricity load forecasting. The weights of hidden layers are randomly
initialized and kept fixed during the training process. The hidden layers are
stacked to enforce deep representation learning. Then, the model generates the
forecasts by ensembling the outputs of each layer. Moreover, we also propose to
augment the random enhancement features by empirical wavelet transformation
(EWT). The raw load data is decomposed by EWT in a walk-forward fashion, not
introducing future data leakage problems in the decomposition process. Finally,
all the sub-series generated by the EWT, including raw data, are fed into the
edRVFL for forecasting purposes. The proposed model is evaluated on twenty
publicly available time series from the Australian Energy Market Operator of
the year 2020. The simulation results demonstrate the proposed model's superior
performance over eleven forecasting methods in three error metrics and
statistical tests on electricity load forecasting tasks.

    

### [[2107.14398] On the interpretation of linear Riemannian tangent space model parameters in M/EEG](http://arxiv.org/abs/2107.14398)


  Riemannian tangent space methods offer state-of-the-art performance in
magnetoencephalography (MEG) and electroencephalography (EEG) based
applications such as brain-computer interfaces and biomarker development. One
limitation, particularly relevant for biomarker development, is limited model
interpretability compared to established component-based methods. Here, we
propose a method to transform the parameters of linear tangent space models
into interpretable patterns. Using typical assumptions, we show that this
approach identifies the true patterns of latent sources, encoding a target
signal. In simulations and two real MEG and EEG datasets, we demonstrate the
validity of the proposed approach and investigate its behavior when the model
assumptions are violated. Our results confirm that Riemannian tangent space
methods are robust to differences in the source patterns across observations.
We found that this robustness property also transfers to the associated
patterns.

    

### [[2107.14410] The Adaptive Multi-Factor Model and the Financial Market](http://arxiv.org/abs/2107.14410)


  Modern evolvements of the technologies have been leading to a profound
influence on the financial market. The introduction of constituents like
Exchange-Traded Funds, and the wide-use of advanced technologies such as
algorithmic trading, results in a boom of the data which provides more
opportunities to reveal deeper insights. However, traditional statistical
methods always suffer from the high-dimensional, high-correlation, and
time-varying instinct of the financial data. In this dissertation, we focus on
developing techniques to stress these difficulties. With the proposed
methodologies, we can have more interpretable models, clearer explanations, and
better predictions.

    

### [[2107.14412] Towards the Unification and Data-Driven Synthesis of Autonomous Vehicle Safety Concepts](http://arxiv.org/abs/2107.14412)


  As safety-critical autonomous vehicles (AVs) will soon become pervasive in
our society, a number of safety concepts for trusted AV deployment have been
recently proposed throughout industry and academia. Yet, agreeing upon an
"appropriate" safety concept is still an elusive task. In this paper, we
advocate for the use of Hamilton Jacobi (HJ) reachability as a unifying
mathematical framework for comparing existing safety concepts, and propose ways
to expand its modeling premises in a data-driven fashion. Specifically, we show
that (i) existing predominant safety concepts can be embedded in the HJ
reachability framework, thereby enabling a common language for comparing and
contrasting modeling assumptions, and (ii) HJ reachability can serve as an
inductive bias to effectively reason, in a data-driven context, about two
critical, yet often overlooked aspects of safety: responsibility and
context-dependency.

    

### [[2107.14417] Creating Powerful and Interpretable Models withRegression Networks](http://arxiv.org/abs/2107.14417)


  As the discipline has evolved, research in machine learning has been focused
more and more on creating more powerful neural networks, without regard for the
interpretability of these networks. Such "black-box models" yield
state-of-the-art results, but we cannot understand why they make a particular
decision or prediction. Sometimes this is acceptable, but often it is not.
We propose a novel architecture, Regression Networks, which combines the
power of neural networks with the understandability of regression analysis.
While some methods for combining these exist in the literature, our
architecture generalizes these approaches by taking interactions into account,
offering the power of a dense neural network without forsaking
interpretability. We demonstrate that the models exceed the state-of-the-art
performance of interpretable models on several benchmark datasets, matching the
power of a dense neural network. Finally, we discuss how these techniques can
be generalized to other neural architectures, such as convolutional and
recurrent neural networks.

    

### [[2107.14432] Adaptive Optimizers with Sparse Group Lasso for Neural Networks in CTR Prediction](http://arxiv.org/abs/2107.14432)


  We develop a novel framework that adds the regularizers of the sparse group
lasso to a family of adaptive optimizers in deep learning, such as Momentum,
Adagrad, Adam, AMSGrad, AdaHessian, and create a new class of optimizers, which
are named Group Momentum, Group Adagrad, Group Adam, Group AMSGrad and Group
AdaHessian, etc., accordingly. We establish theoretically proven convergence
guarantees in the stochastic convex settings, based on primal-dual methods. We
evaluate the regularized effect of our new optimizers on three large-scale
real-world ad click datasets with state-of-the-art deep learning models. The
experimental results reveal that compared with the original optimizers with the
post-processing procedure which uses the magnitude pruning method, the
performance of the models can be significantly improved on the same sparsity
level. Furthermore, in comparison to the cases without magnitude pruning, our
methods can achieve extremely high sparsity with significantly better or highly
competitive performance.

    

### [[2107.14442] Distribution free optimality intervals for clustering](http://arxiv.org/abs/2107.14442)


  We address the problem of validating the ouput of clustering algorithms.
Given data $\mathcal{D}$ and a partition $\mathcal{C}$ of these data into $K$
clusters, when can we say that the clusters obtained are correct or meaningful
for the data? This paper introduces a paradigm in which a clustering
$\mathcal{C}$ is considered meaningful if it is good with respect to a loss
function such as the K-means distortion, and stable, i.e. the only good
clustering up to small perturbations. Furthermore, we present a generic method
to obtain post-inference guarantees of near-optimality and stability for a
clustering $\mathcal{C}$. The method can be instantiated for a variety of
clustering criteria (also called loss functions) for which convex relaxations
exist. Obtaining the guarantees amounts to solving a convex optimization
problem. We demonstrate the practical relevance of this method by obtaining
guarantees for the K-means and the Normalized Cut clustering criteria on
realistic data sets. We also prove that asymptotic instability implies finite
sample instability w.h.p., allowing inferences about the population
clusterability from a sample. The guarantees do not depend on any
distributional assumptions, but they depend on the data set $\mathcal{D}$
admitting a stable clustering.

    

### [[2107.14444] Manipulating Identical Filter Redundancy for Efficient Pruning on Deep and Complicated CNN](http://arxiv.org/abs/2107.14444)


  The existence of redundancy in Convolutional Neural Networks (CNNs) enables
us to remove some filters/channels with acceptable performance drops. However,
the training objective of CNNs usually tends to minimize an accuracy-related
loss function without any attention paid to the redundancy, making the
redundancy distribute randomly on all the filters, such that removing any of
them may trigger information loss and accuracy drop, necessitating a following
finetuning step for recovery. In this paper, we propose to manipulate the
redundancy during training to facilitate network pruning. To this end, we
propose a novel Centripetal SGD (C-SGD) to make some filters identical,
resulting in ideal redundancy patterns, as such filters become purely redundant
due to their duplicates; hence removing them does not harm the network. As
shown on CIFAR and ImageNet, C-SGD delivers better performance because the
redundancy is better organized, compared to the existing methods. The
efficiency also characterizes C-SGD because it is as fast as regular SGD,
requires no finetuning, and can be conducted simultaneously on all the layers
even in very deep CNNs. Besides, C-SGD can improve the accuracy of CNNs by
first training a model with the same architecture but wider layers then
squeezing it into the original width.

    

### [[2107.14447] T-SVDNet: Exploring High-Order Prototypical Correlations for Multi-Source Domain Adaptation](http://arxiv.org/abs/2107.14447)


  Most existing domain adaptation methods focus on adaptation from only one
source domain, however, in practice there are a number of relevant sources that
could be leveraged to help improve performance on target domain. We propose a
novel approach named T-SVDNet to address the task of Multi-source Domain
Adaptation (MDA), which is featured by incorporating Tensor Singular Value
Decomposition (T-SVD) into a neural network's training pipeline. Overall,
high-order correlations among multiple domains and categories are fully
explored so as to better bridge the domain gap. Specifically, we impose
Tensor-Low-Rank (TLR) constraint on a tensor obtained by stacking up a group of
prototypical similarity matrices, aiming at capturing consistent data structure
across different domains. Furthermore, to avoid negative transfer brought by
noisy source data, we propose a novel uncertainty-aware weighting strategy to
adaptively assign weights to different source domains and samples based on the
result of uncertainty estimation. Extensive experiments conducted on public
benchmarks demonstrate the superiority of our model in addressing the task of
MDA compared to state-of-the-art methods.

    

### [[2107.14449] Synth-by-Reg (SbR): Contrastive learning for synthesis-based registration of paired images](http://arxiv.org/abs/2107.14449)


  Nonlinear inter-modality registration is often challenging due to the lack of
objective functions that are good proxies for alignment. Here we propose a
synthesis-by-registration method to convert this problem into an easier
intra-modality task. We introduce a registration loss for weakly supervised
image translation between domains that does not require perfectly aligned
training data. This loss capitalises on a registration U-Net with frozen
weights, to drive a synthesis CNN towards the desired translation. We
complement this loss with a structure preserving constraint based on
contrastive learning, which prevents blurring and content shifts due to
overfitting. We apply this method to the registration of histological sections
to MRI slices, a key step in 3D histology reconstruction. Results on two
different public datasets show improvements over registration based on mutual
information (13% reduction in landmark error) and synthesis-based algorithms
such as CycleGAN (11% reduction), and are comparable to a registration CNN with
label supervision.

    

### [[2107.14457] Maximum Entropy Dueling Network Architecture](http://arxiv.org/abs/2107.14457)


  In recent years, there have been many deep structures for Reinforcement
Learning, mainly for value function estimation and representations. These
methods achieved great success in Atari 2600 domain. In this paper, we propose
an improved architecture based upon Dueling Networks, in this architecture,
there are two separate estimators, one approximate the state value function and
the other, state advantage function. This improvement based on Maximum Entropy,
shows better policy evaluation compared to the original network and other
value-based architectures in Atari domain.

    

### [[2107.14465] Trusted-Maximizers Entropy Search for Efficient Bayesian Optimization](http://arxiv.org/abs/2107.14465)


  Information-based Bayesian optimization (BO) algorithms have achieved
state-of-the-art performance in optimizing a black-box objective function.
However, they usually require several approximations or simplifying assumptions
(without clearly understanding their effects on the BO performance) and/or
their generalization to batch BO is computationally unwieldy, especially with
an increasing batch size. To alleviate these issues, this paper presents a
novel trusted-maximizers entropy search (TES) acquisition function: It measures
how much an input query contributes to the information gain on the maximizer
over a finite set of trusted maximizers, i.e., inputs optimizing functions that
are sampled from the Gaussian process posterior belief of the objective
function. Evaluating TES requires either only a stochastic approximation with
sampling or a deterministic approximation with expectation propagation, both of
which are investigated and empirically evaluated using synthetic benchmark
objective functions and real-world optimization problems, e.g., hyperparameter
tuning of a convolutional neural network and synthesizing 'physically
realizable' faces to fool a black-box face recognition system. Though TES can
naturally be generalized to a batch variant with either approximation, the
latter is amenable to be scaled to a much larger batch size in our experiments.

    

### [[2107.14483] ManiSkill: Learning-from-Demonstrations Benchmark for Generalizable Manipulation Skills](http://arxiv.org/abs/2107.14483)


  Learning generalizable manipulation skills is central for robots to achieve
task automation in environments with endless scene and object variations.
However, existing robot learning environments are limited in both scale and
diversity of 3D assets (especially of articulated objects), making it difficult
to train and evaluate the generalization ability of agents over novel objects.
In this work, we focus on object-level generalization and propose SAPIEN
Manipulation Skill Benchmark (abbreviated as ManiSkill), a large-scale
learning-from-demonstrations benchmark for articulated object manipulation with
visual input (point cloud and image). ManiSkill supports object-level
variations by utilizing a rich and diverse set of articulated objects, and each
task is carefully designed for learning manipulations on a single category of
objects. We equip ManiSkill with high-quality demonstrations to facilitate
learning-from-demonstrations approaches and perform evaluations on common
baseline algorithms. We believe ManiSkill can encourage the robot learning
community to explore more on learning generalizable object manipulation skills.

    

### [[2107.14527] A Framework for Adversarial Streaming via Differential Privacy and Difference Estimators](http://arxiv.org/abs/2107.14527)


  Streaming algorithms are algorithms for processing large data streams, using
only a limited amount of memory. Classical streaming algorithms operate under
the assumption that the input stream is fixed in advance. Recently, there is a
growing interest in studying streaming algorithms that provide provable
guarantees even when the input stream is chosen by an adaptive adversary. Such
streaming algorithms are said to be {\em adversarially-robust}. We propose a
novel framework for adversarial streaming that hybrids two recently suggested
frameworks by Hassidim et al. (2020) and by Woodruff and Zhou (2021). These
recently suggested frameworks rely on very different ideas, each with its own
strengths and weaknesses. We combine these two frameworks (in a non-trivial
way) into a single hybrid framework that gains from both approaches to obtain
superior performances for turnstile streams.

    

### [[2107.14541] Artist Similarity with Graph Neural Networks](http://arxiv.org/abs/2107.14541)


  Artist similarity plays an important role in organizing, understanding, and
subsequently, facilitating discovery in large collections of music. In this
paper, we present a hybrid approach to computing similarity between artists
using graph neural networks trained with triplet loss. The novelty of using a
graph neural network architecture is to combine the topology of a graph of
artist connections with content features to embed artists into a vector space
that encodes similarity. To evaluate the proposed method, we compile the new
OLGA dataset, which contains artist similarities from AllMusic, together with
content features from AcousticBrainz. With 17,673 artists, this is the largest
academic artist similarity dataset that includes content-based features to
date. Moreover, we also showcase the scalability of our approach by
experimenting with a much larger proprietary dataset. Results show the
superiority of the proposed approach over current state-of-the-art methods for
music similarity. Finally, we hope that the OLGA dataset will facilitate
research on data-driven models for artist similarity.

    

### [[2107.14549] Evaluating the COVID-19 Identification ResNet (CIdeR) on the INTERSPEECH COVID-19 from Audio Challenges](http://arxiv.org/abs/2107.14549)


  We report on cross-running the recent COVID-19 Identification ResNet (CIdeR)
on the two Interspeech 2021 COVID-19 diagnosis from cough and speech audio
challenges: ComParE and DiCOVA. CIdeR is an end-to-end deep learning neural
network originally designed to classify whether an individual is COVID-positive
or COVID-negative based on coughing and breathing audio recordings from a
published crowdsourced dataset. In the current study, we demonstrate the
potential of CIdeR at binary COVID-19 diagnosis from both the COVID-19 Cough
and Speech Sub-Challenges of INTERSPEECH 2021, ComParE and DiCOVA. CIdeR
achieves significant improvements over several baselines.

    

### [[2107.14551] Sensing and Mapping for Better Roads: Initial Plan for Using Federated Learning and Implementing a Digital Twin to Identify the Road Conditions in a Developing Country -- Sri Lanka](http://arxiv.org/abs/2107.14551)


  We propose how a developing country like Sri Lanka can benefit from
privacy-enabled machine learning techniques such as Federated Learning to
detect road conditions using crowd-sourced data collection and proposed the
idea of implementing a Digital Twin for the national road system in Sri Lanka.
Developing countries such as Sri Lanka are far behind in implementing smart
road systems and smart cities compared to the developed countries. The proposed
work discussed in this paper matches the UN Sustainable Development Goal (SDG)
9: "Build Resilient Infrastructure, Promote Inclusive and Sustainable
Industrialization and Foster Innovation". Our proposed work discusses how the
government and private sector vehicles that conduct routine trips to collect
crowd-sourced data using smartphone devices to identify the road conditions and
detect where the potholes, surface unevenness (roughness), and other major
distresses are located on the roads. We explore Mobile Edge Computing (MEC)
techniques that can bring machine learning intelligence closer to the edge
devices where produced data is stored and show how the applications of
Federated Learning can be made to detect and improve road conditions. During
the second phase of this study, we plan to implement a Digital Twin for the
road system in Sri Lanka. We intend to use data provided by both Dedicated and
Non-Dedicated systems in the proposed Digital Twin for the road system. As of
writing this paper, and best to our knowledge, there is no Digital Twin system
implemented for roads and other infrastructure systems in Sri Lanka. The
proposed Digital Twin will be one of the first implementations of such systems
in Sri Lanka. Lessons learned from this pilot project will benefit other
developing countries who wish to follow the same path and make data-driven
decisions.

    

### [[2107.14561] TASK3 DCASE2021 Challenge: Sound event localization and detection using squeeze-excitation residual CNNs](http://arxiv.org/abs/2107.14561)


  Sound event localisation and detection (SELD) is a problem in the field of
automatic listening that aims at the temporal detection and localisation
(direction of arrival estimation) of sound events within an audio clip, usually
of long duration. Due to the amount of data present in the datasets related to
this problem, solutions based on deep learning have positioned themselves at
the top of the state of the art. Most solutions are based on 2D representations
of the audio (different spectrograms) that are processed by a
convolutional-recurrent network. The motivation of this submission is to study
the squeeze-excitation technique in the convolutional part of the network and
how it improves the performance of the system. This study is based on the one
carried out by the same team last year. This year, it has been decided to study
how this technique improves each of the datasets (last year only the MIC
dataset was studied). This modification shows an improvement in the performance
of the system compared to the baseline using MIC dataset.

    

### [[2107.14569] Can You Hear It? Backdoor Attacks via Ultrasonic Triggers](http://arxiv.org/abs/2107.14569)


  Deep neural networks represent a powerful option for many real-world
applications due to their ability to model even complex data relations.
However, such neural networks can also be prohibitively expensive to train,
making it common to either outsource the training process to third parties or
use pretrained neural networks. Unfortunately, such practices make neural
networks vulnerable to various attacks, where one attack is the backdoor
attack. In such an attack, the third party training the model may maliciously
inject hidden behaviors into the model. Still, if a particular input (called
trigger) is fed into a neural network, the network will respond with a wrong
result.
In this work, we explore the option of backdoor attacks to automatic speech
recognition systems where we inject inaudible triggers. By doing so, we make
the backdoor attack challenging to detect for legitimate users, and thus,
potentially more dangerous. We conduct experiments on two versions of datasets
and three neural networks and explore the performance of our attack concerning
the duration, position, and type of the trigger. Our results indicate that less
than 1% of poisoned data is sufficient to deploy a backdoor attack and reach a
100% attack success rate. What is more, while the trigger is inaudible, making
it without limitations with respect to the duration of the signal, we observed
that even short, non-continuous triggers result in highly successful attacks.

    

### [[2107.14574] Surrogate Modelling for Injection Molding Processes using Machine Learning](http://arxiv.org/abs/2107.14574)


  Injection molding is one of the most popular manufacturing methods for the
modeling of complex plastic objects. Faster numerical simulation of the
technological process would allow for faster and cheaper design cycles of new
products. In this work, we propose a baseline for a data processing pipeline
that includes the extraction of data from Moldflow simulation projects and the
prediction of the fill time and deflection distributions over 3-dimensional
surfaces using machine learning models. We propose algorithms for engineering
of features, including information of injector gates parameters that will
mostly affect the time for plastic to reach the particular point of the form
for fill time prediction, and geometrical features for deflection prediction.
We propose and evaluate baseline machine learning models for fill time and
deflection distribution prediction and provide baseline values of MSE and RMSE
metrics. Finally, we measure the execution time of our solution and show that
it significantly exceeds the time of simulation with Moldflow software:
approximately 17 times and 14 times faster for mean and median total times
respectively, comparing the times of all analysis stages for deflection
prediction. Our solution has been implemented in a prototype web application
that was approved by the management board of Fiat Chrysler Automobiles and
Illogic SRL. As one of the promising applications of this surrogate modelling
approach, we envision the use of trained models as a fast objective function in
the task of optimization of technological parameters of the injection molding
process (meaning optimal placement of gates), which could significantly aid
engineers in this task, or even automate it.

    

### [[2107.14575] DQ-SGD: Dynamic Quantization in SGD for Communication-Efficient Distributed Learning](http://arxiv.org/abs/2107.14575)


  Gradient quantization is an emerging technique in reducing communication
costs in distributed learning. Existing gradient quantization algorithms often
rely on engineering heuristics or empirical observations, lacking a systematic
approach to dynamically quantize gradients. This paper addresses this issue by
proposing a novel dynamically quantized SGD (DQ-SGD) framework, enabling us to
dynamically adjust the quantization scheme for each gradient descent step by
exploring the trade-off between communication cost and convergence error. We
derive an upper bound, tight in some cases, of the convergence error for a
restricted family of quantization schemes and loss functions. We design our
DQ-SGD algorithm via minimizing the communication cost under the convergence
error constraints. Finally, through extensive experiments on large-scale
natural language processing and computer vision tasks on AG-News, CIFAR-10, and
CIFAR-100 datasets, we demonstrate that our quantization scheme achieves better
tradeoffs between the communication cost and learning performance than other
state-of-the-art gradient quantization methods.

    

### [[2107.14582] NeuralDP Differentially private neural networks by design](http://arxiv.org/abs/2107.14582)


  The application of differential privacy to the training of deep neural
networks holds the promise of allowing large-scale (decentralized) use of
sensitive data while providing rigorous privacy guarantees to the individual.
The predominant approach to differentially private training of neural networks
is DP-SGD, which relies on norm-based gradient clipping as a method for
bounding sensitivity, followed by the addition of appropriately calibrated
Gaussian noise. In this work we propose NeuralDP, a technique for privatising
activations of some layer within a neural network, which by the post-processing
properties of differential privacy yields a differentially private network. We
experimentally demonstrate on two datasets (MNIST and Pediatric Pneumonia
Dataset (PPD)) that our method offers substantially improved privacy-utility
trade-offs compared to DP-SGD.

    

### [[2107.14586] An Efficient DP-SGD Mechanism for Large Scale NLP Models](http://arxiv.org/abs/2107.14586)


  Recent advances in deep learning have drastically improved performance on
many Natural Language Understanding (NLU) tasks. However, the data used to
train NLU models may contain private information such as addresses or phone
numbers, particularly when drawn from human subjects. It is desirable that
underlying models do not expose private information contained in the training
data. Differentially Private Stochastic Gradient Descent (DP-SGD) has been
proposed as a mechanism to build privacy-preserving models. However, DP-SGD can
be prohibitively slow to train. In this work, we propose a more efficient
DP-SGD for training using a GPU infrastructure and apply it to fine-tuning
models based on LSTM and transformer architectures. We report faster training
times, alongside accuracy, theoretical privacy guarantees and success of
Membership inference attacks for our models and observe that fine-tuning with
proposed variant of DP-SGD can yield competitive models without significant
degradation in training time and improvement in privacy protection. We also
make observations such as looser theoretical $\epsilon, \delta$ can translate
into significant practical privacy gains.

    

### [[2107.14587] Urdu & Hindi Poetry Generation using Neural Networks](http://arxiv.org/abs/2107.14587)


  One of the major problems writers and poets face is the writer's block. It is
a condition in which an author loses the ability to produce new work or
experiences a creative slowdown. The problem is more difficult in the context
of poetry than prose, as in the latter case authors need not be very concise
while expressing their ideas, also the various aspects such as rhyme, poetic
meters are not relevant for prose. One of the most effective ways to overcome
this writing block for poets can be, to have a prompt system, which would help
their imagination and open their minds for new ideas. A prompt system can
possibly generate one liner, two liner or full ghazals. The purpose of this
work is to give an ode to the Urdu, Hindi poets, and helping them start their
next line of poetry, a couplet or a complete ghazal considering various factors
like rhymes, refrain, and meters. The result will help aspiring poets to get
new ideas and help them overcome writer's block by auto-generating pieces of
poetry using Deep Learning techniques. A concern with creative works like this,
especially in the literary context, is to ensure that the output is not
plagiarized. This work also addresses the concern and makes sure that the
resulting odes are not exact match with input data using parameters like
temperature and manual plagiarism check against input corpus. To the best of
our knowledge, although the automatic text generation problem has been studied
quite extensively in the literature, the specific problem of Urdu, Hindi poetry
generation has not been explored much. Apart from developing system to
auto-generate Urdu, Hindi poetry, another key contribution of our work is to
create a cleaned and preprocessed corpus of Urdu, Hindi poetry (derived from
authentic resources) and making it freely available for researchers in the
area.

    

### [[2107.14590] Residual Tree Aggregation of Layers for Neural Machine Translation](http://arxiv.org/abs/2107.14590)


  Although attention-based Neural Machine Translation has achieved remarkable
progress in recent layers, it still suffers from issue of making insufficient
use of the output of each layer. In transformer, it only uses the top layer of
encoder and decoder in the subsequent process, which makes it impossible to
take advantage of the useful information in other layers. To address this
issue, we propose a residual tree aggregation of layers for Transformer(RTAL),
which helps to fuse information across layers. Specifically, we try to fuse the
information across layers by constructing a post-order binary tree. In
additional to the last node, we add the residual connection to the process of
generating child nodes. Our model is based on the Neural Machine Translation
model Transformer and we conduct our experiments on WMT14 English-to-German and
WMT17 English-to-France translation tasks. Experimental results across language
pairs show that the proposed approach outperforms the strong baseline model
significantly

    

### [[2107.14591] Self-supervision for health insurance claims data: a Covid-19 use case](http://arxiv.org/abs/2107.14591)


  In this work, we modify and apply self-supervision techniques to the domain
of medical health insurance claims. We model patients' healthcare claims
history analogous to free-text narratives, and introduce pre-trained `prior
knowledge', later utilized for patient outcome predictions on a challenging
task: predicting Covid-19 hospitalization, given a patient's pre-Covid-19
insurance claims history. Results suggest that pre-training on insurance claims
not only produces better prediction performance, but, more importantly,
improves the model's `clinical trustworthiness' and model
stability/reliability.

    

### [[2107.14593] Neural Variational Learning for Grounded Language Acquisition](http://arxiv.org/abs/2107.14593)


  We propose a learning system in which language is grounded in visual percepts
without specific pre-defined categories of terms. We present a unified
generative method to acquire a shared semantic/visual embedding that enables
the learning of language about a wide range of real-world objects. We evaluate
the efficacy of this learning by predicting the semantics of objects and
comparing the performance with neural and non-neural inputs. We show that this
generative approach exhibits promising results in language grounding without
pre-specifying visual categories under low resource settings. Our experiments
demonstrate that this approach is generalizable to multilingual, highly varied
datasets.

    

### [[2107.14597] Text Classification and Clustering with Annealing Soft Nearest Neighbor Loss](http://arxiv.org/abs/2107.14597)


  We define disentanglement as how far class-different data points from each
other are, relative to the distances among class-similar data points. When
maximizing disentanglement during representation learning, we obtain a
transformed feature representation where the class memberships of the data
points are preserved. If the class memberships of the data points are
preserved, we would have a feature representation space in which a nearest
neighbour classifier or a clustering algorithm would perform well. We take
advantage of this method to learn better natural language representation, and
employ it on text classification and text clustering tasks. Through
disentanglement, we obtain text representations with better-defined clusters
and improve text classification performance. Our approach had a test
classification accuracy of as high as 90.11% and test clustering accuracy of
88% on the AG News dataset, outperforming our baseline models -- without any
other training tricks or regularization.

    

### [[2107.14601] Who's Afraid of Thomas Bayes?](http://arxiv.org/abs/2107.14601)


  In many cases, neural networks perform well on test data, but tend to
overestimate their confidence on out-of-distribution data. This has led to
adoption of Bayesian neural networks, which better capture uncertainty and
therefore more accurately reflect the model's confidence. For machine learning
security researchers, this raises the natural question of how making a model
Bayesian affects the security of the model. In this work, we explore the
interplay between Bayesianism and two measures of security: model privacy and
adversarial robustness. We demonstrate that Bayesian neural networks are more
vulnerable to membership inference attacks in general, but are at least as
robust as their non-Bayesian counterparts to adversarial examples.

    

### [[2107.14608] An iterative coordinate descent algorithm to compute sparse low-rank approximations](http://arxiv.org/abs/2107.14608)


  In this paper, we describe a new algorithm to build a few sparse principal
components from a given data matrix. Our approach does not explicitly create
the covariance matrix of the data and can be viewed as an extension of the
Kogbetliantz algorithm to build an approximate singular value decomposition for
a few principal components. We show the performance of the proposed algorithm
to recover sparse principal components on various datasets from the literature
and perform dimensionality reduction for classification applications.

    

### [[2107.14613] Incorporation of Deep Neural Network & Reinforcement Learning with Domain Knowledge](http://arxiv.org/abs/2107.14613)


  We present a study of the manners by which Domain information has been
incorporated when building models with Neural Networks. Integrating space data
is uniquely important to the development of Knowledge understanding model, as
well as other fields that aid in understanding information by utilizing the
human-machine interface and Reinforcement Learning. On numerous such occasions,
machine-based model development may profit essentially from the human
information on the world encoded in an adequately exact structure. This paper
inspects expansive ways to affect encode such information as sensible and
mathematical limitations and portrays methods and results that came to a couple
of subcategories under all of those methodologies.

    

### [[2107.14642] Practical Attacks on Voice Spoofing Countermeasures](http://arxiv.org/abs/2107.14642)


  Voice authentication has become an integral part in security-critical
operations, such as bank transactions and call center conversations. The
vulnerability of automatic speaker verification systems (ASVs) to spoofing
attacks instigated the development of countermeasures (CMs), whose task is to
tell apart bonafide and spoofed speech. Together, ASVs and CMs form today's
voice authentication platforms, advertised as an impregnable access control
mechanism. We develop the first practical attack on CMs, and show how a
malicious actor may efficiently craft audio samples to bypass voice
authentication in its strictest form. Previous works have primarily focused on
non-proactive attacks or adversarial strategies against ASVs that do not
produce speech in the victim's voice. The repercussions of our attacks are far
more severe, as the samples we generate sound like the victim, eliminating any
chance of plausible deniability. Moreover, the few existing adversarial attacks
against CMs mistakenly optimize spoofed speech in the feature space and do not
take into account the existence of ASVs, resulting in inferior synthetic audio
that fails in realistic settings. We eliminate these obstacles through our key
technical contribution: a novel joint loss function that enables mounting
advanced adversarial attacks against combined ASV/CM deployments directly in
the time domain. Our adversarials achieve concerning black-box success rates
against state-of-the-art authentication platforms (up to 93.57\%). Finally, we
perform the first targeted, over-telephony-network attack on CMs, bypassing
several challenges and enabling various potential threats, given the increased
use of voice biometrics in call centers. Our results call into question the
security of modern voice authentication systems in light of the real threat of
attackers bypassing these measures to gain access to users' most valuable
resources.

    

### [[2107.14653] DadaGP: A Dataset of Tokenized GuitarPro Songs for Sequence Models](http://arxiv.org/abs/2107.14653)


  Originating in the Renaissance and burgeoning in the digital era, tablatures
are a commonly used music notation system which provides explicit
representations of instrument fingerings rather than pitches. GuitarPro has
established itself as a widely used tablature format and software enabling
musicians to edit and share songs for musical practice, learning, and
composition. In this work, we present DadaGP, a new symbolic music dataset
comprising 26,181 song scores in the GuitarPro format covering 739 musical
genres, along with an accompanying tokenized format well-suited for generative
sequence models such as the Transformer. The tokenized format is inspired by
event-based MIDI encodings, often used in symbolic music generation models. The
dataset is released with an encoder/decoder which converts GuitarPro files to
tokens and back. We present results of a use case in which DadaGP is used to
train a Transformer-based model to generate new songs in GuitarPro format. We
discuss other relevant use cases for the dataset (guitar-bass transcription,
music style transfer and artist/genre classification) as well as ethical
implications. DadaGP opens up the possibility to train GuitarPro score
generators, fine-tune models on custom data, create new styles of music,
AI-powered songwriting apps, and human-AI improvisation.

    

### [[2107.14658] Task 1A DCASE 2021: Acoustic Scene Classification with mismatch-devices using squeeze-excitation technique and low-complexity constraint](http://arxiv.org/abs/2107.14658)


  Acoustic scene classification (ASC) is one of the most popular problems in
the field of machine listening. The objective of this problem is to classify an
audio clip into one of the predefined scenes using only the audio data. This
problem has considerably progressed over the years in the different editions of
DCASE. It usually has several subtasks that allow to tackle this problem with
different approaches. The subtask presented in this report corresponds to a ASC
problem that is constrained by the complexity of the model as well as having
audio recorded from different devices, known as mismatch devices (real and
simulated). The work presented in this report follows the research line carried
out by the team in previous years. Specifically, a system based on two steps is
proposed: a two-dimensional representation of the audio using the Gamamtone
filter bank and a convolutional neural network using squeeze-excitation
techniques. The presented system outperforms the baseline by about 17
percentage points.

    

### [[2107.14664] Distributed Representations of Atoms and Materials for Machine Learning](http://arxiv.org/abs/2107.14664)


  The use of machine learning is becoming increasingly common in computational
materials science. To build effective models of the chemistry of materials,
useful machine-based representations of atoms and their compounds are required.
We derive distributed representations of compounds from their chemical formulas
only, via pooling operations of distributed representations of atoms. These
compound representations are evaluated on ten different tasks, such as the
prediction of formation energy and band gap, and are found to be competitive
with existing benchmarks that make use of structure, and even superior in cases
where only composition is available. Finally, we introduce a new approach for
learning distributed representations of atoms, named SkipAtom, which makes use
of the growing information in materials structure databases.

    

### [[2107.14682] Can non-specialists provide high quality gold standard labels in challenging modalities?](http://arxiv.org/abs/2107.14682)


  Probably yes. -- Supervised Deep Learning dominates performance scores for
many computer vision tasks and defines the state-of-the-art. However, medical
image analysis lags behind natural image applications. One of the many reasons
is the lack of well annotated medical image data available to researchers. One
of the first things researchers are told is that we require significant
expertise to reliably and accurately interpret and label such data. We see
significant inter- and intra-observer variability between expert annotations of
medical images. Still, it is a widely held assumption that novice annotators
are unable to provide useful annotations for use by clinical Deep Learning
models. In this work we challenge this assumption and examine the implications
of using a minimally trained novice labelling workforce to acquire annotations
for a complex medical image dataset. We study the time and cost implications of
using novice annotators, the raw performance of novice annotators compared to
gold-standard expert annotators, and the downstream effects on a trained Deep
Learning segmentation model's performance for detecting a specific congenital
heart disease (hypoplastic left heart syndrome) in fetal ultrasound imaging.

    

### [[2107.14695] A data-science-driven short-term analysis of Amazon, Apple, Google, and Microsoft stocks](http://arxiv.org/abs/2107.14695)


  In this paper, we implement a combination of technical analysis and
machine/deep learning-based analysis to build a trend classification model. The
goal of the paper is to apprehend short-term market movement, and incorporate
it to improve the underlying stochastic model. Also, the analysis presented in
this paper can be implemented in a \emph{model-independent} fashion. We execute
a data-science-driven technique that makes short-term forecasts dependent on
the price trends of current stock market data. Based on the analysis, three
different labels are generated for a data set: $+1$ (buy signal), $0$ (hold
signal), or $-1$ (sell signal). We propose a detailed analysis of four major
stocks- Amazon, Apple, Google, and Microsoft. We implement various technical
indicators to label the data set according to the trend and train various
models for trend estimation. Statistical analysis of the outputs and
classification results are obtained.

    

### [[2107.14698] Strategically Efficient Exploration in Competitive Multi-agent Reinforcement Learning](http://arxiv.org/abs/2107.14698)


  High sample complexity remains a barrier to the application of reinforcement
learning (RL), particularly in multi-agent systems. A large body of work has
demonstrated that exploration mechanisms based on the principle of optimism
under uncertainty can significantly improve the sample efficiency of RL in
single agent tasks. This work seeks to understand the role of optimistic
exploration in non-cooperative multi-agent settings. We will show that, in
zero-sum games, optimistic exploration can cause the learner to waste time
sampling parts of the state space that are irrelevant to strategic play, as
they can only be reached through cooperation between both players. To address
this issue, we introduce a formal notion of strategically efficient exploration
in Markov games, and use this to develop two strategically efficient learning
algorithms for finite Markov games. We demonstrate that these methods can be
significantly more sample efficient than their optimistic counterparts.

    

### [[2107.14702] Towards General Function Approximation in Zero-Sum Markov Games](http://arxiv.org/abs/2107.14702)


  This paper considers two-player zero-sum finite-horizon Markov games with
simultaneous moves. The study focuses on the challenging settings where the
value function or the model is parameterized by general function classes.
Provably efficient algorithms for both decoupled and {coordinated} settings are
developed. In the {decoupled} setting where the agent controls a single player
and plays against an arbitrary opponent, we propose a new model-free algorithm.
The sample complexity is governed by the Minimax Eluder dimension -- a new
dimension of the function class in Markov games. As a special case, this method
improves the state-of-the-art algorithm by a $\sqrt{d}$ factor in the regret
when the reward function and transition kernel are parameterized with
$d$-dimensional linear features. In the {coordinated} setting where both
players are controlled by the agent, we propose a model-based algorithm and a
model-free algorithm. In the model-based algorithm, we prove that sample
complexity can be bounded by a generalization of Witness rank to Markov games.
The model-free algorithm enjoys a $\sqrt{K}$-regret upper bound where $K$ is
the number of episodes. Our algorithms are based on new techniques of alternate
optimism.

    

### [[2107.14707] When Deep Learners Change Their Mind: Learning Dynamics for Active Learning](http://arxiv.org/abs/2107.14707)


  Active learning aims to select samples to be annotated that yield the largest
performance improvement for the learning algorithm. Many methods approach this
problem by measuring the informativeness of samples and do this based on the
certainty of the network predictions for samples. However, it is well-known
that neural networks are overly confident about their prediction and are
therefore an untrustworthy source to assess sample informativeness. In this
paper, we propose a new informativeness-based active learning method. Our
measure is derived from the learning dynamics of a neural network. More
precisely we track the label assignment of the unlabeled data pool during the
training of the algorithm. We capture the learning dynamics with a metric
called label-dispersion, which is low when the network consistently assigns the
same label to the sample during the training of the network and high when the
assigned label changes frequently. We show that label-dispersion is a promising
predictor of the uncertainty of the network, and show on two benchmark datasets
that an active learning algorithm based on label-dispersion obtains excellent
results.

    

### [[2107.14742] Connections between Numerical Algorithms for PDEs and Neural Networks](http://arxiv.org/abs/2107.14742)


  We investigate numerous structural connections between numerical algorithms
for partial differential equations (PDEs) and neural architectures. Our goal is
to transfer the rich set of mathematical foundations from the world of PDEs to
neural networks. Besides structural insights we provide concrete examples and
experimental evaluations of the resulting architectures. Using the example of
generalised nonlinear diffusion in 1D, we consider explicit schemes,
acceleration strategies thereof, implicit schemes, and multigrid approaches. We
connect these concepts to residual networks, recurrent neural networks, and
U-net architectures. Our findings inspire a symmetric residual network design
with provable stability guarantees and justify the effectiveness of skip
connections in neural networks from a numerical perspective. Moreover, we
present U-net architectures that implement multigrid techniques for learning
efficient solutions of partial differential equation models, and motivate
uncommon design choices such as trainable nonmonotone activation functions.
Experimental evaluations show that the proposed architectures save half of the
trainable parameters and can thus outperform standard ones with the same model
complexity. Our considerations serve as a basis for explaining the success of
popular neural architectures and provide a blueprint for developing new
mathematically well-founded neural building blocks.

    

### [[2107.14747] A common variable minimax theorem for graphs](http://arxiv.org/abs/2107.14747)


  Let $\mathcal{G} = \{G_1 = (V, E_1), \dots, G_m = (V, E_m)\}$ be a collection
of $m$ graphs defined on a common set of vertices $V$ but with different edge
sets $E_1, \dots, E_m$. Informally, a function $f :V \rightarrow \mathbb{R}$ is
smooth with respect to $G_k = (V,E_k)$ if $f(u) \sim f(v)$ whenever $(u, v) \in
E_k$. We study the problem of understanding whether there exists a nonconstant
function that is smooth with respect to all graphs in $\mathcal{G}$,
simultaneously, and how to find it if it exists.

    

### [[2107.14759] Tiny Machine Learning for Concept Drift](http://arxiv.org/abs/2107.14759)


  Tiny Machine Learning (TML) is a new research area whose goal is to design
machine and deep learning techniques able to operate in Embedded Systems and
IoT units, hence satisfying the severe technological constraints on memory,
computation, and energy characterizing these pervasive devices. Interestingly,
the related literature mainly focused on reducing the computational and memory
demand of the inference phase of machine and deep learning models. At the same
time, the training is typically assumed to be carried out in Cloud or edge
computing systems (due to the larger memory and computational requirements).
This assumption results in TML solutions that might become obsolete when the
process generating the data is affected by concept drift (e.g., due to
periodicity or seasonality effect, faults or malfunctioning affecting sensors
or actuators, or changes in the users' behavior), a common situation in
real-world application scenarios. For the first time in the literature, this
paper introduces a Tiny Machine Learning for Concept Drift (TML-CD) solution
based on deep learning feature extractors and a k-nearest neighbors classifier
integrating a hybrid adaptation module able to deal with concept drift
affecting the data-generating process. This adaptation module continuously
updates (in a passive way) the knowledge base of TML-CD and, at the same time,
employs a Change Detection Test to inspect for changes (in an active way) to
quickly adapt to concept drift by removing the obsolete knowledge. Experimental
results on both image and audio benchmarks show the effectiveness of the
proposed solution, whilst the porting of TML-CD on three off-the-shelf
micro-controller units shows the feasibility of what is proposed in real-world
pervasive systems.

    

### [[2107.14762] On the Efficacy of Small Self-Supervised Contrastive Models without Distillation Signals](http://arxiv.org/abs/2107.14762)


  It is a consensus that small models perform quite poorly under the paradigm
of self-supervised contrastive learning. Existing methods usually adopt a large
off-the-shelf model to transfer knowledge to the small one via knowledge
distillation. Despite their effectiveness, distillation-based methods may not
be suitable for some resource-restricted scenarios due to the huge
computational expenses of deploying a large model. In this paper, we study the
issue of training self-supervised small models without distillation signals. We
first evaluate the representation spaces of the small models and make two
non-negligible observations: (i) small models can complete the pretext task
without overfitting despite its limited capacity; (ii) small models universally
suffer the problem of over-clustering. Then we verify multiple assumptions that
are considered to alleviate the over-clustering phenomenon. Finally, we combine
the validated techniques and improve the baseline of five small architectures
with considerable margins, which indicates that training small self-supervised
contrastive models is feasible even without distillation signals.

    

### [[2107.14768] Debiased Explainable Pairwise Ranking from Implicit Feedback](http://arxiv.org/abs/2107.14768)


  Recent work in recommender systems has emphasized the importance of fairness,
with a particular interest in bias and transparency, in addition to predictive
accuracy. In this paper, we focus on the state of the art pairwise ranking
model, Bayesian Personalized Ranking (BPR), which has previously been found to
outperform pointwise models in predictive accuracy, while also being able to
handle implicit feedback. Specifically, we address two limitations of BPR: (1)
BPR is a black box model that does not explain its outputs, thus limiting the
user's trust in the recommendations, and the analyst's ability to scrutinize a
model's outputs; and (2) BPR is vulnerable to exposure bias due to the data
being Missing Not At Random (MNAR). This exposure bias usually translates into
an unfairness against the least popular items because they risk being
under-exposed by the recommender system. In this work, we first propose a novel
explainable loss function and a corresponding Matrix Factorization-based model
called Explainable Bayesian Personalized Ranking (EBPR) that generates
recommendations along with item-based explanations. Then, we theoretically
quantify additional exposure bias resulting from the explainability, and use it
as a basis to propose an unbiased estimator for the ideal EBPR loss. The result
is a ranking model that aptly captures both debiased and explainable user
preferences. Finally, we perform an empirical study on three real-world
datasets that demonstrate the advantages of our proposed models.

    

### [[2107.14776] Synthetic flow-based cryptomining attack generation through Generative Adversarial Networks](http://arxiv.org/abs/2107.14776)


  Due to the growing rise of cyber attacks in the Internet, flow-based data
sets are crucial to increase the performance of the Machine Learning (ML)
components that run in network-based intrusion detection systems (IDS). To
overcome the existing network traffic data shortage in attack analysis, recent
works propose Generative Adversarial Networks (GANs) for synthetic flow-based
network traffic generation. Data privacy is appearing more and more as a strong
requirement when processing such network data, which suggests to find solutions
where synthetic data can fully replace real data. Because of the
ill-convergence of the GAN training, none of the existing solutions can
generate high-quality fully synthetic data that can totally substitute real
data in the training of IDS ML components. Therefore, they mix real with
synthetic data, which acts only as data augmentation components, leading to
privacy breaches as real data is used. In sharp contrast, in this work we
propose a novel deterministic way to measure the quality of the synthetic data
produced by a GAN both with respect to the real data and to its performance
when used for ML tasks. As a byproduct, we present a heuristic that uses these
metrics for selecting the best performing generator during GAN training,
leading to a stopping criterion. An additional heuristic is proposed to select
the best performing GANs when different types of synthetic data are to be used
in the same ML task. We demonstrate the adequacy of our proposal by generating
synthetic cryptomining attack traffic and normal traffic flow-based data using
an enhanced version of a Wasserstein GAN. We show that the generated synthetic
network traffic can completely replace real data when training a ML-based
cryptomining detector, obtaining similar performance and avoiding privacy
violations, since real data is not used in the training of the ML-based
detector.

    

### [[2107.14795] Perceiver IO: A General Architecture for Structured Inputs & Outputs](http://arxiv.org/abs/2107.14795)


  The recently-proposed Perceiver model obtains good results on several domains
(images, audio, multimodal, point clouds) while scaling linearly in compute and
memory with the input size. While the Perceiver supports many kinds of inputs,
it can only produce very simple outputs such as class scores. Perceiver IO
overcomes this limitation without sacrificing the original's appealing
properties by learning to flexibly query the model's latent space to produce
outputs of arbitrary size and semantics. Perceiver IO still decouples model
depth from data size and still scales linearly with data size, but now with
respect to both input and output sizes. The full Perceiver IO model achieves
strong results on tasks with highly structured output spaces, such as natural
language and visual understanding, StarCraft II, and multi-task and multi-modal
domains. As highlights, Perceiver IO matches a Transformer-based BERT baseline
on the GLUE language benchmark without the need for input tokenization and
achieves state-of-the-art performance on Sintel optical flow estimation.

    

### [[2107.14796] Data-driven modeling of time-domain induced polarization](http://arxiv.org/abs/2107.14796)


  We present a novel approach for data-driven modeling of the time-domain
induced polarization (IP) phenomenon using variational autoencoders (VAE). VAEs
are Bayesian neural networks that aim to learn a latent statistical
distribution to encode extensive data sets as lower dimension representations.
We collected 1 600 319 IP decay curves in various regions of Canada, the United
States and Kazakhstan, and compiled them to train a deep VAE. The proposed deep
learning approach is strictly unsupervised and data-driven: it does not require
manual processing or ground truth labeling of IP data. Moreover, our VAE
approach avoids the pitfalls of IP parametrization with the empirical Cole-Cole
and Debye decomposition models, simple power-law models, or other sophisticated
mechanistic models. We demonstrate four applications of VAEs to model and
process IP data: (1) representative synthetic data generation, (2) unsupervised
Bayesian denoising and data uncertainty estimation, (3) quantitative evaluation
of the signal-to-noise ratio, and (4) automated outlier detection. We also
interpret the IP compilation's latent representation and reveal a strong
correlation between its first dimension and the average chargeability of IP
decays. Finally, we experiment with varying VAE latent space dimensions and
demonstrate that a single real-valued scalar parameter contains sufficient
information to encode our extensive IP data compilation. This new finding
suggests that modeling time-domain IP data using mathematical models governed
by more than one free parameter is ambiguous, whereas modeling only the average
chargeability is justified. A pre-trained implementation of our model --
readily applicable to new IP data from any geolocation -- is available as
open-source Python code for the applied geophysics community.

    

### [[2107.14803] DCT2net: an interpretable shallow CNN for image denoising](http://arxiv.org/abs/2107.14803)


  This work tackles the issue of noise removal from images, focusing on the
well-known DCT image denoising algorithm. The latter, stemming from signal
processing, has been well studied over the years. Though very simple, it is
still used in crucial parts of state-of-the-art "traditional" denoising
algorithms such as BM3D. Since a few years however, deep convolutional neural
networks (CNN) have outperformed their traditional counterparts, making signal
processing methods less attractive. In this paper, we demonstrate that a DCT
denoiser can be seen as a shallow CNN and thereby its original linear transform
can be tuned through gradient descent in a supervised manner, improving
considerably its performance. This gives birth to a fully interpretable CNN
called DCT2net. To deal with remaining artifacts induced by DCT2net, an
original hybrid solution between DCT and DCT2net is proposed combining the best
that these two methods can offer; DCT2net is selected to process non-stationary
image patches while DCT is optimal for piecewise smooth patches. Experiments on
artificially noisy images demonstrate that two-layer DCT2net provides
comparable results to BM3D and is as fast as DnCNN algorithm composed of more
than a dozen of layers.

    

### [[1906.04591] CNN depth analysis with different channel inputs for Acoustic Scene Classification](http://arxiv.org/abs/1906.04591)


  Acoustic scene classification (ASC) has been approached in the last years
using deep learning techniques such as convolutional neural networks or
recurrent neural networks. Many state-of-the-art solutions are based on image
classification frameworks and, as such, a 2D representation of the audio signal
is considered for training these networks. Finding the most suitable audio
representation is still a research area of interest. In this paper, different
log-Mel representations and combinations are analyzed. Experiments show that
the best results are obtained using the harmonic and percussive components plus
the difference between left and right stereo channels, (L-R). On the other
hand, it is a common strategy to ensemble different models in order to increase
the final accuracy. Even though averaging different model predictions is a
common choice, an exhaustive analysis of different ensemble techniques has not
been presented in ASC problems. In this paper, geometric and arithmetic mean
plus the Ordered Weighted Averaging (OWA) operator are studied as aggregation
operators for the output of the different models of the ensemble. Finally, the
work carried out in this paper is highly oriented towards real-time
implementations. In this context, as the number of applications for audio
classification on edge devices is increasing exponentially, we also analyze
different network depths and efficient solutions for aggregating ensemble
predictions.

    

### [[1911.12377] Multimodal Attention Networks for Low-Level Vision-and-Language Navigation](http://arxiv.org/abs/1911.12377)


  Vision-and-Language Navigation (VLN) is a challenging task in which an agent
needs to follow a language-specified path to reach a target destination. The
goal gets even harder as the actions available to the agent get simpler and
move towards low-level, atomic interactions with the environment. This setting
takes the name of low-level VLN. In this paper, we strive for the creation of
an agent able to tackle three key issues: multi-modality, long-term
dependencies, and adaptability towards different locomotive settings. To that
end, we devise "Perceive, Transform, and Act" (PTA): a fully-attentive VLN
architecture that leaves the recurrent approach behind and the first
Transformer-like architecture incorporating three different modalities -
natural language, images, and low-level actions for the agent control. In
particular, we adopt an early fusion strategy to merge lingual and visual
information efficiently in our encoder. We then propose to refine the decoding
phase with a late fusion extension between the agent's history of actions and
the perceptual modalities. We experimentally validate our model on two
datasets: PTA achieves promising results in low-level VLN on R2R and achieves
good performance in the recently proposed R4R benchmark. Our code is publicly
available at this https URL.

    

### [[2005.04954] Propagation Graph Estimation from Individual's Time Series of Observed States](http://arxiv.org/abs/2005.04954)


  Various things propagate through the medium of individuals. Some individuals
follow the others and take the states similar to their states a small number of
time steps later. In this paper, we study the problem of estimating the state
propagation order of individuals from the real-valued state sequences of all
the individuals. We propose a method to estimate the propagation direction
between individuals by the sum of the time delay of one individual's state
positions from the other individual's matched state position averaged over the
minimum cost alignments and show how to calculate it efficiently. The
propagation order estimated by our proposed method is demonstrated to be
significantly more accurate than that by a baseline method for our synthetic
datasets, and also to be consistent with visually recognizable propagation
orders for the dataset of Japanese stock price time series and biological cell
firing state sequences.

    

### [[2006.01005] The Effects of Mild Over-parameterization on the Optimization Landscape of Shallow ReLU Neural Networks](http://arxiv.org/abs/2006.01005)


  We study the effects of mild over-parameterization on the optimization
landscape of a simple ReLU neural network of the form
$\mathbf{x}\mapsto\sum_{i=1}^k\max\{0,\mathbf{w}_i^{\top}\mathbf{x}\}$, in a
well-studied teacher-student setting where the target values are generated by
the same architecture, and when directly optimizing over the population squared
loss with respect to Gaussian inputs. We prove that while the objective is
strongly convex around the global minima when the teacher and student networks
possess the same number of neurons, it is not even \emph{locally convex} after
any amount of over-parameterization. Moreover, related desirable properties
(e.g., one-point strong convexity and the Polyak-Łojasiewicz condition) also
do not hold even locally. On the other hand, we establish that the objective
remains one-point strongly convex in \emph{most} directions (suitably defined),
and show an optimization guarantee under this property. For the non-global
minima, we prove that adding even just a single neuron will turn a non-global
minimum into a saddle point. This holds under some technical conditions which
we validate empirically. These results provide a possible explanation for why
recovering a global minimum becomes significantly easier when we
over-parameterize, even if the amount of over-parameterization is very
moderate.

    

### [[2006.07027] Seq2Tens: An Efficient Representation of Sequences by Low-Rank Tensor Projections](http://arxiv.org/abs/2006.07027)


  Sequential data such as time series, video, or text can be challenging to
analyse as the ordered structure gives rise to complex dependencies. At the
heart of this is non-commutativity, in the sense that reordering the elements
of a sequence can completely change its meaning. We use a classical
mathematical object -- the tensor algebra -- to capture such dependencies. To
address the innate computational complexity of high degree tensors, we use
compositions of low-rank tensor projections. This yields modular and scalable
building blocks for neural networks that give state-of-the-art performance on
standard benchmarks such as multivariate time series classification and
generative models for video.

    

### [[2006.09446] Real-Time Regression with Dividing Local Gaussian Processes](http://arxiv.org/abs/2006.09446)


  The increased demand for online prediction and the growing availability of
large data sets drives the need for computationally efficient models. While
exact Gaussian process regression shows various favorable theoretical
properties (uncertainty estimate, unlimited expressive power), the poor scaling
with respect to the training set size prohibits its application in big data
regimes in real-time. Therefore, this paper proposes dividing local Gaussian
processes, which are a novel, computationally efficient modeling approach based
on Gaussian process regression. Due to an iterative, data-driven division of
the input space, they achieve a sublinear computational complexity in the total
number of training points in practice, while providing excellent predictive
distributions. A numerical evaluation on real-world data sets shows their
advantages over other state-of-the-art methods in terms of accuracy as well as
prediction and update speed.

    

### [[2007.03133] The Sample Complexity of Best-$k$ Items Selection from Pairwise Comparisons](http://arxiv.org/abs/2007.03133)


  This paper studies the sample complexity (aka number of comparisons) bounds
for the active best-$k$ items selection from pairwise comparisons. From a given
set of items, the learner can make pairwise comparisons on every pair of items,
and each comparison returns an independent noisy result about the preferred
item. At any time, the learner can adaptively choose a pair of items to compare
according to past observations (i.e., active learning). The learner's goal is
to find the (approximately) best-$k$ items with a given confidence, while
trying to use as few comparisons as possible. In this paper, we study two
problems: (i) finding the probably approximately correct (PAC) best-$k$ items
and (ii) finding the exact best-$k$ items, both under strong stochastic
transitivity and stochastic triangle inequality. For PAC best-$k$ items
selection, we first show a lower bound and then propose an algorithm whose
sample complexity upper bound matches the lower bound up to a constant factor.
For the exact best-$k$ items selection, we first prove a worst-instance lower
bound. We then propose two algorithms based on our PAC best items selection
algorithms: one works for $k=1$ and is sample complexity optimal up to a loglog
factor, and the other works for all values of $k$ and is sample complexity
optimal up to a log factor.

    

### [[2007.03767] Defending against Backdoors in Federated Learning with Robust Learning Rate](http://arxiv.org/abs/2007.03767)


  Federated learning (FL) allows a set of agents to collaboratively train a
model without sharing their potentially sensitive data. This makes FL suitable
for privacy-preserving applications. At the same time, FL is susceptible to
adversarial attacks due to decentralized and unvetted data. One important line
of attacks against FL is the backdoor attacks. In a backdoor attack, an
adversary tries to embed a backdoor functionality to the model during training
that can later be activated to cause a desired misclassification. To prevent
backdoor attacks, we propose a lightweight defense that requires minimal change
to the FL protocol. At a high level, our defense is based on carefully
adjusting the aggregation server's learning rate, per dimension and per round,
based on the sign information of agents' updates. We first conjecture the
necessary steps to carry a successful backdoor attack in FL setting, and then,
explicitly formulate the defense based on our conjecture. Through experiments,
we provide empirical evidence that supports our conjecture, and we test our
defense against backdoor attacks under different settings. We observe that
either backdoor is completely eliminated, or its accuracy is significantly
reduced. Overall, our experiments suggest that our defense significantly
outperforms some of the recently proposed defenses in the literature. We
achieve this by having minimal influence over the accuracy of the trained
models. In addition, we also provide convergence rate analysis for our proposed
scheme.

    

### [[2007.11972] DeepKriging: Spatially Dependent Deep Neural Networks for Spatial Prediction](http://arxiv.org/abs/2007.11972)


  In spatial statistics, a common objective is to predict the values of a
spatial process at unobserved locations by exploiting spatial dependence. In
geostatistics, Kriging provides the best linear unbiased predictor using
covariance functions and is often associated with Gaussian processes. However,
when considering non-linear prediction for non-Gaussian and categorical data,
the Kriging prediction is not necessarily optimal, and the associated variance
is often overly optimistic. We propose to use deep neural networks (DNNs) for
spatial prediction. Although DNNs are widely used for general classification
and prediction, they have not been studied thoroughly for data with spatial
dependence. In this work, we propose a novel neural network structure for
spatial prediction by adding an embedding layer of spatial coordinates with
basis functions. We show in theory that the proposed DeepKriging method has
multiple advantages over Kriging and classical DNNs only with spatial
coordinates as features. We also provide density prediction for uncertainty
quantification without any distributional assumption and apply the method to
PM$_{2.5}$ concentrations across the continental United States.

    

### [[2008.08903] Generative Adversarial Networks for Spatio-temporal Data: A Survey](http://arxiv.org/abs/2008.08903)


  Generative Adversarial Networks (GANs) have shown remarkable success in
producing realistic-looking images in the computer vision area. Recently,
GAN-based techniques are shown to be promising for spatio-temporal-based
applications such as trajectory prediction, events generation and time-series
data imputation. While several reviews for GANs in computer vision have been
presented, no one has considered addressing the practical applications and
challenges relevant to spatio-temporal data. In this paper, we have conducted a
comprehensive review of the recent developments of GANs for spatio-temporal
data. We summarise the application of popular GAN architectures for
spatio-temporal data and the common practices for evaluating the performance of
spatio-temporal applications with GANs. Finally, we point out future research
directions to benefit researchers in this area.

    

### [[2009.05079] Finding Stable Groups of Cross-Correlated Features in Two Data Sets With Common Samples](http://arxiv.org/abs/2009.05079)


  Data sets in which measurements of different types are obtained from a common
set of samples appear in many scientific applications. In the analysis of such
data, an important problem is to identify groups of features from different
data types that are strongly associated. Given two data types, a bimodule is a
pair $(A,B)$ of feature sets from the two types such that the aggregate
cross-correlation between the features in $A$ and those in $B$ is large. A
bimodule $(A,B)$ is stable if $A$ coincides with the set of features that have
significant aggregate correlation with the features in $B$, and vice-versa. We
develop an, iterative, testing-based procedure called BSP to identify stable
bimodules. BSP relies on approximate p-values derived from the permutation
moments of sums of squared sample correlations between a single feature of one
type and a group of features of the second type. We carry out a thorough
simulation study to assess the performance of BSP, and present an extended
application to the problem of expression quantitative trait loci (eQTL)
analysis using recent data from the GTEx project. In addition, we apply BSP to
climatology data to identify regions in North America where annual temperature
variation affects precipitation.

    

### [[2009.06402] Time-Aware Evidence Ranking for Fact-Checking](http://arxiv.org/abs/2009.06402)


  Truth can vary over time. Fact-checking decisions on claim veracity should
therefore take into account temporal information of both the claim and
supporting or refuting evidence. In this work, we investigate the hypothesis
that the timestamp of a Web page is crucial to how it should be ranked for a
given claim. We delineate four temporal ranking methods that constrain evidence
ranking differently and simulate hypothesis-specific evidence rankings given
the evidence timestamps as gold standard. Evidence ranking in three
fact-checking models is ultimately optimized using a learning-to-rank loss
function. Our study reveals that time-aware evidence ranking not only surpasses
relevance assumptions based purely on semantic similarity or position in a
search results list, but also improves veracity predictions of time-sensitive
claims in particular.

    

### [[2011.00241] Methods for Pruning Deep Neural Networks](http://arxiv.org/abs/2011.00241)


  This paper presents a survey of methods for pruning deep neural networks. It
begins by categorising over 150 studies based on the underlying approach used
and then focuses on three categories: methods that use magnitude based pruning,
methods that utilise clustering to identify redundancy, and methods that use
sensitivity analysis to assess the effect of pruning. Some of the key
influencing studies within these categories are presented to highlight the
underlying approaches and results achieved. Most studies present results which
are distributed in the literature as new architectures, algorithms and data
sets have developed with time, making comparison across different studied
difficult. The paper therefore provides a resource for the community that can
be used to quickly compare the results from many different methods on a variety
of data sets, and a range of architectures, including AlexNet, ResNet, DenseNet
and VGG. The resource is illustrated by comparing the results published for
pruning AlexNet and ResNet50 on ImageNet and ResNet56 and VGG16 on the CIFAR10
data to reveal which pruning methods work well in terms of retaining accuracy
whilst achieving good compression rates. The paper concludes by identifying
some promising directions for future research.

    

### [[2011.00509] PILOT: Efficient Planning by Imitation Learning and Optimisation for Safe Autonomous Driving](http://arxiv.org/abs/2011.00509)


  Achieving a proper balance between planning quality, safety and efficiency is
a major challenge for autonomous driving. Optimisation-based motion planners
are capable of producing safe, smooth and comfortable plans, but often at the
cost of runtime efficiency. On the other hand, naively deploying trajectories
produced by efficient-to-run deep imitation learning approaches might risk
compromising safety. In this paper, we present PILOT -- a planning framework
that comprises an imitation neural network followed by an efficient optimiser
that actively rectifies the network's plan, guaranteeing fulfilment of safety
and comfort requirements. The objective of the efficient optimiser is the same
as the objective of an expensive-to-run optimisation-based planning system that
the neural network is trained offline to imitate. This efficient optimiser
provides a key layer of online protection from learning failures or deficiency
in out-of-distribution situations that might compromise safety or comfort.
Using a state-of-the-art, runtime-intensive optimisation-based method as the
expert, we demonstrate in simulated autonomous driving experiments in CARLA
that PILOT achieves a seven-fold reduction in runtime when compared to the
expert it imitates without sacrificing planning quality.

    

### [[2011.01174] Perceptually Guided End-to-End Text-to-Speech With MOS Prediction](http://arxiv.org/abs/2011.01174)


  Although recent end-to-end text-to-speech (TTS) systems have achieved
high-quality synthesized speech, there are still several factors that degrade
the quality of synthesized speech, including lack of training data or
information loss during knowledge distillation. To address the problem, we
propose a novel way to train a TTS model under the supervision of perceptual
loss, which measures the distance between the maximum speech quality score and
the predicted one. We first pre-train a mean opinion score (MOS) prediction
model and then train a TTS model in the direction of maximizing the MOS of
synthesized speech predicted by the pre-trained MOS prediction model. Through
this method, we can improve the quality of synthesized speech universally
(i.e., regardless of the network architecture or the cause of the speech
quality degradation) and efficiently (i.e., without increasing the inference
time or the model complexity). The evaluation results for MOS and phoneme error
rate demonstrate that our proposed approach improves previous models in terms
of both naturalness and intelligibility.

    

### [[2011.06775] DiGNet: Learning Scalable Self-Driving Policies for Generic Traffic Scenarios with Graph Neural Networks](http://arxiv.org/abs/2011.06775)


  Traditional decision and planning frameworks for self-driving vehicles (SDVs)
scale poorly in new scenarios, thus they require tedious hand-tuning of rules
and parameters to maintain acceptable performance in all foreseeable cases.
Recently, self-driving methods based on deep learning have shown promising
results with better generalization capability but less hand engineering effort.
However, most of the previous learning-based methods are trained and evaluated
in limited driving scenarios with scattered tasks, such as lane-following,
autonomous braking, and conditional driving. In this paper, we propose a
graph-based deep network to achieve scalable self-driving that can handle
massive traffic scenarios. Specifically, more than 7,000 km of evaluation is
conducted in a high-fidelity driving simulator, in which our method can obey
the traffic rules and safely navigate the vehicle in a large variety of urban,
rural, and highway environments, including unprotected left turns, narrow
roads, roundabouts, and pedestrian-rich intersections. Demonstration videos are
available at this https URL.

    

### [[2011.10596] The Impact of Data on the Stability of Learning-Based Control- Extended Version](http://arxiv.org/abs/2011.10596)


  Despite the existence of formal guarantees for learning-based control
approaches, the relationship between data and control performance is still
poorly understood. In this paper, we propose a Lyapunov-based measure for
quantifying the impact of data on the certifiable control performance. By
modeling unknown system dynamics through Gaussian processes, we can determine
the interrelation between model uncertainty and satisfaction of stability
conditions. This allows us to directly asses the impact of data on the provable
stationary control performance, and thereby the value of the data for the
closed-loop system performance. Our approach is applicable to a wide variety of
unknown nonlinear systems that are to be controlled by a generic learning-based
control law, and the results obtained in numerical simulations indicate the
efficacy of the proposed measure.

    

### [[2011.11259] Sparse generative modeling via parameter-reduction of Boltzmann machines: application to protein-sequence families](http://arxiv.org/abs/2011.11259)


  Boltzmann machines (BM) are widely used as generative models. For example,
pairwise Potts models (PM), which are instances of the BM class, provide
accurate statistical models of families of evolutionarily related protein
sequences. Their parameters are the local fields, which describe site-specific
patterns of amino-acid conservation, and the two-site couplings, which mirror
the coevolution between pairs of sites. This coevolution reflects structural
and functional constraints acting on protein sequences during evolution. The
most conservative choice to describe the coevolution signal is to include all
possible two-site couplings into the PM. This choice, typical of what is known
as Direct Coupling Analysis, has been successful for predicting residue
contacts in the three-dimensional structure, mutational effects, and in
generating new functional sequences. However, the resulting PM suffers from
important over-fitting effects: many couplings are small, noisy and hardly
interpretable; the PM is close to a critical point, meaning that it is highly
sensitive to small parameter perturbations. In this work, we introduce a
general parameter-reduction procedure for BMs, via a controlled iterative
decimation of the less statistically significant couplings, identified by an
information-based criterion that selects either weak or statistically
unsupported couplings. For several protein families, our procedure allows one
to remove more than $90\%$ of the PM couplings, while preserving the predictive
and generative properties of the original dense PM, and the resulting model is
far away from criticality, hence more robust to noise.

    

### [[2101.07240] Multimodal Variational Autoencoders for Semi-Supervised Learning: In Defense of Product-of-Experts](http://arxiv.org/abs/2101.07240)


  Multimodal generative models should be able to learn a meaningful latent
representation that enables a coherent joint generation of all modalities
(e.g., images and text). Many applications also require the ability to
accurately sample modalities conditioned on observations of a subset of the
modalities. Often not all modalities may be observed for all training data
points, so semi-supervised learning should be possible. In this study, we
propose a novel product-of-experts (PoE) based variational autoencoder that
have these desired properties. We benchmark it against a mixture-of-experts
(MoE) approach and an approach of combining the modalities with an additional
encoder network. An empirical evaluation shows that the PoE based models can
outperform the contrasted models. Our experiments support the intuition that
PoE models are more suited for a conjunctive combination of modalities.

    

### [[2103.03614] FloMo: Tractable Motion Prediction with Normalizing Flows](http://arxiv.org/abs/2103.03614)


  The future motion of traffic participants is inherently uncertain. To plan
safely, therefore, an autonomous agent must take into account multiple possible
trajectory outcomes and prioritize them. Recently, this problem has been
addressed with generative neural networks. However, most generative models
either do not learn the true underlying trajectory distribution reliably, or do
not allow predictions to be associated with likelihoods. In our work, we model
motion prediction directly as a density estimation problem with a normalizing
flow between a noise distribution and the future motion distribution. Our
model, named FloMo, allows likelihoods to be computed in a single network pass
and can be trained directly with maximum likelihood estimation. Furthermore, we
propose a method to stabilize training flows on trajectory datasets and a new
data augmentation transformation that improves the performance and
generalization of our model. Our method achieves state-of-the-art performance
on three popular prediction datasets, with a significant gap to most competing
models.

    

### [[2103.04032] CAM-GAN: Continual Adaptation Modules for Generative Adversarial Networks](http://arxiv.org/abs/2103.04032)


  We present a continual learning approach for generative adversarial networks
(GANs), by designing and leveraging parameter-efficient feature map
transformations. Our approach is based on learning a set of global and
task-specific parameters. The global parameters are fixed across tasks whereas
the task-specific parameters act as local adapters for each task, and help in
efficiently obtaining task-specific feature maps. Moreover, we propose an
element-wise addition of residual bias in the transformed feature space, which
further helps stabilize GAN training in such settings. Our approach also
leverages task similarity information based on the Fisher information matrix.
Leveraging this knowledge from previous tasks significantly improves the model
performance. In addition, the similarity measure also helps reduce the
parameter growth in continual adaptation and helps to learn a compact model. In
contrast to the recent approaches for continually-learned GANs, the proposed
approach provides a memory-efficient way to perform effective continual data
generation. Through extensive experiments on challenging and diverse datasets,
we show that the feature-map-transformation approach outperforms
state-of-the-art methods for continually-learned GANs, with substantially fewer
parameters. The proposed method generates high-quality samples that can also
improve the generative-replay-based continual learning for discriminative
tasks.

    

### [[2103.10651] SoK: A Modularized Approach to Study the Security of Automatic Speech Recognition Systems](http://arxiv.org/abs/2103.10651)


  With the wide use of Automatic Speech Recognition (ASR) in applications such
as human machine interaction, simultaneous interpretation, audio transcription,
etc., its security protection becomes increasingly important. Although recent
studies have brought to light the weaknesses of popular ASR systems that enable
out-of-band signal attack, adversarial attack, etc., and further proposed
various remedies (signal smoothing, adversarial training, etc.), a systematic
understanding of ASR security (both attacks and defenses) is still missing,
especially on how realistic such threats are and how general existing
protection could be. In this paper, we present our systematization of knowledge
for ASR security and provide a comprehensive taxonomy for existing work based
on a modularized workflow. More importantly, we align the research in this
domain with that on security in Image Recognition System (IRS), which has been
extensively studied, using the domain knowledge in the latter to help
understand where we stand in the former. Generally, both IRS and ASR are
perceptual systems. Their similarities allow us to systematically study
existing literature in ASR security based on the spectrum of attacks and
defense solutions proposed for IRS, and pinpoint the directions of more
advanced attacks and the directions potentially leading to more effective
protection in ASR. In contrast, their differences, especially the complexity of
ASR compared with IRS, help us learn unique challenges and opportunities in ASR
security. Particularly, our experimental study shows that transfer learning
across ASR models is feasible, even in the absence of knowledge about models
(even their types) and training data.

    

### [[2103.14910] MINE: Towards Continuous Depth MPI with NeRF for Novel View Synthesis](http://arxiv.org/abs/2103.14910)


  In this paper, we propose MINE to perform novel view synthesis and depth
estimation via dense 3D reconstruction from a single image. Our approach is a
continuous depth generalization of the Multiplane Images (MPI) by introducing
the NEural radiance fields (NeRF). Given a single image as input, MINE predicts
a 4-channel image (RGB and volume density) at arbitrary depth values to jointly
reconstruct the camera frustum and fill in occluded contents. The reconstructed
and inpainted frustum can then be easily rendered into novel RGB or depth views
using differentiable rendering. Extensive experiments on RealEstate10K, KITTI
and Flowers Light Fields show that our MINE outperforms state-of-the-art by a
large margin in novel view synthesis. We also achieve competitive results in
depth estimation on iBims-1 and NYU-v2 without annotated depth supervision. Our
source code is available at this https URL


### [[2104.07576] Piecewise-linear modelling with feature selection for Li-ion battery end of life prognosis](http://arxiv.org/abs/2104.07576)


  The complex nature of lithium-ion battery degradation has led to many machine
learning based approaches to health forecasting being proposed in literature.
However, machine learning can be computationally intensive. Linear approaches
are faster but have previously been too inflexible for successful prognosis.
For both techniques, the choice and quality of the inputs is a limiting factor
of performance. Piecewise-linear models, combined with automated feature
selection, offer a fast and flexible alternative without being as
computationally intensive as machine learning. Here, a piecewise-linear
approach to battery health forecasting was compared to a Gaussian process
regression tool and found to perform equally well. The input feature selection
process demonstrated the benefit of limiting the correlation between inputs.
Further trials found that the piecewise-linear approach was robust to changing
input size and availability of training data.

    

### [[2104.15106] Latent Factor Decomposition Model: Applications for Questionnaire Data](http://arxiv.org/abs/2104.15106)


  The analysis of clinical questionnaire data comes with many inherent
challenges. These challenges include the handling of data with missing fields,
as well as the overall interpretation of a dataset with many fields of
different scales and forms. While numerous methods have been developed to
address these challenges, they are often not robust, statistically sound, or
easily interpretable. Here, we propose a latent factor modeling framework that
extends the principal component analysis for both categorical and quantitative
data with missing elements. The model simultaneously provides the principal
components (basis) and each patients' projections on these bases in a latent
space. We show an application of our modeling framework through Irritable Bowel
Syndrome (IBS) symptoms, where we find correlations between these projections
and other standardized patient symptom scales. This latent factor model can be
easily applied to different clinical questionnaire datasets for clustering
analysis and interpretable inference.

    

### [[2105.01531] VQCPC-GAN: Variable-Length Adversarial Audio Synthesis Using Vector-Quantized Contrastive Predictive Coding](http://arxiv.org/abs/2105.01531)


  Influenced by the field of Computer Vision, Generative Adversarial Networks
(GANs) are often adopted for the audio domain using fixed-size two-dimensional
spectrogram representations as the "image data". However, in the (musical)
audio domain, it is often desired to generate output of variable duration. This
paper presents VQCPC-GAN, an adversarial framework for synthesizing
variable-length audio by exploiting Vector-Quantized Contrastive Predictive
Coding (VQCPC). A sequence of VQCPC tokens extracted from real audio data
serves as conditional input to a GAN architecture, providing step-wise
time-dependent features of the generated content. The input noise z
(characteristic in adversarial architectures) remains fixed over time, ensuring
temporal consistency of global features. We evaluate the proposed model by
comparing a diverse set of metrics against various strong baselines. Results
show that, even though the baselines score best, VQCPC-GAN achieves comparable
performance even when generating variable-length audio. Numerous sound examples
are provided in the accompanying website, and we release the code for
reproducibility.

    

### [[2106.09296] Voice2Series: Reprogramming Acoustic Models for Time Series Classification](http://arxiv.org/abs/2106.09296)


  Learning to classify time series with limited data is a practical yet
challenging problem. Current methods are primarily based on hand-designed
feature extraction rules or domain-specific data augmentation. Motivated by the
advances in deep speech processing models and the fact that voice data are
univariate temporal signals, in this paper, we propose Voice2Series (V2S), a
novel end-to-end approach that reprograms acoustic models for time series
classification, through input transformation learning and output label mapping.
Leveraging the representation learning power of a large-scale pre-trained
speech processing model, on 30 different time series tasks we show that V2S
either outperforms or is tied with state-of-the-art methods on 20 tasks, and
improves their average accuracy by 1.84%. We further provide a theoretical
justification of V2S by proving its population risk is upper bounded by the
source risk and a Wasserstein distance accounting for feature alignment via
reprogramming. Our results offer new and effective means to time series
classification.

    

### [[2106.10410] Deep Generative Learning via Schrödinger Bridge](http://arxiv.org/abs/2106.10410)


  We propose to learn a generative model via entropy interpolation with a
Schrödinger Bridge. The generative learning task can be formulated as
interpolating between a reference distribution and a target distribution based
on the Kullback-Leibler divergence. At the population level, this entropy
interpolation is characterized via an SDE on $[0,1]$ with a time-varying drift
term. At the sample level, we derive our Schrödinger Bridge algorithm by
plugging the drift term estimated by a deep score estimator and a deep density
ratio estimator into the Euler-Maruyama method. Under some mild smoothness
assumptions of the target distribution, we prove the consistency of both the
score estimator and the density ratio estimator, and then establish the
consistency of the proposed Schrödinger Bridge approach. Our theoretical
results guarantee that the distribution learned by our approach converges to
the target distribution. Experimental results on multimodal synthetic data and
benchmark data support our theoretical findings and indicate that the
generative model via Schrödinger Bridge is comparable with state-of-the-art
GANs, suggesting a new formulation of generative learning. We demonstrate its
usefulness in image interpolation and image inpainting.

    

### [[2106.11760] A Novel Verifiable Fingerprinting Scheme for Generative Adversarial Networks](http://arxiv.org/abs/2106.11760)


  This paper presents a novel fingerprinting scheme for the Intellectual
Property (IP) protection of Generative Adversarial Networks (GANs). Prior
solutions for classification models adopt adversarial examples as the
fingerprints, which can raise stealthiness and robustness problems when they
are applied to the GAN models. Our scheme constructs a composite deep learning
model from the target GAN and a classifier. Then we generate stealthy
fingerprint samples from this composite model, and register them to the
classifier for effective ownership verification. This scheme inspires three
concrete methodologies to practically protect the modern GAN models.
Theoretical analysis proves that these methods can satisfy different security
requirements necessary for IP protection. We also conduct extensive experiments
to show that our solutions outperform existing strategies in terms of
stealthiness, functionality-preserving and unremovability.

    

### [[2106.11930] On the importance of cross-task features for class-incremental learning](http://arxiv.org/abs/2106.11930)


  In class-incremental learning, an agent with limited resources needs to learn
a sequence of classification tasks, forming an ever growing classification
problem, with the constraint of not being able to access data from previous
tasks. The main difference with task-incremental learning, where a task-ID is
available at inference time, is that the learner also needs to perform
cross-task discrimination, i.e. distinguish between classes that have not been
seen together. Approaches to tackle this problem are numerous and mostly make
use of an external memory (buffer) of non-negligible size. In this paper, we
ablate the learning of cross-task features and study its influence on the
performance of basic replay strategies used for class-IL. We also define a new
forgetting measure for class-incremental learning, and see that forgetting is
not the principal cause of low performance. Our experimental results show that
future algorithms for class-incremental learning should not only prevent
forgetting, but also aim to improve the quality of the cross-task features, and
the knowledge transfer between tasks. This is especially important when tasks
contain limited amount of data.

    

### [[2107.14371] Distributed Strategy Selection: A Submodular Set Function Maximization Approach](http://arxiv.org/abs/2107.14371)


  Constrained submodular set function maximization problems often appear in
multi-agent decision-making problems with a discrete feasible set. A prominent
example is the problem of multi-agent mobile sensor placement over a discrete
domain. Submodular set function optimization problems, however, are known to be
NP-hard. This paper considers a class of submodular optimization problems that
consist of maximization of a monotone and submodular set function subject to a
uniform matroid constraint over a group of networked agents that communicate
over a connected undirected graph. We work in the value oracle model where the
only access of the agents to the utility function is through a black box that
returns the utility function value. We propose a distributed suboptimal
polynomial-time algorithm that enables each agent to obtain its respective
strategy via local interactions with its neighboring agents. Our solution is a
fully distributed gradient-based algorithm using the submodular set functions'
multilinear extension followed by a distributed stochastic Pipage rounding
procedure. This algorithm results in a strategy set that when the team utility
function is evaluated at worst case, the utility function value is in
1/c(1-e^(-c)-O(1/T)) of the optimal solution with c to be the curvature of the
submodular function. An example demonstrates our results.

    

### [[2107.14509] On Strong Observational Refinement and Forward Simulation](http://arxiv.org/abs/2107.14509)


  Hyperproperties are correctness conditions for labelled transition systems
that are more expressive than traditional trace properties, with particular
relevance to security. Recently, Attiya and Enea studied a notion of strong
observational refinement that preserves all hyperproperties. They analyse the
correspondence between forward simulation and strong observational refinement
in a setting with finite traces only. We study this correspondence in a setting
with both finite and infinite traces. In particular, we show that forward
simulation does not preserve hyperliveness properties in this setting. We
extend the forward simulation proof obligation with a progress condition, and
prove that this progressive forward simulation does imply strong observational
refinement.

    

### [[2107.14570] Beep-And-Sleep: Message and Energy Efficient Set Cover](http://arxiv.org/abs/2107.14570)


  We observe message-efficient distributed algorithms for the Set Cover
problem. Given a ground set $U$ of $n$ elements and $m$ subsets of $U$, we aim
to find the minimal number of these subsets that contain all elements. In the
default distributed setup of this problem, each set has a bidirected
communication link with each element it contains. Our first result is a
$\tilde{O}(\log^2(\Delta))$-time and $O(\sqrt{\Delta)}(n+m))$-message algorithm
with expected approximation ration of $O(\log(\Delta))$ in the $KT_0$ model.
The value $\Delta$ denotes the maximal cardinality of each subset. Our
algorithm is \emph{almost} optimal with regard to time and message complexity.
Further, we present Set Cover algorithm in the Beeping model that only relies
on carrier-sensing and can trade runtime for approximation ratio similar to the
celebrated algorithm by Kuhn and Wattenhofer [PODC '03].

    

### [[2107.14790] Out-of-Core Surface Reconstruction via Global $TGV$ Minimization](http://arxiv.org/abs/2107.14790)


  We present an out-of-core variational approach for surface reconstruction
from a set of aligned depth maps. Input depth maps are supposed to be
reconstructed from regular photos or/and can be a representation of terrestrial
LIDAR point clouds. Our approach is based on surface reconstruction via total
generalized variation minimization ($TGV$) because of its strong
visibility-based noise-filtering properties and GPU-friendliness. Our main
contribution is an out-of-core OpenCL-accelerated adaptation of this numerical
algorithm which can handle arbitrarily large real-world scenes with scale
diversity.

    

### [[2011.15013] Modularising Verification Of Durable Opacity](http://arxiv.org/abs/2011.15013)


  Non-volatile memory (NVM), also known as persistent memory, is an emerging
paradigm for memory that preserves its contents even after power loss. NVM is
widely expected to become ubiquitous, and hardware architectures are already
providing support for NVM programming. This has stimulated interest in the
design of novel concepts ensuring correctness of concurrent programming
abstractions in the face of persistency and in the development of associated
verification approaches.
Software transactional memory (STM) is a key programming abstraction that
supports concurrent access to shared state. In a fashion similar to
linearizability as the correctness condition for concurrent data structures,
there is an established notion of correctness for STMs known as opacity. We
have recently proposed {\em durable opacity} as the natural extension of
opacity to a setting with non-volatile memory. Together with this novel
correctness condition, we designed a verification technique based on
refinement. In this paper, we extend this work in two directions. First, we
develop a durably opaque version of NOrec (no ownership records), an existing
STM algorithm proven to be opaque. Second, we modularise our existing
verification approach by separating the proof of durability of memory accesses
from the proof of opacity. For NOrec, this allows us to re-use an existing
opacity proof and complement it with a proof of the durability of accesses to
shared state.

    

### [[2103.03013] ECM modeling and performance tuning of SpMV and Lattice QCD on A64FX](http://arxiv.org/abs/2103.03013)


  The A64FX CPU is arguably the most powerful Arm-based processor design to
date. Although it is a traditional cache-based multicore processor, its peak
performance and memory bandwidth rival accelerator devices. A good
understanding of its performance features is of paramount importance for
developers who wish to leverage its full potential. We present an architectural
analysis of the A64FX used in the Fujitsu FX1000 supercomputer at a level of
detail that allows for the construction of Execution-Cache-Memory (ECM)
performance models for steady-state loops. In the process we identify
architectural peculiarities that point to viable generic optimization
strategies. After validating the model using simple streaming loops we apply
the insight gained to sparse matrix-vector multiplication (SpMV) and the domain
wall (DW) kernel from quantum chromodynamics (QCD). For SpMV we show why the
CRS matrix storage format is not a good practical choice on this architecture
and how the SELL-C-sigma format can achieve bandwidth saturation. For the DW
kernel we provide a cache-reuse analysis and show how an appropriate choice of
data layout for complex arrays can realize memory-bandwidth saturation in this
case as well. A comparison with state-of-the-art high-end Intel Cascade Lake AP
and Nvidia V100 systems puts the capabilities of the A64FX into perspective. We
also explore the potential for power optimizations using the tuning knobs
provided by the Fugaku system, achieving energy savings of about 31% for SpMV
and 18% for DW.

    

### [[2104.04473] Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](http://arxiv.org/abs/2104.04473)


  Large language models have led to state-of-the-art accuracies across a range
of tasks. However, training these models efficiently is challenging for two
reasons: a) GPU memory capacity is limited, making it impossible to fit large
models on even a multi-GPU server; b) the number of compute operations required
to train these models can result in unrealistically long training times.
Consequently, new methods of model parallelism such as tensor and pipeline
parallelism have been proposed. Unfortunately, naive usage of these methods
leads to fundamental scaling issues at thousands of GPUs, e.g., due to
expensive cross-node communication or devices spending significant time waiting
on other devices to make progress.
In this paper, we show how different types of parallelism methods (tensor,
pipeline, and data parallelism) can be composed to scale to thousands of GPUs
and models with trillions of parameters. We survey techniques for pipeline
parallelism and propose a novel interleaved pipeline parallelism schedule that
can improve throughput by 10+% with memory footprint comparable to existing
approaches. We quantitatively study the trade-offs between tensor, pipeline,
and data parallelism, and provide intuition as to how to configure distributed
training of a large model. Our approach allows us to perform training
iterations on a model with 1 trillion parameters at 502 petaFLOP/s on 3072 GPUs
with achieved per-GPU throughput of 52% of theoretical peak. Our code is open
sourced at this https URL.

    

### [[2107.14374] Modelling and Reasoning Techniques for Context Aware Computing in Intelligent Transportation System](http://arxiv.org/abs/2107.14374)


  The emergence of Internet of Things technology and recent advancement in
sensor networks enabled transportation systems to a new dimension called
Intelligent Transportation System. Due to increased usage of vehicles and
communication among entities in road traffic scenarios, the amount of raw data
generation in Intelligent Transportation System is huge. This raw data are to
be processed to infer contextual information and provide new services related
to different modes of road transport such as traffic signal management,
accident prediction, object detection etc. To understand the importance of
context, this article aims to study context awareness in the Intelligent
Transportation System. We present a review on prominent applications developed
in the literature concerning context awareness in the intelligent
transportation system. The objective of this research paper is to highlight
context and its features in ITS and to address the applicability of modelling
techniques and reasoning approaches in Intelligent Transportation System. Also
to shed light on impact of Internet of Things and machine learning in
Intelligent Transportation System development.

    

### [[2107.14402] Difficulty-Aware Machine Translation Evaluation](http://arxiv.org/abs/2107.14402)


  The high-quality translation results produced by machine translation (MT)
systems still pose a huge challenge for automatic evaluation. Current MT
evaluation pays the same attention to each sentence component, while the
questions of real-world examinations (e.g., university examinations) have
different difficulties and weightings. In this paper, we propose a novel
difficulty-aware MT evaluation metric, expanding the evaluation dimension by
taking translation difficulty into consideration. A translation that fails to
be predicted by most MT systems will be treated as a difficult one and assigned
a large weight in the final score function, and conversely. Experimental
results on the WMT19 English-German Metrics shared tasks show that our proposed
method outperforms commonly used MT metrics in terms of human correlation. In
particular, our proposed method performs well even when all the MT systems are
very competitive, which is when most existing metrics fail to distinguish
between them. The source code is freely available at
this https URL.

    

### [[2107.14414] Towards Understanding the Impact of Real-Time AI-Powered Educational Dashboards (RAED) on Providing Guidance to Instructors](http://arxiv.org/abs/2107.14414)


  The objectives of this ongoing research are to build Real-Time AI-Powered
Educational Dashboard (RAED) as a decision support tool for instructors, and to
measure its impact on them while making decisions. Current developments in AI
can be combined with the educational dashboards to make them AI-Powered. Thus,
AI can help in providing recommendations based on the students' performances.
AI-Powered educational dashboards can also assist instructors in tracking
real-time student activities. In this ongoing research, our aim is to develop
the AI component as well as improve the existing design component of the RAED.
Further, we will conduct experiments to study its impact on instructors, and
understand how much they trust RAED to guide them while making decisions. This
paper elaborates on the ongoing research and future direction.

    

### [[2107.14425] Enhancing Social Relation Inference with Concise Interaction Graph and Discriminative Scene Representation](http://arxiv.org/abs/2107.14425)


  There has been a recent surge of research interest in attacking the problem
of social relation inference based on images. Existing works classify social
relations mainly by creating complicated graphs of human interactions, or
learning the foreground and/or background information of persons and objects,
but ignore holistic scene context. The holistic scene refers to the
functionality of a place in images, such as dinning room, playground and
office. In this paper, by mimicking human understanding on images, we propose
an approach of \textbf{PR}actical \textbf{I}nference in \textbf{S}ocial
r\textbf{E}lation (PRISE), which concisely learns interactive features of
persons and discriminative features of holistic scenes. Technically, we develop
a simple and fast relational graph convolutional network to capture interactive
features of all persons in one image. To learn the holistic scene feature, we
elaborately design a contrastive learning task based on image scene
classification. To further boost the performance in social relation inference,
we collect and distribute a new large-scale dataset, which consists of about
240 thousand unlabeled images. The extensive experimental results show that our
novel learning framework significantly beats the state-of-the-art methods,
e.g., PRISE achieves 6.8$\%$ improvement for domain classification in PIPA
dataset.

    

### [[2107.14487] Refining Labelled Systems for Modal and Constructive Logics with Applications](http://arxiv.org/abs/2107.14487)


  This thesis introduces the "method of structural refinement", which serves as
a means of transforming the relational semantics of a modal and/or constructive
logic into an 'economical' proof system by connecting two proof-theoretic
paradigms: labelled and nested sequent calculi. The formalism of labelled
sequents has been successful in that cut-free calculi in possession of
desirable proof-theoretic properties can be automatically generated for large
classes of logics. Despite these qualities, labelled systems make use of a
complicated syntax that explicitly incorporates the semantics of the associated
logic, and such systems typically violate the subformula property to a high
degree. By contrast, nested sequent calculi employ a simpler syntax and adhere
to a strict reading of the subformula property, making such systems useful in
the design of automated reasoning algorithms. However, the downside of the
nested sequent paradigm is that a general theory concerning the automated
construction of such calculi (as in the labelled setting) is essentially
absent, meaning that the construction of nested systems and the confirmation of
their properties is usually done on a case-by-case basis. The refinement method
connects both paradigms in a fruitful way, by transforming labelled systems
into nested (or, refined labelled) systems with the properties of the former
preserved throughout the transformation process.
To demonstrate the method of refinement and some of its applications, we
consider grammar logics, first-order intuitionistic logics, and deontic STIT
logics. The introduced refined labelled calculi will be used to provide the
first proof-search algorithms for deontic STIT logics. Furthermore, we employ
our refined labelled calculi for grammar logics to show that every logic in the
class possesses the effective Lyndon interpolation property.

    

### [[2107.14573] Neural Network Based Model Predictive Control for an Autonomous Vehicle](http://arxiv.org/abs/2107.14573)


  We study learning based controllers as a replacement for model predictive
controllers (MPC) for the control of autonomous vehicles. We concentrate for
the experiments on the simple yet representative bicycle model. We compare
training by supervised learning and by reinforcement learning. We also discuss
the neural net architectures so as to obtain small nets with the best
performances. This work aims at producing controllers that can both be embedded
on real-time platforms and amenable to verification by formal methods
techniques.

    

### [[2107.14589] On the Quantum-like Contextuality of Ambiguous Phrases](http://arxiv.org/abs/2107.14589)


  Language is contextual as meanings of words are dependent on their contexts.
Contextuality is, concomitantly, a well-defined concept in quantum mechanics
where it is considered a major resource for quantum computations. We
investigate whether natural language exhibits any of the quantum mechanics'
contextual features. We show that meaning combinations in ambiguous phrases can
be modelled in the sheaf-theoretic framework for quantum contextuality, where
they can become possibilistically contextual. Using the framework of
Contextuality-by-Default (CbD), we explore the probabilistic variants of these
and show that CbD-contextuality is also possible.

    

### [[2107.14654] Brain-Inspired Deep Imitation Learning for Autonomous Driving Systems](http://arxiv.org/abs/2107.14654)


  Autonomous driving has attracted great attention from both academics and
industries. To realise autonomous driving, Deep Imitation Learning (DIL) is
treated as one of the most promising solutions, because it improves autonomous
driving systems by automatically learning a complex mapping from human driving
data, compared to manually designing the driving policy. However, existing DIL
methods cannot generalise well across domains, that is, a network trained on
the data of source domain gives rise to poor generalisation on the data of
target domain. In the present study, we propose a novel brain-inspired deep
imitation method that builds on the evidence from human brain functions, to
improve the generalisation ability of deep neural networks so that autonomous
driving systems can perform well in various scenarios. Specifically, humans
have a strong generalisation ability which is beneficial from the structural
and functional asymmetry of the two sides of the brain. Here, we design dual
Neural Circuit Policy (NCP) architectures in deep neural networks based on the
asymmetry of human neural networks. Experimental results demonstrate that our
brain-inspired method outperforms existing methods regarding generalisation
when dealing with unseen data. Our source codes and pretrained models are
available at
this https URL}{this https URL.

    

### [[2107.14691] EmailSum: Abstractive Email Thread Summarization](http://arxiv.org/abs/2107.14691)


  Recent years have brought about an interest in the challenging task of
summarizing conversation threads (meetings, online discussions, etc.). Such
summaries help analysis of the long text to quickly catch up with the decisions
made and thus improve our work or communication efficiency. To spur research in
thread summarization, we have developed an abstractive Email Thread
Summarization (EmailSum) dataset, which contains human-annotated short (<30
words) and long (<100 words) summaries of 2549 email threads (each containing 3
to 10 emails) over a wide variety of topics. We perform a comprehensive
empirical study to explore different summarization techniques (including
extractive and abstractive methods, single-document and hierarchical models, as
well as transfer and semisupervised learning) and conduct human evaluations on
both short and long summary generation tasks. Our results reveal the key
challenges of current abstractive summarization models in this task, such as
understanding the sender's intent and identifying the roles of sender and
receiver. Furthermore, we find that widely used automatic evaluation metrics
(ROUGE, BERTScore) are weakly correlated with human judgments on this email
thread summarization task. Hence, we emphasize the importance of human
evaluation and the development of better metrics by the community. Our code and
summary data have been made available at:
this https URL


### [[2107.14764] Adaptive Approach Phase Guidance for a Hypersonic Glider via Reinforcement Meta Learning](http://arxiv.org/abs/2107.14764)


  We use Reinforcement Meta Learning to optimize an adaptive guidance system
suitable for the approach phase of a gliding hypersonic vehicle. Adaptability
is achieved by optimizing over a range of off-nominal flight conditions
including perturbation of aerodynamic coefficient parameters, actuator failure
scenarios, and sensor noise. The system maps observations directly to commanded
bank angle and angle of attack rates. These observations include a velocity
field tracking error formulated using parallel navigation, but adapted to work
over long trajectories where the Earth's curvature must be taken into account.
Minimizing the tracking error keeps the curved space line of sight to the
target location aligned with the vehicle's velocity vector. The optimized
guidance system will then induce trajectories that bring the vehicle to the
target location with a high degree of accuracy at the designated terminal
speed, while satisfying heating rate, load, and dynamic pressure constraints.
We demonstrate the adaptability of the guidance system by testing over flight
conditions that were not experienced during optimization. The guidance system's
performance is then compared to that of a linear quadratic regulator tracking
an optimal trajectory.

    

### [[2107.14800] ChrEnTranslate: Cherokee-English Machine Translation Demo with Quality Estimation and Corrective Feedback](http://arxiv.org/abs/2107.14800)


  We introduce ChrEnTranslate, an online machine translation demonstration
system for translation between English and an endangered language Cherokee. It
supports both statistical and neural translation models as well as provides
quality estimation to inform users of reliability, two user feedback interfaces
for experts and common users respectively, example inputs to collect human
translations for monolingual data, word alignment visualization, and relevant
terms from the Cherokee-English dictionary. The quantitative evaluation
demonstrates that our backbone translation models achieve state-of-the-art
translation performance and our quality estimation well correlates with both
BLEU and human judgment. By analyzing 216 pieces of expert feedback, we find
that NMT is preferable because it copies less than SMT, and, in general,
current models can translate fragments of the source sentence but make major
mistakes. When we add these 216 expert-corrected parallel texts into the
training set and retrain models, equal or slightly better performance is
observed, which demonstrates indicates the potential of human-in-the-loop
learning. Our online demo is at this https URL our code is
open-sourced at this https URL and our data is
available at this https URL.

    

### [[2106.07221] Certification of embedded systems based on Machine Learning: A survey](http://arxiv.org/abs/2106.07221)


  Advances in machine learning (ML) open the way to innovating functions in the
avionic domain, such as navigation/surveillance assistance (e.g. vision-based
navigation, obstacle sensing, virtual sensing), speechto-text applications,
autonomous flight, predictive maintenance or cockpit assistance. Current
certification standards and practices, which were defined and refined decades
over decades with classical programming in mind, do not however support this
new development paradigm. This article provides an overview of the main
challenges raised by the use ML in the demonstration of compliance with
regulation requirements, and a survey of literature relevant to these
challenges, with particular focus on the issues of robustness and
explainability of ML results.

    

### [[2106.15307] Deep Random Projection Outlyingness for Unsupervised Anomaly Detection](http://arxiv.org/abs/2106.15307)


  Random projection is a common technique for designing algorithms in a variety
of areas, including information retrieval, compressive sensing and measuring of
outlyingness. In this work, the original random projection outlyingness measure
is modified and associated with a neural network to obtain an unsupervised
anomaly detection method able to handle multimodal normality. Theoretical and
experimental arguments are presented to justify the choice of the anomaly score
estimator. The performance of the proposed neural network approach is
comparable to a state-of-the-art anomaly detection method. Experiments
conducted on the MNIST, Fashion-MNIST and CIFAR-10 datasets show the relevance
of the proposed approach.

    

### [[2107.14655] The bitwise operations in relation to the concept of set](http://arxiv.org/abs/2107.14655)


  We contemplate this article to help the teachers of programming in his
aspiration for giving some appropriate and interesting examples. The work will
be especially useful for students-future programmers, and for their lecturers.
Some of the strong sides of these programming languages C/C++ and Java are the
possibilities of low-level programming. Some of the means for this possibility
are the introduced standard bitwise operations, with the help of which, it is
possible to directly operate with every bit of an arbitrary variable situated
in the computers memory. In the current study, we are going to describe some
methodical aspects for work with the bitwise operations and we will discuss the
benefit of using bitwise operations in programming. The article shows some
advantages of using bitwise operations, realizing various operations with sets.

    

### [[1811.04196] A Domain Theory for Statistical Probabilistic Programming](http://arxiv.org/abs/1811.04196)


  We give an adequate denotational semantics for languages with recursive
higher-order types, continuous probability distributions, and soft constraints.
These are expressive languages for building Bayesian models of the kinds used
in computational statistics and machine learning. Among them are untyped
languages, similar to Church and WebPPL, because our semantics allows recursive
mixed-variance datatypes. Our semantics justifies important program
equivalences including commutativity.
Our new semantic model is based on `quasi-Borel predomains'. These are a
mixture of chain-complete partial orders (cpos) and quasi-Borel spaces.
Quasi-Borel spaces are a recent model of probability theory that focuses on
sets of admissible random elements. Probability is traditionally treated in cpo
models using probabilistic powerdomains, but these are not known to be
commutative on any class of cpos with higher order functions. By contrast,
quasi-Borel predomains do support both a commutative probabilistic powerdomain
and higher-order functions. As we show, quasi-Borel predomains form both a
model of Fiore's axiomatic domain theory and a model of Kock's synthetic
measure theory.

    