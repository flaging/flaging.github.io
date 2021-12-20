
## 2021-12-20

### [[2112.09328] D3PG: Dirichlet DDGP for Task Partitioning and Offloading with Constrained Hybrid Action Space in Mobile Edge Computing](http://arxiv.org/abs/2112.09328)


  Mobile Edge Computing (MEC) has been regarded as a promising paradigm to
reduce service latency for data processing in the Internet of Things, by
provisioning computing resources at the network edge. In this work, we jointly
optimize the task partitioning and computational power allocation for
computation offloading in a dynamic environment with multiple IoT devices and
multiple edge servers. We formulate the problem as a Markov decision process
with constrained hybrid action space, which cannot be well handled by existing
deep reinforcement learning (DRL) algorithms. Therefore, we develop a novel
Deep Reinforcement Learning called Dirichlet Deep Deterministic Policy Gradient
(D3PG), which is built on Deep Deterministic Policy Gradient (DDPG) to solve
the problem. The developed model can learn to solve multi-objective
optimization, including maximizing the number of tasks processed before
expiration and minimizing the energy cost and service latency.} More
importantly, D3PG can effectively deal with constrained distribution-continuous
hybrid action space, where the distribution variables are for the task
partitioning and offloading, while the continuous variables are for
computational frequency control. Moreover, the D3PG can address many similar
issues in MEC and general reinforcement learning problems. Extensive simulation
results show that the proposed D3PG outperforms the state-of-art methods.

    

### [[2112.09330] Node Failure Localisation Problem for Load Balancing Dynamic Networks](http://arxiv.org/abs/2112.09330)


  Network tomography has been used as an approach to the Node Failure
Localisation problem, whereby misbehaving subsets of nodes in a network are to
be determined. Typically approaches in the literature assume a statically
routed network, permitting linear algebraic arguments. In this work, a load
balancing, dynamically routed network is studied, necessitating a stochastic
representation of network dynamics. A network model was developed, permitting a
novel application of Markov Chain Monte Carlo (MCMC) inference to the Node
Failure Localisation (NFL) problem, and the assessment of monitor placement
choices. Two nuanced monitor placement algorithms, including one designed for
the NFL problem by Ma et al. 2014 were tested, with the published algorithm
performing significantly better.

    

### [[2112.09393] An Online Orchestration Mechanism for General-Purpose Edge Computing](http://arxiv.org/abs/2112.09393)


  In recent years, the fast development of mobile communications and cloud
systems has substantially promoted edge computing. By pushing server resources
to the edge, mobile service providers can deliver their content and services
with enhanced performance, and mobile-network carriers can alleviate congestion
in the core networks. Although edge computing has been attracting much
interest, most current research is application-specific, and analysis is
lacking from a business perspective of edge cloud providers (ECPs) that provide
general-purpose edge cloud services to mobile service providers and users. In
this article, we present a vision of general-purpose edge computing realized by
multiple interconnected edge clouds, analyzing the business model from the
viewpoint of ECPs and identifying the main issues to address to maximize
benefits for ECPs. Specifically, we formalize the long-term revenue of ECPs as
a function of server-resource allocation and public data-placement decisions
subject to the amount of physical resources and inter-cloud data-transportation
cost constraints. To optimize the long-term objective, we propose an online
framework that integrates the drift-plus-penalty and primal-dual methods. With
theoretical analysis and simulations, we show that the proposed method
approximates the optimal solution in a challenging environment without having
future knowledge of the system.

    

### [[2112.09403] WIP: Exploring DSME MAC for LoRa -- A System Integration and First Evaluation](http://arxiv.org/abs/2112.09403)


  LoRa is a popular wireless technology that enables low-throughput (bytes)
long-range communication (km) at low energy consumption (mW). Its transmission,
though, is on one side prone to interference during long on-air times, and on
the other side subject to duty cycle restrictions. LoRaWAN defines a MAC and a
vertical stack on top of LoRa. LoRaWAN circumvents the above limitations by
imposing a centralized network architecture, which heavily reduces downlink
capacity and prevents peer-to-peer communication. This makes it unusable for
many deployments. The Deterministic and Synchronous Multichannel Extension
(DSME) of IEEE 802.15.4e benefits of time-slotted communication and
peer-to-peer communication and has the potential to overcome LoRaWAN
limitations. In this work, we implement DSME on top of LoRa in the open source
IoT OS RIOT and open the field for first evaluation experiments on real
hardware. Initial results indicate that DSME-LoRa not only enables reliable
peer-to-peer communication for constrained IoT devices, but also scales with an
increasing number of nodes.

    

### [[2112.09407] Communication-oriented Model Fine-tuning for Packet-loss Resilient Distributed Inference under Highly Lossy IoT Networks](http://arxiv.org/abs/2112.09407)


  The distributed inference (DI) framework has gained traction as a technique
for real-time applications empowered by cutting-edge deep machine learning (ML)
on resource-constrained Internet of things (IoT) devices. In DI, computational
tasks are offloaded from the IoT device to the edge server via lossy IoT
networks. However, generally, there is a communication system-level trade-off
between communication latency and reliability; thus, to provide accurate DI
results, a reliable and high-latency communication system is required to be
adapted, which results in non-negligible end-to-end latency of the DI. This
motivated us to improve the trade-off between the communication latency and
accuracy by efforts on ML techniques. Specifically, we have proposed a
communication-oriented model tuning (COMtune), which aims to achieve highly
accurate DI with low-latency but unreliable communication links. In COMtune,
the key idea is to fine-tune the ML model by emulating the effect of unreliable
communication links through the application of the dropout technique. This
enables the DI system to obtain robustness against unreliable communication
links. Our ML experiments revealed that COMtune enables accurate predictions
with low latency and under lossy networks.

    

### [[2112.09450] Decentralized Identifiers and Self-sovereign Identity in 6G](http://arxiv.org/abs/2112.09450)


  One of the key challenges for mobile network operators in the future will be
to bring together a wide range of new players in the mobile network market
under a common umbrella and to orchestrate their innovative technologies to
provide economically viable and seamless mobile connectivity to the mobile
subscribers. With each new player, be it a cloud, edge or hardware provider,
the need for interfaces with secure authentication and authorization mechanisms
increases, as does the complexity and operational costs of the public key
infrastructures required for the associated identity and key management. While
today's centralized public key infrastructures have proven themselves to be
technically feasible in confined and trusted spaces, they do not provide the
required security once centralized identity providers must be avoided, e.g.,
because of limited cross-domain interoperability or national data protection
legislation, and state-dependent certification authorities can't be commonly
trusted, e.g., because of geopolitical reasons. Recent decentralized identity
management concepts, such as the W3C proposed recommendation of Decentralized
Identifiers, provide a secure, tamper-proof, and cross-domain identity
management alternative for future multitenancy 6G networks without relying on
identity provider or certification authorities. This article introduces the
concept of Decentralized Identifiers together with the principles of
Self-sovereign Identity and discusses opportunities and potential benefits of
their application and usage for cross-actor and privacy-preserving identity and
key management in the next mobile network generation 6G.

    

### [[2112.09489] A Belief Propagation Solution for Beam Coordination in MmWave Vehicular Networks](http://arxiv.org/abs/2112.09489)


  Millimeter-wave communication is widely seen as a promising option to
increase the capacity of vehicular networks, where it is expected that
connected cars will soon need to transmit and receive large amounts of data.
Due to harsh propagation conditions, mmWave systems resort to narrow beams to
serve their users, and such beams need to be configured according to traffic
demand and its spatial distribution, as well as interference. In this work, we
address the beam management problem, considering an urban vehicular network
composed of gNBs. We first build an accurate, yet tractable, system model and
formulate an optimization problem aiming at maximizing the total network data
rate while accounting for the stochastic nature of the network scenario. Then
we develop a graph-based model capturing the main system characteristics and
use it to develop a belief propagation algorithmic framework, called CRAB, that
has low complexity and, hence, can effectively cope with large-scale scenarios.
We assess the performance of our approach under real-world settings and show
that, in comparison to state-of-the-art alternatives, CRAB provides on average
a 50% improvement in the amount of data transferred by the single gNBs and up
to 30% better user coverage.

    

### [[2112.09559] ColO-RAN: Developing Machine Learning-based xApps for Open RAN Closed-loop Control on Programmable Experimental Platforms](http://arxiv.org/abs/2112.09559)


  In spite of the new opportunities brought about by the Open RAN, advances in
ML-based network automation have been slow, mainly because of the
unavailability of large-scale datasets and experimental testing infrastructure.
This slows down the development and widespread adoption of Deep Reinforcement
Learning (DRL) agents on real networks, delaying progress in intelligent and
autonomous RAN control. In this paper, we address these challenges by proposing
practical solutions and software pipelines for the design, training, testing,
and experimental evaluation of DRL-based closed-loop control in the Open RAN.
We introduce ColO-RAN, the first publicly-available large-scale O-RAN testing
framework with software-defined radios-in-the-loop. Building on the scale and
computational capabilities of the Colosseum wireless network emulator, ColO-RAN
enables ML research at scale using O-RAN components, programmable base
stations, and a "wireless data factory". Specifically, we design and develop
three exemplary xApps for DRL-based control of RAN slicing, scheduling and
online model training, and evaluate their performance on a cellular network
with 7 softwarized base stations and 42 users. Finally, we showcase the
portability of ColO-RAN to different platforms by deploying it on Arena, an
indoor programmable testbed. Extensive results from our first-of-its-kind
large-scale evaluation highlight the benefits and challenges of DRL-based
adaptive control. They also provide insights on the development of wireless DRL
pipelines, from data analysis to the design of DRL agents, and on the tradeoffs
associated to training on a live RAN. ColO-RAN and the collected large-scale
dataset will be made publicly available to the research community.

    

### [[2007.04193] Mobility driven Cloud-Fog-Edge Framework for Location-aware Services: A Comprehensive Review](http://arxiv.org/abs/2007.04193)


  With the pervasiveness of IoT devices, smart-phones and improvement of
location-tracking technologies huge volume of heterogeneous geo-tagged
(location specific) data is generated which facilitates several location-aware
services. The analytics with this spatio-temporal (having location and time
dimensions) datasets provide varied important services such as, smart
transportation, emergency services (health-care, national defence or urban
planning). While cloud paradigm is suitable for the capability of storage and
computation, the major bottleneck is network connectivity loss. In
time-critical application, where real-time response is required for emergency
service-provisioning, such connectivity issues increases the latency and thus
affects the overall quality of system (QoS). To overcome the issue, fog/ edge
topology has emerged, where partial computation is carried out in the edge of
the network to reduce the delay in communication. Such fog/ edge based system
complements the cloud technology and extends the features of the system. This
chapter discusses cloud-fog-edge based hierarchical collaborative framework,
where several components are deployed to improve the QoS. On the other side.
mobility is another critical factor to enhance the efficacy of such
location-aware service provisioning. Therefore, this chapter discusses the
concerns and challenges associated with mobility-driven cloud-fog-edge based
framework to provide several location-aware services to the end-users
efficiently.

    

### [[2112.09153] An Empirical Investigation of the Role of Pre-training in Lifelong Learning](http://arxiv.org/abs/2112.09153)


  The lifelong learning paradigm in machine learning is an attractive
alternative to the more prominent isolated learning scheme not only due to its
resemblance to biological learning, but also its potential to reduce energy
waste by obviating excessive model re-training. A key challenge to this
paradigm is the phenomenon of catastrophic forgetting. With the increasing
popularity and success of pre-trained models in machine learning, we pose the
question: What role does pre-training play in lifelong learning, specifically
with respect to catastrophic forgetting? We investigate existing methods in the
context of large, pre-trained models and evaluate their performance on a
variety of text and image classification tasks, including a large-scale study
using a novel dataset of 15 diverse NLP tasks. Across all settings, we observe
that generic pre-training implicitly alleviates the effects of catastrophic
forgetting when learning multiple tasks sequentially compared to randomly
initialized models. We then further investigate why pre-training alleviates
forgetting in this setting. We study this phenomenon by analyzing the loss
landscape, finding that pre-trained weights appear to ease forgetting by
leading to wider minima. Based on this insight, we propose jointly optimizing
for current task loss and loss basin sharpness in order to explicitly encourage
wider basins during sequential fine-tuning. We show that this optimization
approach leads to performance comparable to the state-of-the-art in
task-sequential continual learning across multiple settings, without retaining
a memory that scales in size with the number of tasks.

    

### [[2112.09159] Implementation of a Binary Neural Network on a Passive Array of Magnetic Tunnel Junctions](http://arxiv.org/abs/2112.09159)


  The increasing scale of neural networks and their growing application space
have produced demand for more energy- and memory-efficient
artificial-intelligence-specific hardware. Avenues to mitigate the main issue,
the von Neumann bottleneck, include in-memory and near-memory architectures, as
well as algorithmic approaches. Here we leverage the low-power and the
inherently binary operation of magnetic tunnel junctions (MTJs) to demonstrate
neural network hardware inference based on passive arrays of MTJs. In general,
transferring a trained network model to hardware for inference is confronted by
degradation in performance due to device-to-device variations, write errors,
parasitic resistance, and nonidealities in the substrate. To quantify the
effect of these hardware realities, we benchmark 300 unique weight matrix
solutions of a 2-layer perceptron to classify the Wine dataset for both
classification accuracy and write fidelity. Despite device imperfections, we
achieve software-equivalent accuracy of up to 95.3 % with proper tuning of
network parameters in 15 x 15 MTJ arrays having a range of device sizes. The
success of this tuning process shows that new metrics are needed to
characterize the performance and quality of networks reproduced in mixed signal
hardware.

    

### [[2112.09161] Constraint-based graph network simulator](http://arxiv.org/abs/2112.09161)


  In the rapidly advancing area of learned physical simulators, nearly all
methods train forward models that directly predict future states from input
states. However, many traditional simulation engines use a constraint-based
approach instead of direct prediction. Here we present a framework for
constraint-based learned simulation, where a scalar constraint function is
implemented as a neural network, and future predictions are computed as the
solutions to optimization problems under these learned constraints. We
implement our method using a graph neural network as the constraint function
and gradient descent as the constraint solver. The architecture can be trained
by standard backpropagation. We test the model on a variety of challenging
physical domains, including simulated ropes, bouncing balls, colliding
irregular shapes and splashing fluids. Our model achieves better or comparable
performance to top learned simulators. A key advantage of our model is the
ability to generalize to more solver iterations at test time to improve the
simulation accuracy. We also show how hand-designed constraints can be added at
test time to satisfy objectives which were not present in the training data,
which is not possible with forward approaches. Our constraint-based framework
is applicable to any setting where forward learned simulators are used, and
demonstrates how learned simulators can leverage additional inductive biases as
well as the techniques from the field of numerical methods.

    

### [[2112.09164] High Fidelity Visualization of What Your Self-Supervised Representation Knows About](http://arxiv.org/abs/2112.09164)


  Discovering what is learned by neural networks remains a challenge. In
self-supervised learning, classification is the most common task used to
evaluate how good a representation is. However, relying only on such downstream
task can limit our understanding of how much information is retained in the
representation of a given input. In this work, we showcase the use of a
conditional diffusion based generative model (RCDM) to visualize
representations learned with self-supervised models. We further demonstrate how
this model's generation quality is on par with state-of-the-art generative
models while being faithful to the representation used as conditioning. By
using this new tool to analyze self-supervised models, we can show visually
that i) SSL (backbone) representation are not really invariant to many data
augmentation they were trained on. ii) SSL projector embedding appear too
invariant for tasks like classifications. iii) SSL representations are more
robust to small adversarial perturbation of their inputs iv) there is an
inherent structure learned with SSL model that can be used for image
manipulation.

    

### [[2112.09165] ALEBk: Feasibility Study of Attention Level Estimation via Blink Detection applied to e-Learning](http://arxiv.org/abs/2112.09165)


  This work presents a feasibility study of remote attention level estimation
based on eye blink frequency. We first propose an eye blink detection system
based on Convolutional Neural Networks (CNNs), very competitive with respect to
related works. Using this detector, we experimentally evaluate the relationship
between the eye blink rate and the attention level of students captured during
online sessions. The experimental framework is carried out using a public
multimodal database for eye blink detection and attention level estimation
called mEBAL, which comprises data from 38 students and multiples acquisition
sensors, in particular, i) an electroencephalogram (EEG) band which provides
the time signals coming from the student's cognitive information, and ii) RGB
and NIR cameras to capture the students face gestures. The results achieved
suggest an inverse correlation between the eye blink frequency and the
attention level. This relation is used in our proposed method called ALEBk for
estimating the attention level as the inverse of the eye blink frequency. Our
results open a new research line to introduce this technology for attention
level estimation on future e-learning platforms, among other applications of
this kind of behavioral biometrics based on face analysis.

    

### [[2112.09169] On Optimizing Interventions in Shared Autonomy](http://arxiv.org/abs/2112.09169)


  Shared autonomy refers to approaches for enabling an autonomous agent to
collaborate with a human with the aim of improving human performance. However,
besides improving performance, it may often also be beneficial that the agent
concurrently accounts for preserving the user's experience or satisfaction of
collaboration. In order to address this additional goal, we examine approaches
for improving the user experience by constraining the number of interventions
by the autonomous agent. We propose two model-free reinforcement learning
methods that can account for both hard and soft constraints on the number of
interventions. We show that not only does our method outperform the existing
baseline, but also eliminates the need to manually tune a black-box
hyperparameter for controlling the level of assistance. We also provide an
in-depth analysis of intervention scenarios in order to further illuminate
system understanding.

    

### [[2112.09172] An Audio-Visual Dataset and Deep Learning Frameworks for Crowded Scene Classification](http://arxiv.org/abs/2112.09172)


  This paper presents a task of audio-visual scene classification (SC) where
input videos are classified into one of five real-life crowded scenes: 'Riot',
'Noise-Street', 'Firework-Event', 'Music-Event', and 'Sport-Atmosphere'. To
this end, we firstly collect an audio-visual dataset (videos) of these five
crowded contexts from Youtube (in-the-wild scenes). Then, a wide range of deep
learning frameworks are proposed to deploy either audio or visual input data
independently. Finally, results obtained from high-performed deep learning
frameworks are fused to achieve the best accuracy score. Our experimental
results indicate that audio and visual input factors independently contribute
to the SC task's performance. Significantly, an ensemble of deep learning
frameworks exploring either audio or visual input data can achieve the best
accuracy of 95.7%.

    

### [[2112.09175] Effective prevention of semantic drift as angular distance in memory-less continual deep neural networks](http://arxiv.org/abs/2112.09175)


  Lifelong machine learning or continual learning models attempt to learn
incrementally by accumulating knowledge across a sequence of tasks. Therefore,
these models learn better and faster. They are used in various intelligent
systems that have to interact with humans or any dynamic environment e.g.,
chatbots and self-driving cars. Memory-less approach is more often used with
deep neural networks that accommodates incoming information from tasks within
its architecture. It allows them to perform well on all the seen tasks. These
models suffer from semantic drift or the plasticity-stability dilemma. The
existing models use Minkowski distance measures to decide which nodes to
freeze, update or duplicate. These distance metrics do not provide better
separation of nodes as they are susceptible to high dimensional sparse vectors.
In our proposed approach, we use angular distance to evaluate the semantic
drift in individual nodes that provide better separation of nodes and thus
better balancing between stability and plasticity. The proposed approach
outperforms state-of-the art models by maintaining higher accuracy on standard
datasets.

    

### [[2112.09181] Approximation of functions with one-bit neural networks](http://arxiv.org/abs/2112.09181)


  This paper examines the approximation capabilities of coarsely quantized
neural networks -- those whose parameters are selected from a small set of
allowable values. We show that any smooth multivariate function can be
arbitrarily well approximated by an appropriate coarsely quantized neural
network and provide a quantitative approximation rate. For the quadratic
activation, this can be done with only a one-bit alphabet; for the ReLU
activation, we use a three-bit alphabet. The main theorems rely on important
properties of Bernstein polynomials. We prove new results on approximation of
functions with Bernstein polynomials, noise-shaping quantization on the
Bernstein basis, and implementation of the Bernstein polynomials by coarsely
quantized neural networks.

    

### [[2112.09182] Predicting Shallow Water Dynamics using Echo-State Networks with Transfer Learning](http://arxiv.org/abs/2112.09182)


  In this paper we demonstrate that reservoir computing can be used to learn
the dynamics of the shallow-water equations. In particular, while most previous
applications of reservoir computing have required training on a particular
trajectory to further predict the evolution along that trajectory alone, we
show the capability of reservoir computing to predict trajectories of the
shallow-water equations with initial conditions not seen in the training
process. However, in this setting, we find that the performance of the network
deteriorates for initial conditions with ambient conditions (such as total
water height and average velocity) that are different from those in the
training dataset. To circumvent this deficiency, we introduce a transfer
learning approach wherein a small additional training step with the relevant
ambient conditions is used to improve the predictions.

    

### [[2112.09195] Mitigating the Bias of Centered Objects in Common Datasets](http://arxiv.org/abs/2112.09195)


  Convolutional networks are considered shift invariant, but it was
demonstrated that their response may vary according to the exact location of
the objects. In this paper we will demonstrate that most commonly investigated
datasets have a bias, where objects are over-represented at the center of the
image during training. This bias and the boundary condition of these networks
can have a significant effect on the performance of these architectures and
their accuracy drops significantly as an object approaches the boundary. We
will also demonstrate how this effect can be mitigated with data augmentation
techniques.

    

### [[2112.09196] Benchmarking Uncertainty Qualification on Biosignal Classification Tasks under Dataset Shift](http://arxiv.org/abs/2112.09196)


  A biosignal is a signal that can be continuously measured from human bodies,
such as respiratory sounds, heart activity (ECG), brain waves (EEG), etc, based
on which, machine learning models have been developed with very promising
performance for automatic disease detection and health status monitoring.
However, dataset shift, i.e., data distribution of inference varies from the
distribution of the training, is not uncommon for real biosignal-based
applications. To improve the robustness, probabilistic models with uncertainty
qualification are adapted to capture how reliable a prediction is. Yet,
assessing the quality of the estimated uncertainty remains a challenge. In this
work, we propose a framework to evaluate the capability of the estimated
uncertainty in capturing different types of biosignal dataset shifts with
various degrees. In particular, we use three classification tasks based on
respiratory sounds and electrocardiography signals to benchmark five
representative uncertainty qualification methods. Extensive experiments show
that, although Ensemble and Bayesian models could provide relatively better
uncertainty estimations under dataset shifts, all tested models fail to meet
the promise in trustworthy prediction and model calibration. Our work paves the
way for a comprehensive evaluation for any newly developed biosignal
classifiers.

    

### [[2112.09214] Sparse Coding with Multi-Layer Decoders using Variance Regularization](http://arxiv.org/abs/2112.09214)


  Sparse coding with an $l_1$ penalty and a learned linear dictionary requires
regularization of the dictionary to prevent a collapse in the $l_1$ norms of
the codes. Typically, this regularization entails bounding the Euclidean norms
of the dictionary's elements. In this work, we propose a novel sparse coding
protocol which prevents a collapse in the codes without the need to regularize
the decoder. Our method regularizes the codes directly so that each latent code
component has variance greater than a fixed threshold over a set of sparse
representations for a given set of inputs. Furthermore, we explore ways to
effectively train sparse coding systems with multi-layer decoders since they
can model more complex relationships than linear dictionaries. In our
experiments with MNIST and natural image patches, we show that decoders learned
with our approach have interpretable features both in the linear and
multi-layer case. Moreover, we show that sparse autoencoders with multi-layer
decoders trained using our variance regularization method produce higher
quality reconstructions with sparser representations when compared to
autoencoders with linear dictionaries. Additionally, sparse representations
obtained with our variance regularization approach are useful in the downstream
tasks of denoising and classification in the low-data regime.

    

### [[2112.09217] Marginalization in Bayesian Networks: Integrating Exact and Approximate Inference](http://arxiv.org/abs/2112.09217)


  Bayesian Networks are probabilistic graphical models that can compactly
represent dependencies among random variables. Missing data and hidden
variables require calculating the marginal probability distribution of a subset
of the variables. While knowledge of the marginal probability distribution is
crucial for various problems in statistics and machine learning, its exact
computation is generally not feasible for categorical variables due to the
NP-hardness of this task. We develop a divide-and-conquer approach using the
graphical properties of Bayesian networks to split the computation of the
marginal probability distribution into sub-calculations of lower
dimensionality, reducing the overall computational complexity. Exploiting this
property, we present an efficient and scalable algorithm for estimating the
marginal probability distribution for categorical variables. The novel method
is compared against state-of-the-art approximate inference methods in a
benchmarking study, where it displays superior performance. As an immediate
application, we demonstrate how the marginal probability distribution can be
used to classify incomplete data against Bayesian networks and use this
approach for identifying the cancer subtype of kidney cancer patient samples.

    

### [[2112.09220] Sim2Real Docs: Domain Randomization for Documents in Natural Scenes using Ray-traced Rendering](http://arxiv.org/abs/2112.09220)


  In the past, computer vision systems for digitized documents could rely on
systematically captured, high-quality scans. Today, transactions involving
digital documents are more likely to start as mobile phone photo uploads taken
by non-professionals. As such, computer vision for document automation must now
account for documents captured in natural scene contexts. An additional
challenge is that task objectives for document processing can be highly
use-case specific, which makes publicly-available datasets limited in their
utility, while manual data labeling is also costly and poorly translates
between use cases.
To address these issues we created Sim2Real Docs - a framework for
synthesizing datasets and performing domain randomization of documents in
natural scenes. Sim2Real Docs enables programmatic 3D rendering of documents
using Blender, an open source tool for 3D modeling and ray-traced rendering. By
using rendering that simulates physical interactions of light, geometry,
camera, and background, we synthesize datasets of documents in a natural scene
context. Each render is paired with use-case specific ground truth data
specifying latent characteristics of interest, producing unlimited fit-for-task
training data. The role of machine learning models is then to solve the inverse
problem posed by the rendering pipeline. Such models can be further iterated
upon with real-world data by either fine tuning or making adjustments to domain
randomization parameters.

    

### [[2112.09231] Two-view Graph Neural Networks for Knowledge Graph Completion](http://arxiv.org/abs/2112.09231)


  In this paper, we introduce a novel GNN-based knowledge graph embedding
model, named WGE, to capture entity-focused graph structure and
relation-focused graph structure. In particular, given the knowledge graph, WGE
builds a single undirected entity-focused graph that views entities as nodes.
In addition, WGE also constructs another single undirected graph from
relation-focused constraints, which views entities and relations as nodes. WGE
then proposes a new architecture of utilizing two vanilla GNNs directly on
these two single graphs to better update vector representations of entities and
relations, followed by a weighted score function to return the triple scores.
Experimental results show that WGE obtains state-of-the-art performances on
three new and challenging benchmark datasets CoDEx for knowledge graph
completion.

    

### [[2112.09243] Confidence-Aware Subject-to-Subject Transfer Learning for Brain-Computer Interface](http://arxiv.org/abs/2112.09243)


  The inter/intra-subject variability of electroencephalography (EEG) makes the
practical use of the brain-computer interface (BCI) difficult. In general, the
BCI system requires a calibration procedure to tune the model every time the
system is used. This problem is recognized as a major obstacle to BCI, and to
overcome it, approaches based on transfer learning (TL) have recently emerged.
However, many BCI paradigms are limited in that they consist of a structure
that shows labels first and then measures "imagery", the negative effects of
source subjects containing data that do not contain control signals have been
ignored in many cases of the subject-to-subject TL process. The main purpose of
this paper is to propose a method of excluding subjects that are expected to
have a negative impact on subject-to-subject TL training, which generally uses
data from as many subjects as possible. In this paper, we proposed a BCI
framework using only high-confidence subjects for TL training. In our
framework, a deep neural network selects useful subjects for the TL process and
excludes noisy subjects, using a co-teaching algorithm based on the small-loss
trick. We experimented with leave-one-subject-out validation on two public
datasets (2020 international BCI competition track 4 and OpenBMI dataset). Our
experimental results showed that confidence-aware TL, which selects subjects
with small loss instances, improves the generalization performance of BCI.

    

### [[2112.09245] Automated Deep Learning: Neural Architecture Search Is Not the End](http://arxiv.org/abs/2112.09245)


  Deep learning (DL) has proven to be a highly effective approach for
developing models in diverse contexts, including visual perception, speech
recognition, and machine translation. However, the end-to-end process for
applying DL is not trivial. It requires grappling with problem formulation and
context understanding, data engineering, model development, deployment,
continuous monitoring and maintenance, and so on. Moreover, each of these steps
typically relies heavily on humans, in terms of both knowledge and
interactions, which impedes the further advancement and democratization of DL.
Consequently, in response to these issues, a new field has emerged over the
last few years: automated deep learning (AutoDL). This endeavor seeks to
minimize the need for human involvement and is best known for its achievements
in neural architecture search (NAS), a topic that has been the focus of several
surveys. That stated, NAS is not the be-all and end-all of AutoDL. Accordingly,
this review adopts an overarching perspective, examining research efforts into
automation across the entirety of an archetypal DL workflow. In so doing, this
work also proposes a comprehensive set of ten criteria by which to assess
existing work in both individual publications and broader research areas. These
criteria are: novelty, solution quality, efficiency, stability,
interpretability, reproducibility, engineering quality, scalability,
generalizability, and eco-friendliness. Thus, ultimately, this review provides
an evaluative overview of AutoDL in the early 2020s, identifying where future
opportunities for progress may exist.

    

### [[2112.09266] Link-Intensive Alignment for Incomplete Knowledge Graphs](http://arxiv.org/abs/2112.09266)


  Knowledge graph (KG) alignment - the task of recognizing entities referring
to the same thing in different KGs - is recognized as one of the most important
operations in the field of KG construction and completion. However, existing
alignment techniques often assume that the input KGs are complete and
isomorphic, which is not true due to the real-world heterogeneity in the
domain, size, and sparsity. In this work, we address the problem of aligning
incomplete KGs with representation learning. Our KG embedding framework
exploits two feature channels: transitivity-based and proximity-based. The
former captures the consistency constraints between entities via translation
paths, while the latter captures the neighbourhood structure of KGs via
attention guided relation-aware graph neural network. The two feature channels
are jointly learned to exchange important features between the input KGs while
enforcing the output representations of the input KGs in the same embedding
space. Also, we develop a missing links detector that discovers and recovers
the missing links in the input KGs during the training process, which helps
mitigate the incompleteness issue and thus improve the compatibility of the
learned representations. The embeddings then are fused to generate the
alignment result, and the high-confidence matched node pairs are updated to the
pre-aligned supervision data to improve the embeddings gradually. Empirical
results show that our model is up to 15.2\% more accurate than the SOTA and is
robust against different levels of incompleteness. We also demonstrate that the
knowledge exchanging between the KGs helps reveal the unseen facts from
knowledge graphs (a.k.a. knowledge completion), with the result being 3.5\%
higher than the SOTA knowledge graph completion techniques.

    

### [[2112.09277] DNA: Dynamic Network Augmentation](http://arxiv.org/abs/2112.09277)


  In many classification problems, we want a classifier that is robust to a
range of non-semantic transformations. For example, a human can identify a dog
in a picture regardless of the orientation and pose in which it appears. There
is substantial evidence that this kind of invariance can significantly improve
the accuracy and generalization of machine learning models. A common technique
to teach a model geometric invariances is to augment training data with
transformed inputs. However, which invariances are desired for a given
classification task is not always known. Determining an effective data
augmentation policy can require domain expertise or extensive data
pre-processing. Recent efforts like AutoAugment optimize over a parameterized
search space of data augmentation policies to automate the augmentation
process. While AutoAugment and similar methods achieve state-of-the-art
classification accuracy on several common datasets, they are limited to
learning one data augmentation policy. Often times different classes or
features call for different geometric invariances. We introduce Dynamic Network
Augmentation (DNA), which learns input-conditional augmentation policies.
Augmentation parameters in our model are outputs of a neural network and are
implicitly learned as the network weights are updated. Our model allows for
dynamic augmentation policies and performs well on data with geometric
transformations conditional on input features.

    

### [[2112.09279] A Robust Optimization Approach to Deep Learning](http://arxiv.org/abs/2112.09279)


  Many state-of-the-art adversarial training methods leverage upper bounds of
the adversarial loss to provide security guarantees. Yet, these methods require
computations at each training step that can not be incorporated in the gradient
for backpropagation. We introduce a new, more principled approach to
adversarial training based on a closed form solution of an upper bound of the
adversarial loss, which can be effectively trained with backpropagation. This
bound is facilitated by state-of-the-art tools from robust optimization. We
derive two new methods with our approach. The first method (Approximated Robust
Upper Bound or aRUB) uses the first order approximation of the network as well
as basic tools from linear robust optimization to obtain an approximate upper
bound of the adversarial loss that can be easily implemented. The second method
(Robust Upper Bound or RUB), computes an exact upper bound of the adversarial
loss. Across a variety of tabular and vision data sets we demonstrate the
effectiveness of our more principled approach -- RUB is substantially more
robust than state-of-the-art methods for larger perturbations, while aRUB
matches the performance of state-of-the-art methods for small perturbations.
Also, both RUB and aRUB run faster than standard adversarial training (at the
expense of an increase in memory). All the code to reproduce the results can be
found at this https URL.

    

### [[2112.09290] PeopleSansPeople: A Synthetic Data Generator for Human-Centric Computer Vision](http://arxiv.org/abs/2112.09290)


  In recent years, person detection and human pose estimation have made great
strides, helped by large-scale labeled datasets. However, these datasets had no
guarantees or analysis of human activities, poses, or context diversity.
Additionally, privacy, legal, safety, and ethical concerns may limit the
ability to collect more human data. An emerging alternative to real-world data
that alleviates some of these issues is synthetic data. However, creation of
synthetic data generators is incredibly challenging and prevents researchers
from exploring their usefulness. Therefore, we release a human-centric
synthetic data generator PeopleSansPeople which contains simulation-ready 3D
human assets, a parameterized lighting and camera system, and generates 2D and
3D bounding box, instance and semantic segmentation, and COCO pose labels.
Using PeopleSansPeople, we performed benchmark synthetic data training using a
Detectron2 Keypoint R-CNN variant [1]. We found that pre-training a network
using synthetic data and fine-tuning on target real-world data (few-shot
transfer to limited subsets of COCO-person train [2]) resulted in a keypoint AP
of $60.37 \pm 0.48$ (COCO test-dev2017) outperforming models trained with the
same real data alone (keypoint AP of $55.80$) and pre-trained with ImageNet
(keypoint AP of $57.50$). This freely-available data generator should enable a
wide range of research into the emerging field of simulation to real transfer
learning in the critical area of human-centric computer vision.

    

### [[2112.09293] A Comparative Study of Detecting Anomalies in Time Series Data Using LSTM and TCN Models](http://arxiv.org/abs/2112.09293)


  There exist several data-driven approaches that enable us model time series
data including traditional regression-based modeling approaches (i.e., ARIMA).
Recently, deep learning techniques have been introduced and explored in the
context of time series analysis and prediction. A major research question to
ask is the performance of these many variations of deep learning techniques in
predicting time series data. This paper compares two prominent deep learning
modeling techniques. The Recurrent Neural Network (RNN)-based Long Short-Term
Memory (LSTM) and the convolutional Neural Network (CNN)-based Temporal
Convolutional Networks (TCN) are compared and their performance and training
time are reported. According to our experimental results, both modeling
techniques perform comparably having TCN-based models outperform LSTM slightly.
Moreover, the CNN-based TCN model builds a stable model faster than the
RNN-based LSTM models.

    

### [[2112.09305] Gaussian RBF Centered Kernel Alignment (CKA) in the Large Bandwidth Limit](http://arxiv.org/abs/2112.09305)


  We prove that Centered Kernel Alignment (CKA) based on a Gaussian RBF kernel
converges to linear CKA in the large-bandwidth limit. We show that convergence
onset is sensitive to the geometry of the feature representations, and that
representation eccentricity bounds the range of bandwidths for which Gaussian
CKA behaves nonlinearly.

    

### [[2112.09312] MIDI-DDSP: Detailed Control of Musical Performance via Hierarchical Modeling](http://arxiv.org/abs/2112.09312)


  Musical expression requires control of both what notes are played, and how
they are performed. Conventional audio synthesizers provide detailed expressive
controls, but at the cost of realism. Black-box neural audio synthesis and
concatenative samplers can produce realistic audio, but have few mechanisms for
control. In this work, we introduce MIDI-DDSP a hierarchical model of musical
instruments that enables both realistic neural audio synthesis and detailed
user control. Starting from interpretable Differentiable Digital Signal
Processing (DDSP) synthesis parameters, we infer musical notes and high-level
properties of their expressive performance (such as timbre, vibrato, dynamics,
and articulation). This creates a 3-level hierarchy (notes, performance,
synthesis) that affords individuals the option to intervene at each level, or
utilize trained priors (performance given notes, synthesis given performance)
for creative assistance. Through quantitative experiments and listening tests,
we demonstrate that this hierarchy can reconstruct high-fidelity audio,
accurately predict performance attributes for a note sequence, independently
manipulate the attributes of a given performance, and as a complete system,
generate realistic audio from a novel note sequence. By utilizing an
interpretable hierarchy, with multiple levels of granularity, MIDI-DDSP opens
the door to assistive tools to empower individuals across a diverse range of
musical experience.

    

### [[2112.09315] Optimal discharge of patients from intensive care via a data-driven policy learning framework](http://arxiv.org/abs/2112.09315)


  Clinical decision support tools rooted in machine learning and optimization
can provide significant value to healthcare providers, including through better
management of intensive care units. In particular, it is important that the
patient discharge task addresses the nuanced trade-off between decreasing a
patient's length of stay (and associated hospitalization costs) and the risk of
readmission or even death following the discharge decision. This work
introduces an end-to-end general framework for capturing this trade-off to
recommend optimal discharge timing decisions given a patient's electronic
health records. A data-driven approach is used to derive a parsimonious,
discrete state space representation that captures a patient's physiological
condition. Based on this model and a given cost function, an infinite-horizon
discounted Markov decision process is formulated and solved numerically to
compute an optimal discharge policy, whose value is assessed using off-policy
evaluation strategies. Extensive numerical experiments are performed to
validate the proposed framework using real-life intensive care unit patient
data.

    

### [[2112.09318] Procedural Kernel Networks](http://arxiv.org/abs/2112.09318)


  In the last decade Convolutional Neural Networks (CNNs) have defined the
state of the art for many low level image processing and restoration tasks such
as denoising, demosaicking, upscaling, or inpainting. However, on-device mobile
photography is still dominated by traditional image processing techniques, and
uses mostly simple machine learning techniques or limits the neural network
processing to producing low resolution masks. High computational and memory
requirements of CNNs, limited processing power and thermal constraints of
mobile devices, combined with large output image resolutions (typically 8--12
MPix) prevent their wider application. In this work, we introduce Procedural
Kernel Networks (PKNs), a family of machine learning models which generate
parameters of image filter kernels or other traditional algorithms. A
lightweight CNN processes the input image at a lower resolution, which yields a
significant speedup compared to other kernel-based machine learning methods and
allows for new applications. The architecture is learned end-to-end and is
especially well suited for a wide range of low-level image processing tasks,
where it improves the performance of many traditional algorithms. We also
describe how this framework unifies some previous work applying machine
learning for common image restoration tasks.

    

### [[2112.09327] Incentivizing Collaboration in Machine Learning via Synthetic Data Rewards](http://arxiv.org/abs/2112.09327)


  This paper presents a novel collaborative generative modeling (CGM) framework
that incentivizes collaboration among self-interested parties to contribute
data to a pool for training a generative model (e.g., GAN), from which
synthetic data are drawn and distributed to the parties as rewards commensurate
to their contributions. Distributing synthetic data as rewards (instead of
trained models or money) offers task- and model-agnostic benefits for
downstream learning tasks and is less likely to violate data privacy
regulation. To realize the framework, we firstly propose a data valuation
function using maximum mean discrepancy (MMD) that values data based on its
quantity and quality in terms of its closeness to the true data distribution
and provide theoretical results guiding the kernel choice in our MMD-based data
valuation function. Then, we formulate the reward scheme as a linear
optimization problem that when solved, guarantees certain incentives such as
fairness in the CGM framework. We devise a weighted sampling algorithm for
generating synthetic data to be distributed to each party as reward such that
the value of its data and the synthetic data combined matches its assigned
reward value by the reward scheme. We empirically show using simulated and
real-world datasets that the parties' synthetic data rewards are commensurate
to their contributions.

    

### [[2112.09332] WebGPT: Browser-assisted question-answering with human feedback](http://arxiv.org/abs/2112.09332)


  We fine-tune GPT-3 to answer long-form questions using a text-based
web-browsing environment, which allows the model to search and navigate the
web. By setting up the task so that it can be performed by humans, we are able
to train models on the task using imitation learning, and then optimize answer
quality with human feedback. To make human evaluation of factual accuracy
easier, models must collect references while browsing in support of their
answers. We train and evaluate our models on ELI5, a dataset of questions asked
by Reddit users. Our best model is obtained by fine-tuning GPT-3 using behavior
cloning, and then performing rejection sampling against a reward model trained
to predict human preferences. This model's answers are preferred by humans 56%
of the time to those of our human demonstrators, and 69% of the time to the
highest-voted answer from Reddit.

    

### [[2112.09335] Community-based Layerwise Distributed Training of Graph Convolutional Networks](http://arxiv.org/abs/2112.09335)


  The Graph Convolutional Network (GCN) has been successfully applied to many
graph-based applications. Training a large-scale GCN model, however, is still
challenging: Due to the node dependency and layer dependency of the GCN
architecture, a huge amount of computational time and memory is required in the
training process. In this paper, we propose a parallel and distributed GCN
training algorithm based on the Alternating Direction Method of Multipliers
(ADMM) to tackle the two challenges simultaneously. We first split GCN layers
into independent blocks to achieve layer parallelism. Furthermore, we reduce
node dependency by dividing the graph into several dense communities such that
each of them can be trained with an agent in parallel. Finally, we provide
solutions for all subproblems in the community-based ADMM algorithm.
Preliminary results demonstrate that our proposed community-based ADMM training
algorithm can lead to more than triple speedup while achieving the best
performance compared with state-of-the-art methods.

    

### [[2112.09339] ST2Vec: Spatio-Temporal Trajectory Similarity Learning in Road Networks](http://arxiv.org/abs/2112.09339)


  People and vehicle trajectories embody important information of
transportation infrastructures, and trajectory similarity computation is
functionality in many real-world applications involving trajectory data
analysis. Recently, deep-learning based trajectory similarity techniques hold
the potential to offer improved efficiency and adaptability over traditional
similarity techniques. Nevertheless, the existing trajectory similarity
learning proposals emphasize spatial similarity over temporal similarity,
making them suboptimal for time-aware analyses. To this end, we propose ST2Vec,
a trajectory-representation-learning based architecture that considers
fine-grained spatial and temporal correlations between pairs of trajectories
for spatio-temporal similarity learning in road networks. To the best of our
knowledge, this is the first deep-learning proposal for spatio-temporal
trajectory similarity analytics. Specifically, ST2Vec encompasses three phases:
(i) training data preparation that selects representative training samples;
(ii) spatial and temporal modeling that encode spatial and temporal
characteristics of trajectories, where a generic temporal modeling module (TMM)
is designed; and (iii) spatio-temporal co-attention fusion (STCF), where a
unified fusion (UF) approach is developed to help generating unified
spatio-temporal trajectory embeddings that capture the spatio-temporal
similarity relations between trajectories. Further, inspired by curriculum
concept, ST2Vec employs the curriculum learning for model optimization to
improve both convergence and effectiveness. An experimental study offers
evidence that ST2Vec outperforms all state-of-the-art competitors substantially
in terms of effectiveness, efficiency, and scalability, while showing low
parameter sensitivity and good model robustness.

    

### [[2112.09340] KGBoost: A Classification-based Knowledge Base Completion Method with Negative Sampling](http://arxiv.org/abs/2112.09340)


  Knowledge base completion is formulated as a binary classification problem in
this work, where an XGBoost binary classifier is trained for each relation
using relevant links in knowledge graphs (KGs). The new method, named KGBoost,
adopts a modularized design and attempts to find hard negative samples so as to
train a powerful classifier for missing link prediction. We conduct experiments
on multiple benchmark datasets, and demonstrate that KGBoost outperforms
state-of-the-art methods across most datasets. Furthermore, as compared with
models trained by end-to-end optimization, KGBoost works well under the
low-dimensional setting so as to allow a smaller model size.

    

### [[2112.09341] Personalized On-Device E-health Analytics with Decentralized Block Coordinate Descent](http://arxiv.org/abs/2112.09341)


  Actuated by the growing attention to personal healthcare and the pandemic,
the popularity of E-health is proliferating. Nowadays, enhancement on medical
diagnosis via machine learning models has been highly effective in many aspects
of e-health analytics. Nevertheless, in the classic cloud-based/centralized
e-health paradigms, all the data will be centrally stored on the server to
facilitate model training, which inevitably incurs privacy concerns and high
time delay. Distributed solutions like Decentralized Stochastic Gradient
Descent (D-SGD) are proposed to provide safe and timely diagnostic results
based on personal devices. However, methods like D-SGD are subject to the
gradient vanishing issue and usually proceed slowly at the early training
stage, thereby impeding the effectiveness and efficiency of training. In
addition, existing methods are prone to learning models that are biased towards
users with dense data, compromising the fairness when providing E-health
analytics for minority groups. In this paper, we propose a Decentralized Block
Coordinate Descent (D-BCD) learning framework that can better optimize deep
neural network-based models distributed on decentralized devices for E-health
analytics. Benchmarking experiments on three real-world datasets illustrate the
effectiveness and practicality of our proposed D-BCD, where additional
simulation study showcases the strong applicability of D-BCD in real-life
E-health scenarios.

    

### [[2112.09346] Balancing Fairness and Robustness via Partial Invariance](http://arxiv.org/abs/2112.09346)


  The Invariant Risk Minimization (IRM) framework aims to learn invariant
features from a set of environments for solving the out-of-distribution (OOD)
generalization problem. The underlying assumption is that the causal components
of the data generating distributions remain constant across the environments or
alternately, the data "overlaps" across environments to find meaningful
invariant features. Consequently, when the "overlap" assumption does not hold,
the set of truly invariant features may not be sufficient for optimal
prediction performance. Such cases arise naturally in networked settings and
hierarchical data-generating models, wherein the IRM performance becomes
suboptimal. To mitigate this failure case, we argue for a partial invariance
framework. The key idea is to introduce flexibility into the IRM framework by
partitioning the environments based on hierarchical differences, while
enforcing invariance locally within the partitions. We motivate this framework
in classification settings with causal distribution shifts across environments.
Our results show the capability of the partial invariant risk minimization to
alleviate the trade-off between fairness and risk in certain settings.

    

### [[2112.09348] Expedition: A System for the Unsupervised Learning of a Hierarchy of Concepts](http://arxiv.org/abs/2112.09348)


  We present a system for bottom-up cumulative learning of myriad concepts
corresponding to meaningful character strings, and their part-related and
prediction edges. The learning is self-supervised in that the concepts
discovered are used as predictors as well as targets of prediction. We devise
an objective for segmenting with the learned concepts, derived from comparing
to a baseline prediction system, that promotes making and using larger
concepts, which in turn allows for predicting larger spans of text, and we
describe a simple technique to promote exploration, i.e. trying out newly
generated concepts in the segmentation process. We motivate and explain a
layering of the concepts, to help separate the (conditional) distributions
learnt among concepts. The layering of the concepts roughly corresponds to a
part-whole concept hierarchy. With rudimentary segmentation and learning
algorithms, the system is promising in that it acquires many concepts (tens of
thousands in our small-scale experiments), and it learns to segment text well:
when fed with English text with spaces removed, starting at the character
level, much of what is learned respects word or phrase boundaries, and over
time the average number of "bad" splits within segmentations, i.e. splits
inside words, decreases as larger concepts are discovered and the system learns
when to use them during segmentation. We report on promising experiments when
the input text is converted to binary and the system begins with only two
concepts, "0" and "1". The system is transparent, in the sense that it is easy
to tell what the concepts learned correspond to, and which ones are active in a
segmentation, or how the system "sees" its input. We expect this framework to
be extensible and we discuss the current limitations and a number of directions
for enhancing the learning and inference capabilities.

    

### [[2112.09355] From Deterioration to Acceleration: A Calibration Approach to Rehabilitating Step Asynchronism in Federated Optimization](http://arxiv.org/abs/2112.09355)


  In the setting of federated optimization, where a global model is aggregated
periodically, step asynchronism occurs when participants conduct model training
with fully utilizing their computational resources. It is well acknowledged
that step asynchronism leads to objective inconsistency under non-i.i.d. data,
which degrades the model accuracy. To address this issue, we propose a new
algorithm \texttt{FedaGrac}, which calibrates the local direction to a
predictive global orientation. Taking the advantage of estimated orientation,
we guarantee that the aggregated model does not excessively deviate from the
expected orientation while fully utilizing the local updates of faster nodes.
We theoretically prove that \texttt{FedaGrac} holds an improved order of
convergence rate than the state-of-the-art approaches and eliminates the
negative effect of step asynchronism. Empirical results show that our algorithm
accelerates the training and enhances the final accuracy.

    

### [[2112.09362] Colloquium: Advances in automation of quantum dot devices control](http://arxiv.org/abs/2112.09362)


  Arrays of quantum dots (QDs) are a promising candidate system to realize
scalable, coupled qubits systems and serve as a fundamental building block for
quantum computers. In such semiconductor quantum systems, devices now have tens
of individual electrostatic and dynamical voltages that must be carefully set
to localize the system into the single-electron regime and to realize good
qubit operational performance. The mapping of requisite dot locations and
charges to gate voltages presents a challenging classical control problem. With
an increasing number of QD qubits, the relevant parameter space grows
sufficiently to make heuristic control unfeasible. In recent years, there has
been a considerable effort to automate device control that combines
script-based algorithms with machine learning (ML) techniques. In this
Colloquium, we present a comprehensive overview of the recent progress in the
automation of QD device control, with a particular emphasis on silicon- and
GaAs-based QDs formed in two-dimensional electron gases. Combining
physics-based modeling with modern numerical optimization and ML has proven
quite effective in yielding efficient, scalable control. Further integration of
theoretical, computational, and experimental efforts with computer science and
ML holds tremendous potential in advancing semiconductor and other platforms
for quantum computing.

    

### [[2112.09368] Improving evidential deep learning via multi-task learning](http://arxiv.org/abs/2112.09368)


  The Evidential regression network (ENet) estimates a continuous target and
its predictive uncertainty without costly Bayesian model averaging. However, it
is possible that the target is inaccurately predicted due to the gradient
shrinkage problem of the original loss function of the ENet, the negative log
marginal likelihood (NLL) loss. In this paper, the objective is to improve the
prediction accuracy of the ENet while maintaining its efficient uncertainty
estimation by resolving the gradient shrinkage problem. A multi-task learning
(MTL) framework, referred to as MT-ENet, is proposed to accomplish this aim. In
the MTL, we define the Lipschitz modified mean squared error (MSE) loss
function as another loss and add it to the existing NLL loss. The Lipschitz
modified MSE loss is designed to mitigate the gradient conflict with the NLL
loss by dynamically adjusting its Lipschitz constant. By doing so, the
Lipschitz MSE loss does not disturb the uncertainty estimation of the NLL loss.
The MT-ENet enhances the predictive accuracy of the ENet without losing
uncertainty estimation capability on the synthetic dataset and real-world
benchmarks, including drug-target affinity (DTA) regression. Furthermore, the
MT-ENet shows remarkable calibration and out-of-distribution detection
capability on the DTA benchmarks.

    

### [[2112.09382] Discretization and Re-synthesis: an alternative method to solve the Cocktail Party Problem](http://arxiv.org/abs/2112.09382)


  Deep learning based models have significantly improved the performance of
speech separation with input mixtures like the cocktail party. Prominent
methods (e.g., frequency-domain and time-domain speech separation) usually
build regression models to predict the ground-truth speech from the mixture,
using the masking-based design and the signal-level loss criterion (e.g., MSE
or SI-SNR). This study demonstrates, for the first time, that the
synthesis-based approach can also perform well on this problem, with great
flexibility and strong potential. Specifically, we propose a novel speech
separation/enhancement model based on the recognition of discrete symbols, and
convert the paradigm of the speech separation/enhancement related tasks from
regression to classification. By utilizing the synthesis model with the input
of discrete symbols, after the prediction of discrete symbol sequence, each
target speech could be re-synthesized. Evaluation results based on the
WSJ0-2mix and VCTK-noisy corpora in various settings show that our proposed
method can steadily synthesize the separated speech with high speech quality
and without any interference, which is difficult to avoid in regression-based
methods. In addition, with negligible loss of listening quality, the speaker
conversion of enhanced/separated speech could be easily realized through our
method.

    

### [[2112.09389] Feature extraction and classification algorithm, which one is more essential? An experimental study on a specific task of vibration signal diagnosis](http://arxiv.org/abs/2112.09389)


  With the development of machine learning, a data-driven model has been widely
used in vibration signal fault diagnosis. Most data-driven machine learning
algorithms are built based on well-designed features, but feature extraction is
usually required to be completed in advance. In the deep learning era, feature
extraction and classifier learning are conducted simultaneously, which will
lead to an end-to-end learning system. This paper explores which one of the two
key factors, i.e., feature extraction and classification algorithm, is more
essential for a specific task of vibration signal diagnosis during a learning
system is generated. Feature extractions from vibration signal based on both
well-known Gaussian model and statistical characteristics are discussed,
respectively. And several classification algorithms are selected to
experimentally validate the comparative impact of both feature extraction and
classification algorithm on prediction performance.

    

### [[2112.09391] Can Machine Learning Tools Support the Identification of Sustainable Design Leads From Product Reviews? Opportunities and Challenges](http://arxiv.org/abs/2112.09391)


  The increasing number of product reviews posted online is a gold mine for
designers to know better about the products they develop, by capturing the
voice of customers, and to improve these products accordingly. In the meantime,
product design and development have an essential role in creating a more
sustainable future. With the recent advance of artificial intelligence
techniques in the field of natural language processing, this research aims to
develop an integrated machine learning solution to obtain sustainable design
insights from online product reviews automatically. In this paper, the
opportunities and challenges offered by existing frameworks - including Python
libraries, packages, as well as state-of-the-art algorithms like BERT - are
discussed, illustrated, and positioned along an ad hoc machine learning
process. This contribution discusses the opportunities to reach and the
challenges to address for building a machine learning pipeline, in order to get
insights from product reviews to design more sustainable products, including
the five following stages, from the identification of sustainability-related
reviews to the interpretation of sustainable design leads: data collection,
data formatting, model training, model evaluation, and model deployment.
Examples of sustainable design insights that can be produced out of product
review mining and processing are given. Finally, promising lines for future
research in the field are provided, including case studies putting in parallel
standard products with their sustainable alternatives, to compare the features
valued by customers and to generate in fine relevant sustainable design leads.

    

### [[2112.09397] Semi-Supervised Clustering via Markov Chain Aggregation](http://arxiv.org/abs/2112.09397)


  We connect the problem of semi-supervised clustering to constrained Markov
aggregation, i.e., the task of partitioning the state space of a Markov chain.
We achieve this connection by considering every data point in the dataset as an
element of the Markov chain's state space, by defining the transition
probabilities between states via similarities between corresponding data
points, and by incorporating semi-supervision information as hard constraints
in a Hartigan-style algorithm. The introduced Constrained Markov Clustering
(CoMaC) is an extension of a recent information-theoretic framework for
(unsupervised) Markov aggregation to the semi-supervised case. Instantiating
CoMaC for certain parameter settings further generalizes two previous
information-theoretic objectives for unsupervised clustering. Our results
indicate that CoMaC is competitive with the state-of-the-art. Furthermore, our
approach is less sensitive to hyperparameter settings than the unsupervised
counterpart, which is especially attractive in the semi-supervised setting
characterized by little labeled data.

    

### [[2112.09400] Quality of Data in Machine Learning](http://arxiv.org/abs/2112.09400)


  A common assumption exists according to which machine learning models improve
their performance when they have more data to learn from. In this study, the
authors wished to clarify the dilemma by performing an empirical experiment
utilizing novel vocational student data. The experiment compared different
machine learning algorithms while varying the number of data and feature
combinations available for training and testing the models. The experiment
revealed that the increase of data records or their sample frequency does not
immediately lead to significant increases in the model accuracies or
performance, however the variance of accuracies does diminish in the case of
ensemble models. Similar phenomenon was witnessed while increasing the number
of input features for the models. The study refutes the starting assumption and
continues to state that in this case the significance in data lies in the
quality of the data instead of the quantity of the data.

    

### [[2112.09420] A random energy approach to deep learning](http://arxiv.org/abs/2112.09420)


  We study a generic ensemble of deep belief networks which is parametrized by
the distribution of energy levels of the hidden states of each layer. We show
that, within a random energy approach, statistical dependence can propagate
from the visible to deep layers only if each layer is tuned close to the
critical point during learning. As a consequence, efficiently trained learning
machines are characterised by a broad distribution of energy levels. The
analysis of Deep Belief Networks and Restricted Boltzmann Machines on different
datasets confirms these conclusions.

    

### [[2112.09423] ActKnow: Active External Knowledge Infusion Learning for Question Answering in Low Data Regime](http://arxiv.org/abs/2112.09423)


  Deep learning models have set benchmark results in various Natural Language
Processing tasks. However, these models require an enormous amount of training
data, which is infeasible in many practical problems. While various techniques
like domain adaptation, fewshot learning techniques address this problem, we
introduce a new technique of actively infusing external knowledge into learning
to solve low data regime problems. We propose a technique called ActKnow that
actively infuses knowledge from Knowledge Graphs (KG) based "on-demand" into
learning for Question Answering (QA). By infusing world knowledge from
Concept-Net, we show significant improvements on the ARC Challenge-set
benchmark over purely text-based transformer models like RoBERTa in the low
data regime. For example, by using only 20% training examples, we demonstrate a
4% improvement in the accuracy for both ARC-challenge and OpenBookQA,
respectively.

    

### [[2112.09427] Continual Learning for Monolingual End-to-End Automatic Speech Recognition](http://arxiv.org/abs/2112.09427)


  Adapting Automatic Speech Recognition (ASR) models to new domains leads to a
deterioration of performance on the original domain(s), a phenomenon called
Catastrophic Forgetting (CF). Even monolingual ASR models cannot be extended to
new accents, dialects, topics, etc. without suffering from CF, making them
unable to be continually enhanced without storing all past data. Fortunately,
Continual Learning (CL) methods, which aim to enable continual adaptation while
overcoming CF, can be used. In this paper, we implement an extensive number of
CL methods for End-to-End ASR and test and compare their ability to extend a
monolingual Hybrid CTC-Transformer model across four new tasks. We find that
the best performing CL method closes the gap between the fine-tuned model
(lower bound) and the model trained jointly on all tasks (upper bound) by more
than 40%, while requiring access to only 0.6% of the original data.

    

### [[2112.09429] Federated Learning with Heterogeneous Data: A Superquantile Optimization Approach](http://arxiv.org/abs/2112.09429)


  We present a federated learning framework that is designed to robustly
deliver good predictive performance across individual clients with
heterogeneous data. The proposed approach hinges upon a superquantile-based
learning objective that captures the tail statistics of the error distribution
over heterogeneous clients. We present a stochastic training algorithm which
interleaves differentially private client reweighting steps with federated
averaging steps. The proposed algorithm is supported with finite time
convergence guarantees that cover both convex and non-convex settings.
Experimental results on benchmark datasets for federated learning demonstrate
that our approach is competitive with classical ones in terms of average error
and outperforms them in terms of tail statistics of the error.

    

### [[2112.09436] Privacy preserving n-party scalar product protocol](http://arxiv.org/abs/2112.09436)


  Privacy-preserving machine learning enables the training of models on
decentralized datasets without the need to reveal the data, both on horizontal
and vertically partitioned data. However, it relies on specialized techniques
and algorithms to perform the necessary computations. The privacy preserving
scalar product protocol, which enables the dot product of vectors without
revealing them, is one popular example for its versatility. Unfortunately, the
solutions currently proposed in the literature focus mainly on two-party
scenarios, even though scenarios with a higher number of data parties are
becoming more relevant. For example when performing analyses that require
counting the number of samples which fulfill certain criteria defined across
various sites, such as calculating the information gain at a node in a decision
tree. In this paper we propose a generalization of the protocol for an
arbitrary number of parties, based on an existing two-party method. Our
proposed solution relies on a recursive resolution of smaller scalar products.
After describing our proposed method, we discuss potential scalability issues.
Finally, we describe the privacy guarantees and identify any concerns, as well
as comparing the proposed method to the original solution in this aspect.

    

### [[2112.09438] ML Supported Predictions for SAT Solvers Performance](http://arxiv.org/abs/2112.09438)


  In order to classify the indeterministic termination behavior of the open
source SAT solver CryptoMiniSat in multi-threading mode while processing hard
to solve boolean satisfiability problem instances, internal solver runtime
parameters have been collected and analyzed. A subset of these parameters has
been selected and employed as features vector to successfully create a machine
learning model for the binary classification of the solver's termination
behavior with any single new solving run of a not yet solved instance. The
model can be used for the early estimation of a solving attempt as belonging or
not belonging to the class of candidates with good chances for a fast
termination. In this context a combination of active profiles of runtime
characteristics appear to mirror the influence of the solver's momentary
heuristics on the immediate quality of the solver's resolution process. Because
runtime parameters of already the first two solving iterations are enough to
forecast termination of the attempt with good success scores, the results of
the present work deliver a promising basis which can be further developed in
order to enrich CryptoMiniSat or generally any modern SAT solver with AI
abilities.

    

### [[2112.09442] Adaptively Customizing Activation Functions for Various Layers](http://arxiv.org/abs/2112.09442)


  To enhance the nonlinearity of neural networks and increase their mapping
abilities between the inputs and response variables, activation functions play
a crucial role to model more complex relationships and patterns in the data. In
this work, a novel methodology is proposed to adaptively customize activation
functions only by adding very few parameters to the traditional activation
functions such as Sigmoid, Tanh, and ReLU. To verify the effectiveness of the
proposed methodology, some theoretical and experimental analysis on
accelerating the convergence and improving the performance is presented, and a
series of experiments are conducted based on various network models (such as
AlexNet, VGGNet, GoogLeNet, ResNet and DenseNet), and various datasets (such as
CIFAR10, CIFAR100, miniImageNet, PASCAL VOC and COCO) . To further verify the
validity and suitability in various optimization strategies and usage
scenarios, some comparison experiments are also implemented among different
optimization strategies (such as SGD, Momentum, AdaGrad, AdaDelta and ADAM) and
different recognition tasks like classification and detection. The results show
that the proposed methodology is very simple but with significant performance
in convergence speed, precision and generalization, and it can surpass other
popular methods like ReLU and adaptive functions like Swish in almost all
experiments in terms of overall performance.The code is publicly available at
this https URL. The
package includes the proposed three adaptive activation functions for
reproducibility purposes.

    

### [[2112.09456] Visual Learning-based Planning for Continuous High-Dimensional POMDPs](http://arxiv.org/abs/2112.09456)


  The Partially Observable Markov Decision Process (POMDP) is a powerful
framework for capturing decision-making problems that involve state and
transition uncertainty. However, most current POMDP planners cannot effectively
handle very high-dimensional observations they often encounter in the real
world (e.g. image observations in robotic domains). In this work, we propose
Visual Tree Search (VTS), a learning and planning procedure that combines
generative models learned offline with online model-based POMDP planning. VTS
bridges offline model training and online planning by utilizing a set of deep
generative observation models to predict and evaluate the likelihood of image
observations in a Monte Carlo tree search planner. We show that VTS is robust
to different observation noises and, since it utilizes online, model-based
planning, can adapt to different reward structures without the need to
re-train. This new approach outperforms a baseline state-of-the-art on-policy
planning algorithm while using significantly less offline training time.

    

### [[2112.09466] An overview of active learning methods for insurance with fairness appreciation](http://arxiv.org/abs/2112.09466)


  This paper addresses and solves some challenges in the adoption of machine
learning in insurance with the democratization of model deployment. The first
challenge is reducing the labelling effort (hence focusing on the data quality)
with the help of active learning, a feedback loop between the model inference
and an oracle: as in insurance the unlabeled data is usually abundant, active
learning can become a significant asset in reducing the labelling cost. For
that purpose, this paper sketches out various classical active learning
methodologies before studying their empirical impact on both synthetic and real
datasets. Another key challenge in insurance is the fairness issue in model
inferences. We will introduce and integrate a post-processing fairness for
multi-class tasks in this active learning framework to solve these two issues.
Finally numerical experiments on unfair datasets highlight that the proposed
setup presents a good compromise between model precision and fairness.

    

### [[2112.09467] A Multimodal Approach for Automatic Mania Assessment in Bipolar Disorder](http://arxiv.org/abs/2112.09467)


  Bipolar disorder is a mental health disorder that causes mood swings that
range from depression to mania. Diagnosis of bipolar disorder is usually done
based on patient interviews, and reports obtained from the caregivers of the
patients. Subsequently, the diagnosis depends on the experience of the expert,
and it is possible to have confusions of the disorder with other mental
disorders. Automated processes in the diagnosis of bipolar disorder can help
providing quantitative indicators, and allow easier observations of the
patients for longer periods. Furthermore, the need for remote treatment and
diagnosis became especially important during the COVID-19 pandemic. In this
thesis, we create a multimodal decision system based on recordings of the
patient in acoustic, linguistic, and visual modalities. The system is trained
on the Bipolar Disorder corpus. Comprehensive analysis of unimodal and
multimodal systems, as well as various fusion techniques are performed. Besides
processing entire patient sessions using unimodal features, a task-level
investigation of the clips is studied. Using acoustic, linguistic, and visual
features in a multimodal fusion system, we achieved a 64.8% unweighted average
recall score, which improves the state-of-the-art performance achieved on this
dataset.

    

### [[2112.09468] Towards fuzzification of adaptation rules in self-adaptive architectures](http://arxiv.org/abs/2112.09468)


  In this paper, we focus on exploiting neural networks for the analysis and
planning stage in self-adaptive architectures. The studied motivating cases in
the paper involve existing (legacy) self-adaptive architectures and their
adaptation logic, which has been specified by logical rules. We further assume
that there is a need to endow these systems with the ability to learn based on
examples of inputs and expected outputs. One simple option to address such a
need is to replace the reasoning based on logical rules with a neural network.
However, this step brings several problems that often create at least a
temporary regress. The reason is the logical rules typically represent a large
and tested body of domain knowledge, which may be lost if the logical rules are
replaced by a neural network. Further, the black-box nature of generic neural
networks obfuscates how the systems work inside and consequently introduces
more uncertainty. In this paper, we present a method that makes it possible to
endow an existing self-adaptive architectures with the ability to learn using
neural networks, while preserving domain knowledge existing in the logical
rules. We introduce a continuum between the existing rule-based system and a
system based on a generic neural network. We show how to navigate in this
continuum and create a neural network architecture that naturally embeds the
original logical rules and how to gradually scale the learning potential of the
network, thus controlling the uncertainty inherent to all soft computing
models. We showcase and evaluate the approach on representative excerpts from
two larger real-life use cases.

    

### [[2112.09477] Learning Reward Machines: A Study in Partially Observable Reinforcement Learning](http://arxiv.org/abs/2112.09477)


  Reinforcement learning (RL) is a central problem in artificial intelligence.
This problem consists of defining artificial agents that can learn optimal
behaviour by interacting with an environment -- where the optimal behaviour is
defined with respect to a reward signal that the agent seeks to maximize.
Reward machines (RMs) provide a structured, automata-based representation of a
reward function that enables an RL agent to decompose an RL problem into
structured subproblems that can be efficiently learned via off-policy learning.
Here we show that RMs can be learned from experience, instead of being
specified by the user, and that the resulting problem decomposition can be used
to effectively solve partially observable RL problems. We pose the task of
learning RMs as a discrete optimization problem where the objective is to find
an RM that decomposes the problem into a set of subproblems such that the
combination of their optimal memoryless policies is an optimal policy for the
original problem. We show the effectiveness of this approach on three partially
observable domains, where it significantly outperforms A3C, PPO, and ACER, and
discuss its advantages, limitations, and broader potential.

    

### [[2112.09483] Learning from Heterogeneous Data Based on Social Interactions over Graphs](http://arxiv.org/abs/2112.09483)


  This work proposes a decentralized architecture, where individual agents aim
at solving a classification problem while observing streaming features of
different dimensions and arising from possibly different distributions. In the
context of social learning, several useful strategies have been developed,
which solve decision making problems through local cooperation across
distributed agents and allow them to learn from streaming data. However,
traditional social learning strategies rely on the fundamental assumption that
each agent has significant prior knowledge of the underlying distribution of
the observations. In this work we overcome this issue by introducing a machine
learning framework that exploits social interactions over a graph, leading to a
fully data-driven solution to the distributed classification problem. In the
proposed social machine learning (SML) strategy, two phases are present: in the
training phase, classifiers are independently trained to generate a belief over
a set of hypotheses using a finite number of training samples; in the
prediction phase, classifiers evaluate streaming unlabeled observations and
share their instantaneous beliefs with neighboring classifiers. We show that
the SML strategy enables the agents to learn consistently under this
highly-heterogeneous setting and allows the network to continue learning even
during the prediction phase when it is deciding on unlabeled samples. The
prediction decisions are used to continually improve performance thereafter in
a manner that is markedly different from most existing static classification
schemes where, following training, the decisions on unlabeled data are not
re-used to improve future performance.

    

### [[2112.09484] Learning in Restless Bandits under Exogenous Global Markov Process](http://arxiv.org/abs/2112.09484)


  We consider an extension to the restless multi-armed bandit (RMAB) problem
with unknown arm dynamics, where an unknown exogenous global Markov process
governs the rewards distribution of each arm. Under each global state, the
rewards process of each arm evolves according to an unknown Markovian rule,
which is non-identical among different arms. At each time, a player chooses an
arm out of $N$ arms to play, and receives a random reward from a finite set of
reward states. The arms are restless, that is, their local state evolves
regardless of the player's actions. Motivated by recent studies on related RMAB
settings, the regret is defined as the reward loss with respect to a player
that knows the dynamics of the problem, and plays at each time $t$ the arm that
maximizes the expected immediate value. The objective is to develop an
arm-selection policy that minimizes the regret. To that end, we develop the
Learning under Exogenous Markov Process (LEMP) algorithm. We analyze LEMP
theoretically and establish a finite-sample bound on the regret. We show that
LEMP achieves a logarithmic regret order with time. We further analyze LEMP
numerically and present simulation results that support the theoretical
findings and demonstrate that LEMP significantly outperforms alternative
algorithms.

    

### [[2112.09493] Methods for segmenting cracks in 3d images of concrete: A comparison based on semi-synthetic images](http://arxiv.org/abs/2112.09493)


  Concrete is the standard construction material for buildings, bridges, and
roads. As safety plays a central role in the design, monitoring, and
maintenance of such constructions, it is important to understand the cracking
behavior of concrete. Computed tomography captures the microstructure of
building materials and allows to study crack initiation and propagation. Manual
segmentation of crack surfaces in large 3d images is not feasible. In this
paper, automatic crack segmentation methods for 3d images are reviewed and
compared. Classical image processing methods (edge detection filters, template
matching, minimal path and region growing algorithms) and learning methods
(convolutional neural networks, random forests) are considered and tested on
semi-synthetic 3d images. Their performance strongly depends on parameter
selection which should be adapted to the grayvalue distribution of the images
and the geometric properties of the concrete. In general, the learning methods
perform best, in particular for thin cracks and low grayvalue contrast.

    

### [[2112.09495] Stability Verification in Stochastic Control Systems via Neural Network Supermartingales](http://arxiv.org/abs/2112.09495)


  We consider the problem of formally verifying almost-sure (a.s.) asymptotic
stability in discrete-time nonlinear stochastic control systems. While
verifying stability in deterministic control systems is extensively studied in
the literature, verifying stability in stochastic control systems is an open
problem. The few existing works on this topic either consider only specialized
forms of stochasticity or make restrictive assumptions on the system, rendering
them inapplicable to learning algorithms with neural network policies. In this
work, we present an approach for general nonlinear stochastic control problems
with two novel aspects: (a) instead of classical stochastic extensions of
Lyapunov functions, we use ranking supermartingales (RSMs) to certify
a.s.~asymptotic stability, and (b) we present a method for learning neural
network RSMs. We prove that our approach guarantees a.s.~asymptotic stability
of the system and provides the first method to obtain bounds on the
stabilization time, which stochastic Lyapunov functions do not. Finally, we
validate our approach experimentally on a set of nonlinear stochastic
reinforcement learning environments with neural network policies.

    

### [[2112.09496] Towards Launching AI Algorithms for Cellular Pathology into Clinical & Pharmaceutical Orbits](http://arxiv.org/abs/2112.09496)


  Computational Pathology (CPath) is an emerging field concerned with the study
of tissue pathology via computational algorithms for the processing and
analysis of digitized high-resolution images of tissue slides. Recent deep
learning based developments in CPath have successfully leveraged sheer volume
of raw pixel data in histology images for predicting target parameters in the
domains of diagnostics, prognostics, treatment sensitivity and patient
stratification -- heralding the promise of a new data-driven AI era for both
histopathology and oncology. With data serving as the fuel and AI as the
engine, CPath algorithms are poised to be ready for takeoff and eventual launch
into clinical and pharmaceutical orbits. In this paper, we discuss CPath
limitations and associated challenges to enable the readers distinguish hope
from hype and provide directions for future research to overcome some of the
major challenges faced by this budding field to enable its launch into the two
orbits.

    

### [[2112.09519] Correlated Product of Experts for Sparse Gaussian Process Regression](http://arxiv.org/abs/2112.09519)


  Gaussian processes (GPs) are an important tool in machine learning and
statistics with applications ranging from social and natural science through
engineering. They constitute a powerful kernelized non-parametric method with
well-calibrated uncertainty estimates, however, off-the-shelf GP inference
procedures are limited to datasets with several thousand data points because of
their cubic computational complexity. For this reason, many sparse GPs
techniques have been developed over the past years. In this paper, we focus on
GP regression tasks and propose a new approach based on aggregating predictions
from several local and correlated experts. Thereby, the degree of correlation
between the experts can vary between independent up to fully correlated
experts. The individual predictions of the experts are aggregated taking into
account their correlation resulting in consistent uncertainty estimates. Our
method recovers independent Product of Experts, sparse GP and full GP in the
limiting cases. The presented framework can deal with a general kernel function
and multiple variables, and has a time and space complexity which is linear in
the number of experts and data samples, which makes our approach highly
scalable. We demonstrate superior performance, in a time vs. accuracy sense, of
our proposed method against state-of-the-art GP approximation methods for
synthetic as well as several real-world datasets with deterministic and
stochastic optimization.

    

### [[2112.09532] Pixel Distillation: A New Knowledge Distillation Scheme for Low-Resolution Image Recognition](http://arxiv.org/abs/2112.09532)


  The great success of deep learning is mainly due to the large-scale network
architecture and the high-quality training data. However, it is still
challenging to deploy recent deep models on portable devices with limited
memory and imaging ability. Some existing works have engaged to compress the
model via knowledge distillation. Unfortunately, these methods cannot deal with
images with reduced image quality, such as the low-resolution (LR) images. To
this end, we make a pioneering effort to distill helpful knowledge from a heavy
network model learned from high-resolution (HR) images to a compact network
model that will handle LR images, thus advancing the current knowledge
distillation technique with the novel pixel distillation. To achieve this goal,
we propose a Teacher-Assistant-Student (TAS) framework, which disentangles
knowledge distillation into the model compression stage and the high resolution
representation transfer stage. By equipping a novel Feature Super Resolution
(FSR) module, our approach can learn lightweight network model that can achieve
similar accuracy as the heavy teacher model but with much fewer parameters,
faster inference speed, and lower-resolution inputs. Comprehensive experiments
on three widely-used benchmarks, \ie, CUB-200-2011, PASCAL VOC 2007, and
ImageNetSub, demonstrate the effectiveness of our approach.

    

### [[2112.09568] Nearest neighbor search with compact codes: A decoder perspective](http://arxiv.org/abs/2112.09568)


  Modern approaches for fast retrieval of similar vectors on billion-scaled
datasets rely on compressed-domain approaches such as binary sketches or
product quantization. These methods minimize a certain loss, typically the mean
squared error or other objective functions tailored to the retrieval problem.
In this paper, we re-interpret popular methods such as binary hashing or
product quantizers as auto-encoders, and point out that they implicitly make
suboptimal assumptions on the form of the decoder. We design
backward-compatible decoders that improve the reconstruction of the vectors
from the same codes, which translates to a better performance in nearest
neighbor search. Our method significantly improves over binary hashing methods
or product quantization on popular benchmarks.

    

### [[2112.09569] CPPE-5: Medical Personal Protective Equipment Dataset](http://arxiv.org/abs/2112.09569)


  We present a new challenging dataset, CPPE - 5 (Medical Personal Protective
Equipment), with the goal to allow the study of subordinate categorization of
medical personal protective equipments, which is not possible with other
popular data sets that focus on broad level categories (such as PASCAL VOC,
ImageNet, Microsoft COCO, OpenImages, etc). To make it easy for models trained
on this dataset to be used in practical scenarios in complex scenes, our
dataset mainly contains images that show complex scenes with several objects in
each scene in their natural context. The image collection for this dataset
focusing on: obtaining as many non-iconic images as possible and making sure
all the images are real-life images unlike other existing datasets in this
area. Our dataset includes 5 object categories (coveralls, face shield, gloves,
mask, and goggles) and each image is annotated with a set of bounding boxes and
positive labels. We present a detailed analysis of the dataset in comparison to
other popular broad category datasets as well as datasets focusing on personal
protective equipments, we also find that at present there exist no such
publicly available datasets. Finally we also analyze performance and compare
model complexities on baseline and state-of-the-art models for bounding box
results. Our code, data, and trained models are available at
this https URL .

    

### [[2112.09570] A Binded VAE for Inorganic Material Generation](http://arxiv.org/abs/2112.09570)


  Designing new industrial materials with desired properties can be very
expensive and time consuming. The main difficulty is to generate compounds that
correspond to realistic materials. Indeed, the description of compounds as
vectors of components' proportions is characterized by discrete features and a
severe sparsity. Furthermore, traditional generative model validation processes
as visual verification, FID and Inception scores are tailored for images and
cannot then be used as such in this context. To tackle these issues, we develop
an original Binded-VAE model dedicated to the generation of discrete datasets
with high sparsity. We validate the model with novel metrics adapted to the
problem of compounds generation. We show on a real issue of rubber compound
design that the proposed approach outperforms the standard generative models
which opens new perspectives for material design optimization.

    

### [[2112.09574] Super-resolution reconstruction of cytoskeleton image based on A-net deep learning network](http://arxiv.org/abs/2112.09574)


  To date, live-cell imaging at the nanometer scale remains challenging. Even
though super-resolution microscopy methods have enabled visualization of
subcellular structures below the optical resolution limit, the spatial
resolution is still far from enough for the structural reconstruction of
biomolecules in vivo (i.e. ~24 nm thickness of microtubule fiber). In this
study, we proposed an A-net network and showed that the resolution of
cytoskeleton images captured by a confocal microscope can be significantly
improved by combining the A-net deep learning network with the DWDC algorithm
based on degradation model. Utilizing the DWDC algorithm to construct new
datasets and taking advantage of A-net neural network's features (i.e.,
considerably fewer layers), we successfully removed the noise and flocculent
structures, which originally interfere with the cellular structure in the raw
image, and improved the spatial resolution by 10 times using relatively small
dataset. We, therefore, conclude that the proposed algorithm that combines
A-net neural network with the DWDC method is a suitable and universal approach
for exacting structural details of biomolecules, cells and organs from
low-resolution images.

    

### [[2112.09579] Convergence Rates of Two-Time-Scale Gradient Descent-Ascent Dynamics for Solving Nonconvex Min-Max Problems](http://arxiv.org/abs/2112.09579)


  There are much recent interests in solving noncovnex min-max optimization
problems due to its broad applications in many areas including machine
learning, networked resource allocations, and distributed optimization.
Perhaps, the most popular first-order method in solving min-max optimization is
the so-called simultaneous (or single-loop) gradient descent-ascent algorithm
due to its simplicity in implementation. However, theoretical guarantees on the
convergence of this algorithm is very sparse since it can diverge even in a
simple bilinear problem.
In this paper, our focus is to characterize the finite-time performance (or
convergence rates) of the continuous-time variant of simultaneous gradient
descent-ascent algorithm. In particular, we derive the rates of convergence of
this method under a number of different conditions on the underlying objective
function, namely, two-sided Polyak-L ojasiewicz (PL), one-sided PL,
nonconvex-strongly concave, and strongly convex-nonconcave conditions. Our
convergence results improve the ones in prior works under the same conditions
of objective functions. The key idea in our analysis is to use the classic
singular perturbation theory and coupling Lyapunov functions to address the
time-scale difference and interactions between the gradient descent and ascent
dynamics. Our results on the behavior of continuous-time algorithm may be used
to enhance the convergence properties of its discrete-time counterpart.

    

### [[2112.09581] Watermarking Images in Self-Supervised Latent Spaces](http://arxiv.org/abs/2112.09581)


  We revisit watermarking techniques based on pre-trained deep networks, in the
light of self-supervised approaches. We present a way to embed both marks and
binary messages into their latent spaces, leveraging data augmentation at
marking time. Our method can operate at any resolution and creates watermarks
robust to a broad range of transformations (rotations, crops, JPEG, contrast,
etc). It significantly outperforms the previous zero-bit methods, and its
performance on multi-bit watermarking is on par with state-of-the-art
encoder-decoder architectures trained end-to-end for watermarking. Our
implementation and models will be made publicly available.

    

### [[2112.09605] Autonomous Reinforcement Learning: Formalism and Benchmarking](http://arxiv.org/abs/2112.09605)


  Reinforcement learning (RL) provides a naturalistic framing for learning
through trial and error, which is appealing both because of its simplicity and
effectiveness and because of its resemblance to how humans and animals acquire
skills through experience. However, real-world embodied learning, such as that
performed by humans and animals, is situated in a continual, non-episodic
world, whereas common benchmark tasks in RL are episodic, with the environment
resetting between trials to provide the agent with multiple attempts. This
discrepancy presents a major challenge when attempting to take RL algorithms
developed for episodic simulated environments and run them on real-world
platforms, such as robots. In this paper, we aim to address this discrepancy by
laying out a framework for Autonomous Reinforcement Learning (ARL):
reinforcement learning where the agent not only learns through its own
experience, but also contends with lack of human supervision to reset between
trials. We introduce a simulated benchmark EARL around this framework,
containing a set of diverse and challenging simulated tasks reflective of the
hurdles introduced to learning when only a minimal reliance on extrinsic
intervention can be assumed. We show that standard approaches to episodic RL
and existing approaches struggle as interventions are minimized, underscoring
the need for developing new algorithms for reinforcement learning with a
greater focus on autonomy.

    

### [[2112.09625] Provable Adversarial Robustness in the Quantum Model](http://arxiv.org/abs/2112.09625)


  Modern machine learning systems have been applied successfully to a variety
of tasks in recent years but making such systems robust against adversarially
chosen modifications of input instances seems to be a much harder problem. It
is probably fair to say that no fully satisfying solution has been found up to
date and it is not clear if the standard formulation even allows for a
principled solution. Hence, rather than following the classical path of bounded
perturbations, we consider a model similar to the quantum PAC-learning model
introduced by Bshouty and Jackson [1995]. Our first key contribution shows that
in this model we can reduce adversarial robustness to the conjunction of two
classical learning theory problems, namely (Problem 1) the problem of finding
generative models and (Problem 2) the problem of devising classifiers that are
robust with respect to distributional shifts. Our second key contribution is
that the considered framework does not rely on specific (and hence also
somewhat arbitrary) threat models like $\ell_p$ bounded perturbations. Instead,
our reduction guarantees that in order to solve the adversarial robustness
problem in our model it suffices to consider a single distance notion, i.e. the
Hellinger distance. From the technical perspective our protocols are heavily
based on the recent advances on delegation of quantum computation, e.g. Mahadev
[2018].
Although the considered model is quantum and therefore not immediately
applicable to ``real-world'' situations, one might hope that in the future
either one can find a way to embed ``real-world'' problems into a quantum
framework or that classical algorithms can be found that are capable of
mimicking their powerful quantum counterparts.

    

### [[2112.09631] Sublinear Time Approximation of Text Similarity Matrices](http://arxiv.org/abs/2112.09631)


  We study algorithms for approximating pairwise similarity matrices that arise
in natural language processing. Generally, computing a similarity matrix for
$n$ data points requires $\Omega(n^2)$ similarity computations. This quadratic
scaling is a significant bottleneck, especially when similarities are computed
via expensive functions, e.g., via transformer models. Approximation methods
reduce this quadratic complexity, often by using a small subset of exactly
computed similarities to approximate the remainder of the complete pairwise
similarity matrix.
Significant work focuses on the efficient approximation of positive
semidefinite (PSD) similarity matrices, which arise e.g., in kernel methods.
However, much less is understood about indefinite (non-PSD) similarity
matrices, which often arise in NLP. Motivated by the observation that many of
these matrices are still somewhat close to PSD, we introduce a generalization
of the popular Nyström method to the indefinite setting. Our algorithm can
be applied to any similarity matrix and runs in sublinear time in the size of
the matrix, producing a rank-$s$ approximation with just $O(ns)$ similarity
computations.
We show that our method, along with a simple variant of CUR decomposition,
performs very well in approximating a variety of similarity matrices arising in
NLP tasks. We demonstrate high accuracy of the approximated similarity matrices
in the downstream tasks of document classification, sentence similarity, and
cross-document coreference.

    

### [[2112.09638] Oil Spill SAR Image Segmentation via Probability Distribution Modelling](http://arxiv.org/abs/2112.09638)


  Segmentation of marine oil spills in Synthetic Aperture Radar (SAR) images is
a challenging task because of the complexity and irregularities in SAR images.
In this work, we aim to develop an effective segmentation method which
addresses marine oil spill identification in SAR images by investigating the
distribution representation of SAR images. To seek effective oil spill
segmentation, we revisit the SAR imaging mechanism in order to attain the
probability distribution representation of oil spill SAR images, in which the
characteristics of SAR images are properly modelled. We then exploit the
distribution representation to formulate the segmentation energy functional, by
which oil spill characteristics are incorporated to guide oil spill
segmentation. Moreover, the oil spill segmentation model contains the oil spill
contour regularisation term and the updated level set regularisation term which
enhance the representational power of the segmentation energy functional.
Benefiting from the synchronisation of SAR image representation and oil spill
segmentation, our proposed method establishes an effective oil spill
segmentation framework. Experimental evaluations demonstrate the effectiveness
of our proposed segmentation framework for different types of marine oil spill
SAR image segmentation.

    

### [[2112.09641] Embedding Graph Convolutional Networks in Recurrent Neural Networks for Predictive Monitoring](http://arxiv.org/abs/2112.09641)


  Predictive monitoring of business processes is a subfield of process mining
that aims to predict, among other things, the characteristics of the next event
or the sequence of next events. Although multiple approaches based on deep
learning have been proposed, mainly recurrent neural networks and convolutional
neural networks, none of them really exploit the structural information
available in process models. This paper proposes an approach based on graph
convolutional networks and recurrent neural networks that uses information
directly from the process model. An experimental evaluation on real-life event
logs shows that our approach is more consistent and outperforms the current
state-of-the-art approaches.

    

### [[2112.09645] Local contrastive loss with pseudo-label based self-training for semi-supervised medical image segmentation](http://arxiv.org/abs/2112.09645)


  Supervised deep learning-based methods yield accurate results for medical
image segmentation. However, they require large labeled datasets for this, and
obtaining them is a laborious task that requires clinical expertise.
Semi/self-supervised learning-based approaches address this limitation by
exploiting unlabeled data along with limited annotated data. Recent
self-supervised learning methods use contrastive loss to learn good global
level representations from unlabeled images and achieve high performance in
classification tasks on popular natural image datasets like ImageNet. In
pixel-level prediction tasks such as segmentation, it is crucial to also learn
good local level representations along with global representations to achieve
better accuracy. However, the impact of the existing local contrastive
loss-based methods remains limited for learning good local representations
because similar and dissimilar local regions are defined based on random
augmentations and spatial proximity; not based on the semantic label of local
regions due to lack of large-scale expert annotations in the
semi/self-supervised setting. In this paper, we propose a local contrastive
loss to learn good pixel level features useful for segmentation by exploiting
semantic label information obtained from pseudo-labels of unlabeled images
alongside limited annotated images. In particular, we define the proposed loss
to encourage similar representations for the pixels that have the same
pseudo-label/ label while being dissimilar to the representation of pixels with
different pseudo-label/label in the dataset. We perform pseudo-label based
self-training and train the network by jointly optimizing the proposed
contrastive loss on both labeled and unlabeled sets and segmentation loss on
only the limited labeled set. We evaluated on three public cardiac and prostate
datasets, and obtain high segmentation performance.

    

### [[2112.09646] Generation of data on discontinuous manifolds via continuous stochastic non-invertible networks](http://arxiv.org/abs/2112.09646)


  The generation of discontinuous distributions is a difficult task for most
known frameworks such as generative autoencoders and generative adversarial
networks. Generative non-invertible models are unable to accurately generate
such distributions, require long training and often are subject to mode
collapse. Variational autoencoders (VAEs), which are based on the idea of
keeping the latent space to be Gaussian for the sake of a simple sampling,
allow an accurate reconstruction, while they experience significant limitations
at generation task. In this work, instead of trying to keep the latent space to
be Gaussian, we use a pre-trained contrastive encoder to obtain a clustered
latent space. Then, for each cluster, representing a unimodal submanifold, we
train a dedicated low complexity network to generate this submanifold from the
Gaussian distribution. The proposed framework is based on the
information-theoretic formulation of mutual information maximization between
the input data and latent space representation. We derive a link between the
cost functions and the information-theoretic formulation. We apply our approach
to synthetic 2D distributions to demonstrate both reconstruction and generation
of discontinuous distributions using continuous stochastic networks.

    

### [[2112.09653] Information-theoretic stochastic contrastive conditional GAN: InfoSCC-GAN](http://arxiv.org/abs/2112.09653)


  Conditional generation is a subclass of generative problems where the output
of the generation is conditioned by the attribute information. In this paper,
we present a stochastic contrastive conditional generative adversarial network
(InfoSCC-GAN) with an explorable latent space. The InfoSCC-GAN architecture is
based on an unsupervised contrastive encoder built on the InfoNCE paradigm, an
attribute classifier and an EigenGAN generator. We propose a novel training
method, based on generator regularization using external or internal attributes
every $n$-th iteration, using a pre-trained contrastive encoder and a
pre-trained classifier. The proposed InfoSCC-GAN is derived based on an
information-theoretic formulation of mutual information maximization between
input data and latent space representation as well as latent space and
generated data. Thus, we demonstrate a link between the training objective
functions and the above information-theoretic formulation. The experimental
results show that InfoSCC-GAN outperforms the "vanilla" EigenGAN in the image
generation on AFHQ and CelebA datasets. In addition, we investigate the impact
of discriminator architectures and loss functions by performing ablation
studies. Finally, we demonstrate that thanks to the EigenGAN generator, the
proposed framework enjoys a stochastic generation in contrast to vanilla
deterministic GANs yet with the independent training of encoder, classifier,
and generator in contrast to existing frameworks. Code, experimental results,
and demos are available online at this https URL.

    

### [[2112.09655] Distillation of RL Policies with Formal Guarantees via Variational Abstraction of Markov Decision Processes (Technical Report)](http://arxiv.org/abs/2112.09655)


  We consider the challenge of policy simplification and verification in the
context of policies learned through reinforcement learning (RL) in continuous
environments. In well-behaved settings, RL algorithms have convergence
guarantees in the limit. While these guarantees are valuable, they are
insufficient for safety-critical applications. Furthermore, they are lost when
applying advanced techniques such as deep-RL. To recover guarantees when
applying advanced RL algorithms to more complex environments with (i)
reachability, (ii) safety-constrained reachability, or (iii) discounted-reward
objectives, we build upon the DeepMDP framework introduced by Gelada et al. to
derive new bisimulation bounds between the unknown environment and a learned
discrete latent model of it. Our bisimulation bounds enable the application of
formal methods for Markov decision processes. Finally, we show how one can use
a policy obtained via state-of-the-art RL to efficiently train a variational
autoencoder that yields a discrete latent model with provably approximately
correct bisimulation guarantees. Additionally, we obtain a distilled version of
the policy for the latent model.

    

### [[2112.09668] Deep Learning for Spatiotemporal Modeling of Urbanization](http://arxiv.org/abs/2112.09668)


  Urbanization has a strong impact on the health and wellbeing of populations
across the world. Predictive spatial modeling of urbanization therefore can be
a useful tool for effective public health planning. Many spatial urbanization
models have been developed using classic machine learning and numerical
modeling techniques. However, deep learning with its proven capacity to capture
complex spatiotemporal phenomena has not been applied to urbanization modeling.
Here we explore the capacity of deep spatial learning for the predictive
modeling of urbanization. We treat numerical geospatial data as images with
pixels and channels, and enrich the dataset by augmentation, in order to
leverage the high capacity of deep learning. Our resulting model can generate
end-to-end multi-variable urbanization predictions, and outperforms a
state-of-the-art classic machine learning urbanization model in preliminary
comparisons.

    

### [[2112.09670] An Online Data-Driven Emergency-Response Method for Autonomous Agents in Unforeseen Situations](http://arxiv.org/abs/2112.09670)


  Reinforcement learning agents perform well when presented with inputs within
the distribution of those encountered during training. However, they are unable
to respond effectively when faced with novel, out-of-distribution events, until
they have undergone additional training. This paper presents an online,
data-driven, emergency-response method that aims to provide autonomous agents
the ability to react to unexpected situations that are very different from
those it has been trained or designed to address. In such situations, learned
policies cannot be expected to perform appropriately since the observations
obtained in these novel situations would fall outside the distribution of
inputs that the agent has been optimized to handle. The proposed approach
devises a customized response to the unforeseen situation sequentially, by
selecting actions that minimize the rate of increase of the reconstruction
error from a variational auto-encoder. This optimization is achieved online in
a data-efficient manner (on the order of 30 data-points) using a modified
Bayesian optimization procedure. We demonstrate the potential of this approach
in a simulated 3D car driving scenario, in which the agent devises a response
in under 2 seconds to avoid collisions with objects it has not seen during
training.

    

### [[2112.09684] On the existence of global minima and convergence analyses for gradient descent methods in the training of deep neural networks](http://arxiv.org/abs/2112.09684)


  In this article we study fully-connected feedforward deep ReLU ANNs with an
arbitrarily large number of hidden layers and we prove convergence of the risk
of the GD optimization method with random initializations in the training of
such ANNs under the assumption that the unnormalized probability density
function of the probability distribution of the input data of the considered
supervised learning problem is piecewise polynomial, under the assumption that
the target function (describing the relationship between input data and the
output data) is piecewise polynomial, and under the assumption that the risk
function of the considered supervised learning problem admits at least one
regular global minimum. In addition, in the special situation of shallow ANNs
with just one hidden layer and one-dimensional input we also verify this
assumption by proving in the training of such shallow ANNs that for every
Lipschitz continuous target function there exists a global minimum in the risk
landscape. Finally, in the training of deep ANNs with ReLU activation we also
study solutions of gradient flow (GF) differential equations and we prove that
every non-divergent GF trajectory converges with a polynomial rate of
convergence to a critical point (in the sense of limiting Fréchet
subdifferentiability). Our mathematical convergence analysis builds up on tools
from real algebraic geometry such as the concept of semi-algebraic functions
and generalized Kurdyka-Lojasiewicz inequalities, on tools from functional
analysis such as the Arzelà-Ascoli theorem, on tools from nonsmooth analysis
such as the concept of limiting Fréchet subgradients, as well as on the fact
that the set of realization functions of shallow ReLU ANNs with fixed
architecture forms a closed subset of the set of continuous functions revealed
by Petersen et al.

    

### [[1910.13742] Unifying mirror descent and dual averaging](http://arxiv.org/abs/1910.13742)


  We introduce and analyze a new family of first-order optimization algorithms
which generalizes and unifies both mirror descent and dual averaging. Within
the framework of this family, we define new algorithms for constrained
optimization that combines the advantages of mirror descent and dual averaging.
Our preliminary simulation study shows that these new algorithms significantly
outperform available methods in some situations.

    

### [[1911.12199] FOCUS: Flexible Optimizable Counterfactual Explanations for Tree Ensembles](http://arxiv.org/abs/1911.12199)


  Model interpretability has become an important problem in machine learning
(ML) due to the increased effect that algorithmic decisions have on humans.
Counterfactual explanations can help users understand not only why ML models
make certain decisions, but also how these decisions can be changed. We frame
the problem of finding counterfactual explanations as a gradient-based
optimization task and extend previous work that could only be applied to
differentiable models. In order to accommodate non-differentiable models such
as tree ensembles, we use probabilistic model approximations in the
optimization framework. We introduce an approximation technique that is
effective for finding counterfactual explanations for predictions of the
original model and show that our counterfactual examples are significantly
closer to the original instances than those produced by other methods
specifically designed for tree ensembles.

    

### [[2003.05438] Un-Mix: Rethinking Image Mixtures for Unsupervised Visual Representation Learning](http://arxiv.org/abs/2003.05438)


  The recently advanced unsupervised learning approaches use the siamese-like
framework to compare two "views" from the same image for learning
representations. Making the two views distinctive is a core to guarantee that
unsupervised methods can learn meaningful information. However, such frameworks
are sometimes fragile on overfitting if the augmentations used for generating
two views are not strong enough, causing the over-confident issue on the
training data. This drawback hinders the model from learning subtle variance
and fine-grained information. To address this, in this work we aim to involve
the distance concept on label space in the unsupervised learning and let the
model be aware of the soft degree of similarity between positive or negative
pairs through mixing the input data space, to further work collaboratively for
the input and loss spaces. Despite its conceptual simplicity, we show
empirically that with the solution -- Unsupervised image mixtures (Un-Mix), we
can learn subtler, more robust and generalized representations from the
transformed input and corresponding new label space. Extensive experiments are
conducted on CIFAR-10, CIFAR-100, STL-10, Tiny ImageNet and standard ImageNet
with popular unsupervised methods SimCLR, BYOL, MoCo V1&V2, SwAV, etc. Our
proposed image mixture and label assignment strategy can obtain consistent
improvement by 1~3% following exactly the same hyperparameters and training
procedures of the base methods. Code is publicly available at
this https URL.

    

### [[2008.05458] Deep-Learning-Based, Multi-Timescale Load Forecasting in Buildings: Opportunities and Challenges from Research to Deployment](http://arxiv.org/abs/2008.05458)


  Electricity load forecasting for buildings and campuses is becoming
increasingly important as the penetration of distributed energy resources
(DERs) grows. Efficient operation and dispatch of DERs require reasonably
accurate predictions of future energy consumption in order to conduct
near-real-time optimized dispatch of on-site generation and storage assets.
Electric utilities have traditionally performed load forecasting for load
pockets spanning large geographic areas, and therefore forecasting has not been
a common practice by buildings and campus operators. Given the growing trends
of research and prototyping in the grid-interactive efficient buildings domain,
characteristics beyond simple algorithm forecast accuracy are important in
determining true utility of the algorithm for smart buildings. Other
characteristics include the overall design of the deployed architecture and the
operational efficiency of the forecasting system. In this work, we present a
deep-learning-based load forecasting system that predicts the building load at
1-hour intervals for 18 hours in the future. We also discuss challenges
associated with the real-time deployment of such systems as well as the
research opportunities presented by a fully functional forecasting system that
has been developed within the National Renewable Energy Laboratory Intelligent
Campus program.

    

### [[2009.02835] E-BERT: A Phrase and Product Knowledge Enhanced Language Model for E-commerce](http://arxiv.org/abs/2009.02835)


  Pre-trained language models such as BERT have achieved great success in a
broad range of natural language processing tasks. However, BERT cannot well
support E-commerce related tasks due to the lack of two levels of domain
knowledge, i.e., phrase-level and product-level. On one hand, many E-commerce
tasks require an accurate understanding of domain phrases, whereas such
fine-grained phrase-level knowledge is not explicitly modeled by BERT's
training objective. On the other hand, product-level knowledge like product
associations can enhance the language modeling of E-commerce, but they are not
factual knowledge thus using them indiscriminately may introduce noise. To
tackle the problem, we propose a unified pre-training framework, namely,
E-BERT. Specifically, to preserve phrase-level knowledge, we introduce Adaptive
Hybrid Masking, which allows the model to adaptively switch from learning
preliminary word knowledge to learning complex phrases, based on the fitting
progress of two modes. To utilize product-level knowledge, we introduce
Neighbor Product Reconstruction, which trains E-BERT to predict a product's
associated neighbors with a denoising cross attention layer. Our investigation
reveals promising results in four downstream tasks, i.e., review-based question
answering, aspect extraction, aspect sentiment classification, and product
classification.

    

### [[2009.05530] TREX: Tree-Ensemble Representer-Point Explanations](http://arxiv.org/abs/2009.05530)


  How can we identify the training examples that contribute most to the
prediction of a tree ensemble? In this paper, we introduce TREX, an explanation
system that provides instance-attribution explanations for tree ensembles, such
as random forests and gradient boosted trees. TREX builds on the representer
point framework previously developed for explaining deep neural networks. Since
tree ensembles are non-differentiable, we define a kernel that captures the
structure of the specific tree ensemble. By using this kernel in kernel
logistic regression or a support vector machine, TREX builds a surrogate model
that approximates the original tree ensemble. The weights in the kernel
expansion of the surrogate model are used to define the global or local
importance of each training example.
Our experiments show that TREX's surrogate model accurately approximates the
tree ensemble; its global importance weights are more effective in dataset
debugging than the previous state-of-the-art; its explanations identify the
most influential samples better than alternative methods under the remove and
retrain evaluation framework; it runs orders of magnitude faster than
alternative methods; and its local explanations can identify and explain errors
due to domain mismatch.

    

### [[2009.10924] FusionStitching: Boosting Memory Intensive Computations for Deep Learning Workloads](http://arxiv.org/abs/2009.10924)


  We show in this work that memory intensive computations can result in severe
performance problems due to off-chip memory access and CPU-GPU context switch
overheads in a wide range of deep learning models. For this problem, current
just-in-time (JIT) kernel fusion and code generation techniques have
limitations, such as rough fusion plan exploration strategies and limited code
generation ability. We propose FusionStitching, a deep learning compiler
capable of fusing memory intensive operators, with varied data dependencies and
non-homogeneous parallelism, into large GPU kernels to reduce global memory
access and context switch overhead automatically. FusionStitching widens the
range of operation combinations that fusion can target beyond previous JIT
works by introducing data reuse of intermediate values. It explores large
fusion spaces to decide optimal fusion plans with considerations of memory
access costs, kernel calls and resource usage constraints. FusionStitching
tunes the optimal stitching scheme with a domain-specific cost model
efficiently. Experimental results show that FusionStitching can reach up to
2.21x speedup compared to state-of-the-art, with 1.45x on average. Besides
these experimental results, we integrated our approach into a compiler product
and deployed it onto a production cluster for AI workloads with thousands of
GPUs. The system has been in operation for more than 4 months and saves 7,000
GPU hours on average for approximately 30,000 tasks per month.

    

### [[2010.16143] HyperText: Endowing FastText with Hyperbolic Geometry](http://arxiv.org/abs/2010.16143)


  Natural language data exhibit tree-like hierarchical structures such as the
hypernym-hyponym relations in WordNet. FastText, as the state-of-the-art text
classifier based on shallow neural network in Euclidean space, may not model
such hierarchies precisely with limited representation capacity. Considering
that hyperbolic space is naturally suitable for modeling tree-like hierarchical
data, we propose a new model named HyperText for efficient text classification
by endowing FastText with hyperbolic geometry. Empirically, we show that
HyperText outperforms FastText on a range of text classification tasks with
much reduced parameters.

    

### [[2011.07456] State-Dependent Temperature Control for Langevin Diffusions](http://arxiv.org/abs/2011.07456)


  We study the temperature control problem for Langevin diffusions in the
context of non-convex optimization. The classical optimal control of such a
problem is of the bang-bang type, which is overly sensitive to errors. A remedy
is to allow the diffusions to explore other temperature values and hence smooth
out the bang-bang control. We accomplish this by a stochastic relaxed control
formulation incorporating randomization of the temperature control and
regularizing its entropy. We derive a state-dependent, truncated exponential
distribution, which can be used to sample temperatures in a Langevin algorithm,
in terms of the solution to an HJB partial differential equation. We carry out
a numerical experiment on a one-dimensional baseline example, in which the HJB
equation can be easily solved, to compare the performance of the algorithm with
three other available algorithms in search of a global optimum.

    

### [[2012.00893] Evaluating Explanations: How much do explanations from the teacher aid students?](http://arxiv.org/abs/2012.00893)


  While many methods purport to explain predictions by highlighting salient
features, what aims these explanations serve and how they ought to be evaluated
often go unstated. In this work, we introduce a framework to quantify the value
of explanations via the accuracy gains that they confer on a student model
trained to simulate a teacher model. Crucially, the explanations are available
to the student during training, but are not available at test time. Compared to
prior proposals, our approach is less easily gamed, enabling principled,
automatic, model-agnostic evaluation of attributions. Using our framework, we
compare numerous attribution methods for text classification and question
answering, and observe quantitative differences that are consistent (to a
moderate to high degree) across different student model architectures and
learning strategies.

    

### [[2103.02200] Formalizing Generalization and Robustness of Neural Networks to Weight Perturbations](http://arxiv.org/abs/2103.02200)


  Studying the sensitivity of weight perturbation in neural networks and its
impacts on model performance, including generalization and robustness, is an
active research topic due to its implications on a wide range of machine
learning tasks such as model compression, generalization gap assessment, and
adversarial attacks. In this paper, we provide the first integral study and
analysis for feed-forward neural networks in terms of the robustness in
pairwise class margin and its generalization behavior under weight
perturbation. We further design a new theory-driven loss function for training
generalizable and robust neural networks against weight perturbations.
Empirical experiments are conducted to validate our theoretical analysis. Our
results offer fundamental insights for characterizing the generalization and
robustness of neural networks against weight perturbations.

    

### [[2103.10922] Landscape analysis for shallow neural networks: complete classification of critical points for affine target functions](http://arxiv.org/abs/2103.10922)


  In this paper, we analyze the landscape of the true loss of one-hidden-layer
networks with ReLU, leaky ReLU, as well as with quadratic activation. In all
three cases, we provide a complete classification of the critical points in the
case where the target function is affine. In particular, we show that there
exist no local maxima and clarify the structure of saddle points. Moreover, we
prove that non-global local minima can only be caused by `dead' ReLU neurons.
In particular, they do not appear in the case of leaky ReLU or quadratic
activation. Our approach is of a combinatorial nature and builds on a careful
analysis of the different types of hidden neurons that can occur.

    

### [[2104.00501] NuPS: A Parameter Server for Machine Learning with Non-Uniform Parameter Access](http://arxiv.org/abs/2104.00501)


  Parameter servers (PSs) facilitate the implementation of distributed training
for large machine learning tasks. In this paper, we argue that existing PSs are
inefficient for tasks that exhibit non-uniform parameter access; their
performance may even fall behind that of single node baselines. We identify two
major sources of such non-uniform access: skew and sampling. Existing PSs are
ill-suited for managing skew because they uniformly apply the same parameter
management technique to all parameters. They are inefficient for sampling
because the PS is oblivious to the associated randomized accesses and cannot
exploit locality. To overcome these performance limitations, we introduce NuPS,
a novel PS architecture that (i) integrates multiple management techniques and
employs a suitable technique for each parameter and (ii) supports sampling
directly via suitable sampling primitives and sampling schemes that allow for a
controlled quality--efficiency trade-off. In our experimental study, NuPS
outperformed existing PSs by up to one order of magnitude and provided up to
linear scalability across multiple machine learning tasks.

    

### [[2104.01086] Defending Against Image Corruptions Through Adversarial Augmentations](http://arxiv.org/abs/2104.01086)


  Modern neural networks excel at image classification, yet they remain
vulnerable to common image corruptions such as blur, speckle noise or fog.
Recent methods that focus on this problem, such as AugMix and DeepAugment,
introduce defenses that operate in expectation over a distribution of image
corruptions. In contrast, the literature on $\ell_p$-norm bounded perturbations
focuses on defenses against worst-case corruptions. In this work, we reconcile
both approaches by proposing AdversarialAugment, a technique which optimizes
the parameters of image-to-image models to generate adversarially corrupted
augmented images. We theoretically motivate our method and give sufficient
conditions for the consistency of its idealized version as well as that of
DeepAugment. Our classifiers improve upon the state-of-the-art on common image
corruption benchmarks conducted in expectation on CIFAR-10-C and improve
worst-case performance against $\ell_p$-norm bounded perturbations on both
CIFAR-10 and ImageNet.

    

### [[2104.09172] Direction-Aggregated Attack for Transferable Adversarial Examples](http://arxiv.org/abs/2104.09172)


  Deep neural networks are vulnerable to adversarial examples that are crafted
by imposing imperceptible changes to the inputs. However, these adversarial
examples are most successful in white-box settings where the model and its
parameters are available. Finding adversarial examples that are transferable to
other models or developed in a black-box setting is significantly more
difficult. In this paper, we propose the Direction-Aggregated adversarial
attacks that deliver transferable adversarial examples. Our method utilizes
aggregated direction during the attack process for avoiding the generated
adversarial examples overfitting to the white-box model. Extensive experiments
on ImageNet show that our proposed method improves the transferability of
adversarial examples significantly and outperforms state-of-the-art attacks,
especially against adversarial robust models. The best averaged attack success
rates of our proposed method reaches 94.6\% against three adversarial trained
models and 94.8\% against five defense methods. It also reveals that current
defense approaches do not prevent transferable adversarial attacks.

    

### [[2104.14900] Discrete-Time Mean Field Control with Environment States](http://arxiv.org/abs/2104.14900)


  Multi-agent reinforcement learning methods have shown remarkable potential in
solving complex multi-agent problems but mostly lack theoretical guarantees.
Recently, mean field control and mean field games have been established as a
tractable solution for large-scale multi-agent problems with many agents. In
this work, driven by a motivating scheduling problem, we consider a
discrete-time mean field control model with common environment states. We
rigorously establish approximate optimality as the number of agents grows in
the finite agent case and find that a dynamic programming principle holds,
resulting in the existence of an optimal stationary policy. As exact solutions
are difficult in general due to the resulting continuous action space of the
limiting mean field Markov decision process, we apply established deep
reinforcement learning methods to solve the associated mean field control
problem. The performance of the learned mean field control policy is compared
to typical multi-agent reinforcement learning approaches and is found to
converge to the mean field performance for sufficiently many agents, verifying
the obtained theoretical results and reaching competitive solutions.

    

### [[2104.14912] Nearest-Neighbor-based Collision Avoidance for Quadrotors via Reinforcement Learning](http://arxiv.org/abs/2104.14912)


  Collision avoidance algorithms are of central interest to many drone
applications. In particular, decentralized approaches may be the key to
enabling robust drone swarm solutions in cases where centralized communication
becomes computationally prohibitive. In this work, we draw biological
inspiration from flocks of starlings (Sturnus vulgaris) and apply the insight
to end-to-end learned decentralized collision avoidance. More specifically, we
propose a new, scalable observation model following a biomimetic
nearest-neighbor information constraint that leads to fast learning and good
collision avoidance behavior. By proposing a general reinforcement learning
approach, we obtain an end-to-end learning-based approach to integrating
collision avoidance with arbitrary tasks such as package collection and
formation change. To validate the generality of this approach, we successfully
apply our methodology through motion models of medium complexity, modeling
momentum and nonetheless allowing direct application to real world quadrotors
in conjunction with a standard PID controller. In contrast to prior works, we
find that in our sufficiently rich motion model, nearest-neighbor information
is indeed enough to learn effective collision avoidance behavior. Our learned
policies are tested in simulation and subsequently transferred to real-world
drones to validate their real-world applicability.

    

### [[2105.05566] Structural risk minimization for quantum linear classifiers](http://arxiv.org/abs/2105.05566)


  Quantum machine learning (QML) models based on parameterized quantum circuits
are often highlighted as candidates for quantum computing's near-term "killer
application". However, the understanding of the empirical and generalization
performance of these models is still in its infancy. In this paper, we study
how to balance between training accuracy and generalization performance (also
called structural risk minimization) for two prominent QML models introduced by
Havlíček et al. (Nature, 2019), and Schuld and Killoran (PRL, 2019).
Firstly, using relationships to well-understood classical models, we prove that
two model parameters -- i.e., the dimension of the sum of the images and the
Frobenius norm of the observables used by the model -- closely control the
models' complexity and therefore its generalization performance. Secondly,
using ideas inspired by process tomography, we prove that these model
parameters also closely control the models' ability to capture correlations in
sets of training examples. In summary, our results give rise to new options for
structural risk minimization for QML models.

    

### [[2105.06715] Maximizing Mutual Information Across Feature and Topology Views for Learning Graph Representations](http://arxiv.org/abs/2105.06715)


  Recently, maximizing mutual information has emerged as a powerful method for
unsupervised graph representation learning. The existing methods are typically
effective to capture information from the topology view but ignore the feature
view. To circumvent this issue, we propose a novel approach by exploiting
mutual information maximization across feature and topology views.
Specifically, we first utilize a multi-view representation learning module to
better capture both local and global information content across feature and
topology views on graphs. To model the information shared by the feature and
topology spaces, we then develop a common representation learning module using
mutual information maximization and reconstruction loss minimization. To
explicitly encourage diversity between graph representations from the same
view, we also introduce a disagreement regularization to enlarge the distance
between representations from the same view. Experiments on synthetic and
real-world datasets demonstrate the effectiveness of integrating feature and
topology views. In particular, compared with the previous supervised methods,
our proposed method can achieve comparable or even better performance under the
unsupervised representation and linear evaluation protocol.

    

### [[2105.07059] Momentum Contrastive Voxel-wise Representation Learning for Semi-supervised Volumetric Medical Image Segmentation](http://arxiv.org/abs/2105.07059)


  Automated segmentation in medical image analysis is a challenging task that
requires a large amount of manually labeled data. However, manually annotating
medical data is often laborious, and most existing learning-based approaches
fail to accurately delineate object boundaries without effective geometric
constraints. Contrastive learning, a sub-area of self-supervised learning, has
recently been noted as a promising direction in multiple application fields. In
this work, we present a novel Contrastive Voxel-wise Representation
Distillation (CVRD) method with geometric constraints to learn global-local
visual representations for volumetric medical image segmentation with limited
annotations. Our framework can effectively learn global and local features by
capturing 3D spatial context and rich anatomical information. Specifically, we
introduce a voxel-to-volume contrastive algorithm to learn global information
from 3D images, and propose to perform local voxel-to-voxel distillation to
explicitly make use of local cues in the embedding space. Moreover, we
integrate an elastic interaction-based active contour model as a geometric
regularization term to enable fast and reliable object delineations in an
end-to-end learning manner. Results on the Atrial Segmentation Challenge
dataset demonstrate superiority of our proposed scheme, especially in a setting
with a very limited number of annotated data. The code will be available at
this https URL.

    

### [[2105.08164] Parallel and Flexible Sampling from Autoregressive Models via Langevin Dynamics](http://arxiv.org/abs/2105.08164)


  This paper introduces an alternative approach to sampling from autoregressive
models. Autoregressive models are typically sampled sequentially, according to
the transition dynamics defined by the model. Instead, we propose a sampling
procedure that initializes a sequence with white noise and follows a Markov
chain defined by Langevin dynamics on the global log-likelihood of the
sequence. This approach parallelizes the sampling process and generalizes to
conditional sampling. Using an autoregressive model as a Bayesian prior, we can
steer the output of a generative model using a conditional likelihood or
constraints. We apply these techniques to autoregressive models in the visual
and audio domains, with competitive results for audio source separation,
super-resolution, and inpainting.

    

### [[2105.14073] Task-Guided Inverse Reinforcement Learning Under Partial Information](http://arxiv.org/abs/2105.14073)


  We study the problem of inverse reinforcement learning (IRL), where the
learning agent recovers a reward function using expert demonstrations. Most of
the existing IRL techniques make the often unrealistic assumption that the
agent has access to full information about the environment. We remove this
assumption by developing an algorithm for IRL in partially observable Markov
decision processes (POMDPs). The algorithm addresses several limitations of
existing techniques that do not take the information asymmetry between the
expert and the learner into account. First, it adopts causal entropy as the
measure of the likelihood of the expert demonstrations as opposed to entropy in
most existing IRL techniques, and avoids a common source of algorithmic
complexity. Second, it incorporates task specifications expressed in temporal
logic into IRL. Such specifications may be interpreted as side information
available to the learner a priori in addition to the demonstrations and may
reduce the information asymmetry. Nevertheless, the resulting formulation is
still nonconvex due to the intrinsic nonconvexity of the so-called forward
problem, i.e., computing an optimal policy given a reward function, in POMDPs.
We address this nonconvexity through sequential convex programming and
introduce several extensions to solve the forward problem in a scalable manner.
This scalability allows computing policies that incorporate memory at the
expense of added computational cost yet also outperform memoryless policies. We
demonstrate that, even with severely limited data, the algorithm learns reward
functions and policies that satisfy the task and induce a similar behavior to
the expert by leveraging the side information and incorporating memory into the
policy.

    

### [[2106.03962] Amortized Generation of Sequential Algorithmic Recourses for Black-box Models](http://arxiv.org/abs/2106.03962)


  Explainable machine learning (ML) has gained traction in recent years due to
the increasing adoption of ML-based systems in many sectors. Algorithmic
Recourses (ARs) provide "what if" feedback of the form "if an input datapoint
were x' instead of x, then an ML-based system's output would be y' instead of
y." ARs are attractive due to their actionable feedback, amenability to
existing legal frameworks, and fidelity to the underlying ML model. Yet,
current AR approaches are single shot -- that is, they assume x can change to
x' in a single time period. We propose a novel stochastic-control-based
approach that generates sequential ARs, that is, ARs that allow x to move
stochastically and sequentially across intermediate states to a final state x'.
Our approach is model agnostic and black box. Furthermore, the calculation of
ARs is amortized such that once trained, it applies to multiple datapoints
without the need for re-optimization. In addition to these primary
characteristics, our approach admits optional desiderata such as adherence to
the data manifold, respect for causal relations, and sparsity -- identified by
past research as desirable properties of ARs. We evaluate our approach using
three real-world datasets and show successful generation of sequential ARs that
respect other recourse desiderata.

    

### [[2106.04465] Detecting Anomalous Event Sequences with Temporal Point Processes](http://arxiv.org/abs/2106.04465)


  Automatically detecting anomalies in event data can provide substantial value
in domains such as healthcare, DevOps, and information security. In this paper,
we frame the problem of detecting anomalous continuous-time event sequences as
out-of-distribution (OoD) detection for temporal point processes (TPPs). First,
we show how this problem can be approached using goodness-of-fit (GoF) tests.
We then demonstrate the limitations of popular GoF statistics for TPPs and
propose a new test that addresses these shortcomings. The proposed method can
be combined with various TPP models, such as neural TPPs, and is easy to
implement. In our experiments, we show that the proposed statistic excels at
both traditional GoF testing, as well as at detecting anomalies in simulated
and real-world data.

    

### [[2106.05087] Who Is the Strongest Enemy? Towards Optimal and Efficient Evasion Attacks in Deep RL](http://arxiv.org/abs/2106.05087)


  Evaluating the worst-case performance of a reinforcement learning (RL) agent
under the strongest/optimal adversarial perturbations on state observations
(within some constraints) is crucial for understanding the robustness of RL
agents. However, finding the optimal adversary is challenging, in terms of both
whether we can find the optimal attack and how efficiently we can find it.
Existing works on adversarial RL either use heuristics-based methods that may
not find the strongest adversary, or directly train an RL-based adversary by
treating the agent as a part of the environment, which can find the optimal
adversary but may become intractable in a large state space. This paper
introduces a novel attacking method to find the optimal attacks through
collaboration between a designed function named ''actor'' and an RL-based
learner named "director". The actor crafts state perturbations for a given
policy perturbation direction, and the director learns to propose the best
policy perturbation directions. Our proposed algorithm, PA-AD, is theoretically
optimal and significantly more efficient than prior RL-based works in
environments with large state spaces. Empirical results show that our proposed
PA-AD universally outperforms state-of-the-art attacking methods in various
Atari and MuJoCo environments. By applying PA-AD to adversarial training, we
achieve state-of-the-art empirical robustness in multiple tasks under strong
adversaries.

    

### [[2106.06587] Classification algorithms applied to structure formation simulations](http://arxiv.org/abs/2106.06587)


  Throughout cosmological simulations, the properties of the matter density
field in the initial conditions have a decisive impact on the features of the
structures formed today. In this paper we use a random-forest classification
algorithm to infer whether or not dark matter particles, traced back to the
initial conditions, would end up in dark matter halos whose masses are above
some threshold. This problem might be posed as a binary classification task,
where the initial conditions of the matter density field are mapped into
classification labels provided by a halo finder program. Our results show that
random forests are effective tools to predict the output of cosmological
simulations without running the full process. These techniques might be used in
the future to decrease the computational time and to explore more efficiently
the effect of different dark matter/dark energy candidates on the formation of
cosmological structures.

    

### [[2106.10349] The Perils of Learning Before Optimizing](http://arxiv.org/abs/2106.10349)


  Formulating real-world optimization problems often begins with making
predictions from historical data (e.g., an optimizer that aims to recommend
fast routes relies upon travel-time predictions). Typically, learning the
prediction model used to generate the optimization problem and solving that
problem are performed in two separate stages. Recent work has showed how such
prediction models can be learned end-to-end by differentiating through the
optimization task. Such methods often yield empirical improvements, which are
typically attributed to end-to-end making better error tradeoffs than the
standard loss function used in a two-stage solution. We refine this explanation
and more precisely characterize when end-to-end can improve performance. When
prediction targets are stochastic, a two-stage solution must make an a priori
choice about which statistics of the target distribution to model-we consider
expectations over prediction targets-while an end-to-end solution can make this
choice adaptively. We show that the performance gap between a two-stage and
end-to-end approach is closely related to the price of correlation concept in
stochastic optimization and show the implications of some existing POC results
for the predict-then-optimize problem. We then consider a novel and
particularly practical setting, where multiple prediction targets are combined
to obtain each of the objective function's coefficients. We give explicit
constructions where (1) two-stage performs unboundedly worse than end-to-end;
and (2) two-stage is optimal. We use simulations to experimentally quantify
performance gaps and identify a wide range of real-world applications from the
literature whose objective functions rely on multiple prediction targets,
suggesting that end-to-end learning could yield significant improvements.

    

### [[2112.06654] Toward Open-World Electroencephalogram Decoding Via Deep Learning: A Comprehensive Survey](http://arxiv.org/abs/2112.06654)


  Electroencephalogram (EEG) decoding aims to identify the perceptual,
semantic, and cognitive content of neural processing based on non-invasively
measured brain activity. Traditional EEG decoding methods have achieved
moderate success when applied to data acquired in static, well-controlled lab
environments. However, an open-world environment is a more realistic setting,
where situations affecting EEG recordings can emerge unexpectedly,
significantly weakening the robustness of existing methods. In recent years,
deep learning (DL) has emerged as a potential solution for such problems due to
its superior capacity in feature extraction. It overcomes the limitations of
defining `handcrafted' features or features extracted using shallow
architectures, but typically requires large amounts of costly,
expertly-labelled data - something not always obtainable. Combining DL with
domain-specific knowledge may allow for development of robust approaches to
decode brain activity even with small-sample data. Although various DL methods
have been proposed to tackle some of the challenges in EEG decoding, a
systematic tutorial overview, particularly for open-world applications, is
currently lacking. This article therefore provides a comprehensive survey of DL
methods for open-world EEG decoding, and identifies promising research
directions to inspire future studies for EEG decoding in real-world
applications.

    

### [[1506.04319] Generating and Exploring S-Box Multivariate Quadratic Equation Systems with SageMath](http://arxiv.org/abs/1506.04319)


  A new method to derive Multivariate Quadratic equation systems (MQ) for the
input and output bit variables of a cryptographic S-box from its algebraic
expressions with the aid of the computer mathematics software system SageMath
is presented. We consolidate the deficiency of previously presented MQ metrics,
supposed to quantify the resistance of S-boxes against algebraic attacks.

    

### [[2112.09234] DeFT: A Deadlock-Free and Fault-Tolerant Routing Algorithm for 2.5D Chiplet Networks](http://arxiv.org/abs/2112.09234)


  By interconnecting smaller chiplets through an interposer, 2.5D integration
offers a cost-effective and high-yield solution to implement large-scale
modular systems. Nevertheless, the underlying network is prone to deadlock,
despite deadlock-free chiplets, and to different faults on the vertical links
used for connecting the chiplets to the interposer. Unfortunately, existing
fault-tolerant routing techniques proposed for 2D and 3D on-chip networks
cannot be applied to chiplet networks. To address these problems, this paper
presents the first deadlock-free and fault-tolerant routing algorithm, called
DeFT, for 2.5D integrated chiplet systems. DeFT improves the redundancy in
vertical-link selection to tolerate faults in vertical links while considering
network congestion. Moreover, DeFT can tolerate different vertical-link-fault
scenarios while accounting for vertical-link utilization. Compared to the
state-of-the-art routing algorithms in 2.5D chiplet systems, our simulation
results show that DeFT improves network reachability by up to 75% with a fault
rate of up to 25% and reduces the network latency by up to 40% for
multi-application execution scenarios with less than 2% area overhead.

    

### [[2112.09320] Gate-Level Static Approximate Adders](http://arxiv.org/abs/2112.09320)


  This work compares and analyzes static approximate adders which are suitable
for FPGA and ASIC type implementations. We consider many static approximate
adders and evaluate their performance with respect to a digital image
processing application using standard figures of merit such as peak signal to
noise ratio and structural similarity index metric. We provide the error
metrics of approximate adders, and the design metrics of accurate and
approximate adders corresponding to FPGA and ASIC type implementations. For the
FPGA implementation, we considered a Xilinx Artix-7 FPGA, and for an ASIC type
implementation, we considered a 32-28 nm CMOS standard digital cell library.
While the inferences from this work could serve as a useful reference to
determine an optimum static approximate adder for a practical application, in
particular, we found approximate adders HOAANED, HERLOA and M-HERLOA to be
preferable.

    

### [[2106.11840] Quantum Computing -- from NISQ to PISQ](http://arxiv.org/abs/2106.11840)


  Given the impeding timeline of developing good quality quantum processing
units, it is the moment to rethink the approach to advance quantum computing
research. Rather than waiting for quantum hardware technologies to mature, we
need to start assessing in tandem the impact of the occurrence of quantum
computing in various scientific fields. However, to this purpose, we need to
use a complementary but quite different approach than proposed by the NISQ
vision, which is heavily focused on and burdened by the engineering challenges.
That is why we propose and advocate the PISQ approach: Perfect Intermediate
Scale Quantum computing based on the already known concept of perfect qubits.
This will allow researchers to focus much more on the development of new
applications by defining the algorithms in terms of perfect qubits and evaluate
them on quantum computing simulators that are executed on supercomputers. It is
not the long-term solution but will currently allow universities to research on
quantum logic and algorithms and companies can already start developing their
internal know-how on quantum solutions.

    

### [[2112.09384] An Exact Algorithm for the Linear Tape Scheduling Problem](http://arxiv.org/abs/2112.09384)


  Magnetic tapes are often considered as an outdated storage technology, yet
they are still used to store huge amounts of data. Their main interests are a
large capacity and a low price per gigabyte, which come at the cost of a much
larger file access time than on disks. With tapes, finding the right ordering
of multiple file accesses is thus key to performance. Moving the reading head
back and forth along a kilometer long tape has a non-negligible cost and
unnecessary movements thus have to be avoided. However, the optimization of
tape request ordering has then rarely been studied in the scheduling
literature, much less than I/O scheduling on disks. For instance, minimizing
the average service time for several read requests on a linear tape remains an
open question. Therefore, in this paper, we aim at improving the quality of
service experienced by users of tape storage systems, and not only the peak
performance of such systems. To this end, we propose a reasonable
polynomial-time exact algorithm while this problem and simpler variants have
been conjectured NP-hard. We also refine the proposed model by considering
U-turn penalty costs accounting for inherent mechanical accelerations. Then, we
propose a low-cost variant of our optimal algorithm by restricting the solution
space, yet still yielding an accurate suboptimal solution. Finally, we compare
our algorithms to existing solutions from the literature on logs of the mass
storage management system of a major datacenter. This allows us to assess the
quality of previous solutions and the improvement achieved by our low-cost
algorithm. Aiming for reproducibility, we make available the complete
implementation of the algorithms used in our evaluation, alongside the dataset
of tape requests that is, to the best of our knowledge, the first of its kind
to be publicly released.

    

### [[2112.09437] Detectable Quantum Byzantine Agreement for Any Arbitrary Number of Dishonest Parties](http://arxiv.org/abs/2112.09437)


  Reaching agreement in the presence of arbitrary faults is a fundamental
problem in distributed computation, which has been shown to be unsolvable if
one-third of the processes can fail, unless signed messages are used. In this
paper, we propose a solution to a variation of the original BA problem, called
Detectable Byzantine Agreement (DBA), that does not need to use signed
messages. The proposed algorithm uses what we call $Q$-correlated lists, which
are generated by a quantum source device. Once each process has one of these
lists, they use them to reach the agreement in a classical manner. Although, in
general, the agreement is reached by using $m+1$ rounds (where $m$ is the
number of processes that can fail), if less than one-third of the processes
fail it only needs one round to reach the agreement.

    

### [[2112.09479] Memory Efficient Massively Parallel Algorithms for LCL Problems on Trees](http://arxiv.org/abs/2112.09479)


  In this work, we develop the low-space Massively Parallel Computation (MPC)
complexity landscape for a family of fundamental graph problems on trees. We
present a general method that solves most locally checkable labeling (LCL)
problems exponentially faster in the low-space MPC model than in the LOCAL
message passing model. In particular, we show that all solvable LCL problems on
trees can be solved in $O(\log n)$ time (high-complexity regime) and that all
LCL problems on trees with deterministic complexity $n^{o(1)}$ in the LOCAL
model can be solved in $O(\log \log n)$ time (mid-complexity regime). We
emphasize that we solve LCL problems on constant-degree trees, our algorithms
are deterministic and they work in the low-space MPC model, where local memory
is $O(n^\delta)$ for $\delta \in (0,1)$ and global memory is $O(m)$.
For the high-complexity regime, there are two key ingredients. One is a novel
$O(\log n)$-time tree rooting algorithm, which may be of independent interest.
The other ingredient is a novel pointer-chain technique and analysis that
allows us to solve any solvable LCL problem on trees in $O(\log n)$ time. For
the mid-complexity regime, we adapt the approach by Chang and Pettie [FOCS'17],
who gave a canonical LOCAL algorithm for solving LCL problems on trees. For the
special case of 3-coloring trees, which is a natural LCL problem with LOCAL
time complexity $n^{o(1)}$, we show that our analysis is (conditionally) tight,
as it matches the conditional $\Omega(\log \log n)$-time lower bound for
component-stable algorithms.

    

### [[2112.09509] Mitigating inefficient task mappings with an Adaptive Resource-Moldable Scheduler (ARMS)](http://arxiv.org/abs/2112.09509)


  Efficient runtime task scheduling on complex memory hierarchy becomes
increasingly important as modern and future High-Performance Computing (HPC)
systems are progressively composed of multisocket and multi-chiplet nodes with
nonuniform memory access latencies. Existing locality-aware scheduling schemes
either require control of the data placement policy for memory-bound tasks or
maximize locality for all classes of computations, resulting in a loss of
potential performance. While such approaches are viable, an adaptive scheduling
strategy is preferred to enhance locality and resource sharing efficiency using
a portable programming scheme. In this paper, we propose the Adaptive
Resource-Moldable Scheduler (ARMS) that dynamically maps a task at runtime to a
partition spanning one or more threads, based on the task and DAG requirements.
The scheduler builds an online platform-independent model for the local and
non-local scheduling costs for each tuple consisting of task type (function)
and task topology (task location within DAG). We evaluate ARMS using
task-parallel versions of SparseLU, 2D Stencil, FMM, and MatMul as examples.
Compared to previous approaches, ARMS achieves up to 3.5x performance gain over
state-of-the-art locality-aware scheduling schemes.

    

### [[2112.09514] Call for establishing benchmark science and engineering](http://arxiv.org/abs/2112.09514)


  This article investigates the origin and evolution of the benchmark term.
Five categories of benchmarks are summarized, including measurement standards,
standardized data sets with defined properties, representative workloads,
representative data sets, and best practices, which widely exist in
multi-disciplines. I believe there are two pressing challenges in growing this
discipline: establishing consistent benchmarking across multi-disciplines and
developing meta-benchmark to measure the benchmarks themselves. I propose
establishing benchmark science and engineering; one of the primary goal is to
setup a standard benchmark hierarchy across multi-disciplines. It is the right
time to launch a multi-disciplinary benchmark, standard, and evaluation
journal, TBench, to communicate the state-of-the-art and state-of-the-practice
of benchmark science and engineering.

    

### [[2112.09560] Dynamic resource allocation for efficient parallel CFD simulations](http://arxiv.org/abs/2112.09560)


  CFD users of supercomputers usually resort to rule-of-thumb methods to select
the number of subdomains (partitions) when relying on MPI-based
parallelization. One common approach is to set a minimum number of elements or
cells per subdomain, under which the parallel efficiency of the code is "known"
to fall below a subjective level, say 80%. The situation is even worse when the
user is not aware of the "good" practices for the given code and a huge amount
of resources can thus be wasted. This work presents an elastic computing
methodology to adapt at runtime the resources allocated to a simulation
automatically. The criterion to control the required resources is based on a
runtime measure of the communication efficiency of the execution. According to
some analytical estimates, the resources are then expanded or reduced to fulfil
this criterion and eventually execute an efficient simulation.

    

### [[2112.09580] Continuously Testing Distributed IoT Systems: An Overview of the State of the Art](http://arxiv.org/abs/2112.09580)


  The continuous testing of small changes to systems has proven to be useful
and is widely adopted in the development of software systems. For this,
software is tested in environments that are as close as possible to the
production environments.
When testing IoT systems, this approach is met with unique challenges that
stem from the typically large scale of the deployments, heterogeneity of nodes,
challenging network characteristics, and tight integration with the environment
among others. IoT test environments present a possible solution to these
challenges by emulating the nodes, networks, and possibly domain environments
in which IoT applications can be executed.
This paper gives an overview of the state of the art in IoT testing. We
derive desirable characteristics of IoT test environments, compare 18 tools
that can be used in this respect, and give a research outlook of future trends
in this area.

    

### [[2010.08423] Restless reachability problems in temporal graphs](http://arxiv.org/abs/2010.08423)


  We study a family of reachability problems under waiting-time restrictions in
temporal and vertex-colored temporal graphs. Given a temporal graph and a set
of source vertices, we find the set of vertices that are reachable from a
source via a time-respecting path, where the difference in timestamps between
consecutive edges is at most a resting time. Given a vertex-colored temporal
graph and a multiset query of colors, we find the set of vertices reachable
from a source via a time-respecting path such that the vertex colors of the
path agree with the multiset query and the difference in timestamps between
consecutive edges is at most a resting time. These kind of problems have
several applications in understanding the spread of a disease in a network,
tracing contacts in epidemic outbreaks, finding signaling pathways in the brain
network, and recommending tours for tourists.
We present an algebraic algorithmic framework based on constrained
multilinear sieving for solving the restless reachability problems we propose.
In particular, parameterized by the length of a path $k$ sought, we show the
problems can be solved in $O(2^k k m \Delta)$ time and $O(n \tau)$ space, where
$n$ is the number of vertices, $m$ the number of edges, $\Delta$ the maximum
resting time and $\tau$ the maximum timestamp of an input temporal graph. In
addition, we prove that the algorithms presented for the restless reachability
problems in vertex-colored temporal graphs are optimal under plausible
complexity-theoretic assumptions. Finally, with an open-source implementation,
we demonstrate that our algorithm scales to large graphs with up to one billion
temporal edges, despite the problems being NP-hard. Specifically, we present
extensive experiments to evaluate our scalability claims both on synthetic and
real-world graphs.

    

### [[2112.09135] ASC-Net: Unsupervised Medical Anomaly Segmentation Using an Adversarial-based Selective Cutting Network](http://arxiv.org/abs/2112.09135)


  In this paper we consider the problem of unsupervised anomaly segmentation in
medical images, which has attracted increasing attention in recent years due to
the expensive pixel-level annotations from experts and the existence of a large
amount of unannotated normal and abnormal image scans. We introduce a
segmentation network that utilizes adversarial learning to partition an image
into two cuts, with one of them falling into a reference distribution provided
by the user. This Adversarial-based Selective Cutting network (ASC-Net) bridges
the two domains of cluster-based deep segmentation and adversarial-based
anomaly/novelty detection algorithms. Our ASC-Net learns from normal and
abnormal medical scans to segment anomalies in medical scans without any masks
for supervision. We evaluate this unsupervised anomly segmentation model on
three public datasets, i.e., BraTS 2019 for brain tumor segmentation, LiTS for
liver lesion segmentation, and MS-SEG 2015 for brain lesion segmentation, and
also on a private dataset for brain tumor segmentation. Compared to existing
methods, our model demonstrates tremendous performance gains in unsupervised
anomaly segmentation tasks. Although there is still room to further improve
performance compared to supervised learning algorithms, the promising
experimental results and interesting observations shed light on building an
unsupervised learning algorithm for medical anomaly identification using
user-defined knowledge.

    

### [[2112.09171] Causal Modeling With Infinitely Many Variables](http://arxiv.org/abs/2112.09171)


  Structural-equations models (SEMs) are perhaps the most commonly used
framework for modeling causality. However, as we show, naively extending this
framework to infinitely many variables, which is necessary, for example, to
model dynamical systems, runs into several problems. We introduce GSEMs
(generalized SEMs), a flexible generalization of SEMs that directly specify the
results of interventions, in which (1) systems of differential equations can be
represented in a natural and intuitive manner, (2) certain natural situations,
which cannot be represented by SEMs at all, can be represented easily, (3) the
definition of actual causality in SEMs carries over essentially without change.

    

### [[2112.09197] Verification of Neural-Network Control Systems by Integrating Taylor Models and Zonotopes](http://arxiv.org/abs/2112.09197)


  We study the verification problem for closed-loop dynamical systems with
neural-network controllers (NNCS). This problem is commonly reduced to
computing the set of reachable states. When considering dynamical systems and
neural networks in isolation, there exist precise approaches for that task
based on set representations respectively called Taylor models and zonotopes.
However, the combination of these approaches to NNCS is non-trivial because,
when converting between the set representations, dependency information gets
lost in each control cycle and the accumulated approximation error quickly
renders the result useless. We present an algorithm to chain approaches based
on Taylor models and zonotopes, yielding a precise reachability algorithm for
NNCS. Because the algorithm only acts at the interface of the isolated
approaches, it is applicable to general dynamical systems and neural networks
and can benefit from future advances in these areas. Our implementation
delivers state-of-the-art performance and is the first to successfully analyze
all benchmark problems of an annual reachability competition for NNCS.

    

### [[2112.09201] Semantic-Based Few-Shot Learning by Interactive Psychometric Testing](http://arxiv.org/abs/2112.09201)


  Few-shot classification tasks aim to classify images in query sets based on
only a few labeled examples in support sets. Most studies usually assume that
each image in a task has a single and unique class association. Under these
assumptions, these algorithms may not be able to identify the proper class
assignment when there is no exact matching between support and query classes.
For example, given a few images of lions, bikes, and apples to classify a
tiger. However, in a more general setting, we could consider the higher-level
concept of large carnivores to match the tiger to the lion for semantic
classification. Existing studies rarely considered this situation due to the
incompatibility of label-based supervision with complex conception
relationships. In this work, we advanced the few-shot learning towards this
more challenging scenario, the semantic-based few-shot learning, and proposed a
method to address the paradigm by capturing the inner semantic relationships
using interactive psychometric learning. We evaluate our method on the
CIFAR-100 dataset. The results show the merits of our proposed method.

    

### [[2112.09215] Hyperbolic Disentangled Representation for Fine-Grained Aspect Extraction](http://arxiv.org/abs/2112.09215)


  Automatic identification of salient aspects from user reviews is especially
useful for opinion analysis. There has been significant progress in utilizing
weakly supervised approaches, which require only a small set of seed words for
training aspect classifiers. However, there is always room for improvement.
First, no weakly supervised approaches fully utilize latent hierarchies between
words. Second, each seed words representation should have different latent
semantics and be distinct when it represents a different aspect. In this paper,
we propose HDAE, a hyperbolic disentangled aspect extractor in which a
hyperbolic aspect classifier captures words latent hierarchies, and
aspect-disentangled representation models the distinct latent semantics of each
seed word. Compared to previous baselines, HDAE achieves average F1 performance
gains of 18.2% and 24.1% on Amazon product review and restaurant review
datasets, respectively. In addition, the em-bedding visualization experience
demonstrates that HDAE is a more effective approach to leveraging seed words.
An ablation study and a case study further attest to the effectiveness of the
proposed components

    

### [[2112.09288] Neural Architectures for Biological Inter-Sentence Relation Extraction](http://arxiv.org/abs/2112.09288)


  We introduce a family of deep-learning architectures for inter-sentence
relation extraction, i.e., relations where the participants are not necessarily
in the same sentence. We apply these architectures to an important use case in
the biomedical domain: assigning biological context to biochemical events. In
this work, biological context is defined as the type of biological system
within which the biochemical event is observed. The neural architectures encode
and aggregate multiple occurrences of the same candidate context mentions to
determine whether it is the correct context for a particular event mention. We
propose two broad types of architectures: the first type aggregates multiple
instances that correspond to the same candidate context with respect to event
mention before emitting a classification; the second type independently
classifies each instance and uses the results to vote for the final class, akin
to an ensemble approach. Our experiments show that the proposed neural
classifiers are competitive and some achieve better performance than previous
state of the art traditional machine learning methods without the need for
feature engineering. Our analysis shows that the neural methods particularly
improve precision compared to traditional machine learning classifiers and also
demonstrates how the difficulty of inter-sentence relation extraction increases
as the distance between the event and context mentions increase.

    

### [[2112.09301] Overview of the HASOC Subtrack at FIRE 2021: Hate Speech and Offensive Content Identification in English and Indo-Aryan Languages](http://arxiv.org/abs/2112.09301)


  The widespread of offensive content online such as hate speech poses a
growing societal problem. AI tools are necessary for supporting the moderation
process at online platforms. For the evaluation of these identification tools,
continuous experimentation with data sets in different languages are necessary.
The HASOC track (Hate Speech and Offensive Content Identification) is dedicated
to develop benchmark data for this purpose. This paper presents the HASOC
subtrack for English, Hindi, and Marathi. The data set was assembled from
Twitter. This subtrack has two sub-tasks. Task A is a binary classification
problem (Hate and Not Offensive) offered for all three languages. Task B is a
fine-grained classification problem for three classes (HATE) Hate speech,
OFFENSIVE and PROFANITY offered for English and Hindi. Overall, 652 runs were
submitted by 65 teams. The performance of the best classification algorithms
for task A are F1 measures 0.91, 0.78 and 0.83 for Marathi, Hindi and English,
respectively. This overview presents the tasks and the data development as well
as the detailed results. The systems submitted to the competition applied a
variety of technologies. The best performing algorithms were mainly variants of
transformer architectures.

    

### [[2112.09325] Dilemma of the Artificial Intelligence Regulatory Landscape](http://arxiv.org/abs/2112.09325)


  As a startup company in the autonomous driving space, we have undergone four
years of painful experiences dealing with a broad spectrum of regulatory
requirements. Compared to the software industry norm, which spends 13% of their
overall budget on compliances, we were forced to spend 42% of our budget on
compliances. Our situation is not alone and, in a way, reflects the dilemma of
the artificial intelligence (AI) regulatory landscape. The root cause is the
lack of AI expertise in the legislative and executive branches, leading to a
lack of standardization for the industry to follow. In this article, we share
our first-hand experiences and advocate for the establishment of an FDA-like
agency to regulate AI properly.

    

### [[2112.09385] Full Transformer Framework for Robust Point Cloud Registration with Deep Information Interaction](http://arxiv.org/abs/2112.09385)


  Recent Transformer-based methods have achieved advanced performance in point
cloud registration by utilizing advantages of the Transformer in
order-invariance and modeling dependency to aggregate information. However,
they still suffer from indistinct feature extraction, sensitivity to noise, and
outliers. The reasons are: (1) the adoption of CNNs fails to model global
relations due to their local receptive fields, resulting in extracted features
susceptible to noise; (2) the shallow-wide architecture of Transformers and
lack of positional encoding lead to indistinct feature extraction due to
inefficient information interaction; (3) the omission of geometrical
compatibility leads to inaccurate classification between inliers and outliers.
To address above limitations, a novel full Transformer network for point cloud
registration is proposed, named the Deep Interaction Transformer (DIT), which
incorporates: (1) a Point Cloud Structure Extractor (PSE) to model global
relations and retrieve structural information with Transformer encoders; (2) a
deep-narrow Point Feature Transformer (PFT) to facilitate deep information
interaction across two point clouds with positional encoding, such that
Transformers can establish comprehensive associations and directly learn
relative position between points; (3) a Geometric Matching-based Correspondence
Confidence Evaluation (GMCCE) method to measure spatial consistency and
estimate inlier confidence by designing the triangulated descriptor. Extensive
experiments on clean, noisy, partially overlapping point cloud registration
demonstrate that our method outperforms state-of-the-art methods.

    

### [[2112.09459] Weakly Supervised Semantic Segmentation via Alternative Self-Dual Teaching](http://arxiv.org/abs/2112.09459)


  Current weakly supervised semantic segmentation (WSSS) frameworks usually
contain the separated mask-refinement model and the main semantic region mining
model. These approaches would contain redundant feature extraction backbones
and biased learning objectives, making them computational complex yet
sub-optimal to addressing the WSSS task. To solve this problem, this paper
establishes a compact learning framework that embeds the classification and
mask-refinement components into a unified deep model. With the shared feature
extraction backbone, our model is able to facilitate knowledge sharing between
the two components while preserving a low computational complexity. To
encourage high-quality knowledge interaction, we propose a novel alternative
self-dual teaching (ASDT) mechanism. Unlike the conventional distillation
strategy, the knowledge of the two teacher branches in our model is
alternatively distilled to the student branch by a Pulse Width Modulation
(PWM), which generates PW wave-like selection signal to guide the knowledge
distillation process. In this way, the student branch can help prevent the
model from falling into local minimum solutions caused by the imperfect
knowledge provided of either teacher branch. Comprehensive experiments on the
PASCAL VOC 2012 and COCO-Stuff 10K demonstrate the effectiveness of the
proposed alternative self-dual teaching mechanism as well as the new
state-of-the-art performance of our approach.

    

### [[2112.09462] Contrastive Explanations for Comparing Preferences of Reinforcement Learning Agents](http://arxiv.org/abs/2112.09462)


  In complex tasks where the reward function is not straightforward and
consists of a set of objectives, multiple reinforcement learning (RL) policies
that perform task adequately, but employ different strategies can be trained by
adjusting the impact of individual objectives on reward function. Understanding
the differences in strategies between policies is necessary to enable users to
choose between offered policies, and can help developers understand different
behaviors that emerge from various reward functions and training
hyperparameters in RL systems. In this work we compare behavior of two policies
trained on the same task, but with different preferences in objectives. We
propose a method for distinguishing between differences in behavior that stem
from different abilities from those that are a consequence of opposing
preferences of two RL agents. Furthermore, we use only data on preference-based
differences in order to generate contrasting explanations about agents'
preferences. Finally, we test and evaluate our approach on an autonomous
driving task and compare the behavior of a safety-oriented policy and one that
prefers speed.

    

### [[2112.09515] Symmetry-aware Neural Architecture for Embodied Visual Navigation](http://arxiv.org/abs/2112.09515)


  Visual exploration is a task that seeks to visit all the navigable areas of
an environment as quickly as possible. The existing methods employ deep
reinforcement learning (RL) as the standard tool for the task. However, they
tend to be vulnerable to statistical shifts between the training and test data,
resulting in poor generalization over novel environments that are
out-of-distribution (OOD) from the training data. In this paper, we attempt to
improve the generalization ability by utilizing the inductive biases available
for the task. Employing the active neural SLAM (ANS) that learns exploration
policies with the advantage actor-critic (A2C) method as the base framework, we
first point out that the mappings represented by the actor and the critic
should satisfy specific symmetries. We then propose a network design for the
actor and the critic to inherently attain these symmetries. Specifically, we
use $G$-convolution instead of the standard convolution and insert the
semi-global polar pooling (SGPP) layer, which we newly design in this study, in
the last section of the critic network. Experimental results show that our
method increases area coverage by $8.1 m^2$ when trained on the Gibson dataset
and tested on the MP3D dataset, establishing the new state-of-the-art.

    

### [[2112.09573] cgSpan: Closed Graph-Based Substructure Pattern Mining](http://arxiv.org/abs/2112.09573)


  gSpan is a popular algorithm for mining frequent subgraphs. cgSpan (closed
graph-based substructure pattern mining) is a gSpan extension that only mines
closed subgraphs. A subgraph g is closed in the graphs database if there is no
proper frequent supergraph of g that has equivalent occurrence with g. cgSpan
adds the Early Termination pruning method to the gSpan pruning methods, while
leaving the original gSpan steps unchanged. cgSpan also detects and handles
cases in which Early Termination should not be applied. To the best of our
knowledge, cgSpan is the first publicly available implementation for closed
graphs mining

    

### [[2112.09591] Global explainability in aligned image modalities](http://arxiv.org/abs/2112.09591)


  Deep learning (DL) models are very effective on many computer vision problems
and increasingly used in critical applications. They are also inherently black
box. A number of methods exist to generate image-wise explanations that allow
practitioners to understand and verify model predictions for a given image.
Beyond that, it would be desirable to validate that a DL model
\textit{generally} works in a sensible way, i.e. consistent with domain
knowledge and not relying on undesirable data artefacts. For this purpose, the
model needs to be explained globally. In this work, we focus on image
modalities that are naturally aligned such that each pixel position represents
a similar relative position on the imaged object, as is common in medical
imaging. We propose the pixel-wise aggregation of image-wise explanations as a
simple method to obtain label-wise and overall global explanations. These can
then be used for model validation, knowledge discovery, and as an efficient way
to communicate qualitative conclusions drawn from inspecting image-wise
explanations. We further propose Progressive Erasing Plus Progressive
Restoration (PEPPR) as a method to quantitatively validate that these global
explanations are faithful to how the model makes its predictions. We then apply
these methods to ultra-widefield retinal images, a naturally aligned modality.
We find that the global explanations are consistent with domain knowledge and
faithfully reflect the model's workings.

    

### [[2112.09616] Explanation as Question Answering based on Design Knowledge](http://arxiv.org/abs/2112.09616)


  Explanation of an AI agent requires knowledge of its design and operation. An
open question is how to identify, access and use this design knowledge for
generating explanations. Many AI agents used in practice, such as intelligent
tutoring systems fielded in educational contexts, typically come with a User
Guide that explains what the agent does, how it works and how to use the agent.
However, few humans actually read the User Guide in detail. Instead, most users
seek answers to their questions on demand. In this paper, we describe a
question answering agent (AskJill) that uses the User Guide for an interactive
learning environment (VERA) to automatically answer questions and thereby
explains the domain, functioning, and operation of VERA. We present a
preliminary assessment of AskJill in VERA.

    

### [[2008.09943] Quantum Language Model with Entanglement Embedding for Question Answering](http://arxiv.org/abs/2008.09943)


  Quantum Language Models (QLMs) in which words are modelled as quantum
superposition of sememes have demonstrated a high level of model transparency
and good post-hoc interpretability. Nevertheless, in the current literature
word sequences are basically modelled as a classical mixture of word states,
which cannot fully exploit the potential of a quantum probabilistic
description. A full quantum model is yet to be developed to explicitly capture
the non-classical correlations within the word sequences. We propose a neural
network model with a novel Entanglement Embedding (EE) module, whose function
is to transform the word sequences into entangled pure states of many-body
quantum systems. Strong quantum entanglement, which is the central concept of
quantum information and an indication of parallelized correlations among the
words, is observed within the word sequences. Numerical experiments show that
the proposed QLM with EE (QLM-EE) achieves superior performance compared with
the classical deep neural network models and other QLMs on Question Answering
(QA) datasets. In addition, the post-hoc interpretability of the model can be
improved by quantizing the degree of entanglement among the words.

    

### [[2104.07719] Meta Faster R-CNN: Towards Accurate Few-Shot Object Detection with Attentive Feature Alignment](http://arxiv.org/abs/2104.07719)


  Few-shot object detection (FSOD) aims to detect objects using only a few
examples. How to adapt state-of-the-art object detectors to the few-shot domain
remains challenging. Object proposal is a key ingredient in modern object
detectors. However, the quality of proposals generated for few-shot classes
using existing methods is far worse than that of many-shot classes, e.g.,
missing boxes for few-shot classes due to misclassification or inaccurate
spatial locations with respect to true objects. To address the noisy proposal
problem, we propose a novel meta-learning based FSOD model by jointly
optimizing the few-shot proposal generation and fine-grained few-shot proposal
classification. To improve proposal generation for few-shot classes, we propose
to learn a lightweight metric-learning based prototype matching network,
instead of the conventional simple linear object/nonobject classifier, e.g.,
used in RPN. Our non-linear classifier with the feature fusion network could
improve the discriminative prototype matching and the proposal recall for
few-shot classes. To improve the fine-grained few-shot proposal classification,
we propose a novel attentive feature alignment method to address the spatial
misalignment between the noisy proposals and few-shot classes, thus improving
the performance of few-shot object detection. Meanwhile we learn a separate
Faster R-CNN detection head for many-shot base classes and show strong
performance of maintaining base-classes knowledge. Our model achieves
state-of-the-art performance on multiple FSOD benchmarks over most of the shots
and metrics.

    

### [[2101.08939] A Rich Type System for Quantum Programs](http://arxiv.org/abs/2101.08939)


  We show that Gottesman's semantics (GROUP22, 1998) for Clifford circuits
based on the Heisenberg representation can be treated as a type system that can
efficiently characterize a common subset of quantum programs. Our applications
include (i) certifying whether auxiliary qubits can be safely disposed of, (ii)
determining if a system is separable across a given bi-partition, (iii)
checking the transversality of a gate with respect to a given stabilizer code,
and (iv) typing post-measurement states for computational basis measurements.
Further, this type system is extended to accommodate universal quantum
computing by deriving types for the $T$-gate, multiply-controlled unitaries
such as the Toffoli gate, and some gate injection circuits that use associated
magic states. These types allow us to prove a lower bound on the number of $T$
gates necessary to perform a multiply-controlled $Z$ gate.

    

### [<title>Save model in python, and use it in spark - XGBoost</title>](https://discuss.xgboost.ai/t/save-model-in-python-and-use-it-in-spark/2611/1)