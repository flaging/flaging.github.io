
## 2021-12-15

### [[2112.07048] Placement and Allocation of Communications Resources in Slicing-aware Flying Networks](http://arxiv.org/abs/2112.07048)


  Network slicing emerged in 5G networks as a key component to enable the use
of multiple services with different performance requirements on top of a shared
physical network infrastructure. A major challenge lies on ensuring wireless
coverage and enough communications resources to meet the target Quality of
Service (QoS) levels demanded by these services, including throughput and delay
guarantees. The challenge is exacerbated in temporary events, such as disaster
management scenarios and outdoor festivities, where the existing wireless
infrastructures may collapse, fail to provide sufficient wireless coverage, or
lack the required communications resources. Flying networks, composed of
Unmanned Aerial Vehicles (UAVs), emerged as a solution to provide on-demand
wireless coverage and communications resources anywhere, anytime. However,
existing solutions mostly rely on best-effort networks. The main contribution
of this paper is SLICER, an algorithm enabling the placement and allocation of
communications resources in slicing-aware flying networks. The evaluation
carried out by means of ns-3 simulations shows SLICER can meet the targeted QoS
levels, while using the minimum amount of communications resources.

    

### [[2112.07092] A Quantum Internet Architecture](http://arxiv.org/abs/2112.07092)


  Entangled quantum communication is advancing rapidly, with laboratory and
metropolitan testbeds under development, but to date there is no unifying
Quantum Internet architecture. We propose a Quantum Internet architecture
centered around the Quantum Recursive Network Architecture (QRNA), using
RuleSet-based connections established using a two-pass connection setup.
Scalability and internetworking (for both technological and administrative
boundaries) are achieved using recursion in naming and connection control. In
the near term, this architecture will support end-to-end, two-party
entanglement on minimal hardware, and it will extend smoothly to multi-party
entanglement and the use of quantum error correction on advanced hardware in
the future. For a network internal gateway protocol, we recommend (but do not
require) qDijkstra with seconds per Bell pair as link cost for routing; the
external gateway protocol is designed to build recursively. The strength of our
architecture is shown by assessing extensibility and demonstrating how robust
protocol operation can be confirmed using the RuleSet paradigm.

    

### [[2112.07093] QuISP: a Quantum Internet Simulation Package](http://arxiv.org/abs/2112.07093)


  We present an event-driven simulation package called QuISP for large-scale
quantum networks built on top of the OMNeT++ discrete event simulation
framework. Although the behavior of quantum networking devices have been
revealed by recent research, it is still an open question how they will work in
networks of a practical size. QuISP is designed to simulate large-scale quantum
networks to investigate their behavior under realistic, noisy and heterogeneous
configurations. The protocol architecture we propose enables studies of
different choices for error management and other key decisions. Our confidence
in the simulator is supported by comparing its output to analytic results for a
small network. A key reason for simulation is to look for emergent behavior
when large numbers of individually characterized devices are combined. QuISP
can handle thousands of qubits in dozens of nodes on a laptop computer,
preparing for full Quantum Internet simulation. This simulator promotes the
development of protocols for larger and more complex quantum networks.

    

### [[2112.07115] Dynamic Coherence-Based EM Ray Tracing Simulations in Vehicular Environments](http://arxiv.org/abs/2112.07115)


  5G applications have become increasingly popular in recent years as the
spread of 5G network deployment has grown. For vehicular networks, mmWave band
signals have been well studied and used for communication and sensing. In this
work, we propose a new dynamic ray tracing algorithm that exploits spatial and
temporal coherence. We evaluate the performance by comparing the results on
typical vehicular communication scenarios with NYUSIM, which builds on
stochastic models, and Winprop, which utilizes the deterministic model for
simulations with given environment information. We compare the performance of
our algorithm on complex, urban models and observe the reduction in computation
time by 60% compared to NYUSIM and 30% compared to Winprop, while maintaining
similar prediction accuracy.

    

### [[2112.07164] Contention Based Proportional Fairness (CBPF) Transmission Scheme for Time Slotted Channel Hopping Networks](http://arxiv.org/abs/2112.07164)


  Time Slotted Channel Hopping (TSCH) is a Medium Access Control (MAC) protocol
introduced in IEEE802.15.4e standard, addressing low power requirements of the
Internet of Things (IoT) and Low Power Lossy Networks (LLNs). The 6TiSCH
Operation sublayer (6top) of IEEE802.15.4e defines the schedule that includes
sleep, transmit and receive routines of the nodes. However, the design of
schedule is not specified by the standard. In this paper, we propose a
contention based proportional fairness (CBPF) transmission scheme for TSCH
networks to maximize the system throughput addressing fair allocation of
resources to the nodes. We propose a convex programming based method to achieve
the fairness and throughput objectives. We model TSCH MAC as a multichannel
slotted aloha and analyse it for a schedule given by the 6top layer.
Performance metrics like throughput, delay and energy spent per successful
transmission are derived and validated through simulations. The proposed CBPF
transmission scheme has been implemented in the IoT-LAB public testbed to
evaluate its performance and to compare with the existing scheduling
algorithms.

    

### [[2112.07170] Performance evaluation of the QOS provisioning ability of IEEE 802.11e WLAN standard for multimedia traffic](http://arxiv.org/abs/2112.07170)


  This paper presents an analytical model for the average frame transmission
delay and the jitter for the different Access Categories (ACs) of the IEEE
802.11e Enhanced Distributed Channel Access (EDCA) mechanism. Following are the
salient features of our model. As defined by the standard we consider (1) the
virtual collisions among different ACs inside each EDCA station in addition to
external collisions. (2) the effect of priority parameters, such as minimum and
maximum values of Contention Window (CW) sizes, Arbitration Inter Frame Space
(AIFS). (3) the role of Transmission Opportunity (TXOP) of different ACs. (4)
the finite number of retrials a packet experiences before being dropped. Our
model and analytical results provide an in-depth understanding of the EDCA
mechanism and the effect of Quality of Service (QoS) parameters in the
performance of IEEE 802.11e protocol.

    

### [[2112.07185] Towards End-to-End Error Management for a Quantum Internet](http://arxiv.org/abs/2112.07185)


  Error management in the quantum Internet requires stateful and stochastic
processing across multiple nodes, which is a significant burden. In view of the
history of the current Internet, the end-to-end principle was devised for error
management, simplifying the work inside the network and contributing
significantly to the scalability of the Internet. In this paper, we propose to
bring the end-to-end principle into the error management of quantum Internet to
improve the communication resource utilization efficiency of a quantum
Internet. The simulation results show that the error management using the
end-to-end principle and locality can be more resource-efficient than other
settings. In addition, when end-to-end error management is used, if the error
probability of qubits in the end node is sufficiently low, there is no problem
even if the error probability on the network side is higher than that in the
end node, and the load on the network can be reduced. Our proposal will
contribute to improving the communication capacity and scalability of the
quantum Internet, as well as to improve the interoperability of quantum
Autonomous Systems. In addition, existing studies on routing and other aspects
of the quantum Internet may exclude error management from their scope due to
its complexity. The results of this study provide validity to the assumptions
of such studies.

    

### [[2112.07320] Sherman: A Write-Optimized Distributed B+Tree Index on Disaggregated Memory](http://arxiv.org/abs/2112.07320)


  Memory disaggregation architecture physically separates CPU and memory into
independent components, which are connected via high-speed RDMA networks,
greatly improving resource utilization of databases. However, such an
architecture poses unique challenges to data indexing in databases due to
limited RDMA semantics and near-zero computation power at memory-side. Existing
indexes supporting disaggregated memory either suffer from low write
performance, or require hardware modification.
This paper presents Sherman, a write-optimized distributed B+Tree index on
disaggregated memory that delivers high performance with commodity RDMA NICs.
Sherman combines RDMA hardware features and RDMA-friendly software techniques
to boost index write performance from three angles. First, to reduce round
trips, Sherman coalesces dependent RDMA commands by leveraging in-order
delivery property of RDMA. Second, to accelerate concurrent accesses, Sherman
introduces a hierarchical lock that exploits on-chip memory of RDMA NICs.
Finally, to mitigate write amplification, Sherman tailors the data structure
layout of B+Tree with a two-level version mechanism. Our evaluation shows that,
Sherman is one order of magnitude faster in terms of both throughput and 99th
percentile latency on typical write-intensive workloads, compared with
state-of-the-art designs.

    

### [[2112.07339] Speeding up enclave transitions for IO-intensive applications](http://arxiv.org/abs/2112.07339)


  Process-based confidential computing enclaves such as Intel SGX can be used
to protect the confidentiality and integrity of workloads, without the overhead
of virtualisation. However, they introduce a notable performance overhead,
especially when it comes to transitions in and out of the enclave context. Such
overhead makes the use of enclaves impractical for running IO-intensive
applications, such as network packet processing or biological sequence
analysis. We build on earlier approaches to improve the IO performance of
work-loads in Intel SGX enclaves and propose the SGX-Bundler library, which
helps reduce the cost of both individual single enclave transitions well as of
the total number of enclave transitions in trusted applications running in
Intel SGX enclaves. We describe the implementation of the SGX-Bundler library,
evaluate its performance and demonstrate its practicality using the case study
of Open vSwitch, a widely used software switch implementation.

    

### [[2112.07363] An Advanced Parallel PageRank Algorithm](http://arxiv.org/abs/2112.07363)


  Initially used to rank web pages, PageRank has now been applied in many
fields. In general case, there are plenty of special vertices such as dangling
vertices and unreferenced vertices in the graph. Existing PageRank algorithms
usually consider them as `bad` vertices since they may take troubles. However,
in this paper, we propose a parallel PageRank algorithm which can take
advantage of these special vertices. For this end, we firstly interpret
PageRank from the information transmitting perspective and give a constructive
definition of PageRank. Then, based on the information transmitting
interpretation, a parallel PageRank algorithm which we call the Information
Transmitting Algorithm(ITA) is proposed. We prove that the dangling vertices
can increase ITA's convergence rate and the unreferenced vertices and weak
unreferenced vertices can decrease ITA's calculations. Compared with the MONTE
CARLO method, ITA has lower bandwidth requirement. Compared with the power
method, ITA has higher convergence rate and generates less calculations.
Finally, experimental results on four data sets demonstrate that ITA is 1.5-4
times faster than the power method and converges more uniformly.

    

### [[2112.07586] Real-time SIL Emulation Architecture for Cooperative Automated Vehicles](http://arxiv.org/abs/2112.07586)


  The development of safety applications for Connected Automated Vehicles
requires testing in many different scenarios. However, the recreation of test
scenarios for evaluating safety applications is a very challenging task. This
is mainly due to the randomness in communication, difficulty in recreating
vehicle movements precisely, and safety concerns for certain scenarios. We
propose to develop a standalone Remote Vehicle Emulator that can reproduce V2V
messages of remote vehicles from simulations or previous tests. This is
expected to accelerate the development cycle significantly. Remote Vehicle
Emulator is a unique and easily configurable emulation cum simulation setup to
allow Software in the Loop (SIL) testing of connected vehicle applications
realistically and safely. It will help in tailoring numerous test scenarios,
expediting algorithm development and validation, and increasing the probability
of finding failure modes. This, in turn, will help improve the quality of
safety applications while saving testing time and reducing cost.

    

### [[2112.07663] Learning Connectivity-Maximizing Network Configurations](http://arxiv.org/abs/2112.07663)


  In this work we propose a data-driven approach to optimizing the algebraic
connectivity of a team of robots. While a considerable amount of research has
been devoted to this problem, we lack a method that scales in a manner suitable
for online applications for more than a handful of agents. To that end, we
propose a supervised learning approach with a convolutional neural network
(CNN) that learns to place communication agents from an expert that uses an
optimization-based strategy. We demonstrate the performance of our CNN on
canonical line and ring topologies, 105k randomly generated test cases, and
larger teams not seen during training. We also show how our system can be
applied to dynamic robot teams through a Unity-based simulation. After
training, our system produces connected configurations 2 orders of magnitude
faster than the optimization-based scheme for teams of 10-20 agents.

    

### [[2007.12284] Energy-aware Relay Positioning in Flying Networks](http://arxiv.org/abs/2007.12284)


  The ability to move and hover has made rotary-wing Unmanned Aerial Vehicles
(UAVs) suitable platforms to act as Flying Communications Relays (FCR), aiming
at providing on-demand, temporary wireless connectivity when there is no
network infrastructure available or a need to reinforce the capacity of
existing networks. However, since UAVs rely on their on-board batteries, which
can be drained quickly, they typically need to land frequently for recharging
or replacing them, limiting their endurance and the flying network
availability. The problem is exacerbated when a single FCR UAV is used. The FCR
UAV energy is used for two main tasks: communications and propulsion. The
literature has been focused on optimizing both the flying network performance
and energy-efficiency from the communications point of view, overlooking the
energy spent for the UAV propulsion. Yet, the energy spent for communications
is typically negligible when compared with the energy spent for the UAV
propulsion.
In this article we propose Energy-aware RElay Positioning (EREP), an
algorithm for positioning the FCR taking into account the energy spent for the
UAV propulsion. Building upon the conclusion that hovering is not the most
energy-efficient state, EREP defines the trajectory and speed that minimize the
energy spent by the FCR UAV on propulsion, without compromising the Quality of
Service offered by the flying network. The EREP algorithm is evaluated using
simulations. The obtained results show significant gains in the FCR UAV
endurance.

    

### [[2106.02156] Trading Throughput for Freshness: Freshness-Aware Traffic Engineering and In-Network Freshness Control](http://arxiv.org/abs/2106.02156)


  In addition to traditional concerns such as throughput and latency, freshness
is becoming increasingly important. To stay fresh, applications stream status
updates among their components. Existing studies propose the metric age of
information (AoI) to gauge the freshness and design systems to achieve low AoI.
Despite active research in this area, existing results are not applicable to
general wired networks for two reasons. First, they focus on wireless settings
where AoI is mostly affected by interference and collision while queueing is
more dominant in wired settings. Second, the legacy drop-adverse flows are not
taken into account in the literature. Scheduling mixed flows with distinct
performance objective is not yet addressed.
In this paper, we study wired networks shared by two classes of flows, aiming
for high throughput and low AoI respectively, and achieve a good trade-off
between their throughput and AoI. Our approach to the problem consists of two
layers: freshness-aware traffic engineering (FATE) and in-network freshness
control (IFC). FATE derives sending rate/update frequency for flows via
optimization, and its solution is then enforced by IFC through efficient
scheduling mechanisms at each outport of in-network nodes. We also present
efficient Linux implementation of IFC and demonstrate the effectiveness of
FATE/IFC through extensive emulations. Our results show that it is possible to
trade a little throughput (5 % lower) for much shorter AoI (49 to 71% shorter)
compared to state-of-the-art traffic engineering.

    

### [[2112.06918] Automated Customization of On-Thing Inference for Quality-of-Experience Enhancement](http://arxiv.org/abs/2112.06918)


  The rapid uptake of intelligent applications is pushing deep learning (DL)
capabilities to Internet-of-Things (IoT). Despite the emergence of new tools
for embedding deep neural networks (DNNs) into IoT devices, providing
satisfactory Quality of Experience (QoE) to users is still challenging due to
the heterogeneity in DNN architectures, IoT devices, and user preferences. This
paper studies automated customization for DL inference on IoT devices (termed
as on-thing inference), and our goal is to enhance user QoE by configuring the
on-thing inference with an appropriate DNN for users under different usage
scenarios. The core of our method is a DNN selection module that learns user
QoE patterns on-the-fly and identifies the best-fit DNN for on-thing inference
with the learned knowledge. It leverages a novel online learning algorithm,
NeuralUCB, that has excellent generalization ability for handling various user
QoE patterns. We also embed the knowledge transfer technique in NeuralUCB to
expedite the learning process. However, NeuralUCB frequently solicits QoE
ratings from users, which incurs non-negligible inconvenience. To address this
problem, we design feedback solicitation schemes to reduce the number of QoE
solicitations while maintaining the learning efficiency of NeuralUCB. A
pragmatic problem, aggregated QoE, is further investigated to improve the
practicality of our framework. We conduct experiments on both synthetic and
real-world data. The results indicate that our method efficiently learns the
user QoE pattern with few solicitations and provides drastic QoE enhancement
for IoT devices.

    

### [[2112.06920] Boosting Independent Component Analysis](http://arxiv.org/abs/2112.06920)


  Independent component analysis is intended to recover the unknown components
as independent as possible from their linear mixtures. This technique has been
widely used in many fields, such as data analysis, signal processing, and
machine learning. In this paper, we present a novel boosting-based algorithm
for independent component analysis. Our algorithm fills the gap in the
nonparametric independent component analysis by introducing boosting to maximum
likelihood estimation. A variety of experiments validate its performance
compared with many of the presently known algorithms.

    

### [[2112.06924] Generating Fluent Fact Checking Explanations with Unsupervised Post-Editing](http://arxiv.org/abs/2112.06924)


  Fact-checking systems have become important tools to verify fake and
misguiding news. These systems become more trustworthy when human-readable
explanations accompany the veracity labels. However, manual collection of such
explanations is expensive and time-consuming. Recent works frame explanation
generation as extractive summarization, and propose to automatically select a
sufficient subset of the most important facts from the ruling comments (RCs) of
a professional journalist to obtain fact-checking explanations. However, these
explanations lack fluency and sentence coherence. In this work, we present an
iterative edit-based algorithm that uses only phrase-level edits to perform
unsupervised post-editing of disconnected RCs. To regulate our editing
algorithm, we use a scoring function with components including fluency and
semantic preservation. In addition, we show the applicability of our approach
in a completely unsupervised setting. We experiment with two benchmark
datasets, LIAR-PLUS and PubHealth. We show that our model generates
explanations that are fluent, readable, non-redundant, and cover important
information for the fact check.

    

### [[2112.06925] CGAN-EB: A Non-parametric Empirical Bayes Method for Crash Hotspot Identification Using Conditional Generative Adversarial Networks: A Simulated Crash Data Study](http://arxiv.org/abs/2112.06925)


  In this paper, a new non-parametric empirical Bayes approach called CGAN-EB
is proposed for approximating empirical Bayes (EB) estimates in traffic
locations (e.g., road segments) which benefits from the modeling advantages of
deep neural networks, and its performance is compared in a simulation study
with the traditional approach based on negative binomial model (NB-EB). The
NB-EB uses negative binomial model in order to model the crash data and is the
most common approach in practice. To model the crash data in the proposed
CGAN-EB, conditional generative adversarial network is used, which is a
powerful deep neural network based method that can model any types of
distributions. A number of simulation experiments are designed and conducted to
evaluate the CGAN-EB performance in different conditions and compare it with
the NB-EB. The results show that CGAN-EB performs as well as NB-EB when
conditions favor the NB-EB model (i.e. data conform to the assumptions of the
NB model) and outperforms NB-EB in experiments reflecting conditions frequently
encountered in practice, specifically low sample means, and when crash
frequency does not follow a log-linear relationship with covariates.

    

### [[2112.06926] Addressing Bias in Active Learning with Depth Uncertainty Networks... or Not](http://arxiv.org/abs/2112.06926)


  Farquhar et al. [2021] show that correcting for active learning bias with
underparameterised models leads to improved downstream performance. For
overparameterised models such as NNs, however, correction leads either to
decreased or unchanged performance. They suggest that this is due to an
"overfitting bias" which offsets the active learning bias. We show that depth
uncertainty networks operate in a low overfitting regime, much like
underparameterised models. They should therefore see an increase in performance
with bias correction. Surprisingly, they do not. We propose that this negative
result, as well as the results Farquhar et al. [2021], can be explained via the
lens of the bias-variance decomposition of generalisation error.

    

### [[2112.06953] Controlled Cue Generation for Play Scripts](http://arxiv.org/abs/2112.06953)


  In this paper, we use a large-scale play scripts dataset to propose the novel
task of theatrical cue generation from dialogues. Using over one million lines
of dialogue and cues, we approach the problem of cue generation as a controlled
text generation task, and show how cues can be used to enhance the impact of
dialogue using a language model conditioned on a dialogue/cue discriminator. In
addition, we explore the use of topic keywords and emotions for controlled text
generation. Extensive quantitative and qualitative experiments show that
language models can be successfully used to generate plausible and
attribute-controlled texts in highly specialised domains such as play scripts.
Supporting materials can be found at: this https URL.

    

### [[2112.06978] Exploring Latent Dimensions of Crowd-sourced Creativity](http://arxiv.org/abs/2112.06978)


  Recently, the discovery of interpretable directions in the latent spaces of
pre-trained GANs has become a popular topic. While existing works mostly
consider directions for semantic image manipulations, we focus on an abstract
property: creativity. Can we manipulate an image to be more or less creative?
We build our work on the largest AI-based creativity platform, Artbreeder,
where users can generate images using pre-trained GAN models. We explore the
latent dimensions of images generated on this platform and present a novel
framework for manipulating images to make them more creative. Our code and
dataset are available at this http URL.

    

### [[2112.06986] On The Reliability Of Machine Learning Applications In Manufacturing Environments](http://arxiv.org/abs/2112.06986)


  The increasing deployment of advanced digital technologies such as Internet
of Things (IoT) devices and Cyber-Physical Systems (CPS) in industrial
environments is enabling the productive use of machine learning (ML) algorithms
in the manufacturing domain. As ML applications transcend from research to
productive use in real-world industrial environments, the question of
reliability arises. Since the majority of ML models are trained and evaluated
on static datasets, continuous online monitoring of their performance is
required to build reliable systems. Furthermore, concept and sensor drift can
lead to degrading accuracy of the algorithm over time, thus compromising
safety, acceptance and economics if undetected and not properly addressed. In
this work, we exemplarily highlight the severity of the issue on a publicly
available industrial dataset which was recorded over the course of 36 months
and explain possible sources of drift. We assess the robustness of ML
algorithms commonly used in manufacturing and show, that the accuracy strongly
declines with increasing drift for all tested algorithms. We further
investigate how uncertainty estimation may be leveraged for online performance
estimation as well as drift detection as a first step towards continually
learning applications. The results indicate, that ensemble algorithms like
random forests show the least decay of confidence calibration under drift.

    

### [[2112.06989] Analyzing a Caching Model](http://arxiv.org/abs/2112.06989)


  Machine Learning has been successfully applied in systems applications such
as memory prefetching and caching, where learned models have been shown to
outperform heuristics. However, the lack of understanding the inner workings of
these models -- interpretability -- remains a major obstacle for adoption in
real-world deployments. Understanding a model's behavior can help system
administrators and developers gain confidence in the model, understand risks,
and debug unexpected behavior in production. Interpretability for models used
in computer systems poses a particular challenge: Unlike ML models trained on
images or text, the input domain (e.g., memory access patterns, program
counters) is not immediately interpretable. A major challenge is therefore to
explain the model in terms of concepts that are approachable to a human
practitioner. By analyzing a state-of-the-art caching model, we provide
evidence that the model has learned concepts beyond simple statistics that can
be leveraged for explanations. Our work provides a first step towards
explanability of system ML models and highlights both promises and challenges
of this emerging research area.

    

### [[2112.06997] ELF: Exact-Lipschitz Based Universal Density Approximator Flow](http://arxiv.org/abs/2112.06997)


  Normalizing flows have grown more popular over the last few years; however,
they continue to be computationally expensive, making them difficult to be
accepted into the broader machine learning community. In this paper, we
introduce a simple one-dimensional one-layer network that has closed form
Lipschitz constants; using this, we introduce a new Exact-Lipschitz Flow (ELF)
that combines the ease of sampling from residual flows with the strong
performance of autoregressive flows. Further, we show that ELF is provably a
universal density approximator, more computationally and parameter efficient
compared to a multitude of other flows, and achieves state-of-the-art
performance on multiple large-scale datasets.

    

### [[2112.06999] Designing weighted and multiplex networks for deep learning user geolocation in Twitter](http://arxiv.org/abs/2112.06999)


  Predicting the geographical location of users of social media like Twitter
has found several applications in health surveillance, emergency monitoring,
content personalization, and social studies in general. In this work we
contribute to the research in this area by designing and evaluating new methods
based on the literature of weighted multigraphs combined with state-of-the-art
deep learning techniques. The explored methods depart from a similar underlying
structure (that of an extended mention and/or follower network) but use
different information processing strategies, e.g., information diffusion
through transductive and inductive algorithms -- RGCNs and GraphSAGE,
respectively -- and node embeddings with Node2vec+. These graphs are then
combined with attention mechanisms to incorporate the users' text view into the
models. We assess the performance of each of these methods and compare them to
baseline models in the publicly available Twitter-US dataset; we also make a
new dataset available based on a large Twitter capture in Latin America.
Finally, our work discusses the limitations and validity of the comparisons
among methods in the context of different label definitions and metrics.

    

### [[2112.07007] Acceleration techniques for optimization over trained neural network ensembles](http://arxiv.org/abs/2112.07007)


  We study optimization problems where the objective function is modeled
through feedforward neural networks with rectified linear unit (ReLU)
activation. Recent literature has explored the use of a single neural network
to model either uncertain or complex elements within an objective function.
However, it is well known that ensembles of neural networks produce more stable
predictions and have better generalizability than models with single neural
networks, which suggests the application of ensembles of neural networks in a
decision-making pipeline. We study how to incorporate a neural network ensemble
as the objective function of an optimization model and explore computational
approaches for the ensuing problem. We present a mixed-integer linear program
based on existing popular big-$M$ formulations for optimizing over a single
neural network. We develop two acceleration techniques for our model, the first
one is a preprocessing procedure to tighten bounds for critical neurons in the
neural network while the second one is a set of valid inequalities based on
Benders decomposition. Experimental evaluations of our solution methods are
conducted on one global optimization problem and two real-world data sets; the
results suggest that our optimization algorithm outperforms the adaption of an
state-of-the-art approach in terms of computational time and optimality gaps.

    

### [[2112.07013] PantheonRL: A MARL Library for Dynamic Training Interactions](http://arxiv.org/abs/2112.07013)


  We present PantheonRL, a multiagent reinforcement learning software package
for dynamic training interactions such as round-robin, adaptive, and ad-hoc
training. Our package is designed around flexible agent objects that can be
easily configured to support different training interactions, and handles fully
general multiagent environments with mixed rewards and n agents. Built on top
of StableBaselines3, our package works directly with existing powerful deep RL
algorithms. Finally, PantheonRL comes with an intuitive yet functional web user
interface for configuring experiments and launching multiple asynchronous jobs.
Our package can be found at this https URL.

    

### [[2112.07022] Learning Body-Aware 3D Shape Generative Models](http://arxiv.org/abs/2112.07022)


  The shape of many objects in the built environment is dictated by their
relationships to the human body: how will a person interact with this object?
Existing data-driven generative models of 3D shapes produce plausible objects
but do not reason about the relationship of those objects to the human body. In
this paper, we learn body-aware generative models of 3D shapes. Specifically,
we train generative models of chairs, an ubiquitous shape category, which can
be conditioned on a given body shape or sitting pose. The
body-shape-conditioned models produce chairs which will be comfortable for a
person with the given body shape; the pose-conditioned models produce chairs
which accommodate the given sitting pose. To train these models, we define a
"sitting pose matching" metric and a novel "sitting comfort" metric.
Calculating these metrics requires an expensive optimization to sit the body
into the chair, which is too slow to be used as a loss function for training a
generative model. Thus, we train neural networks to efficiently approximate
these metrics. We use our approach to train three body-aware generative shape
models: a structured part-based generator, a point cloud generator, and an
implicit surface generator. In all cases, our approach produces models which
adapt their output chair shapes to input human body specifications.

    

### [[2112.07031] Teaching a Robot to Walk Using Reinforcement Learning](http://arxiv.org/abs/2112.07031)


  Classical control techniques such as PID and LQR have been used effectively
in maintaining a system state, but these techniques become more difficult to
implement when the model dynamics increase in complexity and sensitivity. For
adaptive robotic locomotion tasks with several degrees of freedom, this task
becomes infeasible with classical control techniques. Instead, reinforcement
learning can train optimal walking policies with ease. We apply deep Q-learning
and augmented random search (ARS) to teach a simulated two-dimensional bipedal
robot how to walk using the OpenAI Gym BipedalWalker-v3 environment. Deep
Q-learning did not yield a high reward policy, often prematurely converging to
suboptimal local maxima likely due to the coarsely discretized action space.
ARS, however, resulted in a better trained robot, and produced an optimal
policy which officially "solves" the BipedalWalker-v3 problem. Various naive
policies, including a random policy, a manually encoded inch forward policy,
and a stay still policy, were used as benchmarks to evaluate the proficiency of
the learning algorithm results.

    

### [[2112.07041] Survey of Generative Methods for Social Media Analysis](http://arxiv.org/abs/2112.07041)


  This survey draws a broad-stroke, panoramic picture of the State of the Art
(SoTA) of the research in generative methods for the analysis of social media
data. It fills a void, as the existing survey articles are either much narrower
in their scope or are dated. We included two important aspects that currently
gain importance in mining and modeling social media: dynamics and networks.
Social dynamics are important for understanding the spreading of influence or
diseases, formation of friendships, the productivity of teams, etc. Networks,
on the other hand, may capture various complex relationships providing
additional insight and identifying important patterns that would otherwise go
unnoticed.

    

### [[2112.07042] How to Learn when Data Gradually Reacts to Your Model](http://arxiv.org/abs/2112.07042)


  A recent line of work has focused on training machine learning (ML) models in
the performative setting, i.e. when the data distribution reacts to the
deployed model. The goal in this setting is to learn a model which both induces
a favorable data distribution and performs well on the induced distribution,
thereby minimizing the test loss. Previous work on finding an optimal model
assumes that the data distribution immediately adapts to the deployed model. In
practice, however, this may not be the case, as the population may take time to
adapt to the model. In many applications, the data distribution depends on both
the currently deployed ML model and on the "state" that the population was in
before the model was deployed. In this work, we propose a new algorithm,
Stateful Performative Gradient Descent (Stateful PerfGD), for minimizing the
performative loss even in the presence of these effects. We provide theoretical
guarantees for the convergence of Stateful PerfGD. Our experiments confirm that
Stateful PerfGD substantially outperforms previous state-of-the-art methods.

    

### [[2112.07054] Graph network for simultaneous learning of forward and inverse physics](http://arxiv.org/abs/2112.07054)


  In this work, we propose an end-to-end graph network that learns forward and
inverse models of particle-based physics using interpretable inductive biases.
Physics-informed neural networks are often engineered to solve specific
problems through problem-specific regularization and loss functions. Such
explicit learning biases the network to learn data specific patterns and may
require a change in the loss function or neural network architecture hereby
limiting their generalizabiliy. While recent studies have proposed graph
networks to study forward dynamics, they rely on particle specific parameters
such as mass, etc. to approximate the dynamics of the system. Our graph network
is implicitly biased by learning to solve several tasks, thereby sharing
representations between tasks in order to learn the forward dynamics as well as
infer the probability distribution of unknown particle specific properties. We
evaluate our approach on one-step next state prediction tasks across diverse
datasets that feature different particle interactions. Our comparison against
related data-driven physics learning approaches reveals that our model is able
to predict the forward dynamics with at least an order of magnitude higher
accuracy. We also show that our approach is able to recover multi-modal
probability distributions of unknown physical parameters using orders of
magnitude fewer samples.

    

### [[2112.07055] Language Models are not Models of Language](http://arxiv.org/abs/2112.07055)


  Natural Language Processing (NLP) has become one of the leading application
areas in the current Artificial Intelligence boom. Transfer learning has
enabled large deep learning neural networks trained on the language modeling
task to vastly improve performance in almost all language tasks. Interestingly,
when the models are trained with data that includes software code, they
demonstrate remarkable abilities in generating functioning computer code from
natural language specifications. We argue that this creates a conundrum for
claims that neural models provide an alternative theory to generative phrase
structure grammars in explaining how language works. Since the syntax of
programming languages is determined by phrase structure grammars, successful
neural models are apparently uninformative about the theoretical foundations of
programming languages, and by extension, natural languages. We argue that the
term language model is misleading because deep learning models are not
theoretical models of language and propose the adoption of corpus model
instead, which better reflects the genesis and contents of the model.

    

### [[2112.07057] NEORL: NeuroEvolution Optimization with Reinforcement Learning](http://arxiv.org/abs/2112.07057)


  We present an open-source Python framework for NeuroEvolution Optimization
with Reinforcement Learning (NEORL) developed at the Massachusetts Institute of
Technology. NEORL offers a global optimization interface of state-of-the-art
algorithms in the field of evolutionary computation, neural networks through
reinforcement learning, and hybrid neuroevolution algorithms. NEORL features
diverse set of algorithms, user-friendly interface, parallel computing support,
automatic hyperparameter tuning, detailed documentation, and demonstration of
applications in mathematical and real-world engineering optimization. NEORL
encompasses various optimization problems from combinatorial, continuous, mixed
discrete/continuous, to high-dimensional, expensive, and constrained
engineering optimization. NEORL is tested in variety of engineering
applications relevant to low carbon energy research in addressing solutions to
climate change. The examples include nuclear reactor control and fuel cell
power production. The results demonstrate NEORL competitiveness against other
algorithms and optimization frameworks in the literature, and a potential tool
to solve large-scale optimization problems. More examples and benchmarking of
NEORL can be found here: this https URL


### [[2112.07066] Continual Learning In Environments With Polynomial Mixing Times](http://arxiv.org/abs/2112.07066)


  The mixing time of the Markov chain induced by a policy limits performance in
real-world continual learning scenarios. Yet, the effect of mixing times on
learning in continual reinforcement learning (RL) remains underexplored. In
this paper, we characterize problems that are of long-term interest to the
development of continual RL, which we call scalable MDPs, through the lens of
mixing times. In particular, we establish that scalable MDPs have mixing times
that scale polynomially with the size of the problem. We go on to demonstrate
that polynomial mixing times present significant difficulties for existing
approaches and propose a family of model-based algorithms that speed up
learning by directly optimizing for the average reward through a novel
bootstrapping procedure. Finally, we perform empirical regret analysis of our
proposed approaches, demonstrating clear improvements over baselines and also
how scalable MDPs can be used for analysis of RL algorithms as mixing times
scale.

    

### [[2112.07067] Dynamic Learning of Correlation Potentials for a Time-Dependent Kohn-Sham System](http://arxiv.org/abs/2112.07067)


  We develop methods to learn the correlation potential for a time-dependent
Kohn-Sham (TDKS) system in one spatial dimension. We start from a
low-dimensional two-electron system for which we can numerically solve the
time-dependent Schrödinger equation; this yields electron densities suitable
for training models of the correlation potential. We frame the learning problem
as one of optimizing a least-squares objective subject to the constraint that
the dynamics obey the TDKS equation. Applying adjoints, we develop efficient
methods to compute gradients and thereby learn models of the correlation
potential. Our results show that it is possible to learn values of the
correlation potential such that the resulting electron densities match ground
truth densities. We also show how to learn correlation potential functionals
with memory, demonstrating one such model that yields reasonable results for
trajectories outside the training set.

    

### [[2112.07068] Score-Based Generative Modeling with Critically-Damped Langevin Diffusion](http://arxiv.org/abs/2112.07068)


  Score-based generative models (SGMs) have demonstrated remarkable synthesis
quality. SGMs rely on a diffusion process that gradually perturbs the data
towards a tractable distribution, while the generative model learns to denoise.
The complexity of this denoising task is, apart from the data distribution
itself, uniquely determined by the diffusion process. We argue that current
SGMs employ overly simplistic diffusions, leading to unnecessarily complex
denoising processes, which limit generative modeling performance. Based on
connections to statistical mechanics, we propose a novel critically-damped
Langevin diffusion (CLD) and show that CLD-based SGMs achieve superior
performance. CLD can be interpreted as running a joint diffusion in an extended
space, where the auxiliary variables can be considered "velocities" that are
coupled to the data variables as in Hamiltonian dynamics. We derive a novel
score matching objective for CLD and show that the model only needs to learn
the score function of the conditional distribution of the velocity given data,
an easier task than learning scores of the data directly. We also derive a new
sampling scheme for efficient synthesis from CLD-based diffusion models. We
find that CLD outperforms previous SGMs in synthesis quality for similar
network architectures and sampling compute budgets. We show that our novel
sampler for CLD significantly outperforms solvers such as Euler--Maruyama. Our
framework provides new insights into score-based denoising diffusion models and
can be readily used for high-resolution image synthesis. Project page and code:
this https URL.

    

### [[2112.07074] Towards a Unified Foundation Model: Jointly Pre-Training Transformers on Unpaired Images and Text](http://arxiv.org/abs/2112.07074)


  In this paper, we explore the possibility of building a unified foundation
model that can be adapted to both vision-only and text-only tasks. Starting
from BERT and ViT, we design a unified transformer consisting of
modality-specific tokenizers, a shared transformer encoder, and task-specific
output heads. To efficiently pre-train the proposed model jointly on unpaired
images and text, we propose two novel techniques: (i) We employ the
separately-trained BERT and ViT models as teachers and apply knowledge
distillation to provide additional, accurate supervision signals for the joint
training; (ii) We propose a novel gradient masking strategy to balance the
parameter updates from the image and text pre-training losses. We evaluate the
jointly pre-trained transformer by fine-tuning it on image classification tasks
and natural language understanding tasks, respectively. The experiments show
that the resultant unified foundation transformer works surprisingly well on
both the vision-only and text-only tasks, and the proposed knowledge
distillation and gradient masking strategy can effectively lift the performance
to approach the level of separately-trained models.

    

### [[2112.07076] Real-Time Neural Voice Camouflage](http://arxiv.org/abs/2112.07076)


  Automatic speech recognition systems have created exciting possibilities for
applications, however they also enable opportunities for systematic
eavesdropping. We propose a method to camouflage a person's voice over-the-air
from these systems without inconveniencing the conversation between people in
the room. Standard adversarial attacks are not effective in real-time streaming
situations because the characteristics of the signal will have changed by the
time the attack is executed. We introduce predictive attacks, which achieve
real-time performance by forecasting the attack that will be the most effective
in the future. Under real-time constraints, our method jams the established
speech recognition system DeepSpeech 4.17x more than baselines as measured
through word error rate, and 7.27x more as measured through character error
rate. We furthermore demonstrate our approach is practically effective in
realistic environments over physical distances.

    

### [[2112.07087] Heuristic Hyperparameter Optimization for Convolutional Neural Networks using Genetic Algorithm](http://arxiv.org/abs/2112.07087)


  In recent years, people from all over the world are suffering from one of the
most severe diseases in history, known as Coronavirus disease 2019, COVID-19
for short. When the virus reaches the lungs, it has a higher probability to
cause lung pneumonia and sepsis. X-ray image is a powerful tool in identifying
the typical features of the infection for COVID-19 patients. The radiologists
and pathologists observe that ground-glass opacity appears in the chest X-ray
for infected patient \cite{cozzi2021ground}, and it could be used as one of the
criteria during the diagnosis process. In the past few years, deep learning has
proven to be one of the most powerful methods in the field of image
classification. Due to significant differences in Chest X-Ray between normal
and infected people \cite{rousan2020chest}, deep models could be used to
identify the presence of the disease given a patient's Chest X-Ray. Many deep
models are complex, and it evolves with lots of input parameters. Designers
sometimes struggle with the tuning process for deep models, especially when
they build up the model from scratch. Genetic Algorithm, inspired by the
biological evolution process, plays a key role in solving such complex
problems. In this paper, I proposed a genetic-based approach to optimize the
Convolutional Neural Network(CNN) for the Chest X-Ray classification task.

    

### [[2112.07096] Adaptive Projected Residual Networks for Learning Parametric Maps from Sparse Data](http://arxiv.org/abs/2112.07096)


  We present a parsimonious surrogate framework for learning high dimensional
parametric maps from limited training data. The need for parametric surrogates
arises in many applications that require repeated queries of complex
computational models. These applications include such "outer-loop" problems as
Bayesian inverse problems, optimal experimental design, and optimal design and
control under uncertainty, as well as real time inference and control problems.
Many high dimensional parametric mappings admit low dimensional structure,
which can be exploited by mapping-informed reduced bases of the inputs and
outputs. Exploiting this property, we develop a framework for learning low
dimensional approximations of such maps by adaptively constructing ResNet
approximations between reduced bases of their inputs and output. Motivated by
recent approximation theory for ResNets as discretizations of control flows, we
prove a universal approximation property of our proposed adaptive projected
ResNet framework, which motivates a related iterative algorithm for the ResNet
construction. This strategy represents a confluence of the approximation theory
and the algorithm since both make use of sequentially minimizing flows. In
numerical examples we show that these parsimonious, mapping-informed
architectures are able to achieve remarkably high accuracy given few training
data, making them a desirable surrogate strategy to be implemented for minimal
computational investment in training data generation.

    

### [[2112.07102] COVID-19 Pneumonia and Influenza Pneumonia Detection Using Convolutional Neural Networks](http://arxiv.org/abs/2112.07102)


  In the research, we developed a computer vision solution to support
diagnostic radiology in differentiating between COVID-19 pneumonia, influenza
virus pneumonia, and normal biomarkers. The chest radiograph appearance of
COVID-19 pneumonia is thought to be nonspecific, having presented a challenge
to identify an optimal architecture of a convolutional neural network (CNN)
that would classify with a high sensitivity among the pulmonary inflammation
features of COVID-19 and non-COVID-19 types of pneumonia. Rahman (2021) states
that COVID-19 radiography images observe unavailability and quality issues
impacting the diagnostic process and affecting the accuracy of the deep
learning detection models. A significant scarcity of COVID-19 radiography
images introduced an imbalance in data motivating us to use over-sampling
techniques. In the study, we include an extensive set of X-ray imaging of human
lungs (CXR) with COVID-19 pneumonia, influenza virus pneumonia, and normal
biomarkers to achieve an extensible and accurate CNN model. In the
experimentation phase of the research, we evaluated a variety of convolutional
network architectures, selecting a sequential convolutional network with two
traditional convolutional layers and two pooling layers with maximum function.
In its classification performance, the best performing model demonstrated a
validation accuracy of 93% and an F1 score of 0.95. We chose the Azure Machine
Learning service to perform network experimentation and solution deployment.
The auto-scaling compute clusters offered a significant time reduction in
network training. We would like to see scientists across fields of artificial
intelligence and human biology collaborating and expanding on the proposed
solution to provide rapid and comprehensive diagnostics, effectively mitigating
the spread of the virus

    

### [[2112.07110] Non Asymptotic Bounds for Optimization via Online Multiplicative Stochastic Gradient Descent](http://arxiv.org/abs/2112.07110)


  The gradient noise of Stochastic Gradient Descent (SGD) is considered to play
a key role in its properties (e.g. escaping low potential points and
regularization). Past research has indicated that the covariance of the SGD
error done via minibatching plays a critical role in determining its
regularization and escape from low potential points. It is however not much
explored how much the distribution of the error influences the behavior of the
algorithm. Motivated by some new research in this area, we prove universality
results by showing that noise classes that have the same mean and covariance
structure of SGD via minibatching have similar properties. We mainly consider
the Multiplicative Stochastic Gradient Descent (M-SGD) algorithm as introduced
by Wu et al., which has a much more general noise class than the SGD algorithm
done via minibatching. We establish nonasymptotic bounds for the M-SGD
algorithm mainly with respect to the Stochastic Differential Equation
corresponding to SGD via minibatching. We also show that the M-SGD error is
approximately a scaled Gaussian distribution with mean $0$ at any fixed point
of the M-SGD algorithm.

    

### [[2112.07116] Joint 3D Object Detection and Tracking Using Spatio-Temporal Representation of Camera Image and LiDAR Point Clouds](http://arxiv.org/abs/2112.07116)


  In this paper, we propose a new joint object detection and tracking (JoDT)
framework for 3D object detection and tracking based on camera and LiDAR
sensors. The proposed method, referred to as 3D DetecTrack, enables the
detector and tracker to cooperate to generate a spatio-temporal representation
of the camera and LiDAR data, with which 3D object detection and tracking are
then performed. The detector constructs the spatio-temporal features via the
weighted temporal aggregation of the spatial features obtained by the camera
and LiDAR fusion. Then, the detector reconfigures the initial detection results
using information from the tracklets maintained up to the previous time step.
Based on the spatio-temporal features generated by the detector, the tracker
associates the detected objects with previously tracked objects using a graph
neural network (GNN). We devise a fully-connected GNN facilitated by a
combination of rule-based edge pruning and attention-based edge gating, which
exploits both spatial and temporal object contexts to improve tracking
performance. The experiments conducted on both KITTI and nuScenes benchmarks
demonstrate that the proposed 3D DetecTrack achieves significant improvements
in both detection and tracking performances over baseline methods and achieves
state-of-the-art performance among existing methods through collaboration
between the detector and tracker.

    

### [[2112.07144] GEO-BLEU: Similarity Measure for Geospatial Sequences](http://arxiv.org/abs/2112.07144)


  In recent geospatial research, the importance of modeling large-scale human
mobility data via self-supervised learning is rising, in parallel with progress
in natural language processing driven by self-supervised approaches using
large-scale corpora. Whereas there are already plenty of feasible approaches
applicable to geospatial sequence modeling itself, there seems to be room to
improve with regard to evaluation, specifically about how to measure the
similarity between generated and reference sequences. In this work, we propose
a novel similarity measure, GEO-BLEU, which can be especially useful in the
context of geospatial sequence modeling and generation. As the name suggests,
this work is based on BLEU, one of the most popular measures used in machine
translation research, while introducing spatial proximity to the idea of
n-gram. We compare this measure with an established baseline, dynamic time
warping, applying it to actual generated geospatial sequences. Using
crowdsourced annotated data on the similarity between geospatial sequences
collected from over 12,000 cases, we quantitatively and qualitatively show the
proposed method's superiority.

    

### [[2112.07146] PP-HumanSeg: Connectivity-Aware Portrait Segmentation with a Large-Scale Teleconferencing Video Dataset](http://arxiv.org/abs/2112.07146)


  As the COVID-19 pandemic rampages across the world, the demands of video
conferencing surge. To this end, real-time portrait segmentation becomes a
popular feature to replace backgrounds of conferencing participants. While
feature-rich datasets, models and algorithms have been offered for segmentation
that extract body postures from life scenes, portrait segmentation has yet not
been well covered in a video conferencing context. To facilitate the progress
in this field, we introduce an open-source solution named PP-HumanSeg. This
work is the first to construct a large-scale video portrait dataset that
contains 291 videos from 23 conference scenes with 14K fine-labeled frames and
extensions to multi-camera teleconferencing. Furthermore, we propose a novel
Semantic Connectivity-aware Learning (SCL) for semantic segmentation, which
introduces a semantic connectivity-aware loss to improve the quality of
segmentation results from the perspective of connectivity. And we propose an
ultra-lightweight model with SCL for practical portrait segmentation, which
achieves the best trade-off between IoU and the speed of inference. Extensive
evaluations on our dataset demonstrate the superiority of SCL and our model.
The source code is available at this https URL.

    

### [[2112.07156] ImportantAug: a data augmentation agent for speech](http://arxiv.org/abs/2112.07156)


  We introduce ImportantAug, a technique to augment training data for speech
classification and recognition models by adding noise to unimportant regions of
the speech and not to important regions. Importance is predicted for each
utterance by a data augmentation agent that is trained to maximize the amount
of noise it adds while minimizing its impact on recognition performance. The
effectiveness of our method is illustrated on version two of the Google Speech
Commands (GSC) dataset. On the standard GSC test set, it achieves a 23.3%
relative error rate reduction compared to conventional noise augmentation which
applies noise to speech without regard to where it might be most effective. It
also provides a 25.4% error rate reduction compared to a baseline without data
augmentation. Additionally, the proposed ImportantAug outperforms the
conventional noise augmentation and the baseline on two test sets with
additional noise added.

    

### [[2112.07157] Federated Nearest Neighbor Classification with a Colony of Fruit-Flies: With Supplement](http://arxiv.org/abs/2112.07157)


  The mathematical formalization of a neurological mechanism in the olfactory
circuit of a fruit-fly as a locality sensitive hash (Flyhash) and bloom filter
(FBF) has been recently proposed and "reprogrammed" for various machine
learning tasks such as similarity search, outlier detection and text
embeddings. We propose a novel reprogramming of this hash and bloom filter to
emulate the canonical nearest neighbor classifier (NNC) in the challenging
Federated Learning (FL) setup where training and test data are spread across
parties and no data can leave their respective parties. Specifically, we
utilize Flyhash and FBF to create the FlyNN classifier, and theoretically
establish conditions where FlyNN matches NNC. We show how FlyNN is trained
exactly in a FL setup with low communication overhead to produce FlyNNFL, and
how it can be differentially private. Empirically, we demonstrate that (i)
FlyNN matches NNC accuracy across 70 OpenML datasets, (ii) FlyNNFL training is
highly scalable with low communication overhead, providing up to $8\times$
speedup with $16$ parties.

    

### [[2112.07160] Improving Spectral Graph Convolution for Learning Graph-level Representation](http://arxiv.org/abs/2112.07160)


  From the original theoretically well-defined spectral graph convolution to
the subsequent spatial bassed message-passing model, spatial locality (in
vertex domain) acts as a fundamental principle of most graph neural networks
(GNNs). In the spectral graph convolution, the filter is approximated by
polynomials, where a $k$-order polynomial covers $k$-hop neighbors. In the
message-passing, various definitions of neighbors used in aggregations are
actually an extensive exploration of the spatial locality information. For
learning node representations, the topological distance seems necessary since
it characterizes the basic relations between nodes. However, for learning
representations of the entire graphs, is it still necessary to hold? In this
work, we show that such a principle is not necessary, it hinders most existing
GNNs from efficiently encoding graph structures. By removing it, as well as the
limitation of polynomial filters, the resulting new architecture significantly
boosts performance on learning graph representations. We also study the effects
of graph spectrum on signals and interpret various existing improvements as
different spectrum smoothing techniques. It serves as a spatial understanding
that quantitatively measures the effects of the spectrum to input signals in
comparison to the well-known spectral understanding as high/low-pass filters.
More importantly, it sheds the light on developing powerful graph
representation models.

    

### [[2112.07163] Minimization of Stochastic First-order Oracle Complexity of Adaptive Methods for Nonconvex Optimization](http://arxiv.org/abs/2112.07163)


  Numerical evaluations have definitively shown that, for deep learning
optimizers such as stochastic gradient descent, momentum, and adaptive methods,
the number of steps needed to train a deep neural network halves for each
doubling of the batch size and that there is a region of diminishing returns
beyond the critical batch size. In this paper, we determine the actual critical
batch size by using the global minimizer of the stochastic first-order oracle
(SFO) complexity of the optimizer. To prove the existence of the actual
critical batch size, we set the lower and upper bounds of the SFO complexity
and prove that there exist critical batch sizes in the sense of minimizing the
lower and upper bounds. This proof implies that, if the SFO complexity fits the
lower and upper bounds, then the existence of these critical batch sizes
demonstrates the existence of the actual critical batch size. We also discuss
the conditions needed for the SFO complexity to fit the lower and upper bounds
and provide numerical results that support our theoretical results.

    

### [[2112.07184] Calibrated and Sharp Uncertainties in Deep Learning via Simple Density Estimation](http://arxiv.org/abs/2112.07184)


  Predictive uncertainties can be characterized by two properties--calibration
and sharpness. This paper argues for reasoning about uncertainty in terms these
properties and proposes simple algorithms for enforcing them in deep learning.
Our methods focus on the strongest notion of calibration--distribution
calibration--and enforce it by fitting a low-dimensional density or quantile
function with a neural estimator. The resulting approach is much simpler and
more broadly applicable than previous methods across both classification and
regression. Empirically, we find that our methods improve predictive
uncertainties on several tasks with minimal computational and implementation
overhead. Our insights suggest simple and improved ways of training deep
learning models that lead to accurate uncertainties that should be leveraged to
improve performance across downstream applications.

    

### [[2112.07207] Modeling Image Quantization Tradeoffs for Optimal Compression](http://arxiv.org/abs/2112.07207)


  All Lossy compression algorithms employ similar compression schemes --
frequency domain transform followed by quantization and lossless encoding
schemes. They target tradeoffs by quantizating high frequency data to increase
compression rates which come at the cost of higher image distortion. We propose
a new method of optimizing quantization tables using Deep Learning and a
minimax loss function that more accurately measures the tradeoffs between rate
and distortion parameters (RD) than previous methods. We design a convolutional
neural network (CNN) that learns a mapping between image blocks and
quantization tables in an unsupervised manner. By processing images across all
channels at once, we can achieve stronger performance by also measuring
tradeoffs in information loss between different channels. We initially target
optimization on JPEG images but feel that this can be expanded to any lossy
compressor.

    

### [[2112.07209] ACE-BERT: Adversarial Cross-modal Enhanced BERT for E-commerce Retrieval](http://arxiv.org/abs/2112.07209)


  Nowadays on E-commerce platforms, products are presented to the customers
with multiple modalities. These multiple modalities are significant for a
retrieval system while providing attracted products for customers. Therefore,
how to take into account those multiple modalities simultaneously to boost the
retrieval performance is crucial. This problem is a huge challenge to us due to
the following reasons: (1) the way of extracting patch features with the
pre-trained image model (e.g., CNN-based model) has much inductive bias. It is
difficult to capture the efficient information from the product image in
E-commerce. (2) The heterogeneity of multimodal data makes it challenging to
construct the representations of query text and product including title and
image in a common subspace. We propose a novel Adversarial Cross-modal Enhanced
BERT (ACE-BERT) for efficient E-commerce retrieval. In detail, ACE-BERT
leverages the patch features and pixel features as image representation. Thus
the Transformer architecture can be applied directly to the raw image
sequences. With the pre-trained enhanced BERT as the backbone network, ACE-BERT
further adopts adversarial learning by adding a domain classifier to ensure the
distribution consistency of different modality representations for the purpose
of narrowing down the representation gap between query and product.
Experimental results demonstrate that ACE-BERT outperforms the state-of-the-art
approaches on the retrieval task. It is remarkable that ACE-BERT has already
been deployed in our E-commerce's search engine, leading to 1.46% increase in
revenue.

    

### [[2112.07221] HET: Scaling out Huge Embedding Model Training via Cache-enabled Distributed Framework](http://arxiv.org/abs/2112.07221)


  Embedding models have been an effective learning paradigm for
high-dimensional data. However, one open issue of embedding models is that
their representations (latent factors) often result in large parameter space.
We observe that existing distributed training frameworks face a scalability
issue of embedding models since updating and retrieving the shared embedding
parameters from servers usually dominates the training cycle. In this paper, we
propose HET, a new system framework that significantly improves the scalability
of huge embedding model training. We embrace skewed popularity distributions of
embeddings as a performance opportunity and leverage it to address the
communication bottleneck with an embedding cache. To ensure consistency across
the caches, we incorporate a new consistency model into HET design, which
provides fine-grained consistency guarantees on a per-embedding basis. Compared
to previous work that only allows staleness for read operations, HET also
utilizes staleness for write operations. Evaluations on six representative
tasks show that HET achieves up to 88% embedding communication reductions and
up to 20.68x performance speedup over the state-of-the-art baselines.

    

### [[2112.07222] Meta-CPR: Generalize to Unseen Large Number of Agents with Communication Pattern Recognition Module](http://arxiv.org/abs/2112.07222)


  Designing an effective communication mechanism among agents in reinforcement
learning has been a challenging task, especially for real-world applications.
The number of agents can grow or an environment sometimes needs to interact
with a changing number of agents in real-world scenarios. To this end, a
multi-agent framework needs to handle various scenarios of agents, in terms of
both scales and dynamics, for being practical to real-world applications. We
formulate the multi-agent environment with a different number of agents as a
multi-tasking problem and propose a meta reinforcement learning (meta-RL)
framework to tackle this problem. The proposed framework employs a meta-learned
Communication Pattern Recognition (CPR) module to identify communication
behavior and extract information that facilitates the training process.
Experimental results are poised to demonstrate that the proposed framework (a)
generalizes to an unseen larger number of agents and (b) allows the number of
agents to change between episodes. The ablation study is also provided to
reason the proposed CPR design and show such design is effective.

    

### [[2112.07225] Margin Calibration for Long-Tailed Visual Recognition](http://arxiv.org/abs/2112.07225)


  The long-tailed class distribution in visual recognition tasks poses great
challenges for neural networks on how to handle the biased predictions between
head and tail classes, i.e., the model tends to classify tail classes as head
classes. While existing research focused on data resampling and loss function
engineering, in this paper, we take a different perspective: the classification
margins. We study the relationship between the margins and logits
(classification scores) and empirically observe the biased margins and the
biased logits are positively correlated. We propose MARC, a simple yet
effective MARgin Calibration function to dynamically calibrate the biased
margins for unbiased logits. We validate MARC through extensive experiments on
common long-tailed benchmarks including CIFAR-LT, ImageNet-LT, Places-LT, and
iNaturalist-LT. Experimental results demonstrate that our MARC achieves
favorable results on these benchmarks. In addition, MARC is extremely easy to
implement with just three lines of code. We hope this simple method will
motivate people to rethink the biased margins and biased logits in long-tailed
visual recognition.

    

### [[2112.07227] Unsupervised feature selection via self-paced learning and low-redundant regularization](http://arxiv.org/abs/2112.07227)


  Much more attention has been paid to unsupervised feature selection nowadays
due to the emergence of massive unlabeled data. The distribution of samples and
the latent effect of training a learning method using samples in more effective
order need to be considered so as to improve the robustness of the method.
Self-paced learning is an effective method considering the training order of
samples. In this study, an unsupervised feature selection is proposed by
integrating the framework of self-paced learning and subspace learning.
Moreover, the local manifold structure is preserved and the redundancy of
features is constrained by two regularization terms. $L_{2,1/2}$-norm is
applied to the projection matrix, which aims to retain discriminative features
and further alleviate the effect of noise in the data. Then, an iterative
method is presented to solve the optimization problem. The convergence of the
method is proved theoretically and experimentally. The proposed method is
compared with other state of the art algorithms on nine real-world datasets.
The experimental results show that the proposed method can improve the
performance of clustering methods and outperform other compared algorithms.

    

### [[2112.07239] Compensating trajectory bias for unsupervised patient stratification using adversarial recurrent neural networks](http://arxiv.org/abs/2112.07239)


  Electronic healthcare records are an important source of information which
can be used in patient stratification to discover novel disease phenotypes.
However, they can be challenging to work with as data is often sparse and
irregularly sampled. One approach to solve these limitations is learning dense
embeddings that represent individual patient trajectories using a recurrent
neural network autoencoder (RNN-AE). This process can be susceptible to
unwanted data biases. We show that patient embeddings and clusters using
previously proposed RNN-AE models might be impacted by a trajectory bias,
meaning that results are dominated by the amount of data contained in each
patients trajectory, instead of clinically relevant details. We investigate
this bias on 2 datasets (from different hospitals) and 2 disease areas as well
as using different parts of the patient trajectory. Our results using 2
previously published baseline methods indicate a particularly strong bias in
case of an event-to-end trajectory. We present a method that can overcome this
issue using an adversarial training scheme on top of a RNN-AE. Our results show
that our approach can reduce the trajectory bias in all cases.

    

### [[2112.07259] TopNet: Learning from Neural Topic Model to Generate Long Stories](http://arxiv.org/abs/2112.07259)


  Long story generation (LSG) is one of the coveted goals in natural language
processing. Different from most text generation tasks, LSG requires to output a
long story of rich content based on a much shorter text input, and often
suffers from information sparsity. In this paper, we propose \emph{TopNet} to
alleviate this problem, by leveraging the recent advances in neural topic
modeling to obtain high-quality skeleton words to complement the short input.
In particular, instead of directly generating a story, we first learn to map
the short text input to a low-dimensional topic distribution (which is
pre-assigned by a topic model). Based on this latent topic distribution, we can
use the reconstruction decoder of the topic model to sample a sequence of
inter-related words as a skeleton for the story. Experiments on two benchmark
datasets show that our proposed framework is highly effective in skeleton word
selection and significantly outperforms the state-of-the-art models in both
automatic evaluation and human evaluation.

    

### [[2112.07262] Inductive Semi-supervised Learning Through Optimal Transport](http://arxiv.org/abs/2112.07262)


  In this paper, we tackle the inductive semi-supervised learning problem that
aims to obtain label predictions for out-of-sample data. The proposed approach,
called Optimal Transport Induction (OTI), extends efficiently an optimal
transport based transductive algorithm (OTP) to inductive tasks for both binary
and multi-class settings. A series of experiments are conducted on several
datasets in order to compare the proposed approach with state-of-the-art
methods. Experiments demonstrate the effectiveness of our approach. We make our
code publicly available (Code is available at:
this https URL).

    

### [[2112.07263] Quantifying Multimodality in World Models](http://arxiv.org/abs/2112.07263)


  Model-based Deep Reinforcement Learning (RL) assumes the availability of a
model of an environment's underlying transition dynamics. This model can be
used to predict future effects of an agent's possible actions. When no such
model is available, it is possible to learn an approximation of the real
environment, e.g. by using generative neural networks, sometimes also called
World Models. As most real-world environments are stochastic in nature and the
transition dynamics are oftentimes multimodal, it is important to use a
modelling technique that is able to reflect this multimodal uncertainty. In
order to safely deploy such learning systems in the real world, especially in
an industrial context, it is paramount to consider these uncertainties. In this
work, we analyze existing and propose new metrics for the detection and
quantification of multimodal uncertainty in RL based World Models. The correct
modelling & detection of uncertain future states lays the foundation for
handling critical situations in a safe way, which is a prerequisite for
deploying RL systems in real-world settings.

    

### [[2112.07285] Automatic COVID-19 disease diagnosis using 1D convolutional neural network and augmentation with human respiratory sound based on parameters: cough, breath, and voice](http://arxiv.org/abs/2112.07285)


  The issue in respiratory sound classification has attained good attention
from the clinical scientists and medical researcher's group in the last year to
diagnosing COVID-19 disease. To date, various models of Artificial Intelligence
(AI) entered into the real-world to detect the COVID-19 disease from
human-generated sounds such as voice/speech, cough, and breath. The
Convolutional Neural Network (CNN) model is implemented for solving a lot of
real-world problems on machines based on Artificial Intelligence (AI). In this
context, one dimension (1D) CNN is suggested and implemented to diagnose
respiratory diseases of COVID-19 from human respiratory sounds such as a voice,
cough, and breath. An augmentation-based mechanism is applied to improve the
preprocessing performance of the COVID-19 sounds dataset and to automate
COVID-19 disease diagnosis using the 1D convolutional network. Furthermore, a
DDAE (Data De-noising Auto Encoder) technique is used to generate deep sound
features such as the input function to the 1D CNN instead of adopting the
standard input of MFCC (Mel-frequency cepstral coefficient), and it is
performed better accuracy and performance than previous models.

    

### [[2112.07313] Autonomous Navigation and Configuration of Integrated Access Backhauling for UAV Base Station Using Reinforcement Learning](http://arxiv.org/abs/2112.07313)


  Fast and reliable connectivity is essential to enhancing situational
awareness and operational efficiency for public safety mission-critical (MC)
users. In emergency or disaster circumstances, where existing cellular network
coverage and capacity may not be available to meet MC communication demands,
deployable-network-based solutions such as cells-on-wheels/wings can be
utilized swiftly to ensure reliable connection for MC users. In this paper, we
consider a scenario where a macro base station (BS) is destroyed due to a
natural disaster and an unmanned aerial vehicle carrying BS (UAV-BS) is set up
to provide temporary coverage for users in the disaster area. The UAV-BS is
integrated into the mobile network using the 5G integrated access and backhaul
(IAB) technology. We propose a framework and signalling procedure for applying
machine learning to this use case. A deep reinforcement learning algorithm is
designed to jointly optimize the access and backhaul antenna tilt as well as
the three-dimensional location of the UAV-BS in order to best serve the
on-ground MC users while maintaining a good backhaul connection. Our result
shows that the proposed algorithm can autonomously navigate and configure the
UAV-BS to improve the throughput and reduce the drop rate of MC users.

    

### [[2112.07324] On the Impact of Hard Adversarial Instances on Overfitting in Adversarial Training](http://arxiv.org/abs/2112.07324)


  Adversarial training is a popular method to robustify models against
adversarial attacks. However, it exhibits much more severe overfitting than
training on clean inputs. In this work, we investigate this phenomenon from the
perspective of training instances, i.e., training input-target pairs. Based on
a quantitative metric measuring instances' difficulty, we analyze the model's
behavior on training instances of different difficulty levels. This lets us
show that the decay in generalization performance of adversarial training is a
result of the model's attempt to fit hard adversarial instances. We
theoretically verify our observations for both linear and general nonlinear
models, proving that models trained on hard instances have worse generalization
performance than ones trained on easy instances. Furthermore, we prove that the
difference in the generalization gap between models trained by instances of
different difficulty levels increases with the size of the adversarial budget.
Finally, we conduct case studies on methods mitigating adversarial overfitting
in several scenarios. Our analysis shows that methods successfully mitigating
adversarial overfitting all avoid fitting hard adversarial instances, while
ones fitting hard adversarial instances do not achieve true robustness.

    

### [[2112.07328] Biased Gradient Estimate with Drastic Variance Reduction for Meta Reinforcement Learning](http://arxiv.org/abs/2112.07328)


  Despite the empirical success of meta reinforcement learning (meta-RL), there
are still a number poorly-understood discrepancies between theory and practice.
Critically, biased gradient estimates are almost always implemented in
practice, whereas prior theory on meta-RL only establishes convergence under
unbiased gradient estimates. In this work, we investigate such a discrepancy.
In particular, (1) We show that unbiased gradient estimates have variance
$\Theta(N)$ which linearly depends on the sample size $N$ of the inner loop
updates; (2) We propose linearized score function (LSF) gradient estimates,
which have bias $\mathcal{O}(1/\sqrt{N})$ and variance $\mathcal{O}(1/N)$; (3)
We show that most empirical prior work in fact implements variants of the LSF
gradient estimates. This implies that practical algorithms "accidentally"
introduce bias to achieve better performance; (4) We establish theoretical
guarantees for the LSF gradient estimates in meta-RL regarding its convergence
to stationary points, showing better dependency on $N$ than prior work when $N$
is large.

    

### [[2112.07342] Learning to Guide and to Be Guided in the Architect-Builder Problem](http://arxiv.org/abs/2112.07342)


  We are interested in interactive agents that learn to coordinate, namely, a
$builder$ -- which performs actions but ignores the goal of the task -- and an
$architect$ which guides the builder towards the goal of the task. We define
and explore a formal setting where artificial agents are equipped with
mechanisms that allow them to simultaneously learn a task while at the same
time evolving a shared communication protocol. The field of Experimental
Semiotics has shown the extent of human proficiency at learning from a priori
unknown instructions meanings. Therefore, we take inspiration from it and
present the Architect-Builder Problem (ABP): an asymmetrical setting in which
an architect must learn to guide a builder towards constructing a specific
structure. The architect knows the target structure but cannot act in the
environment and can only send arbitrary messages to the builder. The builder on
the other hand can act in the environment but has no knowledge about the task
at hand and must learn to solve it relying only on the messages sent by the
architect. Crucially, the meaning of messages is initially not defined nor
shared between the agents but must be negotiated throughout learning. Under
these constraints, we propose Architect-Builder Iterated Guiding (ABIG), a
solution to the Architect-Builder Problem where the architect leverages a
learned model of the builder to guide it while the builder uses self-imitation
learning to reinforce its guided behavior. We analyze the key learning
mechanisms of ABIG and test it in a 2-dimensional instantiation of the ABP
where tasks involve grasping cubes, placing them at a given location, or
building various shapes. In this environment, ABIG results in a low-level,
high-frequency, guiding communication protocol that not only enables an
architect-builder pair to solve the task at hand, but that can also generalize
to unseen tasks.

    

### [[2112.07344] SC-Reg: Training Overparameterized Neural Networks under Self-Concordant Regularization](http://arxiv.org/abs/2112.07344)


  In this paper we propose the SC-Reg (self-concordant regularization)
framework for learning overparameterized feedforward neural networks by
incorporating second-order information in the \emph{Newton decrement} framework
for convex problems. We propose the generalized Gauss-Newton with
Self-Concordant Regularization (SCoRe-GGN) algorithm that updates the network
parameters each time it receives a new input batch. The proposed algorithm
exploits the structure of the second-order information in the Hessian matrix,
thereby reducing the training computational overhead. Although our current
analysis considers only the convex case, numerical experiments show the
efficiency of our method and its fast convergence under both convex and
non-convex settings, which compare favorably against baseline first-order
methods and a quasi-Newton method.

    

### [[2112.07353] Machine Learning-based Prediction of Porosity for Concrete Containing Supplementary Cementitious Materials](http://arxiv.org/abs/2112.07353)


  Porosity has been identified as the key indicator of the durability
properties of concrete exposed to aggressive environments. This paper applies
ensemble learning to predict porosity of high-performance concrete containing
supplementary cementitious materials. The concrete samples utilized in this
study are characterized by eight composition features including w/b ratio,
binder content, fly ash, GGBS, superplasticizer, coarse/fine aggregate ratio,
curing condition and curing days. The assembled database consists of 240 data
records, featuring 74 unique concrete mixture designs. The proposed machine
learning algorithms are trained on 180 observations (75%) chosen randomly from
the data set and then tested on the remaining 60 observations (25%). The
numerical experiments suggest that the regression tree ensembles can accurately
predict the porosity of concrete from its mixture compositions. Gradient
boosting trees generally outperforms random forests in terms of prediction
accuracy. For random forests, the out-of-bag error based hyperparameter tuning
strategy is found to be much more efficient than k-Fold Cross-Validation.

    

### [[2112.07356] Technical Language Supervision for Intelligent Fault Diagnosis in Process Industry](http://arxiv.org/abs/2112.07356)


  In the process industry, condition monitoring systems with automated fault
diagnosis methods assisthuman experts and thereby improve maintenance
efficiency, process sustainability, and workplace safety.Improving the
automated fault diagnosis methods using data and machine learning-based models
is a centralaspect of intelligent fault diagnosis (IFD). A major challenge in
IFD is to develop realistic datasets withaccurate labels needed to train and
validate models, and to transfer models trained with labeled lab datato
heterogeneous process industry environments. However, fault descriptions and
work-orders written bydomain experts are increasingly digitized in modern
condition monitoring systems, for example in the contextof rotating equipment
monitoring. Thus, domain-specific knowledge about fault characteristics and
severitiesexists as technical language annotations in industrial datasets.
Furthermore, recent advances in naturallanguage processing enable weakly
supervised model optimization using natural language annotations, mostnotably
in the form ofnatural language supervision(NLS). This creates a timely
opportunity to developtechnical language supervision(TLS) solutions for IFD
systems grounded in industrial data, for exampleas a complement to pre-training
with lab data to address problems like overfitting and inaccurate out-of-sample
generalisation. We surveyed the literature and identify a considerable
improvement in the maturityof NLS over the last two years, facilitating
applications beyond natural language; a rapid development ofweak supervision
methods; and transfer learning as a current trend in IFD which can benefit from
thesedevelopments. Finally, we describe a framework for integration of TLS in
IFD which is inspired by recentNLS innovations.

    

### [[2112.07368] Simple and Robust Loss Design for Multi-Label Learning with Missing Labels](http://arxiv.org/abs/2112.07368)


  Multi-label learning in the presence of missing labels (MLML) is a
challenging problem. Existing methods mainly focus on the design of network
structures or training schemes, which increase the complexity of
implementation. This work seeks to fulfill the potential of loss function in
MLML without increasing the procedure and complexity. Toward this end, we
propose two simple yet effective methods via robust loss design based on an
observation that a model can identify missing labels during training with a
high precision. The first is a novel robust loss for negatives, namely the Hill
loss, which re-weights negatives in the shape of a hill to alleviate the effect
of false negatives. The second is a self-paced loss correction (SPLC) method,
which uses a loss derived from the maximum likelihood criterion under an
approximate distribution of missing labels. Comprehensive experiments on a vast
range of multi-label image classification datasets demonstrate that our methods
can remarkably boost the performance of MLML and achieve new state-of-the-art
loss functions in MLML.

    

### [[2112.07369] Convergence proof for stochastic gradient descent in the training of deep neural networks with ReLU activation for constant target functions](http://arxiv.org/abs/2112.07369)


  In many numerical simulations stochastic gradient descent (SGD) type
optimization methods perform very effectively in the training of deep neural
networks (DNNs) but till this day it remains an open problem of research to
provide a mathematical convergence analysis which rigorously explains the
success of SGD type optimization methods in the training of DNNs. In this work
we study SGD type optimization methods in the training of fully-connected
feedforward DNNs with rectified linear unit (ReLU) activation. We first
establish general regularity properties for the risk functions and their
generalized gradient functions appearing in the training of such DNNs and,
thereafter, we investigate the plain vanilla SGD optimization method in the
training of such DNNs under the assumption that the target function under
consideration is a constant function. Specifically, we prove under the
assumption that the learning rates (the step sizes of the SGD optimization
method) are sufficiently small but not $L^1$-summable and under the assumption
that the target function is a constant function that the expectation of the
riskof the considered SGD process converges in the training of such DNNs to
zero as the number of SGD steps increases to infinity.

    

### [[2112.07400] Robustifying automatic speech recognition by extracting slowly varying features](http://arxiv.org/abs/2112.07400)


  In the past few years, it has been shown that deep learning systems are
highly vulnerable under attacks with adversarial examples. Neural-network-based
automatic speech recognition (ASR) systems are no exception. Targeted and
untargeted attacks can modify an audio input signal in such a way that humans
still recognise the same words, while ASR systems are steered to predict a
different transcription. In this paper, we propose a defense mechanism against
targeted adversarial attacks consisting in removing fast-changing features from
the audio signals, either by applying slow feature analysis, a low-pass filter,
or both, before feeding the input to the ASR system. We perform an empirical
analysis of hybrid ASR models trained on data pre-processed in such a way.
While the resulting models perform quite well on benign data, they are
significantly more robust against targeted adversarial attacks: Our final,
proposed model shows a performance on clean data similar to the baseline model,
while being more than four times more robust.

    

### [[2112.07406] Branching Time Active Inference with Bayesian Filtering](http://arxiv.org/abs/2112.07406)


  Branching Time Active Inference (Champion et al., 2021b,a) is a framework
proposing to look at planning as a form of Bayesian model expansion. Its root
can be found in Active Inference (Friston et al., 2016; Da Costa et al., 2020;
Champion et al., 2021c), a neuroscientific framework widely used for brain
modelling, as well as in Monte Carlo Tree Search (Browne et al., 2012), a
method broadly applied in the Reinforcement Learning literature. Up to now, the
inference of the latent variables was carried out by taking advantage of the
flexibility offered by Variational Message Passing (Winn and Bishop, 2005), an
iterative process that can be understood as sending messages along the edges of
a factor graph (Forney, 2001). In this paper, we harness the efficiency of an
alternative method for inference called Bayesian Filtering (Fox et al., 2003),
which does not require the iteration of the update equations until convergence
of the Variational Free Energy. Instead, this scheme alternates between two
phases: integration of evidence and prediction of future states. Both of those
phases can be performed efficiently and this provides a seventy times speed up
over the state-of-the-art.

    

### [[2112.07424] Conjugated Discrete Distributions for Distributional Reinforcement Learning](http://arxiv.org/abs/2112.07424)


  In this work we continue to build upon recent advances in reinforcement
learning for finite Markov processes. A common approach among previous existing
algorithms, both single-actor and distributed, is to either clip rewards or to
apply a transformation method on Q-functions to handle a large variety of
magnitudes in real discounted returns. We theoretically show that one of the
most successful methods may not yield an optimal policy if we have a
non-deterministic process. As a solution, we argue that distributional
reinforcement learning lends itself to remedy this situation completely. By the
introduction of a conjugated distributional operator we may handle a large
class of transformations for real returns with guaranteed theoretical
convergence. We propose an approximating single-actor algorithm based on this
operator that trains agents directly on unaltered rewards using a proper
distributional metric given by the Cramér distance. To evaluate its
performance in a stochastic setting we train agents on a suite of 55 Atari 2600
games using sticky-actions and obtain state-of-the-art performance compared to
other well-known algorithms in the Dopamine framework.

    

### [[2112.07426] Direct Training via Backpropagation for Ultra-low Latency Spiking Neural Networks with Multi-threshold](http://arxiv.org/abs/2112.07426)


  Spiking neural networks (SNNs) can utilize spatio-temporal information and
have a nature of energy efficiency which is a good alternative to deep neural
networks(DNNs). The event-driven information processing makes SNNs can reduce
the expensive computation of DNNs and save a lot of energy consumption.
However, high training and inference latency is a limitation of the development
of deeper SNNs. SNNs usually need tens or even hundreds of time steps during
the training and inference process which causes not only the increase of
latency but also the waste of energy consumption. To overcome this problem, we
proposed a novel training method based on backpropagation (BP) for ultra-low
latency(1-2 time steps) SNN with multi-threshold. In order to increase the
information capacity of each spike, we introduce the multi-threshold Leaky
Integrate and Fired (LIF) model. In our proposed training method, we proposed
three approximated derivative for spike activity to solve the problem of the
non-differentiable issue which cause difficulties for direct training SNNs
based on BP. The experimental results show that our proposed method achieves an
average accuracy of 99.56%, 93.08%, and 87.90% on MNIST, FashionMNIST, and
CIFAR10, respectively with only 2 time steps. For the CIFAR10 dataset, our
proposed method achieve 1.12% accuracy improvement over the previously reported
direct trained SNNs with fewer time steps.

    

### [[2112.07428] Obtaining Calibrated Probabilities with Personalized Ranking Models](http://arxiv.org/abs/2112.07428)


  For personalized ranking models, the well-calibrated probability of an item
being preferred by a user has great practical value. While existing work shows
promising results in image classification, probability calibration has not been
much explored for personalized ranking. In this paper, we aim to estimate the
calibrated probability of how likely a user will prefer an item. We investigate
various parametric distributions and propose two parametric calibration
methods, namely Gaussian calibration and Gamma calibration. Each proposed
method can be seen as a post-processing function that maps the ranking scores
of pre-trained models to well-calibrated preference probabilities, without
affecting the recommendation performance. We also design the unbiased empirical
risk minimization framework that guides the calibration methods to learning of
true preference probability from the biased user-item interaction dataset.
Extensive evaluations with various personalized ranking models on real-world
datasets show that both the proposed calibration methods and the unbiased
empirical risk minimization significantly improve the calibration performance.

    

### [[2112.07434] Exploring the Limits of Natural Language Inference Based Setup for Few-Shot Intent Detection](http://arxiv.org/abs/2112.07434)


  One of the core components of goal-oriented dialog systems is the task of
Intent Detection. Few-shot Learning upon Intent Detection is challenging due to
the scarcity of available annotated utterances. Although recent works making
use of metric-based and optimization-based methods have been proposed, the task
is still challenging in large label spaces and much smaller number of shots.
Generalized Few-shot learning is more difficult due to the presence of both
novel and seen classes during the testing phase. In this work, we propose a
simple and effective method based on Natural Language Inference that not only
tackles the problem of few shot intent detection, but also proves useful in
zero-shot and generalized few shot learning problems. Our extensive experiments
on a number of Natural Language Understanding (NLU) and Spoken Language
Understanding (SLU) datasets show the effectiveness of our approach. In
addition, we highlight the settings in which our NLI based method outperforms
the baselines by huge margins.

    

### [[2112.07436] Graph Kernel Neural Networks](http://arxiv.org/abs/2112.07436)


  The convolution operator at the core of many modern neural architectures can
effectively be seen as performing a dot product between an input matrix and a
filter. While this is readily applicable to data such as images, which can be
represented as regular grids in the Euclidean space, extending the convolution
operator to work on graphs proves more challenging, due to their irregular
structure. In this paper, we propose to use graph kernels, i.e., kernel
functions that compute an inner product on graphs, to extend the standard
convolution operator to the graph domain. This allows us to define an entirely
structural model that does not require computing the embedding of the input
graph. Our architecture allows to plug-in any type and number of graph kernels
and has the added benefit of providing some interpretability in terms of the
structural masks that are learned during the training process, similarly to
what happens for convolutional masks in traditional convolutional neural
networks. We perform an extensive ablation study to investigate the impact of
the model hyper-parameters and we show that our model achieves competitive
performance on standard graph classification datasets.

    

### [[2112.07437] Bayesian Learning of Play Styles in Multiplayer Video Games](http://arxiv.org/abs/2112.07437)


  The complexity of game play in online multiplayer games has generated strong
interest in modeling the different play styles or strategies used by players
for success. We develop a hierarchical Bayesian regression approach for the
online multiplayer game Battlefield 3 where performance is modeled as a
function of the roles, game type, and map taken on by that player in each of
their matches. We use a Dirichlet process prior that enables the clustering of
players that have similar player-specific coefficients in our regression model,
which allows us to discover common play styles amongst our sample of
Battlefield 3 players. This Bayesian semi-parametric clustering approach has
several advantages: the number of common play styles do not need to be
specified, players can move between multiple clusters, and the resulting
groupings often have a straight-forward interpretations. We examine the most
common play styles among Battlefield 3 players in detail and find groups of
players that exhibit overall high performance, as well as groupings of players
that perform particularly well in specific game types, maps and roles. We are
also able to differentiate between players that are stable members of a
particular play style from hybrid players that exhibit multiple play styles
across their matches. Modeling this landscape of different play styles will aid
game developers in developing specialized tutorials for new participants as
well as improving the construction of complementary teams in their online
matching queues.

    

### [[2112.07441] An Interpretive Constrained Linear Model for ResNet and MgNet](http://arxiv.org/abs/2112.07441)


  We propose a constrained linear data-feature-mapping model as an
interpretable mathematical model for image classification using a convolutional
neural network (CNN). From this viewpoint, we establish detailed connections
between the traditional iterative schemes for linear systems and the
architectures of the basic blocks of ResNet- and MgNet-type models. Using these
connections, we present some modified ResNet models that compared with the
original models have fewer parameters and yet can produce more accurate
results, thereby demonstrating the validity of this constrained learning
data-feature-mapping assumption. Based on this assumption, we further propose a
general data-feature iterative scheme to show the rationality of MgNet. We also
provide a systematic numerical study on MgNet to show its success and
advantages in image classification problems and demonstrate its advantages in
comparison with established networks.

    

### [[2112.07447] Measuring Fairness with Biased Rulers: A Survey on Quantifying Biases in Pretrained Language Models](http://arxiv.org/abs/2112.07447)


  An increasing awareness of biased patterns in natural language processing
resources, like BERT, has motivated many metrics to quantify `bias' and
`fairness'. But comparing the results of different metrics and the works that
evaluate with such metrics remains difficult, if not outright impossible. We
survey the existing literature on fairness metrics for pretrained language
models and experimentally evaluate compatibility, including both biases in
language models as in their downstream tasks. We do this by a mixture of
traditional literature survey and correlation analysis, as well as by running
empirical evaluations. We find that many metrics are not compatible and highly
depend on (i) templates, (ii) attribute and target seeds and (iii) the choice
of embeddings. These results indicate that fairness or bias evaluation remains
challenging for contextualized language models, if not at least highly
subjective. To improve future comparisons and fairness evaluations, we
recommend avoiding embedding-based metrics and focusing on fairness evaluations
in downstream tasks.

    

### [[2112.07457] Triangulation candidates for Bayesian optimization](http://arxiv.org/abs/2112.07457)


  Bayesian optimization is a form of sequential design: idealize input-output
relationships with a suitably flexible nonlinear regression model; fit to data
from an initial experimental campaign; devise and optimize a criterion for
selecting the next experimental condition(s) under the fitted model (e.g., via
predictive equations) to target outcomes of interest (say minima); repeat after
acquiring output under those conditions and updating the fit. In many
situations this "inner optimization" over the new-data acquisition criterion is
cumbersome because it is non-convex/highly multi-modal, may be
non-differentiable, or may otherwise thwart numerical optimizers, especially
when inference requires Monte Carlo. In such cases it is not uncommon to
replace continuous search with a discrete one over random candidates. Here we
propose using candidates based on a Delaunay triangulation of the existing
input design. In addition to detailing construction of these "tricands", based
on a simple wrapper around a conventional convex hull library, we promote
several advantages based on properties of the geometric criterion involved. We
then demonstrate empirically how tricands can lead to better Bayesian
optimization performance compared to both numerically optimized acquisitions
and random candidate-based alternatives on benchmark problems.

    

### [[2112.07459] Scale-Aware Neural Architecture Search for Multivariate Time Series Forecasting](http://arxiv.org/abs/2112.07459)


  Multivariate time series (MTS) forecasting has attracted much attention in
many intelligent applications. It is not a trivial task, as we need to consider
both intra-variable dependencies and inter-variable dependencies. However,
existing works are designed for specific scenarios, and require much domain
knowledge and expert efforts, which is difficult to transfer between different
scenarios. In this paper, we propose a scale-aware neural architecture search
framework for MTS forecasting (SNAS4MTF). A multi-scale decomposition module
transforms raw time series into multi-scale sub-series, which can preserve
multi-scale temporal patterns. An adaptive graph learning module infers the
different inter-variable dependencies under different time scales without any
prior knowledge. For MTS forecasting, a search space is designed to capture
both intra-variable dependencies and inter-variable dependencies at each time
scale. The multi-scale decomposition, adaptive graph learning, and neural
architecture search modules are jointly learned in an end-to-end framework.
Extensive experiments on two real-world datasets demonstrate that SNAS4MTF
achieves a promising performance compared with the state-of-the-art methods.

    

### [[2112.07464] Efficient differentiable quadratic programming layers: an ADMM approach](http://arxiv.org/abs/2112.07464)


  Recent advances in neural-network architecture allow for seamless integration
of convex optimization problems as differentiable layers in an end-to-end
trainable neural network. Integrating medium and large scale quadratic programs
into a deep neural network architecture, however, is challenging as solving
quadratic programs exactly by interior-point methods has worst-case cubic
complexity in the number of variables. In this paper, we present an alternative
network layer architecture based on the alternating direction method of
multipliers (ADMM) that is capable of scaling to problems with a moderately
large number of variables. Backward differentiation is performed by implicit
differentiation of the residual map of a modified fixed-point iteration.
Simulated results demonstrate the computational advantage of the ADMM layer,
which for medium scaled problems is approximately an order of magnitude faster
than the OptNet quadratic programming layer. Furthermore, our novel
backward-pass routine is efficient, from both a memory and computation
standpoint, in comparison to the standard approach based on unrolled
differentiation or implicit differentiation of the KKT optimality conditions.
We conclude with examples from portfolio optimization in the integrated
prediction and optimization paradigm.

    

### [[2112.07485] Pruning Coherent Integrated Photonic Neural Networks Using the Lottery Ticket Hypothesis](http://arxiv.org/abs/2112.07485)


  Singular-value-decomposition-based coherent integrated photonic neural
networks (SC-IPNNs) have a large footprint, suffer from high static power
consumption for training and inference, and cannot be pruned using conventional
DNN pruning techniques. We leverage the lottery ticket hypothesis to propose
the first hardware-aware pruning method for SC-IPNNs that alleviates these
challenges by minimizing the number of weight parameters. We prune a
multi-layer perceptron-based SC-IPNN and show that up to 89% of the phase
angles, which correspond to weight parameters in SC-IPNNs, can be pruned with a
negligible accuracy loss (smaller than 5%) while reducing the static power
consumption by up to 86%.

    

### [[2112.07508] Anti-Money Laundering Alert Optimization Using Machine Learning with Graphs](http://arxiv.org/abs/2112.07508)


  Money laundering is a global problem that concerns legitimizing proceeds from
serious felonies (1.7-4 trillion euros annually) such as drug dealing, human
trafficking, or corruption. The anti-money laundering systems deployed by
financial institutions typically comprise rules aligned with regulatory
frameworks. Human investigators review the alerts and report suspicious cases.
Such systems suffer from high false-positive rates, undermining their
effectiveness and resulting in high operational costs. We propose a machine
learning triage model, which complements the rule-based system and learns to
predict the risk of an alert accurately. Our model uses both entity-centric
engineered features and attributes characterizing inter-entity relations in the
form of graph-based features. We leverage time windows to construct the dynamic
graph, optimizing for time and space efficiency. We validate our model on a
real-world banking dataset and show how the triage model can reduce the number
of false positives by 80% while detecting over 90% of true positives. In this
way, our model can significantly improve anti-money laundering operations.

    

### [[2112.07512] Adversarial Examples for Extreme Multilabel Text Classification](http://arxiv.org/abs/2112.07512)


  Extreme Multilabel Text Classification (XMTC) is a text classification
problem in which, (i) the output space is extremely large, (ii) each data point
may have multiple positive labels, and (iii) the data follows a strongly
imbalanced distribution. With applications in recommendation systems and
automatic tagging of web-scale documents, the research on XMTC has been focused
on improving prediction accuracy and dealing with imbalanced data. However, the
robustness of deep learning based XMTC models against adversarial examples has
been largely underexplored.
In this paper, we investigate the behaviour of XMTC models under adversarial
attacks. To this end, first, we define adversarial attacks in multilabel text
classification problems. We categorize attacking multilabel text classifiers as
(a) positive-targeted, where the target positive label should fall out of top-k
predicted labels, and (b) negative-targeted, where the target negative label
should be among the top-k predicted labels. Then, by experiments on APLC-XLNet
and AttentionXML, we show that XMTC models are highly vulnerable to
positive-targeted attacks but more robust to negative-targeted ones.
Furthermore, our experiments show that the success rate of positive-targeted
adversarial attacks has an imbalanced distribution. More precisely, tail
classes are highly vulnerable to adversarial attacks for which an attacker can
generate adversarial samples with high similarity to the actual data-points. To
overcome this problem, we explore the effect of rebalanced loss functions in
XMTC where not only do they increase accuracy on tail classes, but they also
improve the robustness of these classes against adversarial attacks. The code
for our experiments is available at this https URL


### [[2112.07517] A Style and Semantic Memory Mechanism for Domain Generalization](http://arxiv.org/abs/2112.07517)


  Mainstream state-of-the-art domain generalization algorithms tend to
prioritize the assumption on semantic invariance across domains. Meanwhile, the
inherent intra-domain style invariance is usually underappreciated and put on
the shelf. In this paper, we reveal that leveraging intra-domain style
invariance is also of pivotal importance in improving the efficiency of domain
generalization. We verify that it is critical for the network to be informative
on what domain features are invariant and shared among instances, so that the
network sharpens its understanding and improves its semantic discriminative
ability. Correspondingly, we also propose a novel "jury" mechanism, which is
particularly effective in learning useful semantic feature commonalities among
domains. Our complete model called STEAM can be interpreted as a novel
probabilistic graphical model, for which the implementation requires convenient
constructions of two kinds of memory banks: semantic feature bank and style
feature bank. Empirical results show that our proposed framework surpasses the
state-of-the-art methods by clear margins.

    

### [[2112.07528] $n$-CPS: Generalising Cross Pseudo Supervision to $n$ networks for Semi-Supervised Semantic Segmentation](http://arxiv.org/abs/2112.07528)


  We present $n$-CPS - a generalisation of the recent state-of-the-art cross
pseudo supervision (CPS) approach for the task of semi-supervised semantic
segmentation. In $n$-CPS, there are $n$ simultaneously trained subnetworks that
learn from each other through one-hot encoding perturbation and consistency
regularisation. We also show that ensembling techniques applied to subnetworks
outputs can significantly improve the performance. To the best of our
knowledge, $n$-CPS paired with CutMix outperforms CPS and sets the new
state-of-the-art for Pascal VOC 2012 with (1/16, 1/8, 1/4, and 1/2 supervised
regimes) and Cityscapes (1/16 supervised).

    

### [[2112.07529] Improving COVID-19 CXR Detection with Synthetic Data Augmentation](http://arxiv.org/abs/2112.07529)


  Since the beginning of the COVID-19 pandemic, researchers have developed deep
learning models to classify COVID-19 induced pneumonia. As with many medical
imaging tasks, the quality and quantity of the available data is often limited.
In this work we train a deep learning model on publicly available COVID-19
image data and evaluate the model on local hospital chest X-ray data. The data
has been reviewed and labeled by two radiologists to ensure a high quality
estimation of the generalization capabilities of the model. Furthermore, we are
using a Generative Adversarial Network to generate synthetic X-ray images based
on this data. Our results show that using those synthetic images for data
augmentation can improve the model's performance significantly. This can be a
promising approach for many sparse data domains.

    

### [[2112.07535] Scientific Discovery and the Cost of Measurement -- Balancing Information and Cost in Reinforcement Learning](http://arxiv.org/abs/2112.07535)


  The use of reinforcement learning (RL) in scientific applications, such as
materials design and automated chemistry, is increasing. A major challenge,
however, lies in fact that measuring the state of the system is often costly
and time consuming in scientific applications, whereas policy learning with RL
requires a measurement after each time step. In this work, we make the
measurement costs explicit in the form of a costed reward and propose a
framework that enables off-the-shelf deep RL algorithms to learn a policy for
both selecting actions and determining whether or not to measure the current
state of the system at each time step. In this way, the agents learn to balance
the need for information with the cost of information. Our results show that
when trained under this regime, the Dueling DQN and PPO agents can learn
optimal action policies whilst making up to 50\% fewer state measurements, and
recurrent neural networks can produce a greater than 50\% reduction in
measurements. We postulate the these reduction can help to lower the barrier to
applying RL to real-world scientific applications.

    

### [[2112.07544] Modeling Strong and Human-Like Gameplay with KL-Regularized Search](http://arxiv.org/abs/2112.07544)


  We consider the task of building strong but human-like policies in
multi-agent decision-making problems, given examples of human behavior.
Imitation learning is effective at predicting human actions but may not match
the strength of expert humans, while self-play learning and search techniques
(e.g. AlphaZero) lead to strong performance but may produce policies that are
difficult for humans to understand and coordinate with. We show in chess and Go
that regularizing search policies based on the KL divergence from an
imitation-learned policy by applying Monte Carlo tree search produces policies
that have higher human prediction accuracy and are stronger than the imitation
policy. We then introduce a novel regret minimization algorithm that is
regularized based on the KL divergence from an imitation-learned policy, and
show that applying this algorithm to no-press Diplomacy yields a policy that
maintains the same human prediction accuracy as imitation learning while being
substantially stronger.

    

### [[2112.07555] Classification of histopathology images using ConvNets to detect Lupus Nephritis](http://arxiv.org/abs/2112.07555)


  Systemic lupus erythematosus (SLE) is an autoimmune disease in which the
immune system of the patient starts attacking healthy tissues of the body.
Lupus Nephritis (LN) refers to the inflammation of kidney tissues resulting in
renal failure due to these attacks. The International Society of
Nephrology/Renal Pathology Society (ISN/RPS) has released a classification
system based on various patterns observed during renal injury in SLE.
Traditional methods require meticulous pathological assessment of the renal
biopsy and are time-consuming. Recently, computational techniques have helped
to alleviate this issue by using virtual microscopy or Whole Slide Imaging
(WSI). With the use of deep learning and modern computer vision techniques, we
propose a pipeline that is able to automate the process of 1) detection of
various glomeruli patterns present in these whole slide images and 2)
classification of each image using the extracted glomeruli features.

    

### [[2112.07569] Cooperation for Scalable Supervision of Autonomy in Mixed Traffic](http://arxiv.org/abs/2112.07569)


  Improvements in autonomy offer the potential for positive outcomes in a
number of domains, yet guaranteeing their safe deployment is difficult. This
work investigates how humans can intelligently supervise agents to achieve some
level of safety even when performance guarantees are elusive. The motivating
research question is: In safety-critical settings, can we avoid the need to
have one human supervise one machine at all times? The paper formalizes this
'scaling supervision' problem, and investigates its application to the
safety-critical context of autonomous vehicles (AVs) merging into traffic. It
proposes a conservative, reachability-based method to reduce the burden on the
AVs' human supervisors, which allows for the establishment of high-confidence
upper bounds on the supervision requirements in this setting. Order statistics
and traffic simulations with deep reinforcement learning show analytically and
numerically that teaming of AVs enables supervision time sublinear in AV
adoption. A key takeaway is that, despite present imperfections of AVs,
supervision becomes more tractable as AVs are deployed en masse. While this
work focuses on AVs, the scalable supervision framework is relevant to a
broader array of autonomous control challenges.

    

### [[2112.07571] Epigenomic language models powered by Cerebras](http://arxiv.org/abs/2112.07571)


  Large scale self-supervised pre-training of Transformer language models has
advanced the field of Natural Language Processing and shown promise in
cross-application to the biological `languages' of proteins and DNA. Learning
effective representations of DNA sequences using large genomic sequence
corpuses may accelerate the development of models of gene regulation and
function through transfer learning. However, to accurately model cell
type-specific gene regulation and function, it is necessary to consider not
only the information contained in DNA nucleotide sequences, which is mostly
invariant between cell types, but also how the local chemical and structural
`epigenetic state' of chromosomes varies between cell types. Here, we introduce
a Bidirectional Encoder Representations from Transformers (BERT) model that
learns representations based on both DNA sequence and paired epigenetic state
inputs, which we call Epigenomic BERT (or EBERT). We pre-train EBERT with a
masked language model objective across the entire human genome and across 127
cell types. Training this complex model with a previously prohibitively large
dataset was made possible for the first time by a partnership with Cerebras
Systems, whose CS-1 system powered all pre-training experiments. We show
EBERT's transfer learning potential by demonstrating strong performance on a
cell type-specific transcription factor binding prediction task. Our fine-tuned
model exceeds state of the art performance on 4 of 13 evaluation datasets from
ENCODE-DREAM benchmarks and earns an overall rank of 3rd on the challenge
leaderboard. We explore how the inclusion of epigenetic data and task specific
feature augmentation impact transfer learning performance.

    

### [[2112.07574] M3E2: Multi-gate Mixture-of-experts for Multi-treatment Effect Estimation](http://arxiv.org/abs/2112.07574)


  This work proposes the M3E2, a multi-task learning neural network model to
estimate the effect of multiple treatments. In contrast to existing methods,
M3E2 is robust to multiple treatment effects applied simultaneously to the same
unit, continuous and binary treatments, and many covariates. We compared M3E2
with three baselines in three synthetic benchmark datasets: two with multiple
treatments and one with one treatment. Our analysis showed that our method has
superior performance, making more assertive estimations of the true treatment
effects. The code is available at this http URL.

    

### [[2112.07575] Robust Graph Neural Networks via Probabilistic Lipschitz Constraints](http://arxiv.org/abs/2112.07575)


  Graph neural networks (GNNs) have recently been demonstrated to perform well
on a variety of network-based tasks such as decentralized control and resource
allocation, and provide computationally efficient methods for these tasks which
have traditionally been challenging in that regard. However, like many
neural-network based systems, GNNs are susceptible to shifts and perturbations
on their inputs, which can include both node attributes and graph structure. In
order to make them more useful for real-world applications, it is important to
ensure their robustness post-deployment. Motivated by controlling the Lipschitz
constant of GNN filters with respect to the node attributes, we propose to
constrain the frequency response of the GNN's filter banks. We extend this
formulation to the dynamic graph setting using a continuous frequency response
constraint, and solve a relaxed variant of the problem via the scenario
approach. This allows for the use of the same computationally efficient
algorithm on sampled constraints, which provides PAC-style guarantees on the
stability of the GNN using results in scenario optimization. We also highlight
an important connection between this setup and GNN stability to graph
perturbations, and provide experimental results which demonstrate the efficacy
and broadness of our approach.

    

### [[2112.07611] Speeding up Learning Quantum States through Group Equivariant Convolutional Quantum Ans{ä}tze](http://arxiv.org/abs/2112.07611)


  We develop a theoretical framework for $S_n$-equivariant quantum
convolutional circuits, building on and significantly generalizing Jordan's
Permutational Quantum Computing (PQC) formalism. We show that quantum circuits
are a natural choice for Fourier space neural architectures affording a
super-exponential speedup in computing the matrix elements of $S_n$-Fourier
coefficients compared to the best known classical Fast Fourier Transform (FFT)
over the symmetric group. In particular, we utilize the Okounkov-Vershik
approach to prove Harrow's statement (Ph.D. Thesis 2005 p.160) on the
equivalence between $\operatorname{SU}(d)$- and $S_n$-irrep bases and to
establish the $S_n$-equivariant Convolutional Quantum Alternating Ans{ä}tze
($S_n$-CQA) using Young-Jucys-Murphy (YJM) elements. We prove that $S_n$-CQA
are dense, thus expressible within each $S_n$-irrep block, which may serve as a
universal model for potential future quantum machine learning and optimization
applications. Our method provides another way to prove the universality of
Quantum Approximate Optimization Algorithm (QAOA), from the
representation-theoretical point of view. Our framework can be naturally
applied to a wide array of problems with global $\operatorname{SU}(d)$
symmetry. We present numerical simulations to showcase the effectiveness of the
ans{ä}tze to find the sign structure of the ground state of the $J_1$--$J_2$
antiferromagnetic Heisenberg model on the rectangular and Kagome lattices. Our
work identifies quantum advantage for a specific machine learning problem, and
provides the first application of the celebrated Okounkov-Vershik's
representation theory to machine learning and quantum physics.

    

### [[2112.07615] Cold Item Integration in Deep Hybrid Recommenders via Tunable Stochastic Gates](http://arxiv.org/abs/2112.07615)


  A major challenge in collaborative filtering methods is how to produce
recommendations for cold items (items with no ratings), or integrate cold item
into an existing catalog. Over the years, a variety of hybrid recommendation
models have been proposed to address this problem by utilizing items' metadata
and content along with their ratings or usage patterns. In this work, we wish
to revisit the cold start problem in order to draw attention to an overlooked
challenge: the ability to integrate and balance between (regular) warm items
and completely cold items. In this case, two different challenges arise: (1)
preserving high quality performance on warm items, while (2) learning to
promote cold items to relevant users. First, we show that these two objectives
are in fact conflicting, and the balance between them depends on the business
needs and the application at hand. Next, we propose a novel hybrid
recommendation algorithm that bridges these two conflicting objectives and
enables a harmonized balance between preserving high accuracy for warm items
while effectively promoting completely cold items. We demonstrate the
effectiveness of the proposed algorithm on movies, apps, and articles
recommendations, and provide an empirical analysis of the cold-warm trade-off.

    

### [[2112.07616] DiPS: Differentiable Policy for Sketching in Recommender Systems](http://arxiv.org/abs/2112.07616)


  In sequential recommender system applications, it is important to develop
models that can capture users' evolving interest over time to successfully
recommend future items that they are likely to interact with. For users with
long histories, typical models based on recurrent neural networks tend to
forget important items in the distant past. Recent works have shown that
storing a small sketch of past items can improve sequential recommendation
tasks. However, these works all rely on static sketching policies, i.e.,
heuristics to select items to keep in the sketch, which are not necessarily
optimal and cannot improve over time with more training data. In this paper, we
propose a differentiable policy for sketching (DiPS), a framework that learns a
data-driven sketching policy in an end-to-end manner together with the
recommender system model to explicitly maximize recommendation quality in the
future. We also propose an approximate estimator of the gradient for optimizing
the sketching algorithm parameters that is computationally efficient. We verify
the effectiveness of DiPS on real-world datasets under various practical
settings and show that it requires up to $50\%$ fewer sketch items to reach the
same predictive quality than existing sketching policies.

    

### [[2112.07617] A cross-domain recommender system using deep coupled autoencoders](http://arxiv.org/abs/2112.07617)


  Long-standing data sparsity and cold-start constitute thorny and perplexing
problems for the recommendation systems. Cross-domain recommendation as a
domain adaptation framework has been utilized to efficiently address these
challenging issues, by exploiting information from multiple domains. In this
study, an item-level relevance cross-domain recommendation task is explored,
where two related domains, that is, the source and the target domain contain
common items without sharing sensitive information regarding the users'
behavior, and thus avoiding the leak of user privacy. In light of this
scenario, two novel coupled autoencoder-based deep learning methods are
proposed for cross-domain recommendation. The first method aims to
simultaneously learn a pair of autoencoders in order to reveal the intrinsic
representations of the items in the source and target domains, along with a
coupled mapping function to model the non-linear relationships between these
representations, thus transferring beneficial information from the source to
the target domain. The second method is derived based on a new joint
regularized optimization problem, which employs two autoencoders to generate in
a deep and non-linear manner the user and item-latent factors, while at the
same time a data-driven function is learnt to map the item-latent factors
across domains. Extensive numerical experiments on two publicly available
benchmark datasets are conducted illustrating the superior performance of our
proposed methods compared to several state-of-the-art cross-domain
recommendation frameworks.

    

### [[2112.07618] Robust Information Retrieval for False Claims with Distracting Entities In Fact Extraction and Verification](http://arxiv.org/abs/2112.07618)


  Accurate evidence retrieval is essential for automated fact checking. Little
previous research has focused on the differences between true and false claims
and how they affect evidence retrieval. This paper shows that, compared with
true claims, false claims more frequently contain irrelevant entities which can
distract evidence retrieval model. A BERT-based retrieval model made more
mistakes in retrieving refuting evidence for false claims than supporting
evidence for true claims. When tested with adversarial false claims
(synthetically generated) containing irrelevant entities, the recall of the
retrieval model is significantly lower than that for original claims. These
results suggest that the vanilla BERT-based retrieval model is not robust to
irrelevant entities in the false claims. By augmenting the training data with
synthetic false claims containing irrelevant entities, the trained model
achieved higher evidence recall, including that of false claims with irrelevant
entities. In addition, using separate models to retrieve refuting and
supporting evidence and then aggregating them can also increase the evidence
recall, including that of false claims with irrelevant entities. These results
suggest that we can increase the BERT-based retrieval model's robustness to
false claims with irrelevant entities via data augmentation and model ensemble.

    

### [[2112.07620] Tree-based Focused Web Crawling with Reinforcement Learning](http://arxiv.org/abs/2112.07620)


  A focused crawler aims at discovering as many web pages relevant to a target
topic as possible, while avoiding irrelevant ones; i.e. maximizing the harvest
rate. Reinforcement Learning (RL) has been utilized to optimize the crawling
process, yet it deals with huge state and action spaces, which can constitute a
serious challenge. In this paper, we propose TRES, an end-to-end RL-empowered
framework for focused crawling. Unlike other approaches, we properly model a
crawling environment as a Markov Decision Process, by representing the state as
a subgraph of the Web and actions as its expansion edges. TRES adopts a keyword
expansion strategy based on the cosine similarity of keyword embeddings. To
learn a reward function, we propose a deep neural network, called KwBiLSTM,
leveraging the discovered keywords. To reduce the time complexity of selecting
a best action, we propose Tree-Frontier, a two-fold decision tree, which also
speeds up training by discretizing the state and action spaces. Experimentally,
we show that TRES outperforms state-of-the-art methods in terms of harvest rate
by at least 58%, while it has competitive results in the domain maximization.
Our implementation code can be found on this https URL.

    

### [[2112.07621] Re-ranking With Constraints on Diversified Exposures for Homepage Recommender System](http://arxiv.org/abs/2112.07621)


  The homepage recommendation on most E-commerce applications places items in a
hierarchical manner, where different channels display items in different
styles. Existing algorithms usually optimize the performance of a single
channel. So designing the model to achieve the optimal recommendation list
which maximize the Click-Through Rate (CTR) of whole homepage is a challenge
problem. Other than the accuracy objective, display diversity on the homepage
is also important since homogeneous display usually hurts user experience. In
this paper, we propose a two-stage architecture of the homepage recommendation
system. In the first stage, we develop efficient algorithms for recommending
items to proper channels while maintaining diversity. The two methods can be
combined: user-channel-item predictive model with diversity constraint. In the
second stage, we provide an ordered list of items in each channel. Existing
re-ranking models are hard to describe the mutual influence between items in
both intra-channel and inter-channel. Therefore, we propose a Deep \&
Hierarchical Attention Network Re-ranking (DHANR) model for homepage
recommender systems. The Hierarchical Attention Network consists of an item
encoder, an item-level attention layer, a channel encoder and a channel-level
attention layer. Our method achieves a significant improvement in terms of
precision, intra-list average distance(ILAD) and channel-wise Precision@k in
offline experiments and in terms of CTR and ILAD in our online systems.

    

### [[2112.07628] Training Multi-Layer Over-Parametrized Neural Network in Subquadratic Time](http://arxiv.org/abs/2112.07628)


  We consider the problem of training a multi-layer over-parametrized neural
networks to minimize the empirical risk induced by a loss function. In the
typical setting of over-parametrization, the network width $m$ is much larger
than the data dimension $d$ and number of training samples $n$
($m=\mathrm{poly}(n,d)$), which induces a prohibitive large weight matrix $W\in
\mathbb{R}^{m\times m}$ per layer. Naively, one has to pay $O(m^2)$ time to
read the weight matrix and evaluate the neural network function in both forward
and backward computation. In this work, we show how to reduce the training cost
per iteration, specifically, we propose a framework that uses $m^2$ cost only
in the initialization phase and achieves a truly subquadratic cost per
iteration in terms of $m$, i.e., $m^{2-\Omega(1)}$ per iteration.
To obtain this result, we make use of various techniques, including a shifted
ReLU-based sparsifier, a lazy low rank maintenance data structure, fast
rectangular matrix multiplication, tensor-based sketching techniques and
preconditioning.

    

### [[2112.07640] How and Why to Manipulate Your Own Agent](http://arxiv.org/abs/2112.07640)


  We consider strategic settings where several users engage in a repeated
online interaction, assisted by regret-minimizing agents that repeatedly play a
"game" on their behalf. We study the dynamics and average outcomes of the
repeated game of the agents, and view it as inducing a meta-game between the
users. Our main focus is on whether users can benefit in this meta-game from
"manipulating" their own agent by mis-reporting their parameters to it. We
formally define this "user-agent meta-game" model for general games, discuss
its properties under different notions of convergence of the dynamics of the
automated agents and analyze the equilibria induced on the users in 2x2 games
in which the dynamics converge to a single equilibrium.

    

### [[2112.07648] On the Use of External Data for Spoken Named Entity Recognition](http://arxiv.org/abs/2112.07648)


  Spoken language understanding (SLU) tasks involve mapping from speech audio
signals to semantic labels. Given the complexity of such tasks, good
performance might be expected to require large labeled datasets, which are
difficult to collect for each new task and domain. However, recent advances in
self-supervised speech representations have made it feasible to consider
learning SLU models with limited labeled data. In this work we focus on
low-resource spoken named entity recognition (NER) and address the question:
Beyond self-supervised pre-training, how can we use external speech and/or text
data that are not annotated for the task? We draw on a variety of approaches,
including self-training, knowledge distillation, and transfer learning, and
consider their applicability to both end-to-end models and pipeline (speech
recognition followed by text NER model) approaches. We find that several of
these approaches improve performance in resource-constrained settings beyond
the benefits from pre-trained representations alone. Compared to prior work, we
find improved F1 scores of up to 16%. While the best baseline model is a
pipeline approach, the best performance when using external data is ultimately
achieved by an end-to-end model. We provide detailed comparisons and analyses,
showing for example that end-to-end models are able to focus on the more
NER-specific words.

    

### [[2112.07658] AdaViT: Adaptive Tokens for Efficient Vision Transformer](http://arxiv.org/abs/2112.07658)


  We introduce AdaViT, a method that adaptively adjusts the inference cost of
vision transformer (ViT) for images of different complexity. AdaViT achieves
this by automatically reducing the number of tokens in vision transformers that
are processed in the network as inference proceeds. We reformulate Adaptive
Computation Time (ACT) for this task, extending halting to discard redundant
spatial tokens. The appealing architectural properties of vision transformers
enables our adaptive token reduction mechanism to speed up inference without
modifying the network architecture or inference hardware. We demonstrate that
AdaViT requires no extra parameters or sub-network for halting, as we base the
learning of adaptive halting on the original network parameters. We further
introduce distributional prior regularization that stabilizes training compared
to prior ACT approaches. On the image classification task (ImageNet1K), we show
that our proposed AdaViT yields high efficacy in filtering informative spatial
features and cutting down on the overall compute. The proposed method improves
the throughput of DeiT-Tiny by 62% and DeiT-Small by 38% with only 0.3%
accuracy drop, outperforming prior art by a large margin.

    

### [[2112.07662] Out-of-Distribution Detection without Class Labels](http://arxiv.org/abs/2112.07662)


  Anomaly detection methods identify samples that deviate from the normal
behavior of the dataset. It is typically tackled either for training sets
containing normal data from multiple labeled classes or a single unlabeled
class. Current methods struggle when faced with training data consisting of
multiple classes but no labels. In this work, we first discover that
classifiers learned by self-supervised image clustering methods provide a
strong baseline for anomaly detection on unlabeled multi-class datasets.
Perhaps surprisingly, we find that initializing clustering methods with
pre-trained features does not improve over their self-supervised counterparts.
This is due to the phenomenon of catastrophic forgetting. Instead, we suggest a
two stage approach. We first cluster images using self-supervised methods and
obtain a cluster label for every image. We use the cluster labels as "pseudo
supervision" for out-of-distribution (OOD) methods. Specifically, we finetune
pretrained features on the task of classifying images by their cluster labels.
We provide extensive analyses of our method and demonstrate the necessity of
our two-stage approach. We evaluate it against the state-of-the-art
self-supervised and pretrained methods and demonstrate superior performance.

    

### [[1909.06296] Bayesian parameter estimation using conditional variational autoencoders for gravitational-wave astronomy](http://arxiv.org/abs/1909.06296)


  Gravitational wave (GW) detection is now commonplace and as the sensitivity
of the global network of GW detectors improves, we will observe
$\mathcal{O}(100)$s of transient GW events per year. The current methods used
to estimate their source parameters employ optimally sensitive but
computationally costly Bayesian inference approaches where typical analyses
have taken between 6 hours and 5 days. For binary neutron star and neutron star
black hole systems prompt counterpart electromagnetic (EM) signatures are
expected on timescales of 1 second -- 1 minute and the current fastest method
for alerting EM follow-up observers, can provide estimates in $\mathcal{O}(1)$
minute, on a limited range of key source parameters. Here we show that a
conditional variational autoencoder pre-trained on binary black hole signals
can return Bayesian posterior probability estimates. The training procedure
need only be performed once for a given prior parameter space and the resulting
trained machine can then generate samples describing the posterior distribution
$\sim 6$ orders of magnitude faster than existing techniques.

    

### [[1911.00569] Mitigating the Effects of Non-Identifiability on Inference for Bayesian Neural Networks with Latent Variables](http://arxiv.org/abs/1911.00569)


  Bayesian Neural Networks with Latent Variables (BNN+LVs) capture predictive
uncertainty by explicitly modeling model uncertainty (via priors on network
weights) and environmental stochasticity (via a latent input noise variable).
In this work, we first show that BNN+LV suffers from a serious form of
non-identifiability: explanatory power can be transferred between the model
parameters and latent variables while fitting the data equally well. We
demonstrate that as a result, in the limit of infinite data, the posterior mode
over the network weights and latent variables is asymptotically biased away
from the ground-truth. Due to this asymptotic bias, traditional inference
methods may in practice yield parameters that generalize poorly and misestimate
uncertainty. Next, we develop a novel inference procedure that explicitly
mitigates the effects of likelihood non-identifiability during training and
yields high-quality predictions as well as uncertainty estimates. We
demonstrate that our inference method improves upon benchmark methods across a
range of synthetic and real data-sets.

    

### [[1911.06722] Bayesian nonparametric discontinuity design](http://arxiv.org/abs/1911.06722)


  Quasi-experimental research designs, such as regression discontinuity and
interrupted time series, allow for causal inference in the absence of a
randomized controlled trial, at the cost of additional assumptions. In this
paper, we provide a framework for discontinuity-based designs using Bayesian
model comparison and Gaussian process regression, which we refer to as
'Bayesian nonparametric discontinuity design', or BNDD for short. BNDD
addresses the two major shortcomings in most implementations of such designs:
overconfidence due to implicit conditioning on the alleged effect, and model
misspecification due to reliance on overly simplistic regression models. With
the appropriate Gaussian process covariance function, our approach can detect
discontinuities of any order, and in spectral features. We demonstrate the
usage of BNDD in simulations, and apply the framework to determine the effect
of running for political positions on longevity, of the effect of an alleged
historical phantom border in the Netherlands on Dutch voting behaviour, and of
Kundalini Yoga meditation on heart rate.

    

### [[2004.06493] Solving Newton's Equations of Motion with Large Timesteps using Recurrent Neural Networks based Operators](http://arxiv.org/abs/2004.06493)


  Classical molecular dynamics simulations are based on solving Newton's
equations of motion. Using a small timestep, numerical integrators such as
Verlet generate trajectories of particles as solutions to Newton's equations.
We introduce operators derived using recurrent neural networks that accurately
solve Newton's equations utilizing sequences of past trajectory data, and
produce energy-conserving dynamics of particles using timesteps up to 4000
times larger compared to the Verlet timestep. We demonstrate significant
speedup in many example problems including 3D systems of up to 16 particles.

    

### [[2004.07543] Classify and Generate: Using Classification Latent Space Representations for Image Generations](http://arxiv.org/abs/2004.07543)


  Utilization of classification latent space information for downstream
reconstruction and generation is an intriguing and a relatively unexplored
area. In general, discriminative representations are rich in class-specific
features but are too sparse for reconstruction, whereas, in autoencoders the
representations are dense but have limited indistinguishable class-specific
features, making them less suitable for classification. In this work, we
propose a discriminative modeling framework that employs manipulated supervised
latent representations to reconstruct and generate new samples belonging to a
given class. Unlike generative modeling approaches such as GANs and VAEs that
aim to model the data manifold distribution, Representation based Generations
(ReGene) directly represent the given data manifold in the classification
space. Such supervised representations, under certain constraints, allow for
reconstructions and controlled generations using an appropriate decoder without
enforcing any prior distribution. Theoretically, given a class, we show that
these representations when smartly manipulated using convex combinations retain
the same class label. Furthermore, they also lead to the novel generation of
visually realistic images. Extensive experiments on datasets of varying
resolutions demonstrate that ReGene has higher classification accuracy than
existing conditional generative models while being competitive in terms of FID.

    

### [[2005.12386] Customized Graph Neural Networks](http://arxiv.org/abs/2005.12386)


  Recently, Graph Neural Networks (GNNs) have greatly advanced the task of
graph classification. Typically, we first build a unified GNN model with graphs
in a given training set and then use this unified model to predict labels of
all the unseen graphs in the test set. However, graphs in the same dataset
often have dramatically distinct structures, which indicates that a unified
model may be sub-optimal given an individual graph. Therefore, in this paper,
we aim to develop customized graph neural networks for graph classification.
Specifically, we propose a novel customized graph neural network framework,
i.e., Customized-GNN. Given a graph sample, Customized-GNN can generate a
sample-specific model for this graph based on its structure. Meanwhile, the
proposed framework is very general that can be applied to numerous existing
graph neural network models. Comprehensive experiments on various graph
classification benchmarks demonstrate the effectiveness of the proposed
framework.

    

### [[2006.10175] An Empirical Comparison of GANs and Normalizing Flows for Density Estimation](http://arxiv.org/abs/2006.10175)


  Generative adversarial networks (GANs) and normalizing flows are both
approaches to density estimation that use deep neural networks to transform
samples from an uninformative prior distribution to an approximation of the
data distribution. There is great interest in both for general-purpose
statistical modeling, but the two approaches have seldom been compared to each
other for modeling non-image data. The difficulty of computing likelihoods with
GANs, which are implicit models, makes conducting such a comparison
challenging. We work around this difficulty by considering several
low-dimensional synthetic datasets. An extensive grid search over GAN
architectures, hyperparameters, and training procedures suggests that no GAN is
capable of modeling our simple low-dimensional data well, a task we view as a
prerequisite for an approach to be considered suitable for general-purpose
statistical modeling. Several normalizing flows, on the other hand, excelled at
these tasks, even substantially outperforming WGAN in terms of Wasserstein
distance -- the metric that WGAN alone targets. Scientists and other
practitioners should be wary of relying on WGAN for applications that require
accurate density estimation.

    

### [[2006.16375] Improving Calibration through the Relationship with Adversarial Robustness](http://arxiv.org/abs/2006.16375)


  Neural networks lack adversarial robustness, i.e., they are vulnerable to
adversarial examples that through small perturbations to inputs cause incorrect
predictions. Further, trust is undermined when models give miscalibrated
predictions, i.e., the predicted probability is not a good indicator of how
much we should trust our model. In this paper, we study the connection between
adversarial robustness and calibration and find that the inputs for which the
model is sensitive to small perturbations (are easily attacked) are more likely
to have poorly calibrated predictions. Based on this insight, we examine if
calibration can be improved by addressing those adversarially unrobust inputs.
To this end, we propose Adversarial Robustness based Adaptive Label Smoothing
(AR-AdaLS) that integrates the correlations of adversarial robustness and
calibration into training by adaptively softening labels for an example based
on how easily it can be attacked by an adversary. We find that our method,
taking the adversarial robustness of the in-distribution data into
consideration, leads to better calibration over the model even under
distributional shifts. In addition, AR-AdaLS can also be applied to an ensemble
model to further improve model calibration.

    

### [[2007.03797] Personalized Cross-Silo Federated Learning on Non-IID Data](http://arxiv.org/abs/2007.03797)


  Non-IID data present a tough challenge for federated learning. In this paper,
we explore a novel idea of facilitating pairwise collaborations between clients
with similar data. We propose FedAMP, a new method employing federated
attentive message passing to facilitate similar clients to collaborate more. We
establish the convergence of FedAMP for both convex and non-convex models, and
propose a heuristic method to further improve the performance of FedAMP when
clients adopt deep neural networks as personalized models. Our extensive
experiments on benchmark data sets demonstrate the superior performance of the
proposed methods.

    

### [[2007.08031] Optimal Coresets for Gaussian Kernel Density Estimates](http://arxiv.org/abs/2007.08031)


  Given a point set $P\subset \mathbb{R}^d$, the kernel density estimate of $P$
is defined as \[ \overline{\mathcal{G}}_P(x) =
\frac{1}{\left|P\right|}\sum_{p\in P}e^{-\left\lVert x-p \right\rVert^2} \] for
any $x\in\mathbb{R}^d$. We study how to construct a small subset $Q$ of $P$
such that the kernel density estimate of $P$ is approximated by the kernel
density estimate of $Q$. This subset $Q$ is called a coreset. The main
technique in this work is constructing a $\pm 1$ coloring on the point set $P$
by discrepancy theory and we leverage Banaszczyk's Theorem. When $d>1$ is a
constant, our construction gives a coreset of size
$O\left(\frac{1}{\varepsilon}\right)$ as opposed to the best-known result of
$O\left(\frac{1}{\varepsilon}\sqrt{\log\frac{1}{\varepsilon}}\right)$. It is
the first result to give a breakthrough on the barrier of $\sqrt{\log}$ factor
even when $d=2$.

    

### [[2007.10675] Trade-off on Sim2Real Learning: Real-world Learning Faster than Simulations](http://arxiv.org/abs/2007.10675)


  Deep Reinforcement Learning (DRL) experiments are commonly performed in
simulated environments due to the tremendous training sample demands from deep
neural networks. In contrast, model-based Bayesian Learning allows a robot to
learn good policies within a few trials in the real world. Although it takes
fewer iterations, Bayesian methods pay a relatively higher computational cost
per trial, and the advantage of such methods is strongly tied to dimensionality
and noise. In here, we compare a Deep Bayesian Learning algorithm with a
model-free DRL algorithm while analyzing our results collected from both
simulations and real-world experiments. While considering Sim and Real
learning, our experiments show that the sample-efficient Deep Bayesian RL
performance is better than DRL even when computation time (as opposed to number
of iterations) is taken in consideration. Additionally, the difference in
computation time between Deep Bayesian RL performed in simulation and in
experiments point to a viable path to traverse the reality gap. We also show
that a mix between Sim and Real does not outperform a purely Real approach,
pointing to the possibility that reality can provide the best prior knowledge
to a Bayesian Learning. Roboticists design and build robots every day, and our
results show that a higher learning efficiency in the real-world will shorten
the time between design and deployment by skipping simulations.

    

### [[2010.01264] HeteroFL: Computation and Communication Efficient Federated Learning for Heterogeneous Clients](http://arxiv.org/abs/2010.01264)


  Federated Learning (FL) is a method of training machine learning models on
private data distributed over a large number of possibly heterogeneous clients
such as mobile phones and IoT devices. In this work, we propose a new federated
learning framework named HeteroFL to address heterogeneous clients equipped
with very different computation and communication capabilities. Our solution
can enable the training of heterogeneous local models with varying computation
complexities and still produce a single global inference model. For the first
time, our method challenges the underlying assumption of existing work that
local models have to share the same architecture as the global model. We
demonstrate several strategies to enhance FL training and conduct extensive
empirical evaluations, including five computation complexity levels of three
model architecture on three datasets. We show that adaptively distributing
subnetworks according to clients' capabilities is both computation and
communication efficient.

    

### [[2010.03957] Transformers for Modeling Physical Systems](http://arxiv.org/abs/2010.03957)


  Transformers are widely used in natural language processing due to their
ability to model longer-term dependencies in text. Although these models
achieve state-of-the-art performance for many language related tasks, their
applicability outside of the natural language processing field has been
minimal. In this work, we propose the use of transformer models for the
prediction of dynamical systems representative of physical phenomena. The use
of Koopman based embeddings provide a unique and powerful method for projecting
any dynamical system into a vector representation which can then be predicted
by a transformer. The proposed model is able to accurately predict various
dynamical systems and outperform classical methods that are commonly used in
the scientific machine learning literature.

    

### [[2010.16310] Multiscale Fractal Analysis on EEG Signals for Music-Induced Emotion Recognition](http://arxiv.org/abs/2010.16310)


  Emotion Recognition from EEG signals has long been researched as it can
assist numerous medical and rehabilitative applications. However, their complex
and noisy structure has proven to be a serious barrier for traditional modeling
methods. In this paper, we employ multifractal analysis to examine the behavior
of EEG signals in terms of presence of fluctuations and the degree of
fragmentation along their major frequency bands, for the task of emotion
recognition. In order to extract emotion-related features we utilize two novel
algorithms for EEG analysis, based on Multiscale Fractal Dimension and
Multifractal Detrended Fluctuation Analysis. The proposed feature extraction
methods perform efficiently, surpassing some widely used baseline features on
the competitive DEAP dataset, indicating that multifractal analysis could serve
as basis for the development of robust models for affective state recognition.

    

### [[2011.09464] Counterfactual Credit Assignment in Model-Free Reinforcement Learning](http://arxiv.org/abs/2011.09464)


  Credit assignment in reinforcement learning is the problem of measuring an
action's influence on future rewards. In particular, this requires separating
skill from luck, i.e. disentangling the effect of an action on rewards from
that of external factors and subsequent actions. To achieve this, we adapt the
notion of counterfactuals from causality theory to a model-free RL setup. The
key idea is to condition value functions on future events, by learning to
extract relevant information from a trajectory. We formulate a family of policy
gradient algorithms that use these future-conditional value functions as
baselines or critics, and show that they are provably low variance. To avoid
the potential bias from conditioning on future information, we constrain the
hindsight information to not contain information about the agent's actions. We
demonstrate the efficacy and validity of our algorithm on a number of
illustrative and challenging problems.

    

### [[2011.12090] AI Discovering a Coordinate System of Chemical Elements: Dual Representation by Variational Autoencoders](http://arxiv.org/abs/2011.12090)


  The periodic table is a fundamental representation of chemical elements that
plays essential theoretical and practical roles. The research article discusses
the experiences of unsupervised training of neural networks to represent
elements on the 2D latent space based on their electron configurations. To
emphasize chemical properties of the elements, the original data of electron
configurations has been realigned towards valence orbitals. Recognizing seven
shells and four subshells, the input data has been arranged as 7x4 images.
Latent space representation has been performed using a convolutional beta
variational autoencoder (beta-VAE). Despite discrete and sparse input data, the
beta-VAE disentangles elements of different periods, blocks, groups, and types.
The unsupervised representation of elements on the latent space reveals
pairwise symmetries of periods and elements related to the invariance of
quantum numbers of corresponding elements. In addition, it isolates outliers
that turned out to be known cases of Madelung's rule violations for lanthanide
and actinide elements. Considering the generative capabilities of beta-VAE, the
supervised machine learning has been set to find out if there are insightful
patterns distinguishing electron configurations between real elements and
decoded artificial ones. Also, the article addresses the capability of dual
representation by autoencoders. Conventionally, autoencoders represent
observations of input data on the latent space. By transposing and duplicating
original input data, it is possible to represent variables on the latent space
which can lead to the discovery of meaningful patterns among input variables.
Applying that unsupervised learning for transposed data of electron
configurations, the order of input variables that has been arranged by the
encoder on the latent space has turned out to exactly match the sequence of
Madelung's rule.

    

### [[2102.08127] Learning curves of generic features maps for realistic datasets with a teacher-student model](http://arxiv.org/abs/2102.08127)


  Teacher-student models provide a framework in which the typical-case
performance of high-dimensional supervised learning can be described in closed
form. The assumptions of Gaussian i.i.d. input data underlying the canonical
teacher-student model may, however, be perceived as too restrictive to capture
the behaviour of realistic data sets. In this paper, we introduce a Gaussian
covariate generalisation of the model where the teacher and student can act on
different spaces, generated with fixed, but generic feature maps. While still
solvable in a closed form, this generalization is able to capture the learning
curves for a broad range of realistic data sets, thus redeeming the potential
of the teacher-student framework. Our contribution is then two-fold: First, we
prove a rigorous formula for the asymptotic training loss and generalisation
error. Second, we present a number of situations where the learning curve of
the model captures the one of a realistic data set learned with kernel
regression and classification, with out-of-the-box feature maps such as random
projections or scattering transforms, or with pre-learned ones - such as the
features learned by training multi-layer neural networks. We discuss both the
power and the limitations of the framework.

    

### [[2103.03864] Learning to Extend Molecular Scaffolds with Structural Motifs](http://arxiv.org/abs/2103.03864)


  Recent advancements in deep learning-based modeling of molecules promise to
accelerate in silico drug discovery. A plethora of generative models is
available, building molecules either atom-by-atom and bond-by-bond or
fragment-by-fragment. However, many drug discovery projects require a fixed
scaffold to be present in the generated molecule, and incorporating that
constraint has only recently been explored. Here, we propose MoLeR, a
graph-based model that naturally supports scaffolds as initial seed of the
generative procedure, which is possible because it is not conditioned on the
generation history. Our experiments show that MoLeR performs comparably to
state-of-the-art methods on unconstrained molecular optimization tasks, and
outperforms them on scaffold-based tasks, while being an order of magnitude
faster to train and sample from than existing approaches. Furthermore, we show
the influence of a number of seemingly minor design choices on the overall
performance.

    

### [[2104.08142] Supervising Model Attention with Human Explanations for Robust Natural Language Inference](http://arxiv.org/abs/2104.08142)


  Natural Language Inference (NLI) models are known to learn from biases and
artefacts within their training data, impacting how well they generalise to
other unseen datasets. Existing de-biasing approaches focus on preventing the
models from learning these biases, which can result in restrictive models and
lower performance. We instead investigate teaching the model how a human would
approach the NLI task, in order to learn features that will generalise better
to previously unseen examples. Using natural language explanations, we
supervise the model's attention weights to encourage more attention to be paid
to the words present in the explanations, significantly improving model
performance. Our experiments show that the in-distribution improvements of this
method are also accompanied by out-of-distribution improvements, with the
supervised models learning from features that generalise better to other NLI
datasets. Analysis of the model indicates that human explanations encourage
increased attention on the important words, with more attention paid to words
in the premise and less attention paid to punctuation and stop-words.

    

### [[2104.09856] Permutation-Invariant Variational Autoencoder for Graph-Level Representation Learning](http://arxiv.org/abs/2104.09856)


  Recently, there has been great success in applying deep neural networks on
graph structured data. Most work, however, focuses on either node- or
graph-level supervised learning, such as node, link or graph classification or
node-level unsupervised learning (e.g. node clustering). Despite its wide range
of possible applications, graph-level unsupervised learning has not received
much attention yet. This might be mainly attributed to the high representation
complexity of graphs, which can be represented by n! equivalent adjacency
matrices, where n is the number of nodes. In this work we address this issue by
proposing a permutation-invariant variational autoencoder for graph structured
data. Our proposed model indirectly learns to match the node ordering of input
and output graph, without imposing a particular node ordering or performing
expensive graph matching. We demonstrate the effectiveness of our proposed
model on various graph reconstruction and generation tasks and evaluate the
expressive power of extracted representations for downstream graph-level
classification and regression.

    

### [[2105.06165] PassFlow: Guessing Passwords with Generative Flows](http://arxiv.org/abs/2105.06165)


  Recent advances in generative machine learning models rekindled research
interest in the area of password guessing. Data-driven password guessing
approaches based on GANs, language models and deep latent variable models have
shown impressive generalization performance and offer compelling properties for
the task of password guessing. In this paper, we propose PassFlow, a flow-based
generative model approach to password guessing. Flow-based models allow for
precise log-likelihood computation and optimization, which enables exact latent
variable inference. Additionally, flow-based models provide meaningful latent
space representation, which enables operations such as exploration of specific
subspaces of the latent space and interpolation. We demonstrate the
applicability of generative flows to the context of password guessing,
departing from previous applications of flow-networks which are mainly limited
to the continuous space of image generation. We show that PassFlow is able to
outperform prior state-of-the-art GAN-based approaches in the password guessing
task while using a training set that is orders of magnitudes smaller than that
of previous art. Furthermore, a qualitative analysis of the generated samples
shows that PassFlow can accurately model the distribution of the original
passwords, with even non-matched samples closely resembling human-like
passwords.

    

### [[2106.03791] Learning Gaussian Mixtures with Generalised Linear Models: Precise Asymptotics in High-dimensions](http://arxiv.org/abs/2106.03791)


  Generalised linear models for multi-class classification problems are one of
the fundamental building blocks of modern machine learning tasks. In this
manuscript, we characterise the learning of a mixture of $K$ Gaussians with
generic means and covariances via empirical risk minimisation (ERM) with any
convex loss and regularisation. In particular, we prove exact asymptotics
characterising the ERM estimator in high-dimensions, extending several previous
results about Gaussian mixture classification in the literature. We exemplify
our result in two tasks of interest in statistical learning: a) classification
for a mixture with sparse means, where we study the efficiency of $\ell_1$
penalty with respect to $\ell_2$; b) max-margin multi-class classification,
where we characterise the phase transition on the existence of the multi-class
logistic maximum likelihood estimator for $K>2$. Finally, we discuss how our
theory can be applied beyond the scope of synthetic data, showing that in
different cases Gaussian mixtures capture closely the learning curve of
classification tasks in real data sets.

    

### [[2106.09675] Gone Fishing: Neural Active Learning with Fisher Embeddings](http://arxiv.org/abs/2106.09675)


  There is an increasing need for effective active learning algorithms that are
compatible with deep neural networks. This paper motivates and revisits a
classic, Fisher-based active selection objective, and proposes BAIT, a
practical, tractable, and high-performing algorithm that makes it viable for
use with neural models. BAIT draws inspiration from the theoretical analysis of
maximum likelihood estimators (MLE) for parametric models. It selects batches
of samples by optimizing a bound on the MLE error in terms of the Fisher
information, which we show can be implemented efficiently at scale by
exploiting linear-algebraic structure especially amenable to execution on
modern hardware. Our experiments demonstrate that BAIT outperforms the previous
state of the art on both classification and regression problems, and is
flexible enough to be used with a variety of model architectures.

    

### [[2110.09050] Strategizing University Rank Improvement using Interpretable Machine Learning and Data Visualization](http://arxiv.org/abs/2110.09050)


  Annual ranking of higher educational institutions (HEIs) is a global
phenomenon and have significant impact on higher education landscape. Most of
the HEIs pay close attention to ranking results and look forward to improving
their ranks. However, maintaining a good rank and ascending in the rankings is
a difficult task because it requires considerable resources, efforts and
performance improvement plan. In this work, firstly, we show how exploratory
data analysis (EDA) using correlation heatmaps and box plots can aid in
understanding the broad trends in the ranking data. Subsequently, we present a
novel idea of classifying the rankings data using Decision Tree (DT) based
algorithms and retrieve decision paths for rank improvement using data
visualization techniques. Using Laplace correction to the probability estimate,
we quantify the amount of certainty attached with different decision paths
obtained from interpretable DT models. The proposed methodology can aid
Universities and HEIs to quantitatively assess the scope of improvement,
adumbrate a fine-grained long-term action plan and prepare a suitable road-map.

    

### [[2110.09468] Improving Robustness using Generated Data](http://arxiv.org/abs/2110.09468)


  Recent work argues that robust training requires substantially larger
datasets than those required for standard classification. On CIFAR-10 and
CIFAR-100, this translates into a sizable robust-accuracy gap between models
trained solely on data from the original training set and those trained with
additional data extracted from the "80 Million Tiny Images" dataset (TI-80M).
In this paper, we explore how generative models trained solely on the original
training set can be leveraged to artificially increase the size of the original
training set and improve adversarial robustness to $\ell_p$ norm-bounded
perturbations. We identify the sufficient conditions under which incorporating
additional generated data can improve robustness, and demonstrate that it is
possible to significantly reduce the robust-accuracy gap to models trained with
additional real data. Surprisingly, we even show that even the addition of
non-realistic random data (generated by Gaussian sampling) can improve
robustness. We evaluate our approach on CIFAR-10, CIFAR-100, SVHN and
TinyImageNet against $\ell_\infty$ and $\ell_2$ norm-bounded perturbations of
size $\epsilon = 8/255$ and $\epsilon = 128/255$, respectively. We show large
absolute improvements in robust accuracy compared to previous state-of-the-art
methods. Against $\ell_\infty$ norm-bounded perturbations of size $\epsilon =
8/255$, our models achieve 66.10% and 33.49% robust accuracy on CIFAR-10 and
CIFAR-100, respectively (improving upon the state-of-the-art by +8.96% and
+3.29%). Against $\ell_2$ norm-bounded perturbations of size $\epsilon =
128/255$, our model achieves 78.31% on CIFAR-10 (+3.81%). These results beat
most prior works that use external data.

    

### [[2111.11297] Teaching Humans When To Defer to a Classifier via Exemplars](http://arxiv.org/abs/2111.11297)


  Expert decision makers are starting to rely on data-driven automated agents
to assist them with various tasks. For this collaboration to perform properly,
the human decision maker must have a mental model of when and when not to rely
on the agent. In this work, we aim to ensure that human decision makers learn a
valid mental model of the agent's strengths and weaknesses. To accomplish this
goal, we propose an exemplar-based teaching strategy where humans solve the
task with the help of the agent and try to formulate a set of guidelines of
when and when not to defer. We present a novel parameterization of the human's
mental model of the AI that applies a nearest neighbor rule in local regions
surrounding the teaching examples. Using this model, we derive a near-optimal
strategy for selecting a representative teaching set. We validate the benefits
of our teaching strategy on a multi-hop question answering task using crowd
workers and find that when workers draw the right lessons from the teaching
stage, their task performance improves, we furthermore validate our method on a
set of synthetic experiments.

    

### [[2112.02905] Parameter Efficient Deep Probabilistic Forecasting](http://arxiv.org/abs/2112.02905)


  Probabilistic time series forecasting is crucial in many application domains
such as retail, ecommerce, finance, or biology. With the increasing
availability of large volumes of data, a number of neural architectures have
been proposed for this problem. In particular, Transformer-based methods
achieve state-of-the-art performance on real-world benchmarks. However, these
methods require a large number of parameters to be learned, which imposes high
memory requirements on the computational resources for training such models.
To address this problem, we introduce a novel Bidirectional Temporal
Convolutional Network (BiTCN), which requires an order of magnitude less
parameters than a common Transformer-based approach. Our model combines two
Temporal Convolutional Networks (TCNs): the first network encodes future
covariates of the time series, whereas the second network encodes past
observations and covariates. We jointly estimate the parameters of an output
distribution via these two networks.
Experiments on four real-world datasets show that our method performs on par
with four state-of-the-art probabilistic forecasting methods, including a
Transformer-based approach and WaveNet, on two point metrics (sMAPE, NRMSE) as
well as on a set of range metrics (quantile loss percentiles) in the majority
of cases. Secondly, we demonstrate that our method requires significantly less
parameters than Transformer-based methods, which means the model can be trained
faster with significantly lower memory requirements, which as a consequence
reduces the infrastructure cost for deploying these models.

    

### [[2112.06517] Top $K$ Ranking for Multi-Armed Bandit with Noisy Evaluations](http://arxiv.org/abs/2112.06517)


  We consider a multi-armed bandit setting where, at the beginning of each
round, the learner receives noisy independent, and possibly biased,
\emph{evaluations} of the true reward of each arm and it selects $K$ arms with
the objective of accumulating as much reward as possible over $T$ rounds. Under
the assumption that at each round the true reward of each arm is drawn from a
fixed distribution, we derive different algorithmic approaches and theoretical
guarantees depending on how the evaluations are generated. First, we show a
$\widetilde{O}(T^{2/3})$ regret in the general case when the observation
functions are a genearalized linear function of the true rewards. On the other
hand, we show that an improved $\widetilde{O}(\sqrt{T})$ regret can be derived
when the observation functions are noisy linear functions of the true rewards.
Finally, we report an empirical validation that confirms our theoretical
findings, provides a thorough comparison to alternative approaches, and further
supports the interest of this setting in practice.

    

### [[2112.06560] HiClass: a Python library for local hierarchical classification compatible with scikit-learn](http://arxiv.org/abs/2112.06560)


  HiClass is an open-source Python package for local hierarchical
classification fully compatible with scikit-learn. It provides implementations
of the most popular machine learning models for local hierarchical
classification, including Local Classifier Per Node, Local Classifier Per
Parent Node and Local Classifier Per Level. In addition, the library includes
tools to evaluate model performance on hierarchical data. The documentation
contains installation instructions, interactive notebooks, and a complete
description of the API. HiClass is distributed under the simplified BSD
license, encouraging its use in both academic and commercial settings. Source
code and documentation are available at this https URL.

    

### [[2107.09961] Quantum Pattern Recognition in Photonic Circuits](http://arxiv.org/abs/2107.09961)


  This paper proposes a machine learning method to characterize photonic states
via a simple optical circuit and data processing of photon number
distributions, such as photonic patterns. The input states consist of two
coherent states used as references and a two-mode unknown state to be studied.
We successfully trained supervised learning algorithms that can predict the
degree of entanglement in the two-mode state as well as perform the full
tomography of one photonic mode, obtaining satisfactory values in the
considered regression metrics.

    

### [[2109.08612] Active Learning for the Optimal Design of Multinomial Classification in Physics](http://arxiv.org/abs/2109.08612)


  Optimal design for model training is a critical topic in machine learning.
Active Learning aims at obtaining improved models by querying samples with
maximum uncertainty according to the estimation model for artificially
labeling; this has the additional advantage of achieving successful
performances with a reduced number of labeled samples. We analyze its
capability as an assistant for the design of experiments, extracting maximum
information for learning with the minimal cost in fidelity loss, or reducing
total operation costs of labeling in the laboratory. We present two typical
applications as quantum information retrieval in qutrits and phase boundary
prediction in many-body physics. For an equivalent multinomial classification
problem, we achieve the correct rate of 99% with less than 2% samples labeled.
We reckon that active-learning-inspired physics experiments will remarkably
save budget without loss of accuracy.

    

### [[2112.06074] Early Stopping for Deep Image Prior](http://arxiv.org/abs/2112.06074)


  Deep image prior (DIP) and its variants have showed remarkable potential for
solving inverse problems in computer vision, without any extra training data.
Practical DIP models are often substantially overparameterized. During the
fitting process, these models learn mostly the desired visual content first,
and then pick up the potential modeling and observational noise, i.e.,
overfitting. Thus, the practicality of DIP often depends critically on good
early stopping (ES) that captures the transition period. In this regard, the
majority of DIP works for vision tasks only demonstrates the potential of the
models -- reporting the peak performance against the ground truth, but provides
no clue about how to operationally obtain near-peak performance without access
to the groundtruth. In this paper, we set to break this practicality barrier of
DIP, and propose an efficient ES strategy, which consistently detects near-peak
performance across several vision tasks and DIP variants. Based on a simple
measure of dispersion of consecutive DIP reconstructions, our ES method not
only outpaces the existing ones -- which only work in very narrow domains, but
also remains effective when combined with a number of methods that try to
mitigate the overfitting. The code is available at
this https URL.

    

### [[2112.06628] Quantum Stream Learning](http://arxiv.org/abs/2112.06628)


  The exotic nature of quantum mechanics makes machine learning (ML) be
different in the quantum realm compared to classical applications. ML can be
used for knowledge discovery using information continuously extracted from a
quantum system in a broad range of tasks. The model receives streaming quantum
information for learning and decision-making, resulting in instant feedback on
the quantum system. As a stream learning approach, we present a deep
reinforcement learning on streaming data from a continuously measured qubit at
the presence of detuning, dephasing, and relaxation. We also investigate how
the agent adapts to another quantum noise pattern by transfer learning. Stream
learning provides a better understanding of closed-loop quantum control, which
may pave the way for advanced quantum technologies.

    

### [[2112.06981] Public Release and Validation of SPEC CPU2017 PinPoints](http://arxiv.org/abs/2112.06981)


  Phase-based statistical sampling methods such as SimPoints have proven to be
effective at dramatically reducing the long time for architectural simulators
to run large workloads such as SPEC CPU2017. However, generating and validating
them is a long and tenuous process. While checkpoints of program phases, or
"pinballs", of SPEC CPU2017 have been collected by other researchers and shared
with the research community, they are outdated and produce errors when used
with the latest versions of the Sniper architectural simulator. To facilitate
our own research as well as contribute to the community, we collect and
validate our own pinballs for the SPEC CPU2017 SPECspeed suite and release them
to the public domain. In this work we document our methodology, the hardware
and software details of the collection process, and our validation results. In
terms of CPI, our pinballs have an average error rate of 12% when compared with
the native whole-program benchmark execution.

    

### [[2112.07019] Synapse Compression for Event-Based Convolutional-Neural-Network Accelerators](http://arxiv.org/abs/2112.07019)


  Manufacturing-viable neuromorphic chips require novel computer architectures
to achieve the massively parallel and efficient information processing the
brain supports so effortlessly. Emerging event-based architectures are making
this dream a reality. However, the large memory requirements for synaptic
connectivity are a showstopper for the execution of modern convolutional neural
networks (CNNs) on massively parallel, event-based (spiking) architectures.
This work overcomes this roadblock by contributing a lightweight hardware
scheme to compress the synaptic memory requirements by several thousand times,
enabling the execution of complex CNNs on a single chip of small form factor. A
silicon implementation in a 12-nm technology shows that the technique increases
the system's implementation cost by only 2%, despite achieving a total
memory-footprint reduction of up to 374x compared to the best previously
published technique.

    

### [[2112.07159] Birds Eye View Social Distancing Analysis System](http://arxiv.org/abs/2112.07159)


  Social distancing can reduce the infection rates in respiratory pandemics
such as COVID-19. Traffic intersections are particularly suitable for
monitoring and evaluation of social distancing behavior in metropolises. We
propose and evaluate a privacy-preserving social distancing analysis system
(B-SDA), which uses bird's-eye view video recordings of pedestrians who cross
traffic intersections. We devise algorithms for video pre-processing, object
detection and tracking which are rooted in the known computer-vision and deep
learning techniques, but modified to address the problem of detecting very
small objects/pedestrians captured by a highly elevated camera. We propose a
method for incorporating pedestrian grouping for detection of social distancing
violations. B-SDA is used to compare pedestrian behavior based on pre-pandemic
and pandemic videos in a major metropolitan area. The accomplished pedestrian
detection performance is $63.0\%$ $AP_{50}$ and the tracking performance is
$47.6\%$ MOTA. The social distancing violation rate of $15.6\%$ during the
pandemic is notably lower than $31.4\%$ pre-pandemic baseline, indicating that
pedestrians followed CDC-prescribed social distancing recommendations. The
proposed system is suitable for deployment in real-world applications.

    

### [[2112.07269] MCDS: AI Augmented Workflow Scheduling in Mobile Edge Cloud Computing Systems](http://arxiv.org/abs/2112.07269)


  Workflow scheduling is a long-studied problem in parallel and distributed
computing (PDC), aiming to efficiently utilize compute resources to meet user's
service requirements. Recently proposed scheduling methods leverage the low
response times of edge computing platforms to optimize application Quality of
Service (QoS). However, scheduling workflow applications in mobile edge-cloud
systems is challenging due to computational heterogeneity, changing latencies
of mobile devices and the volatile nature of workload resource requirements. To
overcome these difficulties, it is essential, but at the same time challenging,
to develop a long-sighted optimization scheme that efficiently models the QoS
objectives. In this work, we propose MCDS: Monte Carlo Learning using Deep
Surrogate Models to efficiently schedule workflow applications in mobile
edge-cloud computing systems. MCDS is an Artificial Intelligence (AI) based
scheduling approach that uses a tree-based search strategy and a deep neural
network-based surrogate model to estimate the long-term QoS impact of immediate
actions for robust optimization of scheduling decisions. Experiments on
physical and simulated edge-cloud testbeds show that MCDS can improve over the
state-of-the-art methods in terms of energy consumption, response time, SLA
violations and cost by at least 6.13, 4.56, 45.09 and 30.71 percent
respectively.

    

### [[2112.07303] MMO: Meta Multi-Objectivization for Software Configuration Tuning](http://arxiv.org/abs/2112.07303)


  Software configuration tuning is essential for optimizing a given performance
objective (e.g., minimizing latency). Yet, due to the software's intrinsically
complex configuration landscape and expensive measurement, there has been a
rather mild success, particularly in preventing the search from being trapped
in local optima. To address this issue, in this paper we take a different
perspective. Instead of focusing on improving the optimizer, we work on the
level of optimization model and propose a meta multi-objectivization (MMO)
model that considers an auxiliary performance objective (e.g., throughput in
addition to latency). What makes this model unique is that we do not optimize
the auxiliary performance objective, but rather use it to make
similarly-performing while different configurations less comparable (i.e.
Pareto nondominated to each other), thus preventing the search from being
trapped in local optima. Importantly, we show how to effectively use the MMO
model without worrying about its weight -- the only yet highly sensitive
parameter that can affect its effectiveness. Experiments on 22 cases from 11
real-world software systems/environments confirm that our MMO model with the
new normalization performs better than its state-of-the-art single-objective
counterparts on 82% cases while achieving up to 2.09x speedup. For 67% of the
cases, the new normalization also enables the MMO model to outperform the
instance when using it with the normalization used in our prior FSE work under
pre-tuned best weights, saving a great amount of resources which would be
otherwise necessary to find a good weight. We also demonstrate that the MMO
model with the new normalization can consolidate Flash, a recent model-based
tuning tool, on 68% of the cases with 1.22x speedup in general.

    

### [[2105.04909] Accountability and Reconfiguration: Self-Healing Lattice Agreement](http://arxiv.org/abs/2105.04909)


  An accountable distributed system provides means to detect deviations of
system components from their expected behavior. It is natural to complement
fault detection with a reconfiguration mechanism, so that the system could heal
itself, by replacing malfunctioning parts with new ones. In this paper, we
describe a framework that can be used to implement a large class of accountable
and reconfigurable replicated services. We build atop the fundamental lattice
agreement abstraction lying at the core of storage systems and
cryptocurrencies.
Our asynchronous implementation of accountable lattice agreement ensures that
every violation of consistency is followed by an undeniable evidence of
misbehavior of a faulty replica. The system can then be seamlessly reconfigured
by evicting faulty replicas, adding new ones and merging inconsistent states.
We believe that this paper opens a direction towards asynchronous
"self-healing" systems that combine accountability and reconfiguration.

    

### [[2112.06917] Branching Strategy Selection Approach Based on Vivification Ratio](http://arxiv.org/abs/2112.06917)


  The two most effective branching strategies LRB and VSIDS perform differently
on different types of instances. Generally, LRB is more effective on crafted
instances, while VSIDS is more effective on application ones. However,
distinguishing the types of instances is difficult. To overcome this drawback,
we propose a branching strategy selection approach based on the vivification
ratio. This approach uses the LRB branching strategy more to solve the
instances with a very low vivification ratio. We tested the instances from the
main track of SAT competitions in recent years. The results show that the
proposed approach is robust and it significantly increases the number of solved
instances. It is worth mentioning that, with the help of our approach, the
solver Maple\_CM can solve more than 16 instances for the benchmark from the
2020 SAT competition.

    

### [[2112.07045] Fuzzy Win-Win: A Novel Approach to Quantify Win-Win Using Fuzzy Logic](http://arxiv.org/abs/2112.07045)


  The classic win-win has a key flaw in that it cannot offer the parties the
right amounts of winning because each party believes they are winners. In
reality, one party may win more than the other. This strategy is not limited to
a single product or negotiation; it may be applied to a variety of situations
in life. We present a novel way to measure the win-win situation in this paper.
The proposed method employs Fuzzy logic to create a mathematical model that
aids negotiators in quantifying their winning percentages. The model is put to
the test on real-life negotiations scenarios such as the Iranian uranium
enrichment negotiations, the Iraqi-Jordanian oil deal, and the iron ore
negotiation (2005-2009). The presented model has shown to be a useful tool in
practice and can be easily generalized to be utilized in other domains as well.

    

### [[2112.07089] Building on Huang et al. GlossBERT for Word Sense Disambiguation](http://arxiv.org/abs/2112.07089)


  We propose to take on the problem ofWord Sense Disambiguation (WSD). In
language, words of the same form can take different meanings depending on
context. While humans easily infer the meaning or gloss of such words by their
context, machines stumble on this this http URL such, we intend to replicated and
expand upon the results of Huang et al.GlossBERT, a model which they design to
disambiguate these words (Huang et al.,2019). Specifically, we propose the
following augmentations: data-set tweaking(alpha hyper-parameter), ensemble
methods, and replacement of BERT with BART andALBERT. The following GitHub
repository contains all code used in this report, which extends on the code
made available by Huang et al.

    

### [[2112.07173] On the use of Cortical Magnification and Saccades as Biological Proxies for Data Augmentation](http://arxiv.org/abs/2112.07173)


  Self-supervised learning is a powerful way to learn useful representations
from natural data. It has also been suggested as one possible means of building
visual representation in humans, but the specific objective and algorithm are
unknown. Currently, most self-supervised methods encourage the system to learn
an invariant representation of different transformations of the same image in
contrast to those of other images. However, such transformations are generally
non-biologically plausible, and often consist of contrived perceptual schemes
such as random cropping and color jittering. In this paper, we attempt to
reverse-engineer these augmentations to be more biologically or perceptually
plausible while still conferring the same benefits for encouraging robust
representation. Critically, we find that random cropping can be substituted by
cortical magnification, and saccade-like sampling of the image could also
assist the representation learning. The feasibility of these transformations
suggests a potential way that biological visual systems could implement
self-supervision. Further, they break the widely accepted spatially-uniform
processing assumption used in many computer vision algorithms, suggesting a
role for spatially-adaptive computation in humans and machines alike. Our code
and demo can be found here.

    

### [[2112.07191] An Adaptive Graph Pre-training Framework for Localized Collaborative Filtering](http://arxiv.org/abs/2112.07191)


  Graph neural networks (GNNs) have been widely applied in the recommendation
tasks and have obtained very appealing performance. However, most GNN-based
recommendation methods suffer from the problem of data sparsity in practice.
Meanwhile, pre-training techniques have achieved great success in mitigating
data sparsity in various domains such as natural language processing (NLP) and
computer vision (CV). Thus, graph pre-training has the great potential to
alleviate data sparsity in GNN-based recommendations. However, pre-training
GNNs for recommendations face unique challenges. For example, user-item
interaction graphs in different recommendation tasks have distinct sets of
users and items, and they often present different properties. Therefore, the
successful mechanisms commonly used in NLP and CV to transfer knowledge from
pre-training tasks to downstream tasks such as sharing learned embeddings or
feature extractors are not directly applicable to existing GNN-based
recommendations models. To tackle these challenges, we delicately design an
adaptive graph pre-training framework for localized collaborative filtering
(ADAPT). It does not require transferring user/item embeddings, and is able to
capture both the common knowledge across different graphs and the uniqueness
for each graph. Extensive experimental results have demonstrated the
effectiveness and superiority of ADAPT.

    

### [[2112.07198] From Dense to Sparse: Contrastive Pruning for Better Pre-trained Language Model Compression](http://arxiv.org/abs/2112.07198)


  Pre-trained Language Models (PLMs) have achieved great success in various
Natural Language Processing (NLP) tasks under the pre-training and fine-tuning
paradigm. With large quantities of parameters, PLMs are computation-intensive
and resource-hungry. Hence, model pruning has been introduced to compress
large-scale PLMs. However, most prior approaches only consider task-specific
knowledge towards downstream tasks, but ignore the essential task-agnostic
knowledge during pruning, which may cause catastrophic forgetting problem and
lead to poor generalization ability. To maintain both task-agnostic and
task-specific knowledge in our pruned model, we propose ContrAstive Pruning
(CAP) under the paradigm of pre-training and fine-tuning. It is designed as a
general framework, compatible with both structured and unstructured pruning.
Unified in contrastive learning, CAP enables the pruned model to learn from the
pre-trained model for task-agnostic knowledge, and fine-tuned model for
task-specific knowledge. Besides, to better retain the performance of the
pruned model, the snapshots (i.e., the intermediate models at each pruning
iteration) also serve as effective supervisions for pruning. Our extensive
experiments show that adopting CAP consistently yields significant
improvements, especially in extremely high sparsity scenarios. With only 3%
model parameters reserved (i.e., 97% sparsity), CAP successfully achieves 99.2%
and 96.3% of the original BERT performance in QQP and MNLI tasks. In addition,
our probing experiments demonstrate that the model pruned by CAP tends to
achieve better generalization ability.

    

### [[2112.07200] Weakly Supervised High-Fidelity Clothing Model Generation](http://arxiv.org/abs/2112.07200)


  The development of online economics arouses the demand of generating images
of models on product clothes, to display new clothes and promote sales.
However, the expensive proprietary model images challenge the existing image
virtual try-on methods in this scenario, as most of them need to be trained on
considerable amounts of model images accompanied with paired clothes images. In
this paper, we propose a cheap yet scalable weakly-supervised method called
Deep Generative Projection (DGP) to address this specific scenario. Lying in
the heart of the proposed method is to imitate the process of human predicting
the wearing effect, which is an unsupervised imagination based on life
experience rather than computation rules learned from supervisions. Here a
pretrained StyleGAN is used to capture the practical experience of wearing.
Experiments show that projecting the rough alignment of clothing and body onto
the StyleGAN space can yield photo-realistic wearing results. Experiments on
real scene proprietary model images demonstrate the superiority of DGP over
several state-of-the-art supervised methods when generating clothing model
images.

    

### [[2112.07219] A real-time spatiotemporal AI model analyzes skill in open surgical videos](http://arxiv.org/abs/2112.07219)


  Open procedures represent the dominant form of surgery worldwide. Artificial
intelligence (AI) has the potential to optimize surgical practice and improve
patient outcomes, but efforts have focused primarily on minimally invasive
techniques. Our work overcomes existing data limitations for training AI models
by curating, from YouTube, the largest dataset of open surgical videos to date:
1997 videos from 23 surgical procedures uploaded from 50 countries. Using this
dataset, we developed a multi-task AI model capable of real-time understanding
of surgical behaviors, hands, and tools - the building blocks of procedural
flow and surgeon skill. We show that our model generalizes across diverse
surgery types and environments. Illustrating this generalizability, we directly
applied our YouTube-trained model to analyze open surgeries prospectively
collected at an academic medical center and identified kinematic descriptors of
surgical skill related to efficiency of hand motion. Our Annotated Videos of
Open Surgery (AVOS) dataset and trained model will be made available for
further development of surgical AI.

    

### [[2112.07252] A Deep Knowledge Distillation framework for EEG assisted enhancement of single-lead ECG based sleep staging](http://arxiv.org/abs/2112.07252)


  Automatic Sleep Staging study is presently done with the help of
Electroencephalogram (EEG) signals. Recently, Deep Learning (DL) based
approaches have enabled significant progress in this area, allowing for
near-human accuracy in automated sleep staging. However, EEG based sleep
staging requires an extensive as well as an expensive clinical setup. Moreover,
the requirement of an expert for setup and the added inconvenience to the
subject under study renders it unfavourable in a point of care context.
Electrocardiogram (ECG), an unobtrusive alternative to EEG, is more suitable,
but its performance, unsurprisingly, remains sub-par compared to EEG-based
sleep staging. Naturally, it would be helpful to transfer knowledge from EEG to
ECG, ultimately enhancing the model's performance on ECG based inputs.
Knowledge Distillation (KD) is a renowned concept in DL that looks to transfer
knowledge from a better but potentially more cumbersome teacher model to a
compact student model. Building on this concept, we propose a cross-modal KD
framework to improve ECG-based sleep staging performance with assistance from
features learned through models trained on EEG. Additionally, we also conducted
multiple experiments on the individual components of the proposed model to get
better insight into the distillation approach. Data of 200 subjects from the
Montreal Archive of Sleep Studies (MASS) was utilized for our study. The
proposed model showed a 14.3\% and 13.4\% increase in weighted-F1-score in
4-class and 3-class sleep staging, respectively. This demonstrates the
viability of KD for performance improvement of single-channel ECG based sleep
staging in 4-class(W-L-D-R) and 3-class(W-N-R) classification.

    

### [[2112.07315] Kernel-aware Raw Burst Blind Super-Resolution](http://arxiv.org/abs/2112.07315)


  Burst super-resolution (SR) provides a possibility of restoring rich details
from low-quality images. However, since low-resolution (LR) images in practical
applications have multiple complicated and unknown degradations, existing
non-blind (e.g., bicubic) designed networks usually lead to a severe
performance drop in recovering high-resolution (HR) images. Moreover, handling
multiple misaligned noisy raw inputs is also challenging. In this paper, we
address the problem of reconstructing HR images from raw burst sequences
acquired from modern handheld devices. The central idea is a kernel-guided
strategy which can solve the burst SR with two steps: kernel modeling and HR
restoring. The former estimates burst kernels from raw inputs, while the latter
predicts the super-resolved image based on the estimated kernels. Furthermore,
we introduce a kernel-aware deformable alignment module which can effectively
align the raw images with consideration of the blurry priors. Extensive
experiments on synthetic and real-world datasets demonstrate that the proposed
method can perform favorable state-of-the-art performance in the burst SR
problem.

    

### [[2112.07327] Model Uncertainty-Aware Knowledge Amalgamation for Pre-Trained Language Models](http://arxiv.org/abs/2112.07327)


  As many fine-tuned pre-trained language models~(PLMs) with promising
performance are generously released, investigating better ways to reuse these
models is vital as it can greatly reduce the retraining computational cost and
the potential environmental side-effects. In this paper, we explore a novel
model reuse paradigm, Knowledge Amalgamation~(KA) for PLMs. Without human
annotations available, KA aims to merge the knowledge from different
teacher-PLMs, each of which specializes in a different classification problem,
into a versatile student model. The achieve this, we design a Model
Uncertainty--aware Knowledge Amalgamation~(MUKA) framework, which identifies
the potential adequate teacher using Monte-Carlo Dropout for approximating the
golden supervision to guide the student. Experimental results demonstrate that
MUKA achieves substantial improvements over baselines on benchmark datasets.
Further analysis shows that MUKA can generalize well under several complicate
settings with multiple teacher models, heterogeneous teachers, and even
cross-dataset teachers.

    

### [[2112.07337] Multi-Instance Training for Question Answering Across Table and Linked Text](http://arxiv.org/abs/2112.07337)


  Answering natural language questions using information from tables (TableQA)
is of considerable recent interest. In many applications, tables occur not in
isolation, but embedded in, or linked to unstructured text. Often, a question
is best answered by matching its parts to either table cell contents or
unstructured text spans, and extracting answers from either source. This leads
to a new space of TextTableQA problems that was introduced by the HybridQA
dataset. Existing adaptations of table representation to transformer-based
reading comprehension (RC) architectures fail to tackle the diverse modalities
of the two representations through a single system. Training such systems is
further challenged by the need for distant supervision. To reduce cognitive
burden, training instances usually include just the question and answer, the
latter matching multiple table rows and text passages. This leads to a noisy
multi-instance training regime involving not only rows of the table, but also
spans of linked text. We respond to these challenges by proposing MITQA, a new
TextTableQA system that explicitly models the different but closely-related
probability spaces of table row selection and text span selection. Our
experiments indicate the superiority of our approach compared to recent
baselines. The proposed method is currently at the top of the HybridQA
leaderboard with a held out test set, achieving 21 % absolute improvement on
both EM and F1 scores over previous published results.

    

### [[2112.07381] You Only Need One Model for Open-domain Question Answering](http://arxiv.org/abs/2112.07381)


  Recent works for Open-domain Question Answering refer to an external
knowledge base using a retriever model, optionally rerank the passages with a
separate reranker model and generate an answer using an another reader model.
Despite performing related tasks, the models have separate parameters and are
weakly-coupled during training. In this work, we propose casting the retriever
and the reranker as hard-attention mechanisms applied sequentially within the
transformer architecture and feeding the resulting computed representations to
the reader. In this singular model architecture the hidden representations are
progressively refined from the retriever to the reranker to the reader, which
is more efficient use of model capacity and also leads to better gradient flow
when we train it in an end-to-end manner. We also propose a pre-training
methodology to effectively train this architecture. We evaluate our model on
Natural Questions and TriviaQA open datasets and for a fixed parameter budget,
our model outperforms the previous state-of-the-art model by 1.0 and 0.7 exact
match scores.

    

### [[2112.07415] Stochastic Planner-Actor-Critic for Unsupervised Deformable Image Registration](http://arxiv.org/abs/2112.07415)


  Large deformations of organs, caused by diverse shapes and nonlinear shape
changes, pose a significant challenge for medical image registration.
Traditional registration methods need to iteratively optimize an objective
function via a specific deformation model along with meticulous parameter
tuning, but which have limited capabilities in registering images with large
deformations. While deep learning-based methods can learn the complex mapping
from input images to their respective deformation field, it is regression-based
and is prone to be stuck at local minima, particularly when large deformations
are involved. To this end, we present Stochastic Planner-Actor-Critic (SPAC), a
novel reinforcement learning-based framework that performs step-wise
registration. The key notion is warping a moving image successively by each
time step to finally align to a fixed image. Considering that it is challenging
to handle high dimensional continuous action and state spaces in the
conventional reinforcement learning (RL) framework, we introduce a new concept
`Plan' to the standard Actor-Critic model, which is of low dimension and can
facilitate the actor to generate a tractable high dimensional action. The
entire framework is based on unsupervised training and operates in an
end-to-end manner. We evaluate our method on several 2D and 3D medical image
datasets, some of which contain large deformations. Our empirical results
highlight that our work achieves consistent, significant gains and outperforms
state-of-the-art methods.

    

### [[2112.07435] Multi-Leader Congestion Games with an Adversary](http://arxiv.org/abs/2112.07435)


  We study a multi-leader single-follower congestion game where multiple users
(leaders) choose one resource out of a set of resources and, after observing
the realized loads, an adversary (single-follower) attacks the resources with
maximum loads, causing additional costs for the leaders. For the resulting
strategic game among the leaders, we show that pure Nash equilibria may fail to
exist and therefore, we consider approximate equilibria instead. As our first
main result, we show that the existence of a $K$-approximate equilibrium can
always be guaranteed, where $K \approx 1.1974$ is the unique solution of a
cubic polynomial equation. To this end, we give a polynomial time combinatorial
algorithm which computes a $K$-approximate equilibrium. The factor $K$ is
tight, meaning that there is an instance that does not admit an
$\alpha$-approximate equilibrium for any $\alpha<K$. Thus $\alpha=K$ is the
smallest possible value of $\alpha$ such that the existence of an
$\alpha$-approximate equilibrium can be guaranteed for any instance of the
considered game. Secondly, we focus on approximate equilibria of a given fixed
instance. We show how to compute efficiently a best approximate equilibrium,
that is, with smallest possible $\alpha$ among all $\alpha$-approximate
equilibria of the given instance.

    

### [[2112.07467] AI Ethics Principles in Practice: Perspectives of Designers and Developers](http://arxiv.org/abs/2112.07467)


  As consensus across the various published AI ethics principles is approached,
a gap remains between high-level principles and practical techniques that can
be readily adopted to design and develop responsible AI systems. We examine the
practices and experiences of researchers and engineers from Australia's
national scientific research agency (CSIRO), who are involved in designing and
developing AI systems for a range of purposes. Semi-structured interviews were
used to examine how the practices of the participants relate to and align with
a set of high-level AI ethics principles that are proposed by the Australian
Government. The principles comprise: Privacy Protection & Security, Reliability
& Safety, Transparency & Explainability, Fairness, Contestability,
Accountability, Human-centred Values, and Human, Social & Environmental
Wellbeing. The insights of the researchers and engineers as well as the
challenges that arose for them in the practical application of the principles
are examined. Finally, a set of organisational responses are provided to
support the implementation of high-level AI ethics principles into practice.

    

### [[2112.07493] EABlock: A Declarative Entity Alignment Block for Knowledge Graph Creation Pipelines](http://arxiv.org/abs/2112.07493)


  Despite encoding enormous amount of rich and valuable data, existing data
sources are mostly created independently, being a significant challenge to
their integration. Mapping languages, e.g., RML and R2RML, facilitate
declarative specification of the process of applying meta-data and integrating
data into a knowledge graph. Mapping rules can also include knowledge
extraction functions in addition to expressing correspondences among data
sources and a unified schema. Combining mapping rules and functions represents
a powerful formalism to specify pipelines for integrating data into a knowledge
graph transparently. Surprisingly, these formalisms are not fully adapted, and
many knowledge graphs are created by executing ad-hoc programs to pre-process
and integrate data. In this paper, we present EABlock, an approach integrating
Entity Alignment (EA) as part of RML mapping rules. EABlock includes a block of
functions performing entity recognition from textual attributes and link the
recognized entities to the corresponding resources in Wikidata, DBpedia, and
domain specific thesaurus, e.g., UMLS. EABlock provides agnostic and efficient
techniques to evaluate the functions and transfer the mappings to facilitate
its application in any RML-compliant engine. We have empirically evaluated
EABlock performance, and results indicate that EABlock speeds up knowledge
graph creation pipelines that require entity recognition and linking in
state-of-the-art RML-compliant engines. EABlock is also publicly available as a
tool through a GitHub repository(this https URL) and a
DOI(this https URL).

    

### [[2112.07499] Reconfiguring Shortest Paths in Graphs](http://arxiv.org/abs/2112.07499)


  Reconfiguring two shortest paths in a graph means modifying one shortest path
to the other by changing one vertex at a time so that all the intermediate
paths are also shortest paths. This problem has several natural applications,
namely: (a) revamping road networks, (b) rerouting data packets in synchronous
multiprocessing setting, (c) the shipping container stowage problem, and (d)
the train marshalling problem.
When modelled as graph problems, (a) is the most general case while (b), (c)
and (d) are restrictions to different graph classes. We show that (a) is
intractable, even for relaxed variants of the problem. For (b), (c) and (d), we
present efficient algorithms to solve the respective problems. We also
generalize the problem to when at most $k$ (for a fixed integer $k\geq 2$)
contiguous vertices on a shortest path can be changed at a time.

    

### [[2112.07513] CORE-Text: Improving Scene Text Detection with Contrastive Relational Reasoning](http://arxiv.org/abs/2112.07513)


  Localizing text instances in natural scenes is regarded as a fundamental
challenge in computer vision. Nevertheless, owing to the extremely varied
aspect ratios and scales of text instances in real scenes, most conventional
text detectors suffer from the sub-text problem that only localizes the
fragments of text instance (i.e., sub-texts). In this work, we quantitatively
analyze the sub-text problem and present a simple yet effective design,
COntrastive RElation (CORE) module, to mitigate that issue. CORE first
leverages a vanilla relation block to model the relations among all text
proposals (sub-texts of multiple text instances) and further enhances
relational reasoning via instance-level sub-text discrimination in a
contrastive manner. Such way naturally learns instance-aware representations of
text proposals and thus facilitates scene text detection. We integrate the CORE
module into a two-stage text detector of Mask R-CNN and devise our text
detector CORE-Text. Extensive experiments on four benchmarks demonstrate the
superiority of CORE-Text. Code is available:
\url{this https URL}.

    

### [[2112.07515] CoCo-BERT: Improving Video-Language Pre-training with Contrastive Cross-modal Matching and Denoising](http://arxiv.org/abs/2112.07515)


  BERT-type structure has led to the revolution of vision-language pre-training
and the achievement of state-of-the-art results on numerous vision-language
downstream tasks. Existing solutions dominantly capitalize on the multi-modal
inputs with mask tokens to trigger mask-based proxy pre-training tasks (e.g.,
masked language modeling and masked object/frame prediction). In this work, we
argue that such masked inputs would inevitably introduce noise for cross-modal
matching proxy task, and thus leave the inherent vision-language association
under-explored. As an alternative, we derive a particular form of cross-modal
proxy objective for video-language pre-training, i.e., Contrastive Cross-modal
matching and denoising (CoCo). By viewing the masked frame/word sequences as
the noisy augmentation of primary unmasked ones, CoCo strengthens
video-language association by simultaneously pursuing inter-modal matching and
intra-modal denoising between masked and unmasked inputs in a contrastive
manner. Our CoCo proxy objective can be further integrated into any BERT-type
encoder-decoder structure for video-language pre-training, named as Contrastive
Cross-modal BERT (CoCo-BERT). We pre-train CoCo-BERT on TV dataset and a newly
collected large-scale GIF video dataset (ACTION). Through extensive experiments
over a wide range of downstream tasks (e.g., cross-modal retrieval, video
question answering, and video captioning), we demonstrate the superiority of
CoCo-BERT as a pre-trained structure.

    

### [[2112.07516] Transferrable Contrastive Learning for Visual Domain Adaptation](http://arxiv.org/abs/2112.07516)


  Self-supervised learning (SSL) has recently become the favorite among feature
learning methodologies. It is therefore appealing for domain adaptation
approaches to consider incorporating SSL. The intuition is to enforce
instance-level feature consistency such that the predictor becomes somehow
invariant across domains. However, most existing SSL methods in the regime of
domain adaptation usually are treated as standalone auxiliary components,
leaving the signatures of domain adaptation unattended. Actually, the optimal
region where the domain gap vanishes and the instance level constraint that SSL
peruses may not coincide at all. From this point, we present a particular
paradigm of self-supervised learning tailored for domain adaptation, i.e.,
Transferrable Contrastive Learning (TCL), which links the SSL and the desired
cross-domain transferability congruently. We find contrastive learning
intrinsically a suitable candidate for domain adaptation, as its instance
invariance assumption can be conveniently promoted to cross-domain class-level
invariance favored by domain adaptation tasks. Based on particular memory bank
constructions and pseudo label strategies, TCL then penalizes cross-domain
intra-class domain discrepancy between source and target through a clean and
novel contrastive loss. The free lunch is, thanks to the incorporation of
contrastive learning, TCL relies on a moving-averaged key encoder that
naturally achieves a temporally ensembled version of pseudo labels for target
data, which avoids pseudo label error propagation at no extra cost. TCL
therefore efficiently reduces cross-domain gaps. Through extensive experiments
on benchmarks (Office-Home, VisDA-2017, Digits-five, PACS and DomainNet) for
both single-source and multi-source domain adaptation tasks, TCL has
demonstrated state-of-the-art performances.

    

### [[2112.07596] Rushing and Strolling among Answer Sets -- Navigation Made Easy](http://arxiv.org/abs/2112.07596)


  Answer set programming (ASP) is a popular declarative programming paradigm
with a wide range of applications in artificial intelligence. Oftentimes, when
modeling an AI problem with ASP, and in particular when we are interested
beyond simple search for optimal solutions, an actual solution, differences
between solutions, or number of solutions of the ASP program matter. For
example, when a user aims to identify a specific answer set according to her
needs, or requires the total number of diverging solutions to comprehend
probabilistic applications such as reasoning in medical domains. Then, there
are only certain problem specific and handcrafted encoding techniques available
to navigate the solution space of ASP programs, which is oftentimes not enough.
In this paper, we propose a formal and general framework for interactive
navigation towards desired subsets of answer sets analogous to faceted
browsing. Our approach enables the user to explore the solution space by
consciously zooming in or out of sub-spaces of solutions at a certain
configurable pace. We illustrate that weighted faceted navigation is
computationally hard. Finally, we provide an implementation of our approach
that demonstrates the feasibility of our framework for incomprehensible
solution spaces.

    

### [[2112.07599] Learning to Deblur and Rotate Motion-Blurred Faces](http://arxiv.org/abs/2112.07599)


  We propose a solution to the novel task of rendering sharp videos from new
viewpoints from a single motion-blurred image of a face. Our method handles the
complexity of face blur by implicitly learning the geometry and motion of faces
through the joint training on three large datasets: FFHQ and 300VW, which are
publicly available, and a new Bern Multi-View Face Dataset (BMFD) that we
built. The first two datasets provide a large variety of faces and allow our
model to generalize better. BMFD instead allows us to introduce multi-view
constraints, which are crucial to synthesizing sharp videos from a new camera
view. It consists of high frame rate synchronized videos from multiple views of
several subjects displaying a wide range of facial expressions. We use the high
frame rate videos to simulate realistic motion blur through averaging. Thanks
to this dataset, we train a neural network to reconstruct a 3D video
representation from a single image and the corresponding face gaze. We then
provide a camera viewpoint relative to the estimated gaze and the blurry image
as input to an encoder-decoder network to generate a video of sharp frames with
a novel camera viewpoint. We demonstrate our approach on test subjects of our
multi-view dataset and VIDTIMIT.

    

### [[2112.07605] The King is Naked: on the Notion of Robustness for Natural Language Processing](http://arxiv.org/abs/2112.07605)


  There is growing evidence that the classical notion of adversarial robustness
originally introduced for images has been adopted as a de facto standard by a
large part of the NLP research community. We show that this notion is
problematic in the context of NLP as it considers a narrow spectrum of
linguistic phenomena. In this paper, we argue for semantic robustness, which is
better aligned with the human concept of linguistic fidelity. We characterize
semantic robustness in terms of biases that it is expected to induce in a
model. We study semantic robustness of a range of vanilla and robustly trained
architectures using a template-based generative test bed. We complement the
analysis with empirical evidence that, despite being harder to implement,
semantic robustness can improve performance %gives guarantees for on complex
linguistic phenomena where models robust in the classical sense fail.

    

### [[2112.07606] Semantic Answer Type and Relation Prediction Task (SMART 2021)](http://arxiv.org/abs/2112.07606)


  Each year the International Semantic Web Conference organizes a set of
Semantic Web Challenges to establish competitions that will advance
state-of-the-art solutions in some problem domains. The Semantic Answer Type
and Relation Prediction Task (SMART) task is one of the ISWC 2021 Semantic Web
challenges. This is the second year of the challenge after a successful SMART
2020 at ISWC 2020. This year's version focuses on two sub-tasks that are very
important to Knowledge Base Question Answering (KBQA): Answer Type Prediction
and Relation Prediction. Question type and answer type prediction can play a
key role in knowledge base question answering systems providing insights about
the expected answer that are helpful to generate correct queries or rank the
answer candidates. More concretely, given a question in natural language, the
first task is, to predict the answer type using a target ontology (e.g.,
DBpedia or Wikidata. Similarly, the second task is to identify relations in the
natural language query and link them to the relations in a target ontology.
This paper discusses the task descriptions, benchmark datasets, and evaluation
metrics. For more information, please visit this https URL.

    

### [[2112.07622] ISEEQ: Information Seeking Question Generation using Dynamic Meta-Information Retrieval and Knowledge Graphs](http://arxiv.org/abs/2112.07622)


  Conversational Information Seeking (CIS) is a relatively new research area
within conversational AI that attempts to seek information from end-users in
order to understand and satisfy users' needs. If realized, such a system has
far-reaching benefits in the real world; for example, a CIS system can assist
clinicians in pre-screening or triaging patients in healthcare. A key open
sub-problem in CIS that remains unaddressed in the literature is generating
Information Seeking Questions (ISQs) based on a short initial query from the
end-user. To address this open problem, we propose Information SEEking Question
generator (ISEEQ), a novel approach for generating ISQs from just a short user
query, given a large text corpus relevant to the user query. Firstly, ISEEQ
uses a knowledge graph to enrich the user query. Secondly, ISEEQ uses the
knowledge-enriched query to retrieve relevant context passages to ask coherent
ISQs adhering to a conceptual flow. Thirdly, ISEEQ introduces a new deep
generative-adversarial reinforcement learning-based approach for generating
ISQs. We show that ISEEQ can generate high-quality ISQs to promote the
development of CIS agents. ISEEQ significantly outperforms comparable baselines
on five ISQ evaluation metrics across four datasets having user queries from
diverse domains. Further, we argue that ISEEQ is transferable across domains
for generating ISQs, as it shows the acceptable performance when trained and
tested on different pairs of domains. The qualitative human evaluation confirms
ISEEQ-generated ISQs are comparable in quality to human-generated questions and
outperform the best comparable baseline.

    

### [[2112.07627] Visualizing Ensemble Predictions of Music Mood](http://arxiv.org/abs/2112.07627)


  Music mood classification has been a challenging problem in comparison with
some other classification problems (e.g., genre, composer, or period). One
solution for addressing this challenging is to use an of ensemble machine
learning models. In this paper, we show that visualization techniques can
effectively convey the popular prediction as well as uncertainty at different
music sections along the temporal axis, while enabling the analysis of
individual ML models in conjunction with their application to different musical
data. In addition to the traditional visual designs, such as stacked line
graph, ThemeRiver, and pixel-based visualization, we introduced a new variant
of ThemeRiver, called "dual-flux ThemeRiver", which allows viewers to observe
and measure the most popular prediction more easily than stacked line graph and
ThemeRiver. Testing indicates that visualizing ensemble predictions is helpful
both in model-development workflows and for annotating music using model
predictions.

    

### [[2112.07642] EgoBody: Human Body Shape, Motion and Social Interactions from Head-Mounted Devices](http://arxiv.org/abs/2112.07642)


  Understanding social interactions from first-person views is crucial for many
applications, ranging from assistive robotics to AR/VR. A first step for
reasoning about interactions is to understand human pose and shape. However,
research in this area is currently hindered by the lack of data. Existing
datasets are limited in terms of either size, annotations, ground-truth capture
modalities or the diversity of interactions. We address this shortcoming by
proposing EgoBody, a novel large-scale dataset for social interactions in
complex 3D scenes. We employ Microsoft HoloLens2 headsets to record rich
egocentric data streams (including RGB, depth, eye gaze, head and hand
tracking). To obtain accurate 3D ground-truth, we calibrate the headset with a
multi-Kinect rig and fit expressive SMPL-X body meshes to multi-view RGB-D
frames, reconstructing 3D human poses and shapes relative to the scene. We
collect 68 sequences, spanning diverse sociological interaction categories, and
propose the first benchmark for 3D full-body pose and shape estimation from
egocentric views. Our dataset and code will be available for research at
this https URL.

    

### [[1905.10924] Naive probability](http://arxiv.org/abs/1905.10924)


  We describe a rational, but low resolution model of probability.

    

### [[2011.08772] KddRES: A Multi-level Knowledge-driven Dialogue Dataset for Restaurant Towards Customized Dialogue System](http://arxiv.org/abs/2011.08772)


  Compared with CrossWOZ (Chinese) and MultiWOZ (English) dataset which have
coarse-grained information, there is no dataset which handle fine-grained and
hierarchical level information properly. In this paper, we publish a first
Cantonese knowledge-driven Dialogue Dataset for REStaurant (KddRES) in Hong
Kong, which grounds the information in multi-turn conversations to one specific
restaurant. Our corpus contains 0.8k conversations which derive from 10
restaurants with various styles in different regions. In addition to that, we
designed fine-grained slots and intents to better capture semantic information.
The benchmark experiments and data statistic analysis show the diversity and
rich annotations of our dataset. We believe the publish of KddRES can be a
necessary supplement of current dialogue datasets and more suitable and
valuable for small and middle enterprises (SMEs) of society, such as build a
customized dialogue system for each restaurant. The corpus and benchmark models
are publicly available.

    

### [[2101.08169] mt5se: An Open Source Framework for Building Autonomous Traders](http://arxiv.org/abs/2101.08169)


  Autonomous trading robots have been studied in artificial intelligence area
for quite some time. Many AI techniques have been tested for building
autonomous agents able to trade financial assets. These initiatives include
traditional neural networks, fuzzy logic, reinforcement learning but also more
recent approaches like deep neural networks and deep reinforcement learning.
Many developers claim to be successful in creating robots with great
performance when simulating execution with historical price series, so called
backtesting. However, when these robots are used in real markets frequently
they present poor performance in terms of risks and return. In this paper, we
propose an open source framework, called mt5se, that helps the development,
backtesting, live testing and real operation of autonomous traders. We built
and tested several traders using mt5se. The results indicate that it may help
the development of better traders. Furthermore, we discuss the simple
architecture that is used in many studies and propose an alternative multiagent
architecture. Such architecture separates two main concerns for portfolio
manager (PM) : price prediction and capital allocation. More than achieve a
high accuracy, a PM should increase profits when it is right and reduce loss
when it is wrong. Furthermore, price prediction is highly dependent of asset's
nature and history, while capital allocation is dependent only on analyst's
prediction performance and assets' correlation. Finally, we discuss some
promising technologies in the area.

    

### [[2105.10872] CMUA-Watermark: A Cross-Model Universal Adversarial Watermark for Combating Deepfakes](http://arxiv.org/abs/2105.10872)


  Malicious applications of deepfakes (i.e., technologies generating target
facial attributes or entire faces from facial images) have posed a huge threat
to individuals' reputation and security. To mitigate these threats, recent
studies have proposed adversarial watermarks to combat deepfake models, leading
them to generate distorted outputs. Despite achieving impressive results, these
adversarial watermarks have low image-level and model-level transferability,
meaning that they can protect only one facial image from one specific deepfake
model. To address these issues, we propose a novel solution that can generate a
Cross-Model Universal Adversarial Watermark (CMUA-Watermark), protecting a
large number of facial images from multiple deepfake models. Specifically, we
begin by proposing a cross-model universal attack pipeline that attacks
multiple deepfake models iteratively. Then, we design a two-level perturbation
fusion strategy to alleviate the conflict between the adversarial watermarks
generated by different facial images and models. Moreover, we address the key
problem in cross-model optimization with a heuristic approach to automatically
find the suitable attack step sizes for different models, further weakening the
model-level conflict. Finally, we introduce a more reasonable and comprehensive
evaluation method to fully test the proposed method and compare it with
existing ones. Extensive experimental results demonstrate that the proposed
CMUA-Watermark can effectively distort the fake facial images generated by
multiple deepfake models while achieving a better performance than existing
methods.

    

### [[2112.06963] Meterstick: Benchmarking Performance Variability in Cloud and Self-hosted Minecraft-like Games Extended Technical Report](http://arxiv.org/abs/2112.06963)


  Due to increasing popularity and strict performance requirements, online
games have become a topic of interest for the performance engineering
community. One of the most popular types of online games is the modifiable
virtual environment (MVE), in which players can terraform the environment. The
most popular MVE, Minecraft, provides not only entertainment, but also
educational support and social interaction, to over 130 million people
world-wide. MVEs currently support their many players by replicating isolated
instances that support each only up to a few hundred players under favorable
conditions. In practice, as we show here, the real upper limit of supported
players can be much lower. In this work, we posit that performance variability
is a key cause for the lack of scalability in MVEs, investigate experimentally
causes of performance variability, and derive actionable insights. We propose
an operational model for MVEs, which extends the state-of-the-art with
essential aspects, e.g., through the consideration of environment-based
workloads, which are sizable workload components that do not depend on player
input (once set in action). Starting from this model, we design the first
benchmark that focuses on MVE performance variability, defining specialized
workloads, metrics, and processes. We conduct real-world benchmarking of
Minecraft-like MVEs, both cloud-based and self-hosted. We find
environment-based workloads and cloud deployment are significant sources of
performance variability: peak-latency degrades sharply to 20.7 times the
arithmetic mean and exceeds by a factor of 7.4 the performance requirements. We
derive actionable insights for game-developers, game-operators, and other
stakeholders to tame performance variability.

    

### [[2112.06984] Implementing a Category-Theoretic Framework for Typed Abstract Syntax](http://arxiv.org/abs/2112.06984)


  In previous work ("From signatures to monads in UniMath"), we described a
category-theoretic construction of abstract syntax from a signature, mechanized
in the UniMath library based on the Coq proof assistant.
In the present work, we describe what was necessary to generalize that work
to account for simply-typed languages. First, some definitions had to be
generalized to account for the natural appearance of non-endofunctors in the
simply-typed case. As it turns out, in many cases our mechanized results
carried over to the generalized definitions without any code change. Second, an
existing mechanized library on $\omega$-cocontinuous functors had to be
extended by constructions and theorems necessary for constructing multi-sorted
syntax. Third, the theoretical framework for the semantical signatures had to
be generalized from a monoidal to a bicategorical setting, again to account for
non-endofunctors arising in the typed case. This uses actions of endofunctors
on functors with given source, and the corresponding notion of strong functors
between actions, all formalized in UniMath using a recently developed library
of bicategory theory. We explain what needed to be done to plug all of these
ingredients together, modularly.
The main result of our work is a general construction that, when fed with a
signature for a simply-typed language, returns an implementation of that
language together with suitable boilerplate code, in particular, a certified
monadic substitution operation.

    

### [[2112.07064] Ergo -- a programming language for Smart Legal Contracts](http://arxiv.org/abs/2112.07064)


  We present a smart legal contract platform to support a wide range of smart
legal contract use cases. We see this as a step towards improving existing
approaches to representing the complexity of legal agreements and executing
aspects of these agreements.

    

### [[2112.07292] Verifying a Minimalist Reverse-Mode AD Library](http://arxiv.org/abs/2112.07292)


  By exploiting a number of relatively subtle programming language features,
including dynamically-allocated mutable state, first-class functions, and
effect handlers, reverse-mode automatic differentiation can be implemented as a
library. One outstanding question, however, is: with which logical tools can
one specify what this code is expected to compute and verify that it behaves as
expected? We answer this question by using a modern variant of Separation Logic
to specify and verify a minimalist (but concise and elegant) reverse-mode
automatic differentiation library. We view this result as an advanced exercise
in program verification, with potential future applications to more realistic
automatic differentiation systems.

    

### [[2112.07636] Forwarders as Process Compatibility, Logically](http://arxiv.org/abs/2112.07636)


  Session types define protocols that processes must follow when communicating.
The special case of binary session types, i.e. type annotations of protocols
between two parties, is known to be in a propositions-as-types correspondence
with linear logic. In previous work, we have shown that the generalization to
multiparty session types can be expressed either by coherence proofs or by
arbiters, processes that act as middleware by forwarding messages according to
the given protocol. In this paper, following the propositions-as-types fashion,
we generalize arbiters to a logic, which we call forwarder logic, a fragment of
classical linear logic still satisfying cut-elimination. Our main result is
summarized as follows: forwarders generalize coherence and give an elegant
proof-theoretic characterization of multiparty compatibility, a property of
concurrent systems guaranteeing that all sent messages are eventually received
and no deadlock ever occurs.

    

### [[1912.05601] Is Sized Typing for Coq Practical?](http://arxiv.org/abs/1912.05601)


  Contemporary proof assistants such as Coq require that recursive functions be
terminating and corecursive functions be productive to maintain logical
consistency of their type theories, and some ensure these properties using
syntactic checks. However, being syntactic, they are inherently delicate and
restrictive, preventing users from easily writing obviously terminating or
productive functions at their whim.
Meanwhile, there exist many sized type theories that perform type-based
termination and productivity checking, including theories based on the Calculus
of (Co)Inductive Constructions (CIC), the core calculus underlying Coq. These
theories are more robust and compositional in comparison. So why haven't they
been adapted to Coq?
In this paper, we venture to answer this question with CIC$\widehat{*}$, a
sized type theory based on CIC. It extends past work on sized types in CIC with
additional Coq features such as global and local definitions. We also present a
corresponding size inference algorithm and implement it within Coq's kernel;
for maximal backward compatibility with existing Coq developments, it requires
no additional annotations from the user.
In our evaluation of the implementation, we find a severe performance
degradation when compiling parts of the Coq standard library, inherent to the
algorithm itself. We conclude that if we wish to maintain backward
compatibility, using size inference as a replacement for syntactic checking is
wildly impractical in terms of performance.

    

### [<title>Full path to source file is used instead of relative path in verbose message for default build - XGBoost</title>](https://discuss.xgboost.ai/t/full-path-to-source-file-is-used-instead-of-relative-path-in-verbose-message-for-default-build/2601/1)