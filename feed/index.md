
## 2021-7-15

### [[2107.06372] On the Analysis of MUD-Files' Interactions, Conflicts, and Configuration Requirements Before Deployment](http://arxiv.org/abs/2107.06372)


  Manufacturer Usage Description (MUD) is an Internet Engineering Task Force
(IETF) standard designed to protect IoT devices and networks by creating an
out-of-the-box access control list for an IoT device. %The protocol defines a
conceptually straightforward method to implement an isolation-based defensive
mechanism based on the rules that are introduced by the manufacturer of the
device. However, in practice, the access control list of each device is defined
in its MUD-File and may contain possibly hundreds of access control rules. As a
result, reading and validating these files is a challenge; and determining how
multiple IoT devices interact is difficult for the developer and infeasible for
the consumer. To address this we introduce the MUD-Visualizer to provide a
visualization of any number of MUD-Files. MUD-Visualizer is designed to enable
developers to produce correct MUD-Files by providing format correction,
integrating them with other MUD-Files, and identifying conflicts through
visualization. MUD-Visualizer is scalable and its core task is to merge and
illustrate ACEs for multiple devices; both within and beyond the local area
network. MUD-Visualizer is made publicly available and can be found on GitHub.

    

### [[2107.06512] On the Analysis of Adaptive-Rate Applications in Data-Centric Wireless Ad-Hoc Networks](http://arxiv.org/abs/2107.06512)


  Adapting applications' data rates in multi-hop wireless ad-hoc networks is
inherently challenging. Packet collision, channel contention, and queue buildup
contribute to packet loss but are difficult to manage in conventional TCP/IP
architecture. This work explores a data-centric approach based on Name Data
Networking (NDN) architecture, which is considered more suitable for wireless
ad-hoc networks. We show that the default NDN transport offers better
performance in linear topologies but struggles in more extensive networks due
to high collision and contention caused by excessive Interests from
out-of-order data retrieval and redundant data transmission from improper
Interest lifetime setting as well as in-network caching. To fix these, we use
round-trip hop count to limit Interest rate and Dynamic Interest Lifetime to
minimize the negative effect of improper Interest lifetime. Finally, we analyze
the effect of in-network caching on transport performance and which scenarios
may benefit or suffer from it.

    

### [[2107.06537] Age of Information in Physical-Layer Network Coding Enabled Two-Way Relay Networks](http://arxiv.org/abs/2107.06537)


  This paper investigates the information freshness of two-way relay networks
(TWRN) operated with physical-layer network coding (PNC). Information freshness
is quantified by age of information (AoI), defined as the time elapsed since
the generation time of the latest received information update. PNC reduces
communication latency of TWRNs by turning superimposed electromagnetic waves
into network-coded messages so that end users can send update packets to each
other via the relay more frequently. Although sending update packets more
frequently is potential to reduce AoI, how to deal with packet corruption has
not been well investigated. Specifically, if old packets are corrupted in any
hop of a TWRN, one needs to decide the old packets to be dropped or to be
retransmitted, e.g., new packets have recent information, but may require more
time to be delivered. We study the average AoI with and without ARQ in
PNC-enabled TWRNs. We first consider a non-ARQ scheme where old packets are
always dropped when corrupted, referred to once-lost-then-drop (OLTD), and a
classical ARQ scheme with no packet lost, referred to as reliable packet
transmission (RPT). Interestingly, our analysis shows that neither the non-ARQ
scheme nor the pure ARQ scheme achieves good average AoI. We then put forth an
uplink-lost-then-drop (ULTD) protocol that combines packet drop and ARQ.
Experiments on software-defined radio indicate that ULTD significantly
outperforms OLTD and RPT in terms of average AoI. Although this paper focuses
on TWRNs, we believe the insight of ULTD applies generally to other two-hop
networks. Our insight is that to achieve high information freshness, when
packets are corrupted in the first hop, new packets should be generated and
sent (i.e., old packets are discarded); when packets are corrupted in the
second hop, old packets should be retransmitted until successful reception.

    

### [[2107.06548] Communication-Efficient Hierarchical Federated Learning for IoT Heterogeneous Systems with Imbalanced Data](http://arxiv.org/abs/2107.06548)


  Federated learning (FL) is a distributed learning methodology that allows
multiple nodes to cooperatively train a deep learning model, without the need
to share their local data. It is a promising solution for telemonitoring
systems that demand intensive data collection, for detection, classification,
and prediction of future events, from different locations while maintaining a
strict privacy constraint. Due to privacy concerns and critical communication
bottlenecks, it can become impractical to send the FL updated models to a
centralized server. Thus, this paper studies the potential of hierarchical FL
in IoT heterogeneous systems and propose an optimized solution for user
assignment and resource allocation on multiple edge nodes. In particular, this
work focuses on a generic class of machine learning models that are trained
using gradient-descent-based schemes while considering the practical
constraints of non-uniformly distributed data across different users. We
evaluate the proposed system using two real-world datasets, and we show that it
outperforms state-of-the-art FL solutions. In particular, our numerical results
highlight the effectiveness of our approach and its ability to provide 4-6%
increase in the classification accuracy, with respect to hierarchical FL
schemes that consider distance-based user assignment. Furthermore, the proposed
approach could significantly accelerate FL training and reduce communication
overhead by providing 75-85% reduction in the communication rounds between edge
nodes and the centralized server, for the same model accuracy.

    

### [[2107.06570] QoS-Aware Scheduling in New Radio Using Deep Reinforcement Learning](http://arxiv.org/abs/2107.06570)


  Fifth-generation (5G) New Radio (NR) cellular networks support a wide range
of new services, many of which require an application-specific quality of
service (QoS), e.g. in terms of a guaranteed minimum bit-rate or a maximum
tolerable delay. Therefore, scheduling multiple parallel data flows, each
serving a unique application instance, is bound to become an even more
challenging task compared to the previous generations. Leveraging recent
advances in deep reinforcement learning, in this paper, we propose a QoS-Aware
Deep Reinforcement learning Agent (QADRA) scheduler for NR networks. In
contrast to state-of-the-art scheduling heuristics, the QADRA scheduler
explicitly optimizes for the QoS satisfaction rate while simultaneously
maximizing the network performance. Moreover, we train our algorithm end-to-end
on these objectives. We evaluate QADRA in a full scale, near-product, system
level NR simulator and demonstrate a significant boost in network performance.
In our particular evaluation scenario, the QADRA scheduler improves network
throughput by 30% while simultaneously maintaining the QoS satisfaction rate of
VoIP users served by the network, compared to state-of-the-art baselines.

    

### [[2107.06790] Governing Decentralized Complex Queries Through a DAO](http://arxiv.org/abs/2107.06790)


  Recently, a new generation of P2P systems capable of addressing data
integrity and authenticity has emerged for the development of new applications
for a "more" decentralized Internet, i.e., Distributed Ledger Technologies
(DLT) and Decentralized File Systems (DFS). However, these technologies still
have some unanswered issues, mostly related to data lookup and discovery. In
this paper, first, we propose a Distributed Hash Table (DHT) system that
efficiently manages decentralized keyword-based queries executed on data stored
in DFS. Through a hypercube logical layout, queries are efficiently routed
among the network, where each node is responsible for a specific keywords set
and the related contents. Second, we provide a framework for the governance of
the above network, based on a Decentralized Autonomous Organization (DAO)
implementation. We show how the use of smart contracts enables organizational
decision making and rewards for nodes that have actively contributed to the
DHT. Finally, we provide experimental validation of an implementation of our
proposal, where the execution of the same protocol for different logical nodes
of the hypercube allows us to evaluate the efficiency of communication within
the network.

    

### [[2107.06881] Evolution of Non-Terrestrial Networks From 5G to 6G: A Survey](http://arxiv.org/abs/2107.06881)


  Non-terrestrial networks (NTNs) traditionally had certain limited
applications. However, the recent technological advancements opened up myriad
applications of NTNs for 5G and beyond networks, especially when integrated
into terrestrial networks (TNs). This article comprehensively surveys the
evolution of NTNs highlighting its relevance to 5G networks and essentially,
how it will play a pivotal role in the development of 6G and beyond wireless
networks. The survey discusses important features of NTNs integration into TNs
by delving into the new range of services and use cases, various architectures,
and new approaches being adopted to develop a new wireless ecosystem. Our
survey includes the major progresses and outcomes from academic research as
well as industrial efforts. We first start with introducing the relevant 5G use
cases and general integration challenges such as handover and deployment
difficulties. Then, we review the NTNs operations in mmWave and their potential
for the internet of things (IoT). Further, we discuss the significance of
mobile edge computing (MEC) and machine learning (ML) in NTNs by reviewing the
relevant research works. Furthermore, we also discuss the corresponding higher
layer advancements and relevant field trials/prototyping at both academic and
industrial levels. Finally, we identify and review 6G and beyond application
scenarios, novel architectures, technological enablers, and higher layer
aspects pertinent to NTNs integration.

    

### [[2005.07778] Access Control for Distributed Ledgers in the Internet of Things: A Networking Approach](http://arxiv.org/abs/2005.07778)


  In the Internet of Things (IoT) domain, devices need a platform to transact
seamlessly without a trusted intermediary. Although Distributed Ledger
Technologies (DLTs) could provide such a platform, blockchains, such as
Bitcoin, were not designed with IoT networks in mind, hence are often
unsuitable for such applications: they offer poor transaction throughput and
confirmation times, put stress on constrained computing and storage resources,
and require high transaction fees. In this work, we consider a class of
IoT-friendly DLTs based on directed acyclic graphs, rather than a blockchain,
and with a reputation system in the place of Proof of Work (PoW). However,
without PoW, implementation of these DLTs requires an access control algorithm
to manage the rate at which nodes can add new transactions to the ledger. We
model the access control problem and present an algorithm that is fair,
efficient and secure. Our algorithm represents a new design paradigm for DLTs
in which concepts from networking are applied to the DLT setting for the first
time. For example, our algorithm uses distributed rate setting which is similar
in nature to transmission control used in the Internet. However, our solution
features novel adaptations to cope with the adversarial environment of DLTs in
which no individual agent can be trusted. Our algorithm guarantees utilisation
of resources, consistency, fairness, and resilience against attackers. All of
this is achieved efficiently and with regard for the limitations of IoT
devices. We perform extensive simulations to validate these claims.

    

### [[2107.01004] AdaptSky: A DRL Based Resource Allocation Framework in NOMA-UAV Networks](http://arxiv.org/abs/2107.01004)


  Unmanned aerial vehicle (UAV) has recently attracted a lot of attention as a
candidate to meet the 6G ubiquitousconnectivity demand and boost the resiliency
of terrestrialnetworks. Thanks to the high spectral efficiency and low latency,
non-orthogonal multiple access (NOMA) is a potential access technique for
future communication networks. In this paper, we propose to use the UAV as a
moving base station (BS) to serve multiple users using NOMA and jointly solve
for the 3D-UAV placement and resource allocation problem. Since the
corresponding optimization problem is non-convex, we rely on the recent
advances in artificial intelligence (AI) and propose AdaptSky, a deep
reinforcement learning (DRL)-based framework, to efficiently solve it. To the
best of our knowledge, AdaptSky is the first framework that optimizes NOMA
power allocation jointly with 3D-UAV placement using both sub-6GHz and
millimeter wave mmWave spectrum. Furthermore, for the first time in NOMA-UAV
networks, AdaptSky integrates the dueling network (DN) architecture to the DRL
technique to improve its learning capabilities. Our findings show that AdaptSky
does not only exhibit a fast-adapting learning and outperform the
state-of-the-art baseline approach in data rate and fairness, but also it
generalizes very well.

    

### [[2107.06281] Non-isomorphic Inter-modality Graph Alignment and Synthesis for Holistic Brain Mapping](http://arxiv.org/abs/2107.06281)


  Brain graph synthesis marked a new era for predicting a target brain graph
from a source one without incurring the high acquisition cost and processing
time of neuroimaging data. However, existing multi-modal graph synthesis
frameworks have several limitations. First, they mainly focus on generating
graphs from the same domain (intra-modality), overlooking the rich multimodal
representations of brain connectivity (inter-modality). Second, they can only
handle isomorphic graph generation tasks, limiting their generalizability to
synthesizing target graphs with a different node size and topological structure
from those of the source one. More importantly, both target and source domains
might have different distributions, which causes a domain fracture between them
(i.e., distribution misalignment). To address such challenges, we propose an
inter-modality aligner of non-isomorphic graphs (IMANGraphNet) framework to
infer a target graph modality based on a given modality. Our three core
contributions lie in (i) predicting a target graph (e.g., functional) from a
source graph (e.g., morphological) based on a novel graph generative
adversarial network (gGAN); (ii) using non-isomorphic graphs for both source
and target domains with a different number of nodes, edges and structure; and
(iii) enforcing the predicted target distribution to match that of the ground
truth graphs using a graph autoencoder to relax the designed loss oprimization.
To handle the unstable behavior of gGAN, we design a new Ground
Truth-Preserving (GT-P) loss function to guide the generator in learning the
topological structure of ground truth brain graphs. Our comprehensive
experiments on predicting functional from morphological graphs demonstrate the
outperformance of IMANGraphNet in comparison with its variants. This can be
further leveraged for integrative and holistic brain mapping in health and
disease.

    

### [[2107.06304] Deep Neural Networks are Surprisingly Reversible: A Baseline for Zero-Shot Inversion](http://arxiv.org/abs/2107.06304)


  Understanding the behavior and vulnerability of pre-trained deep neural
networks (DNNs) can help to improve them. Analysis can be performed via
reversing the network's flow to generate inputs from internal representations.
Most existing work relies on priors or data-intensive optimization to invert a
model, yet struggles to scale to deep architectures and complex datasets. This
paper presents a zero-shot direct model inversion framework that recovers the
input to the trained model given only the internal representation. The crux of
our method is to inverse the DNN in a divide-and-conquer manner while
re-syncing the inverted layers via cycle-consistency guidance with the help of
synthesized data. As a result, we obtain a single feed-forward model capable of
inversion with a single forward pass without seeing any real data of the
original task. With the proposed approach, we scale zero-shot direct inversion
to deep architectures and complex datasets. We empirically show that modern
classification models on ImageNet can, surprisingly, be inverted, allowing an
approximate recovery of the original 224x224px images from a representation
after more than 20 layers. Moreover, inversion of generators in GANs unveils
latent code of a given synthesized face image at 128x128px, which can even, in
turn, improve defective synthesized images from GANs.

    

### [[2107.06317] Inverse Contextual Bandits: Learning How Behavior Evolves over Time](http://arxiv.org/abs/2107.06317)


  Understanding an agent's priorities by observing their behavior is critical
for transparency and accountability in decision processes, such as in
healthcare. While conventional approaches to policy learning almost invariably
assume stationarity in behavior, this is hardly true in practice: Medical
practice is constantly evolving, and clinical professionals are constantly
fine-tuning their priorities. We desire an approach to policy learning that
provides (1) interpretable representations of decision-making, accounts for (2)
non-stationarity in behavior, as well as operating in an (3) offline manner.
First, we model the behavior of learning agents in terms of contextual bandits,
and formalize the problem of inverse contextual bandits (ICB). Second, we
propose two algorithms to tackle ICB, each making varying degrees of
assumptions regarding the agent's learning strategy. Finally, through both real
and simulated data for liver transplantations, we illustrate the applicability
and explainability of our method, as well as validating its accuracy.

    

### [[2107.06319] On the Performance Analysis of the Adversarial System Variant Approximation Method to Quantify Process Model Generalization](http://arxiv.org/abs/2107.06319)


  Process mining algorithms discover a process model from an event log. The
resulting process model is supposed to describe all possible event sequences of
the underlying system. Generalization is a process model quality dimension of
interest. A generalization metric should quantify the extent to which a process
model represents the observed event sequences contained in the event log and
the unobserved event sequences of the system. Most of the available metrics in
the literature cannot properly quantify the generalization of a process model.
A recently published method [1] called Adversarial System Variant Approximation
leverages Generative Adversarial Networks to approximate the underlying event
sequence distribution of a system from an event log. While this method
demonstrated performance gains over existing methods in measuring the
generalization of process models, its experimental evaluations have been
performed under ideal conditions. This paper experimentally investigates the
performance of Adversarial System Variant Approximation under non-ideal
conditions such as biased and limited event logs. Moreover, experiments are
performed to investigate the originally proposed sampling hyperparameter value
of the method on its performance to measure the generalization. The results
confirm the need to raise awareness about the working conditions of the
Adversarial System Variant Approximation method. The outcomes of this paper
also serve to initiate future research directions.
[1] Theis, Julian, and Houshang Darabi. "Adversarial System Variant
Approximation to Quantify Process Model Generalization." IEEE Access 8 (2020):
194410-194427.

    

### [[2107.06327] Contextual Games: Multi-Agent Learning with Side Information](http://arxiv.org/abs/2107.06327)


  We formulate the novel class of contextual games, a type of repeated games
driven by contextual information at each round. By means of kernel-based
regularity assumptions, we model the correlation between different contexts and
game outcomes and propose a novel online (meta) algorithm that exploits such
correlations to minimize the contextual regret of individual players. We define
game-theoretic notions of contextual Coarse Correlated Equilibria (c-CCE) and
optimal contextual welfare for this new class of games and show that c-CCEs and
optimal welfare can be approached whenever players' contextual regrets vanish.
Finally, we empirically validate our results in a traffic routing experiment,
where our algorithm leads to better performance and higher welfare compared to
baselines that do not exploit the available contextual information or the
correlations present in the game.

    

### [[2107.06336] Learnability of Learning Performance and Its Application to Data Valuation](http://arxiv.org/abs/2107.06336)


  For most machine learning (ML) tasks, evaluating learning performance on a
given dataset requires intensive computation. On the other hand, the ability to
efficiently estimate learning performance may benefit a wide spectrum of
applications, such as active learning, data quality management, and data
valuation. Recent empirical studies show that for many common ML models, one
can accurately learn a parametric model that predicts learning performance for
any given input datasets using a small amount of samples. However, the
theoretical underpinning of the learnability of such performance prediction
models is still missing. In this work, we develop the first theoretical
analysis of the ML performance learning problem. We propose a relaxed notion
for submodularity that can well describe the behavior of learning performance
as a function of input datasets. We give a learning algorithm that achieves a
constant-factor approximation under certain assumptions. Further, we give a
learning algorithm that achieves arbitrarily small error based on a newly
derived structural result. We then discuss a natural, important use case of
learning performance learning -- data valuation, which is known to suffer
computational challenges due to the requirement of estimating learning
performance for many data combinations. We show that performance learning can
significantly improve the accuracy of data valuation.

    

### [[2107.06344] Inverse Reinforcement Learning Based Stochastic Driver Behavior Learning](http://arxiv.org/abs/2107.06344)


  Drivers have unique and rich driving behaviors when operating vehicles in
traffic. This paper presents a novel driver behavior learning approach that
captures the uniqueness and richness of human driver behavior in realistic
driving scenarios. A stochastic inverse reinforcement learning (SIRL) approach
is proposed to learn a distribution of cost function, which represents the
richness of the human driver behavior with a given set of driver-specific
demonstrations. Evaluations are conducted on the realistic driving data
collected from the 3D driver-in-the-loop driving simulation. The results show
that the learned stochastic driver model is capable of expressing the richness
of the human driving strategies under different realistic driving scenarios.
Compared to the deterministic baseline driver model, the results reveal that
the proposed stochastic driver behavior model can better replicate the driver's
unique and rich driving strategies in a variety of traffic conditions.

    

### [[2107.06351] BRIMA: low-overhead BRowser-only IMage Annotation tool (Preprint)](http://arxiv.org/abs/2107.06351)


  Image annotation and large annotated datasets are crucial parts within the
Computer Vision and Artificial Intelligence this http URL the same time, it is
well-known and acknowledged by the research community that the image annotation
process is challenging, time-consuming and hard to scale. Therefore, the
researchers and practitioners are always seeking ways to perform the
annotations easier, faster, and at higher quality. Even though several widely
used tools exist and the tools' landscape evolved considerably, most of the
tools still require intricate technical setups and high levels of technical
savviness from its operators and crowdsource contributors.
In order to address such challenges, we develop and present BRIMA -- a
flexible and open-source browser extension that allows BRowser-only IMage
Annotation at considerably lower overheads. Once added to the browser, it
instantly allows the user to annotate images easily and efficiently directly
from the browser without any installation or setup on the client-side. It also
features cross-browser and cross-platform functionality thus presenting itself
as a neat tool for researchers within the Computer Vision, Artificial
Intelligence, and privacy-related fields.

    

### [[2107.06353] Distributionally Robust Policy Learning via Adversarial Environment Generation](http://arxiv.org/abs/2107.06353)


  Our goal is to train control policies that generalize well to unseen
environments. Inspired by the Distributionally Robust Optimization (DRO)
framework, we propose DRAGEN - Distributionally Robust policy learning via
Adversarial Generation of ENvironments - for iteratively improving robustness
of policies to realistic distribution shifts by generating adversarial
environments. The key idea is to learn a generative model for environments
whose latent variables capture cost-predictive and realistic variations in
environments. We perform DRO with respect to a Wasserstein ball around the
empirical distribution of environments by generating realistic adversarial
environments via gradient ascent on the latent space. We demonstrate strong
Out-of-Distribution (OoD) generalization in simulation for (i) swinging up a
pendulum with onboard vision and (ii) grasping realistic 2D/3D objects.
Grasping experiments on hardware demonstrate better sim2real performance
compared to domain randomization.

    

### [[2107.06356] Real-Time Pothole Detection Using Deep Learning](http://arxiv.org/abs/2107.06356)


  Roads are connecting line between different places, and used daily. Roads'
periodic maintenance keeps them safe and functional. Detecting and reporting
the existence of potholes to responsible departments can help in eliminating
them. This study deployed and tested on different deep learning architecture to
detect potholes. The images used for training were collected by cellphone
mounted on the windshield of the car, in addition to many images downloaded
from the internet to increase the size and variability of the database. Second,
various object detection algorithms are employed and compared to detect
potholes in real-time like SDD-TensorFlow, YOLOv3Darknet53 and YOLOv4Darknet53.
YOLOv4 achieved the best performance with 81% recall, 85% precision and 85.39%
mean Average Precision (mAP). The speed of processing was 20 frame per second.
The system was able to detect potholes from a range on 100 meters away from the
camera. The system can increase the safety of drivers and improve the
performance of self-driving cars by detecting pothole time ahead.

    

### [[2107.06383] How Much Can CLIP Benefit Vision-and-Language Tasks?](http://arxiv.org/abs/2107.06383)


  Most existing Vision-and-Language (V&L) models rely on pre-trained visual
encoders, using a relatively small set of manually-annotated data (as compared
to web-crawled data), to perceive the visual world. However, it has been
observed that large-scale pretraining usually can result in better
generalization performance, e.g., CLIP (Contrastive Language-Image
Pre-training), trained on a massive amount of image-caption pairs, has shown a
strong zero-shot capability on various vision tasks. To further study the
advantage brought by CLIP, we propose to use CLIP as the visual encoder in
various V&L models in two typical scenarios: 1) plugging CLIP into
task-specific fine-tuning; 2) combining CLIP with V&L pre-training and
transferring to downstream tasks. We show that CLIP significantly outperforms
widely-used visual encoders trained with in-domain annotated data, such as
BottomUp-TopDown. We achieve competitive or better results on diverse V&L
tasks, while establishing new state-of-the-art results on Visual Question
Answering, Visual Entailment, and V&L Navigation tasks. We release our code at
this https URL.

    

### [[2107.06386] Geometry and Generalization: Eigenvalues as predictors of where a network will fail to generalize](http://arxiv.org/abs/2107.06386)


  We study the deformation of the input space by a trained autoencoder via the
Jacobians of the trained weight matrices. In doing so, we prove bounds for the
mean squared errors for points in the input space, under assumptions regarding
the orthogonality of the eigenvectors. We also show that the trace and the
product of the eigenvalues of the Jacobian matrices is a good predictor of the
MSE on test points. This is a dataset independent means of testing an
autoencoder's ability to generalize on new input. Namely, no knowledge of the
dataset on which the network was trained is needed, only the parameters of the
trained model.

    

### [[2107.06393] Hybrid Memoised Wake-Sleep: Approximate Inference at the Discrete-Continuous Interface](http://arxiv.org/abs/2107.06393)


  Modeling complex phenomena typically involves the use of both discrete and
continuous variables. Such a setting applies across a wide range of problems,
from identifying trends in time-series data to performing effective
compositional scene understanding in images. Here, we propose Hybrid Memoised
Wake-Sleep (HMWS), an algorithm for effective inference in such hybrid
discrete-continuous models. Prior approaches to learning suffer as they need to
perform repeated expensive inner-loop discrete inference. We build on a recent
approach, Memoised Wake-Sleep (MWS), which alleviates part of the problem by
memoising discrete variables, and extend it to allow for a principled and
effective way to handle continuous variables by learning a separate recognition
model used for importance-sampling based approximate inference and
marginalization. We evaluate HMWS in the GP-kernel learning and 3D scene
understanding domains, and show that it outperforms current state-of-the-art
inference methods.

    

### [[2107.06396] Forecasting Thermoacoustic Instabilities in Liquid Propellant Rocket Engines Using Multimodal Bayesian Deep Learning](http://arxiv.org/abs/2107.06396)


  The 100 MW cryogenic liquid oxygen/hydrogen multi-injector combustor BKD
operated by the DLR Institute of Space Propulsion is a research platform that
allows the study of thermoacoustic instabilities under realistic conditions,
representative of small upper stage rocket engines. We use data from BKD
experimental campaigns in which the static chamber pressure and fuel-oxidizer
ratio are varied such that the first tangential mode of the combustor is
excited under some conditions. We train an autoregressive Bayesian neural
network model to forecast the amplitude of the dynamic pressure time series,
inputting multiple sensor measurements (injector pressure/ temperature
measurements, static chamber pressure, high-frequency dynamic pressure
measurements, high-frequency OH* chemiluminescence measurements) and future
flow rate control signals. The Bayesian nature of our algorithms allows us to
work with a dataset whose size is restricted by the expense of each
experimental run, without making overconfident extrapolations. We find that the
networks are able to accurately forecast the evolution of the pressure
amplitude and anticipate instability events on unseen experimental runs 500
milliseconds in advance. We compare the predictive accuracy of multiple models
using different combinations of sensor inputs. We find that the high-frequency
dynamic pressure signal is particularly informative. We also use the technique
of integrated gradients to interpret the influence of different sensor inputs
on the model prediction. The negative log-likelihood of data points in the test
dataset indicates that predictive uncertainties are well-characterized by our
Bayesian model and simulating a sensor failure event results as expected in a
dramatic increase in the epistemic component of the uncertainty.

    

### [[2107.06405] Shortest-Path Constrained Reinforcement Learning for Sparse Reward Tasks](http://arxiv.org/abs/2107.06405)


  We propose the k-Shortest-Path (k-SP) constraint: a novel constraint on the
agent's trajectory that improves the sample efficiency in sparse-reward MDPs.
We show that any optimal policy necessarily satisfies the k-SP constraint.
Notably, the k-SP constraint prevents the policy from exploring state-action
pairs along the non-k-SP trajectories (e.g., going back and forth). However, in
practice, excluding state-action pairs may hinder the convergence of RL
algorithms. To overcome this, we propose a novel cost function that penalizes
the policy violating SP constraint, instead of completely excluding it. Our
numerical experiment in a tabular RL setting demonstrates that the SP
constraint can significantly reduce the trajectory space of policy. As a
result, our constraint enables more sample efficient learning by suppressing
redundant exploration and exploitation. Our experiments on MiniGrid, DeepMind
Lab, Atari, and Fetch show that the proposed method significantly improves
proximal policy optimization (PPO) and outperforms existing novelty-seeking
exploration methods including count-based exploration even in continuous
control tasks, indicating that it improves the sample efficiency by preventing
the agent from taking redundant actions.

    

### [[2107.06409] The Foes of Neural Network's Data Efficiency Among Unnecessary Input Dimensions](http://arxiv.org/abs/2107.06409)


  Datasets often contain input dimensions that are unnecessary to predict the
output label, e.g. background in object recognition, which lead to more
trainable parameters. Deep Neural Networks (DNNs) are robust to increasing the
number of parameters in the hidden layers, but it is unclear whether this holds
true for the input layer. In this letter, we investigate the impact of
unnecessary input dimensions on a central issue of DNNs: their data efficiency,
ie. the amount of examples needed to achieve certain generalization
performance. Our results show that unnecessary input dimensions that are
task-unrelated substantially degrade data efficiency. This highlights the need
for mechanisms that remove {task-unrelated} dimensions to enable data
efficiency gains.

    

### [[2107.06419] ATTACC the Quadratic Bottleneck of Attention Layers](http://arxiv.org/abs/2107.06419)


  Attention mechanisms form the backbone of state-of-the-art machine learning
models for a variety of tasks. Deploying them on deep neural network (DNN)
accelerators, however, is prohibitively challenging especially under long
sequences. Operators in attention layers exhibit limited reuse and quadratic
growth in memory footprint, leading to severe memory-boundedness. This paper
introduces a new attention-tailored dataflow, termed FLAT, which leverages
operator fusion, loop-nest optimizations, and interleaved execution. It
increases the effective memory bandwidth by efficiently utilizing the
high-bandwidth, low-capacity on-chip buffer and thus achieves better run time
and compute resource utilization. We term FLAT-compatible accelerators ATTACC.
In our evaluation, ATTACC achieves 1.94x and 1.76x speedup and 49% and 42% of
energy reduction comparing to state-of-the-art edge and cloud accelerators.

    

### [[2107.06424] Tourbillon: a Physically Plausible Neural Architecture](http://arxiv.org/abs/2107.06424)


  In a physical neural system, backpropagation is faced with a number of
obstacles including: the need for labeled data, the violation of the locality
learning principle, the need for symmetric connections, and the lack of
modularity. Tourbillon is a new architecture that addresses all these
limitations. At its core, it consists of a stack of circular autoencoders
followed by an output layer. The circular autoencoders are trained in
self-supervised mode by recirculation algorithms and the top layer in
supervised mode by stochastic gradient descent, with the option of propagating
error information through the entire stack using non-symmetric connections.
While the Tourbillon architecture is meant primarily to address physical
constraints, and not to improve current engineering applications of deep
learning, we demonstrate its viability on standard benchmark datasets including
MNIST, Fashion MNIST, and CIFAR10. We show that Tourbillon can achieve
comparable performance to models trained with backpropagation and outperform
models that are trained with other physically plausible algorithms, such as
feedback alignment.

    

### [[2107.06428] For high-dimensional hierarchical models, consider exchangeability of effects across covariates instead of across datasets](http://arxiv.org/abs/2107.06428)


  Hierarchical Bayesian methods enable information sharing across multiple
related regression problems. While standard practice is to model regression
parameters (effects) as (1) exchangeable across datasets and (2) correlated to
differing degrees across covariates, we show that this approach exhibits poor
statistical performance when the number of covariates exceeds the number of
datasets. For instance, in statistical genetics, we might regress dozens of
traits (defining datasets) for thousands of individuals (responses) on up to
millions of genetic variants (covariates). When an analyst has more covariates
than datasets, we argue that it is often more natural to instead model effects
as (1) exchangeable across covariates and (2) correlated to differing degrees
across datasets. To this end, we propose a hierarchical model expressing our
alternative perspective. We devise an empirical Bayes estimator for learning
the degree of correlation between datasets. We develop theory that demonstrates
that our method outperforms the classic approach when the number of covariates
dominates the number of datasets, and corroborate this result empirically on
several high-dimensional multiple regression and classification problems.

    

### [[2107.06433] A New Parallel Algorithm for Sinkhorn Word-Movers Distance and Its Performance on PIUMA and Xeon CPU](http://arxiv.org/abs/2107.06433)


  The Word Movers Distance (WMD) measures the semantic dissimilarity between
two text documents by computing the cost of optimally moving all words of a
source/query document to the most similar words of a target document. Computing
WMD between two documents is costly because it requires solving an optimization
problem that costs $O (V^3 \log(V)) $ where $V$ is the number of unique words
in the document. Fortunately, WMD can be framed as an Earth Mover's Distance
(EMD) for which the algorithmic complexity can be reduced to $O(V^2)$ by adding
an entropy penalty to the optimization problem and solving it using the
Sinkhorn-Knopp algorithm. Additionally, the computation can be made highly
parallel by computing the WMD of a single query document against multiple
target documents at once, for example by finding whether a given tweet is
similar to any other tweets of a given day.
In this paper, we first present a shared-memory parallel Sinkhorn-Knopp
algorithm to compute the WMD of one document against many other documents by
adopting the $ O(V^2)$ EMD algorithm. We then algorithmically transform the
original $O(V^2)$ dense compute-heavy version into an equivalent sparse one
which is mapped onto the new Intel Programmable Integrated Unified Memory
Architecture (PIUMA) system. The WMD parallel implementation achieves 67x
speedup on 96 cores across 4 NUMA sockets of an Intel Cascade Lake system. We
also show that PIUMA cores are around 1.2-2.6x faster than Xeon cores on
Sinkhorn-WMD and also provide better strong scaling.

    

### [[2107.06446] Hierarchical Associative Memory](http://arxiv.org/abs/2107.06446)


  Dense Associative Memories or Modern Hopfield Networks have many appealing
properties of associative memory. They can do pattern completion, store a large
number of memories, and can be described using a recurrent neural network with
a degree of biological plausibility and rich feedback between the neurons. At
the same time, up until now all the models of this class have had only one
hidden layer, and have only been formulated with densely connected network
architectures, two aspects that hinder their machine learning applications.
This paper tackles this gap and describes a fully recurrent model of
associative memory with an arbitrary large number of layers, some of which can
be locally connected (convolutional), and a corresponding energy function that
decreases on the dynamical trajectory of the neurons' activations. The memories
of the full network are dynamically "assembled" using primitives encoded in the
synaptic weights of the lower layers, with the "assembling rules" encoded in
the synaptic weights of the higher layers. In addition to the bottom-up
propagation of information, typical of commonly used feedforward neural
networks, the model described has rich top-down feedback from higher layers
that help the lower-layer neurons to decide on their response to the input
stimuli.

    

### [[2107.06456] AID-Purifier: A Light Auxiliary Network for Boosting Adversarial Defense](http://arxiv.org/abs/2107.06456)


  We propose an AID-purifier that can boost the robustness of
adversarially-trained networks by purifying their inputs. AID-purifier is an
auxiliary network that works as an add-on to an already trained main
classifier. To keep it computationally light, it is trained as a discriminator
with a binary cross-entropy loss. To obtain additionally useful information
from the adversarial examples, the architecture design is closely related to
information maximization principles where two layers of the main classification
network are piped to the auxiliary network. To assist the iterative
optimization procedure of purification, the auxiliary network is trained with
AVmixup. AID-purifier can be used together with other purifiers such as
PixelDefend for an extra enhancement. The overall results indicate that the
best performing adversarially-trained networks can be enhanced by the best
performing purification networks, where AID-purifier is a competitive candidate
that is light and robust.

    

### [[2107.06466] Going Beyond Linear RL: Sample Efficient Neural Function Approximation](http://arxiv.org/abs/2107.06466)


  Deep Reinforcement Learning (RL) powered by neural net approximation of the Q
function has had enormous empirical success. While the theory of RL has
traditionally focused on linear function approximation (or eluder dimension)
approaches, little is known about nonlinear RL with neural net approximations
of the Q functions. This is the focus of this work, where we study function
approximation with two-layer neural networks (considering both ReLU and
polynomial activation functions). Our first result is a computationally and
statistically efficient algorithm in the generative model setting under
completeness for two-layer neural networks. Our second result considers this
setting but under only realizability of the neural net function class. Here,
assuming deterministic dynamics, the sample complexity scales linearly in the
algebraic dimension. In all cases, our results significantly improve upon what
can be attained with linear (or eluder dimension) methods.

    

### [[2107.06469] Model-Parallel Model Selection for Deep Learning Systems](http://arxiv.org/abs/2107.06469)


  As deep learning becomes more expensive, both in terms of time and compute,
inefficiencies in machine learning (ML) training prevent practical usage of
state-of-the-art models for most users. The newest model architectures are
simply too large to be fit onto a single processor. To address the issue, many
ML practitioners have turned to model parallelism as a method of distributing
the computational requirements across several devices. Unfortunately, the
sequential nature of neural networks causes very low efficiency and device
utilization in model parallel training jobs. We propose a new form of "shard
parallelism" combining task and model parallelism, then package it into a
framework we name Hydra. Hydra recasts the problem of model parallelism in the
multi-model context to produce a fine-grained parallel workload of independent
model shards, rather than independent models. This new parallel design promises
dramatic speedups relative to the traditional model parallelism paradigm.

    

### [[2107.06473] Spectrum Gaussian Processes Based On Tunable Basis Functions](http://arxiv.org/abs/2107.06473)


  Spectral approximation and variational inducing learning for the Gaussian
process are two popular methods to reduce computational complexity. However, in
previous research, those methods always tend to adopt the orthonormal basis
functions, such as eigenvectors in the Hilbert space, in the spectrum method,
or decoupled orthogonal components in the variational framework. In this paper,
inspired by quantum physics, we introduce a novel basis function, which is
tunable, local and bounded, to approximate the kernel function in the Gaussian
process. There are two adjustable parameters in these functions, which control
their orthogonality to each other and limit their boundedness. And we conduct
extensive experiments on open-source datasets to testify its performance.
Compared to several state-of-the-art methods, it turns out that the proposed
method can obtain satisfactory or even better results, especially with poorly
chosen kernel functions.

    

### [[2107.06475] Generative and reproducible benchmarks for comprehensive evaluation of machine learning classifiers](http://arxiv.org/abs/2107.06475)


  Understanding the strengths and weaknesses of machine learning (ML)
algorithms is crucial for determine their scope of application. Here, we
introduce the DIverse and GENerative ML Benchmark (DIGEN) - a collection of
synthetic datasets for comprehensive, reproducible, and interpretable
benchmarking of machine learning algorithms for classification of binary
outcomes. The DIGEN resource consists of 40 mathematical functions which map
continuous features to discrete endpoints for creating synthetic datasets.
These 40 functions were discovered using a heuristic algorithm designed to
maximize the diversity of performance among multiple popular machine learning
algorithms thus providing a useful test suite for evaluating and comparing new
methods. Access to the generative functions facilitates understanding of why a
method performs poorly compared to other algorithms thus providing ideas for
improvement. The resource with extensive documentation and analyses is
open-source and available on GitHub.

    

### [[2107.06481] A Convolutional Neural Network Approach to the Classification of Engineering Models](http://arxiv.org/abs/2107.06481)


  This paper presents a deep learning approach for the classification of
Engineering (CAD) models using Convolutional Neural Networks (CNNs). Owing to
the availability of large annotated datasets and also enough computational
power in the form of GPUs, many deep learning-based solutions for object
classification have been proposed of late, especially in the domain of images
and graphical models. Nevertheless, very few solutions have been proposed for
the task of functional classification of CAD models. Hence, for this research,
CAD models have been collected from Engineering Shape Benchmark (ESB), National
Design Repository (NDR) and augmented with newer models created using a
modelling software to form a dataset - 'CADNET'. It is proposed to use a
residual network architecture for CADNET, inspired by the popular ResNet. A
weighted Light Field Descriptor (LFD) scheme is chosen as the method of feature
extraction, and the generated images are fed as inputs to the CNN. The problem
of class imbalance in the dataset is addressed using a class weights approach.
Experiments have been conducted with other signatures such as geodesic distance
etc. using deep networks as well as other network architectures on the CADNET.
The LFD-based CNN approach using the proposed network architecture, along with
gradient boosting yielded the best classification accuracy on CADNET.

    

### [[2107.06499] Deduplicating Training Data Makes Language Models Better](http://arxiv.org/abs/2107.06499)


  We find that existing language modeling datasets contain many near-duplicate
examples and long repetitive substrings. As a result, over 1% of the unprompted
output of language models trained on these datasets is copied verbatim from the
training data. We develop two tools that allow us to deduplicate training
datasets -- for example removing from C4 a single 61 word English sentence that
is repeated over 60,000 times. Deduplication allows us to train models that
emit memorized text ten times less frequently and require fewer train steps to
achieve the same or better accuracy. We can also reduce train-test overlap,
which affects over 4% of the validation set of standard datasets, thus allowing
for more accurate evaluation. We release code for reproducing our work and
performing dataset deduplication at
this https URL.

    

### [[2107.06501] AdvFilter: Predictive Perturbation-aware Filtering against Adversarial Attack via Multi-domain Learning](http://arxiv.org/abs/2107.06501)


  High-level representation-guided pixel denoising and adversarial training are
independent solutions to enhance the robustness of CNNs against adversarial
attacks by pre-processing input data and re-training models, respectively. Most
recently, adversarial training techniques have been widely studied and improved
while the pixel denoising-based method is getting less attractive. However, it
is still questionable whether there exists a more advanced pixel
denoising-based method and whether the combination of the two solutions
benefits each other. To this end, we first comprehensively investigate two
kinds of pixel denoising methods for adversarial robustness enhancement (i.e.,
existing additive-based and unexplored filtering-based methods) under the loss
functions of image-level and semantic-level restorations, respectively, showing
that pixel-wise filtering can obtain much higher image quality (e.g., higher
PSNR) as well as higher robustness (e.g., higher accuracy on adversarial
examples) than existing pixel-wise additive-based method. However, we also
observe that the robustness results of the filtering-based method rely on the
perturbation amplitude of adversarial examples used for training. To address
this problem, we propose predictive perturbation-aware pixel-wise filtering,
where dual-perturbation filtering and an uncertainty-aware fusion module are
designed and employed to automatically perceive the perturbation amplitude
during the training and testing process. The proposed method is termed as
AdvFilter. Moreover, we combine adversarial pixel denoising methods with three
adversarial training-based methods, hinting that considering data and models
jointly is able to achieve more robust CNNs. The experiments conduct on
NeurIPS-2017DEV, SVHN, and CIFAR10 datasets and show the advantages over
enhancing CNNs' robustness, high generalization to different models, and noise
levels.

    

### [[2107.06511] CNN-Cap: Effective Convolutional Neural Network Based Capacitance Models for Full-Chip Parasitic Extraction](http://arxiv.org/abs/2107.06511)


  Accurate capacitance extraction is becoming more important for designing
integrated circuits under advanced process technology. The pattern matching
based full-chip extraction methodology delivers fast computational speed, but
suffers from large error, and tedious efforts on building capacitance models of
the increasing structure patterns. In this work, we propose an effective method
for building convolutional neural network (CNN) based capacitance models
(called CNN-Cap) for two-dimensional (2-D) structures in full-chip capacitance
extraction. With a novel grid-based data representation, the proposed method is
able to model the pattern with a variable number of conductors, so that largely
reduce the number of patterns. Based on the ability of ResNet architecture on
capturing spatial information and the proposed training skills, the obtained
CNN-Cap exhibits much better performance over the multilayer perception neural
network based capacitance model while being more versatile. Extensive
experiments on a 55nm and a 15nm process technologies have demonstrated that
the error of total capacitance produced with CNN-Cap is always within 1.3% and
the error of produced coupling capacitance is less than 10% in over 99.5%
probability. CNN-Cap runs more than 4000X faster than 2-D field solver on a GPU
server, while it consumes negligible memory compared to the look-up table based
capacitance model.

    

### [[2107.06530] Detection of Abnormal Behavior with Self-Supervised Gaze Estimation](http://arxiv.org/abs/2107.06530)


  Due to the recent outbreak of COVID-19, many classes, exams, and meetings
have been conducted non-face-to-face. However, the foundation for video
conferencing solutions is still insufficient. So this technology has become an
important issue. In particular, these technologies are essential for
non-face-to-face testing, and technology dissemination is urgent. In this
paper, we present a single video conferencing solution using gaze estimation in
preparation for these problems. Gaze is an important cue for the tasks such as
analysis of human behavior. Hence, numerous studies have been proposed to solve
gaze estimation using deep learning, which is one of the most prominent methods
up to date. We use these gaze estimation methods to detect abnormal behavior of
video conferencing participants. Our contribution is as follows. i) We find and
apply the optimal network for the gaze estimation method and apply a
self-supervised method to improve accuracy. ii) For anomaly detection, we
present a new dataset that aggregates the values of a new gaze, head pose, etc.
iii) We train newly created data on Multi Layer Perceptron (MLP) models to
detect anomaly behavior based on deep learning. We demonstrate the robustness
of our method through experiments.

    

### [[2107.06534] Zeroth and First Order Stochastic Frank-Wolfe Algorithms for Constrained Optimization](http://arxiv.org/abs/2107.06534)


  This paper considers stochastic convex optimization problems with two sets of
constraints: (a) deterministic constraints on the domain of the optimization
variable, which are difficult to project onto; and (b) deterministic or
stochastic constraints that admit efficient projection. Problems of this form
arise frequently in the context of semidefinite programming as well as when
various NP-hard problems are solved approximately via semidefinite relaxation.
Since projection onto the first set of constraints is difficult, it becomes
necessary to explore projection-free algorithms, such as the stochastic
Frank-Wolfe (FW) algorithm. On the other hand, the second set of constraints
cannot be handled in the same way, and must be incorporated as an indicator
function within the objective function, thereby complicating the application of
FW methods. Similar problems have been studied before, and solved using
first-order stochastic FW algorithms by applying homotopy and Nesterov's
smoothing techniques to the indicator function. This work improves upon these
existing results and puts forth momentum-based first-order methods that yield
improved convergence rates, at par with the best known rates for problems
without the second set of constraints. Zeroth-order variants of the proposed
algorithms are also developed and again improve upon the state-of-the-art rate
results. The efficacy of the proposed algorithms is tested on relevant
applications of sparse matrix estimation, clustering via semidefinite
relaxation, and uniform sparsest cut problem.

    

### [[2107.06543] TEACHING -- Trustworthy autonomous cyber-physical applications through human-centred intelligence](http://arxiv.org/abs/2107.06543)


  This paper discusses the perspective of the H2020 TEACHING project on the
next generation of autonomous applications running in a distributed and highly
heterogeneous environment comprising both virtual and physical resources
spanning the edge-cloud continuum. TEACHING puts forward a human-centred vision
leveraging the physiological, emotional, and cognitive state of the users as a
driver for the adaptation and optimization of the autonomous applications. It
does so by building a distributed, embedded and federated learning system
complemented by methods and tools to enforce its dependability, security and
privacy preservation. The paper discusses the main concepts of the TEACHING
approach and singles out the main AI-related research challenges associated
with it. Further, we provide a discussion of the design choices for the
TEACHING system to tackle the aforementioned challenges

    

### [[2107.06546] ZR-2021VG: Zero-Resource Speech Challenge, Visually-Grounded Language Modelling track, 2021 edition](http://arxiv.org/abs/2107.06546)


  We present the visually-grounded language modelling track that was introduced
in the Zero-Resource Speech challenge, 2021 edition, 2nd round. We motivate the
new track and discuss participation rules in detail. We also present the two
baseline systems that were developed for this track.

    

### [[2107.06566] MESS: Manifold Embedding Motivated Super Sampling](http://arxiv.org/abs/2107.06566)


  Many approaches in the field of machine learning and data analysis rely on
the assumption that the observed data lies on lower-dimensional manifolds. This
assumption has been verified empirically for many real data sets. To make use
of this manifold assumption one generally requires the manifold to be locally
sampled to a certain density such that features of the manifold can be
observed. However, for increasing intrinsic dimensionality of a data set the
required data density introduces the need for very large data sets, resulting
in one of the many faces of the curse of dimensionality. To combat the
increased requirement for local data density we propose a framework to generate
virtual data points that faithful to an approximate embedding function
underlying the manifold observable in the data.

    

### [[2107.06573] A Note on Learning Rare Events in Molecular Dynamics using LSTM and Transformer](http://arxiv.org/abs/2107.06573)


  Recurrent neural networks for language models like long short-term memory
(LSTM) have been utilized as a tool for modeling and predicting long term
dynamics of complex stochastic molecular systems. Recently successful examples
on learning slow dynamics by LSTM are given with simulation data of low
dimensional reaction coordinate. However, in this report we show that the
following three key factors significantly affect the performance of language
model learning, namely dimensionality of reaction coordinates, temporal
resolution and state partition. When applying recurrent neural networks to
molecular dynamics simulation trajectories of high dimensionality, we find that
rare events corresponding to the slow dynamics might be obscured by other
faster dynamics of the system, and cannot be efficiently learned. Under such
conditions, we find that coarse graining the conformational space into
metastable states and removing recrossing events when estimating transition
probabilities between states could greatly help improve the accuracy of slow
dynamics learning in molecular dynamics. Moreover, we also explore other models
like Transformer, which do not show superior performance than LSTM in
overcoming these issues. Therefore, to learn rare events of slow molecular
dynamics by LSTM and Transformer, it is critical to choose proper temporal
resolution (i.e., saving intervals of MD simulation trajectories) and state
partition in high resolution data, since deep neural network models might not
automatically disentangle slow dynamics from fast dynamics when both are
present in data influencing each other.

    

### [[2107.06578] A Distance Measure for Privacy-preserving Process Mining based on Feature Learning](http://arxiv.org/abs/2107.06578)


  To enable process analysis based on an event log without compromising the
privacy of individuals involved in process execution, a log may be anonymized.
Such anonymization strives to transform a log so that it satisfies provable
privacy guarantees, while largely maintaining its utility for process analysis.
Existing techniques perform anonymization using simple, syntactic measures to
identify suitable transformation operations. This way, the semantics of the
activities referenced by the events in a trace are neglected, potentially
leading to transformations in which events of unrelated activities are merged.
To avoid this and incorporate the semantics of activities during anonymization,
we propose to instead incorporate a distance measure based on feature learning.
Specifically, we show how embeddings of events enable the definition of a
distance measure for traces to guide event log anonymization. Our experiments
with real-world data indicate that anonymization using this measure, compared
to a syntactic one, yields logs that are closer to the original log in various
dimensions and, hence, have higher utility for process analysis.

    

### [[2107.06580] IFedAvg: Interpretable Data-Interoperability for Federated Learning](http://arxiv.org/abs/2107.06580)


  Recently, the ever-growing demand for privacy-oriented machine learning has
motivated researchers to develop federated and decentralized learning
techniques, allowing individual clients to train models collaboratively without
disclosing their private datasets. However, widespread adoption has been
limited in domains relying on high levels of user trust, where assessment of
data compatibility is essential. In this work, we define and address low
interoperability induced by underlying client data inconsistencies in federated
learning for tabular data. The proposed method, iFedAvg, builds on federated
averaging adding local element-wise affine layers to allow for a personalized
and granular understanding of the collaborative learning process. Thus,
enabling the detection of outlier datasets in the federation and also learning
the compensation for local data distribution shifts without sharing any
original data. We evaluate iFedAvg using several public benchmarks and a
previously unstudied collection of real-world datasets from the 2014 - 2016
West African Ebola epidemic, jointly forming the largest such dataset in the
world. In all evaluations, iFedAvg achieves competitive average performance
with negligible overhead. It additionally shows substantial improvement on
outlier clients, highlighting increased robustness to individual dataset
shifts. Most importantly, our method provides valuable client-specific insights
at a fine-grained level to guide interoperable federated learning.

    

### [[2107.06608] Continuous vs. Discrete Optimization of Deep Neural Networks](http://arxiv.org/abs/2107.06608)


  Existing analyses of optimization in deep learning are either continuous,
focusing on (variants of) gradient flow, or discrete, directly treating
(variants of) gradient descent. Gradient flow is amenable to theoretical
analysis, but is stylized and disregards computational efficiency. The extent
to which it represents gradient descent is an open question in deep learning
theory. The current paper studies this question. Viewing gradient descent as an
approximate numerical solution to the initial value problem of gradient flow,
we find that the degree of approximation depends on the curvature along the
latter's trajectory. We then show that over deep neural networks with
homogeneous activations, gradient flow trajectories enjoy favorable curvature,
suggesting they are well approximated by gradient descent. This finding allows
us to translate an analysis of gradient flow over deep linear neural networks
into a guarantee that gradient descent efficiently converges to global minimum
almost surely under random initialization. Experiments suggest that over simple
deep neural networks, gradient descent with conventional step size is indeed
close to the continuous limit. We hypothesize that the theory of gradient flows
will be central to unraveling mysteries behind deep learning.

    

### [[2107.06615] Oblivious sketching for logistic regression](http://arxiv.org/abs/2107.06615)


  What guarantees are possible for solving logistic regression in one pass over
a data stream? To answer this question, we present the first data oblivious
sketch for logistic regression. Our sketch can be computed in input sparsity
time over a turnstile data stream and reduces the size of a $d$-dimensional
data set from $n$ to only $\operatorname{poly}(\mu d\log n)$ weighted points,
where $\mu$ is a useful parameter which captures the complexity of compressing
the data. Solving (weighted) logistic regression on the sketch gives an $O(\log
n)$-approximation to the original problem on the full data set. We also show
how to obtain an $O(1)$-approximation with slight modifications. Our sketches
are fast, simple, easy to implement, and our experiments demonstrate their
practicality.

    

### [[2107.06618] Hierarchical Analysis of Visual COVID-19 Features from Chest Radiographs](http://arxiv.org/abs/2107.06618)


  Chest radiography has been a recommended procedure for patient triaging and
resource management in intensive care units (ICUs) throughout the COVID-19
pandemic. The machine learning efforts to augment this workflow have been long
challenged due to deficiencies in reporting, model evaluation, and failure mode
analysis. To address some of those shortcomings, we model radiological features
with a human-interpretable class hierarchy that aligns with the radiological
decision process. Also, we propose the use of a data-driven error analysis
methodology to uncover the blind spots of our model, providing further
transparency on its clinical utility. For example, our experiments show that
model failures highly correlate with ICU imaging conditions and with the
inherent difficulty in distinguishing certain types of radiological features.
Also, our hierarchical interpretation and analysis facilitates the comparison
with respect to radiologists' findings and inter-variability, which in return
helps us to better assess the clinical applicability of models.

    

### [[2107.06626] Optimality of the Johnson-Lindenstrauss Dimensionality Reduction for Practical Measures](http://arxiv.org/abs/2107.06626)


  It is well known that the Johnson-Lindenstrauss dimensionality reduction
method is optimal for worst case distortion. While in practice many other
methods and heuristics are used, not much is known in terms of bounds on their
performance. The question of whether the JL method is optimal for practical
measures of distortion was recently raised in \cite{BFN19} (NeurIPS'19). They
provided upper bounds on its quality for a wide range of practical measures and
showed that indeed these are best possible in many cases. Yet, some of the most
important cases, including the fundamental case of average distortion were left
open. In particular, they show that the JL transform has $1+\epsilon$ average
distortion for embedding into $k$-dimensional Euclidean space, where
$k=O(1/\eps^2)$, and for more general $q$-norms of distortion, $k =
O(\max\{1/\eps^2,q/\eps\})$, whereas tight lower bounds were established only
for large values of $q$ via reduction to the worst case.
In this paper we prove that these bounds are best possible for any
dimensionality reduction method, for any $1 \leq q \leq O(\frac{\log (2\eps^2
n)}{\eps})$ and $\epsilon \geq \frac{1}{\sqrt{n}}$, where $n$ is the size of
the subset of Euclidean space.
Our results imply that the JL method is optimal for various distortion
measures commonly used in practice, such as {\it stress, energy} and {\it
relative error}. We prove that if any of these measures is bounded by $\eps$
then $k=\Omega(1/\eps^2)$, for any $\epsilon \geq \frac{1}{\sqrt{n}}$, matching
the upper bounds of \cite{BFN19} and extending their tightness results for the
full range moment analysis.
Our results may indicate that the JL dimensionality reduction method should
be considered more often in practical applications, and the bounds we provide
for its quality should be served as a measure for comparison when evaluating
the performance of other methods and heuristics.

    

### [[2107.06629] Model-free Reinforcement Learning for Robust Locomotion Using Trajectory Optimization for Exploration](http://arxiv.org/abs/2107.06629)


  In this work we present a general, two-stage reinforcement learning approach
for going from a single demonstration trajectory to a robust policy that can be
deployed on hardware without any additional training. The demonstration is used
in the first stage as a starting point to facilitate initial exploration. In
the second stage, the relevant task reward is optimized directly and a policy
robust to environment uncertainties is computed. We demonstrate and examine in
detail performance and robustness of our approach on highly dynamic hopping and
bounding tasks on a real quadruped robot.

    

### [[2107.06630] Online Evaluation Methods for the Causal Effect of Recommendations](http://arxiv.org/abs/2107.06630)


  Evaluating the causal effect of recommendations is an important objective
because the causal effect on user interactions can directly leads to an
increase in sales and user engagement. To select an optimal recommendation
model, it is common to conduct A/B testing to compare model performance.
However, A/B testing of causal effects requires a large number of users, making
such experiments costly and risky. We therefore propose the first interleaving
methods that can efficiently compare recommendation models in terms of causal
effects. In contrast to conventional interleaving methods, we measure the
outcomes of both items on an interleaved list and items not on the interleaved
list, since the causal effect is the difference between outcomes with and
without recommendations. To ensure that the evaluations are unbiased, we either
select items with equal probability or weight the outcomes using inverse
propensity scores. We then verify the unbiasedness and efficiency of online
evaluation methods through simulated online experiments. The results indicate
that our proposed methods are unbiased and that they have superior efficiency
to A/B testing.

    

### [[2107.06639] You Only Write Thrice: Creating Documents, Computational Notebooks and Presentations From a Single Source](http://arxiv.org/abs/2107.06639)


  Academic trade requires juggling multiple variants of the same content
published in different formats: manuscripts, presentations, posters and
computational notebooks. The need to track versions to accommodate for the
write--review--rebut--revise life-cycle adds another layer of complexity. We
propose to significantly reduce this burden by maintaining a single source
document in a version-controlled environment (such as git), adding
functionality to generate a collection of output formats popular in academia.
To this end, we utilise various open-source tools from the Jupyter scientific
computing ecosystem and operationalise selected software engineering concepts.
We offer a proof-of-concept workflow that composes Jupyter Book (an online
document), Jupyter Notebook (a computational narrative) and reveal.js slides
from a single markdown source file. Hosted on GitHub, our approach supports
change tracking and versioning, as well as a transparent review process based
on the underlying code issue management infrastructure. An exhibit of our
workflow can be previewed at this https URL.

    

### [[2107.06642] Many-to-Many Voice Conversion based Feature Disentanglement using Variational Autoencoder](http://arxiv.org/abs/2107.06642)


  Voice conversion is a challenging task which transforms the voice
characteristics of a source speaker to a target speaker without changing
linguistic content. Recently, there have been many works on many-to-many Voice
Conversion (VC) based on Variational Autoencoder (VAEs) achieving good results,
however, these methods lack the ability to disentangle speaker identity and
linguistic content to achieve good performance on unseen speaker scenarios. In
this paper, we propose a new method based on feature disentanglement to tackle
many to many voice conversion. The method has the capability to disentangle
speaker identity and linguistic content from utterances, it can convert from
many source speakers to many target speakers with a single autoencoder network.
Moreover, it naturally deals with the unseen target speaker scenarios. We
perform both objective and subjective evaluations to show the competitive
performance of our proposed method compared with other state-of-the-art models
in terms of naturalness and target speaker similarity.

    

### [[2107.06650] An Efficient Deep Distribution Network for Bid Shading in First-Price Auctions](http://arxiv.org/abs/2107.06650)


  Since 2019, most ad exchanges and sell-side platforms (SSPs), in the online
advertising industry, shifted from second to first price auctions. Due to the
fundamental difference between these auctions, demand-side platforms (DSPs)
have had to update their bidding strategies to avoid bidding unnecessarily high
and hence overpaying. Bid shading was proposed to adjust the bid price intended
for second-price auctions, in order to balance cost and winning probability in
a first-price auction setup. In this study, we introduce a novel deep
distribution network for optimal bidding in both open (non-censored) and closed
(censored) online first-price auctions. Offline and online A/B testing results
show that our algorithm outperforms previous state-of-art algorithms in terms
of both surplus and effective cost per action (eCPX) metrics. Furthermore, the
algorithm is optimized in run-time and has been deployed into VerizonMedia DSP
as production algorithm, serving hundreds of billions of bid requests per day.
Online A/B test shows that advertiser's ROI are improved by +2.4%, +2.4%, and
+8.6% for impression based (CPM), click based (CPC), and conversion based (CPA)
campaigns respectively.

    

### [[2107.06657] DeepMutants: Training neural bug detectors with contextual mutations](http://arxiv.org/abs/2107.06657)


  Learning-based bug detectors promise to find bugs in large code bases by
exploiting natural hints such as names of variables and functions or comments.
Still, existing techniques tend to underperform when presented with realistic
bugs. We believe bug detector learning to currently suffer from a lack of
realistic defective training examples. In fact, real world bugs are scarce
which has driven existing methods to train on artificially created and mostly
unrealistic mutants. In this work, we propose a novel contextual mutation
operator which incorporates knowledge about the mutation context to dynamically
inject natural and more realistic faults into code. Our approach employs a
masked language model to produce a context-dependent distribution over feasible
token replacements. The evaluation shows that sampling from a language model
does not only produce mutants which more accurately represent real bugs but
also lead to better performing bug detectors, both on artificial benchmarks and
on real world source code.

    

### [[2107.06658] A Framework for Machine Learning of Model Error in Dynamical Systems](http://arxiv.org/abs/2107.06658)


  The development of data-informed predictive models for dynamical systems is
of widespread interest in many disciplines. We present a unifying framework for
blending mechanistic and machine-learning approaches to identify dynamical
systems from data. We compare pure data-driven learning with hybrid models
which incorporate imperfect domain knowledge. We cast the problem in both
continuous- and discrete-time, for problems in which the model error is
memoryless and in which it has significant memory, and we compare data-driven
and hybrid approaches experimentally. Our formulation is agnostic to the chosen
machine learning model.
Using Lorenz '63 and Lorenz '96 Multiscale systems, we find that hybrid
methods substantially outperform solely data-driven approaches in terms of data
hunger, demands for model complexity, and overall predictive performance. We
also find that, while a continuous-time framing allows for robustness to
irregular sampling and desirable domain-interpretability, a discrete-time
framing can provide similar or better predictive performance, especially when
data are undersampled and the vector field cannot be resolved.
We study model error from the learning theory perspective, defining excess
risk and generalization error; for a linear model of the error used to learn
about ergodic dynamical systems, both errors are bounded by terms that diminish
with the square-root of T. We also illustrate scenarios that benefit from
modeling with memory, proving that continuous-time recurrent neural networks
(RNNs) can, in principle, learn memory-dependent model error and reconstruct
the original system arbitrarily well; numerical results depict challenges in
representing memory by this approach. We also connect RNNs to reservoir
computing and thereby relate the learning of memory-dependent error to recent
work on supervised learning between Banach spaces using random features.

    

### [[2107.06661] Plan-Based Relaxed Reward Shaping for Goal-Directed Tasks](http://arxiv.org/abs/2107.06661)


  In high-dimensional state spaces, the usefulness of Reinforcement Learning
(RL) is limited by the problem of exploration. This issue has been addressed
using potential-based reward shaping (PB-RS) previously. In the present work,
we introduce Final-Volume-Preserving Reward Shaping (FV-RS). FV-RS relaxes the
strict optimality guarantees of PB-RS to a guarantee of preserved long-term
behavior. Being less restrictive, FV-RS allows for reward shaping functions
that are even better suited for improving the sample efficiency of RL
algorithms. In particular, we consider settings in which the agent has access
to an approximate plan. Here, we use examples of simulated robotic manipulation
tasks to demonstrate that plan-based FV-RS can indeed significantly improve the
sample efficiency of RL over plan-based PB-RS.

    

### [[2107.06665] Disparity Between Batches as a Signal for Early Stopping](http://arxiv.org/abs/2107.06665)


  We propose a metric for evaluating the generalization ability of deep neural
networks trained with mini-batch gradient descent. Our metric, called gradient
disparity, is the $\ell_2$ norm distance between the gradient vectors of two
mini-batches drawn from the training set. It is derived from a probabilistic
upper bound on the difference between the classification errors over a given
mini-batch, when the network is trained on this mini-batch and when the network
is trained on another mini-batch of points sampled from the same dataset. We
empirically show that gradient disparity is a very promising early-stopping
criterion (i) when data is limited, as it uses all the samples for training and
(ii) when available data has noisy labels, as it signals overfitting better
than the validation data. Furthermore, we show in a wide range of experimental
settings that gradient disparity is strongly related to the generalization
error between the training and test sets, and that it is also very informative
about the level of label noise.

    

### [[2107.06668] Thinkback: Task-SpecificOut-of-Distribution Detection](http://arxiv.org/abs/2107.06668)


  The increased success of Deep Learning (DL) has recently sparked large-scale
deployment of DL models in many diverse industry segments. Yet, a crucial
weakness of supervised model is the inherent difficulty in handling
out-of-distribution samples, i.e., samples belonging to classes that were not
presented to the model at training time. We propose in this paper a novel way
to formulate the out-of-distribution detection problem, tailored for DL models.
Our method does not require fine tuning process on training data, yet is
significantly more accurate than the state of the art for out-of-distribution
detection.

    

### [[2107.06675] M5 Competition Uncertainty: Overdispersion, distributional forecasting, GAMLSS and beyond](http://arxiv.org/abs/2107.06675)


  The M5 competition uncertainty track aims for probabilistic forecasting of
sales of thousands of Walmart retail goods. We show that the M5 competition
data faces strong overdispersion and sporadic demand, especially zero demand.
We discuss resulting modeling issues concerning adequate probabilistic
forecasting of such count data processes. Unfortunately, the majority of
popular prediction methods used in the M5 competition (e.g. lightgbm and
xgboost GBMs) fails to address the data characteristics due to the considered
objective functions. The distributional forecasting provides a suitable
modeling approach for to the overcome those problems. The GAMLSS framework
allows flexible probabilistic forecasting using low dimensional distributions.
We illustrate, how the GAMLSS approach can be applied for the M5 competition
data by modeling the location and scale parameter of various distributions,
e.g. the negative binomial distribution. Finally, we discuss software packages
for distributional modeling and their drawback, like the R package gamlss with
its package extensions, and (deep) distributional forecasting libraries such as
TensorFlow Probability.

    

### [[2107.06676] Higgs Boson Classification: Brain-inspired BCPNN Learning with StreamBrain](http://arxiv.org/abs/2107.06676)


  One of the most promising approaches for data analysis and exploration of
large data sets is Machine Learning techniques that are inspired by brain
models. Such methods use alternative learning rules potentially more
efficiently than established learning rules. In this work, we focus on the
potential of brain-inspired ML for exploiting High-Performance Computing (HPC)
resources to solve ML problems: we discuss the BCPNN and an HPC implementation,
called StreamBrain, its computational cost, suitability to HPC systems. As an
example, we use StreamBrain to analyze the Higgs Boson dataset from High Energy
Physics and discriminate between background and signal classes in collisions of
high-energy particle colliders. Overall, we reach up to 69.15% accuracy and
76.4% Area Under the Curve (AUC) performance.

    

### [[2107.06677] Hybrid Model and Data Driven Algorithm for Online Learning of Any-to-Any Path Loss Maps](http://arxiv.org/abs/2107.06677)


  Learning any-to-any (A2A) path loss maps, where the objective is the
reconstruction of path loss between any two given points in a map, might be a
key enabler for many applications that rely on device-to-device (D2D)
communication. Such applications include machine-type communications (MTC) or
vehicle-to-vehicle (V2V) communications. Current approaches for learning A2A
maps are either model-based methods, or pure data-driven methods. Model-based
methods have the advantage that they can generate reliable estimations with low
computational complexity, but they cannot exploit information coming from data.
Pure data-driven methods can achieve good performance without assuming any
physical model, but their complexity and their lack of robustness is not
acceptable for many applications. In this paper, we propose a novel hybrid
model and data-driven approach that fuses information obtained from datasets
and models in an online fashion. To that end, we leverage the framework of
stochastic learning to deal with the sequential arrival of samples and propose
an online algorithm that alternatively and sequentially minimizes the original
non-convex problem. A proof of convergence is presented, along with experiments
based firstly on synthetic data, and secondly on a more realistic dataset for
V2X, with both experiments showing promising results.

    

### [[2107.06686] Safer Reinforcement Learning through Transferable Instinct Networks](http://arxiv.org/abs/2107.06686)


  Random exploration is one of the main mechanisms through which reinforcement
learning (RL) finds well-performing policies. However, it can lead to
undesirable or catastrophic outcomes when learning online in safety-critical
environments. In fact, safe learning is one of the major obstacles towards
real-world agents that can learn during deployment. One way of ensuring that
agents respect hard limitations is to explicitly configure boundaries in which
they can operate. While this might work in some cases, we do not always have
clear a-priori information which states and actions can lead dangerously close
to hazardous states. Here, we present an approach where an additional policy
can override the main policy and offer a safer alternative action. In our
instinct-regulated RL (IR^2L) approach, an "instinctual" network is trained to
recognize undesirable situations, while guarding the learning policy against
entering them. The instinct network is pre-trained on a single task where it is
safe to make mistakes, and transferred to environments in which learning a new
task safely is critical. We demonstrate IR^2L in the OpenAI Safety gym domain,
in which it receives a significantly lower number of safety violations during
training than a baseline RL approach while reaching similar task performance.

    

### [[2107.06692] Deep Adaptive Multi-Intention Inverse Reinforcement Learning](http://arxiv.org/abs/2107.06692)


  This paper presents a deep Inverse Reinforcement Learning (IRL) framework
that can learn an a priori unknown number of nonlinear reward functions from
unlabeled experts' demonstrations. For this purpose, we employ the tools from
Dirichlet processes and propose an adaptive approach to simultaneously account
for both complex and unknown number of reward functions. Using the conditional
maximum entropy principle, we model the experts' multi-intention behaviors as a
mixture of latent intention distributions and derive two algorithms to estimate
the parameters of the deep reward network along with the number of experts'
intentions from unlabeled demonstrations. The proposed algorithms are evaluated
on three benchmarks, two of which have been specifically extended in this study
for multi-intention IRL, and compared with well-known baselines. We demonstrate
through several experiments the advantages of our algorithms over the existing
approaches and the benefits of online inferring, rather than fixing beforehand,
the number of expert's intentions.

    

### [[2107.06700] Differential-Critic GAN: Generating What You Want by a Cue of Preferences](http://arxiv.org/abs/2107.06700)


  This paper proposes Differential-Critic Generative Adversarial Network
(DiCGAN) to learn the distribution of user-desired data when only partial
instead of the entire dataset possesses the desired property, which generates
desired data that meets user's expectations and can assist in designing
biological products with desired properties. Existing approaches select the
desired samples first and train regular GANs on the selected samples to derive
the user-desired data distribution. However, the selection of the desired data
relies on an expert criterion and supervision over the entire dataset. DiCGAN
introduces a differential critic that can learn the preference direction from
the pairwise preferences, which is amateur knowledge and can be defined on part
of the training data. The resultant critic guides the generation of the desired
data instead of the whole data. Specifically, apart from the Wasserstein GAN
loss, a ranking loss of the pairwise preferences is defined over the critic. It
endows the difference of critic values between each pair of samples with the
pairwise preference relation. The higher critic value indicates that the sample
is preferred by the user. Thus training the generative model for higher critic
values encourages the generation of user-preferred samples. Extensive
experiments show that our DiCGAN achieves state-of-the-art performance in
learning the user-desired data distributions, especially in the cases of
insufficient desired data and limited supervision.

    

### [[2107.06703] Zero-Round Active Learning](http://arxiv.org/abs/2107.06703)


  Active learning (AL) aims at reducing labeling effort by identifying the most
valuable unlabeled data points from a large pool. Traditional AL frameworks
have two limitations: First, they perform data selection in a multi-round
manner, which is time-consuming and impractical. Second, they usually assume
that there are a small amount of labeled data points available in the same
domain as the data in the unlabeled pool. Recent work proposes a solution for
one-round active learning based on data utility learning and optimization,
which fixes the first issue but still requires the initially labeled data
points in the same domain. In this paper, we propose $\mathrm{D^2ULO}$ as a
solution that solves both issues. Specifically, $\mathrm{D^2ULO}$ leverages the
idea of domain adaptation (DA) to train a data utility model which can
effectively predict the utility for any given unlabeled data in the target
domain once labeled. The trained data utility model can then be used to select
high-utility data and at the same time, provide an estimate for the utility of
the selected data. Our algorithm does not rely on any feedback from annotators
in the target domain and hence, can be used to perform zero-round active
learning or warm-start existing multi-round active learning strategies. Our
experiments show that $\mathrm{D^2ULO}$ outperforms the existing
state-of-the-art AL strategies equipped with domain adaptation over various
domain shift settings (e.g., real-to-real data and synthetic-to-real data).
Particularly, $\mathrm{D^2ULO}$ are applicable to the scenario where source and
target labels have mismatches, which is not supported by the existing works.

    

### [[2107.06720] Fairness in Ranking under Uncertainty](http://arxiv.org/abs/2107.06720)


  Fairness has emerged as an important consideration in algorithmic
decision-making. Unfairness occurs when an agent with higher merit obtains a
worse outcome than an agent with lower merit. Our central point is that a
primary cause of unfairness is uncertainty. A principal or algorithm making
decisions never has access to the agents' true merit, and instead uses proxy
features that only imperfectly predict merit (e.g., GPA, star ratings,
recommendation letters). None of these ever fully capture an agent's merit; yet
existing approaches have mostly been defining fairness notions directly based
on observed features and outcomes.
Our primary point is that it is more principled to acknowledge and model the
uncertainty explicitly. The role of observed features is to give rise to a
posterior distribution of the agents' merits. We use this viewpoint to define a
notion of approximate fairness in ranking. We call an algorithm $\phi$-fair
(for $\phi \in [0,1]$) if it has the following property for all agents $x$ and
all $k$: if agent $x$ is among the top $k$ agents with respect to merit with
probability at least $\rho$ (according to the posterior merit distribution),
then the algorithm places the agent among the top $k$ agents in its ranking
with probability at least $\phi \rho$.
We show how to compute rankings that optimally trade off approximate fairness
against utility to the principal. In addition to the theoretical
characterization, we present an empirical analysis of the potential impact of
the approach in simulation studies. For real-world validation, we applied the
approach in the context of a paper recommendation system that we built and
fielded at a large conference.

    

### [[2107.06724] Federated Mixture of Experts](http://arxiv.org/abs/2107.06724)


  Federated learning (FL) has emerged as the predominant approach for
collaborative training of neural network models across multiple users, without
the need to gather the data at a central location. One of the important
challenges in this setting is data heterogeneity, i.e. different users have
different data characteristics. For this reason, training and using a single
global model might be suboptimal when considering the performance of each of
the individual user's data. In this work, we tackle this problem via Federated
Mixture of Experts, FedMix, a framework that allows us to train an ensemble of
specialized models. FedMix adaptively selects and trains a user-specific
selection of the ensemble members. We show that users with similar data
characteristics select the same members and therefore share statistical
strength while mitigating the effect of non-i.i.d data. Empirically, we show
through an extensive experimental evaluation that FedMix improves performance
compared to using a single global model across a variety of different sources
of non-i.i.d.-ness.

    

### [[2107.06744] Efficient Learning of Pinball TWSVM using Privileged Information and its applications](http://arxiv.org/abs/2107.06744)


  In any learning framework, an expert knowledge always plays a crucial role.
But, in the field of machine learning, the knowledge offered by an expert is
rarely used. Moreover, machine learning algorithms (SVM based) generally use
hinge loss function which is sensitive towards the noise. Thus, in order to get
the advantage from an expert knowledge and to reduce the sensitivity towards
the noise, in this paper, we propose privileged information based Twin Pinball
Support Vector Machine classifier (Pin-TWSVMPI) where expert's knowledge is in
the form of privileged information. The proposed Pin-TWSVMPI incorporates
privileged information by using correcting function so as to obtain two
nonparallel decision hyperplanes. Further, in order to make computations more
efficient and fast, we use Sequential Minimal Optimization (SMO) technique for
obtaining the classifier and have also shown its application for Pedestrian
detection and Handwritten digit recognition. Further, for UCI datasets, we
first implement a procedure which extracts privileged information from the
features of the dataset which are then further utilized by Pin-TWSVMPI that
leads to enhancement in classification accuracy with comparatively lesser
computational time.

    

### [[2107.06755] DIT4BEARs Smart Roads Internship](http://arxiv.org/abs/2107.06755)


  The research internship at UiT - The Arctic University of Norway was offered
for our team being the winner of the 'Smart Roads - Winter Road Maintenance
2021' Hackathon. The internship commenced on 3 May 2021 and ended on 21 May
2021 with meetings happening twice each week. In spite of having different
nationalities and educational backgrounds, we both interns tried to collaborate
as a team as much as possible. The most alluring part was working on this
project made us realize the critical conditions faced by the arctic people,
where it was hard to gain such a unique experience from our residence. We
developed and implemented several deep learning models to classify the states
(dry, moist, wet, icy, snowy, slushy). Depending upon the best model, the
weather forecast app will predict the state taking the Ta, Tsurf, Height,
Speed, Water, etc. into consideration. The crucial part was to define a safety
metric which is the product of the accident rates based on friction and the
accident rates based on states. We developed a regressor that will predict the
safety metric depending upon the state obtained from the classifier and the
friction obtained from the sensor data. A pathfinding algorithm has been
designed using the sensor data, open street map data, weather data.

    

### [[2107.06762] Modelling Neuronal Behaviour with Time Series Regression: Recurrent Neural Networks on C. Elegans Data](http://arxiv.org/abs/2107.06762)


  Given the inner complexity of the human nervous system, insight into the
dynamics of brain activity can be gained from understanding smaller and simpler
organisms, such as the nematode C. Elegans. The behavioural and structural
biology of these organisms is well-known, making them prime candidates for
benchmarking modelling and simulation techniques. In these complex neuronal
collections, classical, white-box modelling techniques based on intrinsic
structural or behavioural information are either unable to capture the profound
nonlinearities of the neuronal response to different stimuli or generate
extremely complex models, which are computationally intractable. In this paper
we show how the nervous system of C. Elegans can be modelled and simulated with
data-driven models using different neural network architectures. Specifically,
we target the use of state of the art recurrent neural networks architectures
such as LSTMs and GRUs and compare these architectures in terms of their
properties and their accuracy as well as the complexity of the resulting
models. We show that GRU models with a hidden layer size of 4 units are able to
accurately reproduce with high accuracy the system's response to very different
stimuli.

    

### [[2107.06767] Correlated Stochastic Block Models: Exact Graph Matching with Applications to Recovering Communities](http://arxiv.org/abs/2107.06767)


  We consider the task of learning latent community structure from multiple
correlated networks. First, we study the problem of learning the latent vertex
correspondence between two edge-correlated stochastic block models, focusing on
the regime where the average degree is logarithmic in the number of vertices.
We derive the precise information-theoretic threshold for exact recovery: above
the threshold there exists an estimator that outputs the true correspondence
with probability close to 1, while below it no estimator can recover the true
correspondence with probability bounded away from 0. As an application of our
results, we show how one can exactly recover the latent communities using
multiple correlated graphs in parameter regimes where it is
information-theoretically impossible to do so using just a single graph.

    

### [[2107.06773] Relational graph convolutional networks for predicting blood-brain barrier penetration of drug molecules](http://arxiv.org/abs/2107.06773)


  The evaluation of the BBB penetrating ability of drug molecules is a critical
step in brain drug development. Computational prediction based on machine
learning has proved to be an efficient way to conduct the evaluation. However,
performance of the established models has been limited by their incapability of
dealing with the interactions between drugs and proteins, which play an
important role in the mechanism behind BBB penetrating behaviors. To address
this issue, we employed the relational graph convolutional network (RGCN) to
handle the drug-protein (denoted by the encoding gene) relations as well as the
features of each individual drug. In addition, drug-drug similarity was also
introduced to connect structurally similar drugs in the graph. The RGCN model
was initially trained without input of any drug features. And the performance
was already promising, demonstrating the significant role of the
drug-protein/drug-drug relations in the prediction of BBB permeability.
Moreover, molecular embeddings from a pre-trained knowledge graph were used as
the drug features to further enhance the predictive ability of the model.
Finally, the best performing RGCN model was built with a large number of
unlabeled drugs integrated into the graph.

    

### [[2107.06782] Clustering and attention model based for Intelligent Trading](http://arxiv.org/abs/2107.06782)


  The foreign exchange market has taken an important role in the global
financial market. While foreign exchange trading brings high-yield
opportunities to investors, it also brings certain risks. Since the
establishment of the foreign exchange market in the 20th century, foreign
exchange rate forecasting has become a hot issue studied by scholars from all
over the world. Due to the complexity and number of factors affecting the
foreign exchange market, technical analysis cannot respond to administrative
intervention or unexpected events. Our team chose several pairs of foreign
currency historical data and derived technical indicators from 2005 to 2021 as
the dataset and established different machine learning models for event-driven
price prediction for oversold scenario.

    

### [[2107.06785] Large-Scale News Classification using BERT Language Model: Spark NLP Approach](http://arxiv.org/abs/2107.06785)


  The rise of big data analytics on top of NLP increases the computational
burden for text processing at scale. The problems faced in NLP are very high
dimensional text, so it takes a high computation resource. The MapReduce allows
parallelization of large computations and can improve the efficiency of text
processing. This research aims to study the effect of big data processing on
NLP tasks based on a deep learning approach. We classify a big text of news
topics with fine-tuning BERT used pre-trained models. Five pre-trained models
with a different number of parameters were used in this study. To measure the
efficiency of this method, we compared the performance of the BERT with the
pipelines from Spark NLP. The result shows that BERT without Spark NLP gives
higher accuracy compared to BERT with Spark NLP. The accuracy average and
training time of all models using BERT is 0.9187 and 35 minutes while using
BERT with Spark NLP pipeline is 0.8444 and 9 minutes. The bigger model will
take more computation resources and need a longer time to complete the tasks.
However, the accuracy of BERT with Spark NLP only decreased by an average of
5.7%, while the training time was reduced significantly by 62.9% compared to
BERT without Spark NLP.

    

### [[2107.06796] Indonesia's Fake News Detection using Transformer Network](http://arxiv.org/abs/2107.06796)


  Fake news is a problem faced by society in this era. It is not rare for fake
news to cause provocation and problem for the people. Indonesia, as a country
with the 4th largest population, has a problem in dealing with fake news. More
than 30% of rural and urban population are deceived by this fake news problem.
As we have been studying, there is only few literatures on preventing the
spread of fake news in Bahasa Indonesia. So, this research is conducted to
prevent these problems. The dataset used in this research was obtained from a
news portal that identifies fake news, this http URL. Using Web Scrapping on
this page, we got 1116 data consisting of valid news and fake news. The dataset
can be accessed at this https URL. This
dataset will be combined with other available datasets. The methods used are
CNN, BiLSTM, Hybrid CNN-BiLSTM, and BERT with Transformer Network. This
research shows that the BERT method with Transformer Network has the best
results with an accuracy of up to 90%.

    

### [[2107.06802] BERT Fine-Tuning for Sentiment Analysis on Indonesian Mobile Apps Reviews](http://arxiv.org/abs/2107.06802)


  User reviews have an essential role in the success of the developed mobile
apps. User reviews in the textual form are unstructured data, creating a very
high complexity when processed for sentiment analysis. Previous approaches that
have been used often ignore the context of reviews. In addition, the relatively
small data makes the model overfitting. A new approach, BERT, has been
introduced as a transfer learning model with a pre-trained model that has
previously been trained to have a better context representation. This study
examines the effectiveness of fine-tuning BERT for sentiment analysis using two
different pre-trained models. Besides the multilingual pre-trained model, we
use the pre-trained model that only has been trained in Indonesian. The dataset
used is Indonesian user reviews of the ten best apps in 2020 in Google Play
sites. We also perform hyper-parameter tuning to find the optimum trained
model. Two training data labeling approaches were also tested to determine the
effectiveness of the model, which is score-based and lexicon-based. The
experimental results show that pre-trained models trained in Indonesian have
better average accuracy on lexicon-based data. The pre-trained Indonesian model
highest accuracy is 84%, with 25 epochs and a training time of 24 minutes.
These results are better than all of the machine learning and multilingual
pre-trained models.

    

### [[2107.06825] A Generalized Lottery Ticket Hypothesis](http://arxiv.org/abs/2107.06825)


  We introduce a generalization to the lottery ticket hypothesis in which the
notion of "sparsity" is relaxed by choosing an arbitrary basis in the space of
parameters. We present evidence that the original results reported for the
canonical basis continue to hold in this broader setting. We describe how
structured pruning methods, including pruning units or factorizing
fully-connected layers into products of low-rank matrices, can be cast as
particular instances of this "generalized" lottery ticket hypothesis. The
investigations reported here are preliminary and are provided to encourage
further research along this direction.

    

### [[2107.06845] Meta-Optimization of Deep CNN for Image Denoising Using LSTM](http://arxiv.org/abs/2107.06845)


  The recent application of deep learning (DL) to various tasks has seen the
performance of classical techniques surpassed by their DL-based counterparts.
As a result, DL has equally seen application in the removal of noise from
images. In particular, the use of deep feed-forward convolutional neural
networks (DnCNNs) has been investigated for denoising. It utilizes advances in
DL techniques such as deep architecture, residual learning, and batch
normalization to achieve better denoising performance when compared with the
other classical state-of-the-art denoising algorithms. However, its deep
architecture resulted in a huge set of trainable parameters. Meta-optimization
is a training approach of enabling algorithms to learn to train themselves by
themselves. Training algorithms using meta-optimizers have been shown to enable
algorithms to achieve better performance when compared to the classical
gradient descent-based training approach. In this work, we investigate the
application of the meta-optimization training approach to the DnCNN denoising
algorithm to enhance its denoising capability. Our preliminary experiments on
simpler algorithms reveal the prospects of utilizing the meta-optimization
training approach towards the enhancement of the DnCNN denoising capability.

    

### [[2107.06846] Extreme Precipitation Seasonal Forecast Using a Transformer Neural Network](http://arxiv.org/abs/2107.06846)


  An impact of climate change is the increase in frequency and intensity of
extreme precipitation events. However, confidently predicting the likelihood of
extreme precipitation at seasonal scales remains an outstanding challenge.
Here, we present an approach to forecasting the quantiles of the maximum daily
precipitation in each week up to six months ahead using the temporal fusion
transformer (TFT) model. Through experiments in two regions, we compare TFT
predictions with those of two baselines: climatology and a calibrated ECMWF
SEAS5 ensemble forecast (S5). Our results show that, in terms of quantile risk
at six month lead time, the TFT predictions significantly outperform those from
S5 and show an overall small improvement compared to climatology. The TFT also
responds positively to departures from normal that climatology cannot.

    

### [[2107.06859] A novel approach for modelling and classifying sit-to-stand kinematics using inertial sensors](http://arxiv.org/abs/2107.06859)


  Sit-to-stand transitions are an important part of activities of daily living
and play a key role in functional mobility in humans. The sit-to-stand movement
is often affected in older adults due to frailty and in patients with motor
impairments such as Parkinson's disease leading to falls. Studying kinematics
of sit-to-stand transitions can provide insight in assessment, monitoring and
developing rehabilitation strategies for the affected populations. We propose a
three-segment body model for estimating sit-to-stand kinematics using only two
wearable inertial sensors, placed on the shank and back. Reducing the number of
sensors to two instead of one per body segment facilitates monitoring and
classifying movements over extended periods, making it more comfortable to wear
while reducing the power requirements of sensors. We applied this model on 10
younger healthy adults (YH), 12 older healthy adults (OH) and 12 people with
Parkinson's disease (PwP). We have achieved this by incorporating unique
sit-to-stand classification technique using unsupervised learning in the model
based reconstruction of angular kinematics using extended Kalman filter. Our
proposed model showed that it was possible to successfully estimate thigh
kinematics despite not measuring the thigh motion with inertial sensor. We
classified sit-to-stand transitions, sitting and standing states with the
accuracies of 98.67%, 94.20% and 91.41% for YH, OH and PwP respectively. We
have proposed a novel integrated approach of modelling and classification for
estimating the body kinematics during sit-to-stand motion and successfully
applied it on YH, OH and PwP groups.

    

### [[2107.06862] Differentiable Programming of Reaction-Diffusion Patterns](http://arxiv.org/abs/2107.06862)


  Reaction-Diffusion (RD) systems provide a computational framework that
governs many pattern formation processes in nature. Current RD system design
practices boil down to trial-and-error parameter search. We propose a
differentiable optimization method for learning the RD system parameters to
perform example-based texture synthesis on a 2D plane. We do this by
representing the RD system as a variant of Neural Cellular Automata and using
task-specific differentiable loss functions. RD systems generated by our method
exhibit robust, non-trivial 'life-like' behavior.

    

### [[2107.06865] Exploiting Spiking Dynamics with Spatial-temporal Feature Normalization in Graph Learning](http://arxiv.org/abs/2107.06865)


  Biological spiking neurons with intrinsic dynamics underlie the powerful
representation and learning capabilities of the brain for processing multimodal
information in complex environments. Despite recent tremendous progress in
spiking neural networks (SNNs) for handling Euclidean-space tasks, it still
remains challenging to exploit SNNs in processing non-Euclidean-space data
represented by graph data, mainly due to the lack of effective modeling
framework and useful training techniques. Here we present a general spike-based
modeling framework that enables the direct training of SNNs for graph learning.
Through spatial-temporal unfolding for spiking data flows of node features, we
incorporate graph convolution filters into spiking dynamics and formalize a
synergistic learning paradigm. Considering the unique features of spike
representation and spiking dynamics, we propose a spatial-temporal feature
normalization (STFN) technique suitable for SNN to accelerate convergence. We
instantiate our methods into two spiking graph models, including graph
convolution SNNs and graph attention SNNs, and validate their performance on
three node-classification benchmarks, including Cora, Citeseer, and Pubmed. Our
model can achieve comparable performance with the state-of-the-art graph neural
network (GNN) models with much lower computation costs, demonstrating great
benefits for the execution on neuromorphic hardware and prompting neuromorphic
applications in graphical scenarios.

    

### [[2107.06869] Core-set Sampling for Efficient Neural Architecture Search](http://arxiv.org/abs/2107.06869)


  Neural architecture search (NAS), an important branch of automatic machine
learning, has become an effective approach to automate the design of deep
learning models. However, the major issue in NAS is how to reduce the large
search time imposed by the heavy computational burden. While most recent
approaches focus on pruning redundant sets or developing new search
methodologies, this paper attempts to formulate the problem based on the data
curation manner. Our key strategy is to search the architecture using
summarized data distribution, i.e., core-set. Typically, many NAS algorithms
separate searching and training stages, and the proposed core-set methodology
is only used in search stage, thus their performance degradation can be
minimized. In our experiments, we were able to save overall computational time
from 30.8 hours to 3.5 hours, 8.8x reduction, on a single RTX 3090 GPU without
sacrificing accuracy.

    

### [[2107.06871] Uncertainty Modeling of Emerging Device-based Computing-in-Memory Neural Accelerators with Application to Neural Architecture Search](http://arxiv.org/abs/2107.06871)


  Emerging device-based Computing-in-memory (CiM) has been proved to be a
promising candidate for high-energy efficiency deep neural network (DNN)
computations. However, most emerging devices suffer uncertainty issues,
resulting in a difference between actual data stored and the weight value it is
designed to be. This leads to an accuracy drop from trained models to actually
deployed platforms. In this work, we offer a thorough analysis of the effect of
such uncertainties-induced changes in DNN models. To reduce the impact of
device uncertainties, we propose UAE, an uncertainty-aware Neural Architecture
Search scheme to identify a DNN model that is both accurate and robust against
device uncertainties.

    

### [[2107.06872] Generalisation in Neural Networks Does not Require Feature Overlap](http://arxiv.org/abs/2107.06872)


  That shared features between train and test data are required for
generalisation in artificial neural networks has been a common assumption of
both proponents and critics of these models. Here, we show that convolutional
architectures avoid this limitation by applying them to two well known
challenges, based on learning the identity function and learning rules
governing sequences of words. In each case, successful performance on the test
set requires generalising to features that were not present in the training
data, which is typically not feasible for standard connectionist models.
However, our experiments demonstrate that neural networks can succeed on such
problems when they incorporate the weight sharing employed by convolutional
architectures. In the image processing domain, such architectures are intended
to reflect the symmetry under spatial translations of the natural world that
such images depict. We discuss the role of symmetry in the two tasks and its
connection to generalisation.

    

### [[2107.06875] DULA: A Differentiable Ergonomics Model for Postural Optimization in Physical HRI](http://arxiv.org/abs/2107.06875)


  Ergonomics and human comfort are essential concerns in physical human-robot
interaction applications. Defining an accurate and easy-to-use ergonomic
assessment model stands as an important step in providing feedback for postural
correction to improve operator health and comfort. In order to enable efficient
computation, previously proposed automated ergonomic assessment and correction
tools make approximations or simplifications to gold-standard assessment tools
used by ergonomists in practice. In order to retain assessment quality, while
improving computational considerations, we introduce DULA, a differentiable and
continuous ergonomics model learned to replicate the popular and scientifically
validated RULA assessment. We show that DULA provides assessment comparable to
RULA while providing computational benefits. We highlight DULA's strength in a
demonstration of gradient-based postural optimization for a simulated
teleoperation task.

    

### [[2107.06876] Scalable Optimal Transport in High Dimensions for Graph Distances, Embedding Alignment, and More](http://arxiv.org/abs/2107.06876)


  The current best practice for computing optimal transport (OT) is via entropy
regularization and Sinkhorn iterations. This algorithm runs in quadratic time
as it requires the full pairwise cost matrix, which is prohibitively expensive
for large sets of objects. In this work we propose two effective log-linear
time approximations of the cost matrix: First, a sparse approximation based on
locality-sensitive hashing (LSH) and, second, a Nyström approximation with
LSH-based sparse corrections, which we call locally corrected Nyström (LCN).
These approximations enable general log-linear time algorithms for
entropy-regularized OT that perform well even for the complex, high-dimensional
spaces common in deep learning. We analyse these approximations theoretically
and evaluate them experimentally both directly and end-to-end as a component
for real-world applications. Using our approximations for unsupervised word
embedding alignment enables us to speed up a state-of-the-art method by a
factor of 3 while also improving the accuracy by 3.1 percentage points without
any additional model changes. For graph distance regression we propose the
graph transport network (GTN), which combines graph neural networks (GNNs) with
enhanced Sinkhorn. GTN outcompetes previous models by 48% and still scales
log-linearly in the number of nodes.

    

### [[2107.06877] Federated Self-Training for Semi-Supervised Audio Recognition](http://arxiv.org/abs/2107.06877)


  Federated Learning is a distributed machine learning paradigm dealing with
decentralized and personal datasets. Since data reside on devices like
smartphones and virtual assistants, labeling is entrusted to the clients, or
labels are extracted in an automated way. Specifically, in the case of audio
data, acquiring semantic annotations can be prohibitively expensive and
time-consuming. As a result, an abundance of audio data remains unlabeled and
unexploited on users' devices. Most existing federated learning approaches
focus on supervised learning without harnessing the unlabeled data. In this
work, we study the problem of semi-supervised learning of audio models via
self-training in conjunction with federated learning. We propose FedSTAR to
exploit large-scale on-device unlabeled data to improve the generalization of
audio recognition models. We further demonstrate that self-supervised
pre-trained models can accelerate the training of on-device models,
significantly improving convergence to within fewer training rounds. We conduct
experiments on diverse public audio classification datasets and investigate the
performance of our models under varying percentages of labeled and unlabeled
data. Notably, we show that with as little as 3% labeled data available,
FedSTAR on average can improve the recognition rate by 13.28% compared to the
fully supervised federated model.

    

### [[2107.06882] Conservative Objective Models for Effective Offline Model-Based Optimization](http://arxiv.org/abs/2107.06882)


  Computational design problems arise in a number of settings, from synthetic
biology to computer architectures. In this paper, we aim to solve data-driven
model-based optimization (MBO) problems, where the goal is to find a design
input that maximizes an unknown objective function provided access to only a
static dataset of prior experiments. Such data-driven optimization procedures
are the only practical methods in many real-world domains where active data
collection is expensive (e.g., when optimizing over proteins) or dangerous
(e.g., when optimizing over aircraft designs). Typical methods for MBO that
optimize the design against a learned model suffer from distributional shift:
it is easy to find a design that "fools" the model into predicting a high
value. To overcome this, we propose conservative objective models (COMs), a
method that learns a model of the objective function that lower bounds the
actual value of the ground-truth objective on out-of-distribution inputs, and
uses it for optimization. Structurally, COMs resemble adversarial training
methods used to overcome adversarial examples. COMs are simple to implement and
outperform a number of existing methods on a wide range of MBO problems,
including optimizing protein sequences, robot morphologies, neural network
weights, and superconducting materials.

    

### [[1806.00421] Solving the Kolmogorov PDE by means of deep learning](http://arxiv.org/abs/1806.00421)


  Stochastic differential equations (SDEs) and the Kolmogorov partial
differential equations (PDEs) associated to them have been widely used in
models from engineering, finance, and the natural sciences. In particular, SDEs
and Kolmogorov PDEs, respectively, are highly employed in models for the
approximative pricing of financial derivatives. Kolmogorov PDEs and SDEs,
respectively, can typically not be solved explicitly and it has been and still
is an active topic of research to design and analyze numerical methods which
are able to approximately solve Kolmogorov PDEs and SDEs, respectively. Nearly
all approximation methods for Kolmogorov PDEs in the literature suffer under
the curse of dimensionality or only provide approximations of the solution of
the PDE at a single fixed space-time point. In this paper we derive and propose
a numerical approximation method which aims to overcome both of the above
mentioned drawbacks and intends to deliver a numerical approximation of the
Kolmogorov PDE on an entire region $[a,b]^d$ without suffering from the curse
of dimensionality. Numerical results on examples including the heat equation,
the Black-Scholes model, the stochastic Lorenz equation, and the Heston model
suggest that the proposed approximation algorithm is quite effective in high
dimensions in terms of both accuracy and speed.

    

### [[1904.08352] MOSNet: Deep Learning based Objective Assessment for Voice Conversion](http://arxiv.org/abs/1904.08352)


  Existing objective evaluation metrics for voice conversion (VC) are not
always correlated with human perception. Therefore, training VC models with
such criteria may not effectively improve naturalness and similarity of
converted speech. In this paper, we propose deep learning-based assessment
models to predict human ratings of converted speech. We adopt the convolutional
and recurrent neural network models to build a mean opinion score (MOS)
predictor, termed as MOSNet. The proposed models are tested on large-scale
listening test results of the Voice Conversion Challenge (VCC) 2018.
Experimental results show that the predicted scores of the proposed MOSNet are
highly correlated with human MOS ratings at the system level while being fairly
correlated with human MOS ratings at the utterance level. Meanwhile, we have
modified MOSNet to predict the similarity scores, and the preliminary results
show that the predicted scores are also fairly correlated with human ratings.
These results confirm that the proposed models could be used as a computational
evaluator to measure the MOS of VC systems to reduce the need for expensive
human rating.

    

### [[1908.03464] Zero-Shot Feature Selection via Transferring Supervised Knowledge](http://arxiv.org/abs/1908.03464)


  Feature selection, an effective technique for dimensionality reduction, plays
an important role in many machine learning systems. Supervised knowledge can
significantly improve the performance. However, faced with the rapid growth of
newly emerging concepts, existing supervised methods might easily suffer from
the scarcity and validity of labeled data for training. In this paper, the
authors study the problem of zero-shot feature selection (i.e., building a
feature selection model that generalizes well to "unseen" concepts with limited
training data of "seen" concepts). Specifically, they adopt class-semantic
descriptions (i.e., attributes) as supervision for feature selection, so as to
utilize the supervised knowledge transferred from the seen concepts. For more
reliable discriminative features, they further propose the
center-characteristic loss which encourages the selected features to capture
the central characteristics of seen concepts. Extensive experiments conducted
on various real-world datasets demonstrate the effectiveness of the method.

    

### [[1911.03886] Performance Analysis on Machine Learning-Based Channel Estimation](http://arxiv.org/abs/1911.03886)


  Recently, machine learning-based channel estimation has attracted much
attention. The performance of machine learning-based estimation has been
validated by simulation experiments. However, little attention has been paid to
the theoretical performance analysis. In this paper, we investigate the mean
square error (MSE) performance of machine learning-based estimation. Hypothesis
testing is employed to analyze its MSE upper bound. Furthermore, we build a
statistical model for hypothesis testing, which holds when the linear learning
module with a low input dimension is used in machine learning-based channel
estimation, and derive a clear analytical relation between the size of the
training data and performance. Then, we simulate the machine learning-based
channel estimation in orthogonal frequency division multiplexing (OFDM) systems
to verify our analysis results. Finally, the design considerations for the
situation where only limited training data is available are discussed. In this
situation, our analysis results can be applied to assess the performance and
support the design of machine learning-based channel estimation.

    

### [[2003.08938] Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations](http://arxiv.org/abs/2003.08938)


  A deep reinforcement learning (DRL) agent observes its states through
observations, which may contain natural measurement errors or adversarial
noises. Since the observations deviate from the true states, they can mislead
the agent into making suboptimal actions. Several works have shown this
vulnerability via adversarial attacks, but existing approaches on improving the
robustness of DRL under this setting have limited success and lack for
theoretical principles. We show that naively applying existing techniques on
improving robustness for classification tasks, like adversarial training, is
ineffective for many RL tasks. We propose the state-adversarial Markov decision
process (SA-MDP) to study the fundamental properties of this problem, and
develop a theoretically principled policy regularization which can be applied
to a large family of DRL algorithms, including proximal policy optimization
(PPO), deep deterministic policy gradient (DDPG) and deep Q networks (DQN), for
both discrete and continuous action control problems. We significantly improve
the robustness of PPO, DDPG and DQN agents under a suite of strong white box
adversarial attacks, including new attacks of our own. Additionally, we find
that a robust policy noticeably improves DRL performance even without an
adversary in a number of environments. Our code is available at
this https URL.

    

### [[2006.02804] Exploring the Potential of Low-bit Training of Convolutional Neural Networks](http://arxiv.org/abs/2006.02804)


  In this work, we propose a low-bit training framework for convolutional
neural networks, which is built around a novel multi-level scaling (MLS) tensor
format. Our framework focuses on reducing the energy consumption of convolution
operations by quantizing all the convolution operands to low bit-width format.
Specifically, we propose the MLS tensor format, in which the element-wise
bit-width can be largely reduced. Then, we describe the dynamic quantization
and the low-bit tensor convolution arithmetic to leverage the MLS tensor format
efficiently. Experiments show that our framework achieves a superior trade-off
between the accuracy and the bit-width than previous low-bit training
frameworks. For training a variety of models on CIFAR-10, using 1-bit mantissa
and 2-bit exponent is adequate to keep the accuracy loss within $1\%$. And on
larger datasets like ImageNet, using 4-bit mantissa and 2-bit exponent is
adequate to keep the accuracy loss within $1\%$. Through the energy consumption
simulation of the computing units, we can estimate that training a variety of
models with our framework could achieve $8.3\sim10.2\times$ and
$1.9\sim2.3\times$ higher energy efficiency than training with full-precision
and 8-bit floating-point arithmetic, respectively.

    

### [[2006.05732] Object Detection in the DCT Domain: is Luminance the Solution?](http://arxiv.org/abs/2006.05732)


  Object detection in images has reached unprecedented performances. The
state-of-the-art methods rely on deep architectures that extract salient
features and predict bounding boxes enclosing the objects of interest. These
methods essentially run on RGB images. However, the RGB images are often
compressed by the acquisition devices for storage purpose and transfer
efficiency. Hence, their decompression is required for object detectors. To
gain in efficiency, this paper proposes to take advantage of the compressed
representation of images to carry out object detection usable in constrained
resources conditions.
Specifically, we focus on JPEG images and propose a thorough analysis of
detection architectures newly designed in regard of the peculiarities of the
JPEG norm. This leads to a $\times 1.7$ speed up in comparison with a standard
RGB-based architecture, while only reducing the detection performance by 5.5%.
Additionally, our empirical findings demonstrate that only part of the
compressed JPEG information, namely the luminance component, may be required to
match detection accuracy of the full input methods.

    

### [[2007.06093] Interval Universal Approximation for Neural Networks](http://arxiv.org/abs/2007.06093)


  To verify safety and robustness of neural networks, researchers have
successfully applied abstract interpretation, primarily using the interval
abstract domain. In this paper, we study the theoretical power and limits of
the interval domain for neural-network verification.
First, we introduce the interval universal approximation (IUA) theorem. IUA
shows that neural networks not only can approximate any continuous function $f$
(universal approximation) as we have known for decades, but we can find a
neural network, using any well-behaved activation function, whose interval
bounds are an arbitrarily close approximation of the set semantics of $f$ (the
result of applying $f$ to a set of inputs). We call this notion of
approximation interval approximation. Our theorem generalizes the recent result
of Baader et al. (2020) from ReLUs to a rich class of activation functions that
we call squashable functions. Additionally, the IUA theorem implies that we can
always construct provably robust neural networks under $\ell_\infty$-norm using
almost any practical activation function.
Second, we study the computational complexity of constructing neural networks
that are amenable to precise interval analysis. This is a crucial question, as
our constructive proof of IUA is exponential in the size of the approximation
domain. We boil this question down to the problem of approximating the range of
a neural network with squashable activation functions. We show that the range
approximation problem (RA) is a $\Delta_2$-intermediate problem, which is
strictly harder than $\mathsf{NP}$-complete problems, assuming
$\mathsf{coNP}\not\subset \mathsf{NP}$. As a result, IUA is an inherently hard
problem: No matter what abstract domain or computational tools we consider to
achieve interval approximation, there is no efficient construction of such a
universal approximator.

    

### [[2009.13370] Replica Analysis of the Linear Model with Markov or Hidden Markov Signal Priors](http://arxiv.org/abs/2009.13370)


  This paper estimates free energy, average mutual information, and minimum
mean square error (MMSE) of a linear model under two assumptions: (1) the
source is generated by a Markov chain, (2) the source is generated via a hidden
Markov model. Our estimates are based on the replica method in statistical
physics. We show that under the posterior mean estimator, the linear model with
Markov sources or hidden Markov sources is decoupled into single-input AWGN
channels with state information available at both encoder and decoder where the
state distribution follows the left Perron-Frobenius eigenvector with unit
Manhattan norm of the stochastic matrix of Markov chains. Numerical results
show that the free energies and MSEs obtained via the replica method closely
approximate to their counterparts achieved by the Metropolis-Hastings algorithm
or some well-known approximate message passing algorithms in the research
literature.

    

### [[2009.13998] How Do You Want Your Greedy: Simultaneous or Repeated?](http://arxiv.org/abs/2009.13998)


  We present SimultaneousGreedys, a deterministic algorithm for constrained
submodular maximization. At a high level, the algorithm maintains $\ell$
solutions and greedily updates them in a simultaneous fashion.
SimultaneousGreedys achieves the tightest known approximation guarantees for
both $k$-extendible systems and the more general $k$-systems, which are
$(k+1)^2/k = k + \mathcal{O}(1)$ and $(1 + \sqrt{k+2})^2 = k +
\mathcal{O}(\sqrt{k})$, respectively. This is in contrast to previous
algorithms, which are designed to provide tight approximation guarantees in one
setting, but not both. We also improve the analysis of RepeatedGreedy, showing
that it achieves an approximation ratio of $k + \mathcal{O}(\sqrt{k})$ for
$k$-systems when allowed to run for $\mathcal{O}(\sqrt{k})$ iterations, an
improvement in both the runtime and approximation over previous analyses. We
demonstrate that both algorithms may be modified to run in nearly linear time
with an arbitrarily small loss in the approximation.
Both SimultaneousGreedys and RepeatedGreedy are flexible enough to
incorporate the intersection of $m$ additional knapsack constraints, while
retaining similar approximation guarantees: both algorithms yield an
approximation guarantee of roughly $k + 2m + \mathcal{O}(\sqrt{k+m})$ for
$k$-systems and SimultaneousGreedys enjoys an improved approximation guarantee
of $k+2m + \mathcal{O}(\sqrt{m})$ for $k$-extendible systems. To complement our
algorithmic contributions, we provide a hardness result which states that no
algorithm making polynomially many oracle queries can achieve an approximation
better than $k + 1/2 + \varepsilon$. We also present SubmodularGreedy.jl, a
Julia package which implements these algorithms and may be downloaded at
this https URL . Finally, we test the
effectiveness of these algorithms on real datasets.

    

### [[2010.13275] Asymptotic Behavior of Adversarial Training in Binary Classification](http://arxiv.org/abs/2010.13275)


  It has been consistently reported that many machine learning models are
susceptible to adversarial attacks i.e., small additive adversarial
perturbations applied to data points can cause misclassification. Adversarial
training using empirical risk minimization is considered to be the
state-of-the-art method for defense against adversarial attacks. Despite being
successful in practice, several problems in understanding generalization
performance of adversarial training remain open. In this paper, we derive
precise theoretical predictions for the performance of adversarial training in
binary classification. We consider the high-dimensional regime where the
dimension of data grows with the size of the training data-set at a constant
ratio. Our results provide exact asymptotics for standard and adversarial test
errors of the estimators obtained by adversarial training with $\ell_q$-norm
bounded perturbations ($q \ge 1$) for both discriminative binary models and
generative Gaussian-mixture models with correlated features. Furthermore, we
use these sharp predictions to uncover several intriguing observations on the
role of various parameters including the over-parameterization ratio, the data
model, and the attack budget on the adversarial and standard errors.

    

### [[2010.14227] Efficient, Simple and Automated Negative Sampling for Knowledge Graph Embedding](http://arxiv.org/abs/2010.14227)


  Negative sampling, which samples negative triplets from non-observed ones in
knowledge graph (KG), is an essential step in KG embedding. Recently,
generative adversarial network (GAN), has been introduced in negative sampling.
By sampling negative triplets with large gradients, these methods avoid the
problem of vanishing gradient and thus obtain better performance. However, they
make the original model more complex and harder to train. In this paper,
motivated by the observation that negative triplets with large gradients are
important but rare, we propose to directly keep track of them with the cache.
In this way, our method acts as a "distilled" version of previous GAN-based
methods, which does not waste training time on additional parameters to fit the
full distribution of negative triplets. However, how to sample from and update
the cache are two critical questions. We propose to solve these issues by
automated machine learning techniques. The automated version also covers
GAN-based methods as special cases. Theoretical explanation of NSCaching is
also provided, justifying the superior over fixed sampling scheme. Besides, we
further extend NSCaching with skip-gram model for graph embedding. Finally,
extensive experiments show that our method can gain significant improvements on
various KG embedding models and the skip-gram model, and outperforms the
state-of-the-art negative sampling methods.

    

### [[2011.08694] Reactive Long Horizon Task Execution via Visual Skill and Precondition Models](http://arxiv.org/abs/2011.08694)


  Zero-shot execution of unseen robotic tasks is important to allowing robots
to perform a wide variety of tasks in human environments, but collecting the
amounts of data necessary to train end-to-end policies in the real-world is
often infeasible. We describe an approach for sim-to-real training that can
accomplish unseen robotic tasks using models learned in simulation to ground
components of a simple task planner. We learn a library of parameterized
skills, along with a set of predicates-based preconditions and termination
conditions, entirely in simulation. We explore a block-stacking task because it
has a clear structure, where multiple skills must be chained together, but our
methods are applicable to a wide range of other problems and domains, and can
transfer from simulation to the real-world with no fine tuning. The system is
able to recognize failures and accomplish long-horizon tasks from perceptual
input, which is critical for real-world execution. We evaluate our proposed
approach in both simulation and in the real-world, showing an increase in
success rate from 91.6% to 98% in simulation and from 10% to 80% success rate
in the real-world as compared with naive baselines. For experiment videos
including both real-world and simulation, see:
this https URL


### [[2011.14696] On Initial Pools for Deep Active Learning](http://arxiv.org/abs/2011.14696)


  Active Learning (AL) techniques aim to minimize the training data required to
train a model for a given task. Pool-based AL techniques start with a small
initial labeled pool and then iteratively pick batches of the most informative
samples for labeling. Generally, the initial pool is sampled randomly and
labeled to seed the AL iterations. While recent studies have focused on
evaluating the robustness of various query functions in AL, little to no
attention has been given to the design of the initial labeled pool for deep
active learning. Given the recent successes of learning representations in
self-supervised/unsupervised ways, we study if an intelligently sampled initial
labeled pool can improve deep AL performance. We investigate the effect of
intelligently sampled initial labeled pools, including the use of
self-supervised and unsupervised strategies, on deep AL methods. The setup,
hypotheses, methodology, and implementation details were evaluated by peer
review before experiments were conducted. Experimental results could not
conclusively prove that intelligently sampled initial pools are better for AL
than random initial pools in the long run, although a Variational
Autoencoder-based initial pool sampling strategy showed interesting trends that
merit deeper investigation.

    

### [[2012.08791] MINIROCKET: A Very Fast (Almost) Deterministic Transform for Time Series Classification](http://arxiv.org/abs/2012.08791)


  Until recently, the most accurate methods for time series classification were
limited by high computational complexity. ROCKET achieves state-of-the-art
accuracy with a fraction of the computational expense of most existing methods
by transforming input time series using random convolutional kernels, and using
the transformed features to train a linear classifier. We reformulate ROCKET
into a new method, MINIROCKET, making it up to 75 times faster on larger
datasets, and making it almost deterministic (and optionally, with additional
computational expense, fully deterministic), while maintaining essentially the
same accuracy. Using this method, it is possible to train and test a classifier
on all of 109 datasets from the UCR archive to state-of-the-art accuracy in
less than 10 minutes. MINIROCKET is significantly faster than any other method
of comparable accuracy (including ROCKET), and significantly more accurate than
any other method of even roughly-similar computational expense. As such, we
suggest that MINIROCKET should now be considered and used as the default
variant of ROCKET.

    

### [[2012.13376] A Physics-Informed Deep Learning Paradigm for Car-Following Models](http://arxiv.org/abs/2012.13376)


  Car-following behavior has been extensively studied using physics-based
models, such as the Intelligent Driver Model. These models successfully
interpret traffic phenomena observed in the real-world but may not fully
capture the complex cognitive process of driving. Deep learning models, on the
other hand, have demonstrated their power in capturing observed traffic
phenomena but require a large amount of driving data to train. This paper aims
to develop a family of neural network based car-following models that are
informed by physics-based models, which leverage the advantage of both
physics-based (being data-efficient and interpretable) and deep learning based
(being generalizable) models. We design physics-informed deep learning
car-following (PIDL-CF) architectures encoded with two popular physics-based
models - IDM and OVM, on which acceleration is predicted for four traffic
regimes: acceleration, deceleration, cruising, and emergency braking. Two types
of PIDL-CFM problems are studied, one to predict acceleration only and the
other to jointly predict acceleration and discover model parameters. We also
demonstrate the superior performance of PIDL with the Next Generation
SIMulation (NGSIM) dataset over baselines, especially when the training data is
sparse. The results demonstrate the superior performance of neural networks
informed by physics over those without. The developed PIDL-CF framework holds
the potential for system identification of driving models and for the
development of driving-based controls for automated vehicles.

    

### [[2012.13869] Neural Closure Models for Dynamical Systems](http://arxiv.org/abs/2012.13869)


  Complex dynamical systems are used for predictions in many domains. Because
of computational costs, models are truncated, coarsened, or aggregated. As the
neglected and unresolved terms become important, the utility of model
predictions diminishes. We develop a novel, versatile, and rigorous methodology
to learn non-Markovian closure parameterizations for known-physics/low-fidelity
models using data from high-fidelity simulations. The new "neural closure
models" augment low-fidelity models with neural delay differential equations
(nDDEs), motivated by the Mori-Zwanzig formulation and the inherent delays in
complex dynamical systems. We demonstrate that neural closures efficiently
account for truncated modes in reduced-order-models, capture the effects of
subgrid-scale processes in coarse models, and augment the simplification of
complex biological and physical-biogeochemical models. We find that using
non-Markovian over Markovian closures improves long-term prediction accuracy
and requires smaller networks. We derive adjoint equations and network
architectures needed to efficiently implement the new discrete and distributed
nDDEs, for any time-integration schemes and allowing nonuniformly-spaced
temporal training data. The performance of discrete over distributed delays in
closure models is explained using information theory, and we find an optimal
amount of past information for a specified architecture. Finally, we analyze
computational complexity and explain the limited additional cost due to neural
closure models.

    

### [[2101.02137] Smoothed functional-based gradient algorithms for off-policy reinforcement learning: A non-asymptotic viewpoint](http://arxiv.org/abs/2101.02137)


  We propose two policy gradient algorithms for solving the problem of control
in an off-policy reinforcement learning (RL) context. Both algorithms
incorporate a smoothed functional (SF) based gradient estimation scheme. The
first algorithm is a straightforward combination of importance sampling-based
off-policy evaluation with SF-based gradient estimation. The second algorithm,
inspired by the stochastic variance-reduced gradient (SVRG) algorithm,
incorporates variance reduction in the update iteration. For both algorithms,
we derive non-asymptotic bounds that establish convergence to an approximate
stationary point. From these results, we infer that the first algorithm
converges at a rate that is comparable to the well-known REINFORCE algorithm in
an off-policy RL context, while the second algorithm exhibits an improved rate
of convergence.

    

### [[2101.11156] Fundamental limits and algorithms for sparse linear regression with sublinear sparsity](http://arxiv.org/abs/2101.11156)


  We establish exact asymptotic expressions for the normalized mutual
information and minimum mean-square-error (MMSE) of sparse linear regression in
the sub-linear sparsity regime. Our result is achieved by a generalization of
the adaptive interpolation method in Bayesian inference for linear regimes to
sub-linear ones. A modification of the well-known approximate message passing
algorithm to approach the MMSE fundamental limit is also proposed, and its
state evolution is rigorously analyzed. Our results show that the traditional
linear assumption between the signal dimension and number of observations in
the replica and adaptive interpolation methods is not necessary for sparse
signals. They also show how to modify the existing well-known AMP algorithms
for linear regimes to sub-linear ones.

    

### [[2102.00904] Text-to-hashtag Generation using Seq2seq Learning](http://arxiv.org/abs/2102.00904)


  In this paper, we studied whether models based on BiLSTM and BERT can predict
hashtags in Brazilian Portuguese for Ecommerce websites. Hashtags have a
sizable financial impact on Ecommerce. We processed a corpus of Ecommerce
reviews as inputs, and predicted hashtags as outputs. We evaluated the results
using four quantitative metrics: NIST, BLEU, METEOR and a crowdsourced score. A
word cloud was used as a qualitative metric. While all computer-generated
metrics (NIST, BLEU and METEOR) indicated bad results, the crowdsourced results
produced amazing scores. We concluded that the texts predicted by the neural
networks are very promising for use as hashtags for products on Ecommerce
websites. The code for this work is available at
this https URL.

    

### [[2102.04313] Long-time simulations with high fidelity on quantum hardware](http://arxiv.org/abs/2102.04313)


  Moderate-size quantum computers are now publicly accessible over the cloud,
opening the exciting possibility of performing dynamical simulations of quantum
systems. However, while rapidly improving, these devices have short coherence
times, limiting the depth of algorithms that may be successfully implemented.
Here we demonstrate that, despite these limitations, it is possible to
implement long-time, high fidelity simulations on current hardware.
Specifically, we simulate an XY-model spin chain on the Rigetti and IBM quantum
computers, maintaining a fidelity of at least 0.9 for over 600 time steps. This
is a factor of 150 longer than is possible using the iterated Trotter method.
Our simulations are performed using a new algorithm that we call the fixed
state Variational Fast Forwarding (fsVFF) algorithm. This algorithm decreases
the circuit depth and width required for a quantum simulation by finding an
approximate diagonalization of a short time evolution unitary. Crucially, fsVFF
only requires finding a diagonalization on the subspace spanned by the initial
state, rather than on the total Hilbert space as with previous methods,
substantially reducing the required resources. We further demonstrate the
viability of fsVFF through large numerical implementations of the algorithm, as
well as an analysis of its noise resilience and the scaling of simulation
errors.

    

### [[2102.05291] Clusterability as an Alternative to Anchor Points When Learning with Noisy Labels](http://arxiv.org/abs/2102.05291)


  The label noise transition matrix, characterizing the probabilities of a
training instance being wrongly annotated, is crucial to designing popular
solutions to learning with noisy labels. Existing works heavily rely on finding
"anchor points" or their approximates, defined as instances belonging to a
particular class almost surely. Nonetheless, finding anchor points remains a
non-trivial task, and the estimation accuracy is also often throttled by the
number of available anchor points. In this paper, we propose an alternative
option to the above task. Our main contribution is the discovery of an
efficient estimation procedure based on a clusterability condition. We prove
that with clusterable representations of features, using up to third-order
consensuses of noisy labels among neighbor representations is sufficient to
estimate a unique transition matrix. Compared with methods using anchor points,
our approach uses substantially more instances and benefits from a much better
sample complexity. We demonstrate the estimation accuracy and advantages of our
estimates using both synthetic noisy labels (on CIFAR-10/100) and real
human-level noisy labels (on Clothing1M and our self-collected human-annotated
CIFAR-10). Our code and human-level noisy CIFAR-10 labels are available at
this https URL.

    

### [[2102.08817] Dissecting Supervised Contrastive Learning](http://arxiv.org/abs/2102.08817)


  Minimizing cross-entropy over the softmax scores of a linear map composed
with a high-capacity encoder is arguably the most popular choice for training
neural networks on supervised learning tasks. However, recent works show that
one can directly optimize the encoder instead, to obtain equally (or even more)
discriminative representations via a supervised variant of a contrastive
objective. In this work, we address the question whether there are fundamental
differences in the sought-for representation geometry in the output space of
the encoder at minimal loss. Specifically, we prove, under mild assumptions,
that both losses attain their minimum once the representations of each class
collapse to the vertices of a regular simplex, inscribed in a hypersphere. We
provide empirical evidence that this configuration is attained in practice and
that reaching a close-to-optimal state typically indicates good generalization
performance. Yet, the two losses show remarkably different optimization
behavior. The number of iterations required to perfectly fit to data scales
superlinearly with the amount of randomly flipped labels for the supervised
contrastive loss. This is in contrast to the approximately linear scaling
previously reported for networks trained with cross-entropy.

    

### [[2103.01926] Slow-Growing Trees](http://arxiv.org/abs/2103.01926)


  Random Forest's performance can be matched by a single slow-growing tree
(SGT), which uses a learning rate to tame CART's greedy algorithm. SGT exploits
the view that CART is an extreme case of an iterative weighted least square
procedure. Moreover, a unifying view of Boosted Trees (BT) and Random Forests
(RF) is presented. Greedy ML algorithms' outcomes can be improved using either
"slow learning" or diversification. SGT applies the former to estimate a single
deep tree, and Booging (bagging stochastic BT with a high learning rate) uses
the latter with additive shallow trees. The performance of this tree ensemble
quaternity (Booging, BT, SGT, RF) is assessed on simulated and real regression
tasks.

    

### [[2103.03158] A Structural Causal Model for MR Images of Multiple Sclerosis](http://arxiv.org/abs/2103.03158)


  Precision medicine involves answering counterfactual questions such as "Would
this patient respond better to treatment A or treatment B?" These types of
questions are causal in nature and require the tools of causal inference to be
answered, e.g., with a structural causal model (SCM). In this work, we develop
an SCM that models the interaction between demographic information, disease
covariates, and magnetic resonance (MR) images of the brain for people with
multiple sclerosis. Inference in the SCM generates counterfactual images that
show what an MR image of the brain would look like if demographic or disease
covariates are changed. These images can be used for modeling disease
progression or used for image processing tasks where controlling for
confounders is necessary.

    

### [[2103.04217] Spectral Tensor Train Parameterization of Deep Learning Layers](http://arxiv.org/abs/2103.04217)


  We study low-rank parameterizations of weight matrices with embedded spectral
properties in the Deep Learning context. The low-rank property leads to
parameter efficiency and permits taking computational shortcuts when computing
mappings. Spectral properties are often subject to constraints in optimization
problems, leading to better models and stability of optimization. We start by
looking at the compact SVD parameterization of weight matrices and identifying
redundancy sources in the parameterization. We further apply the Tensor Train
(TT) decomposition to the compact SVD components, and propose a non-redundant
differentiable parameterization of fixed TT-rank tensor manifolds, termed the
Spectral Tensor Train Parameterization (STTP). We demonstrate the effects of
neural network compression in the image classification setting and both
compression and improved training stability in the generative adversarial
training setting.

    

### [[2103.05630] ForgeryNet: A Versatile Benchmark for Comprehensive Forgery Analysis](http://arxiv.org/abs/2103.05630)


  The rapid progress of photorealistic synthesis techniques has reached at a
critical point where the boundary between real and manipulated images starts to
blur. Thus, benchmarking and advancing digital forgery analysis have become a
pressing issue. However, existing face forgery datasets either have limited
diversity or only support coarse-grained analysis. To counter this emerging
threat, we construct the ForgeryNet dataset, an extremely large face forgery
dataset with unified annotations in image- and video-level data across four
tasks: 1) Image Forgery Classification, including two-way (real / fake),
three-way (real / fake with identity-replaced forgery approaches / fake with
identity-remained forgery approaches), and n-way (real and 15 respective
forgery approaches) classification. 2) Spatial Forgery Localization, which
segments the manipulated area of fake images compared to their corresponding
source real images. 3) Video Forgery Classification, which re-defines the
video-level forgery classification with manipulated frames in random positions.
This task is important because attackers in real world are free to manipulate
any target frame. and 4) Temporal Forgery Localization, to localize the
temporal segments which are manipulated. ForgeryNet is by far the largest
publicly available deep face forgery dataset in terms of data-scale (2.9
million images, 221,247 videos), manipulations (7 image-level approaches, 8
video-level approaches), perturbations (36 independent and more mixed
perturbations) and annotations (6.3 million classification labels, 2.9 million
manipulated area annotations and 221,247 temporal forgery segment labels). We
perform extensive benchmarking and studies of existing face forensics methods
and obtain several valuable observations.

    

### [[2104.00120] Multi-Encoder Learning and Stream Fusion for Transformer-Based End-to-End Automatic Speech Recognition](http://arxiv.org/abs/2104.00120)


  Stream fusion, also known as system combination, is a common technique in
automatic speech recognition for traditional hybrid hidden Markov model
approaches, yet mostly unexplored for modern deep neural network end-to-end
model architectures. Here, we investigate various fusion techniques for the
all-attention-based encoder-decoder architecture known as the transformer,
striving to achieve optimal fusion by investigating different fusion levels in
an example single-microphone setting with fusion of standard magnitude and
phase features. We introduce a novel multi-encoder learning method that
performs a weighted combination of two encoder-decoder multi-head attention
outputs only during training. Employing then only the magnitude feature encoder
in inference, we are able to show consistent improvement on Wall Street Journal
(WSJ) with language model and on Librispeech, without increase in runtime or
parameters. Combining two such multi-encoder trained models by a simple late
fusion in inference, we achieve state-of-the-art performance for
transformer-based models on WSJ with a significant WER reduction of 19%
relative compared to the current benchmark approach.

    

### [[2105.04019] Differentiable Sorting Networks for Scalable Sorting and Ranking Supervision](http://arxiv.org/abs/2105.04019)


  Sorting and ranking supervision is a method for training neural networks
end-to-end based on ordering constraints. That is, the ground truth order of
sets of samples is known, while their absolute values remain unsupervised. For
that, we propose differentiable sorting networks by relaxing their pairwise
conditional swap operations. To address the problems of vanishing gradients and
extensive blurring that arise with larger numbers of layers, we propose mapping
activations to regions with moderate gradients. We consider odd-even as well as
bitonic sorting networks, which outperform existing relaxations of the sorting
operation. We show that bitonic sorting networks can achieve stable training on
large input sets of up to 1024 elements.

    

### [[2105.08583] Machine Learning in weakly nonlinear systems: A Case study on Significant wave heights](http://arxiv.org/abs/2105.08583)


  This paper proposes a machine learning method based on the Extra Trees (ET)
algorithm for forecasting Significant Wave Heights in oceanic waters. To derive
multiple features from the CDIP buoys, which make point measurements, we first
nowcast various parameters and then forecast them at 30-min intervals. The
proposed algorithm has Scatter Index (SI), Bias, Correlation Coefficient, Root
Mean Squared Error (RMSE) of 0.130, -0.002, 0.97, and 0.14, respectively, for
one day ahead prediction and 0.110, -0.001, 0.98, and 0.122, respectively, for
14-day ahead prediction on the testing dataset. While other state-of-the-art
methods can only forecast up to 120 hours ahead, we extend it further to 14
days. Our proposed setup includes spectral features, hv-block cross-validation,
and stringent QC criteria. The proposed algorithm performs significantly better
than the state-of-the-art methods commonly used for significant wave height
forecasting for one-day ahead prediction. Moreover, the improved performance of
the proposed machine learning method compared to the numerical methods shows
that this performance can be extended to even longer periods allowing for early
prediction of significant wave heights in oceanic waters.

    

### [[2105.08721] A LightGBM based Forecasting of Dominant Wave Periods in Oceanic Waters](http://arxiv.org/abs/2105.08721)


  In this paper, we propose a Light Gradient Boosting (LightGBM) to forecast
dominant wave periods in oceanic waters. First, we use the data collected from
CDIP buoys and apply various data filtering methods. The data filtering methods
allow us to obtain a high-quality dataset for training and validation purposes.
We then extract various wave-based features like wave heights, periods,
skewness, kurtosis, etc., and atmospheric features like humidity, pressure, and
air temperature for the buoys. Afterward, we train algorithms that use LightGBM
and Extra Trees through a hv-block cross-validation scheme to forecast dominant
wave periods for up to 30 days ahead. LightGBM has the R2 score of 0.94, 0.94,
and 0.94 for 1-day ahead, 15-day ahead, and 30-day ahead prediction. Similarly,
Extra Trees (ET) has an R2 score of 0.88, 0.86, and 0.85 for 1-day ahead,
15-day ahead, and 30 day ahead prediction. In case of the test dataset,
LightGBM has R2 score of 0.94, 0.94, and 0.94 for 1-day ahead, 15-day ahead and
30-day ahead prediction. ET has R2 score of 0.88, 0.86, and 0.85 for 1-day
ahead, 15-day ahead, and 30-day ahead prediction. A similar R2 score for both
training and the test dataset suggests that the machine learning models
developed in this paper are robust. Since the LightGBM algorithm outperforms ET
for all the windows tested, it is taken as the final algorithm. Note that the
performance of both methods does not decrease significantly as the forecast
horizon increases. Likewise, the proposed method outperforms the numerical
approaches included in this paper in the test dataset. For 1 day ahead
prediction, the proposed algorithm has SI, Bias, CC, and RMSE of 0.09, 0.00,
0.97, and 1.78 compared to 0.268, 0.40, 0.63, and 2.18 for the European Centre
for Medium-range Weather Forecasts (ECMWF) model, which outperforms all the
other methods in the test dataset.

    

### [[2105.11730] Exploring Autoencoder-based Error-bounded Compression for Scientific Data](http://arxiv.org/abs/2105.11730)


  Error-bounded lossy compression is becoming an indispensable technique for
the success of today's scientific projects with vast volumes of data produced
during the simulations or instrument data acquisitions. Not only can it
significantly reduce data size, but it also can control the compression errors
based on user-specified error bounds. Autoencoder (AE) models have been widely
used in image compression, but few AE-based compression approaches support
error-bounding features, which are highly required by scientific applications.
To address this issue, we explore using convolutional autoencoders to improve
error-bounded lossy compression for scientific data, with the following three
key contributions. (1) We provide an in-depth investigation of the
characteristics of various autoencoder models and develop an error-bounded
autoencoder-based framework in terms of the SZ model. (2) We optimize the
compression quality for main stages in our designed AE-based error-bounded
compression framework, fine-tuning the block sizes and latent sizes and also
optimizing the compression efficiency of latent vectors. (3) We evaluate our
proposed solution using five real-world scientific datasets and comparing them
with six other related works. Experiments show that our solution exhibits a
very competitive compression quality from among all the compressors in our
tests. In absolute terms, it can obtain a much better compression quality (100%
~ 800% improvement in compression ratio with the same data distortion) compared
with SZ2.1 and ZFP in cases with a high compression ratio.

    

### [[2106.03593] Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising](http://arxiv.org/abs/2106.03593)


  In e-commerce advertising, it is crucial to jointly consider various
performance metrics, e.g., user experience, advertiser utility, and platform
revenue. Traditional auction mechanisms, such as GSP and VCG auctions, can be
suboptimal due to their fixed allocation rules to optimize a single performance
metric (e.g., revenue or social welfare). Recently, data-driven auctions,
learned directly from auction outcomes to optimize multiple performance
metrics, have attracted increasing research interests. However, the procedure
of auction mechanisms involves various discrete calculation operations, making
it challenging to be compatible with continuous optimization pipelines in
machine learning. In this paper, we design \underline{D}eep \underline{N}eural
\underline{A}uctions (DNAs) to enable end-to-end auction learning by proposing
a differentiable model to relax the discrete sorting operation, a key component
in auctions. We optimize the performance metrics by developing deep models to
efficiently extract contexts from auctions, providing rich features for auction
design. We further integrate the game theoretical conditions within the model
design, to guarantee the stability of the auctions. DNAs have been successfully
deployed in the e-commerce advertising system at Taobao. Experimental
evaluation results on both large-scale data set as well as online A/B test
demonstrated that DNAs significantly outperformed other mechanisms widely
adopted in industry.

    

### [[2106.04887] Interaction-Grounded Learning](http://arxiv.org/abs/2106.04887)


  Consider a prosthetic arm, learning to adapt to its user's control signals.
We propose Interaction-Grounded Learning for this novel setting, in which a
learner's goal is to interact with the environment with no grounding or
explicit reward to optimize its policies. Such a problem evades common RL
solutions which require an explicit reward. The learning agent observes a
multidimensional context vector, takes an action, and then observes a
multidimensional feedback vector. This multidimensional feedback vector has no
explicit reward information. In order to succeed, the algorithm must learn how
to evaluate the feedback vector to discover a latent reward signal, with which
it can ground its policies without supervision. We show that in an
Interaction-Grounded Learning setting, with certain natural assumptions, a
learner can discover the latent reward and ground its policy for successful
interaction. We provide theoretical guarantees and a proof-of-concept empirical
evaluation to demonstrate the effectiveness of our proposed approach.

    

### [[2106.07900] Augmented Tensor Decomposition with Stochastic Optimization](http://arxiv.org/abs/2106.07900)


  Tensor decompositions are powerful tools for dimensionality reduction and
feature interpretation of multidimensional data such as signals. Existing
tensor decomposition objectives (e.g., Frobenius norm) are designed for fitting
raw data under statistical assumptions, which may not align with downstream
classification tasks. Also, real-world tensor data are usually high-ordered and
have large dimensions with millions or billions of entries. Thus, it is
expensive to decompose the whole tensor with traditional algorithms. In
practice, raw tensor data also contains redundant information while data
augmentation techniques may be used to smooth out noise in samples. This paper
addresses the above challenges by proposing augmented tensor decomposition
(ATD), which effectively incorporates data augmentations to boost downstream
classification. To reduce the memory footprint of the decomposition, we propose
a stochastic algorithm that updates the factor matrices in a batch fashion. We
evaluate ATD on multiple signal datasets. It shows comparable or better
performance (e.g., up to 15% in accuracy) over self-supervised and autoencoder
baselines with less than 5% of model parameters, achieves 0.6% ~ 1.3% accuracy
gain over other tensor-based baselines, and reduces the memory footprint by 9X
when compared to standard tensor decomposition algorithms.

    

### [[2106.08038] Mean Embeddings with Test-Time Data Augmentation for Ensembling of Representations](http://arxiv.org/abs/2106.08038)


  Averaging predictions over a set of models -- an ensemble -- is widely used
to improve predictive performance and uncertainty estimation of deep learning
models. At the same time, many machine learning systems, such as search,
matching, and recommendation systems, heavily rely on embeddings.
Unfortunately, due to misalignment of features of independently trained models,
embeddings, cannot be improved with a naive deep ensemble like approach. In
this work, we look at the ensembling of representations and propose mean
embeddings with test-time augmentation (MeTTA) simple yet well-performing
recipe for ensembling representations. Empirically we demonstrate that MeTTA
significantly boosts the quality of linear evaluation on ImageNet for both
supervised and self-supervised models. Even more exciting, we draw connections
between MeTTA, image retrieval, and transformation invariant models. We believe
that spreading the success of ensembles to inference higher-quality
representations is the important step that will open many new applications of
ensembling.

    

### [[2106.13456] Interpreting Criminal Charge Prediction and Its Algorithmic Bias via Quantum-Inspired Complex Valued Networks](http://arxiv.org/abs/2106.13456)


  While predictive policing has become increasingly common in assisting with
decisions in the criminal justice system, the use of these results is still
controversial. Some software based on deep learning lacks accuracy (e.g., in
F-1), and importantly many decision processes are not transparent, causing
doubt about decision bias, such as perceived racial and age disparities. This
paper addresses bias issues with post-hoc explanations to provide a trustable
prediction of whether a person will receive future criminal charges given one's
previous criminal records by learning temporal behavior patterns over twenty
years. Bi-LSTM relieves the vanishing gradient problem, attentional mechanisms
allow learning and interpretation of feature importance, and complex-valued
networks inspired quantum physics to facilitate a certain level of transparency
in modeling the decision process. Our approach shows a consistent and reliable
prediction precision and recall on a real-life dataset. Our analysis of the
importance of each input feature shows the critical causal impact on
decision-making, suggesting that criminal histories are statistically
significant factors, while identifiers, such as race and age, are not. Finally,
our algorithm indicates that a suspect tends to rather than suddenly increase
crime severity level over time gradually.

    

### [[2106.15158] End-to-end Waveform Learning Through Joint Optimization of Pulse and Constellation Shaping](http://arxiv.org/abs/2106.15158)


  As communication systems are foreseen to enable new services such as joint
communication and sensing and utilize parts of the sub-THz spectrum, the design
of novel waveforms that can support these emerging applications becomes
increasingly challenging. We present in this work an end-to-end learning
approach to design waveforms through joint learning of pulse shaping and
constellation geometry, together with a neural network (NN)-based receiver.
Optimization is performed to maximize an achievable information rate, while
satisfying constraints on out-of-band emission and power envelope. Our results
show that the proposed approach enables up to orders of magnitude smaller
adjacent channel leakage ratios (ACLRs) with peak-to-average power ratios
(PAPRs) competitive with traditional filters, without significant loss of
information rate on an additive white Gaussian noise (AWGN) channel, and no
additional complexity at the transmitter.

    

### [[2106.15499] Self-Contrastive Learning](http://arxiv.org/abs/2106.15499)


  This paper proposes a novel contrastive learning framework, coined as
Self-Contrastive (SelfCon) Learning, that self-contrasts within multiple
outputs from the different levels of a network. We confirmed that SelfCon loss
guarantees the lower bound of mutual information (MI) between the intermediate
and last representations. Besides, we empirically showed, via various MI
estimators, that SelfCon loss highly correlates to the increase of MI and
better classification performance. In our experiments, SelfCon surpasses
supervised contrastive (SupCon) learning without the need for a multi-viewed
batch and with the cheaper computational cost. Especially on ResNet-18, we
achieved top-1 classification accuracy of 76.45% for the CIFAR-100 dataset,
which is 2.87% and 4.36% higher than SupCon and cross-entropy loss,
respectively. We found that mitigating both vanishing gradient and overfitting
issue makes our method outperform the counterparts.

    

### [[2107.05585] Differentially Private Stochastic Optimization: New Results in Convex and Non-Convex Settings](http://arxiv.org/abs/2107.05585)


  We study differentially private stochastic optimization in convex and
non-convex settings. For the convex case, we focus on the family of non-smooth
generalized linear losses (GLLs). Our algorithm for the $\ell_2$ setting
achieves optimal excess population risk in near-linear time, while the best
known differentially private algorithms for general convex losses run in
super-linear time. Our algorithm for the $\ell_1$ setting has nearly-optimal
excess population risk $\tilde{O}\big(\sqrt{\frac{\log{d}}{n}}\big)$, and
circumvents the dimension dependent lower bound of [AFKT21] for general
non-smooth convex losses. In the differentially private non-convex setting, we
provide several new algorithms for approximating stationary points of the
population risk. For the $\ell_1$-case with smooth losses and polyhedral
constraint, we provide the first nearly dimension independent rate, $\tilde
O\big(\frac{\log^{2/3}{d}}{n^{1/3}}\big)$ in linear time. For the constrained
$\ell_2$-case, with smooth losses, we obtain a linear-time algorithm with rate
$\tilde O\big(\frac{1}{n^{3/10}d^{1/10}}+\big(\frac{d}{n^2}\big)^{1/5}\big)$.
Finally, for the $\ell_2$-case we provide the first method for {\em non-smooth
weakly convex} stochastic optimization with rate $\tilde
O\big(\frac{1}{n^{1/4}}+\big(\frac{d}{n^2}\big)^{1/6}\big)$ which matches the
best existing non-private algorithm when $d= O(\sqrt{n})$. We also extend all
our results above for the non-convex $\ell_2$ setting to the $\ell_p$ setting,
where $1 < p \leq 2$, with only polylogarithmic (in the dimension) overhead in
the rates.

    

### [[2107.06814] Gain and Pain of a Reliable Delay Model](http://arxiv.org/abs/2107.06814)


  State-of-the-art digital circuit design tools almost exclusively rely on pure
and inertial delay for timing simulations. While these provide reasonable
estimations at very low execution time in the average case, their ability to
cover complex signal traces is limited. Research has provided the dynamic
Involution Delay Model (IDM) as a promising alternative, which was shown (i) to
depict reality more closely and recently (ii) to be compatible with modern
simulation suites. In this paper we complement these encouraging results by
experimentally exploring the behavioral coverage for more advanced circuits. In
detail we apply the IDM to three simple circuits (a combinatorial loop, an SR
latch and an adder), interpret the delivered results and evaluate the overhead
in realistic settings. Comparisons to digital (inertial delay) and analog
(SPICE) simulations reveal, that the IDM delivers very fine-grained results,
which match analog simulations very closely. Moreover, severe shortcomings of
inertial delay become apparent in our simulations, as it fails to depict a
range of malicious behaviors. Overall the Involution Delay Model hence
represents a viable upgrade to the available delay models in modern digital
timing simulation tools.

    

### [[2107.06533] Accelerating Distributed K-FAC with Smart Parallelism of Computing and Communication Tasks](http://arxiv.org/abs/2107.06533)


  Distributed training with synchronous stochastic gradient descent (SGD) on
GPU clusters has been widely used to accelerate the training process of deep
models. However, SGD only utilizes the first-order gradient in model parameter
updates, which may take days or weeks. Recent studies have successfully
exploited approximate second-order information to speed up the training
process, in which the Kronecker-Factored Approximate Curvature (KFAC) emerges
as one of the most efficient approximation algorithms for training deep models.
Yet, when leveraging GPU clusters to train models with distributed KFAC
(D-KFAC), it incurs extensive computation as well as introduces extra
communications during each iteration. In this work, we propose D-KFAC
(SPD-KFAC) with smart parallelism of computing and communication tasks to
reduce the iteration time. Specifically, 1) we first characterize the
performance bottlenecks of D-KFAC, 2) we design and implement a pipelining
mechanism for Kronecker factors computation and communication with dynamic
tensor fusion, and 3) we develop a load balancing placement for inverting
multiple matrices on GPU clusters. We conduct real-world experiments on a
64-GPU cluster with 100Gb/s InfiniBand interconnect. Experimental results show
that our proposed SPD-KFAC training scheme can achieve 10%-35% improvement over
state-of-the-art algorithms.

    

### [[2107.06640] 3D Acoustic-Elastic Coupling with Gravity: The Dynamics of the 2018 Palu, Sulawesi Earthquake and Tsunami](http://arxiv.org/abs/2107.06640)


  We present a highly scalable 3D fully-coupled Earth & ocean model of
earthquake rupture and tsunami generation and perform the first fully coupled
simulation of an actual earthquake-tsunami event and a 3D benchmark problem of
tsunami generation by a mega-thrust dynamic earthquake rupture. Multi-petascale
simulations, with excellent performance demonstrated on three different
platforms, allow high-resolution forward modeling. Our largest mesh has
$\approx$261 billion degrees of freedom, resolving at least 15 Hz of the
acoustic wave field. We self-consistently model seismic, acoustic and surface
gravity wave propagation in elastic (Earth) and acoustic (ocean) materials
sourced by physics-based non-linear earthquake dynamic rupture, thereby gaining
insight into the tsunami generation process without relying on approximations
that have previously been applied to permit solution of this challenging
problem. Complicated geometries, including high-resolution bathymetry,
coastlines and segmented earthquake faults are discretized by adaptive
unstructured tetrahedral meshes. This leads inevitably to large differences in
element sizes and wave speeds which can be mitigated by ADER local
time-stepping and a Discontinuous Galerkin discretisation yielding high-order
accuracy in time and space.

    

### [[2107.06771] Simulation of Dissemination Strategies on Temporal Networks](http://arxiv.org/abs/2107.06771)


  In distributed environments, such as distributed ledgers technologies and
other peer-to-peer architectures, communication represents a crucial topic. The
ability to efficiently disseminate contents is strongly influenced by the type
of system architecture, the protocol used to spread such contents over the
network and the actual dynamicity of the communication links (i.e. static vs.
temporal nets). In particular, the dissemination strategies either focus on
achieving an optimal coverage, minimizing the network traffic or providing
assurances on anonymity (that is a fundamental requirement of many
cryptocurrencies). In this work, the behaviour of multiple dissemination
protocols is discussed and studied through simulation. The performance
evaluation has been carried out on temporal networks with the help of
LUNES-temporal, a discrete event simulator that allows to test algorithms
running on a distributed environment. The experiments show that some gossip
protocols allow to either save a considerable number of messages or to provide
better anonymity guarantees, at the cost of a little lower coverage achieved
and/or a little increase of the delivery time.

    

### [[2107.06835] A Review on Edge Analytics: Issues, Challenges, Opportunities, Promises, Future Directions, and Applications](http://arxiv.org/abs/2107.06835)


  Edge technology aims to bring Cloud resources (specifically, the compute,
storage, and network) to the closed proximity of the Edge devices, i.e., smart
devices where the data are produced and consumed. Embedding computing and
application in Edge devices lead to emerging of two new concepts in Edge
technology, namely, Edge computing and Edge analytics. Edge analytics uses some
techniques or algorithms to analyze the data generated by the Edge devices.
With the emerging of Edge analytics, the Edge devices have become a complete
set. Currently, Edge analytics is unable to provide full support for the
execution of the analytic techniques. The Edge devices cannot execute advanced
and sophisticated analytic algorithms following various constraints such as
limited power supply, small memory size, limited resources, etc. This article
aims to provide a detailed discussion on Edge analytics. A clear explanation to
distinguish between the three concepts of Edge technology, namely, Edge
devices, Edge computing, and Edge analytics, along with their issues.
Furthermore, the article discusses the implementation of Edge analytics to
solve many problems in various areas such as retail, agriculture, industry, and
healthcare. In addition, the research papers of the state-of-the-art edge
analytics are rigorously reviewed in this article to explore the existing
issues, emerging challenges, research opportunities and their directions, and
applications.

    

### [[2107.06836] Consistent RDMA-Friendly Hashing on Remote Persistent Memory](http://arxiv.org/abs/2107.06836)


  Coalescing RDMA and Persistent Memory (PM) delivers high end-to-end
performance for networked storage systems, which requires rethinking the design
of efficient hash structures. In general, existing hashing schemes separately
optimize RDMA and PM, thus partially addressing the problems of RDMA Access
Amplification and High-Overhead PM Consistency. In order to address these
problems, we propose a continuity hashing, which is a "one-stone-two-birds"
design to optimize both RDMA and PM. The continuity hashing leverages a
fine-grained contiguous shared region, called SBuckets, to provide standby
positions for the neighbouring two buckets in case of hash collisions. In the
continuity hashing, remote read only needs a single RDMA read to directly fetch
the home bucket and the neighbouring SBuckets, which contain all the positions
of maintaining a key-value item, thus alleviating RDMA access amplification.
Continuity hashing further leverages indicators that can be atomically modified
to support log-free PM consistency for all the write operations. Evaluation
results demonstrate that compared with state-of-the-art schemes, continuity
hashing achieves high throughput (i.e., 1.45X -- 2.43X improvement), low
latency (about 1.7X speedup) and the smallest number of PM writes with various
workloads, while has acceptable load factors of about 70%.

    

### [[2107.06307] HDMapNet: An Online HD Map Construction and Evaluation Framework](http://arxiv.org/abs/2107.06307)


  High-definition map (HD map) construction is a crucial problem for autonomous
driving. This problem typically involves collecting high-quality point clouds,
fusing multiple point clouds of the same scene, annotating map elements, and
updating maps constantly. This pipeline, however, requires a vast amount of
human efforts and resources which limits its scalability. Additionally,
traditional HD maps are coupled with centimeter-level accurate localization
which is unreliable in many scenarios. In this paper, we argue that online map
learning, which dynamically constructs the HD maps based on local sensor
observations, is a more scalable way to provide semantic and geometry priors to
self-driving vehicles than traditional pre-annotated HD maps. Meanwhile, we
introduce an online map learning method, titled HDMapNet. It encodes image
features from surrounding cameras and/or point clouds from LiDAR, and predicts
vectorized map elements in the bird's-eye view. We benchmark HDMapNet on the
nuScenes dataset and show that in all settings, it performs better than
baseline methods. Of note, our fusion-based HDMapNet outperforms existing
methods by more than 50% in all metrics. To accelerate future research, we
develop customized metrics to evaluate map learning performance, including both
semantic-level and instance-level ones. By introducing this method and metrics,
we invite the community to study this novel map learning problem. We will
release our code and evaluation kit to facilitate future development.

    

### [[2107.06329] Efficient exact computation of the conjunctive and disjunctive decompositions of D-S Theory for information fusion: Translation and extension](http://arxiv.org/abs/2107.06329)


  Dempster-Shafer Theory (DST) generalizes Bayesian probability theory,
offering useful additional information, but suffers from a high computational
burden. A lot of work has been done to reduce the complexity of computations
used in information fusion with Dempster's rule. Yet, few research had been
conducted to reduce the complexity of computations for the conjunctive and
disjunctive decompositions of evidence, which are at the core of other
important methods of information fusion. In this paper, we propose a method
designed to exploit the actual evidence (information) contained in these
decompositions in order to compute them. It is based on a new notion that we
call focal point, derived from the notion of focal set. With it, we are able to
reduce these computations up to a linear complexity in the number of focal sets
in some cases. In a broader perspective, our formulas have the potential to be
tractable when the size of the frame of discernment exceeds a few dozen
possible states, contrary to the existing litterature. This article extends
(and translates) our work published at the french conference GRETSI in 2019.

    

### [[2107.06413] Monotonicity and Noise-Tolerance in Case-Based Reasoning with Abstract Argumentation (with Appendix)](http://arxiv.org/abs/2107.06413)


  Recently, abstract argumentation-based models of case-based reasoning
($AA{\text -} CBR$ in short) have been proposed, originally inspired by the
legal domain, but also applicable as classifiers in different scenarios.
However, the formal properties of $AA{\text -} CBR$ as a reasoning system
remain largely unexplored. In this paper, we focus on analysing the
non-monotonicity properties of a regular version of $AA{\text -} CBR$ (that we
call $AA{\text -} CBR_{\succeq}$). Specifically, we prove that $AA{\text -}
CBR_{\succeq}$ is not cautiously monotonic, a property frequently considered
desirable in the literature. We then define a variation of $AA{\text -}
CBR_{\succeq}$ which is cautiously monotonic. Further, we prove that such
variation is equivalent to using $AA{\text -} CBR_{\succeq}$ with a restricted
casebase consisting of all "surprising" and "sufficient" cases in the original
casebase. As a by-product, we prove that this variation of $AA{\text -}
CBR_{\succeq}$ is cumulative, rationally monotonic, and empowers a principled
treatment of noise in "incoherent" casebases. Finally, we illustrate $AA{\text
-} CBR$ and cautious monotonicity questions on a case study on the U.S. Trade
Secrets domain, a legal casebase.

    

### [[2107.06426] TSCAN : Dialog Structure discovery using SCAN](http://arxiv.org/abs/2107.06426)


  Can we discover dialog structure by dividing utterances into labelled
clusters. Can these labels be generated from the data. Typically for dialogs we
need an ontology and use that to discover structure, however by using
unsupervised classification and self-labelling we are able to intuit this
structure without any labels or ontology. In this paper we apply SCAN (Semantic
Clustering using Nearest Neighbors) to dialog data. We used BERT for pretext
task and an adaptation of SCAN for clustering and self labeling. These clusters
are used to identify transition probabilities and create the dialog structure.
The self-labelling method used for SCAN makes these structures interpretable as
every cluster has a label. As the approach is unsupervised, evaluation metrics
is a challenge, we use statistical measures as proxies for structure quality

    

### [[2107.06434] Centralized Model and Exploration Policy for Multi-Agent RL](http://arxiv.org/abs/2107.06434)


  Reinforcement learning (RL) in partially observable, fully cooperative
multi-agent settings (Dec-POMDPs) can in principle be used to address many
real-world challenges such as controlling a swarm of rescue robots or a
synchronous team of quadcopters. However, Dec-POMDPs are significantly harder
to solve than single-agent problems, with the former being NEXP-complete and
the latter, MDPs, being just P-complete. Hence, current RL algorithms for
Dec-POMDPs suffer from poor sample complexity, thereby reducing their
applicability to practical problems where environment interaction is costly.
Our key insight is that using just a polynomial number of samples, one can
learn a centralized model that generalizes across different policies. We can
then optimize the policy within the learned model instead of the true system,
reducing the number of environment interactions. We also learn a centralized
exploration policy within our model that learns to collect additional data in
state-action regions with high model uncertainty. Finally, we empirically
evaluate the proposed model-based algorithm, MARCO, in three cooperative
communication tasks, where it improves sample efficiency by up to 20x.

    

### [[2107.06442] GREN: Graph-Regularized Embedding Network for Weakly-Supervised Disease Localization in X-ray images](http://arxiv.org/abs/2107.06442)


  Locating diseases in chest X-ray images with few careful annotations saves
large human effort. Recent works approached this task with innovative
weakly-supervised algorithms such as multi-instance learning (MIL) and class
activation maps (CAM), however, these methods often yield inaccurate or
incomplete regions. One of the reasons is the neglection of the pathological
implications hidden in the relationship across anatomical regions within each
image and the relationship across images. In this paper, we argue that the
cross-region and cross-image relationship, as contextual and compensating
information, is vital to obtain more consistent and integral regions. To model
the relationship, we propose the Graph Regularized Embedding Network (GREN),
which leverages the intra-image and inter-image information to locate diseases
on chest X-ray images. GREN uses a pre-trained U-Net to segment the lung lobes,
and then models the intra-image relationship between the lung lobes using an
intra-image graph to compare different regions. Meanwhile, the relationship
between in-batch images is modeled by an inter-image graph to compare multiple
images. This process mimics the training and decision-making process of a
radiologist: comparing multiple regions and images for diagnosis. In order for
the deep embedding layers of the neural network to retain structural
information (important in the localization task), we use the Hash coding and
Hamming distance to compute the graphs, which are used as regularizers to
facilitate training. By means of this, our approach achieves the
state-of-the-art result on NIH chest X-ray dataset for weakly-supervised
disease localization. Our codes are accessible online.

    

### [[2107.06505] Few-shot Neural Human Performance Rendering from Sparse RGBD Videos](http://arxiv.org/abs/2107.06505)


  Recent neural rendering approaches for human activities achieve remarkable
view synthesis results, but still rely on dense input views or dense training
with all the capture frames, leading to deployment difficulty and inefficient
training overload. However, existing advances will be ill-posed if the input is
both spatially and temporally sparse. To fill this gap, in this paper we
propose a few-shot neural human rendering approach (FNHR) from only sparse RGBD
inputs, which exploits the temporal and spatial redundancy to generate
photo-realistic free-view output of human activities. Our FNHR is trained only
on the key-frames which expand the motion manifold in the input sequences. We
introduce a two-branch neural blending to combine the neural point render and
classical graphics texturing pipeline, which integrates reliable observations
over sparse key-frames. Furthermore, we adopt a patch-based adversarial
training process to make use of the local redundancy and avoids over-fitting to
the key-frames, which generates fine-detailed rendering results. Extensive
experiments demonstrate the effectiveness of our approach to generate
high-quality free view-point results for challenging human performances under
the sparse setting.

    

### [[2107.06516] Learning Algebraic Recombination for Compositional Generalization](http://arxiv.org/abs/2107.06516)


  Neural sequence models exhibit limited compositional generalization ability
in semantic parsing tasks. Compositional generalization requires algebraic
recombination, i.e., dynamically recombining structured expressions in a
recursive manner. However, most previous studies mainly concentrate on
recombining lexical units, which is an important but not sufficient part of
algebraic recombination. In this paper, we propose LeAR, an end-to-end neural
model to learn algebraic recombination for compositional generalization. The
key insight is to model the semantic parsing task as a homomorphism between a
latent syntactic algebra and a semantic algebra, thus encouraging algebraic
recombination. Specifically, we learn two modules jointly: a Composer for
producing latent syntax, and an Interpreter for assigning semantic operations.
Experiments on two realistic and comprehensive compositional generalization
benchmarks demonstrate the effectiveness of our model. The source code is
publicly available at this https URL.

    

### [[2107.06547] The I-ADOPT Interoperability Framework for FAIRer data descriptions of biodiversity](http://arxiv.org/abs/2107.06547)


  Biodiversity, the variation within and between species and ecosystems, is
essential for human well-being and the equilibrium of the planet. It is
critical for the sustainable development of human society and is an important
global challenge. Biodiversity research has become increasingly data-intensive
and it deals with heterogeneous and distributed data made available by global
and regional initiatives, such as GBIF, ILTER, LifeWatch, BODC, PANGAEA, and
TERN, that apply different data management practices. In particular, a variety
of metadata and semantic resources have been produced by these initiatives to
describe biodiversity observations, introducing interoperability issues across
data management systems. To address these challenges, the InteroperAble
Descriptions of Observable Property Terminology WG (I-ADOPT WG) was formed by a
group of international terminology providers and data center managers in 2019
with the aim to build a common approach to describe what is observed, measured,
calculated, or derived. Based on an extensive analysis of existing semantic
representations of variables, the WG has recently published the I-ADOPT
framework ontology to facilitate interoperability between existing semantic
resources and support the provision of machine-readable variable descriptions
whose components are mapped to FAIR vocabulary terms. The I-ADOPT framework
ontology defines a set of high level semantic components that can be used to
describe a variety of patterns commonly found in scientific observations. This
contribution will focus on how the I-ADOPT framework can be applied to
represent variables commonly used in the biodiversity domain.

    

### [[2107.06552] Domain Generalization with Pseudo-Domain Label for Face Anti-Spoofing](http://arxiv.org/abs/2107.06552)


  Face anti-spoofing (FAS) plays an important role in protecting face
recognition systems from face representation attacks. Many recent studies in
FAS have approached this problem with domain generalization technique. Domain
generalization aims to increase generalization performance to better detect
various types of attacks and unseen attacks. However, previous studies in this
area have defined each domain simply as an anti-spoofing datasets and focused
on developing learning techniques. In this paper, we proposed a method that
enables network to judge its domain by itself with the clustered convolutional
feature statistics from intermediate layers of the network, without labeling
domains as datasets. We obtained pseudo-domain labels by not only using the
network extracting features, but also using depth estimators, which were
previously used only as an auxiliary task in FAS. In our experiments, we
trained with three datasets and evaluated the performance with the remaining
one dataset to demonstrate the effectiveness of the proposed method by
conducting a total of four sets of experiments.

    

### [[2107.06581] A Granular Sieving Algorithm for Deterministic Global Optimization](http://arxiv.org/abs/2107.06581)


  A gradient-free deterministic method is developed to solve global
optimization problems for Lipschitz continuous functions defined in arbitrary
path-wise connected compact sets in Euclidean spaces. The method can be
regarded as granular sieving with synchronous analysis in both the domain and
range of the objective function. With straightforward mathematical formulation
applicable to both univariate and multivariate objective functions, the global
minimum value and all the global minimizers are located through two decreasing
sequences of compact sets in, respectively, the domain and range spaces. The
algorithm is easy to implement with moderate computational cost. The method is
tested against extensive benchmark functions in the literature. The
experimental results show remarkable effectiveness and applicability of the
algorithm.

    

### [[2107.06638] Procedural Content Generation using Behavior Trees (PCGBT)](http://arxiv.org/abs/2107.06638)


  Behavior trees (BTs) are a popular method of modeling the behavior of NPCs
and enemy AI and have found widespread use in a large number of commercial
games. In this paper, rather than use BTs to model game-playing agents, we
demonstrate their use for modeling game design agents, defining behaviors as
executing content generation tasks rather than in-game actions. Similar to how
traditional BTs enable modeling behaviors in a modular and dynamic manner, BTs
for PCG enable simple subtrees for generating parts of levels to be combined
modularly to form more complex trees for generating whole levels as well as
generators that can dynamically vary the generated content. We demonstrate this
approach by using BTs to model generators for Super Mario Bros., Mega Man and
Metroid levels as well as dungeon layouts and discuss several ways in which
this PCGBT paradigm could be applied and extended in the future.

    

### [[2107.06641] Trustworthy AI: A Computational Perspective](http://arxiv.org/abs/2107.06641)


  In the past few decades, artificial intelligence (AI) technology has
experienced swift developments, changing everyone's daily life and profoundly
altering the course of human society. The intention of developing AI is to
benefit humans, by reducing human labor, bringing everyday convenience to human
lives, and promoting social good. However, recent research and AI applications
show that AI can cause unintentional harm to humans, such as making unreliable
decisions in safety-critical scenarios or undermining fairness by inadvertently
discriminating against one group. Thus, trustworthy AI has attracted immense
attention recently, which requires careful consideration to avoid the adverse
effects that AI may bring to humans, so that humans can fully trust and live in
harmony with AI technologies.
Recent years have witnessed a tremendous amount of research on trustworthy
AI. In this survey, we present a comprehensive survey of trustworthy AI from a
computational perspective, to help readers understand the latest technologies
for achieving trustworthy AI. Trustworthy AI is a large and complex area,
involving various dimensions. In this work, we focus on six of the most crucial
dimensions in achieving trustworthy AI: (i) Safety & Robustness, (ii)
Non-discrimination & Fairness, (iii) Explainability, (iv) Privacy, (v)
Accountability & Auditability, and (vi) Environmental Well-Being. For each
dimension, we review the recent related technologies according to a taxonomy
and summarize their applications in real-world systems. We also discuss the
accordant and conflicting interactions among different dimensions and discuss
potential aspects for trustworthy AI to investigate in the future.

    

### [[2107.06672] Improved SAT models for NFA learning](http://arxiv.org/abs/2107.06672)


  Grammatical inference is concerned with the study of algorithms for learning
automata and grammars from words. We focus on learning Nondeterministic Finite
Automaton of size k from samples of words. To this end, we formulate the
problem as a SAT model. The generated SAT instances being enormous, we propose
some model improvements, both in terms of the number of variables, the number
of clauses, and clauses size. These improvements significantly reduce the
instances, but at the cost of longer generation time. We thus try to balance
instance size vs. generation and solving time. We also achieved some
experimental comparisons and we analyzed our various model improvements.

    

### [[2107.06708] MDE4QAI: Towards Model-Driven Engineering for Quantum Artificial Intelligence](http://arxiv.org/abs/2107.06708)


  Over the past decade, Artificial Intelligence (AI) has provided enormous new
possibilities and opportunities, but also new demands and requirements for
software systems. In particular, Machine Learning (ML) has proven useful in
almost every vertical application domain. Although other sub-disciplines of AI,
such as intelligent agents and Multi-Agent Systems (MAS) did not become
promoted to the same extent, they still possess the potential to be integrated
into the mainstream technology stacks and ecosystems, for example, due to the
ongoing prevalence of the Internet of Things (IoT) and smart Cyber-Physical
Systems (CPS). However, in the decade ahead, an unprecedented paradigm shift
from classical computing towards Quantum Computing (QC) is expected, with
perhaps a quantum-classical hybrid model. We expect the Model-Driven
Engineering (MDE) paradigm to be an enabler and a facilitator, when it comes to
the quantum and the quantum-classical hybrid applications as it has already
proven beneficial in the highly complex domains of IoT, smart CPS and AI with
inherently heterogeneous hardware and software platforms, and APIs. This
includes not only automated code generation, but also automated model checking
and verification, as well as model analysis in the early design phases, and
model-to-model transformations both at the design-time and at the runtime. In
this paper, the vision is focused on MDE for Quantum AI, and a holistic
approach integrating all of the above.

    

### [[2107.06747] Artificial Intelligence in PET: an Industry Perspective](http://arxiv.org/abs/2107.06747)


  Artificial intelligence (AI) has significant potential to positively impact
and advance medical imaging, including positron emission tomography (PET)
imaging applications. AI has the ability to enhance and optimize all aspects of
the PET imaging chain from patient scheduling, patient setup, protocoling, data
acquisition, detector signal processing, reconstruction, image processing and
interpretation. AI poses industry-specific challenges which will need to be
addressed and overcome to maximize the future potentials of AI in PET. This
paper provides an overview of these industry-specific challenges for the
development, standardization, commercialization, and clinical adoption of AI,
and explores the potential enhancements to PET imaging brought on by AI in the
near future. In particular, the combination of on-demand image reconstruction,
AI, and custom designed data processing workflows may open new possibilities
for innovation which would positively impact the industry and ultimately
patients.

    

### [[2107.06750] Fast and Slow Enigmas and Parental Guidance](http://arxiv.org/abs/2107.06750)


  We describe several additions to the ENIGMA system that guides clause
selection in the E automated theorem prover. First, we significantly speed up
its neural guidance by adding server-based GPU evaluation. The second addition
is motivated by fast weight-based rejection filters that are currently used in
systems like E and Prover9. Such systems can be made more intelligent by
instead training fast versions of ENIGMA that implement more intelligent
pre-filtering. This results in combinations of trainable fast and slow thinking
that improves over both the fast-only and slow-only methods. The third addition
is based on "judging the children by their parents", i.e., possibly rejecting
an inference before it produces a clause. This is motivated by standard
evolutionary mechanisms, where there is always a cost to producing all possible
offsprings in the current population. This saves time by not evaluating all
clauses by more expensive methods and provides a complementary view of the
generated clauses. The methods are evaluated on a large benchmark coming from
the Mizar Mathematical Library, showing good improvements over the state of the
art.

    

### [[2107.06777] Synthesis in Style: Semantic Segmentation of Historical Documents using Synthetic Data](http://arxiv.org/abs/2107.06777)


  One of the most pressing problems in the automated analysis of historical
documents is the availability of annotated training data. In this paper, we
propose a novel method for the synthesis of training data for semantic
segmentation of document images. We utilize clusters found in intermediate
features of a StyleGAN generator for the synthesis of RGB and label images at
the same time. Our model can be applied to any dataset of scanned documents
without the need for manual annotation of individual images, as each model is
custom-fit to the dataset. In our experiments, we show that models trained on
our synthetic data can reach competitive performance on open benchmark datasets
for line segmentation.

    

### [[2107.06817] Efficient Set of Vectors Search](http://arxiv.org/abs/2107.06817)


  We consider a similarity measure between two sets $A$ and $B$ of vectors,
that balances the average and maximum cosine distance between pairs of vectors,
one from set $A$ and one from set $B$. As a motivation for this measure, we
present lineage tracking in a database. To practically realize this measure, we
need an approximate search algorithm that given a set of vectors $A$ and sets
of vectors $B_1,...,B_n$, the algorithm quickly locates the set $B_i$ that
maximizes the similarity measure. For the case where all sets are singleton
sets, essentially each is a single vector, there are known efficient
approximate search algorithms, e.g., approximated versions of tree search
algorithms, locality-sensitive hashing (LSH), vector quantization (VQ) and
proximity graph algorithms. In this work, we present approximate search
algorithms for the general case. The underlying idea in these algorithms is
encoding a set of vectors via a "long" single vector.

    

### [[2107.06833] A Review-based Taxonomy for Secure Health Care Monitoring: Wireless Smart Cameras](http://arxiv.org/abs/2107.06833)


  Health records data security is one of the main challenges in e-health
systems. Authentication is one of the essential security services to support
the stored data confidentiality, integrity, and availability. This research
focuses on the secure storage of patient and medical records in the healthcare
sector where data security and unauthorized access is an ongoing issue. A
potential solution comes from biometrics, although their use may be
time-consuming and can slow down data retrieval. This research aims to overcome
these challenges and enhance data access control in the healthcare sector
through the addition of biometrics in the form of fingerprints. The proposed
model for application in the healthcare sector consists of Collection, Network
communication, and Authentication (CNA) using biometrics, which replaces an
existing password-based access control method. A sensor then collects data and
by using a network (wireless or Zig-bee), a connection is established, after
connectivity analytics and data management work which processes and aggregate
the data. Subsequently, access is granted to authenticated users of the
application. This IoT-based biometric authentication system facilitates
effective recognition and ensures confidentiality, integrity, and reliability
of patients, records and other sensitive data. The proposed solution provides
reliable access to healthcare data and enables secure access through the
process of user and device authentication. The proposed model has been
developed for access control to data through the authentication of users in
healthcare to reduce data manipulation or theft.

    

### [[2107.06840] Mixing Human Demonstrations with Self-Exploration in Experience Replay for Deep Reinforcement Learning](http://arxiv.org/abs/2107.06840)


  We investigate the effect of using human demonstration data in the replay
buffer for Deep Reinforcement Learning. We use a policy gradient method with a
modified experience replay buffer where a human demonstration experience is
sampled with a given probability. We analyze different ratios of using
demonstration data in a task where an agent attempts to reach a goal while
avoiding obstacles. Our results suggest that while the agents trained by pure
self-exploration and pure demonstration had similar success rates, the pure
demonstration model converged faster to solutions with less number of steps.

    

### [[2107.06857] Scalable Evaluation of Multi-Agent Reinforcement Learning with Melting Pot](http://arxiv.org/abs/2107.06857)


  Existing evaluation suites for multi-agent reinforcement learning (MARL) do
not assess generalization to novel situations as their primary objective
(unlike supervised-learning benchmarks). Our contribution, Melting Pot, is a
MARL evaluation suite that fills this gap, and uses reinforcement learning to
reduce the human labor required to create novel test scenarios. This works
because one agent's behavior constitutes (part of) another agent's environment.
To demonstrate scalability, we have created over 80 unique test scenarios
covering a broad range of research topics such as social dilemmas, reciprocity,
resource sharing, and task partitioning. We apply these test scenarios to
standard MARL training algorithms, and demonstrate how Melting Pot reveals
weaknesses not apparent from training performance alone.

    

### [[2107.06870] Reinforced Hybrid Genetic Algorithm for the Traveling Salesman Problem](http://arxiv.org/abs/2107.06870)


  We propose a powerful Reinforced Hybrid Genetic Algorithm (RHGA) for the
famous NP-hard Traveling Salesman Problem (TSP). RHGA combines reinforcement
learning technique with the well-known Edge Assembly Crossover genetic
algorithm (EAX-GA) and the Lin-Kernighan-Helsgaun (LKH) local search heuristic.
With the help of the proposed hybrid mechanism, the genetic evolution of EAX-GA
and the local search of LKH can boost each other's performance. And the
reinforcement learning technique based on Q-learning further promotes the
hybrid genetic algorithm. Experimental results on 138 well-known and widely
used TSP benchmarks, with the number of cities ranging from 1,000 to 85,900,
demonstrate the excellent performance of the proposed method.

    

### [[2002.09598] A characterization of proportionally representative committees](http://arxiv.org/abs/2002.09598)


  A well-known axiom for proportional representation is Proportionality of
Solid Coalitions (PSC). We characterize committees satisfying PSC as possible
outcomes of the Minimal Demand rule, which generalizes an approach pioneered by
Michael Dummett.

    

### [[2012.03709] Reference Knowledgeable Network for Machine Reading Comprehension](http://arxiv.org/abs/2012.03709)


  Multi-choice Machine Reading Comprehension (MRC) as a challenge requires
model to select the most appropriate answer from a set of candidates given
passage and question. Most of the existing researches focus on the modeling of
the task datasets without explicitly referring to external fine-grained
knowledge sources, which is supposed to greatly make up the deficiency of the
given passage. Thus we propose a novel reference-based knowledge enhancement
model called Reference Knowledgeable Network (RekNet), which refines critical
information from the passage and quote explicit knowledge in necessity. In
detail, RekNet refines fine-grained critical information and defines it as
Reference Span, then quotes explicit knowledge quadruples by the co-occurrence
information of Reference Span and candidates. The proposed RekNet is evaluated
on three multi-choice MRC benchmarks: RACE, DREAM and Cosmos QA, which shows
consistent and remarkable performance improvement with observable statistical
significance level over strong baselines.

    

### [[2101.07067] Data Obsolescence Detection in the Light of Newly Acquired Valid Observations](http://arxiv.org/abs/2101.07067)


  The information describing the conditions of a system or a person is
constantly evolving and may become obsolete and contradict other information. A
database, therefore, must be consistently updated upon the acquisition of new
valid observations that contradict obsolete ones contained in the database. In
this paper, we propose a novel approach for dealing with the information
obsolescence problem. Our approach aims to detect, in real-time, contradictions
between observations and then identify the obsolete ones, given a
representation model. Since we work within an uncertain environment
characterized by the lack of information, we choose to use a Bayesian network
as our representation model and propose a new approximate concept,
$\epsilon$-Contradiction. The new concept is parameterised by a confidence
level of having a contradiction in a set of observations. We propose a
polynomial-time algorithm for detecting obsolete information. We show that the
resulting obsolete information is better represented by an AND-OR tree than a
simple set of observations. Finally, we demonstrate the effectiveness of our
approach on a real elderly fall-prevention database and showcase how this tree
can be used to give reliable recommendations to doctors. Our experiments give
systematically and substantially very good results.

    

### [[2104.09124] DisCo: Remedy Self-supervised Learning on Lightweight Models with Distilled Contrastive Learning](http://arxiv.org/abs/2104.09124)


  While self-supervised representation learning (SSL) has received widespread
attention from the community, recent research argue that its performance will
suffer a cliff fall when the model size decreases. The current method mainly
relies on contrastive learning to train the network and in this work, we
propose a simple yet effective Distilled Contrastive Learning (DisCo) to ease
the issue by a large margin. Specifically, we find the final embedding obtained
by the mainstream SSL methods contains the most fruitful information, and
propose to distill the final embedding to maximally transmit a teacher's
knowledge to a lightweight model by constraining the last embedding of the
student to be consistent with that of the teacher. In addition, in the
experiment, we find that there exists a phenomenon termed Distilling BottleNeck
and present to enlarge the embedding dimension to alleviate this problem. Our
method does not introduce any extra parameter to lightweight models during
deployment. Experimental results demonstrate that our method achieves the
state-of-the-art on all lightweight models. Particularly, when
ResNet-101/ResNet-50 is used as teacher to teach EfficientNet-B0, the linear
result of EfficientNet-B0 on ImageNet is very close to ResNet-101/ResNet-50,
but the number of parameters of EfficientNet-B0 is only 9.4%/16.3% of
ResNet-101/ResNet-50.

    

### [[2106.00390] On the KLM properties of a fuzzy DL with Typicality](http://arxiv.org/abs/2106.00390)


  The paper investigates the properties of a fuzzy logic of typicality. The
extension of fuzzy logic with a typicality operator was proposed in recent work
to define a fuzzy multipreference semantics for Multilayer Perceptrons, by
regarding the deep neural network as a conditional knowledge base. In this
paper, we study its properties. First, a monotonic extension of a fuzzy ALC
with typicality is considered (called ALC^FT) and a reformulation the KLM
properties of a preferential consequence relation for this logic is devised.
Most of the properties are satisfied, depending on the reformulation and on the
fuzzy combination functions considered. We then strengthen ALC^FT with a
closure construction by introducing a notion of faithful model of a weighted
knowledge base, which generalizes the notion of coherent model of a conditional
knowledge base previously introduced, and we study its properties.

    

### [[2107.06591] Useful Open Call-by-Need](http://arxiv.org/abs/2107.06591)


  This paper studies useful sharing, which is a sophisticated optimization for
lambda-calculi, in the context of call-by-need evaluation in presence of open
terms. Useful sharing turns out to be harder to manipulate in call-by-need than
in call-by-name or call-by-value, because call-by-need evaluates inside
environments, making it harder to specify when a substitution step is useful.
We isolate the key involved concepts and prove the correctness of useful
sharing in this setting.

    

### [[2010.01240] Proving Quantum Programs Correct](http://arxiv.org/abs/2010.01240)


  As quantum computing progresses steadily from theory into practice,
programmers will face a common problem: How can they be sure that their code
does what they intend it to do? This paper presents encouraging results in the
application of mechanized proof to the domain of quantum programming in the
context of the SQIR development. It verifies the correctness of a range of a
quantum algorithms including Grover's algorithm and quantum phase estimation, a
key component of Shor's algorithm. In doing so, it aims to highlight both the
successes and challenges of formal verification in the quantum context and
motivate the theorem proving community to target quantum computing as an
application domain.

    