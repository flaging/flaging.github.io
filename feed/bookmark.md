
## 2021-10-8

### [[2110.03011] Empirical Analysis of Bi-directional Wi-Fi Network Performance on Mobile Robots and Connected Vehicles](http://arxiv.org/abs/2110.03011)


  This paper proposes a framework to measure the important metrics (throughput,
delay, packet retransmits, signal strength, etc.) to determine Wi-Fi network
performance of mobile robots supported by the Robot Operating Systems (ROS)
middleware. We analyze the bidirectional network performance of mobile robots
and connected vehicles through an experimental setup, where a mobile robot is
communicating vital sensor data such as video streaming from the camera(s) and
LiDAR scan values to a command station while it navigates an indoor environment
through teleoperated velocity commands received from the command station. The
experiments evaluate the performance under 2.4 GHz and 5 GHz channels with
different placement of Access Points (AP) with up to two network devices on
each side. The discussions and insights from this study apply to the general
vehicular networks and the field robotics community, where the wireless network
plays a key role in enabling the success of robotic missions.

    

### [[2110.03537] A Sidelink-Aided Approach for Secure Multicast Service Delivery: from Human-Oriented Multimedia Traffic to Machine Type Communications](http://arxiv.org/abs/2110.03537)


  To date, group-oriented communications have been mainly exploited for
delivering multimedia services in human-oriented communications while, in
future fifth generation (5G) cellular networks, objects will be the main
target. Internet of Things (IoT) will undoubtedly play a key role in 5G
networks, wherein massive machine-type communications (mMTC) feature a use case
as crucial as challenging since cellular IoT connections are predicted to grow
heavily in the next future. To boost capacity and energy efficiency, the 5G
network can leverage device-to-device (D2D) communications which are recognized
as an effective offloading technique. This is achieved thanks to the fact that,
in legacy D2D communications, data are directly sent from one device to
another, avoiding the crossing of the network. Obviously, the distributed
nature of such a communication paradigm and the inherent broadcast nature of
the wireless channel make it necessary to think how to secure the so called
"sidelink" transmissions. This work proposes a protocol for the efficient and
reliable management of multicast services in a 5G-oriented IoT scenario, in
which security is a crucial requirement to be met. The proposed protocol is
tailored to Narrowband IoT (NB-IoT) and makes use of D2D communications with
the aim of improving network efficiency and optimizing network resource
utilization. In addition, cyber security and social trustworthiness mechanisms
are exploited to secure D2D communications.

    

### [[2110.03542] A novel approach for MBSFN Area Formation aided by D2D Communications for eMBB Service Delivery in 5G NR Systems](http://arxiv.org/abs/2110.03542)


  Forthcoming 5G New Radio (NR) systems will be asked to handle a huge number
of devices accessing or delivering "resource-hungry" and high-quality services.
In view of this, the new 5G Radio Access Technology (RAT) aims to support, in
next releases, Multimedia Broadcast/Multicast Service Single Frequency Network
(MBSFN) to enable the simultaneous delivery of the same content to a set of
users covered by different cells. According to MBSFN, all cells belonging to
the same MBSFN Area are synchronized in time and the MBSFN transmission occurs
over the same radio resources. In such a way, the same content flow is
delivered by several cells to all the receivers in the MBSFN Area. A further
means to enhance the network coverage and provide high data rate and low
latency in future 5G-enabled MBSFN networks is Device-to-Device (D2D)
connectivity. Along these lines, in this paper we propose a D2D-aided MBSFN
Area Formation (D2D-MAF) algorithm to dynamically create MBSFN Areas with the
aim to improve the system aggregate data rate while satisfying all user
requests. The proposed D2D-MAF foresees that users could receive the service
through either MBSFN, or D2D, or unicast transmissions. Performance evaluation
results, carried out under a wide range of conditions, testify to the high
effectiveness of the proposed algorithm.

    

### [[2101.11246] A Survey on 5G Radio Access Network Energy Efficiency: Massive MIMO, Lean Carrier Design, Sleep Modes, and Machine Learning](http://arxiv.org/abs/2101.11246)


  Cellular networks have changed the world we are living in, and the fifth
generation (5G) of radio technology is expected to further revolutionise our
everyday lives, by enabling a high degree of automation, through its larger
capacity, massive connectivity, and ultra-reliable low-latency communications.
In addition, the third generation partnership project (3GPP) new radio (NR)
specification also provides tools to significantly decrease the energy
consumption and the green house emissions of next generations networks, thus
contributing towards information and communication technology (ICT)
sustainability targets. In this survey paper, we thoroughly review the
state-of-the-art on current energy efficiency research. We first categorise and
carefully analyse the different power consumption models and energy efficiency
metrics, which have helped to make progress on the understanding of green
networks. Then, as a main contribution, we survey in detail -- from a
theoretical and a practical viewpoint -- the main energy efficiency enabling
technologies that 3GPP NR provides, together with their main benefits and
challenges. Special attention is paid to four key enabling technologies, i.e.,
massive multiple-input multiple-output (MIMO), lean carrier design, and
advanced idle modes, together with the role of artificial intelligence
capabilities. We dive into their implementation and operational details, and
thoroughly discuss their optimal operation points and theoretical-trade-offs
from an energy consumption perspective. This will help the reader to grasp the
fundamentals of -- and the status on -- green networking. Finally, the areas of
research where more effort is needed to make future networks greener are also
discussed.

    

### [[2110.02954] A Stochastic Newton Algorithm for Distributed Convex Optimization](http://arxiv.org/abs/2110.02954)


  We propose and analyze a stochastic Newton algorithm for homogeneous
distributed stochastic convex optimization, where each machine can calculate
stochastic gradients of the same population objective, as well as stochastic
Hessian-vector products (products of an independent unbiased estimator of the
Hessian of the population objective with arbitrary vectors), with many such
stochastic computations performed between rounds of communication. We show that
our method can reduce the number, and frequency, of required communication
rounds compared to existing methods without hurting performance, by proving
convergence guarantees for quasi-self-concordant objectives (e.g., logistic
regression), alongside empirical evidence.

    

### [[2110.02987] Distributed Optimization of Graph Convolutional Network using Subgraph Variance](http://arxiv.org/abs/2110.02987)


  In recent years, Graph Convolutional Networks (GCNs) have achieved great
success in learning from graph-structured data. With the growing tendency of
graph nodes and edges, GCN training by single processor cannot meet the demand
for time and memory, which led to a boom into distributed GCN training
frameworks research. However, existing distributed GCN training frameworks
require enormous communication costs between processors since multitudes of
dependent nodes and edges information need to be collected and transmitted for
GCN training from other processors. To address this issue, we propose a Graph
Augmentation based Distributed GCN framework(GAD). In particular, GAD has two
main components, GAD-Partition and GAD-Optimizer. We first propose a graph
augmentation-based partition (GAD-Partition) that can divide original graph
into augmented subgraphs to reduce communication by selecting and storing as
few significant nodes of other processors as possible while guaranteeing the
accuracy of the training. In addition, we further design a subgraph
variance-based importance calculation formula and propose a novel weighted
global consensus method, collectively referred to as GAD-Optimizer. This
optimizer adaptively reduces the importance of subgraphs with large variances
for the purpose of reducing the effect of extra variance introduced by
GAD-Partition on distributed GCN training. Extensive experiments on four
large-scale real-world datasets demonstrate that our framework significantly
reduces the communication overhead (50%), improves the convergence speed (2X)
of distributed GCN training, and slight gain in accuracy (0.45%) based on
minimal redundancy compared to the state-of-the-art methods.

    

### [[2110.02994] Learning Canonical Embedding for Non-rigid Shape Matching](http://arxiv.org/abs/2110.02994)


  This paper provides a novel framework that learns canonical embeddings for
non-rigid shape matching. In contrast to prior work in this direction, our
framework is trained end-to-end and thus avoids instabilities and constraints
associated with the commonly-used Laplace-Beltrami basis or sequential
optimization schemes. On multiple datasets, we demonstrate that learning self
symmetry maps with a deep functional map projects 3D shapes into a low
dimensional canonical embedding that facilitates non-rigid shape correspondence
via a simple nearest neighbor search. Our framework outperforms multiple recent
learning based methods on FAUST and SHREC benchmarks while being
computationally cheaper, data-efficient, and robust.

    

### [[2110.02998] Federated Learning via Plurality Vote](http://arxiv.org/abs/2110.02998)


  Federated learning allows collaborative workers to solve a machine learning
problem while preserving data privacy. Recent studies have tackled various
challenges in federated learning, but the joint optimization of communication
overhead, learning reliability, and deployment efficiency is still an open
problem. To this end, we propose a new scheme named federated learning via
plurality vote (FedVote). In each communication round of FedVote, workers
transmit binary or ternary weights to the server with low communication
overhead. The model parameters are aggregated via weighted voting to enhance
the resilience against Byzantine attacks. When deployed for inference, the
model with binary or ternary weights is resource-friendly to edge devices. We
show that our proposed method can reduce quantization error and converges
faster compared with the methods directly quantizing the model updates.

    

### [[2110.02999] Generative Modeling with Optimal Transport Maps](http://arxiv.org/abs/2110.02999)


  With the discovery of Wasserstein GANs, Optimal Transport (OT) has become a
powerful tool for large-scale generative modeling tasks. In these tasks, OT
cost is typically used as the loss for training GANs. In contrast to this
approach, we show that the OT map itself can be used as a generative model,
providing comparable performance. Previous analogous approaches consider OT
maps as generative models only in the latent spaces due to their poor
performance in the original high-dimensional ambient space. In contrast, we
apply OT maps directly in the ambient space, e.g., a space of high-dimensional
images. First, we derive a min-max optimization algorithm to efficiently
compute OT maps for the quadratic cost (Wasserstein-2 distance). Next, we
extend the approach to the case when the input and output distributions are
located in the spaces of different dimensions and derive error bounds for the
computed OT map. We evaluate the algorithm on image generation and unpaired
image restoration tasks. In particular, we consider denoising, colorization,
and inpainting, where the optimality of the restoration map is a desired
attribute, since the output (restored) image is expected to be close to the
input (degraded) one.

    

### [[2110.03006] Data-Centric Semi-Supervised Learning](http://arxiv.org/abs/2110.03006)


  We study unsupervised data selection for semi-supervised learning (SSL),
where a large-scale unlabeled data is available and a small subset of data is
budgeted for label acquisition. Existing SSL methods focus on learning a model
that effectively integrates information from given small labeled data and large
unlabeled data, whereas we focus on selecting the right data for SSL without
any label or task information, in an also stark contrast to supervised data
selection for active learning. Intuitively, instances to be labeled shall
collectively have maximum diversity and coverage for downstream tasks, and
individually have maximum information propagation utility for SSL. We formalize
these concepts in a three-step data-centric SSL method that improves FixMatch
in stability and accuracy by 8% on CIFAR-10 (0.08% labeled) and 14% on
ImageNet-1K (0.2% labeled). Our work demonstrates that a small compute spent on
careful labeled data selection brings big annotation efficiency and model
performance gain without changing the learning pipeline. Our completely
unsupervised data selection can be easily extended to other weakly supervised
learning settings.

    

### [[2110.03007] Unsupervised Multimodal Language Representations using Convolutional Autoencoders](http://arxiv.org/abs/2110.03007)


  Multimodal Language Analysis is a demanding area of research, since it is
associated with two requirements: combining different modalities and capturing
temporal information. During the last years, several works have been proposed
in the area, mostly centered around supervised learning in downstream tasks. In
this paper we propose extracting unsupervised Multimodal Language
representations that are universal and can be applied to different tasks.
Towards this end, we map the word-level aligned multimodal sequences to 2-D
matrices and then use Convolutional Autoencoders to learn embeddings by
combining multiple datasets. Extensive experimentation on Sentiment Analysis
(MOSEI) and Emotion Recognition (IEMOCAP) indicate that the learned
representations can achieve near-state-of-the-art performance with just the use
of a Logistic Regression algorithm for downstream classification. It is also
shown that our method is extremely lightweight and can be easily generalized to
other tasks and unseen data with small performance drop and almost the same
number of parameters. The proposed multimodal representation models are
open-sourced and will help grow the applicability of Multimodal Language.

    

### [[2110.03014] Active Learning of Markov Decision Processes using Baum-Welch algorithm (Extended)](http://arxiv.org/abs/2110.03014)


  Cyber-physical systems (CPSs) are naturally modelled as reactive systems with
nondeterministic and probabilistic dynamics. Model-based verification
techniques have proved effective in the deployment of safety-critical CPSs.
Central for a successful application of such techniques is the construction of
an accurate formal model for the system. Manual construction can be a
resource-demanding and error-prone process, thus motivating the design of
automata learning algorithms to synthesise a system model from observed system
behaviours.
This paper revisits and adapts the classic Baum-Welch algorithm for learning
Markov decision processes and Markov chains. For the case of MDPs, which
typically demand more observations, we present a model-based active learning
sampling strategy that choses examples which are most informative w.r.t.\ the
current model hypothesis. We empirically compare our approach with
state-of-the-art tools and demonstrate that the proposed active learning
procedure can significantly reduce the number of observations required to
obtain accurate models.

    

### [[2110.03017] Two-Bit Aggregation for Communication Efficient and Differentially Private Federated Learning](http://arxiv.org/abs/2110.03017)


  In federated learning (FL), a machine learning model is trained on multiple
nodes in a decentralized manner, while keeping the data local and not shared
with other nodes. However, FL requires the nodes to also send information on
the model parameters to a central server for aggregation. However, the
information sent from the nodes to the server may reveal some details about
each node's local data, thus raising privacy concerns. Furthermore, the
repetitive uplink transmission from the nodes to the server may result in a
communication overhead and network congestion. To address these two challenges,
in this paper, a novel two-bit aggregation algorithm is proposed with
guaranteed differential privacy and reduced uplink communication overhead.
Extensive experiments demonstrate that the proposed aggregation algorithm can
achieve the same performance as state-of-the-art approaches on datasets such as
MNIST, Fashion MNIST, CIFAR-10, and CIFAR-100, while ensuring differential
privacy and improving communication efficiency.

    

### [[2110.03020] Efficient Methods for Online Multiclass Logistic Regression](http://arxiv.org/abs/2110.03020)


  Multiclass logistic regression is a fundamental task in machine learning with
applications in classification and boosting. Previous work (Foster et al.,
2018) has highlighted the importance of improper predictors for achieving "fast
rates" in the online multiclass logistic regression problem without suffering
exponentially from secondary problem parameters, such as the norm of the
predictors in the comparison class. While Foster et al. (2018) introduced a
statistically optimal algorithm, it is in practice computationally intractable
due to its run-time complexity being a large polynomial in the time horizon and
dimension of input feature vectors. In this paper, we develop a new algorithm,
FOLKLORE, for the problem which runs significantly faster than the algorithm of
Foster et al.(2018) -- the running time per iteration scales quadratically in
the dimension -- at the cost of a linear dependence on the norm of the
predictors in the regret bound. This yields the first practical algorithm for
online multiclass logistic regression, resolving an open problem of Foster et
al.(2018). Furthermore, we show that our algorithm can be applied to online
bandit multiclass prediction and online multiclass boosting, yielding more
practical algorithms for both problems compared to the ones in Foster et
al.(2018) with similar performance guarantees. Finally, we also provide an
online-to-batch conversion result for our algorithm.

    

### [[2110.03022] Tribuo: Machine Learning with Provenance in Java](http://arxiv.org/abs/2110.03022)


  Machine Learning models are deployed across a wide range of industries,
performing a wide range of tasks. Tracking these models and ensuring they
behave appropriately is becoming increasingly difficult as the number of
deployed models increases. There are also new regulatory burdens for ML systems
which affect human lives, requiring a link between a model and its training
data in high-risk situations. Current ML monitoring systems often provide
provenance and experiment tracking as a layer on top of an ML library, allowing
room for imperfect tracking and skew between the tracked object and the
metadata. In this paper we introduce Tribuo, a Java ML library that integrates
model training, inference, strong type-safety, runtime checking, and automatic
provenance recording into a single framework. All Tribuo's models and
evaluations record the full processing pipeline for input data, along with the
training algorithms, hyperparameters and data transformation steps
automatically. The provenance lives inside the model object and can be
persisted separately using common markup formats. Tribuo implements many
popular ML algorithms for classification, regression, clustering, multi-label
classification and anomaly detection, along with interfaces to XGBoost,
TensorFlow and ONNX Runtime. Tribuo's source code is available at
this https URL under an Apache 2.0 license with documentation
and tutorials available at this https URL.

    

### [[2110.03031] RieszNet and ForestRiesz: Automatic Debiased Machine Learning with Neural Nets and Random Forests](http://arxiv.org/abs/2110.03031)


  Many causal and policy effects of interest are defined by linear functionals
of high-dimensional or non-parametric regression functions.
$\sqrt{n}$-consistent and asymptotically normal estimation of the object of
interest requires debiasing to reduce the effects of regularization and/or
model selection on the object of interest. Debiasing is typically achieved by
adding a correction term to the plug-in estimator of the functional, that is
derived based on a functional-specific theoretical derivation of what is known
as the influence function and which leads to properties such as double
robustness and Neyman orthogonality. We instead implement an automatic
debiasing procedure based on automatically learning the Riesz representation of
the linear functional using Neural Nets and Random Forests. Our method solely
requires value query oracle access to the linear functional. We propose a
multi-tasking Neural Net debiasing method with stochastic gradient descent
minimization of a combined Riesz representer and regression loss, while sharing
representation layers for the two functions. We also propose a Random Forest
method which learns a locally linear representation of the Riesz function. Even
though our methodology applies to arbitrary functionals, we experimentally find
that it beats state of the art performance of the prior neural net based
estimator of Shi et al. (2019) for the case of the average treatment effect
functional. We also evaluate our method on the more challenging problem of
estimating average marginal effects with continuous treatments, using
semi-synthetic data of gasoline price changes on gasoline demand.

    

### [[2110.03032] Learning Multi-Objective Curricula for Deep Reinforcement Learning](http://arxiv.org/abs/2110.03032)


  Various automatic curriculum learning (ACL) methods have been proposed to
improve the sample efficiency and final performance of deep reinforcement
learning (DRL). They are designed to control how a DRL agent collects data,
which is inspired by how humans gradually adapt their learning processes to
their capabilities. For example, ACL can be used for subgoal generation, reward
shaping, environment generation, or initial state generation. However, prior
work only considers curriculum learning following one of the aforementioned
predefined paradigms. It is unclear which of these paradigms are complementary,
and how the combination of them can be learned from interactions with the
environment. Therefore, in this paper, we propose a unified automatic
curriculum learning framework to create multi-objective but coherent curricula
that are generated by a set of parametric curriculum modules. Each curriculum
module is instantiated as a neural network and is responsible for generating a
particular curriculum. In order to coordinate those potentially conflicting
modules in unified parameter space, we propose a multi-task hyper-net learning
framework that uses a single hyper-net to parameterize all those curriculum
modules. In addition to existing hand-designed curricula paradigms, we further
design a flexible memory mechanism to learn an abstract curriculum, which may
otherwise be difficult to design manually. We evaluate our method on a series
of robotic manipulation tasks and demonstrate its superiority over other
state-of-the-art ACL methods in terms of sample efficiency and final
performance.

    

### [[2110.03039] Optimized Recommender Systems with Deep Reinforcement Learning](http://arxiv.org/abs/2110.03039)


  Recommender Systems have been the cornerstone of online retailers.
Traditionally they were based on rules, relevance scores, ranking algorithms,
and supervised learning algorithms, but now it is feasible to use reinforcement
learning algorithms to generate meaningful recommendations. This work
investigates and develops means to setup a reproducible testbed, and evaluate
different state of the art algorithms in a realistic environment. It entails a
proposal, literature review, methodology, results, and comments.

    

### [[2110.03049] Physics-informed neural network simulation of multiphase poroelasticity using stress-split sequential training](http://arxiv.org/abs/2110.03049)


  Physics-informed neural networks (PINNs) have received significant attention
as a unified framework for forward, inverse, and surrogate modeling of problems
governed by partial differential equations (PDEs). Training PINNs for forward
problems, however, pose significant challenges, mainly because of the complex
non-convex and multi-objective loss function. In this work, we present a PINN
approach to solving the equations of coupled flow and deformation in porous
media for both single-phase and multiphase flow. To this end, we construct the
solution space using multi-layer neural networks. Due to the dynamics of the
problem, we find that incorporating multiple differential relations into the
loss function results in an unstable optimization problem, meaning that
sometimes it converges to the trivial null solution, other times it moves very
far from the expected solution. We report a dimensionless form of the coupled
governing equations that we find most favourable to the optimizer.
Additionally, we propose a sequential training approach based on the
stress-split algorithms of poromechanics. Notably, we find that sequential
training based on stress-split performs well for different problems, while the
classical strain-split algorithm shows an unstable behaviour similar to what is
reported in the context of finite element solvers. We use the approach to solve
benchmark problems of poroelasticity, including Mandel's consolidation problem,
Barry-Mercer's injection-production problem, and a reference two-phase drainage
problem. The Python-SciANN codes reproducing the results reported in this
manuscript will be made publicly available at
this https URL.

    

### [[2110.03051] A Survey on Evidential Deep Learning For Single-Pass Uncertainty Estimation](http://arxiv.org/abs/2110.03051)


  Popular approaches for quantifying predictive uncertainty in deep neural
networks often involve a set of weights or models, for instance via ensembling
or Monte Carlo Dropout. These techniques usually produce overhead by having to
train multiple model instances or do not produce very diverse predictions. This
survey aims to familiarize the reader with an alternative class of models based
on the concept of Evidential Deep Learning: For unfamiliar data, they admit
"what they don't know" and fall back onto a prior belief. Furthermore, they
allow uncertainty estimation in a single model and forward pass by
parameterizing distributions over distributions. This survey recapitulates
existing works, focusing on the implementation in a classification setting.
Finally, we survey the application of the same paradigm to regression problems.
We also provide a reflection on the strengths and weaknesses of the mentioned
approaches compared to existing ones and provide the most central theoretical
results in order to inform future research.

    

### [[2110.03061] Automatic Tuning of Federated Learning Hyper-Parameters from System Perspective](http://arxiv.org/abs/2110.03061)


  Federated learning (FL) is a distributed model training paradigm that
preserves clients' data privacy. FL hyper-parameters significantly affect the
training overheads in terms of time, computation, and communication. However,
the current practice of manually selecting FL hyper-parameters puts a high
burden on FL practitioners since various applications prefer different training
preferences. In this paper, we propose FedTuning, an automatic FL
hyper-parameter tuning algorithm tailored to applications' diverse system
requirements of FL training. FedTuning is lightweight and flexible, achieving
an average of 41% improvement for different training preferences on time,
computation, and communication compared to fixed FL hyper-parameters. FedTuning
is available at this https URL.

    

### [[2110.03068] Learning the Optimal Recommendation from Explorative Users](http://arxiv.org/abs/2110.03068)


  We propose a new problem setting to study the sequential interactions between
a recommender system and a user. Instead of assuming the user is omniscient,
static, and explicit, as the classical practice does, we sketch a more
realistic user behavior model, under which the user: 1) rejects recommendations
if they are clearly worse than others; 2) updates her utility estimation based
on rewards from her accepted recommendations; 3) withholds realized rewards
from the system. We formulate the interactions between the system and such an
explorative user in a $K$-armed bandit framework and study the problem of
learning the optimal recommendation on the system side. We show that efficient
system learning is still possible but is more difficult. In particular, the
system can identify the best arm with probability at least $1-\delta$ within
$O(1/\delta)$ interactions, and we prove this is tight. Our finding contrasts
the result for the problem of best arm identification with fixed confidence, in
which the best arm can be identified with probability $1-\delta$ within
$O(\log(1/\delta))$ interactions. This gap illustrates the inevitable cost the
system has to pay when it learns from an explorative user's revealed
preferences on its recommendations rather than from the realized rewards.

    

### [[2110.03070] Robust Algorithms for GMM Estimation: A Finite Sample Viewpoint](http://arxiv.org/abs/2110.03070)


  For many inference problems in statistics and econometrics, the unknown
parameter is identified by a set of moment conditions. A generic method of
solving moment conditions is the Generalized Method of Moments (GMM). However,
classical GMM estimation is potentially very sensitive to outliers. Robustified
GMM estimators have been developed in the past, but suffer from several
drawbacks: computational intractability, poor dimension-dependence, and no
quantitative recovery guarantees in the presence of a constant fraction of
outliers. In this work, we develop the first computationally efficient GMM
estimator (under intuitive assumptions) that can tolerate a constant $\epsilon$
fraction of adversarially corrupted samples, and that has an $\ell_2$ recovery
guarantee of $O(\sqrt{\epsilon})$. To achieve this, we draw upon and extend a
recent line of work on algorithmic robust statistics for related but simpler
problems such as mean estimation, linear regression and stochastic
optimization. As two examples of the generality of our algorithm, we show how
our estimation algorithm and assumptions apply to instrumental variables linear
and logistic regression. Moreover, we experimentally validate that our
estimator outperforms classical IV regression and two-stage Huber regression on
synthetic and semi-synthetic datasets with corruption.

    

### [[2110.03072] FOD-A: A Dataset for Foreign Object Debris in Airports](http://arxiv.org/abs/2110.03072)


  Foreign Object Debris (FOD) detection has attracted increased attention in
the area of machine learning and computer vision. However, a robust and
publicly available image dataset for FOD has not been initialized. To this end,
this paper introduces an image dataset of FOD, named FOD in Airports (FOD-A).
FOD-A object categories have been selected based on guidance from prior
documentation and related research by the Federal Aviation Administration
(FAA). In addition to the primary annotations of bounding boxes for object
detection, FOD-A provides labeled environmental conditions. As such, each
annotation instance is further categorized into three light level categories
(bright, dim, and dark) and two weather categories (dry and wet). Currently,
FOD-A has released 31 object categories and over 30,000 annotation instances.
This paper presents the creation methodology, discusses the publicly available
dataset extension process, and demonstrates the practicality of FOD-A with
widely used machine learning models for object detection.

    

### [[2110.03092] A Uniform Framework for Anomaly Detection in Deep Neural Networks](http://arxiv.org/abs/2110.03092)


  Deep neural networks (DNN) can achieve high performance when applied to
In-Distribution (ID) data which come from the same distribution as the training
set. When presented with anomaly inputs not from the ID, the outputs of a DNN
should be regarded as meaningless. However, modern DNN often predict anomaly
inputs as an ID class with high confidence, which is dangerous and misleading.
In this work, we consider three classes of anomaly inputs, (1) natural inputs
from a different distribution than the DNN is trained for, known as
Out-of-Distribution (OOD) samples, (2) crafted inputs generated from ID by
attackers, often known as adversarial (AD) samples, and (3) noise (NS) samples
generated from meaningless data. We propose a framework that aims to detect all
these anomalies for a pre-trained DNN. Unlike some of the existing works, our
method does not require preprocessing of input data, nor is it dependent to any
known OOD set or adversarial attack algorithm. Through extensive experiments
over a variety of DNN models for the detection of aforementioned anomalies, we
show that in most cases our method outperforms state-of-the-art anomaly
detection methods in identifying all three classes of anomalies.

    

### [[2110.03095] Which Shortcut Cues Will DNNs Choose? A Study from the Parameter-Space Perspective](http://arxiv.org/abs/2110.03095)


  Deep neural networks (DNNs) often rely on easy-to-learn discriminatory
features, or cues, that are not necessarily essential to the problem at hand.
For example, ducks in an image may be recognized based on their typical
background scenery, such as lakes or streams. This phenomenon, also known as
shortcut learning, is emerging as a key limitation of the current generation of
machine learning models. In this work, we introduce a set of experiments to
deepen our understanding of shortcut learning and its implications. We design a
training setup with several shortcut cues, named WCST-ML, where each cue is
equally conducive to the visual recognition problem at hand. Even under equal
opportunities, we observe that (1) certain cues are preferred to others, (2)
solutions biased to the easy-to-learn cues tend to converge to relatively flat
minima on the loss surface, and (3) the solutions focusing on those preferred
cues are far more abundant in the parameter space. We explain the abundance of
certain cues via their Kolmogorov (descriptional) complexity: solutions
corresponding to Kolmogorov-simple cues are abundant in the parameter space and
are thus preferred by DNNs. Our studies are based on the synthetic dataset
DSprites and the face dataset UTKFace. In our WCST-ML, we observe that the
inborn bias of models leans toward simple cues, such as color and ethnicity.
Our findings emphasize the importance of active human intervention to remove
the inborn model biases that may cause negative societal impacts.

    

### [[2110.03097] SWAT Watershed Model Calibration using Deep Learning](http://arxiv.org/abs/2110.03097)


  Watershed models such as the Soil and Water Assessment Tool (SWAT) consist of
high-dimensional physical and empirical parameters. These parameters need to be
accurately calibrated for models to produce reliable predictions for
streamflow, evapotranspiration, snow water equivalent, and nutrient loading.
Existing parameter estimation methods are time-consuming, inefficient, and
computationally intensive, with reduced accuracy when estimating
high-dimensional parameters. In this paper, we present a fast, accurate, and
reliable methodology to calibrate the SWAT model (i.e., 21 parameters) using
deep learning (DL). We develop DL-enabled inverse models based on convolutional
neural networks to ingest streamflow data and estimate the SWAT model
parameters. Hyperparameter tuning is performed to identify the optimal neural
network architecture and the nine next best candidates. We use ensemble SWAT
simulations to train, validate, and test the above DL models. We estimated the
actual parameters of the SWAT model using observational data. We test and
validate the proposed DL methodology on the American River Watershed, located
in the Pacific Northwest-based Yakima River basin. Our results show that the DL
models-based calibration is better than traditional parameter estimation
methods, such as generalized likelihood uncertainty estimation (GLUE). The
behavioral parameter sets estimated by DL have narrower ranges than GLUE and
produce values within the sampling range even under high relative observational
errors. This narrow range of parameters shows the reliability of the proposed
workflow to estimate sensitive parameters accurately even under noise. Due to
its fast and reasonably accurate estimations of process parameters, the
proposed DL workflow is attractive for calibrating integrated hydrologic models
for large spatial-scale applications.

    

### [[2110.03098] CTC Variations Through New WFST Topologies](http://arxiv.org/abs/2110.03098)


  This paper presents novel Weighted Finite-State Transducer (WFST) topologies
to implement Connectionist Temporal Classification (CTC)-like algorithms for
automatic speech recognition. Three new CTC variants are proposed: (1) the
"compact-CTC", in which direct transitions between units are replaced with
<epsilon> back-off transitions; (2) the "minimal-CTC", that only adds <blank>
self-loops when used in WFST-composition; and (3) "selfless-CTC", that
disallows self-loop for non-blank units. The new CTC variants have several
benefits, such as reducing decoding graph size and GPU memory required for
training while keeping model accuracy.

    

### [[2110.03101] SPEED+: Next Generation Dataset for Spacecraft Pose Estimation across Domain Gap](http://arxiv.org/abs/2110.03101)


  Autonomous vision-based spaceborne navigation is an enabling technology for
future on-orbit servicing and space logistics missions. While computer vision
in general has benefited from Machine Learning (ML), training and validating
spaceborne ML models are extremely challenging due to the impracticality of
acquiring a large-scale labeled dataset of images of the intended target in the
space environment. Existing datasets, such as Spacecraft PosE Estimation
Dataset (SPEED), have so far mostly relied on synthetic images for both
training and validation, which are easy to mass-produce but fail to resemble
the visual features and illumination variability inherent to the target
spaceborne images. In order to bridge the gap between the current practices and
the intended applications in future space missions, this paper introduces
SPEED+: the next generation spacecraft pose estimation dataset with specific
emphasis on domain gap. In addition to 60,000 synthetic images for training,
SPEED+ includes 9,531 simulated images of a spacecraft mockup model captured
from the Testbed for Rendezvous and Optical Navigation (TRON) facility. TRON is
a first-of-a-kind robotic testbed capable of capturing an arbitrary number of
target images with accurate and maximally diverse pose labels and high-fidelity
spaceborne illumination conditions. SPEED+ will be used in the upcoming
international Satellite Pose Estimation Challenge co-hosted with the Advanced
Concepts Team of the European Space Agency to evaluate and compare the
robustness of spaceborne ML models trained on synthetic images.

    

### [[2110.03104] Hybrid Pointer Networks for Traveling Salesman Problems Optimization](http://arxiv.org/abs/2110.03104)


  In this work, a novel idea is presented for combinatorial optimization
problems, a hybrid network, which results in a superior outcome. We applied
this method to graph pointer networks [1], expanding its capabilities to a
higher level. We proposed a hybrid pointer network (HPN) to solve the
travelling salesman problem trained by reinforcement learning. Furthermore, HPN
builds upon graph pointer networks which is an extension of pointer networks
with an additional graph embedding layer. HPN outperforms the graph pointer
network in solution quality due to the hybrid encoder, which provides our model
with a verity encoding type, allowing our model to converge to a better policy.
Our network significantly outperforms the original graph pointer network for
small and large-scale problems increasing its performance for TSP50 from 5.959
to 5.706 without utilizing 2opt, Pointer networks, Attention model, and a wide
range of models, producing results comparable to highly tuned and specialized
algorithms. We make our data, models, and code publicly available [2].

    

### [[2110.03106] Multi-Trigger-Key: Towards Multi-Task Privacy Preserving In Deep Learning](http://arxiv.org/abs/2110.03106)


  Deep learning-based Multi-Task Classification (MTC) is widely used in
applications like facial attributes and healthcare that warrant strong privacy
guarantees. In this work, we aim to protect sensitive information in the
inference phase of MTC and propose a novel Multi-Trigger-Key (MTK) framework to
achieve the privacy-preserving objective. MTK associates each secured task in
the multi-task dataset with a specifically designed trigger-key. The true
information can be revealed by adding the trigger-key if the user is
authorized. We obtain such an MTK model by training it with a newly generated
training set. To address the information leakage malaise resulting from
correlations among different tasks, we generalize the training process by
incorporating an MTK decoupling process with a controllable trade-off between
the protective efficacy and the model performance. Theoretical guarantees and
experimental results demonstrate the effectiveness of the privacy protection
without appreciable hindering on the model performance.

    

### [[2110.03109] Consistent Counterfactuals for Deep Models](http://arxiv.org/abs/2110.03109)


  Counterfactual examples are one of the most commonly-cited methods for
explaining the predictions of machine learning models in key areas such as
finance and medical diagnosis. Counterfactuals are often discussed under the
assumption that the model on which they will be used is static, but in
deployment models may be periodically retrained or fine-tuned. This paper
studies the consistency of model prediction on counterfactual examples in deep
networks under small changes to initial training conditions, such as weight
initialization and leave-one-out variations in data, as often occurs during
model deployment. We demonstrate experimentally that counterfactual examples
for deep models are often inconsistent across such small changes, and that
increasing the cost of the counterfactual, a stability-enhancing mitigation
suggested by prior work in the context of simpler models, is not a reliable
heuristic in deep networks. Rather, our analysis shows that a model's local
Lipschitz continuity around the counterfactual is key to its consistency across
related models. To this end, we propose Stable Neighbor Search as a way to
generate more consistent counterfactual explanations, and illustrate the
effectiveness of this approach on several benchmark datasets.

    

### [[2110.03120] Assurance Monitoring of Learning Enabled Cyber-Physical Systems Using Inductive Conformal Prediction based on Distance Learning](http://arxiv.org/abs/2110.03120)


  Machine learning components such as deep neural networks are used extensively
in Cyber-Physical Systems (CPS). However, such components may introduce new
types of hazards that can have disastrous consequences and need to be addressed
for engineering trustworthy systems. Although deep neural networks offer
advanced capabilities, they must be complemented by engineering methods and
practices that allow effective integration in CPS. In this paper, we proposed
an approach for assurance monitoring of learning-enabled CPS based on the
conformal prediction framework. In order to allow real-time assurance
monitoring, the approach employs distance learning to transform
high-dimensional inputs into lower size embedding representations. By
leveraging conformal prediction, the approach provides well-calibrated
confidence and ensures a bounded small error rate while limiting the number of
inputs for which an accurate prediction cannot be made. We demonstrate the
approach using three data sets of mobile robot following a wall, speaker
recognition, and traffic sign recognition. The experimental results demonstrate
that the error rates are well-calibrated while the number of alarms is very
small. Further, the method is computationally efficient and allows real-time
assurance monitoring of CPS.

    

### [[2110.03123] Improving Prediction Confidence in Learning-Enabled Autonomous Systems](http://arxiv.org/abs/2110.03123)


  Autonomous systems use extensively learning-enabled components such as deep
neural networks (DNNs) for prediction and decision making. In this paper, we
utilize a feedback loop between learning-enabled components used for
classification and the sensors of an autonomous system in order to improve the
confidence of the predictions. We design a classifier using Inductive Conformal
Prediction (ICP) based on a triplet network architecture in order to learn
representations that can be used to quantify the similarity between test and
training examples. The method allows computing confident set predictions with
an error rate predefined using a selected significance level. A feedback loop
that queries the sensors for a new input is used to further refine the
predictions and increase the classification accuracy. The method is
computationally efficient, scalable to high-dimensional inputs, and can be
executed in a feedback loop with the system in real-time. The approach is
evaluated using a traffic sign recognition dataset and the results show that
the error rate is reduced.

    

### [[2110.03124] Improving Adversarial Robustness for Free with Snapshot Ensemble](http://arxiv.org/abs/2110.03124)


  Adversarial training, as one of the few certified defenses against
adversarial attacks, can be quite complicated and time-consuming, while the
results might not be robust enough. To address the issue of lack of robustness,
ensemble methods were proposed, aiming to get the final output by weighting the
selected results from repeatedly trained processes. It is proved to be very
useful in achieving robust and accurate results, but the computational and
memory costs are even higher. Snapshot ensemble, a new ensemble method that
combines several local minima in a single training process to make the final
prediction, was proposed recently, which reduces the time spent on training
multiple networks and the memory to store the results. Based on the snapshot
ensemble, we present a new method that is easier to implement: unlike original
snapshot ensemble that seeks for local minima, our snapshot ensemble focuses on
the last few iterations of a training and stores the sets of parameters from
them. Our algorithm is much simpler but the results are no less accurate than
the original ones: based on different hyperparameters and datasets, our
snapshot ensemble has shown a 5% to 30% increase in accuracy when compared to
the traditional adversarial training.

    

### [[2110.03127] Reliable Probability Intervals For Classification Using Inductive Venn Predictors Based on Distance Learning](http://arxiv.org/abs/2110.03127)


  Deep neural networks are frequently used by autonomous systems for their
ability to learn complex, non-linear data patterns and make accurate
predictions in dynamic environments. However, their use as black boxes
introduces risks as the confidence in each prediction is unknown. Different
frameworks have been proposed to compute accurate confidence measures along
with the predictions but at the same time introduce a number of limitations
like execution time overhead or inability to be used with high-dimensional
data. In this paper, we use the Inductive Venn Predictors framework for
computing probability intervals regarding the correctness of each prediction in
real-time. We propose taxonomies based on distance metric learning to compute
informative probability intervals in applications involving high-dimensional
inputs. Empirical evaluation on image classification and botnet attacks
detection in Internet-of-Things (IoT) applications demonstrates improved
accuracy and calibration. The proposed method is computationally efficient, and
therefore, can be used in real-time.

    

### [[2110.03128] On the Generalization of Models Trained with SGD: Information-Theoretic Bounds and Implications](http://arxiv.org/abs/2110.03128)


  This paper follows up on a recent work of (Neu, 2021) and presents new and
tighter information-theoretic upper bounds for the generalization error of
machine learning models, such as neural networks, trained with SGD. We apply
these bounds to analyzing the generalization behaviour of linear and two-layer
ReLU networks. Experimental study based on these bounds provide some insights
on the SGD training of neural networks. They also point to a new and simple
regularization scheme which we show performs comparably to the current state of
the art.

    

### [[2110.03135] Double Descent in Adversarial Training: An Implicit Label Noise Perspective](http://arxiv.org/abs/2110.03135)


  Here, we show that the robust overfitting shall be viewed as the early part
of an epoch-wise double descent -- the robust test error will start to decrease
again after training the model for a considerable number of epochs. Inspired by
our observations, we further advance the analyses of double descent to
understand robust overfitting better. In standard training, double descent has
been shown to be a result of label flipping noise. However, this reasoning is
not applicable in our setting, since adversarial perturbations are believed not
to change the label. Going beyond label flipping noise, we propose to measure
the mismatch between the assigned and (unknown) true label distributions,
denoted as \emph{implicit label noise}. We show that the traditional labeling
of adversarial examples inherited from their clean counterparts will lead to
implicit label noise. Towards better labeling, we show that predicted
distribution from a classifier, after scaling and interpolation, can provably
reduce the implicit label noise under mild assumptions. In light of our
analyses, we tailored the training objective accordingly to effectively
mitigate the double descent and verified its effectiveness on three benchmark
datasets.

    

### [[2110.03144] Conceptual Expansion Neural Architecture Search (CENAS)](http://arxiv.org/abs/2110.03144)


  Architecture search optimizes the structure of a neural network for some task
instead of relying on manual authoring. However, it is slow, as each potential
architecture is typically trained from scratch. In this paper we present an
approach called Conceptual Expansion Neural Architecture Search (CENAS) that
combines a sample-efficient, computational creativity-inspired transfer
learning approach with neural architecture search. This approach finds models
faster than naive architecture search via transferring existing weights to
approximate the parameters of the new model. It outperforms standard transfer
learning by allowing for the addition of features instead of only modifying
existing features. We demonstrate that our approach outperforms standard neural
architecture search and transfer learning methods in terms of efficiency,
performance, and parameter counts on a variety of transfer learning tasks.

    

### [[2110.03146] Solving Multistage Stochastic Linear Programming via Regularized Linear Decision Rules: An Application to Hydrothermal Dispatch Planning](http://arxiv.org/abs/2110.03146)


  The solution of multistage stochastic linear problems (MSLP) represents a
challenge for many applications. Long-term hydrothermal dispatch planning
(LHDP) materializes this challenge in a real-world problem that affects
electricity markets, economies, and natural resources worldwide. No closed-form
solutions are available for MSLP and the definition of non-anticipative
policies with high-quality out-of-sample performance is crucial. Linear
decision rules (LDR) provide an interesting simulation-based framework for
finding high-quality policies to MSLP through two-stage stochastic models. In
practical applications, however, the number of parameters to be estimated when
using an LDR may be close or higher than the number of scenarios, thereby
generating an in-sample overfit and poor performances in out-of-sample
simulations. In this paper, we propose a novel regularization scheme for LDR
based on the AdaLASSO (adaptive least absolute shrinkage and selection
operator). The goal is to use the parsimony principle as largely studied in
high-dimensional linear regression models to obtain better out-of-sample
performance for an LDR applied to MSLP. Computational experiments show that the
overfit threat is non-negligible when using the classical non-regularized LDR
to solve MSLP. For the LHDP problem, our analysis highlights the following
benefits of the proposed framework in comparison to the non-regularized
benchmark: 1) significant reductions in the number of non-zero coefficients
(model parsimony), 2) substantial cost reductions in out-of-sample evaluations,
and 3) improved spot-price profiles.

    

### [[2110.03147] PRRS Outbreak Prediction via Deep Switching Auto-Regressive Factorization Modeling](http://arxiv.org/abs/2110.03147)


  We propose an epidemic analysis framework for the outbreak prediction in the
livestock industry, focusing on the study of the most costly and viral
infectious disease in the swine industry -- the PRRS virus. Using this
framework, we can predict the PRRS outbreak in all farms of a swine production
system by capturing the spatio-temporal dynamics of infection transmission
based on the intra-farm pig-level virus transmission dynamics, and inter-farm
pig shipment network. We simulate a PRRS infection epidemic based on the
shipment network and the SEIR epidemic model using the statistics extracted
from real data provided by the swine industry. We develop a hierarchical
factorized deep generative model that approximates high dimensional data by a
product between time-dependent weights and spatially dependent low dimensional
factors to perform per farm time series prediction. The prediction results
demonstrate the ability of the model in forecasting the virus spread
progression with average error of NRMSE = 2.5\%.

    

### [[2110.03149] Data-driven behavioural biometrics for continuous and adaptive user verification using Smartphone and Smartwatch](http://arxiv.org/abs/2110.03149)


  Recent studies have shown how motion-based biometrics can be used as a form
of user authentication and identification without requiring any human
cooperation. This category of behavioural biometrics deals with the features we
learn in our life as a result of our interaction with the environment and
nature. This modality is related to change in human behaviour over time. The
developments in these methods aim to amplify continuous authentication such as
biometrics to protect their privacy on user devices. Various Continuous
Authentication (CA) systems have been proposed in the literature. They
represent a new generation of security mechanisms that continuously monitor
user behaviour and use this as the basis to re-authenticate them periodically
throughout a login session. However, these methods usually constitute a single
classification model which is used to identify or verify a user. This work
proposes an algorithm to blend behavioural biometrics with multi-factor
authentication (MFA) by introducing a two-step user verification algorithm that
verifies the user's identity using motion-based biometrics and complements the
multi-factor authentication, thus making it more secure and flexible. This
two-step user verification algorithm is also immune to adversarial attacks,
based on our experimental results which show how the rate of misclassification
drops while using this model with adversarial data.

    

### [[2110.03155] Towards Understanding Distributional Reinforcement Learning: Regularization, Optimization, Acceleration and Sinkhorn Algorithm](http://arxiv.org/abs/2110.03155)


  Distributional reinforcement learning~(RL) is a class of state-of-the-art
algorithms that estimate the whole distribution of the total return rather than
only its expectation. Despite the remarkable performance of distributional RL,
a theoretical understanding of its advantages over expectation-based RL remains
elusive. In this paper, we interpret distributional RL as entropy-regularized
maximum likelihood estimation in the \textit{neural Z-fitted iteration}
framework, and establish the connection of the resulting risk-aware
regularization with maximum entropy RL. In addition, We shed light on the
stability-promoting distributional loss with desirable smoothness properties in
distributional RL, which can yield stable optimization and guaranteed
generalization. We also analyze the acceleration behavior while optimizing
distributional RL algorithms and show that an appropriate approximation to the
true target distribution can speed up the convergence. From the perspective of
representation, we find that distributional RL encourages state representation
from the same action class classified by the policy in tighter clusters.
Finally, we propose a class of \textit{Sinkhorn distributional RL} algorithm
that interpolates between the Wasserstein distance and maximum mean
discrepancy~(MMD). Experiments on a suite of Atari games reveal the competitive
performance of our algorithm relative to existing state-of-the-art
distributional RL algorithms.

    

### [[2110.03165] Offline RL With Resource Constrained Online Deployment](http://arxiv.org/abs/2110.03165)


  Offline reinforcement learning is used to train policies in scenarios where
real-time access to the environment is expensive or impossible. As a natural
consequence of these harsh conditions, an agent may lack the resources to fully
observe the online environment before taking an action. We dub this situation
the resource-constrained setting. This leads to situations where the offline
dataset (available for training) can contain fully processed features (using
powerful language models, image models, complex sensors, etc.) which are not
available when actions are actually taken online. This disconnect leads to an
interesting and unexplored problem in offline RL: Is it possible to use a
richly processed offline dataset to train a policy which has access to fewer
features in the online environment? In this work, we introduce and formalize
this novel resource-constrained problem setting. We highlight the performance
gap between policies trained using the full offline dataset and policies
trained using limited features. We address this performance gap with a policy
transfer algorithm which first trains a teacher agent using the offline dataset
where features are fully available, and then transfers this knowledge to a
student agent that only uses the resource-constrained features. To better
capture the challenge of this setting, we propose a data collection procedure:
Resource Constrained-Datasets for RL (RC-D4RL). We evaluate our transfer
algorithm on RC-D4RL and the popular D4RL benchmarks and observe consistent
improvement over the baseline (TD3+BC without transfer). The code for the
experiments is available at
this https URL}{this http URL.

    

### [[2110.03171] Assemblies of neurons can learn to classify well-separated distributions](http://arxiv.org/abs/2110.03171)


  Assemblies are patterns of coordinated firing across large populations of
neurons, believed to represent higher-level information in the brain, such as
memories, concepts, words, and other cognitive categories. Recently, a
computational system called the Assembly Calculus (AC) has been proposed, based
on a set of biologically plausible operations on assemblies. This system is
capable of simulating arbitrary space-bounded computation, and describes quite
naturally complex cognitive phenomena such as language. However, the question
of whether assemblies can perform the brain's greatest trick -- its ability to
learn -- has been open. We show that the AC provides a mechanism for learning
to classify samples from well-separated classes. We prove rigorously that for
simple classification problems, a new assembly that represents each class can
be reliably formed in response to a few stimuli from it; this assembly is
henceforth reliably recalled in response to new stimuli from the same class.
Furthermore, such class assemblies will be distinguishable as long as the
respective classes are reasonably separated, in particular when they are
clusters of similar assemblies, or more generally divided by a halfspace with
margin. Experimentally, we demonstrate the successful formation of assemblies
which represent concept classes on synthetic data drawn from these
distributions, and also on MNIST, which lends itself to classification through
one assembly per digit. Seen as a learning algorithm, this mechanism is
entirely online, generalizes from very few samples, and requires only mild
supervision -- all key attributes of learning in a model of the brain.

    

### [[2110.03173] Multi-objective Optimization by Learning Space Partitions](http://arxiv.org/abs/2110.03173)


  In contrast to single-objective optimization (SOO), multi-objective
optimization (MOO) requires an optimizer to find the Pareto frontier, a subset
of feasible solutions that are not dominated by other feasible solutions. In
this paper, we propose LaMOO, a novel multi-objective optimizer that learns a
model from observed samples to partition the search space and then focus on
promising regions that are likely to contain a subset of the Pareto frontier.
The partitioning is based on the dominance number, which measures "how close" a
data point is to the Pareto frontier among existing samples. To account for
possible partition errors due to limited samples and model mismatch, we
leverage Monte Carlo Tree Search (MCTS) to exploit promising regions while
exploring suboptimal regions that may turn out to contain good solutions later.
Theoretically, we prove the efficacy of learning space partitioning via LaMOO
under certain assumptions. Empirically, on the HyperVolume (HV) benchmark, a
popular MOO metric, LaMOO substantially outperforms strong baselines on
multiple real-world MOO tasks, by up to 225% in sample efficiency for neural
architecture search on Nasbench201, and up to 10% for molecular design.

    

### [[2110.03177] EE-Net: Exploitation-Exploration Neural Networks in Contextual Bandits](http://arxiv.org/abs/2110.03177)


  Contextual multi-armed bandits have been studied for decades and adapted to
various applications such as online advertising and personalized
recommendation. To solve the exploitation-exploration tradeoff in bandits,
there are three main techniques: epsilon-greedy, Thompson Sampling (TS), and
Upper Confidence Bound (UCB). In recent literature, linear contextual bandits
have adopted ridge regression to estimate the reward function and combine it
with TS or UCB strategies for exploration. However, this line of works
explicitly assumes the reward is based on a linear function of arm vectors,
which may not be true in real-world datasets. To overcome this challenge, a
series of neural-based bandit algorithms have been proposed, where a neural
network is assigned to learn the underlying reward function and TS or UCB are
adapted for exploration. In this paper, we propose "EE-Net", a neural-based
bandit approach with a novel exploration strategy. In addition to utilizing a
neural network (Exploitation network) to learn the reward function, EE-Net
adopts another neural network (Exploration network) to adaptively learn
potential gains compared to currently estimated reward. Then, a decision-maker
is constructed to combine the outputs from the Exploitation and Exploration
networks. We prove that EE-Net achieves $\mathcal{O}(\sqrt{T\log T})$ regret,
which is tighter than existing state-of-the-art neural bandit algorithms
($\mathcal{O}(\sqrt{T}\log T)$ for both UCB-based and TS-based). Through
extensive experiments on four real-world datasets, we show that EE-Net
outperforms existing linear and neural bandit approaches.

    

### [[2110.03181] Tile Embedding: A General Representation for Procedural Level Generation via Machine Learning](http://arxiv.org/abs/2110.03181)


  In recent years, Procedural Level Generation via Machine Learning (PLGML)
techniques have been applied to generate game levels with machine learning.
These approaches rely on human-annotated representations of game levels.
Creating annotated datasets for games requires domain knowledge and is
time-consuming. Hence, though a large number of video games exist, annotated
datasets are curated only for a small handful. Thus current PLGML techniques
have been explored in limited domains, with Super Mario Bros. as the most
common example. To address this problem, we present tile embeddings, a unified,
affordance-rich representation for tile-based 2D games. To learn this
embedding, we employ autoencoders trained on the visual and semantic
information of tiles from a set of existing, human-annotated games. We evaluate
this representation on its ability to predict affordances for unseen tiles, and
to serve as a PLGML representation for annotated and unannotated games.

    

### [[2110.03183] Attention is All You Need? Good Embeddings with Statistics are enough: Audio Understanding WITHOUT Convolutions/Transformers/BERTs/Mixers/Attention/RNNs or ....](http://arxiv.org/abs/2110.03183)


  This paper presents a way of doing large scale audio understanding without
traditional state of the art neural architectures. Ever since the introduction
of deep learning for understanding audio signals in the past decade,
convolutional architectures have been able to achieve state of the art results
surpassing traditional hand-crafted features. In the recent past, there has
been a similar shift away from traditional convolutional and recurrent neural
networks towards purely end-to-end Transformer architectures. We, in this work,
explore an approach, based on Bag-of-Words model. Our approach does not have
any convolutions, recurrence, attention, transformers or other approaches such
as BERT. We utilize micro and macro level clustered vanilla embeddings, and use
a MLP head for classification. We only use feed-forward encoder-decoder models
to get the bottlenecks of spectral envelops, spectral patches and slices as
well as multi-resolution spectra. A classification head (a feed-forward layer),
similar to the approach in SimCLR is trained on a learned representation. Using
simple codes learned on latent representations, we show how we surpass
traditional convolutional neural network architectures, and come strikingly
close to outperforming powerful Transformer architectures. This work hopefully
would pave way for exciting advancements in the field of representation
learning without massive, end-to-end neural architectures.

    

### [[2110.03184] Explaining Deep Reinforcement Learning Agents In The Atari Domain through a Surrogate Model](http://arxiv.org/abs/2110.03184)


  One major barrier to applications of deep Reinforcement Learning (RL) both
inside and outside of games is the lack of explainability. In this paper, we
describe a lightweight and effective method to derive explanations for deep RL
agents, which we evaluate in the Atari domain. Our method relies on a
transformation of the pixel-based input of the RL agent to an interpretable,
percept-like input representation. We then train a surrogate model, which is
itself interpretable, to replicate the behavior of the target, deep RL agent.
Our experiments demonstrate that we can learn an effective surrogate that
accurately approximates the underlying decision making of a target agent on a
suite of Atari games.

    

### [[2110.03187] On the Optimal Memorization Power of ReLU Neural Networks](http://arxiv.org/abs/2110.03187)


  We study the memorization power of feedforward ReLU neural networks. We show
that such networks can memorize any $N$ points that satisfy a mild separability
assumption using $\tilde{O}\left(\sqrt{N}\right)$ parameters. Known
VC-dimension upper bounds imply that memorizing $N$ samples requires
$\Omega(\sqrt{N})$ parameters, and hence our construction is optimal up to
logarithmic factors. We also give a generalized construction for networks with
depth bounded by $1 \leq L \leq \sqrt{N}$, for memorizing $N$ samples using
$\tilde{O}(N/L)$ parameters. This bound is also optimal up to logarithmic
factors. Our construction uses weights with large bit complexity. We prove that
having such a large bit complexity is both necessary and sufficient for
memorization with a sub-linear number of parameters.

    

### [[2110.03195] Coresets for Decision Trees of Signals](http://arxiv.org/abs/2110.03195)


  A $k$-decision tree $t$ (or $k$-tree) is a recursive partition of a matrix
(2D-signal) into $k\geq 1$ block matrices (axis-parallel rectangles, leaves)
where each rectangle is assigned a real label. Its regression or classification
loss to a given matrix $D$ of $N$ entries (labels) is the sum of squared
differences over every label in $D$ and its assigned label by $t$. Given an
error parameter $\varepsilon\in(0,1)$, a $(k,\varepsilon)$-coreset $C$ of $D$
is a small summarization that provably approximates this loss to \emph{every}
such tree, up to a multiplicative factor of $1\pm\varepsilon$. In particular,
the optimal $k$-tree of $C$ is a $(1+\varepsilon)$-approximation to the optimal
$k$-tree of $D$.
We provide the first algorithm that outputs such a $(k,\varepsilon)$-coreset
for \emph{every} such matrix $D$. The size $|C|$ of the coreset is polynomial
in $k\log(N)/\varepsilon$, and its construction takes $O(Nk)$ time. This is by
forging a link between decision trees from machine learning -- to partition
trees in computational geometry.
Experimental results on \texttt{sklearn} and \texttt{lightGBM} show that
applying our coresets on real-world data-sets boosts the computation time of
random forests and their parameter tuning by up to x$10$, while keeping similar
accuracy. Full open source code is provided.

    

### [[2110.03210] Universality of Deep Neural Network Lottery Tickets: A Renormalization Group Perspective](http://arxiv.org/abs/2110.03210)


  Foundational work on the Lottery Ticket Hypothesis has suggested an exciting
corollary: winning tickets found in the context of one task can be transferred
to similar tasks, possibly even across different architectures. While this has
become of broad practical and theoretical interest, to date, there exists no
detailed understanding of why winning ticket universality exists, or any way of
knowing \textit{a priori} whether a given ticket can be transferred to a given
task. To address these outstanding open questions, we make use of
renormalization group theory, one of the most successful tools in theoretical
physics. We find that iterative magnitude pruning, the method used for
discovering winning tickets, is a renormalization group scheme. This opens the
door to a wealth of existing numerical and theoretical tools, some of which we
leverage here to examine winning ticket universality in large scale lottery
ticket experiments, as well as sheds new light on the success iterative
magnitude pruning has found in the field of sparse machine learning.

    

### [[2110.03215] Towards Continual Knowledge Learning of Language Models](http://arxiv.org/abs/2110.03215)


  Large Language Models (LMs) are known to encode world knowledge in their
parameters as they pretrain on a vast amount of web corpus, which is often
utilized for performing knowledge-dependent downstream tasks such as question
answering, fact-checking, and open dialogue. In real-world scenarios, the world
knowledge stored in the LMs can quickly become outdated as the world changes,
but it is non-trivial to avoid catastrophic forgetting and reliably acquire new
knowledge while preserving invariant knowledge. To push the community towards
better maintenance of ever-changing LMs, we formulate a new continual learning
(CL) problem called Continual Knowledge Learning (CKL). We construct a new
benchmark and metric to quantify the retention of time-invariant world
knowledge, the update of outdated knowledge, and the acquisition of new
knowledge. We adopt applicable recent methods from literature to create several
strong baselines. Through extensive experiments, we find that CKL exhibits
unique challenges that are not addressed in previous CL setups, where parameter
expansion is necessary to reliably retain and learn knowledge simultaneously.
By highlighting the critical causes of knowledge forgetting, we show that CKL
is a challenging and important problem that helps us better understand and
train ever-changing LMs.

    

### [[2110.03218] Joint optimization of system design and reconstruction in MIMO radar imaging](http://arxiv.org/abs/2110.03218)


  Multiple-input multiple-output (MIMO) radar is one of the leading depth
sensing modalities. However, the usage of multiple receive channels lead to
relative high costs and prevent the penetration of MIMOs in many areas such as
the automotive industry. Over the last years, few studies concentrated on
designing reduced measurement schemes and image reconstruction schemes for MIMO
radars, however these problems have been so far addressed separately. On the
other hand, recent works in optical computational imaging have demonstrated
growing success of simultaneous learning-based design of the acquisition and
reconstruction schemes, manifesting significant improvement in the
reconstruction quality. Inspired by these successes, in this work, we propose
to learn MIMO acquisition parameters in the form of receive (Rx) antenna
elements locations jointly with an image neural-network based reconstruction.
To this end, we propose an algorithm for training the combined
acquisition-reconstruction pipeline end-to-end in a differentiable way. We
demonstrate the significance of using our learned acquisition parameters with
and without the neural-network reconstruction.

    

### [[2110.03224] Darts: User-Friendly Modern Machine Learning for Time Series](http://arxiv.org/abs/2110.03224)


  We present Darts, a Python machine learning library for time series, with a
focus on forecasting. Darts offers a variety of models, from classics such as
ARIMA to state-of-the-art deep neural networks. The emphasis of the library is
on offering modern machine learning functionalities, such as supporting
multidimensional series, meta-learning on multiple series, training on large
datasets, incorporating external data, ensembling models, and providing a rich
support for probabilistic forecasting. At the same time, great care goes into
the API design to make it user-friendly and easy to use. For instance, all
models can be used using fit()/predict(), similar to scikit-learn.

    

### [[2110.03237] Score-based Generative Neural Networks for Large-Scale Optimal Transport](http://arxiv.org/abs/2110.03237)


  We consider the fundamental problem of sampling the optimal transport
coupling between given source and target distributions. In certain cases, the
optimal transport plan takes the form of a one-to-one mapping from the source
support to the target support, but learning or even approximating such a map is
computationally challenging for large and high-dimensional datasets due to the
high cost of linear programming routines and an intrinsic curse of
dimensionality. We study instead the Sinkhorn problem, a regularized form of
optimal transport whose solutions are couplings between the source and the
target distribution. We introduce a novel framework for learning the Sinkhorn
coupling between two distributions in the form of a score-based generative
model. Conditioned on source data, our procedure iterates Langevin Dynamics to
sample target data according to the regularized optimal coupling. Key to this
approach is a neural network parametrization of the Sinkhorn problem, and we
prove convergence of gradient descent with respect to network parameters in
this formulation. We demonstrate its empirical success on a variety of large
scale optimal transport tasks.

    

### [[2110.03239] Understanding Domain Randomization for Sim-to-real Transfer](http://arxiv.org/abs/2110.03239)


  Reinforcement learning encounters many challenges when applied directly in
the real world. Sim-to-real transfer is widely used to transfer the knowledge
learned from simulation to the real world. Domain randomization -- one of the
most popular algorithms for sim-to-real transfer -- has been demonstrated to be
effective in various tasks in robotics and autonomous driving. Despite its
empirical successes, theoretical understanding on why this simple algorithm
works is limited. In this paper, we propose a theoretical framework for
sim-to-real transfers, in which the simulator is modeled as a set of MDPs with
tunable parameters (corresponding to unknown physical parameters such as
friction). We provide sharp bounds on the sim-to-real gap -- the difference
between the value of policy returned by domain randomization and the value of
an optimal policy for the real world. We prove that sim-to-real transfer can
succeed under mild conditions without any real-world training samples. Our
theory also highlights the importance of using memory (i.e., history-dependent
policies) in domain randomization. Our proof is based on novel techniques that
reduce the problem of bounding the sim-to-real gap to the problem of designing
efficient learning algorithms for infinite-horizon MDPs, which we believe are
of independent interest.

    

### [[2110.03244] Near-Optimal Reward-Free Exploration for Linear Mixture MDPs with Plug-in Solver](http://arxiv.org/abs/2110.03244)


  Although model-based reinforcement learning (RL) approaches are considered
more sample efficient, existing algorithms are usually relying on sophisticated
planning algorithm to couple tightly with the model-learning procedure. Hence
the learned models may lack the ability of being re-used with more specialized
planners. In this paper we address this issue and provide approaches to learn
an RL model efficiently without the guidance of a reward signal. In particular,
we take a plug-in solver approach, where we focus on learning a model in the
exploration phase and demand that \emph{any planning algorithm} on the learned
model can give a near-optimal policy. Specicially, we focus on the linear
mixture MDP setting, where the probability transition matrix is a (unknown)
convex combination of a set of existing models. We show that, by establishing a
novel exploration algorithm, the plug-in approach learns a model by taking
$\tilde{O}(d^2H^3/\epsilon^2)$ interactions with the environment and \emph{any}
$\epsilon$-optimal planner on the model gives an $O(\epsilon)$-optimal policy
on the original model. This sample complexity matches lower bounds for
non-plug-in approaches and is \emph{statistically optimal}. We achieve this
result by leveraging a careful maximum total-variance bound using Bernstein
inequality and properties specified to linear mixture MDP.

    

### [[2110.03260] Improving MC-Dropout Uncertainty Estimates with Calibration Error-based Optimization](http://arxiv.org/abs/2110.03260)


  Uncertainty quantification of machine learning and deep learning methods
plays an important role in enhancing trust to the obtained result. In recent
years, a numerous number of uncertainty quantification methods have been
introduced. Monte Carlo dropout (MC-Dropout) is one of the most well-known
techniques to quantify uncertainty in deep learning methods. In this study, we
propose two new loss functions by combining cross entropy with Expected
Calibration Error (ECE) and Predictive Entropy (PE). The obtained results
clearly show that the new proposed loss functions lead to having a calibrated
MC-Dropout method. Our results confirmed the great impact of the new hybrid
loss functions for minimising the overlap between the distributions of
uncertainty estimates for correct and incorrect predictions without sacrificing
the model's overall performance.

    

### [[2110.03266] Lagrangian Neural Network with Differential Symmetries and Relational Inductive Bias](http://arxiv.org/abs/2110.03266)


  Realistic models of physical world rely on differentiable symmetries that, in
turn, correspond to conservation laws. Recent works on Lagrangian and
Hamiltonian neural networks show that the underlying symmetries of a system can
be easily learned by a neural network when provided with an appropriate
inductive bias. However, these models still suffer from issues such as
inability to generalize to arbitrary system sizes, poor interpretability, and
most importantly, inability to learn translational and rotational symmetries,
which lead to the conservation laws of linear and angular momentum,
respectively. Here, we present a momentum conserving Lagrangian neural network
(MCLNN) that learns the Lagrangian of a system, while also preserving the
translational and rotational symmetries. We test our approach on linear and
non-linear spring systems, and a gravitational system, demonstrating the energy
and momentum conservation. We also show that the model developed can generalize
to systems of any arbitrary size. Finally, we discuss the interpretability of
the MCLNN, which directly provides physical insights into the interactions of
multi-particle systems.

    

### [[2110.03267] Propagating State Uncertainty Through Trajectory Forecasting](http://arxiv.org/abs/2110.03267)


  Uncertainty pervades through the modern robotic autonomy stack, with nearly
every component (e.g., sensors, detection, classification, tracking, behavior
prediction) producing continuous or discrete probabilistic distributions.
Trajectory forecasting, in particular, is surrounded by uncertainty as its
inputs are produced by (noisy) upstream perception and its outputs are
predictions that are often probabilistic for use in downstream planning.
However, most trajectory forecasting methods do not account for upstream
uncertainty, instead taking only the most-likely values. As a result,
perceptual uncertainties are not propagated through forecasting and predictions
are frequently overconfident. To address this, we present a novel method for
incorporating perceptual state uncertainty in trajectory forecasting, a key
component of which is a new statistical distance-based loss function which
encourages predicting uncertainties that better match upstream perception. We
evaluate our approach both in illustrative simulations and on large-scale,
real-world data, demonstrating its efficacy in propagating perceptual state
uncertainty through prediction and producing more calibrated predictions.

    

### [[2110.03270] Injecting Planning-Awareness into Prediction and Detection Evaluation](http://arxiv.org/abs/2110.03270)


  Detecting other agents and forecasting their behavior is an integral part of
the modern robotic autonomy stack, especially in safety-critical scenarios
entailing human-robot interaction such as autonomous driving. Due to the
importance of these components, there has been a significant amount of interest
and research in perception and trajectory forecasting, resulting in a wide
variety of approaches. Common to most works, however, is the use of the same
few accuracy-based evaluation metrics, e.g., intersection-over-union,
displacement error, log-likelihood, etc. While these metrics are informative,
they are task-agnostic and outputs that are evaluated as equal can lead to
vastly different outcomes in downstream planning and decision making. In this
work, we take a step back and critically assess current evaluation metrics,
proposing task-aware metrics as a better measure of performance in systems
where they are deployed. Experiments on an illustrative simulation as well as
real-world autonomous driving data validate that our proposed task-aware
metrics are able to account for outcome asymmetry and provide a better estimate
of a model's closed-loop performance.

    

### [[2110.03273] AgFlow: Fast Model Selection of Penalized PCA via Implicit Regularization Effects of Gradient Flow](http://arxiv.org/abs/2110.03273)


  Principal component analysis (PCA) has been widely used as an effective
technique for feature extraction and dimension reduction. In the High Dimension
Low Sample Size (HDLSS) setting, one may prefer modified principal components,
with penalized loadings, and automated penalty selection by implementing model
selection among these different models with varying penalties. The earlier work
[1, 2] has proposed penalized PCA, indicating the feasibility of model
selection in $L_2$- penalized PCA through the solution path of Ridge
regression, however, it is extremely time-consuming because of the intensive
calculation of matrix inverse. In this paper, we propose a fast model selection
method for penalized PCA, named Approximated Gradient Flow (AgFlow), which
lowers the computation complexity through incorporating the implicit
regularization effect introduced by (stochastic) gradient flow [3, 4] and
obtains the complete solution path of $L_2$-penalized PCA under varying
$L_2$-regularization. We perform extensive experiments on real-world datasets.
AgFlow outperforms existing methods (Oja [5], Power [6], and Shamir [7] and the
vanilla Ridge estimators) in terms of computation costs.

    

### [[2110.03281] Detecting Autism Spectrum Disorders with Machine Learning Models Using Speech Transcripts](http://arxiv.org/abs/2110.03281)


  Autism spectrum disorder (ASD) can be defined as a neurodevelopmental
disorder that affects how children interact, communicate and socialize with
others. This disorder can occur in a broad spectrum of symptoms, with varying
effects and severity. While there is no permanent cure for ASD, early detection
and proactive treatment can substantially improve the lives of many children.
Current methods to accurately diagnose ASD are invasive, time-consuming, and
tedious. They can also be subjective perspectives of a number of clinicians
involved, including pediatricians, speech pathologists, psychologists, and
psychiatrists. New technologies are rapidly emerging that include machine
learning models using speech, computer vision from facial, retinal, and brain
MRI images of patients to accurately and timely detect this disorder. Our
research focuses on computational linguistics and machine learning using speech
data from TalkBank, the world's largest spoken language database. We used data
of both ASD and Typical Development (TD) in children from TalkBank to develop
machine learning models to accurately predict ASD. More than 50 features were
used from specifically two datasets in TalkBank to run our experiments using
five different classifiers. Logistic Regression and Random Forest models were
found to be the most effective for each of these two main datasets, with an
accuracy of 0.75. These experiments confirm that while significant
opportunities exist for improving the accuracy, machine learning models can
reliably predict ASD status in children for effective diagnosis.

    

### [[2110.03292] Robotic Lever Manipulation using Hindsight Experience Replay and Shapley Additive Explanations](http://arxiv.org/abs/2110.03292)


  This paper deals with robotic lever control using Explainable Deep
Reinforcement Learning. First, we train a policy by using the Deep
Deterministic Policy Gradient algorithm and the Hindsight Experience Replay
technique, where the goal is to control a robotic manipulator to manipulate a
lever. This enables us both to use continuous states and actions and to learn
with sparse rewards. Being able to learn from sparse rewards is especially
desirable for Deep Reinforcement Learning because designing a reward function
for complex tasks such as this is challenging. We first train in the PyBullet
simulator, which accelerates the training procedure, but is not accurate on
this task compared to the real-world environment. After completing the training
in PyBullet, we further train in the Gazebo simulator, which runs more slowly
than PyBullet, but is more accurate on this task. We then transfer the policy
to the real-world environment, where it achieves comparable performance to the
simulated environments for most episodes. To explain the decisions of the
policy we use the SHAP method to create an explanation model based on the
episodes done in the real-world environment. This gives us some results that
agree with intuition, and some that do not. We also question whether the
independence assumption made when approximating the SHAP values influences the
accuracy of these values for a system such as this, where there are some
correlations between the states.

    

### [[2110.03294] EF21 with Bells & Whistles: Practical Algorithmic Extensions of Modern Error Feedback](http://arxiv.org/abs/2110.03294)


  First proposed by Seide (2014) as a heuristic, error feedback (EF) is a very
popular mechanism for enforcing convergence of distributed gradient-based
optimization methods enhanced with communication compression strategies based
on the application of contractive compression operators. However, existing
theory of EF relies on very strong assumptions (e.g., bounded gradients), and
provides pessimistic convergence rates (e.g., while the best known rate for EF
in the smooth nonconvex regime, and when full gradients are compressed, is
$O(1/T^{2/3})$, the rate of gradient descent in the same regime is $O(1/T)$).
Recently, Richtárik et al. (2021) proposed a new error feedback mechanism,
EF21, based on the construction of a Markov compressor induced by a contractive
compressor. EF21 removes the aforementioned theoretical deficiencies of EF and
at the same time works better in practice. In this work we propose six
practical extensions of EF21, all supported by strong convergence theory:
partial participation, stochastic approximation, variance reduction, proximal
setting, momentum and bidirectional compression. Several of these techniques
were never analyzed in conjunction with EF before, and in cases where they were
(e.g., bidirectional compression), our rates are vastly superior.

    

### [[2110.03298] End-to-End Supermask Pruning: Learning to Prune Image Captioning Models](http://arxiv.org/abs/2110.03298)


  With the advancement of deep models, research work on image captioning has
led to a remarkable gain in raw performance over the last decade, along with
increasing model complexity and computational cost. However, surprisingly works
on compression of deep networks for image captioning task has received little
to no attention. For the first time in image captioning research, we provide an
extensive comparison of various unstructured weight pruning methods on three
different popular image captioning architectures, namely Soft-Attention,
Up-Down and Object Relation Transformer. Following this, we propose a novel
end-to-end weight pruning method that performs gradual sparsification based on
weight sensitivity to the training loss. The pruning schemes are then extended
with encoder pruning, where we show that conducting both decoder pruning and
training simultaneously prior to the encoder pruning provides good overall
performance. Empirically, we show that an 80% to 95% sparse network (up to 75%
reduction in model size) can either match or outperform its dense counterpart.
The code and pre-trained models for Up-Down and Object Relation Transformer
that are capable of achieving CIDEr scores >120 on the MS-COCO dataset but with
only 8.7 MB and 14.5 MB in model size (size reduction of 96% and 94%
respectively against dense versions) are publicly available at
this https URL.

    

### [[2110.03299] End-to-end label uncertainty modeling for speech emotion recognition using Bayesian neural networks](http://arxiv.org/abs/2110.03299)


  Emotions are subjective constructs. Recent end-to-end speech emotion
recognition systems are typically agnostic to the subjective nature of
emotions, despite their state-of-the-art performances. In this work, we
introduce an end-to-end Bayesian neural network architecture to capture the
inherent subjectivity in emotions. To the best of our knowledge, this work is
the first to use Bayesian neural networks for speech emotion recognition. At
training, the network learns a distribution of weights to capture the inherent
uncertainty related to subjective emotion annotations. For this, we introduce a
loss term which enables the model to be explicitly trained on a distribution of
emotion annotations, rather than training them exclusively on mean or
gold-standard labels. We evaluate the proposed approach on the AVEC'16 emotion
recognition dataset. Qualitative and quantitative analysis of the results
reveal that the proposed model can aptly capture the distribution of subjective
emotion annotations with a compromise between mean and standard deviation
estimations.

    

### [[2110.03300] Permutation Compressors for Provably Faster Distributed Nonconvex Optimization](http://arxiv.org/abs/2110.03300)


  We study the MARINA method of Gorbunov et al (2021) -- the current
state-of-the-art distributed non-convex optimization method in terms of
theoretical communication complexity. Theoretical superiority of this method
can be largely attributed to two sources: the use of a carefully engineered
biased stochastic gradient estimator, which leads to a reduction in the number
of communication rounds, and the reliance on {\em independent} stochastic
communication compression operators, which leads to a reduction in the number
of transmitted bits within each communication round. In this paper we i) extend
the theory of MARINA to support a much wider class of potentially {\em
correlated} compressors, extending the reach of the method beyond the classical
independent compressors setting, ii) show that a new quantity, for which we
coin the name {\em Hessian variance}, allows us to significantly refine the
original analysis of MARINA without any additional assumptions, and iii)
identify a special class of correlated compressors based on the idea of {\em
random permutations}, for which we coin the term Perm$K$, the use of which
leads to $O(\sqrt{n})$ (resp. $O(1 + d/\sqrt{n})$) improvement in the
theoretical communication complexity of MARINA in the low Hessian variance
regime when $d\geq n$ (resp. $d \leq n$), where $n$ is the number of workers
and $d$ is the number of parameters describing the model we are learning. We
corroborate our theoretical results with carefully engineered synthetic
experiments with minimizing the average of nonconvex quadratics, and on
autoencoder training with the MNIST dataset.

    

### [[2110.03301] EvadeDroid: A Practical Evasion Attack on Machine Learning for Black-box Android Malware Detection](http://arxiv.org/abs/2110.03301)


  Over the last decade, several studies have investigated the weaknesses of
Android malware detectors against adversarial examples by proposing novel
evasion attacks; however, the practicality of most studies in manipulating
real-world malware is arguable. The majority of studies have assumed attackers
know the details of the target classifiers used for malware detection, while in
real life, malicious actors have limited access to the target classifiers. This
paper presents a practical evasion attack, EvadeDroid, to circumvent black-box
Android malware detectors. In addition to generating real-world adversarial
malware, the proposed evasion attack can preserve the functionality of the
original malware samples. EvadeDroid applies a set of functionality-preserving
transformations to morph malware instances into benign ones using an iterative
and incremental manipulation strategy. The proposed manipulation technique is a
novel, query-efficient optimization algorithm with the aim of finding and
injecting optimal sequences of transformations into malware samples. Our
empirical evaluation demonstrates the efficacy of EvadeDroid under hard- and
soft-label attacks. Moreover, EvadeDroid is capable to generate practical
adversarial examples with only a small number of queries, with evasion rate of
81%, 73%, and 75% for DREBIN, Sec-SVM, and MaMaDroid, respectively. Finally, we
show that EvadeDroid is able to preserve its stealthiness against four popular
commercial antivirus, thus demonstrating its feasibility in the real world.

    

### [[2110.03303] Universal Approximation Under Constraints is Possible with Transformers](http://arxiv.org/abs/2110.03303)


  Many practical problems need the output of a machine learning model to
satisfy a set of constraints, $K$. Nevertheless, there is no known guarantee
that classical neural network architectures can exactly encode constraints
while simultaneously achieving universality. We provide a quantitative
constrained universal approximation theorem which guarantees that for any
non-convex compact set $K$ and any continuous function
$f:\mathbb{R}^n\rightarrow K$, there is a probabilistic transformer $\hat{F}$
whose randomized outputs all lie in $K$ and whose expected output uniformly
approximates $f$. Our second main result is a "deep neural version" of Berge's
Maximum Theorem (1963). The result guarantees that given an objective function
$L$, a constraint set $K$, and a family of soft constraint sets, there is a
probabilistic transformer $\hat{F}$ that approximately minimizes $L$ and whose
outputs belong to $K$; moreover, $\hat{F}$ approximately satisfies the soft
constraints. Our results imply the first universal approximation theorem for
classical transformers with exact convex constraint satisfaction. They also
yield that a chart-free universal approximation theorem for Riemannian
manifold-valued functions subject to suitable geodesically convex constraints.

    

### [[2110.03310] Solving the Dirichlet problem for the Monge-Ampère equation using neural networks](http://arxiv.org/abs/2110.03310)


  The Monge-Ampère equation is a fully nonlinear partial differential
equation (PDE) of fundamental importance in analysis, geometry and in the
applied sciences. In this paper we solve the Dirichlet problem associated with
the Monge-Ampère equation using neural networks and we show that an ansatz
using deep input convex neural networks can be used to find the unique convex
solution. As part of our analysis we study the effect of singularities and
noise in the source function, we consider nontrivial domains, and we
investigate how the method performs in higher dimensions. We also compare this
method to an alternative approach in which standard feed-forward networks are
used together with a loss function which penalizes lack of convexity.

    

### [[2110.03313] Distributed Methods with Compressed Communication for Solving Variational Inequalities, with Theoretical Guarantees](http://arxiv.org/abs/2110.03313)


  Variational inequalities in general and saddle point problems in particular
are increasingly relevant in machine learning applications, including
adversarial learning, GANs, transport and robust optimization. With increasing
data and problem sizes necessary to train high performing models across these
and other applications, it is necessary to rely on parallel and distributed
computing. However, in distributed training, communication among the compute
nodes is a key bottleneck during training, and this problem is exacerbated for
high dimensional and over-parameterized models models. Due to these
considerations, it is important to equip existing methods with strategies that
would allow to reduce the volume of transmitted information during training
while obtaining a model of comparable quality. In this paper, we present the
first theoretically grounded distributed methods for solving variational
inequalities and saddle point problems using compressed communication: MASHA1
and MASHA2. Our theory and methods allow for the use of both unbiased (such as
Rand$k$; MASHA1) and contractive (such as Top$k$; MASHA2) compressors. We
empirically validate our conclusions using two experimental setups: a standard
bilinear min-max problem, and large-scale distributed adversarial training of
transformers.

    

### [[2110.03318] On the Latent Holes of VAEs for Text Generation](http://arxiv.org/abs/2110.03318)


  In this paper, we provide the first focused study on the discontinuities
(aka. holes) in the latent space of Variational Auto-Encoders (VAEs), a
phenomenon which has been shown to have a detrimental effect on model capacity.
When investigating latent holes, existing works are exclusively centred around
the encoder network and they merely explore the existence of holes. We tackle
these limitations by proposing a highly efficient Tree-based Decoder-Centric
(TDC) algorithm for latent hole identification, with a focal point on the text
domain. In contrast to past studies, our approach pays attention to the decoder
network, as a decoder has a direct impact on the model's output quality.
Furthermore, we provide, for the first time, in-depth empirical analysis of the
latent hole phenomenon, investigating several important aspects such as how the
holes impact VAE algorithms' performance on text generation, and how the holes
are distributed in the latent space.

    

### [[2110.03319] Towards Federated Learning-Enabled Visible Light Communication in 6G Systems](http://arxiv.org/abs/2110.03319)


  Visible light communication (VLC) technology was introduced as a key enabler
for the next generation of wireless networks, mainly thanks to its simple and
low-cost implementation. However, several challenges prohibit the realization
of the full potentials of VLC, namely, limited modulation bandwidth, ambient
light interference, optical diffuse reflection effects, devices non-linearity,
and random receiver orientation. On the contrary, centralized machine learning
(ML) techniques have demonstrated a significant potential in handling different
challenges relating to wireless communication systems. Specifically, it was
shown that ML algorithms exhibit superior capabilities in handling complicated
network tasks, such as channel equalization, estimation and modeling, resources
allocation, and opportunistic spectrum access control, to name a few.
Nevertheless, concerns pertaining to privacy and communication overhead when
sharing raw data of the involved clients with a server constitute major
bottlenecks in the implementation of centralized ML techniques. This has
motivated the emergence of a new distributed ML paradigm, namely federated
learning (FL), which can reduce the cost associated with transferring raw data,
and preserve privacy by training ML models locally and collaboratively at the
clients' side. Hence, it becomes evident that integrating FL into VLC networks
can provide ubiquitous and reliable implementation of VLC systems. With this
motivation, this is the first in-depth review in the literature on the
application of FL in VLC networks. To that end, besides the different
architectures and related characteristics of FL, we provide a thorough overview
on the main design aspects of FL based VLC systems. Finally, we also highlight
some potential future research directions of FL that are envisioned to
substantially enhance the performance and robustness of VLC systems.

    

### [[2110.03321] Robustness and reliability when training with noisy labels](http://arxiv.org/abs/2110.03321)


  Labelling of data for supervised learning can be costly and time-consuming
and the risk of incorporating label noise in large data sets is imminent. If
training a flexible discriminative model using a strictly proper loss, such
noise will inevitably shift the solution towards the conditional distribution
over noisy labels. Nevertheless, while deep neural networks have proved capable
of fitting random labels, regularisation and the use of robust loss functions
empirically mitigate the effects of label noise. However, such observations
concern robustness in accuracy, which is insufficient if reliable uncertainty
quantification is critical. We demonstrate this by analysing the properties of
the conditional distribution over noisy labels for an input-dependent noise
model. In addition, we evaluate the set of robust loss functions characterised
by an overlap in asymptotic risk minimisers under the clean and noisy data
distributions. We find that strictly proper and robust loss functions both
offer asymptotic robustness in accuracy, but neither guarantee that the
resulting model is calibrated. Moreover, overfitting is an issue in practice.
With these results, we aim to explain inherent robustness of algorithms to
label noise and to give guidance in the development of new noise-robust
algorithms.

    

### [[2110.03327] Improving Confidence Estimation on Out-of-Domain Data for End-to-End Speech Recognition](http://arxiv.org/abs/2110.03327)


  As end-to-end automatic speech recognition (ASR) models reach promising
performance, various downstream tasks rely on good confidence estimators for
these systems. Recent research has shown that model-based confidence estimators
have a significant advantage over using the output softmax probabilities. If
the input data to the speech recogniser is from mismatched acoustic and
linguistic conditions, the ASR performance and the corresponding confidence
estimators may exhibit severe degradation. Since confidence models are often
trained on the same in-domain data as the ASR, generalising to out-of-domain
(OOD) scenarios is challenging. By keeping the ASR model untouched, this paper
proposes two approaches to improve the model-based confidence estimators on OOD
data: using pseudo transcriptions and an additional OOD language model. With an
ASR model trained on LibriSpeech, experiments show that the proposed methods
can significantly improve the confidence metrics on TED-LIUM and Switchboard
datasets while preserving in-domain performance. Furthermore, the improved
confidence estimators are better calibrated on OOD data and can provide a much
more reliable criterion for data selection.

    

### [[2110.03331] CLEVA-Compass: A Continual Learning EValuation Assessment Compass to Promote Research Transparency and Comparability](http://arxiv.org/abs/2110.03331)


  What is the state of the art in continual machine learning? Although a
natural question for predominant static benchmarks, the notion to train systems
in a lifelong manner entails a plethora of additional challenges with respect
to set-up and evaluation. The latter have recently sparked a growing amount of
critiques on prominent algorithm-centric perspectives and evaluation protocols
being too narrow, resulting in several attempts at constructing guidelines in
favor of specific desiderata or arguing against the validity of prevalent
assumptions. In this work, we depart from this mindset and argue that the goal
of a precise formulation of desiderata is an ill-posed one, as diverse
applications may always warrant distinct scenarios. Instead, we introduce the
Continual Learning EValuation Assessment Compass, CLEVA-Compass for short. The
compass provides the visual means to both identify how approaches are
practically reported and how works can simultaneously be contextualized in the
broader literature landscape. In addition to promoting compact specification in
the spirit of recent replication trends, the CLEVA-Compass thus provides an
intuitive chart to understand the priorities of individual systems, where they
resemble each other, and what elements are missing towards a fair comparison.

    

### [[2110.03336] Frame Averaging for Invariant and Equivariant Network Design](http://arxiv.org/abs/2110.03336)


  Many machine learning tasks involve learning functions that are known to be
invariant or equivariant to certain symmetries of the input data. However, it
is often challenging to design neural network architectures that respect these
symmetries while being expressive and computationally efficient. For example,
Euclidean motion invariant/equivariant graph or point cloud neural networks. We
introduce Frame Averaging (FA), a general purpose and systematic framework for
adapting known (backbone) architectures to become invariant or equivariant to
new symmetry types. Our framework builds on the well known group averaging
operator that guarantees invariance or equivariance but is intractable. In
contrast, we observe that for many important classes of symmetries, this
operator can be replaced with an averaging operator over a small subset of the
group elements, called a frame. We show that averaging over a frame guarantees
exact invariance or equivariance while often being much simpler to compute than
averaging over the entire group. Furthermore, we prove that FA-based models
have maximal expressive power in a broad setting and in general preserve the
expressive power of their backbone architectures. Using frame averaging, we
propose a new class of universal Graph Neural Networks (GNNs), universal
Euclidean motion invariant point cloud networks, and Euclidean motion invariant
Message Passing (MP) GNNs. We demonstrate the practical effectiveness of FA on
several applications including point cloud normal estimation, beyond $2$-WL
graph separation, and $n$-body dynamics prediction, achieving state-of-the-art
results in all of these benchmarks.

    

### [[2110.03343] Uncertainty-aware GAN with Adaptive Loss for Robust MRI Image Enhancement](http://arxiv.org/abs/2110.03343)


  Image-to-image translation is an ill-posed problem as unique one-to-one
mapping may not exist between the source and target images. Learning-based
methods proposed in this context often evaluate the performance on test data
that is similar to the training data, which may be impractical. This demands
robust methods that can quantify uncertainty in the prediction for making
informed decisions, especially for critical areas such as medical imaging.
Recent works that employ conditional generative adversarial networks (GANs)
have shown improved performance in learning photo-realistic image-to-image
mappings between the source and the target images. However, these methods do
not focus on (i)~robustness of the models to out-of-distribution (OOD)-noisy
data and (ii)~uncertainty quantification. This paper proposes a GAN-based
framework that (i)~models an adaptive loss function for robustness to OOD-noisy
data that automatically tunes the spatially varying norm for penalizing the
residuals and (ii)~estimates the per-voxel uncertainty in the predictions. We
demonstrate our method on two key applications in medical imaging:
(i)~undersampled magnetic resonance imaging (MRI) reconstruction (ii)~MRI
modality propagation. Our experiments with two different real-world datasets
show that the proposed method (i)~is robust to OOD-noisy test data and provides
improved accuracy and (ii)~quantifies voxel-level uncertainty in the
predictions.

    

### [[2110.03352] Optimized U-Net for Brain Tumor Segmentation](http://arxiv.org/abs/2110.03352)


  We propose an optimized U-Net architecture for a brain \mbox{tumor}
segmentation task in the BraTS21 Challenge. To find the \mbox{optimal} model
architecture and learning schedule we ran an extensive ablation study to test:
deep supervision loss, Focal loss, decoder attention, drop block, and residual
connections. Additionally, we have searched for the optimal depth of the U-Net
and number of convolutional channels. Our solution was the winner of the
challenge validation phase, with the normalized statistical ranking score of
0.267 and mean Dice score of 0.8855

    

### [[2110.03353] Noisy Text Data: Achilles' Heel of popular transformer based NLP models](http://arxiv.org/abs/2110.03353)


  In the last few years, the ML community has created a number of new NLP
models based on transformer architecture. These models have shown great
performance for various NLP tasks on benchmark datasets, often surpassing SOTA
results. Buoyed with this success, one often finds industry practitioners
actively experimenting with fine-tuning these models to build NLP applications
for industry use cases. However, for most datasets that are used by
practitioners to build industrial NLP applications, it is hard to guarantee the
presence of any noise in the data. While most transformer based NLP models have
performed exceedingly well in transferring the learnings from one dataset to
another, it remains unclear how these models perform when fine-tuned on noisy
text. We address the open question by Kumar et al. (2020) to explore the
sensitivity of popular transformer based NLP models to noise in the text data.
We continue working with the noise as defined by them -- spelling mistakes &
typos (which are the most commonly occurring noise). We show (via experimental
results) that these models perform badly on most common NLP tasks namely text
classification, textual similarity, NER, question answering, text summarization
on benchmark datasets. We further show that as the noise in data increases, the
performance degrades. Our findings suggest that one must be vary of the
presence of noise in their datasets while fine-tuning popular transformer based
NLP models.

    

### [[2110.03354] $\bar{G}_{mst}$:An Unbiased Stratified Statistic and a Fast Gradient Optimization Algorithm Based on It](http://arxiv.org/abs/2110.03354)


  -The fluctuation effect of gradient expectation and variance caused by
parameter update between consecutive iterations is neglected or confusing by
current mainstream gradient optimization algorithms. The work in this paper
remedy this issue by introducing a novel unbiased stratified statistic \
$\bar{G}_{mst}$\ , a sufficient condition of fast convergence for \
$\bar{G}_{mst}$\ also is established. A novel algorithm named MSSG designed
based on \ $\bar{G}_{mst}$\ outperforms other sgd-like algorithms. Theoretical
conclusions and experimental evidence strongly suggest to employ MSSG when
training deep model.

    

### [[2110.03358] Uncertainty Set Prediction of Aggregated Wind Power Generation based on Bayesian LSTM and Spatio-Temporal Analysis](http://arxiv.org/abs/2110.03358)


  Aggregated stochastic characteristics of geographically distributed wind
generation will provide valuable information for secured and economical system
operation in electricity markets. This paper focuses on the uncertainty set
prediction of the aggregated generation of geographically distributed wind
farms. A Spatio-temporal model is proposed to learn the dynamic features from
partial observation in near-surface wind fields of neighboring wind farms. We
use Bayesian LSTM, a probabilistic prediction model, to obtain the uncertainty
set of the generation in individual wind farms. Then, spatial correlation
between different wind farms is presented to correct the output results.
Numerical testing results based on the actual data with 6 wind farms in
northwest China show that the uncertainty set of aggregated wind generation of
distributed wind farms is less volatile than that of a single wind farm.

    

### [[2110.03360] Sparse MoEs meet Efficient Ensembles](http://arxiv.org/abs/2110.03360)


  Machine learning models based on the aggregated outputs of submodels, either
at the activation or prediction levels, lead to strong performance. We study
the interplay of two popular classes of such models: ensembles of neural
networks and sparse mixture of experts (sparse MoEs). First, we show that these
two approaches have complementary features whose combination is beneficial.
Then, we present partitioned batch ensembles, an efficient ensemble of sparse
MoEs that takes the best of both classes of models. Extensive experiments on
fine-tuned vision transformers demonstrate the accuracy, log-likelihood,
few-shot learning, robustness, and uncertainty calibration improvements of our
approach over several challenging baselines. Partitioned batch ensembles not
only scale to models with up to 2.7B parameters, but also provide larger
performance gains for larger models.

    

### [[2110.03363] Evaluating model-based planning and planner amortization for continuous control](http://arxiv.org/abs/2110.03363)


  There is a widespread intuition that model-based control methods should be
able to surpass the data efficiency of model-free approaches. In this paper we
attempt to evaluate this intuition on various challenging locomotion tasks. We
take a hybrid approach, combining model predictive control (MPC) with a learned
model and model-free policy learning; the learned policy serves as a proposal
for MPC. We find that well-tuned model-free agents are strong baselines even
for high DoF control problems but MPC with learned proposals and models
(trained on the fly or transferred from related tasks) can significantly
improve performance and data efficiency in hard multi-task/multi-goal settings.
Finally, we show that it is possible to distil a model-based planner into a
policy that amortizes the planning computation without any loss of performance.
Videos of agents performing different tasks can be seen at
this https URL.

    

### [[2110.03369] The Connection between Out-of-Distribution Generalization and Privacy of ML Models](http://arxiv.org/abs/2110.03369)


  With the goal of generalizing to out-of-distribution (OOD) data, recent
domain generalization methods aim to learn "stable" feature representations
whose effect on the output remains invariant across domains. Given the
theoretical connection between generalization and privacy, we ask whether
better OOD generalization leads to better privacy for machine learning models,
where privacy is measured through robustness to membership inference (MI)
attacks. In general, we find that the relationship does not hold. Through
extensive evaluation on a synthetic dataset and image datasets like MNIST,
Fashion-MNIST, and Chest X-rays, we show that a lower OOD generalization gap
does not imply better robustness to MI attacks. Instead, privacy benefits are
based on the extent to which a model captures the stable features. A model that
captures stable features is more robust to MI attacks than models that exhibit
better OOD generalization but do not learn stable features. Further, for the
same provable differential privacy guarantees, a model that learns stable
features provides higher utility as compared to others. Our results offer the
first extensive empirical study connecting stable features and privacy, and
also have a takeaway for the domain generalization community; MI attack can be
used as a complementary metric to measure model quality.

    

### [[2110.03372] Unifying Likelihood-free Inference with Black-box Sequence Design and Beyond](http://arxiv.org/abs/2110.03372)


  Black-box optimization formulations for biological sequence design have drawn
recent attention due to their promising potential impact on the pharmaceutical
industry. In this work, we propose to unify two seemingly distinct worlds:
likelihood-free inference and black-box sequence design, under one
probabilistic framework. In tandem, we provide a recipe for constructing
various sequence design methods based on this framework. We show how previous
drug discovery approaches can be "reinvented" in our framework, and further
propose new probabilistic sequence design algorithms. Extensive experiments
illustrate the benefits of the proposed methodology.

    

### [[2110.03375] Learning Pessimism for Robust and Efficient Off-Policy Reinforcement Learning](http://arxiv.org/abs/2110.03375)


  Popular off-policy deep reinforcement learning algorithms compensate for
overestimation bias during temporal-difference learning by utilizing
pessimistic estimates of the expected target returns. In this work, we propose
a novel learnable penalty to enact such pessimism, based on a new way to
quantify the critic's epistemic uncertainty. Furthermore, we propose to learn
the penalty alongside the critic with dual TD-learning, a strategy to estimate
and minimize the bias magnitude in the target returns. Our method enables us to
accurately counteract overestimation bias throughout training without incurring
the downsides of overly pessimistic targets. Empirically, by integrating our
method and other orthogonal improvements with popular off-policy algorithms, we
achieve state-of-the-art results in continuous control tasks from both
proprioceptive and pixel observations.

    

### [[2110.03384] Deep Learning Model Explainability for Inspection Accuracy Improvement in the Automotive Industry](http://arxiv.org/abs/2110.03384)


  The welding seams visual inspection is still manually operated by humans in
different companies, so the result of the test is still highly subjective and
expensive. At present, the integration of deep learning methods for welds
classification is a research focus in engineering applications. This work
intends to apprehend and emphasize the contribution of deep learning model
explainability to the improvement of welding seams classification accuracy and
reliability, two of the various metrics affecting the production lines and cost
in the automotive industry. For this purpose, we implement a novel hybrid
method that relies on combining the model prediction scores and visual
explanation heatmap of the model in order to make a more accurate
classification of welding seam defects and improve both its performance and its
reliability. The results show that the hybrid model performance is relatively
above our target performance and helps to increase the accuracy by at least
18%, which presents new perspectives to the developments of deep Learning
explainability and interpretability.

    

### [[2110.03393] Multivariate Anomaly Detection based on Prediction Intervals Constructed using Deep Learning](http://arxiv.org/abs/2110.03393)


  It has been shown that deep learning models can under certain circumstances
outperform traditional statistical methods at forecasting. Furthermore, various
techniques have been developed for quantifying the forecast uncertainty
(prediction intervals). In this paper, we utilize prediction intervals
constructed with the aid of artificial neural networks to detect anomalies in
the multivariate setting. Challenges with existing deep learning-based anomaly
detection approaches include $(i)$ large sets of parameters that may be
computationally intensive to tune, $(ii)$ returning too many false positives
rendering the techniques impractical for use, $(iii)$ requiring labeled
datasets for training which are often not prevalent in real life. Our approach
overcomes these challenges. We benchmark our approach against the oft-preferred
well-established statistical models. We focus on three deep learning
architectures, namely, cascaded neural networks, reservoir computing and long
short-term memory recurrent neural networks. Our finding is deep learning
outperforms (or at the very least is competitive to) the latter.

    

### [[2110.03403] Disentangling deep neural networks with rectified linear units using duality](http://arxiv.org/abs/2110.03403)


  Despite their success deep neural networks (DNNs) are still largely
considered as black boxes. The main issue is that the linear and non-linear
operations are entangled in every layer, making it hard to interpret the hidden
layer outputs. In this paper, we look at DNNs with rectified linear units
(ReLUs), and focus on the gating property (`on/off' states) of the ReLUs. We
extend the recently developed dual view in which the computation is broken
path-wise to show that learning in the gates is more crucial, and learning the
weights given the gates is characterised analytically via the so called neural
path kernel (NPK) which depends on inputs and gates. In this paper, we present
novel results to show that convolution with global pooling and skip connection
provide respectively rotational invariance and ensemble structure to the NPK.
To address `black box'-ness, we propose a novel interpretable counterpart of
DNNs with ReLUs namely deep linearly gated networks (DLGN): the pre-activations
to the gates are generated by a deep linear network, and the gates are then
applied as external masks to learn the weights in a different network. The DLGN
is not an alternative architecture per se, but a disentanglement and an
interpretable re-arrangement of the computations in a DNN with ReLUs. The DLGN
disentangles the computations into two `mathematically' interpretable
linearities (i) the `primal' linearity between the input and the
pre-activations in the gating network and (ii) the `dual' linearity in the path
space in the weights network characterised by the NPK. We compare the
performance of DNN, DGN and DLGN on CIFAR-10 and CIFAR-100 to show that, the
DLGN recovers more than $83.5\%$ of the performance of state-of-the-art DNNs.
This brings us to an interesting question: `Is DLGN a universal spectral
approximator?'

    

### [[2110.03405] Joint calibration and mapping of satellite altimetry data using trainable variational models](http://arxiv.org/abs/2110.03405)


  Satellite radar altimeters are a key source of observation of ocean surface
dynamics. However, current sensor technology and mapping techniques do not yet
allow to systematically resolve scales smaller than 100km. With their new
sensors, upcoming wide-swath altimeter missions such as SWOT should help
resolve finer scales. Current mapping techniques rely on the quality of the
input data, which is why the raw data go through multiple preprocessing stages
before being used. Those calibration stages are improved and refined over many
years and represent a challenge when a new type of sensor start acquiring data.
Here we show how a data-driven variational data assimilation framework could be
used to jointly learn a calibration operator and an interpolator from
non-calibrated data . The proposed framework significantly outperforms the
operational state-of-the-art mapping pipeline and truly benefits from
wide-swath data to resolve finer scales on the global map as well as in the
SWOT sensor geometry.

    

### [[2110.03413] Curved Markov Chain Monte Carlo for Network Learning](http://arxiv.org/abs/2110.03413)


  We present a geometrically enhanced Markov chain Monte Carlo sampler for
networks based on a discrete curvature measure defined on graphs. Specifically,
we incorporate the concept of graph Forman curvature into sampling procedures
on both the nodes and edges of a network explicitly, via the transition
probability of the Markov chain, as well as implicitly, via the target
stationary distribution, which gives a novel, curved Markov chain Monte Carlo
approach to learning networks. We show that integrating curvature into the
sampler results in faster convergence to a wide range of network statistics
demonstrated on deterministic networks drawn from real-world data.

    

### [[2110.03414] SERAB: A multi-lingual benchmark for speech emotion recognition](http://arxiv.org/abs/2110.03414)


  Recent developments in speech emotion recognition (SER) often leverage deep
neural networks (DNNs). Comparing and benchmarking different DNN models can
often be tedious due to the use of different datasets and evaluation protocols.
To facilitate the process, here, we present the Speech Emotion Recognition
Adaptation Benchmark (SERAB), a framework for evaluating the performance and
generalization capacity of different approaches for utterance-level SER. The
benchmark is composed of nine datasets for SER in six languages. Since the
datasets have different sizes and numbers of emotional classes, the proposed
setup is particularly suitable for estimating the generalization capacity of
pre-trained DNN-based feature extractors. We used the proposed framework to
evaluate a selection of standard hand-crafted feature sets and state-of-the-art
DNN representations. The results highlight that using only a subset of the data
included in SERAB can result in biased evaluation, while compliance with the
proposed protocol can circumvent this issue.

    

### [[2110.03422] Modeling Effect of Lockdowns and Other Effects on India Covid-19 Infections Using SEIR Model and Machine Learning](http://arxiv.org/abs/2110.03422)


  The SEIR model is a widely used epidemiological model used to predict the
rise in infections. This model has been widely used in different countries to
predict the number of Covid-19 cases. But the original SEIR model does not take
into account the effect of factors such as lockdowns, vaccines, and
re-infections. In India the first wave of Covid started in March 2020 and the
second wave in April 2021. In this paper, we modify the SEIR model equations to
model the effect of lockdowns and other influencers, and fit the model on data
of the daily Covid-19 infections in India using lmfit, a python library for
least squares minimization for curve fitting. We modify R0 parameter in the
standard SEIR model as a rectangle in order to account for the effect of
lockdowns. Our modified SEIR model accurately fits the available data of
infections.

    

### [[2110.03423] Efficient GPU implementation of randomized SVD and its applications](http://arxiv.org/abs/2110.03423)


  Matrix decompositions are ubiquitous in machine learning, including
applications in dimensionality reduction, data compression and deep learning
algorithms. Typical solutions for matrix decompositions have polynomial
complexity which significantly increases their computational cost and time. In
this work, we leverage efficient processing operations that can be run in
parallel on modern Graphical Processing Units (GPUs), predominant computing
architecture used e.g. in deep learning, to reduce the computational burden of
computing matrix decompositions. More specifically, we reformulate the
randomized decomposition problem to incorporate fast matrix multiplication
operations (BLAS-3) as building blocks. We show that this formulation, combined
with fast random number generators, allows to fully exploit the potential of
parallel processing implemented in GPUs. Our extensive evaluation confirms the
superiority of this approach over the competing methods and we release the
results of this research as a part of the official CUDA implementation
(this https URL).

    

### [[2110.03424] Bad-Policy Density: A Measure of Reinforcement Learning Hardness](http://arxiv.org/abs/2110.03424)


  Reinforcement learning is hard in general. Yet, in many specific
environments, learning is easy. What makes learning easy in one environment,
but difficult in another? We address this question by proposing a simple
measure of reinforcement-learning hardness called the bad-policy density. This
quantity measures the fraction of the deterministic stationary policy space
that is below a desired threshold in value. We prove that this simple quantity
has many properties one would expect of a measure of learning hardness.
Further, we prove it is NP-hard to compute the measure in general, but there
are paths to polynomial-time approximation. We conclude by summarizing
potential directions and uses for this measure.

    

### [[2110.03426] Fast learning from label proportions with small bags](http://arxiv.org/abs/2110.03426)


  In learning from label proportions (LLP), the instances are grouped into
bags, compared with supervised learning and the task is to learn an instance
classifier given relative class proportions in training bags. LLP is useful
when obtaining individual instance labels is impossible or costly.
In this work, we focus on the case of small bags, which allows designing more
efficient algorithms by explicitly considering all consistent label
combinations. In particular, we propose an EM algorithm alternating between
optimizing a general neural network instance classifier and incorporating
bag-level annotations. In comparison to existing deep LLP methods, our approach
converges faster to a comparable or better solution. Several experiments were
performed on two different datasets.

    

### [[2110.03427] Is Attention always needed? A Case Study on Language Identification from Speech](http://arxiv.org/abs/2110.03427)


  Language Identification (LID), a recommended initial step to Automatic Speech
Recognition (ASR), is used to detect a spoken language from audio specimens. In
state-of-the-art systems capable of multilingual speech processing, however,
users have to explicitly set one or more languages before using them. LID,
therefore, plays a very important role in situations where ASR based systems
cannot parse the uttered language in multilingual contexts causing failure in
speech recognition. We propose an attention based convolutional recurrent
neural network (CRNN with Attention) that works on Mel-frequency Cepstral
Coefficient (MFCC) features of audio specimens. Additionally, we reproduce some
state-of-the-art approaches, namely Convolutional Neural Network (CNN) and
Convolutional Recurrent Neural Network (CRNN), and compare them to our proposed
method. We performed extensive evaluation on thirteen different Indian
languages and our model achieves classification accuracy over 98%. Our LID
model is robust to noise and provides 91.2% accuracy in a noisy scenario. The
proposed model is easily extensible to new languages.

    

### [[2110.03431] Cloud Failure Prediction with Hierarchical Temporary Memory: An Empirical Assessment](http://arxiv.org/abs/2110.03431)


  Hierarchical Temporary Memory (HTM) is an unsupervised learning algorithm
inspired by the features of the neocortex that can be used to continuously
process stream data and detect anomalies, without requiring a large amount of
data for training nor requiring labeled data. HTM is also able to continuously
learn from samples, providing a model that is always up-to-date with respect to
observations. These characteristics make HTM particularly suitable for
supporting online failure prediction in cloud systems, which are systems with a
dynamically changing behavior that must be monitored to anticipate problems.
This paper presents the first systematic study that assesses HTM in the context
of failure prediction. The results that we obtained considering 72
configurations of HTM applied to 12 different types of faults introduced in the
Clearwater cloud system show that HTM can help to predict failures with
sufficient effectiveness (F-measure = 0.76), representing an interesting
practical alternative to (semi-)supervised algorithms.

    

### [[2110.03440] Towards Robust and Transferable IIoT Sensor based Anomaly Classification using Artificial Intelligence](http://arxiv.org/abs/2110.03440)


  The increasing deployment of low-cost industrial IoT (IIoT) sensor platforms
on industrial assets enables great opportunities for anomaly classification in
industrial plants. The performance of such a classification model depends
highly on the available training data. Models perform well when the training
data comes from the same machine. However, as soon as the machine is changed,
repaired, or put into operation in a different environment, the prediction
often fails. For this reason, we investigate whether it is feasible to have a
robust and transferable method for AI based anomaly classification using
different models and pre-processing steps on centrifugal pumps which are
dismantled and put back into operation in the same as well as in different
environments. Further, we investigate the model performance on different pumps
from the same type compared to those from the training data.

    

### [[2110.03442] A Comparison of Neural Network Architectures for Data-Driven Reduced-Order Modeling](http://arxiv.org/abs/2110.03442)


  The popularity of deep convolutional autoencoders (CAEs) has engendered
effective reduced-order models (ROMs) for the simulation of large-scale
dynamical systems. However, it is not known whether deep CAEs provide superior
performance in all ROM scenarios. To elucidate this, the effect of autoencoder
architecture on its associated ROM is studied through the comparison of deep
CAEs against two alternatives: a simple fully connected autoencoder, and a
novel graph convolutional autoencoder. Through benchmark experiments, it is
shown that the superior autoencoder architecture for a given ROM application is
highly dependent on the size of the latent space and the structure of the
snapshot data, with the proposed architecture demonstrating benefits on data
with irregular connectivity when the latent space is sufficiently large.

    

### [[2110.03443] Unpacking the Black Box: Regulating Algorithmic Decisions](http://arxiv.org/abs/2110.03443)


  We characterize optimal oversight of algorithms in a world where an agent
designs a complex prediction function but a principal is limited in the amount
of information she can learn about the prediction function. We show that
limiting agents to prediction functions that are simple enough to be fully
transparent is inefficient as long as the bias induced by misalignment between
principal's and agent's preferences is small relative to the uncertainty about
the true state of the world. Algorithmic audits can improve welfare, but the
gains depend on the design of the audit tools. Tools that focus on minimizing
overall information loss, the focus of many post-hoc explainer tools, will
generally be inefficient since they focus on explaining the average behavior of
the prediction function rather than sources of mis-prediction, which matter for
welfare-relevant outcomes. Targeted tools that focus on the source of incentive
misalignment, e.g., excess false positives or racial disparities, can provide
first-best solutions. We provide empirical support for our theoretical findings
using an application in consumer lending.

    

### [[2110.03446] A Hierarchical Variational Neural Uncertainty Model for Stochastic Video Prediction](http://arxiv.org/abs/2110.03446)


  Predicting the future frames of a video is a challenging task, in part due to
the underlying stochastic real-world phenomena. Prior approaches to solve this
task typically estimate a latent prior characterizing this stochasticity,
however do not account for the predictive uncertainty of the (deep learning)
model. Such approaches often derive the training signal from the mean-squared
error (MSE) between the generated frame and the ground truth, which can lead to
sub-optimal training, especially when the predictive uncertainty is high.
Towards this end, we introduce Neural Uncertainty Quantifier (NUQ) - a
stochastic quantification of the model's predictive uncertainty, and use it to
weigh the MSE loss. We propose a hierarchical, variational framework to derive
NUQ in a principled manner using a deep, Bayesian graphical model. Our
experiments on four benchmark stochastic video prediction datasets show that
our proposed framework trains more effectively compared to the state-of-the-art
models (especially when the training sets are small), while demonstrating
better video generation quality and diversity against several evaluation
metrics.

    

### [[2110.03448] Multi-Head ReLU Implicit Neural Representation Networks](http://arxiv.org/abs/2110.03448)


  In this paper, a novel multi-head multi-layer perceptron (MLP) structure is
presented for implicit neural representation (INR). Since conventional
rectified linear unit (ReLU) networks are shown to exhibit spectral bias
towards learning low-frequency features of the signal, we aim at mitigating
this defect by taking advantage of the local structure of the signals. To be
more specific, an MLP is used to capture the global features of the underlying
generator function of the desired signal. Then, several heads are utilized to
reconstruct disjoint local features of the signal, and to reduce the
computational complexity, sparse layers are deployed for attaching heads to the
body. Through various experiments, we show that the proposed model does not
suffer from the special bias of conventional ReLU networks and has superior
generalization capabilities. Finally, simulation results confirm that the
proposed multi-head structure outperforms existing INR methods with
considerably less computational cost.

    

### [[2110.03450] Efficient and Private Federated Learning with Partially Trainable Networks](http://arxiv.org/abs/2110.03450)


  Federated learning is used for decentralized training of machine learning
models on a large number (millions) of edge mobile devices. It is challenging
because mobile devices often have limited communication bandwidth and local
computation resources. Therefore, improving the efficiency of federated
learning is critical for scalability and usability. In this paper, we propose
to leverage partially trainable neural networks, which freeze a portion of the
model parameters during the entire training process, to reduce the
communication cost with little implications on model performance. Through
extensive experiments, we empirically show that Federated learning of Partially
Trainable neural networks (FedPT) can result in superior communication-accuracy
trade-offs, with up to $46\times$ reduction in communication cost, at a small
accuracy cost. Our approach also enables faster training, with a smaller memory
footprint, and better utility for strong differential privacy guarantees. The
proposed FedPT method can be particularly interesting for pushing the
limitations of overparameterization in on-device learning.

    

### [[2110.03452] Inter-Domain Alignment for Predicting High-Resolution Brain Networks Using Teacher-Student Learning](http://arxiv.org/abs/2110.03452)


  Accurate and automated super-resolution image synthesis is highly desired
since it has the great potential to circumvent the need for acquiring high-cost
medical scans and a time-consuming preprocessing pipeline of neuroimaging data.
However, existing deep learning frameworks are solely designed to predict
high-resolution (HR) image from a low-resolution (LR) one, which limits their
generalization ability to brain graphs (i.e., connectomes). A small body of
works has focused on superresolving brain graphs where the goal is to predict a
HR graph from a single LR graph. Although promising, existing works mainly
focus on superresolving graphs belonging to the same domain (e.g., functional),
overlooking the domain fracture existing between multimodal brain data
distributions (e.g., morphological and structural). To this aim, we propose a
novel inter-domain adaptation framework namely, Learn to SuperResolve Brain
Graphs with Knowledge Distillation Network (L2S-KDnet), which adopts a
teacher-student paradigm to superresolve brain graphs. Our teacher network is a
graph encoder-decoder that firstly learns the LR brain graph embeddings, and
secondly learns how to align the resulting latent representations to the HR
ground truth data distribution using an adversarial regularization. Ultimately,
it decodes the HR graphs from the aligned embeddings. Next, our student network
learns the knowledge of the aligned brain graphs as well as the topological
structure of the predicted HR graphs transferred from the teacher. We further
leverage the decoder of the teacher to optimize the student network. L2S-KDnet
presents the first TS architecture tailored for brain graph super-resolution
synthesis that is based on inter-domain alignment. Our experimental results
demonstrate substantial performance gains over benchmark methods.

    

### [[2110.03453] Recurrent Multigraph Integrator Network for Predicting the Evolution of Population-Driven Brain Connectivity Templates](http://arxiv.org/abs/2110.03453)


  Learning how to estimate a connectional brain template(CBT) from a population
of brain multigraphs, where each graph (e.g., functional) quantifies a
particular relationship between pairs of brain regions of interest (ROIs),
allows to pin down the unique connectivity patterns shared across individuals.
Specifically, a CBT is viewed as an integral representation of a set of highly
heterogeneous graphs and ideally meeting the centeredness (i.e., minimum
distance to all graphs in the population) and discriminativeness (i.e.,
distinguishes the healthy from the disordered population) criteria. So far,
existing works have been limited to only integrating and fusing a population of
brain multigraphs acquired at a single timepoint. In this paper, we
unprecedentedly tackle the question: Given a baseline multigraph population,
can we learn how to integrate and forecast its CBT representations at follow-up
timepoints? Addressing such question is of paramount in predicting common
alternations across healthy and disordered populations. To fill this gap, we
propose Recurrent Multigraph Integrator Network (ReMI-Net), the first graph
recurrent neural network which infers the baseline CBT of an input population
t1 and predicts its longitudinal evolution over time (ti > t1). Our ReMI-Net is
composed of recurrent neural blocks with graph convolutional layers using a
cross-node message passing to first learn hidden-states embeddings of each CBT
node (i.e., brain region of interest) and then predict its evolution at the
consecutive timepoint. Moreover, we design a novel time-dependent loss to
regularize the CBT evolution trajectory over time and further introduce a
cyclic recursion and learnable normalization layer to generate well-centered
CBTs from time-dependent hidden-state embeddings. Finally, we derive the CBT
adjacency matrix from the learned hidden state graph representation.

    

### [[2110.03464] Differential Anomaly Detection for Facial Images](http://arxiv.org/abs/2110.03464)


  Due to their convenience and high accuracy, face recognition systems are
widely employed in governmental and personal security applications to
automatically recognise individuals. Despite recent advances, face recognition
systems have shown to be particularly vulnerable to identity attacks (i.e.,
digital manipulations and attack presentations). Identity attacks pose a big
security threat as they can be used to gain unauthorised access and spread
misinformation. In this context, most algorithms for detecting identity attacks
generalise poorly to attack types that are unknown at training time. To tackle
this problem, we introduce a differential anomaly detection framework in which
deep face embeddings are first extracted from pairs of images (i.e., reference
and probe) and then combined for identity attack detection. The experimental
evaluation conducted over several databases shows a high generalisation
capability of the proposed method for detecting unknown attacks in both the
digital and physical domains.

    

### [[2110.03469] Federated Learning from Small Datasets](http://arxiv.org/abs/2110.03469)


  Federated learning allows multiple parties to collaboratively train a joint
model without sharing local data. This enables applications of machine learning
in settings of inherently distributed, undisclosable data such as in the
medical domain. In practice, joint training is usually achieved by aggregating
local models, for which local training objectives have to be in expectation
similar to the joint (global) objective. Often, however, local datasets are so
small that local objectives differ greatly from the global objective, resulting
in federated learning to fail. We propose a novel approach that intertwines
model aggregations with permutations of local models. The permutations expose
each local model to a daisy chain of local datasets resulting in more efficient
training in data-sparse domains. This enables training on extremely small local
datasets, such as patient data across hospitals, while retaining the training
efficiency and privacy benefits of federated learning.

    

### [[2110.03473] Unsupervised Image Decomposition with Phase-Correlation Networks](http://arxiv.org/abs/2110.03473)


  The ability to decompose scenes into their object components is a desired
property for autonomous agents, allowing them to reason and act in their
surroundings. Recently, different methods have been proposed to learn
object-centric representations from data in an unsupervised manner. These
methods often rely on latent representations learned by deep neural networks,
hence requiring high computational costs and large amounts of curated data.
Such models are also difficult to interpret. To address these challenges, we
propose the Phase-Correlation Decomposition Network (PCDNet), a novel model
that decomposes a scene into its object components, which are represented as
transformed versions of a set of learned object prototypes. The core building
block in PCDNet is the Phase-Correlation Cell (PC Cell), which exploits the
frequency-domain representation of the images in order to estimate the
transformation between an object prototype and its transformed version in the
image. In our experiments, we show how PCDNet outperforms state-of-the-art
methods for unsupervised object discovery and segmentation on simple benchmark
datasets and on more challenging data, while using a small number of learnable
parameters and being fully interpretable.

    

### [[2110.03478] Complex-valued deep learning with differential privacy](http://arxiv.org/abs/2110.03478)


  We present $\zeta$-DP, an extension of differential privacy (DP) to
complex-valued functions. After introducing the complex Gaussian mechanism,
whose properties we characterise in terms of $(\varepsilon, \delta)$-DP and
Rényi-DP, we present $\zeta$-DP stochastic gradient descent ($\zeta$-DP-SGD),
a variant of DP-SGD for training complex-valued neural networks. We
experimentally evaluate $\zeta$-DP-SGD on three complex-valued tasks, i.e.
electrocardiogram classification, speech classification and magnetic resonance
imaging (MRI) reconstruction. Moreover, we provide $\zeta$-DP-SGD benchmarks
for a large variety of complex-valued activation functions and on a
complex-valued variant of the MNIST dataset. Our experiments demonstrate that
DP training of complex-valued neural networks is possible with rigorous privacy
guarantees and excellent utility.

    

### [[2110.03484] Creating Training Sets via Weak Indirect Supervision](http://arxiv.org/abs/2110.03484)


  Creating labeled training sets has become one of the major roadblocks in
machine learning. To address this, recent Weak Supervision (WS) frameworks
synthesize training labels from multiple potentially noisy supervision sources.
However, existing frameworks are restricted to supervision sources that share
the same output space as the target task. To extend the scope of usable
sources, we formulate Weak Indirect Supervision (WIS), a new research problem
for automatically synthesizing training labels based on indirect supervision
sources that have different output label spaces. To overcome the challenge of
mismatched output spaces, we develop a probabilistic modeling approach, PLRM,
which uses user-provided label relations to model and leverage indirect
supervision sources. Moreover, we provide a theoretically-principled test of
the distinguishability of PLRM for unseen labels, along with an generalization
bound. On both image and text classification tasks as well as an industrial
advertising application, we demonstrate the advantages of PLRM by outperforming
baselines by a margin of 2%-9%.

    

### [[2110.03498] On the relationship between disentanglement and multi-task learning](http://arxiv.org/abs/2110.03498)


  One of the main arguments behind studying disentangled representations is the
assumption that they can be easily reused in different tasks. At the same time
finding a joint, adaptable representation of data is one of the key challenges
in the multi-task learning setting. In this paper, we take a closer look at the
relationship between disentanglement and multi-task learning based on hard
parameter sharing. We perform a thorough empirical study of the representations
obtained by neural networks trained on automatically generated supervised
tasks. Using a set of standard metrics we show that disentanglement appears
naturally during the process of multi-task neural network training.

    

### [[2110.03501] Pretrained Language Models are Symbolic Mathematics Solvers too!](http://arxiv.org/abs/2110.03501)


  Solving symbolic mathematics has always been of in the arena of human
ingenuity that needs compositional reasoning and recurrence. However, recent
studies have shown that large-scale language models such as transformers are
universal and surprisingly can be trained as a sequence-to-sequence task to
solve complex mathematical equations. These large transformer models need
humongous amounts of training data to generalize to unseen symbolic mathematics
problems. In this paper, we present a sample efficient way of solving the
symbolic tasks by first pretraining the transformer model with language
translation and then fine-tuning the pretrained transformer model to solve the
downstream task of symbolic mathematics. We achieve comparable accuracy on the
integration task with our pretrained model while using around $1.5$ orders of
magnitude less number of training samples with respect to the state-of-the-art
deep learning for symbolic mathematics. The test accuracy on differential
equation tasks is considerably lower comparing with integration as they need
higher order recursions that are not present in language translations. We
pretrain our model with different pairs of language translations. Our results
show language bias in solving symbolic mathematics tasks. Finally, we study the
robustness of the fine-tuned model on symbolic math tasks against distribution
shift, and our approach generalizes better in distribution shift scenarios for
the function integration.

    

### [[2110.03511] Peer Collaborative Learning for Polyphonic Sound Event Detection](http://arxiv.org/abs/2110.03511)


  This paper describes that semi-supervised learning called peer collaborative
learning (PCL) can be applied to the polyphonic sound event detection (PSED)
task, which is one of the tasks in the Detection and Classification of Acoustic
Scenes and Events (DCASE) challenge. Many deep learning models have been
studied to find out what kind of sound events occur where and for how long in a
given audio clip. The characteristic of PCL used in this paper is the
combination of ensemble-based knowledge distillation into sub-networks and
student-teacher model-based knowledge distillation, which can train a robust
PSED model from a small amount of strongly labeled data, weakly labeled data,
and a large amount of unlabeled data. We evaluated the proposed PCL model using
the DCASE 2019 Task 4 datasets and achieved an F1-score improvement of about
10% compared to the baseline model.

    

### [[2110.03513] Accelerated Componentwise Gradient Boosting using Efficient Data Representation and Momentum-based Optimization](http://arxiv.org/abs/2110.03513)


  Componentwise boosting (CWB), also known as model-based boosting, is a
variant of gradient boosting that builds on additive models as base learners to
ensure interpretability. CWB is thus often used in research areas where models
are employed as tools to explain relationships in data. One downside of CWB is
its computational complexity in terms of memory and runtime. In this paper, we
propose two techniques to overcome these issues without losing the properties
of CWB: feature discretization of numerical features and incorporating Nesterov
momentum into functional gradient descent. As the latter can be prone to early
overfitting, we also propose a hybrid approach that prevents a possibly
diverging gradient descent routine while ensuring faster convergence. We
perform extensive benchmarks on multiple simulated and real-world data sets to
demonstrate the improvements in runtime and memory consumption while
maintaining state-of-the-art estimation and prediction performance.

    

### [[2110.03515] Use of Deterministic Transforms to Design Weight Matrices of a Neural Network](http://arxiv.org/abs/2110.03515)


  Self size-estimating feedforward network (SSFN) is a feedforward multilayer
network. For the existing SSFN, a part of each weight matrix is trained using a
layer-wise convex optimization approach (a supervised training), while the
other part is chosen as a random matrix instance (an unsupervised training). In
this article, the use of deterministic transforms instead of random matrix
instances for the SSFN weight matrices is explored. The use of deterministic
transforms provides a reduction in computational complexity. The use of several
deterministic transforms is investigated, such as discrete cosine transform,
Hadamard transform, Hartley transform, and wavelet transforms. The choice of a
deterministic transform among a set of transforms is made in an unsupervised
manner. To this end, two methods based on features' statistical parameters are
developed. The proposed methods help to design a neural net where deterministic
transforms can vary across its layers' weight matrices. The effectiveness of
the proposed approach vis-a-vis the SSFN is illustrated for object
classification tasks using several benchmark datasets.

    

### [[2110.03522] Surrogate-Based Black-Box Optimization Method for Costly Molecular Properties](http://arxiv.org/abs/2110.03522)


  AI-assisted molecular optimization is a very active research field as it is
expected to provide the next-generation drugs and molecular materials. An
important difficulty is that the properties to be optimized rely on costly
evaluations. Machine learning methods are investigated with success to predict
these properties, but show generalization issues on less known areas of the
chemical space. We propose here a surrogate-based black box optimization
method, to tackle jointly the optimization and machine learning problems. It
consists in optimizing the expected improvement of the surrogate of a molecular
property using an evolutionary algorithm. The surrogate is defined as a
Gaussian Process Regression (GPR) model, learned on a relevant area of the
search space with respect to the property to be optimized. We show that our
approach can successfully optimize a costly property of interest much faster
than a purely metaheuristic approach.

    

### [[2110.03528] Decoding ECoG signal into 3D hand translation using deep learning](http://arxiv.org/abs/2110.03528)


  Motor brain-computer interfaces (BCIs) are a promising technology that may
enable motor-impaired people to interact with their environment. Designing
real-time and accurate BCI is crucial to make such devices useful, safe, and
easy to use by patients in a real-life environment. Electrocorticography
(ECoG)-based BCIs emerge as a good compromise between invasiveness of the
recording device and good spatial and temporal resolution of the recorded
signal. However, most ECoG signal decoders used to predict continuous hand
movements are linear models. These models have a limited representational
capacity and may fail to capture the relationship between ECoG signal and
continuous hand movements. Deep learning (DL) models, which are
state-of-the-art in many problems, could be a solution to better capture this
relationship. In this study, we tested several DL-based architectures to
predict imagined 3D continuous hand translation using time-frequency features
extracted from ECoG signals. The dataset used in the analysis is a part of a
long-term clinical trial (this http URL identifier: NCT02550522) and was
acquired during a closed-loop experiment with a tetraplegic subject. The
proposed architectures include multilayer perceptron (MLP), convolutional
neural networks (CNN), and long short-term memory networks (LSTM). The accuracy
of the DL-based and multilinear models was compared offline using cosine
similarity. Our results show that CNN-based architectures outperform the
current state-of-the-art multilinear model. The best architecture exploited the
spatial correlation between neighboring electrodes with CNN and benefited from
the sequential character of the desired hand trajectory by using LSTMs.
Overall, DL increased the average cosine similarity, compared to the
multilinear model, by up to 60%, from 0.189 to 0.302 and from 0.157 to 0.249
for the left and right hand, respectively.

    

### [[2110.03535] A Few-shot Learning Graph Multi-Trajectory Evolution Network for Forecasting Multimodal Baby Connectivity Development from a Baseline Timepoint](http://arxiv.org/abs/2110.03535)


  Charting the baby connectome evolution trajectory during the first year after
birth plays a vital role in understanding dynamic connectivity development of
baby brains. Such analysis requires acquisition of longitudinal connectomic
datasets. However, both neonatal and postnatal scans are rarely acquired due to
various difficulties. A small body of works has focused on predicting baby
brain evolution trajectory from a neonatal brain connectome derived from a
single modality. Although promising, large training datasets are essential to
boost model learning and to generalize to a multi-trajectory prediction from
different modalities (i.e., functional and morphological connectomes). Here, we
unprecedentedly explore the question: Can we design a few-shot learning-based
framework for predicting brain graph trajectories across different modalities?
To this aim, we propose a Graph Multi-Trajectory Evolution Network (GmTE-Net),
which adopts a teacher-student paradigm where the teacher network learns on
pure neonatal brain graphs and the student network learns on simulated brain
graphs given a set of different timepoints. To the best of our knowledge, this
is the first teacher-student architecture tailored for brain graph
multi-trajectory growth prediction that is based on few-shot learning and
generalized to graph neural networks (GNNs). To boost the performance of the
student network, we introduce a local topology-aware distillation loss that
forces the predicted graph topology of the student network to be consistent
with the teacher network. Experimental results demonstrate substantial
performance gains over benchmark methods. Hence, our GmTE-Net can be leveraged
to predict atypical brain connectivity trajectory evolution across various
modalities. Our code is available at https: //github.com/basiralab/GmTE-Net.

    

### [[2110.03540] A Broad Ensemble Learning System for Drifting Stream Classification](http://arxiv.org/abs/2110.03540)


  Data stream classification has become a major research topic due to the
increase in temporal data. One of the biggest hurdles of data stream
classification is the development of algorithms that deal with evolving data,
also known as concept drifts. As data changes over time, static prediction
models lose their validity. Adapting to concept drifts provides more robust and
better performing models. The Broad Learning System (BLS) is an effective broad
neural architecture recently developed for incremental learning. BLS cannot
provide instant response since it requires huge data chunks and is unable to
handle concept drifts. We propose a Broad Ensemble Learning System (BELS) for
stream classification with concept drift. BELS uses a novel updating method
that greatly improves best-in-class model accuracy. It employs a dynamic output
ensemble layer to address the limitations of BLS. We present its mathematical
derivation, provide comprehensive experiments with 11 datasets that demonstrate
the adaptability of our model, including a comparison of our model with BLS,
and provide parameter and robustness analysis on several drifting streams,
showing that it statistically significantly outperforms seven state-of-the-art
baselines. We show that our proposed method improves on average 44% compared to
BLS, and 29% compared to other competitive baselines.

    

### [[2110.03549] Bias-Variance Tradeoffs in Single-Sample Binary Gradient Estimators](http://arxiv.org/abs/2110.03549)


  Discrete and especially binary random variables occur in many machine
learning models, notably in variational autoencoders with binary latent states
and in stochastic binary networks. When learning such models, a key tool is an
estimator of the gradient of the expected loss with respect to the
probabilities of binary variables. The straight-through (ST) estimator gained
popularity due to its simplicity and efficiency, in particular in deep networks
where unbiased estimators are impractical. Several techniques were proposed to
improve over ST while keeping the same low computational complexity:
Gumbel-Softmax, ST-Gumbel-Softmax, BayesBiNN, FouST. We conduct a theoretical
analysis of Bias and Variance of these methods in order to understand tradeoffs
and verify the originally claimed properties. The presented theoretical results
are mainly negative, showing limitations of these methods and in some cases
revealing serious issues.

    

### [[2110.03553] Shift-BNN: Highly-Efficient Probabilistic Bayesian Neural Network Training via Memory-Friendly Pattern Retrieving](http://arxiv.org/abs/2110.03553)


  Bayesian Neural Networks (BNNs) that possess a property of uncertainty
estimation have been increasingly adopted in a wide range of safety-critical AI
applications which demand reliable and robust decision making, e.g.,
self-driving, rescue robots, medical image diagnosis. The training procedure of
a probabilistic BNN model involves training an ensemble of sampled DNN models,
which induces orders of magnitude larger volume of data movement than training
a single DNN model. In this paper, we reveal that the root cause for BNN
training inefficiency originates from the massive off-chip data transfer by
Gaussian Random Variables (GRVs). To tackle this challenge, we propose a novel
design that eliminates all the off-chip data transfer by GRVs through the
reversed shifting of Linear Feedback Shift Registers (LFSRs) without incurring
any training accuracy loss. To efficiently support our LFSR reversion strategy
at the hardware level, we explore the design space of the current DNN
accelerators and identify the optimal computation mapping scheme to best
accommodate our strategy. By leveraging this finding, we design and prototype
the first highly efficient BNN training accelerator, named Shift-BNN, that is
low-cost and scalable. Extensive evaluation on five representative BNN models
demonstrates that Shift-BNN achieves an average of 4.9x (up to 10.8x) boost in
energy efficiency and 1.6x (up to 2.8x) speedup over the baseline DNN training
accelerator.

    

### [[2110.03576] Training Stable Graph Neural Networks Through Constrained Learning](http://arxiv.org/abs/2110.03576)


  Graph Neural Networks (GNN) rely on graph convolutions to learn features from
network data. GNNs are stable to different types of perturbations of the
underlying graph, a property that they inherit from graph filters. In this
paper we leverage the stability property of GNNs as a typing point in order to
seek for representations that are stable within a distribution. We propose a
novel constrained learning approach by imposing a constraint on the stability
condition of the GNN within a perturbation of choice. We showcase our framework
in real world data, corroborating that we are able to obtain more stable
representations while not compromising the overall accuracy of the predictor.

    

### [[2110.03580] A Model Selection Approach for Corruption Robust Reinforcement Learning](http://arxiv.org/abs/2110.03580)


  We develop a model selection approach to tackle reinforcement learning with
adversarial corruption in both transition and reward. For finite-horizon
tabular MDPs, without prior knowledge on the total amount of corruption, our
algorithm achieves a regret bound of
$\widetilde{\mathcal{O}}(\min\{\frac{1}{\Delta}, \sqrt{T}\}+C)$ where $T$ is
the number of episodes, $C$ is the total amount of corruption, and $\Delta$ is
the reward gap between the best and the second-best policy. This is the first
worst-case optimal bound achieved without knowledge of $C$, improving previous
results of Lykouris et al. (2021); Chen et al. (2021); Wu et al. (2021). For
finite-horizon linear MDPs, we develop a computationally efficient algorithm
with a regret bound of $\widetilde{\mathcal{O}}(\sqrt{(1+C)T})$, and another
computationally inefficient one with $\widetilde{\mathcal{O}}(\sqrt{T}+C)$,
improving the result of Lykouris et al. (2021) and answering an open question
by Zhang et al. (2021b). Finally, our model selection framework can be easily
applied to other settings including linear bandits, linear contextual bandits,
and MDPs with general function approximation, leading to several improved or
new results.

    

### [[2110.03585] To Charge or To Sell? EV Pack Useful Life Estimation via LSTMs and Autoencoders](http://arxiv.org/abs/2110.03585)


  Electric Vehicles (EVs) are spreading fast as they promise to provide better
performances and comfort, but above all, to help facing climate change. Despite
their success, their cost is still a challenge. One of the most expensive
components of EVs is lithium-ion batteries, which became the standard for
energy storage in a wide range of applications. Precisely estimating the
Remaining Useful Life (RUL) of battery packs can open to their reuse and thus
help to reduce the cost of EVs and improve sustainability. A correct RUL
estimation can be used to quantify the residual market value of the battery
pack. The customer can then decide to sell the battery when it still has a
value, i.e., before it exceeds its end of life of the target application and
can still be reused in a second domain without compromising safety and
reliability. In this paper, we propose to use a Deep Learning approach based on
LSTMs and Autoencoders to estimate the RUL of li-ion batteries. Compared to
what has been proposed so far in the literature, we employ measures to ensure
the applicability of the method also in the real deployed application. Such
measures include (1) avoid using non-measurable variables as input, (2) employ
appropriate datasets with wide variability and different conditions, (3) do not
use cycles to define the RUL.

    

### [[2110.03594] Ship Performance Monitoring using Machine-learning](http://arxiv.org/abs/2110.03594)


  The hydrodynamic performance of a sea-going ship varies over its lifespan due
to factors like marine fouling and the condition of the anti-fouling paint
system. In order to accurately estimate the power demand and fuel consumption
for a planned voyage, it is important to assess the hydrodynamic performance of
the ship. The current work uses machine-learning (ML) methods to estimate the
hydrodynamic performance of a ship using the onboard recorded in-service data.
Three ML methods, NL-PCR, NL-PLSR and probabilistic ANN, are calibrated using
the data from two sister ships. The calibrated models are used to extract the
varying trend in ship's hydrodynamic performance over time and predict the
change in performance through several propeller and hull cleaning events. The
predicted change in performance is compared with the corresponding values
estimated using the fouling friction coefficient ($\Delta C_F$). The ML methods
are found to be performing well while modelling the hydrodynamic state
variables of the ships with probabilistic ANN model performing the best, but
the results from NL-PCR and NL-PLSR are not far behind, indicating that it may
be possible to use simple methods to solve such problems with the help of
domain knowledge.

    

### [[2110.03595] Generalization in Deep RL for TSP Problems via Equivariance and Local Search](http://arxiv.org/abs/2110.03595)


  Deep reinforcement learning (RL) has proved to be a competitive heuristic for
solving small-sized instances of traveling salesman problems (TSP), but its
performance on larger-sized instances is insufficient. Since training on large
instances is impractical, we design a novel deep RL approach with a focus on
generalizability. Our proposition consisting of a simple deep learning
architecture that learns with novel RL training techniques, exploits two main
ideas. First, we exploit equivariance to facilitate training. Second, we
interleave efficient local search heuristics with the usual RL training to
smooth the value landscape. In order to validate the whole approach, we
empirically evaluate our proposition on random and realistic TSP problems
against relevant state-of-the-art deep RL methods. Moreover, we present an
ablation study to understand the contribution of each of its component

    

### [[2110.03604] Online Markov Decision Processes with Non-oblivious Strategic Adversary](http://arxiv.org/abs/2110.03604)


  We study a novel setting in Online Markov Decision Processes (OMDPs) where
the loss function is chosen by a non-oblivious strategic adversary who follows
a no-external regret algorithm. In this setting, we first demonstrate that
MDP-Expert, an existing algorithm that works well with oblivious adversaries
can still apply and achieve a policy regret bound of $\mathcal{O}(\sqrt{T
\log(L)}+\tau^2\sqrt{ T \log(|A|)})$ where $L$ is the size of adversary's pure
strategy set and $|A|$ denotes the size of agent's action space. Considering
real-world games where the support size of a NE is small, we further propose a
new algorithm: MDP-Online Oracle Expert (MDP-OOE), that achieves a policy
regret bound of $\mathcal{O}(\sqrt{T\log(L)}+\tau^2\sqrt{ T k \log(k)})$ where
$k$ depends only on the support size of the NE. MDP-OOE leverages the key
benefit of Double Oracle in game theory and thus can solve games with
prohibitively large action space. Finally, to better understand the learning
dynamics of no-regret methods, under the same setting of no-external regret
adversary in OMDPs, we introduce an algorithm that achieves last-round
convergence result to a NE. To our best knowledge, this is first work leading
to the last iteration result in OMDPs.

    

### [[2110.03605] One Thing to Fool them All: Generating Interpretable, Universal, and Physically-Realizable Adversarial Features](http://arxiv.org/abs/2110.03605)


  It is well understood that modern deep networks are vulnerable to adversarial
attacks. However, conventional methods fail to produce adversarial
perturbations that are intelligible to humans, and they pose limited threats in
the physical world. To study feature-class associations in networks and better
understand the real-world threats they face, we develop feature-level
adversarial perturbations using deep image generators and a novel optimization
objective. We term these feature-fool attacks. We show that they are versatile
and use them to generate targeted feature-level attacks at the ImageNet scale
that are simultaneously interpretable, universal to any source image, and
physically-realizable. These attacks can also reveal spurious,
semantically-describable feature/class associations, and we use them to guide
the design of "copy/paste" adversaries in which one natural image is pasted
into another to cause a targeted misclassification.

    

### [[2110.03608] How to Sense the World: Leveraging Hierarchy in Multimodal Perception for Robust Reinforcement Learning Agents](http://arxiv.org/abs/2110.03608)


  This work addresses the problem of sensing the world: how to learn a
multimodal representation of a reinforcement learning agent's environment that
allows the execution of tasks under incomplete perceptual conditions. To
address such problem, we argue for hierarchy in the design of representation
models and contribute with a novel multimodal representation model, MUSE. The
proposed model learns hierarchical representations: low-level modality-specific
representations, encoded from raw observation data, and a high-level multimodal
representation, encoding joint-modality information to allow robust state
estimation. We employ MUSE as the sensory representation model of deep
reinforcement learning agents provided with multimodal observations in Atari
games. We perform a comparative study over different designs of reinforcement
learning agents, showing that MUSE allows agents to perform tasks under
incomplete perceptual experience with minimal performance loss. Finally, we
evaluate the performance of MUSE in literature-standard multimodal scenarios
with higher number and more complex modalities, showing that it outperforms
state-of-the-art multimodal variational autoencoders in single and
cross-modality generation.

    

### [[2110.03609] Applying Phonological Features in Multilingual Text-To-Speech](http://arxiv.org/abs/2110.03609)


  This study investigates whether phonological features can be applied in
text-to-speech systems to generate native and non-native speech. We present a
mapping between ARPABET/pinyin->SAMPA/SAMPA-SC->phonological features in this
paper, and tested whether native, non-native, and code-switched speech could be
successfully generated using this mapping. We ran two experiments, one with a
small dataset and one with a larger dataset. The results proved that
phonological features can be a feasible input system, although it needs further
investigation to improve model performance. The accented output generated by
the TTS models also helps with understanding human second language acquisition
processes.

    

### [[2110.03618] Causal Direction of Data Collection Matters: Implications of Causal and Anticausal Learning in NLP](http://arxiv.org/abs/2110.03618)


  The principle of independent causal mechanisms (ICM) states that generative
processes of real world data consist of independent modules which do not
influence or inform each other. While this idea has led to fruitful
developments in the field of causal inference, it is not widely-known in the
NLP community. In this work, we argue that the causal direction of the data
collection process bears nontrivial implications that can explain a number of
published NLP findings, such as differences in semi-supervised learning (SSL)
and domain adaptation (DA) performance across different settings. We categorize
common NLP tasks according to their causal direction and empirically assay the
validity of the ICM principle for text data using minimum description length.
We conduct an extensive meta-analysis of over 100 published SSL and 30 DA
studies, and find that the results are consistent with our expectations based
on causal insights. This work presents the first attempt to analyze the ICM
principle in NLP, and provides constructive suggestions for future modeling
choices. Code available at this https URL.

    

### [[2110.03620] Hyperparameter Tuning with Renyi Differential Privacy](http://arxiv.org/abs/2110.03620)


  For many differentially private algorithms, such as the prominent noisy
stochastic gradient descent (DP-SGD), the analysis needed to bound the privacy
leakage of a single training run is well understood. However, few studies have
reasoned about the privacy leakage resulting from the multiple training runs
needed to fine tune the value of the training algorithm's hyperparameters. In
this work, we first illustrate how simply setting hyperparameters based on
non-private training runs can leak private information. Motivated by this
observation, we then provide privacy guarantees for hyperparameter search
procedures within the framework of Renyi Differential Privacy. Our results
improve and extend the work of Liu and Talwar (STOC 2019). Our analysis
supports our previous observation that tuning hyperparameters does indeed leak
private information, but we prove that, under certain assumptions, this leakage
is modest, as long as each candidate training run needed to select
hyperparameters is itself differentially private.

    

### [[2110.03625] Time Series Forecasting Using Manifold Learning](http://arxiv.org/abs/2110.03625)


  We address a three-tier numerical framework based on manifold learning for
the forecasting of high-dimensional time series. At the first step, we embed
the time series into a reduced low-dimensional space using a nonlinear manifold
learning algorithm such as Locally Linear Embedding and Diffusion Maps. At the
second step, we construct reduced-order regression models on the manifold, in
particular Multivariate Autoregressive (MVAR) and Gaussian Process Regression
(GPR) models, to forecast the embedded dynamics. At the final step, we lift the
embedded time series back to the original high-dimensional space using Radial
Basis Functions interpolation and Geometric Harmonics. For our illustrations,
we test the forecasting performance of the proposed numerical scheme with four
sets of time series: three synthetic stochastic ones resembling EEG signals
produced from linear and nonlinear stochastic models with different model
orders, and one real-world data set containing daily time series of 10 key
foreign exchange rates (FOREX) spanning the time period 19/09/2001-29/10/2020.
The forecasting performance of the proposed numerical scheme is assessed using
the combinations of manifold learning, modelling and lifting approaches. We
also provide a comparison with the Principal Component Analysis algorithm as
well as with the naive random walk model and the MVAR and GPR models trained
and implemented directly in the high-dimensional space.

    

### [[2110.03628] Boxhead: A Dataset for Learning Hierarchical Representations](http://arxiv.org/abs/2110.03628)


  Disentanglement is hypothesized to be beneficial towards a number of
downstream tasks. However, a common assumption in learning disentangled
representations is that the data generative factors are statistically
independent. As current methods are almost solely evaluated on toy datasets
where this ideal assumption holds, we investigate their performance in
hierarchical settings, a relevant feature of real-world data. In this work, we
introduce Boxhead, a dataset with hierarchically structured ground-truth
generative factors. We use this novel dataset to evaluate the performance of
state-of-the-art autoencoder-based disentanglement models and observe that
hierarchical models generally outperform single-layer VAEs in terms of
disentanglement of hierarchically arranged factors.

    

### [[2110.03634] Enabling On-Device Training of Speech Recognition Models with Federated Dropout](http://arxiv.org/abs/2110.03634)


  Federated learning can be used to train machine learning models on the edge
on local data that never leave devices, providing privacy by default. This
presents a challenge pertaining to the communication and computation costs
associated with clients' devices. These costs are strongly correlated with the
size of the model being trained, and are significant for state-of-the-art
automatic speech recognition models.
We propose using federated dropout to reduce the size of client models while
training a full-size model server-side. We provide empirical evidence of the
effectiveness of federated dropout, and propose a novel approach to vary the
dropout rate applied at each layer. Furthermore, we find that federated dropout
enables a set of smaller sub-models within the larger model to independently
have low word error rates, making it easier to dynamically adjust the size of
the model deployed for inference.

    

### [[2110.03639] Using Contrastive Learning and Pseudolabels to learn representations for Retail Product Image Classification](http://arxiv.org/abs/2110.03639)


  Retail product Image classification problems are often few shot
classification problems, given retail product classes cannot have the type of
variations across images like a cat or dog or tree could have. Previous works
have shown different methods to finetune Convolutional Neural Networks to
achieve better classification accuracy on such datasets. In this work, we try
to address the problem statement : Can we pretrain a Convolutional Neural
Network backbone which yields good enough representations for retail product
images, so that training a simple logistic regression on these representations
gives us good classifiers ? We use contrastive learning and pseudolabel based
noisy student training to learn representations that get accuracy in order of
finetuning the entire Convnet backbone for retail product image classification.

    

### [[2110.03646] Using Keypoint Matching and Interactive Self Attention Network to verify Retail POSMs](http://arxiv.org/abs/2110.03646)


  Point of Sale Materials(POSM) are the merchandising and decoration items that
are used by companies to communicate product information and offers in retail
stores. POSMs are part of companies' retail marketing strategy and are often
applied as stylized window displays around retail shelves. In this work, we
apply computer vision techniques to the task of verification of POSMs in
supermarkets by telling if all desired components of window display are present
in a shelf image. We use Convolutional Neural Network based unsupervised
keypoint matching as a baseline to verify POSM components and propose a
supervised Neural Network based method to enhance the accuracy of baseline by a
large margin. We also show that the supervised pipeline is not restricted to
the POSM material it is trained on and can generalize. We train and evaluate
our model on a private dataset composed of retail shelf images.

    

### [[2110.03649] Neural Networks, Inside Out: Solving for Inputs Given Parameters (A Preliminary Investigation)](http://arxiv.org/abs/2110.03649)


  Artificial neural network (ANN) is a supervised learning algorithm, where
parameters are learned by several back-and-forth iterations of passing the
inputs through the network, comparing the output with the expected labels, and
correcting the parameters. Inspired by a recent work of Derian and Kramer
(2020), we investigate a different problem: Suppose an observer can view how
the ANN parameters evolve over many iterations, but the dataset is oblivious to
him. For instance, this can be an adversary eavesdropping on a multi-party
computation of an ANN parameters (where intermediate parameters are leaked).
Can he form a system of equations, and solve it to recover the dataset?

    

### [[2110.03655] Augmenting Reinforcement Learning with Behavior Primitives for Diverse Manipulation Tasks](http://arxiv.org/abs/2110.03655)


  Realistic manipulation tasks require a robot to interact with an environment
with a prolonged sequence of motor actions. While deep reinforcement learning
methods have recently emerged as a promising paradigm for automating
manipulation behaviors, they usually fall short in long-horizon tasks due to
the exploration burden. This work introduces MAnipulation Primitive-augmented
reinforcement LEarning (MAPLE), a learning framework that augments standard
reinforcement learning algorithms with a pre-defined library of behavior
primitives. These behavior primitives are robust functional modules specialized
in achieving manipulation goals, such as grasping and pushing. To use these
heterogeneous primitives, we develop a hierarchical policy that involves the
primitives and instantiates their executions with input parameters. We
demonstrate that MAPLE outperforms baseline approaches by a significant margin
on a suite of simulated manipulation tasks. We also quantify the compositional
structure of the learned behaviors and highlight our method's ability to
transfer policies to new task variants and to physical hardware. Videos and
code are available at this https URL


### [[2110.03659] Transform2Act: Learning a Transform-and-Control Policy for Efficient Agent Design](http://arxiv.org/abs/2110.03659)


  An agent's functionality is largely determined by its design, i.e., skeletal
structure and joint attributes (e.g., length, size, strength). However, finding
the optimal agent design for a given function is extremely challenging since
the problem is inherently combinatorial and the design space is prohibitively
large. Additionally, it can be costly to evaluate each candidate design which
requires solving for its optimal controller. To tackle these problems, our key
idea is to incorporate the design procedure of an agent into its
decision-making process. Specifically, we learn a conditional policy that, in
an episode, first applies a sequence of transform actions to modify an agent's
skeletal structure and joint attributes, and then applies control actions under
the new design. To handle a variable number of joints across designs, we use a
graph-based policy where each graph node represents a joint and uses message
passing with its neighbors to output joint-specific actions. Using policy
gradient methods, our approach enables first-order optimization of agent design
and control as well as experience sharing across different designs, which
improves sample efficiency tremendously. Experiments show that our approach,
Transform2Act, outperforms prior methods significantly in terms of convergence
speed and final performance. Notably, Transform2Act can automatically discover
plausible designs similar to giraffes, squids, and spiders. Our project website
is at this https URL.

    

### [[2110.03660] Developing Medical AI : a cloud-native audio-visual data collection study](http://arxiv.org/abs/2110.03660)


  Designing Artificial Intelligence (AI) solutions that can operate in
real-world situations is a highly complex task. Deploying such solutions in the
medical domain is even more challenging. The promise of using AI to improve
patient care and reduce cost has encouraged many companies to undertake such
endeavours. For our team, the goal has been to improve early identification of
deteriorating patients in the hospital. Identifying patient deterioration in
lower acuity wards relies, to a large degree on the attention and intuition of
clinicians, rather than on the presence of physiological monitoring devices. In
these care areas, an automated tool which could continuously observe patients
and notify the clinical staff of suspected deterioration, would be extremely
valuable. In order to develop such an AI-enabled tool, a large collection of
patient images and audio correlated with corresponding vital signs, past
medical history and clinical outcome would be indispensable. To the best of our
knowledge, no such public or for-pay data set currently exists. This lack of
audio-visual data led to the decision to conduct exactly such study. The main
contributions of this paper are, the description of a protocol for audio-visual
data collection study, a cloud-architecture for efficiently processing and
consuming such data, and the design of a specific data collection device.

    

### [[2110.03665] Revisiting SVD to generate powerful Node Embeddings for Recommendation Systems](http://arxiv.org/abs/2110.03665)


  Graph Representation Learning (GRL) is an upcoming and promising area in
recommendation systems. In this paper, we revisit the Singular Value
Decomposition (SVD) of adjacency matrix for embedding generation of users and
items and use a two-layer neural network on top of these embeddings to learn
relevance between user-item pairs. Inspired by the success of higher-order
learning in GRL, we further propose an extension of this method to include
two-hop neighbors for SVD through the second order of the adjacency matrix and
demonstrate improved performance compared with the simple SVD method which only
uses one-hop neighbors. Empirical validation on three publicly available
datasets of recommendation system demonstrates that the proposed methods,
despite being simple, beat many state-of-the-art methods and for two of three
datasets beats all of them up to a margin of 10%. Through our research, we want
to shed light on the effectiveness of matrix factorization approaches,
specifically SVD, in the deep learning era and show that these methods still
contribute as important baselines in recommendation systems.

    

### [[2110.03666] Joint inference of multiple graphs with hidden variables from stationary graph signals](http://arxiv.org/abs/2110.03666)


  Learning graphs from sets of nodal observations represents a prominent
problem formally known as graph topology inference. However, current approaches
are limited by typically focusing on inferring single networks, and they assume
that observations from all nodes are available. First, many contemporary setups
involve multiple related networks, and second, it is often the case that only a
subset of nodes is observed while the rest remain hidden. Motivated by these
facts, we introduce a joint graph topology inference method that models the
influence of the hidden variables. Under the assumptions that the observed
signals are stationary on the sought graphs and the graphs are closely related,
the joint estimation of multiple networks allows us to exploit such
relationships to improve the quality of the learned graphs. Moreover, we
confront the challenging problem of modeling the influence of the hidden nodes
to minimize their detrimental effect. To obtain an amenable approach, we take
advantage of the particular structure of the setup at hand and leverage the
similarity between the different graphs, which affects both the observed and
the hidden nodes. To test the proposed method, numerical simulations over
synthetic and real-world graphs are provided.

    

### [[2110.03673] Tighter Sparse Approximation Bounds for ReLU Neural Networks](http://arxiv.org/abs/2110.03673)


  A well-known line of work (Barron, 1993; Breiman, 1993; Klusowski & Barron,
2018) provides bounds on the width $n$ of a ReLU two-layer neural network
needed to approximate a function $f$ over the ball $\mathcal{B}_R(\R^d)$ up to
error $\epsilon$, when the Fourier based quantity $C_f = \int_{\R^d} \|\xi\|^2
|\hat{f}(\xi)| \ d\xi$ is finite. More recently Ongie et al. (2019) used the
Radon transform as a tool for analysis of infinite-width ReLU two-layer
networks. In particular, they introduce the concept of Radon-based
$\mathcal{R}$-norms and show that a function defined on $\R^d$ can be
represented as an infinite-width two-layer neural network if and only if its
$\mathcal{R}$-norm is finite. In this work, we extend the framework of Ongie et
al. (2019) and define similar Radon-based semi-norms ($\mathcal{R},
\mathcal{U}$-norms) such that a function admits an infinite-width neural
network representation on a bounded open set $\mathcal{U} \subseteq \R^d$ when
its $\mathcal{R}, \mathcal{U}$-norm is finite. Building on this, we derive
sparse (finite-width) neural network approximation bounds that refine those of
Breiman (1993); Klusowski & Barron (2018). Finally, we show that infinite-width
neural network representations on bounded open sets are not unique and study
their structure, providing a functional view of mode connectivity.

    

### [[2110.03677] Large Learning Rate Tames Homogeneity: Convergence and Balancing Effect](http://arxiv.org/abs/2110.03677)


  Recent empirical advances show that training deep models with large learning
rate often improves generalization performance. However, theoretical
justifications on the benefits of large learning rate are highly limited, due
to challenges in analysis. In this paper, we consider using Gradient Descent
(GD) with a large learning rate on a homogeneous matrix factorization problem,
i.e., $\min_{X, Y} \|A - XY^\top\|_{\sf F}^2$. We prove a convergence theory
for constant large learning rates well beyond $2/L$, where $L$ is the largest
eigenvalue of Hessian at the initialization. Moreover, we rigorously establish
an implicit bias of GD induced by such a large learning rate, termed
'balancing', meaning that magnitudes of $X$ and $Y$ at the limit of GD
iterations will be close even if their initialization is significantly
unbalanced. Numerical experiments are provided to support our theory.

    

### [[2110.03681] Neural Tangent Kernel Empowered Federated Learning](http://arxiv.org/abs/2110.03681)


  Federated learning (FL) is a privacy-preserving paradigm where multiple
participants jointly solve a machine learning problem without sharing raw data.
Unlike traditional distributed learning, a unique characteristic of FL is
statistical heterogeneity, namely, data distributions across participants are
different from each other. Meanwhile, recent advances in the interpretation of
neural networks have seen a wide use of neural tangent kernel (NTK) for
convergence and generalization analyses. In this paper, we propose a novel FL
paradigm empowered by the NTK framework. The proposed paradigm addresses the
challenge of statistical heterogeneity by transmitting update data that are
more expressive than those of the traditional FL paradigms. Specifically,
sample-wise Jacobian matrices, rather than model weights/gradients, are
uploaded by participants. The server then constructs an empirical kernel matrix
to update a global model without explicitly performing gradient descent. We
further develop a variant with improved communication efficiency and enhanced
privacy. Numerical results show that the proposed paradigm can achieve the same
accuracy while reducing the number of communication rounds by an order of
magnitude compared to federated averaging.

    

### [[2110.03684] Cross-Domain Imitation Learning via Optimal Transport](http://arxiv.org/abs/2110.03684)


  Cross-domain imitation learning studies how to leverage expert demonstrations
of one agent to train an imitation agent with a different embodiment or
morphology. Comparing trajectories and stationary distributions between the
expert and imitation agents is challenging because they live on different
systems that may not even have the same dimensionality. We propose
Gromov-Wasserstein Imitation Learning (GWIL), a method for cross-domain
imitation that uses the Gromov-Wasserstein distance to align and compare states
between the different spaces of the agents. Our theory formally characterizes
the scenarios where GWIL preserves optimality, revealing its possibilities and
limitations. We demonstrate the effectiveness of GWIL in non-trivial continuous
control domains ranging from simple rigid transformation of the expert domain
to arbitrary transformation of the state-action space.

    

### [[1812.09234] A Primal-dual Learning Algorithm for Personalized Dynamic Pricing with an Inventory Constraint](http://arxiv.org/abs/1812.09234)


  We consider the problem of a firm seeking to use personalized pricing to sell
an exogenously given stock of a product over a finite selling horizon to
different consumer types. We assume that the type of an arriving consumer can
be observed but the demand function associated with each type is initially
unknown. The firm sets personalized prices dynamically for each type and
attempts to maximize the revenue over the season. We provide a learning
algorithm that is near-optimal when the demand and capacity scale in
proportion. The algorithm utilizes the primal-dual formulation of the problem
and learns the dual optimal solution explicitly. It allows the algorithm to
overcome the curse of dimensionality (the rate of regret is independent of the
number of types) and sheds light on novel algorithmic designs for learning
problems with resource constraints.

    

### [[1910.01931] Sparse Popularity Adjusted Stochastic Block Model](http://arxiv.org/abs/1910.01931)


  In the present paper we study a sparse stochastic network enabled with a
block structure. The popular Stochastic Block Model (SBM) and the Degree
Corrected Block Model (DCBM) address sparsity by placing an upper bound on the
maximum probability of connections between any pair of nodes. As a result,
sparsity describes only the behavior of network as a whole, without
distinguishing between the block-dependent sparsity patterns. To the best of
our knowledge, the recently introduced Popularity Adjusted Block Model (PABM)
is the only block model that allows to introduce a {\it structural sparsity}
where some probabilities of connections are identically equal to zero while the
rest of them remain above a certain threshold. The latter presents a more
nuanced view of the network.

    

### [[2002.05954] Learning Functionally Decomposed Hierarchies for Continuous Control Tasks with Path Planning](http://arxiv.org/abs/2002.05954)


  We present HiDe, a novel hierarchical reinforcement learning architecture
that successfully solves long horizon control tasks and generalizes to unseen
test scenarios. Functional decomposition between planning and low-level control
is achieved by explicitly separating the state-action spaces across the
hierarchy, which allows the integration of task-relevant knowledge per layer.
We propose an RL-based planner to efficiently leverage the information in the
planning layer of the hierarchy, while the control layer learns a
goal-conditioned control policy. The hierarchy is trained jointly but allows
for the modular transfer of policy layers across hierarchies of different
agents. We experimentally show that our method generalizes across unseen test
environments and can scale to 3x horizon length compared to both learning and
non-learning based methods. We evaluate on complex continuous control tasks
with sparse rewards, including navigation and robot manipulation.

    

### [[2006.06320] Hypernetwork-Based Augmentation](http://arxiv.org/abs/2006.06320)


  Data augmentation is an effective technique to improve the generalization of
deep neural networks. Recently, AutoAugment proposed a well-designed search
space and a search algorithm that automatically finds augmentation policies in
a data-driven manner. However, AutoAugment is computationally intensive. In
this paper, we propose an efficient gradient-based search algorithm, called
Hypernetwork-Based Augmentation (HBA), which simultaneously learns model
parameters and augmentation hyperparameters in a single training. Our HBA uses
a hypernetwork to approximate a population-based training algorithm, which
enables us to tune augmentation hyperparameters by gradient descent. Besides,
we introduce a weight sharing strategy that simplifies our hypernetwork
architecture and speeds up our search algorithm. We conduct experiments on
CIFAR-10, CIFAR-100, SVHN, and ImageNet. Our results show that HBA is
competitive to the state-of-the-art methods in terms of both search speed and
accuracy.

    

### [[2006.06880] Reintroducing Straight-Through Estimators as Principled Methods for Stochastic Binary Networks](http://arxiv.org/abs/2006.06880)


  Training neural networks with binary weights and activations is a challenging
problem due to the lack of gradients and difficulty of optimization over
discrete weights. Many successful experimental results have been achieved with
empirical straight-through (ST) approaches, proposing a variety of ad-hoc rules
for propagating gradients through non-differentiable activations and updating
discrete weights. At the same time, ST methods can be truly derived as
estimators in the stochastic binary network (SBN) model with Bernoulli weights.
We advance these derivations to a more complete and systematic study. We
analyze properties, estimation accuracy, obtain different forms of correct ST
estimators for activations and weights, explain existing empirical approaches
and their shortcomings, explain how latent weights arise from the mirror
descent method when optimizing over probabilities. This allows to reintroduce
ST methods, long known empirically, as sound approximations, apply them with
clarity and develop further improvements.

    

### [[2006.12000] Self-Knowledge Distillation with Progressive Refinement of Targets](http://arxiv.org/abs/2006.12000)


  The generalization capability of deep neural networks has been substantially
improved by applying a wide spectrum of regularization methods, e.g.,
restricting function space, injecting randomness during training, augmenting
data, etc. In this work, we propose a simple yet effective regularization
method named progressive self-knowledge distillation (PS-KD), which
progressively distills a model's own knowledge to soften hard targets (i.e.,
one-hot vectors) during training. Hence, it can be interpreted within a
framework of knowledge distillation as a student becomes a teacher itself.
Specifically, targets are adjusted adaptively by combining the ground-truth and
past predictions from the model itself. We show that PS-KD provides an effect
of hard example mining by rescaling gradients according to difficulty in
classifying examples. The proposed method is applicable to any supervised
learning tasks with hard targets and can be easily combined with existing
regularization methods to further enhance the generalization performance.
Furthermore, it is confirmed that PS-KD achieves not only better accuracy, but
also provides high quality of confidence estimates in terms of calibration as
well as ordinal ranking. Extensive experimental results on three different
tasks, image classification, object detection, and machine translation,
demonstrate that our method consistently improves the performance of the
state-of-the-art baselines. The code is available at
this https URL.

    

### [[2007.04911] GAMA: a General Automated Machine learning Assistant](http://arxiv.org/abs/2007.04911)


  The General Automated Machine learning Assistant (GAMA) is a modular AutoML
system developed to empower users to track and control how AutoML algorithms
search for optimal machine learning pipelines, and facilitate AutoML research
itself. In contrast to current, often black-box systems, GAMA allows users to
plug in different AutoML and post-processing techniques, logs and visualizes
the search process, and supports easy benchmarking. It currently features three
AutoML search algorithms, two model post-processing steps, and is designed to
allow for more components to be added.

    

### [[2007.08457] Artificial Fingerprinting for Generative Models: Rooting Deepfake Attribution in Training Data](http://arxiv.org/abs/2007.08457)


  Photorealistic image generation has reached a new level of quality due to the
breakthroughs of generative adversarial networks (GANs). Yet, the dark side of
such deepfakes, the malicious use of generated media, raises concerns about
visual misinformation. While existing research work on deepfake detection
demonstrates high accuracy, it is subject to advances in generation techniques
and adversarial iterations on detection countermeasure techniques. Thus, we
seek a proactive and sustainable solution on deepfake detection, that is
agnostic to the evolution of generative models, by introducing artificial
fingerprints into the models.
Our approach is simple and effective. We first embed artificial fingerprints
into training data, then validate a surprising discovery on the transferability
of such fingerprints from training data to generative models, which in turn
appears in the generated deepfakes. Experiments show that our fingerprinting
solution (1) holds for a variety of cutting-edge generative models, (2) leads
to a negligible side effect on generation quality, (3) stays robust against
image-level and model-level perturbations, (4) stays hard to be detected by
adversaries, and (5) converts deepfake detection and attribution into trivial
tasks and outperforms the recent state-of-the-art baselines. Our solution
closes the responsibility loop between publishing pre-trained generative model
inventions and their possible misuses, which makes it independent of the
current arms race.

    

### [[2009.06516] Justicia: A Stochastic SAT Approach to Formally Verify Fairness](http://arxiv.org/abs/2009.06516)


  As a technology ML is oblivious to societal good or bad, and thus, the field
of fair machine learning has stepped up to propose multiple mathematical
definitions, algorithms, and systems to ensure different notions of fairness in
ML applications. Given the multitude of propositions, it has become imperative
to formally verify the fairness metrics satisfied by different algorithms on
different datasets. In this paper, we propose a stochastic satisfiability
(SSAT) framework, Justicia, that formally verifies different fairness measures
of supervised learning algorithms with respect to the underlying data
distribution. We instantiate Justicia on multiple classification and bias
mitigation algorithms, and datasets to verify different fairness metrics, such
as disparate impact, statistical parity, and equalized odds. Justicia is
scalable, accurate, and operates on non-Boolean and compound sensitive
attributes unlike existing distribution-based verifiers, such as FairSquare and
VeriFair. Being distribution-based by design, Justicia is more robust than the
verifiers, such as AIF360, that operate on specific test samples. We also
theoretically bound the finite-sample error of the verified fairness measure.

    

### [[2009.12368] Generate Novel Molecules With Target Properties Using Conditional Generative Models](http://arxiv.org/abs/2009.12368)


  Drug discovery using deep learning has attracted a lot of attention of late
as it has obvious advantages like higher efficiency, less manual guessing and
faster process time. In this paper, we present a novel neural network for
generating small molecules similar to the ones in the training set. Our network
consists of an encoder made up of bi-GRU layers for converting the input
samples to a latent space, predictor for enhancing the capability of encoder
made up of 1D-CNN layers and a decoder comprised of uni-GRU layers for
reconstructing the samples from the latent space representation. Condition
vector in latent space is used for generating molecules with the desired
properties. We present the loss functions used for training our network,
experimental details and property prediction metrics. Our network outperforms
previous methods using Molecular weight, LogP and Quantitative Estimation of
Drug-likeness as the evaluation metrics.

    

### [[2010.03316] Batch Normalization Increases Adversarial Vulnerability and Decreases Adversarial Transferability: A Non-Robust Feature Perspective](http://arxiv.org/abs/2010.03316)


  Batch normalization (BN) has been widely used in modern deep neural networks
(DNNs) due to improved convergence. BN is observed to increase the model
accuracy while at the cost of adversarial robustness. There is an increasing
interest in the ML community to understand the impact of BN on DNNs, especially
related to the model robustness. This work attempts to understand the impact of
BN on DNNs from a non-robust feature perspective. Straightforwardly, the
improved accuracy can be attributed to the better utilization of useful
features. It remains unclear whether BN mainly favors learning robust features
(RFs) or non-robust features (NRFs). Our work presents empirical evidence that
supports that BN shifts a model towards being more dependent on NRFs. To
facilitate the analysis of such a feature robustness shift, we propose a
framework for disentangling robust usefulness into robustness and usefulness.
Extensive analysis under the proposed framework yields valuable insight on the
DNN behavior regarding robustness, e.g. DNNs first mainly learn RFs and then
NRFs. The insight that RFs transfer better than NRFs, further inspires simple
techniques to strengthen transfer-based black-box attacks.

    

### [[2010.11852] Efficient Robust Optimal Transport with Application to Multi-Label Classification](http://arxiv.org/abs/2010.11852)


  Optimal transport (OT) is a powerful geometric tool for comparing two
distributions and has been employed in various machine learning applications.
In this work, we propose a novel OT formulation that takes feature correlations
into account while learning the transport plan between two distributions. We
model the feature-feature relationship via a symmetric positive semi-definite
Mahalanobis metric in the OT cost function. For a certain class of regularizers
on the metric, we show that the optimization strategy can be considerably
simplified by exploiting the problem structure. For high-dimensional data, we
additionally propose suitable low-dimensional modeling of the Mahalanobis
metric. Overall, we view the resulting optimization problem as a non-linear OT
problem, which we solve using the Frank-Wolfe algorithm. Empirical results on
the discriminative learning setting, such as tag prediction and multi-class
classification, illustrate the good performance of our approach.

    

### [[2011.01129] Multi-Agent Reinforcement Learning for Visibility-based Persistent Monitoring](http://arxiv.org/abs/2011.01129)


  The Visibility-based Persistent Monitoring (VPM) problem seeks to find a set
of trajectories (or controllers) for robots to persistently monitor a changing
environment. Each robot has a sensor, such as a camera, with a limited
field-of-view that is obstructed by obstacles in the environment. The robots
may need to coordinate with each other to ensure no point in the environment is
left unmonitored for long periods of time. We model the problem such that there
is a penalty that accrues every time step if a point is left unmonitored.
However, the dynamics of the penalty are unknown to us. We present a
Multi-Agent Reinforcement Learning (MARL) algorithm for the VPM problem.
Specifically, we present a Multi-Agent Graph Attention Proximal Policy
Optimization (MA-G-PPO) algorithm that takes as input the local observations of
all agents combined with a low resolution global map to learn a policy for each
agent. The graph attention allows agents to share their information with others
leading to an effective joint policy. Our main focus is to understand how
effective MARL is for the VPM problem. We investigate five research questions
with this broader goal. We find that MA-G-PPO is able to learn a better policy
than the non-RL baseline in most cases, the effectiveness depends on agents
sharing information with each other, and the policy learnt shows emergent
behavior for the agents.

    

### [[2011.14048] Is Support Set Diversity Necessary for Meta-Learning?](http://arxiv.org/abs/2011.14048)


  Meta-learning is a popular framework for learning with limited data in which
an algorithm is produced by training over multiple few-shot learning tasks. For
classification problems, these tasks are typically constructed by sampling a
small number of support and query examples from a subset of the classes. While
conventional wisdom is that task diversity should improve the performance of
meta-learning, in this work we find evidence to the contrary: we propose a
modification to traditional meta-learning approaches in which we keep the
support sets fixed across tasks, thus reducing task diversity. Surprisingly, we
find that not only does this modification not result in adverse effects, it
almost always improves the performance for a variety of datasets and
meta-learning methods. We also provide several initial analyses to understand
this phenomenon. Our work serves to: (i) more closely investigate the effect of
support set construction for the problem of meta-learning, and (ii) suggest a
simple, general, and competitive baseline for few-shot learning.

    

### [[2012.05846] Full-Glow: Fully conditional Glow for more realistic image generation](http://arxiv.org/abs/2012.05846)


  Autonomous agents, such as driverless cars, require large amounts of labeled
visual data for their training. A viable approach for acquiring such data is
training a generative model with collected real data, and then augmenting the
collected real dataset with synthetic images from the model, generated with
control of the scene layout and ground truth labeling. In this paper we propose
Full-Glow, a fully conditional Glow-based architecture for generating plausible
and realistic images of novel street scenes given a semantic segmentation map
indicating the scene layout. Benchmark comparisons show our model to outperform
recent works in terms of the semantic segmentation performance of a pretrained
PSPNet. This indicates that images from our model are, to a higher degree than
from other models, similar to real images of the same kinds of scenes and
objects, making them suitable as training data for a visual semantic
segmentation or object recognition system.

    

### [[2101.09903] A Two-stage Framework for Compound Figure Separation](http://arxiv.org/abs/2101.09903)


  Scientific literature contains large volumes of complex, unstructured figures
that are compound in nature (i.e. composed of multiple images, graphs, and
drawings). Separation of these compound figures is critical for information
retrieval from these figures. In this paper, we propose a new strategy for
compound figure separation, which decomposes the compound figures into
constituent subfigures while preserving the association between the subfigures
and their respective caption components. We propose a two-stage framework to
address the proposed compound figure separation problem. In particular, the
subfigure label detection module detects all subfigure labels in the first
stage. Then, in the subfigure detection module, the detected subfigure labels
help to detect the subfigures by optimizing the feature selection process and
providing the global layout information as extra features. Extensive
experiments are conducted to validate the effectiveness and superiority of the
proposed framework, which improves the detection precision by 9%.

    

### [[2102.04877] Noisy Recurrent Neural Networks](http://arxiv.org/abs/2102.04877)


  We provide a general framework for studying recurrent neural networks (RNNs)
trained by injecting noise into hidden states. Specifically, we consider RNNs
that can be viewed as discretizations of stochastic differential equations
driven by input data. This framework allows us to study the implicit
regularization effect of general noise injection schemes by deriving an
approximate explicit regularizer in the small noise regime. We find that, under
reasonable assumptions, this implicit regularization promotes flatter minima;
it biases towards models with more stable dynamics; and, in classification
tasks, it favors models with larger classification margin. Sufficient
conditions for global stability are obtained, highlighting the phenomenon of
stochastic stabilization, where noise injection can improve stability during
training. Our theory is supported by empirical results which demonstrate that
the RNNs have improved robustness with respect to various input perturbations.

    

### [[2102.06004] Fairness-Aware PAC Learning from Corrupted Data](http://arxiv.org/abs/2102.06004)


  Addressing fairness concerns about machine learning models is a crucial step
towards their long-term adoption in real-world automated systems. While many
approaches have been developed for training fair models from data, little is
known about the robustness of these methods to data corruption. In this work we
consider fairness-aware learning under worst-case data manipulations. We show
that an adversary can in some situations force any learner to return an overly
biased classifier, regardless of the sample size and with or without degrading
accuracy, and that the strength of the excess bias increases for learning
problems with underrepresented protected groups in the data. We also prove that
our hardness results are tight up to constant factors. To this end, we study
two natural learning algorithms that optimize for both accuracy and fairness
and show that these algorithms enjoy guarantees that are order-optimal in terms
of the corruption ratio and the protected groups frequencies in the large data
limit.

    

### [[2102.07437] Data Quality Matters For Adversarial Training: An Empirical Study](http://arxiv.org/abs/2102.07437)


  Multiple intriguing problems are hovering in adversarial training, including
robust overfitting, robustness overestimation, and robustness-accuracy
trade-off. These problems pose great challenges to both reliable evaluation and
practical deployment. Here, we empirically show that these problems share one
common cause -- low-quality samples in the dataset. Specifically, we first
propose a strategy to measure the data quality based on the learning behaviors
of the data during adversarial training and find that low-quality data may not
be useful and even detrimental to the adversarial robustness. We then design
controlled experiments to investigate the interconnections between data quality
and problems in adversarial training. We find that when low-quality data is
removed, robust overfitting and robustness overestimation can be largely
alleviated; and robustness-accuracy trade-off becomes less significant. These
observations not only verify our intuition about data quality but may also open
new opportunities to advance adversarial training.

    

### [[2102.07824] A Koopman Approach to Understanding Sequence Neural Models](http://arxiv.org/abs/2102.07824)


  We introduce a new approach to understanding trained sequence neural models:
the Koopman Analysis of Neural Networks (KANN) method. Motivated by the
relation between time-series models and self-maps, we compute approximate
Koopman operators that encode well the latent dynamics. Unlike other existing
methods whose applicability is limited, our framework is global, and it has
only weak constraints over the inputs. Moreover, the Koopman operator is
linear, and it is related to a rich mathematical theory. Thus, we can use tools
and insights from linear analysis and Koopman Theory in our study. For
instance, we show that the operator eigendecomposition is instrumental in
exploring the dominant features of the network. Our results extend across tasks
and architectures as we demonstrate for the copy problem, and ECG
classification and sentiment analysis tasks.

    

### [[2102.09310] VAE Approximation Error: ELBO and Conditional Independence](http://arxiv.org/abs/2102.09310)


  The importance of Variational Autoencoders reaches far beyond standalone
generative models -- the approach is also used for learning latent
representations and can be generalized to semi-supervised learning. This
requires a thorough analysis of their commonly known shortcomings: posterior
collapse and approximation errors. This paper analyzes VAE approximation errors
caused by the combination of the ELBO objective with the choice of the encoder
probability family, in particular under conditional independence assumptions.
We identify the subclass of generative models consistent with the encoder
family. We show that the ELBO optimizer is pulled from the likelihood optimizer
towards this consistent subset. Furthermore, this subset can not be enlarged,
and the respective error cannot be decreased, by only considering deeper
encoder networks.

    

### [[2102.12695] Orbital dynamics of binary black hole systems can be learned from gravitational wave measurements](http://arxiv.org/abs/2102.12695)


  We introduce a gravitational waveform inversion strategy that discovers
mechanical models of binary black hole (BBH) systems. We show that only a
single time series of (possibly noisy) waveform data is necessary to construct
the equations of motion for a BBH system. Starting with a class of universal
differential equations parameterized by feed-forward neural networks, our
strategy involves the construction of a space of plausible mechanical models
and a physics-informed constrained optimization within that space to minimize
the waveform error. We apply our method to various BBH systems including
extreme and comparable mass ratio systems in eccentric and non-eccentric
orbits. We show the resulting differential equations apply to time durations
longer than the training interval, and relativistic effects, such as perihelion
precession, radiation reaction, and orbital plunge, are automatically accounted
for. The methods outlined here provide a new, data-driven approach to studying
the dynamics of binary black hole systems.

    

### [[2103.00737] Meta-Learning an Inference Algorithm for Probabilistic Programs](http://arxiv.org/abs/2103.00737)


  We present a meta-algorithm for learning a posterior-inference algorithm for
restricted probabilistic programs. Our meta-algorithm takes a training set of
probabilistic programs that describe models with observations, and attempts to
learn an efficient method for inferring the posterior of a similar program. A
key feature of our approach is the use of what we call a white-box inference
algorithm that extracts information directly from model descriptions
themselves, given as programs. Concretely, our white-box inference algorithm is
equipped with multiple neural networks, one for each type of atomic command,
and computes an approximate posterior of a given probabilistic program by
analysing individual atomic commands in the program using these networks. The
parameters of the networks are learnt from a training set by our
meta-algorithm. We empirically demonstrate that the learnt inference algorithm
generalises well to programs that are new in terms of both parameters and model
structures, and report cases where our approach achieves greater test-time
efficiency than alternative approaches such as HMC. The overall results show
the promise as well as remaining challenges of our approach.

    

### [[2103.02631] RotoGrad: Gradient Homogenization in Multitask Learning](http://arxiv.org/abs/2103.02631)


  Multitask learning is being increasingly adopted in applications domains like
computer vision and reinforcement learning. However, optimally exploiting its
advantages remains a major challenge due to the effect of negative transfer.
Previous works have tracked down this issue to the disparities in gradient
magnitudes and directions across tasks, when optimizing the shared network
parameters. While recent work has acknowledged that negative transfer is a
two-fold problem, existing approaches fall short as they only focus on either
homogenizing the gradient magnitude across tasks; or greedily change the
gradient directions, overlooking future conflicts. In this work, we introduce
RotoGrad, an algorithm that tackles negative transfer as a whole: it jointly
homogenizes gradient magnitudes and directions, while ensuring training
convergence. We show that RotoGrad outperforms competing methods in complex
problems, including multi-label classification in CelebA and computer vision
tasks in the NYUv2 dataset. A Pytorch implementation can be found in
this https URL .

    

### [[2103.04250] Greedy Approximation Algorithms for Active Sequential Hypothesis Testing](http://arxiv.org/abs/2103.04250)


  In the problem of active sequential hypothesis testing (ASHT), a learner
seeks to identify the true hypothesis from among a known set of hypotheses. The
learner is given a set of actions and knows the random distribution of the
outcome of any action under any true hypothesis. Given a target error
$\delta>0$, the goal is to sequentially select the fewest number of actions so
as to identify the true hypothesis with probability at least $1 - \delta$.
Motivated by applications in which the number of hypotheses or actions is
massive (e.g., genomics-based cancer detection), we propose efficient (greedy,
in fact) algorithms and provide the first approximation guarantees for ASHT,
under two types of adaptivity. Both of our guarantees are independent of the
number of actions and logarithmic in the number of hypotheses. We numerically
evaluate the performance of our algorithms using both synthetic and real-world
DNA mutation data, demonstrating that our algorithms outperform previously
proposed heuristic policies by large margins.

    

### [[2103.10796] CoordiNet: uncertainty-aware pose regressor for reliable vehicle localization](http://arxiv.org/abs/2103.10796)


  In this paper, we investigate visual-based camera re-localization with neural
networks for robotics and autonomous vehicles applications. Our solution is a
CNN-based algorithm which predicts camera pose (3D translation and 3D rotation)
directly from a single image. It also provides an uncertainty estimate of the
pose. Pose and uncertainty are learned together with a single loss function and
are fused at test time with an EKF. Furthermore, we propose a new fully
convolutional architecture, named CoordiNet, designed to embed some of the
scene geometry. Our framework outperforms comparable methods on the largest
available benchmark, the Oxford RobotCar dataset, with an average error of 8
meters where previous best was 19 meters. We have also investigated the
performance of our method on large scenes for real time (18 fps) vehicle
localization. In this setup, structure-based methods require a large database,
and we show that our proposal is a reliable alternative, achieving 29cm median
error in a 1.9km loop in a busy urban area

    

### [[2104.00322] Domain Invariant Adversarial Learning](http://arxiv.org/abs/2104.00322)


  The phenomenon of adversarial examples illustrates one of the most basic
vulnerabilities of deep neural networks. Among the variety of techniques
introduced to surmount this inherent weakness, adversarial training has emerged
as the most effective strategy to achieve robustness. Typically, this is
achieved by balancing robust and natural objectives. In this work, we aim to
further optimize the trade-off between robust and standard accuracy by
enforcing a domain-invariant feature representation. We present a new
adversarial training method, Domain Invariant Adversarial Learning (DIAL),
which learns a feature representation that is both robust and domain invariant.
DIAL uses a variant of Domain Adversarial Neural Network (DANN) on the natural
domain and its corresponding adversarial domain. In the case where the source
domain consists of natural examples and the target domain is the adversarially
perturbed examples, our method learns a feature representation constrained not
to discriminate between the natural and adversarial examples, and can therefore
achieve a more robust representation. Our experiments indicate that our method
improves both robustness and standard accuracy, when compared to other
state-of-the-art adversarial training methods.

    

### [[2104.02369] Classification with Runge-Kutta networks and feature space augmentation](http://arxiv.org/abs/2104.02369)


  In this paper we combine an approach based on Runge-Kutta Nets considered in
[Benning et al., J. Comput. Dynamics, 9, 2019] and a technique on augmenting
the input space in [Dupont et al., NeurIPS, 2019] to obtain network
architectures which show a better numerical performance for deep neural
networks in point and image classification problems. The approach is
illustrated with several examples implemented in PyTorch.

    

### [[2104.08171] Safe Exploration in Model-based Reinforcement Learning using Control Barrier Functions](http://arxiv.org/abs/2104.08171)


  This paper studies the problem of developing an approximate dynamic
programming (ADP) framework for learning online the value function of an
infinite-horizon optimal problem while obeying safety constraints expressed as
control barrier functions (CBFs). Our approach is facilitated by the
development of a novel class of CBFs, termed Lyapunov-like CBFs (LCBFs), that
retain the beneficial properties of CBFs for developing minimally-invasive safe
control policies while also possessing desirable Lyapunov-like qualities such
as positive semi-definiteness. We show how these LCBFs can be used to augment a
learning-based control policy so as to guarantee safety and then leverage this
approach to develop a safe exploration framework in a model-based reinforcement
learning setting. We demonstrate that our developed approach can handle more
general safety constraints than state-of-the-art safe ADP methods through a
variety of numerical examples.

    

### [[2105.00925] Hyperspherically Regularized Networks for BYOL Improves Feature Uniformity and Separability](http://arxiv.org/abs/2105.00925)


  Bootstrap Your Own Latent (BYOL) introduced an approach to self-supervised
learning avoiding the contrastive paradigm and subsequently removing the
computational burden of negative sampling. However, feature representations
under this paradigm are poorly distributed on the surface of the
unit-hypersphere representation space compared to contrastive methods. This
work empirically demonstrates that feature diversity enforced by contrastive
losses is beneficial when employed in BYOL, and as such, provides greater
inter-class feature separability. Therefore to achieve a more uniform
distribution of features, we advocate the minimization of hyperspherical energy
(i.e. maximization of entropy) in BYOL network weights. We show that directly
optimizing a measure of uniformity alongside the standard loss, or regularizing
the networks of the BYOL architecture to minimize the hyperspherical energy of
neurons can produce more uniformly distributed and better performing
representations for downstream tasks.

    

### [[2105.10832] Spectral Pruning for Recurrent Neural Networks](http://arxiv.org/abs/2105.10832)


  Recurrent neural networks (RNNs) are a class of neural networks used in
sequential tasks. However, in general, RNNs have a large number of parameters
and involve enormous computational costs by repeating the recurrent structures
in many time steps. As a method to overcome this difficulty, RNN pruning has
attracted increasing attention in recent years, and it brings us benefits in
terms of the reduction of computational cost as the time step progresses.
However, most existing methods of RNN pruning are heuristic. The purpose of
this paper is to study the theoretical scheme for RNN pruning method. We
propose an appropriate pruning algorithm for RNNs inspired by "spectral
pruning", and provide the generalization error bounds for compressed RNNs. We
also provide numerical experiments to demonstrate our theoretical results and
show the effectiveness of our pruning method compared with existing methods.

    

### [[2106.01487] LLC: Accurate, Multi-purpose Learnt Low-dimensional Binary Codes](http://arxiv.org/abs/2106.01487)


  Learning binary representations of instances and classes is a classical
problem with several high potential applications. In modern settings, the
compression of high-dimensional neural representations to low-dimensional
binary codes is a challenging task and often require large bit-codes to be
accurate. In this work, we propose a novel method for Learning Low-dimensional
binary Codes (LLC) for instances as well as classes. Our method does not
require any side-information, like annotated attributes or label meta-data, and
learns extremely low-dimensional binary codes (~20 bits for ImageNet-1K). The
learnt codes are super-efficient while still ensuring nearly optimal
classification accuracy for ResNet50 on ImageNet-1K. We demonstrate that the
learnt codes capture intrinsically important features in the data, by
discovering an intuitive taxonomy over classes. We further quantitatively
measure the quality of our codes by applying it to the efficient image
retrieval as well as out-of-distribution (OOD) detection problems. For
ImageNet-100 retrieval problem, our learnt binary codes outperform 16 bit
HashNet using only 10 bits and also are as accurate as 10 dimensional real
representations. Finally, our learnt binary codes can perform OOD detection,
out-of-the-box, as accurately as a baseline that needs ~3000 samples to tune
its threshold, while we require none. Code is open-sourced at
this https URL.

    

### [[2106.02914] Feature Flow Regularization: Improving Structured Sparsity in Deep Neural Networks](http://arxiv.org/abs/2106.02914)


  Pruning is a model compression method that removes redundant parameters in
deep neural networks (DNNs) while maintaining accuracy. Most available filter
pruning methods require complex treatments such as iterative pruning, features
statistics/ranking, or additional optimization designs in the training process.
In this paper, we propose a simple and effective regularization strategy from a
new perspective of evolution of features, which we call feature flow
regularization (FFR), for improving structured sparsity and filter pruning in
DNNs. Specifically, FFR imposes controls on the gradient and curvature of
feature flow along the neural network, which implicitly increases the sparsity
of the parameters. The principle behind FFR is that coherent and smooth
evolution of features will lead to an efficient network that avoids redundant
parameters. The high structured sparsity obtained from FFR enables us to prune
filters effectively. Experiments with VGGNets, ResNets on CIFAR-10/100, and
Tiny ImageNet datasets demonstrate that FFR can significantly improve both
unstructured and structured sparsity. Our pruning results in terms of reduction
of parameters and FLOPs are comparable to or even better than those of
state-of-the-art pruning methods.

    

### [[2106.03156] Fast and Robust Online Inference with Stochastic Gradient Descent via Random Scaling](http://arxiv.org/abs/2106.03156)


  We develop a new method of online inference for a vector of parameters
estimated by the Polyak-Ruppert averaging procedure of stochastic gradient
descent (SGD) algorithms. We leverage insights from time series regression in
econometrics and construct asymptotically pivotal statistics via random
scaling. Our approach is fully operational with online data and is rigorously
underpinned by a functional central limit theorem. Our proposed inference
method has a couple of key advantages over the existing methods. First, the
test statistic is computed in an online fashion with only SGD iterates and the
critical values can be obtained without any resampling methods, thereby
allowing for efficient implementation suitable for massive online data. Second,
there is no need to estimate the asymptotic variance and our inference method
is shown to be robust to changes in the tuning parameters for SGD algorithms in
simulation experiments with synthetic data.

    

### [[2106.03157] Self-Supervision is All You Need for Solving Rubik's Cube](http://arxiv.org/abs/2106.03157)


  While combinatorial problems are of great academic and practical importance,
previous approaches like explicit heuristics and reinforcement learning have
been complex and costly. To address this, we developed a simple and robust
method to train a Deep Neural Network (DNN) through self-supervised learning
for solving a goal-predefined combinatorial problem. Assuming that more optimal
moves occur more frequently as a path of random moves connecting two problem
states, the DNN can approximate an optimal solver by learning to predict the
last move of a random scramble based on the problem state. Tested on 1,000
scrambled Rubik's Cube instances, a Transformer-based model could solve all of
them near-optimally using a breadth-first search; with a maximum breadth of
$10^3$, the mean solution length was $20.5$ moves. The proposed method may
apply to other goal-predefined combinatorial problems, though it has a few
constraints.

    

### [[2106.06127] Differentially Private Federated Learning via Inexact ADMM](http://arxiv.org/abs/2106.06127)


  Differential privacy (DP) techniques can be applied to the federated learning
model to protect data privacy against inference attacks to communication among
the learning agents. The DP techniques, however, hinder achieving a greater
learning performance while ensuring strong data privacy. In this paper we
develop a DP inexact alternating direction method of multipliers algorithm that
solves a sequence of subproblems with the objective perturbation by random
noises generated from a Laplace distribution. We show that our algorithm
provides $\bar{\epsilon}$-DP for every iteration, where $\bar{\epsilon}$ is a
privacy parameter controlled by a user. Using MNIST and FEMNIST datasets for
the image classification, we demonstrate that our algorithm reduces the testing
error by at most $22\%$ compared with the existing DP algorithm, while
achieving the same level of data privacy. The numerical experiment also shows
that our algorithm converges faster than the existing algorithm.

    

### [[2106.06210] Learning to Pool in Graph Neural Networks for Extrapolation](http://arxiv.org/abs/2106.06210)


  Graph neural networks (GNNs) are one of the most popular approaches to using
deep learning on graph-structured data, and they have shown state-of-the-art
performances on a variety of tasks. However, according to a recent study, a
careful choice of pooling functions, which are used for the aggregation and
readout operations in GNNs, is crucial for enabling GNNs to extrapolate.
Without proper choices of pooling functions, which varies across tasks, GNNs
completely fail to generalize to out-of-distribution data, while the number of
possible choices grows exponentially with the number of layers. In this paper,
we present GNP, a $L^p$ norm-like pooling function that is trainable end-to-end
for any given task. Notably, GNP generalizes most of the widely-used pooling
functions. We verify experimentally that simply using GNP for every aggregation
and readout operation enables GNNs to extrapolate well on many node-level,
graph-level, and set-related tasks; and GNP sometimes performs even better than
the best-performing choices among existing pooling functions.

    

### [[2106.08417] Scene Transformer: A unified architecture for predicting multiple agent trajectories](http://arxiv.org/abs/2106.08417)


  Predicting the motion of multiple agents is necessary for planning in dynamic
environments. This task is challenging for autonomous driving since agents
(e.g. vehicles and pedestrians) and their associated behaviors may be diverse
and influence one another. Most prior work have focused on predicting
independent futures for each agent based on all past motion, and planning
against these independent predictions. However, planning against independent
predictions can make it challenging to represent the future interaction
possibilities between different agents, leading to sub-optimal planning. In
this work, we formulate a model for predicting the behavior of all agents
jointly, producing consistent futures that account for interactions between
agents. Inspired by recent language modeling approaches, we use a masking
strategy as the query to our model, enabling one to invoke a single model to
predict agent behavior in many ways, such as potentially conditioned on the
goal or full future trajectory of the autonomous vehicle or the behavior of
other agents in the environment. Our model architecture employs attention to
combine features across road elements, agent interactions, and time steps. We
evaluate our approach on autonomous driving datasets for both marginal and
joint motion prediction, and achieve state of the art performance across two
popular datasets. Through combining a scene-centric approach, agent permutation
equivariant model, and a sequence masking strategy, we show that our model can
unify a variety of motion prediction tasks from joint motion predictions to
conditioned prediction.

    

### [[2106.08928] Recursive Construction of Stable Assemblies of Recurrent Neural Networks](http://arxiv.org/abs/2106.08928)


  Advanced applications of modern machine learning will likely involve
combinations of trained networks, as are already used in spectacular systems
such as DeepMind's AlphaGo. Recursively building such combinations in an
effective and stable fashion while also allowing for continual refinement of
the individual networks - as nature does for biological networks - will require
new analysis tools. This paper takes a step in this direction by establishing
contraction properties of broad classes of nonlinear recurrent networks and
neural ODEs, and showing how these quantified properties allow in turn to
recursively construct stable networks of networks in a systematic fashion. The
results can also be used to stably combine recurrent networks and physical
systems with quantified contraction properties. Similarly, they may be applied
to modular computational models of cognition. We perform experiments with these
combined networks on benchmark sequential tasks (e.g permuted sequential MNIST)
to demonstrate their capacity for processing information across a long
timescale in a provably stable manner.

    

### [[2106.11257] Secure Distributed Training at Scale](http://arxiv.org/abs/2106.11257)


  Some of the hardest problems in deep learning can be solved via pooling
together computational resources of many independent parties, as is the case
for scientific collaborations and volunteer computing. Unfortunately, any
single participant in such systems can jeopardize the entire training run by
sending incorrect updates, whether deliberately or by mistake. Training in
presence of such peers requires specialized distributed training algorithms
with Byzantine tolerance. These algorithms often sacrifice efficiency by
introducing redundant communication or passing all updates through a trusted
server. As a result, it can be infeasible to apply such algorithms to
large-scale distributed deep learning, where models can have billions of
parameters. In this work, we propose a novel protocol for secure
(Byzantine-tolerant) decentralized training that emphasizes communication
efficiency. We rigorously analyze this protocol: in particular, we provide
theoretical bounds for its resistance against Byzantine and Sybil attacks and
show that it has a marginal communication overhead. To demonstrate its
practical effectiveness, we conduct large-scale experiments on image
classification and language modeling in presence of Byzantine attackers.

    

### [[2110.01467] HyperTeNet: Hypergraph and Transformer-based Neural Network for Personalized List Continuation](http://arxiv.org/abs/2110.01467)


  The personalized list continuation (PLC) task is to curate the next items to
user-generated lists (ordered sequence of items) in a personalized way. The
main challenge in this task is understanding the ternary relationships among
the interacting entities (users, items, and lists) that the existing works do
not consider. Further, they do not take into account the multi-hop
relationships among entities of the same type. In addition, capturing the
sequential information amongst the items already present in the list also plays
a vital role in determining the next relevant items that get curated.
In this work, we propose HyperTeNet -- a self-attention hypergraph and
Transformer-based neural network architecture for the personalized list
continuation task to address the challenges mentioned above. We use graph
convolutions to learn the multi-hop relationship among the entities of the same
type and leverage a self-attention-based hypergraph neural network to learn the
ternary relationships among the interacting entities via hyperlink prediction
in a 3-uniform hypergraph. Further, the entity embeddings are shared with a
Transformer-based architecture and are learned through an alternating
optimization procedure. As a result, this network also learns the sequential
information needed to curate the next items to be added to the list.
Experimental results demonstrate that HyperTeNet significantly outperforms the
other state-of-the-art models on real-world datasets. Our implementation and
datasets are available at this https URL.

    

### [[2101.01673] Characterizing Intersectional Group Fairness with Worst-Case Comparisons](http://arxiv.org/abs/2101.01673)


  Machine Learning or Artificial Intelligence algorithms have gained
considerable scrutiny in recent times owing to their propensity towards
imitating and amplifying existing prejudices in society. This has led to a
niche but growing body of work that identifies and attempts to fix these
biases. A first step towards making these algorithms more fair is designing
metrics that measure unfairness. Most existing work in this field deals with
either a binary view of fairness (protected vs. unprotected groups) or
politically defined categories (race or gender). Such categorization misses the
important nuance of intersectionality - biases can often be amplified in
subgroups that combine membership from different categories, especially if such
a subgroup is particularly underrepresented in historical platforms of
opportunity.
In this paper, we discuss why fairness metrics need to be looked at under the
lens of intersectionality, identify existing work in intersectional fairness,
suggest a simple worst case comparison method to expand the definitions of
existing group fairness metrics to incorporate intersectionality, and finally
conclude with the social, legal and political framework to handle
intersectional fairness in the modern context.

    

### [[2110.03214] MAPA: Multi-Accelerator Pattern Allocation Policy for Multi-Tenant GPU Servers](http://arxiv.org/abs/2110.03214)


  Multi-accelerator servers are increasingly being deployed in shared
multi-tenant environments (such as in cloud data centers) in order to meet the
demands of large-scale compute-intensive workloads. In addition, these
accelerators are increasingly being inter-connected in complex topologies and
workloads are exhibiting a wider variety of inter-accelerator communication
patterns. However, existing allocation policies are ill-suited for these
emerging use-cases. Specifically, this work identifies that multi-accelerator
workloads are commonly fragmented leading to reduced bandwidth and increased
latency for inter-accelerator communication. We propose Multi-Accelerator
Pattern Allocation (MAPA), a graph pattern mining approach towards providing
generalized allocation support for allocating multi-accelerator workloads on
multi-accelerator servers. We demonstrate that MAPA is able to improve the
execution time of multi-accelerator workloads and that MAPA is able to provide
generalized benefits across various accelerator topologies. Finally, we
demonstrate a speedup of 12.4% for 75th percentile of jobs with the worst case
execution time reduced by up to 35% against baseline policy using MAPA.

    

### [[2110.03229] IaaS Signature Change Detection with Performance Noise](http://arxiv.org/abs/2110.03229)


  We propose a novel framework to detect changes in the performance behavior of
an IaaS service. The proposed framework leverages the concept of the IaaS
signature to represent an IaaS service's long-term performance behavior. A new
type of performance signature called categorical IaaS signature is introduced
to represent the performance behavior more accurately. A novel performance
noise model is proposed to accurately identify IaaS performance noise and
accurate changes in the performance behavior of an IaaS service. A set of
experiments based on real-world datasets is carried out to evaluate the
effectiveness of the proposed framework.

    

### [[2110.03636] A Hybrid Direct-Iterative Method for Solving KKT Linear Systems](http://arxiv.org/abs/2110.03636)


  We propose a solution strategy for linear systems arising in interior method
optimization, which is suitable for implementation on hardware accelerators
such as graphical processing units (GPUs). The current gold standard for
solving these systems is the LDL^T factorization. However, LDL^T requires
pivoting during factorization, which substantially increases communication cost
and degrades performance on GPUs. Our novel approach solves a large indefinite
system by solving multiple smaller positive definite systems, using an
iterative solve for the Schur complement and an inner direct solve (via
Cholesky factorization) within each iteration. Cholesky is stable without
pivoting, thereby reducing communication and allowing reuse of the symbolic
factorization. We demonstrate the practicality of our approach and show that on
large systems it can efficiently utilize GPUs and outperform LDL^T
factorization of the full system.

    

### [[2110.03001] Predictability and Fairness in Load Aggregation and Operations of Virtual Power Plants](http://arxiv.org/abs/2110.03001)


  In power systems, one wishes to regulate the aggregate demand of an ensemble
of distributed energy resources (DERs), such as controllable loads and battery
energy storage systems. We suggest a notion of predictability and fairness,
which suggests that the long-term averages of prices or incentives offered
should be independent of the initial states of the operators of the DER, the
aggregator, and the power grid. We show that this notion cannot be guaranteed
with many traditional controllers used by the load aggregator, including the
usual proportional-integral (PI) controller. We show that even considering the
non-linearity of the alternating-current model, this notion of predictability
and fairness can be guaranteed for incrementally input-to-state stable (iISS)
controllers, under mild assumptions.

    

### [[2110.03024] A Fast Randomized Algorithm for Massive Text Normalization](http://arxiv.org/abs/2110.03024)


  Many popular machine learning techniques in natural language processing and
data mining rely heavily on high-quality text sources. However real-world text
datasets contain a significant amount of spelling errors and improperly
punctuated variants where the performance of these models would quickly
deteriorate. Moreover, real-world, web-scale datasets contain hundreds of
millions or even billions of lines of text, where the existing text cleaning
tools are prohibitively expensive to execute over and may require an overhead
to learn the corrections. In this paper, we present FLAN, a scalable randomized
algorithm to clean and canonicalize massive text data. Our algorithm relies on
the Jaccard similarity between words to suggest correction results. We
efficiently handle the pairwise word-to-word comparisons via Locality Sensitive
Hashing (LSH). We also propose a novel stabilization process to address the
issue of hash collisions between dissimilar words, which is a consequence of
the randomized nature of LSH and is exacerbated by the massive scale of
real-world datasets. Compared with existing approaches, our method is more
efficient, both asymptotically and in empirical evaluations, and does not rely
on additional features, such as lexical/phonetic similarity or word embedding
features. In addition, FLAN does not require any annotated data or supervised
learning. We further theoretically show the robustness of our algorithm with
upper bounds on the false positive and false negative rates of corrections. Our
experimental results on real-world datasets demonstrate the efficiency and
efficacy of FLAN.

    

### [[2110.03036] The Low-Resource Double Bind: An Empirical Study of Pruning for Low-Resource Machine Translation](http://arxiv.org/abs/2110.03036)


  A "bigger is better" explosion in the number of parameters in deep neural
networks has made it increasingly challenging to make state-of-the-art networks
accessible in compute-restricted environments. Compression techniques have
taken on renewed importance as a way to bridge the gap. However, evaluation of
the trade-offs incurred by popular compression techniques has been centered on
high-resource datasets. In this work, we instead consider the impact of
compression in a data-limited regime. We introduce the term low-resource double
bind to refer to the co-occurrence of data limitations and compute resource
constraints. This is a common setting for NLP for low-resource languages, yet
the trade-offs in performance are poorly studied. Our work offers surprising
insights into the relationship between capacity and generalization in
data-limited regimes for the task of machine translation. Our experiments on
magnitude pruning for translations from English into Yoruba, Hausa, Igbo and
German show that in low-resource regimes, sparsity preserves performance on
frequent sentences but has a disparate impact on infrequent ones. However, it
improves robustness to out-of-distribution shifts, especially for datasets that
are very distinct from the training distribution. Our findings suggest that
sparsity can play a beneficial role at curbing memorization of low frequency
attributes, and therefore offers a promising solution to the low-resource
double bind.

    

### [[2110.03054] On The Vulnerability of Recurrent Neural Networks to Membership Inference Attacks](http://arxiv.org/abs/2110.03054)


  We study the privacy implications of deploying recurrent neural networks in
machine learning. We consider membership inference attacks (MIAs) in which an
attacker aims to infer whether a given data record has been used in the
training of a learning agent. Using existing MIAs that target feed-forward
neural networks, we empirically demonstrate that the attack accuracy wanes for
data records used earlier in the training history. Alternatively, recurrent
networks are specifically designed to better remember their past experience;
hence, they are likely to be more vulnerable to MIAs than their feed-forward
counterparts. We develop a pair of MIA layouts for two primary applications of
recurrent networks, namely, deep reinforcement learning and
sequence-to-sequence tasks. We use the first attack to provide empirical
evidence that recurrent networks are indeed more vulnerable to MIAs than
feed-forward networks with the same performance level. We use the second attack
to showcase the differences between the effects of overtraining recurrent and
feed-forward networks on the accuracy of their respective MIAs. Finally, we
deploy a differential privacy mechanism to resolve the privacy vulnerability
that the MIAs exploit. For both attack layouts, the privacy mechanism degrades
the attack accuracy from above 80% to 50%, which is equal to guessing the data
membership uniformly at random, while trading off less than 10% utility.

    

### [[2110.03105] Learning a Metacognition for Object Detection](http://arxiv.org/abs/2110.03105)


  In contrast to object recognition models, humans do not blindly trust their
perception when building representations of the world, instead recruiting
metacognition to detect percepts that are unreliable or false, such as when we
realize that we mistook one object for another. We propose METAGEN, an
unsupervised model that enhances object recognition models through a
metacognition. Given noisy output from an object-detection model, METAGEN
learns a meta-representation of how its perceptual system works and uses it to
infer the objects in the world responsible for the detections. METAGEN achieves
this by conditioning its inference on basic principles of objects that even
human infants understand (known as Spelke principles: object permanence,
cohesion, and spatiotemporal continuity). We test METAGEN on a variety of
state-of-the-art object detection neural networks. We find that METAGEN quickly
learns an accurate metacognitive representation of the neural network, and that
this improves detection accuracy by filling in objects that the detection model
missed and removing hallucinated objects. This approach enables generalization
to out-of-sample data and outperforms comparison models that lack a
metacognition.

    

### [[2110.03141] Efficient Sharpness-aware Minimization for Improved Training of Neural Networks](http://arxiv.org/abs/2110.03141)


  Overparametrized Deep Neural Networks (DNNs) often achieve astounding
performances, but may potentially result in severe generalization error.
Recently, the relation between the sharpness of the loss landscape and the
generalization error has been established by Foret et al. (2020), in which the
Sharpness Aware Minimizer (SAM) was proposed to mitigate the degradation of the
generalization. Unfortunately, SAM s computational cost is roughly double that
of base optimizers, such as Stochastic Gradient Descent (SGD). This paper thus
proposes Efficient Sharpness Aware Minimizer (ESAM), which boosts SAM s
efficiency at no cost to its generalization performance. ESAM includes two
novel and efficient training strategies-StochasticWeight Perturbation and
Sharpness-Sensitive Data Selection. In the former, the sharpness measure is
approximated by perturbing a stochastically chosen set of weights in each
iteration; in the latter, the SAM loss is optimized using only a judiciously
selected subset of data that is sensitive to the sharpness. We provide
theoretical explanations as to why these strategies perform well. We also show,
via extensive experiments on the CIFAR and ImageNet datasets, that ESAM
enhances the efficiency over SAM from requiring 100% extra computations to 40%
vis-a-vis base optimizers, while test accuracies are preserved or even
improved.

    

### [[2110.03154] DoubleStar: Long-Range Attack Towards Depth Estimation based Obstacle Avoidance in Autonomous Systems](http://arxiv.org/abs/2110.03154)


  Depth estimation-based obstacle avoidance has been widely adopted by
autonomous systems (drones and vehicles) for safety purpose. It normally relies
on a stereo camera to automatically detect obstacles and make flying/driving
decisions, e.g., stopping several meters ahead of the obstacle in the path or
moving away from the detected obstacle. In this paper, we explore new security
risks associated with the stereo vision-based depth estimation algorithms used
for obstacle avoidance. By exploiting the weaknesses of the stereo matching in
depth estimation algorithms and the lens flare effect in optical imaging, we
propose DoubleStar, a long-range attack that injects fake obstacle depth by
projecting pure light from two complementary light sources.
DoubleStar includes two distinctive attack formats: beams attack and orbs
attack, which leverage projected light beams and lens flare orbs respectively
to cause false depth perception. We successfully attack two commercial stereo
cameras designed for autonomous systems (ZED and Intel RealSense). The
visualization of fake depth perceived by the stereo cameras illustrates the
false stereo matching induced by DoubleStar. We further use Ardupilot to
simulate the attack and demonstrate its impact on drones. To validate the
attack on real systems, we perform a real-world attack towards a commercial
drone equipped with state-of-the-art obstacle avoidance algorithms. Our attack
can continuously bring a flying drone to a sudden stop or drift it away across
a long distance under various lighting conditions, even bypassing sensor fusion
mechanisms. Specifically, our experimental results show that DoubleStar creates
fake depth up to 15 meters in distance at night and up to 8 meters during the
daytime. To mitigate this newly discovered threat, we provide discussions on
potential countermeasures to defend against DoubleStar.

    

### [[2110.03156] StrengthNet: Deep Learning-based Emotion Strength Assessment for Emotional Speech Synthesis](http://arxiv.org/abs/2110.03156)


  Recently, emotional speech synthesis has achieved remarkable performance.
Furthermore, the emotion strength of synthesized speech can be controlled
flexibly using a strength descriptor, which is obtained by an emotion attribute
ranking function. However, a trained ranking function on specific data has poor
generalization, which limit its applicability for more realistic cases. In this
paper, we propose a deep learning based emotion strength assessment network for
strength prediction that is referred to as StrengthNet. Our model conforms to a
multi-task learning framework with a structure that includes an acoustic
encoder, a strength predictor and an auxiliary emotion predictor. A data
augmentation strategy was utilized to improve the model generalization.
Experiments show that the predicted emotion strength of the proposed
StrengthNet are highly correlated with ground truth scores for seen and unseen
speech. Our codes are available at: this https URL.

    

### [[2110.03174] Transferring Voice Knowledge for Acoustic Event Detection: An Empirical Study](http://arxiv.org/abs/2110.03174)


  Detection of common events and scenes from audio is useful for extracting and
understanding human contexts in daily life. Prior studies have shown that
leveraging knowledge from a relevant domain is beneficial for a target acoustic
event detection (AED) process. Inspired by the observation that many
human-centered acoustic events in daily life involve voice elements, this paper
investigates the potential of transferring high-level voice representations
extracted from a public speaker dataset to enrich an AED pipeline. Towards this
end, we develop a dual-branch neural network architecture for the joint
learning of voice and acoustic features during an AED process and conduct
thorough empirical studies to examine the performance on the public AudioSet
[1] with different types of inputs. Our main observations are that: 1) Joint
learning of audio and voice inputs improves the AED performance (mean average
precision) for both a CNN baseline (0.292 vs 0.134 mAP) and a TALNet [2]
baseline (0.361 vs 0.351 mAP); 2) Augmenting the extra voice features is
critical to maximize the model performance with dual inputs.

    

### [[2110.03175] Fingerprinting Multi-exit Deep Neural Network Models via Inference Time](http://arxiv.org/abs/2110.03175)


  Transforming large deep neural network (DNN) models into the multi-exit
architectures can overcome the overthinking issue and distribute a large DNN
model on resource-constrained scenarios (e.g. IoT frontend devices and backend
servers) for inference and transmission efficiency. Nevertheless, intellectual
property (IP) protection for the multi-exit models in the wild is still an
unsolved challenge. Previous efforts to verify DNN model ownership mainly rely
on querying the model with specific samples and checking the responses, e.g.,
DNN watermarking and fingerprinting. However, they are vulnerable to
adversarial settings such as adversarial training and are not suitable for the
IP verification for multi-exit DNN models. In this paper, we propose a novel
approach to fingerprint multi-exit models via inference time rather than
inference predictions. Specifically, we design an effective method to generate
a set of fingerprint samples to craft the inference process with a unique and
robust inference time cost as the evidence for model ownership. We conduct
extensive experiments to prove the uniqueness and robustness of our method on
three structures (ResNet-56, VGG-16, and MobileNet) and three datasets
(CIFAR-10, CIFAR-100, and Tiny-ImageNet) under comprehensive adversarial
settings.

    

### [[2110.03192] GNN is a Counter? Revisiting GNN for Question Answering](http://arxiv.org/abs/2110.03192)


  Question Answering (QA) has been a long-standing research topic in AI and NLP
fields, and a wealth of studies have been conducted to attempt to equip QA
systems with human-level reasoning capability. To approximate the complicated
human reasoning process, state-of-the-art QA systems commonly use pre-trained
language models (LMs) to access knowledge encoded in LMs together with
elaborately designed modules based on Graph Neural Networks (GNNs) to perform
reasoning over knowledge graphs (KGs). However, many problems remain open
regarding the reasoning functionality of these GNN-based modules. Can these
GNN-based modules really perform a complex reasoning process? Are they under-
or over-complicated for QA? To open the black box of GNN and investigate these
problems, we dissect state-of-the-art GNN modules for QA and analyze their
reasoning capability. We discover that even a very simple graph neural counter
can outperform all the existing GNN modules on CommonsenseQA and OpenBookQA,
two popular QA benchmark datasets which heavily rely on knowledge-aware
reasoning. Our work reveals that existing knowledge-aware GNN modules may only
carry out some simple reasoning such as counting. It remains a challenging open
problem to build comprehensive reasoning modules for knowledge-powered QA.

    

### [[2110.03223] Goal-Directed Design Agents: Integrating Visual Imitation with One-Step Lookahead Optimization for Generative Design](http://arxiv.org/abs/2110.03223)


  Engineering design problems often involve large state and action spaces along
with highly sparse rewards. Since an exhaustive search of those spaces is not
feasible, humans utilize relevant domain knowledge to condense the search
space. Previously, deep learning agents (DLAgents) were introduced to use
visual imitation learning to model design domain knowledge. This note builds on
DLAgents and integrates them with one-step lookahead search to develop
goal-directed agents capable of enhancing learned strategies for sequentially
generating designs. Goal-directed DLAgents can employ human strategies learned
from data along with optimizing an objective function. The visual imitation
network from DLAgents is composed of a convolutional encoder-decoder network,
acting as a rough planning step that is agnostic to feedback. Meanwhile, the
lookahead search identifies the fine-tuned design action guided by an
objective. These design agents are trained on an unconstrained truss design
problem that is modeled as a sequential, action-based configuration design
problem. The agents are then evaluated on two versions of the problem: the
original version used for training and an unseen constrained version with an
obstructed construction space. The goal-directed agents outperform the human
designers used to train the network as well as the previous objective-agnostic
versions of the agent in both scenarios. This illustrates a design agent
framework that can efficiently use feedback to not only enhance learned design
strategies but also adapt to unseen design problems.

    

### [[2110.03262] Situated Dialogue Learning through Procedural Environment Generation](http://arxiv.org/abs/2110.03262)


  We teach goal-driven agents to interactively act and speak in situated
environments by training on generated curriculums. Our agents operate in LIGHT
(Urbanek et al. 2019) -- a large-scale crowd-sourced fantasy text adventure
game wherein an agent perceives and interacts with the world through textual
natural language. Goals in this environment take the form of character-based
quests, consisting of personas and motivations. We augment LIGHT by learning to
procedurally generate additional novel textual worlds and quests to create a
curriculum of steadily increasing difficulty for training agents to achieve
such goals. In particular, we measure curriculum difficulty in terms of the
rarity of the quest in the original training distribution -- an easier
environment is one that is more likely to have been found in the unaugmented
dataset. An ablation study shows that this method of learning from the tail of
a distribution results in significantly higher generalization abilities as
measured by zero-shot performance on never-before-seen quests.

    

### [[2110.03276] Inferring Substitutable and Complementary Products with Knowledge-Aware Path Reasoning based on Dynamic Policy Network](http://arxiv.org/abs/2110.03276)


  Inferring the substitutable and complementary products for a given product is
an essential and fundamental concern for the recommender system. To achieve
this, existing approaches take advantage of the knowledge graphs to learn more
evidences for inference, whereas they often suffer from invalid reasoning for
lack of elegant decision making strategies. Therefore, we propose a novel
Knowledge-Aware Path Reasoning (KAPR) model which leverages the dynamic policy
network to make explicit reasoning over knowledge graphs, for inferring the
substitutable and complementary relationships. Our contributions can be
highlighted as three aspects. Firstly, we model this inference scenario as a
Markov Decision Process in order to accomplish a knowledge-aware path reasoning
over knowledge graphs. Secondly,we integrate both structured and unstructured
knowledge to provide adequate evidences for making accurate decision-making.
Thirdly, we evaluate our model on a series of real-world datasets, achieving
competitive performance compared with state-of-the-art approaches. Our code is
released on this https URL flower/kapr/tree/master.

    

### [[2110.03278] Virtual Multi-Modality Self-Supervised Foreground Matting for Human-Object Interaction](http://arxiv.org/abs/2110.03278)


  Most existing human matting algorithms tried to separate pure human-only
foreground from the background. In this paper, we propose a Virtual
Multi-modality Foreground Matting (VMFM) method to learn human-object
interactive foreground (human and objects interacted with him or her) from a
raw RGB image. The VMFM method requires no additional inputs, e.g. trimap or
known background. We reformulate foreground matting as a self-supervised
multi-modality problem: factor each input image into estimated depth map,
segmentation mask, and interaction heatmap using three auto-encoders. In order
to fully utilize the characteristics of each modality, we first train a dual
encoder-to-decoder network to estimate the same alpha matte. Then we introduce
a self-supervised method: Complementary Learning(CL) to predict deviation
probability map and exchange reliable gradients across modalities without
label. We conducted extensive experiments to analyze the effectiveness of each
modality and the significance of different components in complementary
learning. We demonstrate that our model outperforms the state-of-the-art
methods.

    

### [[2110.03320] Automated Testing of AI Models](http://arxiv.org/abs/2110.03320)


  The last decade has seen tremendous progress in AI technology and
applications. With such widespread adoption, ensuring the reliability of the AI
models is crucial. In past, we took the first step of creating a testing
framework called AITEST for metamorphic properties such as fairness, robustness
properties for tabular, time-series, and text classification models. In this
paper, we extend the capability of the AITEST tool to include the testing
techniques for Image and Speech-to-text models along with interpretability
testing for tabular models. These novel extensions make AITEST a comprehensive
framework for testing AI models.

    

### [[2110.03346] MSHCNet: Multi-Stream Hybridized Convolutional Networks with Mixed Statistics in Euclidean/Non-Euclidean Spaces and Its Application to Hyperspectral Image Classification](http://arxiv.org/abs/2110.03346)


  It is well known that hyperspectral images (HSI) contain rich
spatial-spectral contextual information, and how to effectively combine both
spectral and spatial information using DNN for HSI classification has become a
new research hotspot. Compared with CNN with square kernels, GCN have exhibited
exciting potential to model spatial contextual structure and conduct flexible
convolution on arbitrarily irregular image regions. However, current GCN only
using first-order spectral-spatial signatures can result in boundary blurring
and isolated misclassification. To address these, we first designed the
graph-based second-order pooling (GSOP) operation to obtain contextual nodes
information in non-Euclidean space for GCN. Further, we proposed a novel
multi-stream hybridized convolutional network (MSHCNet) with combination of
first and second order statistics in Euclidean/non-Euclidean spaces to learn
and fuse multi-view complementary information to segment HSIs. Specifically,
our MSHCNet adopted four parallel streams, which contained G-stream, utilizing
the irregular correlation between adjacent land covers in terms of first-order
graph in non-Euclidean space; C-stream, adopting convolution operator to learn
regular spatial-spectral features in Euclidean space; N-stream, combining first
and second order features to learn representative and discriminative regular
spatial-spectral features of Euclidean space; S-stream, using GSOP to capture
boundary correlations and obtain graph representations from all nodes in graphs
of non-Euclidean space. Besides, these feature representations learned from
four different streams were fused to integrate the multi-view complementary
information for HSI classification. Finally, we evaluated our proposed MSHCNet
on three hyperspectral datasets, and experimental results demonstrated that our
method significantly outperformed state-of-the-art eight methods.

    

### [[2110.03389] Beam Search with Bidirectional Strategies for Neural Response Generation](http://arxiv.org/abs/2110.03389)


  Sequence-to-sequence neural networks have been widely used in language-based
applications as they have flexible capabilities to learn various language
models. However, when seeking for the optimal language response through trained
neural networks, current existing approaches such as beam-search decoder
strategies are still not able reaching to promising performances. Instead of
developing various decoder strategies based on a "regular sentence order"
neural network (a trained model by outputting sentences from left-to-right
order), we leveraged "reverse" order as additional language model (a trained
model by outputting sentences from right-to-left order) which can provide
different perspectives for the path finding problems. In this paper, we propose
bidirectional strategies in searching paths by combining two networks
(left-to-right and right-to-left language models) making a bidirectional beam
search possible. Besides, our solution allows us using any similarity measure
in our sentence selection criterion. Our approaches demonstrate better
performance compared to the unidirectional beam search strategy.

    

### [[2110.03390] GANtron: Emotional Speech Synthesis with Generative Adversarial Networks](http://arxiv.org/abs/2110.03390)


  Speech synthesis is used in a wide variety of industries. Nonetheless, it
always sounds flat or robotic. The state of the art methods that allow for
prosody control are very cumbersome to use and do not allow easy tuning. To
tackle some of these drawbacks, in this work we target the implementation of a
text-to-speech model where the inferred speech can be tuned with the desired
emotions. To do so, we use Generative Adversarial Networks (GANs) together with
a sequence-to-sequence model using an attention mechanism. We evaluate four
different configurations considering different inputs and training strategies,
study them and prove how our best model can generate speech files that lie in
the same distribution as the initial training dataset. Additionally, a new
strategy to boost the training convergence by applying a guided attention loss
is proposed.

    

### [[2110.03395] SLASH: Embracing Probabilistic Circuits into Neural Answer Set Programming](http://arxiv.org/abs/2110.03395)


  The goal of combining the robustness of neural networks and the expressivity
of symbolic methods has rekindled the interest in neuro-symbolic AI. Recent
advancements in neuro-symbolic AI often consider specifically-tailored
architectures consisting of disjoint neural and symbolic components, and thus
do not exhibit desired gains that can be achieved by integrating them into a
unifying framework. We introduce SLASH -- a novel deep probabilistic
programming language (DPPL). At its core, SLASH consists of
Neural-Probabilistic Predicates (NPPs) and logical programs which are united
via answer set programming. The probability estimates resulting from NPPs act
as the binding element between the logical program and raw input data, thereby
allowing SLASH to answer task-dependent logical queries. This allows SLASH to
elegantly integrate the symbolic and neural components in a unified framework.
We evaluate SLASH on the benchmark data of MNIST addition as well as novel
tasks for DPPLs such as missing data prediction and set prediction with
state-of-the-art performance, thereby showing the effectiveness and generality
of our method.

    

### [[2110.03433] From the Head or the Heart? An Experimental Design on the Impact of Explanation on Cognitive and Affective Trust](http://arxiv.org/abs/2110.03433)


  Automated vehicles (AVs) are social robots that can potentially benefit our
society. According to the existing literature, AV explanations can promote
passengers' trust by reducing the uncertainty associated with the AV's
reasoning and actions. However, the literature on AV explanations and trust has
failed to consider how the type of trust
- cognitive versus affective - might alter this relationship. Yet, the
existing literature has shown that the implications associated with trust vary
widely depending on whether it is cognitive or affective. To address this
shortcoming and better understand the impacts of explanations on trust in AVs,
we designed a study to investigate the effectiveness of explanations on both
cognitive and affective trust. We expect these results to be of great
significance in designing AV explanations to promote AV trust.

    

### [[2110.03445] PWG-IDS: An Intrusion Detection Model for Solving Class Imbalance in IIoT Networks Using Generative Adversarial Networks](http://arxiv.org/abs/2110.03445)


  With the continuous development of industrial IoT (IIoT) technology, network
security is becoming more and more important. And intrusion detection is an
important part of its security. However, since the amount of attack traffic is
very small compared to normal traffic, this imbalance makes intrusion detection
in it very difficult. To address this imbalance, an intrusion detection system
called pretraining Wasserstein generative adversarial network intrusion
detection system (PWG-IDS) is proposed in this paper. This system is divided
into two main modules: 1) In this module, we introduce the pretraining
mechanism in the Wasserstein generative adversarial network with gradient
penalty (WGAN-GP) for the first time, firstly using the normal network traffic
to train the WGAN-GP, and then inputting the imbalance data into the
pre-trained WGAN-GP to retrain and generate the final required data. 2)
Intrusion detection module: We use LightGBM as the classification algorithm to
detect attack traffic in IIoT networks. The experimental results show that our
proposed PWG-IDS outperforms other models, with F1-scores of 99% and 89% on the
2 datasets, respectively. And the pretraining mechanism we proposed can also be
widely used in other GANs, providing a new way of thinking for the training of
GANs.

    

### [[2110.03461] Self-Evolutionary Optimization for Pareto Front Learning](http://arxiv.org/abs/2110.03461)


  Multi-task learning (MTL), which aims to improve performance by learning
multiple tasks simultaneously, inherently presents an optimization challenge
due to multiple objectives. Hence, multi-objective optimization (MOO)
approaches have been proposed for multitasking problems. Recent MOO methods
approximate multiple optimal solutions (Pareto front) with a single unified
model, which is collectively referred to as Pareto front learning (PFL). In
this paper, we show that PFL can be re-formulated into another MOO problem with
multiple objectives, each of which corresponds to different preference weights
for the tasks. We leverage an evolutionary algorithm (EA) to propose a method
for PFL called self-evolutionary optimization (SEO) by directly maximizing the
hypervolume. By using SEO, the neural network learns to approximate the Pareto
front conditioned on multiple hyper-parameters that drastically affect the
hypervolume. Then, by generating a population of approximations simply by
inferencing the network, the hyper-parameters of the network can be optimized
by EA. Utilizing SEO for PFL, we also introduce self-evolutionary Pareto
networks (SEPNet), enabling the unified model to approximate the entire Pareto
front set that maximizes the hypervolume. Extensive experimental results
confirm that SEPNet can find a better Pareto front than the current
state-of-the-art methods while minimizing the increase in model size and
training cost.

    

### [[2110.03468] Belief Evolution Network: Probability Transformation of Basic Belief Assignment and Fusion Conflict Probability](http://arxiv.org/abs/2110.03468)


  We give a new interpretation of basic belief assignment transformation into
probability distribution, and use directed acyclic network called belief
evolution network to describe the causality between the focal elements of a
BBA. On this basis, a new probability transformations method called full
causality probability transformation is proposed, and this method is superior
to all previous method after verification from the process and the result. In
addition, using this method combined with disjunctive combination rule, we
propose a new probabilistic combination rule called disjunctive transformation
combination rule. It has an excellent ability to merge conflicts and an
interesting pseudo-Matthew effect, which offer a new idea to information fusion
besides the combination rule of Dempster.

    

### [[2110.03485] Cartoon Explanations of Image Classifiers](http://arxiv.org/abs/2110.03485)


  We present CartoonX (Cartoon Explanation), a novel model-agnostic explanation
method tailored towards image classifiers and based on the rate-distortion
explanation (RDE) framework. Natural images are roughly piece-wise smooth
signals -- also called cartoon images -- and tend to be sparse in the wavelet
domain. CartoonX is the first explanation method to exploit this by requiring
its explanations to be sparse in the wavelet domain, thus extracting the
\emph{relevant piece-wise smooth} part of an image instead of relevant
pixel-sparse regions. We demonstrate experimentally that CartoonX is not only
highly interpretable due to its piece-wise smooth nature but also particularly
apt at explaining misclassifications.

    

### [[2110.03524] Data-Driven Methods for Balancing Fairness and Efficiency in Ride-Pooling](http://arxiv.org/abs/2110.03524)


  Rideshare and ride-pooling platforms use artificial intelligence-based
matching algorithms to pair riders and drivers. However, these platforms can
induce inequality either through an unequal income distribution or disparate
treatment of riders. We investigate two methods to reduce forms of inequality
in ride-pooling platforms: (1) incorporating fairness constraints into the
objective function and (2) redistributing income to drivers to reduce income
fluctuation and inequality. To evaluate our solutions, we use the New York City
taxi data set. For the first method, we find that optimizing for driver-side
fairness outperforms state-of-the-art models on the number of riders serviced,
both in the worst-off neighborhood and overall, showing that optimizing for
fairness can assist profitability in certain circumstances. For the second
method, we explore income redistribution as a way to combat income inequality
by having drivers keep an $r$ fraction of their income, and contributing the
rest to a redistribution pool. For certain values of $r$, most drivers earn
near their Shapley value, while still incentivizing drivers to maximize value,
thereby avoiding the free-rider problem and reducing income variability. The
first method can be extended to many definitions of fairness and the second
method provably improves fairness without affecting profitability.

    

### [[2110.03546] mRAT-SQL+GAP:A Portuguese Text-to-SQL Transformer](http://arxiv.org/abs/2110.03546)


  The translation of natural language questions to SQL queries has attracted
growing attention, in particular in connection with transformers and similar
language models. A large number of techniques are geared towards the English
language; in this work, we thus investigated translation to SQL when input
questions are given in the Portuguese language. To do so, we properly adapted
state-of-the-art tools and resources. We changed the RAT-SQL+GAP system by
relying on a multilingual BART model (we report tests with other language
models), and we produced a translated version of the Spider dataset. Our
experiments expose interesting phenomena that arise when non-English languages
are targeted; in particular, it is better to train with original and translated
training datasets together, even if a single target language is desired. This
multilingual BART model fine-tuned with a double-size training dataset (English
and Portuguese) achieved 83% of the baseline, making inferences for the
Portuguese test dataset. This investigation can help other researchers to
produce results in Machine Learning in a language different from English. Our
multilingual ready version of RAT-SQL+GAP and the data are available,
open-sourced as mRAT-SQL+GAP at: this https URL


### [[2110.03569] Human in the Loop for Machine Creativity](http://arxiv.org/abs/2110.03569)


  Artificial intelligence (AI) is increasingly utilized in synthesizing
visuals, texts, and audio. These AI-based works, often derived from neural
networks, are entering the mainstream market, as digital paintings, songs,
books, and others. We conceptualize both existing and future human-in-the-loop
(HITL) approaches for creative applications and to develop more expressive,
nuanced, and multimodal models. Particularly, how can our expertise as curators
and collaborators be encoded in AI models in an interactive manner? We examine
and speculate on long term implications for models, interfaces, and machine
creativity. Our selection, creation, and interpretation of AI art inherently
contain our emotional responses, cultures, and contexts. Therefore, the
proposed HITL may help algorithms to learn creative processes that are much
harder to codify or quantify. We envision multimodal HITL processes, where
texts, visuals, sounds, and other information are coupled together, with
automated analysis of humans and environments. Overall, these HITL approaches
will increase interaction between human and AI, and thus help the future AI
systems to better understand our own creative and emotional processes.

    

### [[2110.03613] A Data-Centric Approach for Training Deep Neural Networks with Less Data](http://arxiv.org/abs/2110.03613)


  While the availability of large datasets is perceived to be a key requirement
for training deep neural networks, it is possible to train such models with
relatively little data. However, compensating for the absence of large datasets
demands a series of actions to enhance the quality of the existing samples and
to generate new ones. This paper summarizes our winning submission to the
"Data-Centric AI" competition. We discuss some of the challenges that arise
while training with a small dataset, offer a principled approach for systematic
data quality enhancement, and propose a GAN-based solution for synthesizing new
data points. Our evaluations indicate that the dataset generated by the
proposed pipeline offers 5% accuracy improvement while being significantly
smaller than the baseline.

    

### [[2110.03624] On the Complexity of Inductively Learning Guarded Rules](http://arxiv.org/abs/2110.03624)


  We investigate the computational complexity of mining guarded clauses from
clausal datasets through the framework of inductive logic programming (ILP). We
show that learning guarded clauses is NP-complete and thus one step below the
$\sigma^P_2$-complete task of learning Horn clauses on the polynomial
hierarchy. Motivated by practical applications on large datasets we identify a
natural tractable fragment of the problem. Finally, we also generalise all of
our results to $k$-guarded clauses for constant $k$.

    

### [[2110.03643] From Weighted Conditionals of Multilayer Perceptrons to a Gradual Argumentation Semantics](http://arxiv.org/abs/2110.03643)


  A fuzzy multipreference semantics has been recently proposed for weighted
conditional knowledge bases, and used to develop a logical semantics for
Multilayer Perceptrons, by regarding a deep neural network (after training) as
a weighted conditional knowledge base. This semantics, in its different
variants, suggests some gradual argumentation semantics, which are related to
the family of the gradual semantics. The relationships between weighted
conditional knowledge bases and MLPs extend to the proposed gradual semantics,
which captures the stationary states of MPs, so that a dee neural network can
as well be seen as a weighted argumentation graph.

    

### [[2010.01196] End-to-End Differentiable Molecular Mechanics Force Field Construction](http://arxiv.org/abs/2010.01196)


  Molecular mechanics (MM) potentials have long been a workhorse of
computational chemistry. Leveraging accuracy and speed, these functional forms
find use in a wide variety of applications in biomolecular modeling and drug
discovery, from rapid virtual screening to detailed free energy calculations.
Traditionally, MM potentials have relied on human-curated, inflexible, and
poorly extensible discrete chemical perception rules (atom types}) for applying
parameters to small molecules or biopolymers, making it difficult to optimize
both types and parameters to fit quantum chemical or physical property data.
Here, we propose an alternative approach that uses graph neural networks to
perceive chemical environments, producing continuous atom embeddings from which
valence and nonbonded parameters can be predicted using invariance-preserving
layers. Since all stages are built from smooth neural functions, the entire
process -- spanning chemical perception to parameter assignment -- is modular
and end-to-end differentiable with respect to model parameters, allowing new
force fields to be easily constructed, extended, and applied to arbitrary
molecules. We show that this approach is not only sufficiently expressive to
reproduce legacy atom types, but that it can learn and extend existing
molecular mechanics force fields and construct entirely new force fields
applicable to both biopolymers and small molecules from quantum chemical
calculations, and even learn to accurately predict free energies from
experimental observables. This approach is implemented in the free and open
source package Espaloma, available at this https URL.

    

### [[2102.03082] Achieving Explainability for Plant Disease Classification with Disentangled Variational Autoencoders](http://arxiv.org/abs/2102.03082)


  Agricultural image recognition tasks are becoming increasingly dependent on
deep learning (DL); however, despite the excellent performance of DL, it is
difficult to comprehend the type of logic or features of the input image it
uses during decision making. Knowing the logic or features is highly crucial
for result verification, algorithm improvement, training data improvement, and
knowledge extraction. However, the explanations from the current heatmap-based
algorithms are insufficient for the abovementioned requirements. To address
this, this paper details the development of a classification and explanation
method based on a variational autoencoder (VAE) architecture, which can
visualize the variations of the most important features by visualizing the
generated images that correspond to the variations of those features. Using the
PlantVillage dataset, an acceptable level of explainability was achieved
without sacrificing the classification accuracy. The proposed method can also
be extended to other crops as well as other image classification tasks.
Further, application systems using this method for disease identification
tasks, such as the identification of potato blackleg disease, potato virus Y,
and other image classification tasks, are currently being developed.

    

### [[2104.07412] XTREME-R: Towards More Challenging and Nuanced Multilingual Evaluation](http://arxiv.org/abs/2104.07412)


  Machine learning has brought striking advances in multilingual natural
language processing capabilities over the past year. For example, the latest
techniques have improved the state-of-the-art performance on the XTREME
multilingual benchmark by more than 13 points. While a sizeable gap to
human-level performance remains, improvements have been easier to achieve in
some tasks than in others. This paper analyzes the current state of
cross-lingual transfer learning and summarizes some lessons learned. In order
to catalyze meaningful progress, we extend XTREME to XTREME-R, which consists
of an improved set of ten natural language understanding tasks, including
challenging language-agnostic retrieval tasks, and covers 50 typologically
diverse languages. In addition, we provide a massively multilingual diagnostic
suite (MultiCheckList) and fine-grained multi-dataset evaluation capabilities
through an interactive public leaderboard to gain a better understanding of
such models. The leaderboard and code for XTREME-R will be made available at
https://sites.research.google/xtreme and
this https URL respectively.

    

### [[2104.08401] Enriching a Model's Notion of Belief using a Persistent Memory](http://arxiv.org/abs/2104.08401)


  (This is an old and now obsolete draft. See https://arxiv.org/abs/2109.14723
("BeliefBank: Adding Memory to a Pre-Trained Language Model for a Systematic
Notion of Belief") for the final paper).

    

### [[2106.09847] Disinformation, Stochastic Harm, and Costly Effort: A Principal-Agent Analysis of Regulating Social Media Platforms](http://arxiv.org/abs/2106.09847)


  The spread of disinformation on social media platforms such as Facebook is
harmful to society. This harm may manifest as a gradual degradation of public
discourse; but it can also take the form of sudden dramatic events such as the
recent insurrection on Capitol Hill. The platforms themselves are in the best
position to prevent the spread of disinformation, as they have the best access
to relevant data and the expertise to use it. However, mitigating
disinformation is costly, not only for implementing classification algorithms
or employing manual detection, but also because limiting such highly viral
content impacts user growth and thus potential advertising revenue. Since the
costs of harmful content are borne by other entities, the platform will
therefore have no incentive to exercise the socially-optimal level of effort.
This problem is similar to the problem of environmental regulation, in which
the costs of adverse events are not directly borne by a firm, the mitigation
effort of a firm is not observable, and the causal link between a harmful
consequence and a specific failure is difficult to prove. In the environmental
regulation domain, one solution to this issue is to perform costly monitoring
to ensure that the firm takes adequate precautions according a specified rule.
However, classifying disinformation is performative, and thus a fixed rule
becomes less effective over time. Encoding our domain as a Markov decision
process, we demonstrate that no penalty based on a static rule, no matter how
large, can incentivize adequate effort by the platform. Penalties based on an
adaptive rule can incentivize optimal effort, but counterintuitively, only if
the regulator sufficiently overreacts to harmful events by requiring a
greater-than-optimal level of effort. We therefore push for mechanisms that
elicit platforms' costs of precautionary effort in order to bypass such an
overreaction.

    

### [[2110.01476] Learning Online Visual Invariances for Novel Objects via Supervised and Self-Supervised Training](http://arxiv.org/abs/2110.01476)


  Humans can identify objects following various spatial transformations such as
scale and viewpoint. This extends to novel objects, after a single presentation
at a single pose, sometimes referred to as online invariance. CNNs have been
proposed as a compelling model of human vision, but their ability to identify
objects across transformations is typically tested on held-out samples of
trained categories after extensive data augmentation. This paper assesses
whether standard CNNs can support human-like online invariance by training
models to recognize images of synthetic 3D objects that undergo several
transformations: rotation, scaling, translation, brightness, contrast, and
viewpoint. Through the analysis of models' internal representations, we show
that standard supervised CNNs trained on transformed objects can acquire strong
invariances on novel classes even when trained with as few as 50 objects taken
from 10 classes. This extended to a different dataset of photographs of real
objects. We also show that these invariances can be acquired in a
self-supervised way, through solving the same/different task. We suggest that
this latter approach may be similar to how humans acquire invariances.

    

### [[2110.03471] FaaSter Troubleshooting -- Evaluating Distributed Tracing Approaches for Serverless Applications](http://arxiv.org/abs/2110.03471)


  Serverless applications can be particularly difficult to troubleshoot, as
these applications are often composed of various managed and partly managed
services. Faults are often unpredictable and can occur at multiple points, even
in simple compositions. Each additional function or service in a serverless
composition introduces a new possible fault source and a new layer to obfuscate
faults. Currently, serverless platforms offer only limited support for
identifying runtime faults. Developers looking to observe their serverless
compositions often have to rely on scattered logs and ambiguous error messages
to pinpoint root causes. In this paper, we investigate the use of distributed
tracing for improving the observability of faults in serverless applications.
To this end, we first introduce a model for characterizing fault observability,
then provide a prototypical tracing implementation - specifically, a
developer-driven and a platform-supported tracing approach. We compare both
approaches with our model, measure associated trade-offs (execution latency,
resource utilization), and contribute new insights for troubleshooting
serverless compositions.

    

### [[1901.03371] Optimal Asynchronous Dynamic Policies in Energy-Efficient Data Centers](http://arxiv.org/abs/1901.03371)


  In this paper, we use a Markov decision process to find optimal asynchronous
policy of an energy-efficient data center with two groups of heterogeneous
servers, a finite buffer, and a fast setup process at sleep state. Servers in
Group 1 always work. Servers in Group 2 may either work or sleep, and a fast
setup process occurs when server's states are changed from sleep to work. In
such a data center, an asynchronous dynamic policy is designed as two
sub-policies: The setup policy and the sleep policy, which determine the switch
rule between the work and sleep states for the servers in Group 2. To analyze
the optimal asynchronous dynamic policy, we apply the Markov decision process
to establish a policy-based Poisson equation, which provides expression for the
unique solution of the performance potential by means of the RG-factorization.
Based on this, we can characterize the monotonicity and optimality of the
long-run average profit of the data center with respect to the asynchronous
dynamic policy under different service prices. Furthermore, we prove that the
bang-bang control is always optimal for this optimization problem, and supports
a threshold-type dynamic control in the energy-efficient data center. We hope
that the methodology and results derived in this paper can shed light to the
study of more general energy-efficient data centers.

    