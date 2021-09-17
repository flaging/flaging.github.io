
## 2021-9-17

### [[2101.07034] AGRNet: Adaptive Graph Representation Learning and Reasoning for Face Parsing](http://arxiv.org/abs/2101.07034)


  Face parsing infers a pixel-wise label to each facial component, which has
drawn much attention recently.Previous methods have shown their success in face
parsing, which however overlook the correlation among facial this http URL a
matter of fact, the component-wise relationship is a critical clue in
discriminating ambiguous pixels in facial this http URL address this issue, we
propose adaptive graph representation learning and reasoning over facial
components, aiming to learn representative vertices that describe each
component, exploit the component-wise relationship and thereby produce accurate
parsing results against ambiguity. In particular, we devise an adaptive and
differentiable graph abstraction method to represent the components on a graph
via pixel-to-vertex projection under the initial condition of a predicted
parsing map, where pixel features within a certain facial region are aggregated
onto a vertex. Further, we explicitly incorporate the image edge as a prior in
the model, which helps to discriminate edge and non-edge pixels during the
projection, thus leading to refined parsing results along the edges.Then, our
model learns and reasons over the relations among components by propagating
information across vertices on the graph. Finally, the refined vertex features
are projected back to pixel grids for the prediction of the final parsing
this http URL train our model, we propose a discriminative loss to penalize small
distances between vertices in the feature space, which leads to distinct
vertices with strong semantics. Experimental results show the superior
performance of the proposed model on multiple face parsing datasets, along with
the validation on the human parsing task to demonstrate the generalizability of
our model.

    

### [[2109.07768] Path Loss in Urban LoRa Networks: A Large-Scale Measurement Study](http://arxiv.org/abs/2109.07768)


  Urban LoRa networks promise to provide a cost-efficient and scalable
communication backbone for smart cities. One core challenge in rolling out and
operating these networks is radio network planning, i.e., precise predictions
about possible new locations and their impact on network coverage. Path loss
models aid in this task, but evaluating and comparing different models requires
a sufficiently large set of high-quality received packet power samples. In this
paper, we report on a corresponding large-scale measurement study covering an
urban area of 200km2 over a period of 230 days using sensors deployed on
garbage trucks, resulting in more than 112 thousand high-quality samples for
received packet power. Using this data, we compare eleven previously proposed
path loss models and additionally provide new coefficients for the Log-distance
model. Our results reveal that the Log-distance model and other well-known
empirical models such as Okumura or Winner+ provide reasonable estimations in
an urban environment, and terrain based models such as ITM or ITWOM have no
advantages. In addition, we derive estimations for the needed sample size in
similar measurement campaigns. To stimulate further research in this direction,
we make all our data publicly available.

    

### [[2109.07814] A Lightweight Cell Switching and Traffic Offloading Scheme for Energy Optimization in Ultra-Dense Heterogeneous Networks](http://arxiv.org/abs/2109.07814)


  One of the major capacity boosters for 5G networks is the deployment of
ultra-dense heterogeneous networks (UDHNs). However, this deployment results in
tremendousincrease in the energy consumption of the network due to the large
number of base stations (BSs) involved. In addition to enhanced capacity, 5G
networks must also be energy efficient for it to be economically viable and
environmentally friendly. Dynamic cell switching is a very common way of
reducing the total energy consumption of the network but most of the proposed
methods are computationally demanding which makes them unsuitable for
application in ultra-dense network deployment with massive number of BSs. To
tackle this problem, we propose a lightweight cell switching scheme also known
as Threshold-based Hybrid cEllswItching Scheme (THESIS) for energy optimization
in UDHNs. The developed approach combines the benefits of clustering and
exhaustive search (ES) algorithm to produce a solution whose optimality is
close to that of the ES (which is guaranteed tobe optimal), but is
computationally more efficient than ES and as such can be applied for cell
switching in real networks even when their dimension is large. The performance
evaluation shows that the THESIS produces a significant reduction in the energy
consumption of the UDHN and is able to reduce the complexity of finding a
near-optimal solution from exponential to polynomial complexity.

    

### [[2109.07934] Fast and Secure Routing Algorithms for Quantum Key Distribution Networks](http://arxiv.org/abs/2109.07934)


  This paper considers the problem of secure packet routing at the maximum
achievable rate in a Quantum key distribution (QKD) network. Assume that a QKD
protocol generates symmetric private keys for secure communication over each
link in a multi-hop network. The quantum key generation process, which is
affected by noise, is assumed to be modeled by a stochastic counting process.
Packets are first encrypted with the available quantum keys for each hop and
then transmitted on a point-to-point basis over the communication links. A
fundamental problem that arises in this setting is to design a secure and
capacity-achieving routing policy that accounts for the time-varying
availability of the quantum keys for encryption and finite link capacities for
transmission. In this paper, by combining the QKD protocol with the Universal
Max Weight (UMW) routing policy, we design a new secure throughput-optimal
routing policy, called Tandem Queue Decomposition (TQD). TQD solves the problem
of secure routing efficiently for a wide class of traffic, including unicast,
broadcast, and multicast. One of our main contributions in this paper is to
show that the problem can be reduced to the usual generalized network flow
problem on a transformed network without the key availability constraints.
Simulation results show that the proposed policy incurs a substantially smaller
delay as compared to the state-of-the-art routing and key management policies.
The proof of throughput-optimality of the proposed policy makes use of the
Lyapunov stability theory along with a careful treatment of the key-storage
dynamics.

    

### [[2109.07999] Learning from Peers: Transfer Reinforcement Learning for Joint Radio and Cache Resource Allocation in 5G Network Slicing](http://arxiv.org/abs/2109.07999)


  Radio access network (RAN) slicing is an important part of network slicing in
5G. The evolving network architecture requires the orchestration of multiple
network resources such as radio and cache resources. In recent years, machine
learning (ML) techniques have been widely applied for network slicing. However,
most existing works do not take advantage of the knowledge transfer capability
in ML. In this paper, we propose a transfer reinforcement learning (TRL) scheme
for joint radio and cache resources allocation to serve 5G RAN slicing.We first
define a hierarchical architecture for the joint resources allocation. Then we
propose two TRL algorithms: Q-value transfer reinforcement learning (QTRL) and
action selection transfer reinforcement learning (ASTRL). In the proposed
schemes, learner agents utilize the expert agents' knowledge to improve their
performance on target tasks. The proposed algorithms are compared with both the
model-free Q-learning and the model-based priority proportional fairness and
time-to-live (PPF-TTL) algorithms. Compared with Q-learning, QTRL and ASTRL
present 23.9% lower delay for Ultra Reliable Low Latency Communications slice
and 41.6% higher throughput for enhanced Mobile Broad Band slice, while
achieving significantly faster convergence than Q-learning. Moreover, 40.3%
lower URLLC delay and almost twice eMBB throughput are observed with respect to
PPF-TTL.

    

### [[2109.08139] Adversarial Attacks against Deep Learning Based Power Control in Wireless Communications](http://arxiv.org/abs/2109.08139)


  We consider adversarial machine learning based attacks on power allocation
where the base station (BS) allocates its transmit power to multiple orthogonal
subcarriers by using a deep neural network (DNN) to serve multiple user
equipments (UEs). The DNN that corresponds to a regression model is trained
with channel gains as the input and allocated transmit powers as the output.
While the BS allocates the transmit power to the UEs to maximize rates for all
UEs, there is an adversary that aims to minimize these rates. The adversary may
be an external transmitter that aims to manipulate the inputs to the DNN by
interfering with the pilot signals that are transmitted to measure the channel
gain. Alternatively, the adversary may be a rogue UE that transmits fabricated
channel estimates to the BS. In both cases, the adversary carefully crafts
adversarial perturbations to manipulate the inputs to the DNN of the BS subject
to an upper bound on the strengths of these perturbations. We consider the
attacks targeted on a single UE or all UEs. We compare these attacks with a
benchmark, where the adversary scales down the input to the DNN. We show that
adversarial attacks are much more effective than the benchmark attack in terms
of reducing the rate of communications. We also show that adversarial attacks
are robust to the uncertainty at the adversary including the erroneous
knowledge of channel gains and the potential errors in exercising the attacks
exactly as specified.

    

### [[2012.11783] Scalable Deep Reinforcement Learning for Routing and Spectrum Access in Physical Layer](http://arxiv.org/abs/2012.11783)


  This paper proposes a novel scalable reinforcement learning approach for
simultaneous routing and spectrum access in wireless ad-hoc networks. In most
previous works on reinforcement learning for network optimization, the network
topology is assumed to be fixed, and a different agent is trained for each
transmission node -- this limits scalability and generalizability. Further,
routing and spectrum access are typically treated as separate tasks. Moreover,
the optimization objective is usually a cumulative metric along the route,
e.g., number of hops or delay. In this paper, we account for the physical-layer
signal-to-interference-plus-noise ratio (SINR) in a wireless network and
further show that bottleneck objective such as the minimum SINR along the route
can also be optimized effectively using reinforcement learning. Specifically,
we propose a scalable approach in which a single agent is associated with each
flow and makes routing and spectrum access decisions as it moves along the
frontier nodes. The agent is trained according to the physical-layer
characteristics of the environment using a novel rewarding scheme based on the
Monte Carlo estimation of the future bottleneck SINR. It learns to avoid
interference by intelligently making joint routing and spectrum allocation
decisions based on the geographical location information of the neighbouring
nodes.

    

### [[2109.07468] Computationally-Efficient Climate Predictions using Multi-Fidelity Surrogate Modelling](http://arxiv.org/abs/2109.07468)


  Accurately modelling the Earth's climate has widespread applications ranging
from forecasting local weather to understanding global climate change.
Low-fidelity simulations of climate phenomena are readily available, but
high-fidelity simulations are expensive to obtain. We therefore investigate the
potential of Gaussian process-based multi-fidelity surrogate modelling as a way
to produce high-fidelity climate predictions at low cost. Specifically, our
model combines the predictions of a low-fidelity Global Climate Model (GCM) and
those of a high-fidelity Regional Climate Model (RCM) to produce high-fidelity
temperature predictions for a mountainous region on the coastline of Peru. We
are able to produce high-fidelity temperature predictions at significantly
lower computational cost compared to the high-fidelity model alone: our
predictions have an average error of $15.62^\circ\text{C}^2$ yet our approach
only evaluates the high-fidelity model on 6% of the region of interest.

    

### [[2109.07471] Data-Driven Theory-guided Learning of Partial Differential Equations using SimultaNeous Basis Function Approximation and Parameter Estimation (SNAPE)](http://arxiv.org/abs/2109.07471)


  The measured spatiotemporal response of various physical processes is
utilized to infer the governing partial differential equations (PDEs). We
propose SimultaNeous Basis Function Approximation and Parameter Estimation
(SNAPE), a technique of parameter estimation of PDEs that is robust against
high levels of noise nearly 100 %, by simultaneously fitting basis functions to
the measured response and estimating the parameters of both ordinary and
partial differential equations. The domain knowledge of the general
multidimensional process is used as a constraint in the formulation of the
optimization framework. SNAPE not only demonstrates its applicability on
various complex dynamic systems that encompass wide scientific domains
including Schrödinger equation, chaotic duffing oscillator, and Navier-Stokes
equation but also estimates an analytical approximation to the process
response. The method systematically combines the knowledge of well-established
scientific theories and the concepts of data science to infer the properties of
the process from the observed data.

    

### [[2109.07473] Generalized XGBoost Method](http://arxiv.org/abs/2109.07473)


  The XGBoost method has many advantages and is especially suitable for
statistical analysis of big data, but its loss function is limited to convex
functions. In many specific applications, a nonconvex loss function would be
preferable. In this paper, we propose a generalized XGBoost method, which
requires weaker loss function condition and involves more general loss
functions, including convex loss functions and some non-convex loss functions.
Furthermore, this generalized XGBoost method is extended to multivariate loss
function to form a more generalized XGBoost method. This method is a
multivariate regularized tree boosting method, which can model multiple
parameters in most of the frequently-used parametric probability distributions
to be fitted by predictor variables. Meanwhile, the related algorithms and some
examples in non-life insurance pricing are given.

    

### [[2109.07488] Comparing Euclidean and Hyperbolic Embeddings on the WordNet Nouns Hypernymy Graph](http://arxiv.org/abs/2109.07488)


  Nickel and Kiela (2017) present a new method for embedding tree nodes in the
Poincare ball, and suggest that these hyperbolic embeddings are far more
effective than Euclidean embeddings at embedding nodes in large, hierarchically
structured graphs like the WordNet nouns hypernymy tree. This is especially
true in low dimensions (Nickel and Kiela, 2017, Table 1). In this work, we seek
to reproduce their experiments on embedding and reconstructing the WordNet
nouns hypernymy graph. Counter to what they report, we find that Euclidean
embeddings are able to represent this tree at least as well as Poincare
embeddings, when allowed at least 50 dimensions. We note that this does not
diminish the significance of their work given the impressive performance of
hyperbolic embeddings in very low-dimensional settings. However, given the wide
influence of their work, our aim here is to present an updated and more
accurate comparison between the Euclidean and hyperbolic embeddings.

    

### [[2109.07497] Sign-MAML: Efficient Model-Agnostic Meta-Learning by SignSGD](http://arxiv.org/abs/2109.07497)


  We propose a new computationally-efficient first-order algorithm for
Model-Agnostic Meta-Learning (MAML). The key enabling technique is to interpret
MAML as a bilevel optimization (BLO) problem and leverage the sign-based
SGD(signSGD) as a lower-level optimizer of BLO. We show that MAML, through the
lens of signSGD-oriented BLO, naturally yields an alternating optimization
scheme that just requires first-order gradients of a learned meta-model. We
term the resulting MAML algorithm Sign-MAML. Compared to the conventional
first-order MAML (FO-MAML) algorithm, Sign-MAML is theoretically-grounded as it
does not impose any assumption on the absence of second-order derivatives
during meta training. In practice, we show that Sign-MAML outperforms FO-MAML
in various few-shot image classification tasks, and compared to MAML, it
achieves a much more graceful tradeoff between classification accuracy and
computation efficiency.

    

### [[2109.07498] Short Quantum Circuits in Reinforcement Learning Policies for the Vehicle Routing Problem](http://arxiv.org/abs/2109.07498)


  Quantum computing and machine learning have potential for symbiosis. However,
in addition to the hardware limitations from current devices, there are still
basic issues that must be addressed before quantum circuits can usefully
incorporate with current machine learning tasks. We report a new strategy for
such an integration in the context of attention models used for reinforcement
learning. Agents that implement attention mechanisms have successfully been
applied to certain cases of combinatorial routing problems by first encoding
nodes on a graph and then sequentially decoding nodes until a route is
selected. We demonstrate that simple quantum circuits can used in place of
classical attention head layers while maintaining performance. Our method
modifies the networks used in [1] by replacing key and query vectors for every
node with quantum states that are entangled before being measured. The
resulting hybrid classical-quantum agent is tested in the context of vehicle
routing problems where its performance is competitive with the original
classical approach. We regard our model as a prototype that can be scaled up
and as an avenue for further study on the role of quantum computing in
reinforcement learning.

    

### [[2109.07504] Federated Contrastive Learning for Decentralized Unlabeled Medical Images](http://arxiv.org/abs/2109.07504)


  A label-efficient paradigm in computer vision is based on self-supervised
contrastive pre-training on unlabeled data followed by fine-tuning with a small
number of labels. Making practical use of a federated computing environment in
the clinical domain and learning on medical images poses specific challenges.
In this work, we propose FedMoCo, a robust federated contrastive learning (FCL)
framework, which makes efficient use of decentralized unlabeled medical data.
FedMoCo has two novel modules: metadata transfer, an inter-node statistical
data augmentation module, and self-adaptive aggregation, an aggregation module
based on representational similarity analysis. To the best of our knowledge,
this is the first FCL work on medical images. Our experiments show that FedMoCo
can consistently outperform FedAvg, a seminal federated learning framework, in
extracting meaningful representations for downstream tasks. We further show
that FedMoCo can substantially reduce the amount of labeled data required in a
downstream task, such as COVID-19 detection, to achieve a reasonable
performance.

    

### [[2109.07509] Learning to Aggregate and Refine Noisy Labels for Visual Sentiment Analysis](http://arxiv.org/abs/2109.07509)


  Visual sentiment analysis has received increasing attention in recent years.
However, the quality of the dataset is a concern because the sentiment labels
are crowd-sourcing, subjective, and prone to mistakes. This poses a severe
threat to the data-driven models including the deep neural networks which would
generalize poorly on the testing cases if they are trained to over-fit the
samples with noisy sentiment labels. Inspired by the recent progress on
learning with noisy labels, we propose a robust learning method to perform
robust visual sentiment analysis. Our method relies on an external memory to
aggregate and filter noisy labels during training and thus can prevent the
model from overfitting the noisy cases. The memory is composed of the
prototypes with corresponding labels, both of which can be updated online. We
establish a benchmark for visual sentiment analysis with label noise using
publicly available datasets. The experiment results of the proposed benchmark
settings comprehensively show the effectiveness of our method.

    

### [[2109.07513] Tied & Reduced RNN-T Decoder](http://arxiv.org/abs/2109.07513)


  Previous works on the Recurrent Neural Network-Transducer (RNN-T) models have
shown that, under some conditions, it is possible to simplify its prediction
network with little or no loss in recognition accuracy (arXiv:2003.07705
[eess.AS], [2], arXiv:2012.06749 [cs.CL]). This is done by limiting the context
size of previous labels and/or using a simpler architecture for its layers
instead of LSTMs. The benefits of such changes include reduction in model size,
faster inference and power savings, which are all useful for on-device
applications.
In this work, we study ways to make the RNN-T decoder (prediction network +
joint network) smaller and faster without degradation in recognition
performance. Our prediction network performs a simple weighted averaging of the
input embeddings, and shares its embedding matrix weights with the joint
network's output layer (a.k.a. weight tying, commonly used in language modeling
arXiv:1611.01462 [cs.LG]). This simple design, when used in conjunction with
additional Edit-based Minimum Bayes Risk (EMBR) training, reduces the RNN-T
Decoder from 23M parameters to just 2M, without affecting word-error rate
(WER).

    

### [[2109.07519] Discovering Useful Compact Sets of Sequential Rules in a Long Sequence](http://arxiv.org/abs/2109.07519)


  We are interested in understanding the underlying generation process for long
sequences of symbolic events. To do so, we propose COSSU, an algorithm to mine
small and meaningful sets of sequential rules. The rules are selected using an
MDL-inspired criterion that favors compactness and relies on a novel rule-based
encoding scheme for sequences. Our evaluation shows that COSSU can successfully
retrieve relevant sets of closed sequential rules from a long sequence. Such
rules constitute an interpretable model that exhibits competitive accuracy for
the tasks of next-element prediction and classification.

    

### [[2109.07548] Learning the Regularization in DCE-MR Image Reconstruction for Functional Imaging of Kidneys](http://arxiv.org/abs/2109.07548)


  Kidney DCE-MRI aims at both qualitative assessment of kidney anatomy and
quantitative assessment of kidney function by estimating the tracer kinetic
(TK) model parameters. Accurate estimation of TK model parameters requires an
accurate measurement of the arterial input function (AIF) with high temporal
resolution. Accelerated imaging is used to achieve high temporal resolution,
which yields under-sampling artifacts in the reconstructed images. Compressed
sensing (CS) methods offer a variety of reconstruction options. Most commonly,
sparsity of temporal differences is encouraged for regularization to reduce
artifacts. Increasing regularization in CS methods removes the ambient
artifacts but also over-smooths the signal temporally which reduces the
parameter estimation accuracy. In this work, we propose a single image trained
deep neural network to reduce MRI under-sampling artifacts without reducing the
accuracy of functional imaging markers. Instead of regularizing with a penalty
term in optimization, we promote regularization by generating images from a
lower dimensional representation. In this manuscript we motivate and explain
the lower dimensional input design. We compare our approach to CS
reconstructions with multiple regularization weights. Proposed approach results
in kidney biomarkers that are highly correlated with the ground truth markers
estimated using the CS reconstruction which was optimized for functional
analysis. At the same time, the proposed approach reduces the artifacts in the
reconstructed images.

    

### [[2109.07555] RaWaNet: Enriching Graph Neural Network Input via Random Walks on Graphs](http://arxiv.org/abs/2109.07555)


  In recent years, graph neural networks (GNNs) have gained increasing
popularity and have shown very promising results for data that are represented
by graphs. The majority of GNN architectures are designed based on developing
new convolutional and/or pooling layers that better extract the hidden and
deeper representations of the graphs to be used for different prediction tasks.
The inputs to these layers are mainly the three default descriptors of a graph,
node features $(X)$, adjacency matrix $(A)$, and edge features $(W)$ (if
available). To provide a more enriched input to the network, we propose a
random walk data processing of the graphs based on three selected lengths.
Namely, (regular) walks of length 1 and 2, and a fractional walk of length
$\gamma \in (0,1)$, in order to capture the different local and global dynamics
on the graphs. We also calculate the stationary distribution of each random
walk, which is then used as a scaling factor for the initial node features
($X$). This way, for each graph, the network receives multiple adjacency
matrices along with their individual weighting for the node features. We test
our method on various molecular datasets by passing the processed node features
to the network in order to perform several classification and regression tasks.
Interestingly, our method, not using edge features which are heavily exploited
in molecular graph learning, let a shallow network outperform well known deep
GNNs.

    

### [[2109.07557] CounterNet: End-to-End Training of Counterfactual Aware Predictions](http://arxiv.org/abs/2109.07557)


  This work presents CounterNet, a novel end-to-end learning framework which
integrates the predictive model training and counterfactual (CF) explanation
generation into a single end-to-end pipeline. Counterfactual explanations
attempt to find the smallest modification to the feature values of an instance
that changes the prediction of the ML model to a predefined output. Prior CF
explanation techniques rely on solving separate time-intensive optimization
problems for every single input instance to find CF examples, and also suffer
from the misalignment of objectives between model predictions and explanations,
which leads to significant shortcomings in the quality of CF explanations.
CounterNet, on the other hand, integrates both prediction and explanation in
the same framework, which enables the optimization of the CF example generation
only once together with the predictive model. We propose a novel variant of
back-propagation which can help in effectively training CounterNet's network.
Finally, we conduct extensive experiments on multiple real-world datasets. Our
results show that CounterNet generates high-quality predictions, and
corresponding CF examples (with high validity) for any new input instance
significantly faster than existing state-of-the-art baselines.

    

### [[2109.07563] Non-smooth Bayesian Optimization in Tuning Problems](http://arxiv.org/abs/2109.07563)


  Building surrogate models is one common approach when we attempt to learn
unknown black-box functions. Bayesian optimization provides a framework which
allows us to build surrogate models based on sequential samples drawn from the
function and find the optimum. Tuning algorithmic parameters to optimize the
performance of large, complicated "black-box" application codes is a specific
important application, which aims at finding the optima of black-box functions.
Within the Bayesian optimization framework, the Gaussian process model produces
smooth or continuous sample paths. However, the black-box function in the
tuning problem is often non-smooth. This difficult tuning problem is worsened
by the fact that we usually have limited sequential samples from the black-box
function. Motivated by these issues encountered in tuning, we propose a novel
additive Gaussian process model called clustered Gaussian process (cGP), where
the additive components are induced by clustering. In the examples we studied,
the performance can be improved by as much as 90% among repetitive experiments.
By using this surrogate model, we want to capture the non-smoothness of the
black-box function. In addition to an algorithm for constructing this model, we
also apply the model to several artificial and real applications to evaluate
it.

    

### [[2109.07564] Estimation of Warfarin Dosage with Reinforcement Learning](http://arxiv.org/abs/2109.07564)


  In this paper, it has attempted to use Reinforcement learning to model the
proper dosage of Warfarin for patients.The paper first examines two baselines:
a fixed model of 35 mg/week dosages and a linear model that relies on patient
data. We implemented a LinUCB bandit that improved performance measured on
regret and percent incorrect. On top of the LinUCB bandit, we experimented with
online supervised learning and reward reshaping to boost performance. Our
results clearly beat the baselines and show the promise of using multi-armed
bandits and artificial intelligence to aid physicians in deciding proper
dosages.

    

### [[2109.07570] Predicting the outcome of team movements -- Player time series analysis using fuzzy and deep methods for representation learning](http://arxiv.org/abs/2109.07570)


  We extract and use player position time-series data, tagged along with the
action types, to build a competent model for representing team tactics
behavioral patterns and use this representation to predict the outcome of
arbitrary movements. We provide a framework for the useful encoding of short
tactics and space occupations in a more extended sequence of movements or
tactical plans. We investigate game segments during a match in which the team
in possession of the ball regularly attempts to reach a position where they can
take a shot at goal for a single game. A carefully designed and efficient
kernel is employed using a triangular fuzzy membership function to create
multiple time series for players' potential of presence at different court
regions. Unsupervised learning is then used for time series using triplet loss
and deep neural networks with exponentially dilated causal convolutions for the
derived multivariate time series. This works key contribution lies in its
approach to model how short scenes contribute to other longer ones and how
players occupies and creates new spaces in-game court. We discuss the
effectiveness of the proposed approach for prediction and recognition tasks on
the professional basketball SportVU dataset for the 2015-16 half-season. The
proposed system demonstrates descent functionality even with relatively small
data.

    

### [[2109.07571] Secure Your Ride: Real-time Matching Success Rate Prediction for Passenger-Driver Pairs](http://arxiv.org/abs/2109.07571)


  In recent years, online ride-hailing platforms have become an indispensable
part of urban transportation. After a passenger is matched up with a driver by
the platform, both the passenger and the driver have the freedom to simply
accept or cancel a ride with one click. Hence, accurately predicting whether a
passenger-driver pair is a good match turns out to be crucial for ride-hailing
platforms to devise instant order assignments. However, since the users of
ride-hailing platforms consist of two parties, decision-making needs to
simultaneously account for the dynamics from both the driver and the passenger
sides. This makes it more challenging than traditional online advertising
tasks. Moreover, the amount of available data is severely imbalanced across
different cities, creating difficulties for training an accurate model for
smaller cities with scarce data. Though a sophisticated neural network
architecture can help improve the prediction accuracy under data scarcity, the
overly complex design will impede the model's capacity of delivering timely
predictions in a production environment. In the paper, to accurately predict
the MSR of passenger-driver, we propose the Multi-View model (MV) which
comprehensively learns the interactions among the dynamic features of the
passenger, driver, trip order, as well as context. Regarding the data imbalance
problem, we further design the Knowledge Distillation framework (KD) to
supplement the model's predictive power for smaller cities using the knowledge
from cities with denser data and also generate a simple model to support
efficient deployment. Finally, we conduct extensive experiments on real-world
datasets from several different cities, which demonstrates the superiority of
our solution.

    

### [[2109.07573] Differentiable Physics: A Position Piece](http://arxiv.org/abs/2109.07573)


  Differentiable physics provides a new approach for modeling and understanding
the physical systems by pairing the new technology of differentiable
programming with classical numerical methods for physical simulation. We survey
the rapidly growing literature of differentiable physics techniques and
highlight methods for parameter estimation, learning representations, solving
differential equations, and developing what we call scientific foundation
models using data and inductive priors. We argue that differentiable physics
offers a new paradigm for modeling physical phenomena by combining classical
analytic solutions with numerical methodology using the bridge of
differentiable programming.

    

### [[2109.07578] Multi-Task Learning with Sequence-Conditioned Transporter Networks](http://arxiv.org/abs/2109.07578)


  Enabling robots to solve multiple manipulation tasks has a wide range of
industrial applications. While learning-based approaches enjoy flexibility and
generalizability, scaling these approaches to solve such compositional tasks
remains a challenge. In this work, we aim to solve multi-task learning through
the lens of sequence-conditioning and weighted sampling. First, we propose a
new suite of benchmark specifically aimed at compositional tasks, MultiRavens,
which allows defining custom task combinations through task modules that are
inspired by industrial tasks and exemplify the difficulties in vision-based
learning and planning methods. Second, we propose a vision-based end-to-end
system architecture, Sequence-Conditioned Transporter Networks, which augments
Goal-Conditioned Transporter Networks with sequence-conditioning and weighted
sampling and can efficiently learn to solve multi-task long horizon problems.
Our analysis suggests that not only the new framework significantly improves
pick-and-place performance on novel 10 multi-task benchmark problems, but also
the multi-task learning with weighted sampling can vastly improve learning and
agent performances on individual tasks.

    

### [[2109.07582] How to Simplify Search: Classification-wise Pareto Evolution for One-shot Neural Architecture Search](http://arxiv.org/abs/2109.07582)


  In the deployment of deep neural models, how to effectively and automatically
find feasible deep models under diverse design objectives is fundamental. Most
existing neural architecture search (NAS) methods utilize surrogates to predict
the detailed performance (e.g., accuracy and model size) of a candidate
architecture during the search, which however is complicated and inefficient.
In contrast, we aim to learn an efficient Pareto classifier to simplify the
search process of NAS by transforming the complex multi-objective NAS task into
a simple Pareto-dominance classification task. To this end, we propose a
classification-wise Pareto evolution approach for one-shot NAS, where an online
classifier is trained to predict the dominance relationship between the
candidate and constructed reference architectures, instead of using surrogates
to fit the objective functions. The main contribution of this study is to
change supernet adaption into a Pareto classifier. Besides, we design two
adaptive schemes to select the reference set of architectures for constructing
classification boundary and regulate the rate of positive samples over negative
ones, respectively. We compare the proposed evolution approach with
state-of-the-art approaches on widely-used benchmark datasets, and experimental
results indicate that the proposed approach outperforms other approaches and
have found a number of neural architectures with different model sizes ranging
from 2M to 6M under diverse objectives and constraints.

    

### [[2109.07583] Network representation learning systematic review: ancestors and current development state](http://arxiv.org/abs/2109.07583)


  Real-world information networks are increasingly occurring across various
disciplines including online social networks and citation networks. These
network data are generally characterized by sparseness, nonlinearity and
heterogeneity bringing different challenges to the network analytics task to
capture inherent properties from network data. Artificial intelligence and
machine learning have been recently leveraged as powerful systems to learn
insights from network data and deal with presented challenges. As part of
machine learning techniques, graph embedding approaches are originally
conceived for graphs constructed from feature represented datasets, like image
dataset, in which links between nodes are explicitly defined. These traditional
approaches cannot cope with network data challenges. As a new learning
paradigm, network representation learning has been proposed to map a real-world
information network into a low-dimensional space while preserving inherent
properties of the network. In this paper, we present a systematic comprehensive
survey of network representation learning, known also as network embedding,
from birth to the current development state. Through the undertaken survey, we
provide a comprehensive view of reasons behind the emergence of network
embedding and, types of settings and models used in the network embedding
pipeline. Thus, we introduce a brief history of representation learning and
word representation learning ancestor of network embedding. We provide also
formal definitions of basic concepts required to understand network
representation learning followed by a description of network embedding
pipeline. Most commonly used downstream tasks to evaluate embeddings, their
evaluation metrics and popular datasets are highlighted. Finally, we present
the open-source libraries for network embedding.

    

### [[2109.07591] On the Complementarity of Data Selection and Fine Tuning for Domain Adaptation](http://arxiv.org/abs/2109.07591)


  Domain adaptation of neural networks commonly relies on three training
phases: pretraining, selected data training and then fine tuning. Data
selection improves target domain generalization by training further on
pretraining data identified by relying on a small sample of target domain data.
This work examines the benefit of data selection for language modeling and
machine translation. Our experiments assess the complementarity of selection
with fine tuning and result in practical recommendations: (i) selected data
must be similar to the fine-tuning domain but not so much as to erode the
complementary effect of fine-tuning; (ii) there is a trade-off between
selecting little data for fast but limited progress or much data for slow but
long lasting progress; (iii) data selection can be applied early during
pretraining, with performance gains comparable to long pretraining session;
(iv) data selection from domain classifiers is often more effective than the
popular contrastive data selection method.

    

### [[2109.07593] Modern Cybersecurity Solution using Supervised Machine Learning](http://arxiv.org/abs/2109.07593)


  Cybersecurity is essential, and attacks are rapidly growing and getting more
challenging to detect. The traditional Firewall and Intrusion Detection system,
even though it is widely used and recommended but it fails to detect new
attacks, zero-day attacks, and traffic patterns that do not match with any
configured rules. Therefore, Machine Learning (ML) can be an efficient and
cost-reduced solution in cybersecurity.
We used Netflow datasets to extract features after applying data analysis.
Then, a selection process has been applied to compare these features with one
another. Our experiments focus on how efficient machine learning algorithms can
detect Bot traffic, Malware traffic, and background traffic. We managed to get
0.903 precision value from a dataset that has 6.5% Bot flows, 1.57% Normal
flows, 0.18% Command&Control (C&C) flows, and 91.7% background flows, from
2,753,884 total flows. The results show low false-negative with few
false-positive detections.

    

### [[2109.07601] A Column Streaming-Based Convolution Engine and Mapping Algorithm for CNN-based Edge AI accelerators](http://arxiv.org/abs/2109.07601)


  Edge AI accelerators have been emerging as a solution for near customers'
applications in areas such as unmanned aerial vehicles (UAVs), image
recognition sensors, wearable devices, robotics, and remote sensing satellites.
These applications not only require meeting performance targets but also
meeting strict area and power constraints due to their portable mobility
feature and limited power sources. As a result, a column streaming-based
convolution engine has been proposed in this paper that includes column sets of
processing elements design for flexibility in terms of the applicability for
different CNN algorithms in edge AI accelerators. Comparing to a commercialized
CNN accelerator, the key results reveal that the column streaming-based
convolution engine requires similar execution cycles for processing a 227 x 227
feature map with avoiding zero-padding penalties.

    

### [[2109.07602] Interpretable Additive Recurrent Neural Networks For Multivariate Clinical Time Series](http://arxiv.org/abs/2109.07602)


  Time series models with recurrent neural networks (RNNs) can have high
accuracy but are unfortunately difficult to interpret as a result of
feature-interactions, temporal-interactions, and non-linear transformations.
Interpretability is important in domains like healthcare where constructing
models that provide insight into the relationships they have learned are
required to validate and trust model predictions. We want accurate time series
models where users can understand the contribution of individual input
features. We present the Interpretable-RNN (I-RNN) that balances model
complexity and accuracy by forcing the relationship between variables in the
model to be additive. Interactions are restricted between hidden states of the
RNN and additively combined at the final step. I-RNN specifically captures the
unique characteristics of clinical time series, which are unevenly sampled in
time, asynchronously acquired, and have missing data. Importantly, the hidden
state activations represent feature coefficients that correlate with the
prediction target and can be visualized as risk curves that capture the global
relationship between individual input features and the outcome. We evaluate the
I-RNN model on the Physionet 2012 Challenge dataset to predict in-hospital
mortality, and on a real-world clinical decision support task: predicting
hemodynamic interventions in the intensive care unit. I-RNN provides
explanations in the form of global and local feature importances comparable to
highly intelligible models like decision trees trained on hand-engineered
features while significantly outperforming them. I-RNN remains intelligible
while providing accuracy comparable to state-of-the-art decay-based and
interpolation-based recurrent time series models. The experimental results on
real-world clinical datasets refute the myth that there is a tradeoff between
accuracy and interpretability.

    

### [[2109.07611] On-the-Fly Ensemble Pruning in Evolving Data Streams](http://arxiv.org/abs/2109.07611)


  Ensemble pruning is the process of selecting a subset of componentclassifiers
from an ensemble which performs at least as well as theoriginal ensemble while
reducing storage and computational costs.Ensemble pruning in data streams is a
largely unexplored area ofresearch. It requires analysis of ensemble components
as they arerunning on the stream, and differentiation of useful classifiers
fromredundant ones. We present CCRP, an on-the-fly ensemble prun-ing method for
multi-class data stream classification empoweredby an imbalance-aware fusion of
class-wise component rankings.CCRP aims that the resulting pruned ensemble
contains the bestperforming classifier for each target class and hence, reduces
the ef-fects of class imbalance. The conducted experiments on real-worldand
synthetic data streams demonstrate that different types of en-sembles that
integrate CCRP as their pruning scheme consistentlyyield on par or superior
performance with 20% to 90% less averagememory consumption. Lastly, we validate
the proposed pruningscheme by comparing our approach against pruning schemes
basedon ensemble weights and basic rank fusion methods.

    

### [[2109.07622] Towards Zero-shot Cross-lingual Image Retrieval and Tagging](http://arxiv.org/abs/2109.07622)


  There has been a recent spike in interest in multi-modal Language and Vision
problems. On the language side, most of these models primarily focus on English
since most multi-modal datasets are monolingual. We try to bridge this gap with
a zero-shot approach for learning multi-modal representations using
cross-lingual pre-training on the text side. We present a simple yet practical
approach for building a cross-lingual image retrieval model which trains on a
monolingual training dataset but can be used in a zero-shot cross-lingual
fashion during inference. We also introduce a new objective function which
tightens the text embedding clusters by pushing dissimilar texts away from each
other. For evaluation, we introduce a new 1K multi-lingual MSCOCO2014 caption
test dataset (XTD10) in 7 languages that we collected using a crowdsourcing
platform. We use this as the test set for zero-shot model performance across
languages. We also demonstrate how a cross-lingual model can be used for
downstream tasks like multi-lingual image tagging in a zero shot manner. XTD10
dataset is made publicly available here:
this https URL.

    

### [[2109.07623] BacHMMachine: An Interpretable and Scalable Model for Algorithmic Harmonization for Four-part Baroque Chorales](http://arxiv.org/abs/2109.07623)


  Algorithmic harmonization - the automated harmonization of a musical piece
given its melodic line - is a challenging problem that has garnered much
interest from both music theorists and computer scientists. One genre of
particular interest is the four-part Baroque chorales of J.S. Bach. Methods for
algorithmic chorale harmonization typically adopt a black-box, "data-driven"
approach: they do not explicitly integrate principles from music theory but
rely on a complex learning model trained with a large amount of chorale data.
We propose instead a new harmonization model, called BacHMMachine, which
employs a "theory-driven" framework guided by music composition principles,
along with a "data-driven" model for learning compositional features within
this framework. As its name suggests, BacHMMachine uses a novel Hidden Markov
Model based on key and chord transitions, providing a probabilistic framework
for learning key modulations and chordal progressions from a given melodic
line. This allows for the generation of creative, yet musically coherent
chorale harmonizations; integrating compositional principles allows for a much
simpler model that results in vast decreases in computational burden and
greater interpretability compared to state-of-the-art algorithmic harmonization
methods, at no penalty to quality of harmonization or musicality. We
demonstrate this improvement via comprehensive experiments and Turing tests
comparing BacHMMachine to existing methods.

    

### [[2109.07627] Adversarially Regularized Policy Learning Guided by Trajectory Optimization](http://arxiv.org/abs/2109.07627)


  Recent advancement in combining trajectory optimization with function
approximation (especially neural networks) shows promise in learning complex
control policies for diverse tasks in robot systems. Despite their great
flexibility, the large neural networks for parameterizing control policies
impose significant challenges. The learned neural control policies are often
overcomplex and non-smooth, which can easily cause unexpected or diverging
robot motions. Therefore, they often yield poor generalization performance in
practice. To address this issue, we propose adVErsarially Regularized pOlicy
learNIng guided by trajeCtory optimizAtion (VERONICA) for learning smooth
control policies. Specifically, our proposed approach controls the smoothness
(local Lipschitz continuity) of the neural control policies by stabilizing the
output control with respect to the worst-case perturbation to the input state.
Our experiments on robot manipulation show that our proposed approach not only
improves the sample efficiency of neural policy learning but also enhances the
robustness of the policy against various types of disturbances, including
sensor noise, environmental uncertainty, and model mismatch.

    

### [[2109.07628] Subspace Learning for Personalized Federated Optimization](http://arxiv.org/abs/2109.07628)


  As data is generated and stored almost everywhere, learning a model from a
data-decentralized setting is a task of interest for many AI-driven service
providers. Although federated learning is settled down as the main solution in
such situations, there still exists room for improvement in terms of
personalization. Training federated learning systems usually focuses on
optimizing a global model that is identically deployed to all client devices.
However, a single global model is not sufficient for each client to be
personalized on their performance as local data assumes to be not identically
distributed across clients. We propose a method to address this situation
through the lens of ensemble learning based on the construction of a low-loss
subspace continuum that generates a high-accuracy ensemble of two endpoints
(i.e. global model and local model). We demonstrate that our method achieves
consistent gains both in personalized and unseen client evaluation settings
through extensive experiments on several standard benchmark datasets.

    

### [[2109.07630] Adaptive Control of Quadratic Costs in Linear Stochastic Differential Equations](http://arxiv.org/abs/2109.07630)


  We study a canonical problem in adaptive control; design and analysis of
policies for minimizing quadratic costs in unknown continuous-time linear
dynamical systems. We address important challenges including accuracy of
learning the unknown parameters of the underlying stochastic differential
equation, as well as full analyses of performance degradation due to
sub-optimal actions (i.e., regret). Then, an easy-to-implement algorithm for
balancing exploration versus exploitation is proposed, followed by theoretical
guarantees showing a square-root of time regret bound. Further, we present
tight results for assuring system stability and for specifying fundamental
limits for regret. To establish the presented results, multiple novel technical
frameworks are developed, which can be of independent interests.

    

### [[2109.07651] Machine-Learned HASDM Model with Uncertainty Quantification](http://arxiv.org/abs/2109.07651)


  The first thermospheric neutral mass density model with robust and reliable
uncertainty estimates is developed based on the SET HASDM density database.
This database, created by Space Environment Technologies (SET), contains 20
years of outputs from the U.S. Space Force's High Accuracy Satellite Drag Model
(HASDM), which represents the state-of-the-art for density and drag modeling.
We utilize principal component analysis (PCA) for dimensionality reduction,
creating the coefficients upon which nonlinear machine-learned (ML) regression
models are trained. These models use three unique loss functions: mean square
error (MSE), negative logarithm of predictive density (NLPD), and continuous
ranked probability score (CRPS). Three input sets are also tested, showing
improved performance when introducing time histories for geomagnetic indices.
These models leverage Monte Carlo (MC) dropout to provide uncertainty
estimates, and the use of the NLPD loss function results in well-calibrated
uncertainty estimates without sacrificing model accuracy (<10% mean absolute
error). By comparing the best HASDM-ML model to the HASDM database along
satellite orbits, we found that the model provides robust and reliable
uncertainties in the density space over all space weather conditions. A
storm-time comparison shows that HASDM-ML also supplies meaningful uncertainty
measurements during extreme events.

    

### [[2109.07690] The Neural Metric Factorization for Computational Drug Repositioning](http://arxiv.org/abs/2109.07690)


  Computational drug repositioning aims to discover new therapeutic diseases
for marketed drugs and has the advantages of low cost, short development cycle,
and high controllability compared to traditional drug development. The matrix
factorization model has become a mainstream cornerstone technique for
computational drug repositioning due to its ease of implementation and
excellent scalability. However, the matrix factorization model uses the inner
product operation to represent the association between drugs and diseases,
which is lacking in expressive ability. Moreover, the degree of similarity of
drugs or diseases could not be implied on their respective latent factor
vectors, which is not satisfy the common sense of drug discovery. Therefore, a
neural metric factorization model for computational drug repositioning is
proposed in this work. We novelly consider the latent factor vector of drugs
and diseases as a point in a high-dimensional coordinate system and propose a
generalized Euclidean distance to represent the association between drugs and
diseases to compensate for the shortcomings of the inner product operation.
Furthermore, by embedding multiple drug and disease metrics information into
the encoding space of the latent factor vector, the latent factor vectors of
similar drugs or diseases are made closer. Finally, we conduct wide analysis
experiments on two real datasets to demonstrate the effectiveness of the above
improvement points and the superiority of the NMF model.

    

### [[2109.07701] SPIN Road Mapper: Extracting Roads from Aerial Images via Spatial and Interaction Space Graph Reasoning for Autonomous Driving](http://arxiv.org/abs/2109.07701)


  Road extraction is an essential step in building autonomous navigation
systems. Detecting road segments is challenging as they are of varying widths,
bifurcated throughout the image, and are often occluded by terrain, cloud, or
other weather conditions. Using just convolution neural networks (ConvNets) for
this problem is not effective as it is inefficient at capturing distant
dependencies between road segments in the image which is essential to extract
road connectivity. To this end, we propose a Spatial and Interaction Space
Graph Reasoning (SPIN) module which when plugged into a ConvNet performs
reasoning over graphs constructed on spatial and interaction spaces projected
from the feature maps. Reasoning over spatial space extracts dependencies
between different spatial regions and other contextual information. Reasoning
over a projected interaction space helps in appropriate delineation of roads
from other topographies present in the image. Thus, SPIN extracts long-range
dependencies between road segments and effectively delineates roads from other
semantics. We also introduce a SPIN pyramid which performs SPIN graph reasoning
across multiple scales to extract multi-scale features. We propose a network
based on stacked hourglass modules and SPIN pyramid for road segmentation which
achieves better performance compared to existing methods. Moreover, our method
is computationally efficient and significantly boosts the convergence speed
during training, making it feasible for applying on large-scale high-resolution
aerial images. Code available at:
this https URL.

    

### [[2109.07702] A Multi-Task Cross-Task Learning Architecture for Ad-hoc Uncertainty Estimation in 3D Cardiac MRI Image Segmentation](http://arxiv.org/abs/2109.07702)


  Medical image segmentation has significantly benefitted thanks to deep
learning architectures. Furthermore, semi-supervised learning (SSL) has
recently been a growing trend for improving a model's overall performance by
leveraging abundant unlabeled data. Moreover, learning multiple tasks within
the same model further improves model generalizability. To generate smoother
and accurate segmentation masks from 3D cardiac MR images, we present a
Multi-task Cross-task learning consistency approach to enforce the correlation
between the pixel-level (segmentation) and the geometric-level (distance map)
tasks. Our extensive experimentation with varied quantities of labeled data in
the training sets justifies the effectiveness of our model for the segmentation
of the left atrial cavity from Gadolinium-enhanced magnetic resonance (GE-MR)
images. With the incorporation of uncertainty estimates to detect failures in
the segmentation masks generated by CNNs, our study further showcases the
potential of our model to flag low-quality segmentation from a given model.

    

### [[2109.07703] ROS-X-Habitat: Bridging the ROS Ecosystem with Embodied AI](http://arxiv.org/abs/2109.07703)


  We introduce ROS-X-Habitat, a software interface that bridges the AI Habitat
platform for embodied reinforcement learning agents with other robotics
resources via ROS. This interface not only offers standardized communication
protocols between embodied agents and simulators, but also enables
physics-based simulation. With this interface, roboticists are able to train
their own Habitat RL agents in another simulation environment or to develop
their own robotic algorithms inside Habitat Sim. Through in silico experiments,
we demonstrate that ROS-X-Habitat has minimal impact on the navigation
performance and simulation speed of Habitat agents; that a standard set of ROS
mapping, planning and navigation tools can run in the Habitat simulator, and
that a Habitat agent can run in the standard ROS simulator Gazebo.

    

### [[2109.07704] Federated Submodel Averaging](http://arxiv.org/abs/2109.07704)


  We study practical data characteristics underlying federated learning, where
non-i.i.d. data from clients have sparse features, and a certain client's local
data normally involves only a small part of the full model, called a submodel.
Due to data sparsity, the classical federated averaging (FedAvg) algorithm or
its variants will be severely slowed down, because when updating the global
model, each client's zero update of the full model excluding its submodel is
inaccurately aggregated. Therefore, we propose federated submodel averaging
(FedSubAvg), ensuring that the expectation of the global update of each model
parameter is equal to the average of the local updates of the clients who
involve it. We theoretically proved the convergence rate of FedSubAvg by
deriving an upper bound under a new metric called the element-wise gradient
norm. In particular, this new metric can characterize the convergence of
federated optimization over sparse data, while the conventional metric of
squared gradient norm used in FedAvg and its variants cannot. We extensively
evaluated FedSubAvg over both public and industrial datasets. The evaluation
results demonstrate that FedSubAvg significantly outperforms FedAvg and its
variants.

    

### [[2109.07710] Exploiting Activation based Gradient Output Sparsity to Accelerate Backpropagation in CNNs](http://arxiv.org/abs/2109.07710)


  Machine/deep-learning (ML/DL) based techniques are emerging as a driving
force behind many cutting-edge technologies, achieving high accuracy on
computer vision workloads such as image classification and object detection.
However, training these models involving large parameters is both
time-consuming and energy-hogging. In this regard, several prior works have
advocated for sparsity to speed up the of DL training and more so, the
inference phase. This work begins with the observation that during training,
sparsity in the forward and backward passes are correlated. In that context, we
investigate two types of sparsity (input and output type) inherent in gradient
descent-based optimization algorithms and propose a hardware micro-architecture
to leverage the same. Our experimental results use five state-of-the-art CNN
models on the Imagenet dataset, and show back propagation speedups in the range
of 1.69$\times$ to 5.43$\times$, compared to the dense baseline execution. By
exploiting sparsity in both the forward and backward passes, speedup
improvements range from 1.68$\times$ to 3.30$\times$ over the sparsity-agnostic
baseline execution. Our work also achieves significant reduction in training
iteration time over several previously proposed dense as well as sparse
accelerator based platforms, in addition to achieving order of magnitude energy
efficiency improvements over GPU based execution.

    

### [[2109.07711] DeepMTS: Deep Multi-task Learning for Survival Prediction in Patients with Advanced Nasopharyngeal Carcinoma using Pretreatment PET/CT](http://arxiv.org/abs/2109.07711)


  Nasopharyngeal Carcinoma (NPC) is a worldwide malignant epithelial cancer.
Survival prediction is a major concern for NPC patients, as it provides early
prognostic information that is needed to guide treatments. Recently, deep
learning, which leverages Deep Neural Networks (DNNs) to learn deep
representations of image patterns, has been introduced to the survival
prediction in various cancers including NPC. It has been reported that
image-derived end-to-end deep survival models have the potential to outperform
clinical prognostic indicators and traditional radiomics-based survival models
in prognostic performance. However, deep survival models, especially 3D models,
require large image training data to avoid overfitting. Unfortunately, medical
image data is usually scarce, especially for Positron Emission
Tomography/Computed Tomography (PET/CT) due to the high cost of PET/CT
scanning. Compared to Magnetic Resonance Imaging (MRI) or Computed Tomography
(CT) providing only anatomical information of tumors, PET/CT that provides both
anatomical (from CT) and metabolic (from PET) information is promising to
achieve more accurate survival prediction. However, we have not identified any
3D end-to-end deep survival model that applies to small PET/CT data of NPC
patients. In this study, we introduced the concept of multi-task leaning into
deep survival models to address the overfitting problem resulted from small
data. Tumor segmentation was incorporated as an auxiliary task to enhance the
model's efficiency of learning from scarce PET/CT data. Based on this idea, we
proposed a 3D end-to-end Deep Multi-Task Survival model (DeepMTS) for joint
survival prediction and tumor segmentation. Our DeepMTS can jointly learn
survival prediction and tumor segmentation using PET/CT data of only 170
patients with advanced NPC.

    

### [[2109.07713] Transferable Persona-Grounded Dialogues via Grounded Minimal Edits](http://arxiv.org/abs/2109.07713)


  Grounded dialogue models generate responses that are grounded on certain
concepts. Limited by the distribution of grounded dialogue data, models trained
on such data face the transferability challenges in terms of the data
distribution and the type of grounded concepts. To address the challenges, we
propose the grounded minimal editing framework, which minimally edits existing
responses to be grounded on the given concept. Focusing on personas, we propose
Grounded Minimal Editor (GME), which learns to edit by disentangling and
recombining persona-related and persona-agnostic parts of the response. To
evaluate persona-grounded minimal editing, we present the PersonaMinEdit
dataset, and experimental results show that GME outperforms competitive
baselines by a large margin. To evaluate the transferability, we experiment on
the test set of BlendedSkillTalk and show that GME can edit dialogue models'
responses to largely improve their persona consistency while preserving the use
of knowledge and empathy.

    

### [[2109.07719] Efficient Differentiable Simulation of Articulated Bodies](http://arxiv.org/abs/2109.07719)


  We present a method for efficient differentiable simulation of articulated
bodies. This enables integration of articulated body dynamics into deep
learning frameworks, and gradient-based optimization of neural networks that
operate on articulated bodies. We derive the gradients of the forward dynamics
using spatial algebra and the adjoint method. Our approach is an order of
magnitude faster than autodiff tools. By only saving the initial states
throughout the simulation process, our method reduces memory requirements by
two orders of magnitude. We demonstrate the utility of efficient differentiable
dynamics for articulated bodies in a variety of applications. We show that
reinforcement learning with articulated systems can be accelerated using
gradients provided by our method. In applications to control and inverse
problems, gradient-based optimization enabled by our work accelerates
convergence by more than an order of magnitude.

    

### [[2109.07723] Targeted Attack on Deep RL-based Autonomous Driving with Learned Visual Patterns](http://arxiv.org/abs/2109.07723)


  Recent studies demonstrated the vulnerability of control policies learned
through deep reinforcement learning against adversarial attacks, raising
concerns about the application of such models to risk-sensitive tasks such as
autonomous driving. Threat models for these demonstrations are limited to (1)
targeted attacks through real-time manipulation of the agent's observation, and
(2) untargeted attacks through manipulation of the physical environment. The
former assumes full access to the agent's states/observations at all times,
while the latter has no control over attack outcomes. This paper investigates
the feasibility of targeted attacks through visually learned patterns placed on
physical object in the environment, a threat model that combines the
practicality and effectiveness of the existing ones. Through analysis, we
demonstrate that a pre-trained policy can be hijacked within a time window,
e.g., performing an unintended self-parking, when an adversarial object is
present. To enable the attack, we adopt an assumption that the dynamics of both
the environment and the agent can be learned by the attacker. Lastly, we
empirically show the effectiveness of the proposed attack on different driving
scenarios, perform a location robustness test, and study the tradeoff between
the attack strength and its effectiveness.

    

### [[2109.07729] Beyond 5G RIS mmWave Systems: Where Communication and Localization Meet](http://arxiv.org/abs/2109.07729)


  Upcoming beyond fifth generation (5G) communications systems aim at further
enhancing key performance indicators and fully supporting brand new use cases
by embracing emerging techniques, e.g., reconfigurable intelligent surface
(RIS), integrated communication, localization, and sensing, and mmWave/THz
communications. The wireless intelligence empowered by state-of-the-art
artificial intelligence techniques has been widely considered at the
transceivers, and now the paradigm is deemed to be shifted to the smart control
of radio propagation environment by virtue of RISs. In this article, we argue
that to harness the full potential of RISs, localization and communication must
be tightly coupled. This is in sharp contrast to 5G and earlier generations,
where localization was a minor additional service. To support this, we first
introduce the fundamentals of RIS mmWave channel modeling, followed by RIS
channel state information acquisition and link establishment. Then, we deal
with the connection between localization and communications, from a separate
and joint perspective.

    

### [[2109.07730] Machine learning with quantum field theories](http://arxiv.org/abs/2109.07730)


  The precise equivalence between discretized Euclidean field theories and a
certain class of probabilistic graphical models, namely the mathematical
framework of Markov random fields, opens up the opportunity to investigate
machine learning from the perspective of quantum field theory. In this
contribution we will demonstrate, through the Hammersley-Clifford theorem, that
the $\phi^{4}$ scalar field theory on a square lattice satisfies the local
Markov property and can therefore be recast as a Markov random field. We will
then derive from the $\phi^{4}$ theory machine learning algorithms and neural
networks which can be viewed as generalizations of conventional neural network
architectures. Finally, we will conclude by presenting applications based on
the minimization of an asymmetric distance between the probability distribution
of the $\phi^{4}$ machine learning algorithms and target probability
distributions.

    

### [[2109.07739] A Comparative Study of Machine Learning Methods for Predicting the Evolution of Brain Connectivity from a Baseline Timepoint](http://arxiv.org/abs/2109.07739)


  Predicting the evolution of the brain network, also called connectome, by
foreseeing changes in the connectivity weights linking pairs of anatomical
regions makes it possible to spot connectivity-related neurological disorders
in earlier stages and detect the development of potential connectomic
anomalies. Remarkably, such a challenging prediction problem remains least
explored in the predictive connectomics literature. It is a known fact that
machine learning (ML) methods have proven their predictive abilities in a wide
variety of computer vision problems. However, ML techniques specifically
tailored for the prediction of brain connectivity evolution trajectory from a
single timepoint are almost absent. To fill this gap, we organized a Kaggle
competition where 20 competing teams designed advanced machine learning
pipelines for predicting the brain connectivity evolution from a single
timepoint. The competing teams developed their ML pipelines with a combination
of data pre-processing, dimensionality reduction, and learning methods.
Utilizing an inclusive evaluation approach, we ranked the methods based on two
complementary evaluation metrics (mean absolute error (MAE) and Pearson
Correlation Coefficient (PCC)) and their performances using different training
and testing data perturbation strategies (single random split and
cross-validation). The final rank was calculated using the rank product for
each competing team across all evaluation measures and validation strategies.
In support of open science, the developed 20 ML pipelines along with the
connectomic dataset are made available on GitHub. The outcomes of this
competition are anticipated to lead to the further development of predictive
models that can foresee the evolution of brain connectivity over time, as well
as other types of networks (e.g., genetic networks).

    

### [[2109.07740] Scaling Laws for Neural Machine Translation](http://arxiv.org/abs/2109.07740)


  We present an empirical study of scaling properties of encoder-decoder
Transformer models used in neural machine translation (NMT). We show that
cross-entropy loss as a function of model size follows a certain scaling law.
Specifically (i) We propose a formula which describes the scaling behavior of
cross-entropy loss as a bivariate function of encoder and decoder size, and
show that it gives accurate predictions under a variety of scaling approaches
and languages; we show that the total number of parameters alone is not
sufficient for such purposes. (ii) We observe different power law exponents
when scaling the decoder vs scaling the encoder, and provide recommendations
for optimal allocation of encoder/decoder capacity based on this observation.
(iii) We also report that the scaling behavior of the model is acutely
influenced by composition bias of the train/test sets, which we define as any
deviation from naturally generated text (either via machine generated or human
translated text). We observe that natural text on the target side enjoys
scaling, which manifests as successful reduction of the cross-entropy loss.
(iv) Finally, we investigate the relationship between the cross-entropy loss
and the quality of the generated translations. We find two different behaviors,
depending on the nature of the test data. For test sets which were originally
translated from target language to source language, both loss and BLEU score
improve as model size increases. In contrast, for test sets originally
translated from source language to target language, the loss improves, but the
BLEU score stops improving after a certain threshold. We release generated text
from all models used in this study.

    

### [[2109.07743] Optimal Probing with Statistical Guarantees for Network Monitoring at Scale](http://arxiv.org/abs/2109.07743)


  Cloud networks are difficult to monitor because they grow rapidly and the
budgets for monitoring them are limited. We propose a framework for estimating
network metrics, such as latency and packet loss, with guarantees on estimation
errors for a fixed monitoring budget. Our proposed algorithms produce a
distribution of probes across network paths, which we then monitor; and are
based on A- and E-optimal experimental designs in statistics. Unfortunately,
these designs are too computationally costly to use at production scale. We
propose their scalable and near-optimal approximations based on the Frank-Wolfe
algorithm. We validate our approaches in simulation on real network topologies,
and also using a production probing system in a real cloud network. We show
major gains in reducing the probing budget compared to both production and
academic baselines, while maintaining low estimation errors, even with very low
probing budgets.

    

### [[2109.07747] Neural-network acceleration of projection-based model-order-reduction for finite plasticity: Application to RVEs](http://arxiv.org/abs/2109.07747)


  Compared to conventional projection-based model-order-reduction, its
neural-network acceleration has the advantage that the online simulations are
equation-free, meaning that no system of equations needs to be solved
iteratively. Consequently, no stiffness matrix needs to be constructed and the
stress update needs to be computed only once per increment. In this
contribution, a recurrent neural network is developed to accelerate a
projection-based model-order-reduction of the elastoplastic mechanical
behaviour of an RVE. In contrast to a neural network that merely emulates the
relation between the macroscopic deformation (path) and the macroscopic stress,
the neural network acceleration of projection-based model-order-reduction
preserves all microstructural information, at the price of computing this
information once per increment.

    

### [[2109.07752] End-to-End Partially Observable Visual Navigation in a Diverse Environment](http://arxiv.org/abs/2109.07752)


  How can a robot navigate successfully in a rich and diverse environment,
indoors or outdoors, along an office corridor or a trail in the park, on the
flat ground, the staircase, or the elevator, etc.? To this end, this work aims
at three challenges: (i) complex visual observations, (ii) partial
observability of local sensing, and (iii) multimodal navigation behaviors that
depend on both the local environment and the high-level goal. We propose a
novel neural network (NN) architecture to represent a local controller and
leverage the flexibility of the end-to-end approach to learn a powerful policy.
To tackle complex visual observations, we extract multiscale spatial
information through convolution layers. To deal with partial observability, we
encode rich history information in LSTM-like modules. Importantly, we integrate
the two into a single unified architecture that exploits convolutional memory
cells to track the observation history at multiple spatial scales, which can
capture the complex spatiotemporal dependencies between observations and
controls. We additionally condition the network on the high-level goal in order
to generate different navigation behavior modes. Specifically, we propose to
use independent memory cells for different modes to prevent mode collapse in
the learned policy. We implemented the NN controller on the SPOT robot and
evaluate it on three challenging tasks with partial observations: adversarial
pedestrian avoidance, blind-spot obstacle avoidance, and elevator riding. Our
model significantly outperforms CNNs, conventional LSTMs, or the ablated
versions of our model. A demo video will be publicly available, showing our
SPOT robot traversing many different locations on our university campus.

    

### [[2109.07804] Detection Accuracy for Evaluating Compositional Explanations of Units](http://arxiv.org/abs/2109.07804)


  The recent success of deep learning models in solving complex problems and in
different domains has increased interest in understanding what they learn.
Therefore, different approaches have been employed to explain these models, one
of which uses human-understandable concepts as explanations. Two examples of
methods that use this approach are Network Dissection and Compositional
explanations. The former explains units using atomic concepts, while the latter
makes explanations more expressive, replacing atomic concepts with logical
forms. While intuitively, logical forms are more informative than atomic
concepts, it is not clear how to quantify this improvement, and their
evaluation is often based on the same metric that is optimized during the
search-process and on the usage of hyper-parameters to be tuned. In this paper,
we propose to use as evaluation metric the Detection Accuracy, which measures
units' consistency of detection of their assigned explanations. We show that
this metric (1) evaluates explanations of different lengths effectively, (2)
can be used as a stopping criterion for the compositional explanation search,
eliminating the explanation length hyper-parameter, and (3) exposes new
specialized units whose length 1 explanations are the perceptual abstractions
of their longer explanations.

    

### [[2109.07815] Probability-driven scoring functions in combining linear classifiers](http://arxiv.org/abs/2109.07815)


  Although linear classifiers are one of the oldest methods in machine
learning, they are still very popular in the machine learning community. This
is due to their low computational complexity and robustness to overfitting.
Consequently, linear classifiers are often used as base classifiers of multiple
ensemble classification systems. This research is aimed at building a new
fusion method dedicated to the ensemble of linear classifiers. The fusion
scheme uses both measurement space and geometrical space. Namely, we proposed a
probability-driven scoring function which shape depends on the orientation of
the decision hyperplanes generated by the base classifiers. The proposed fusion
method is compared with the reference method using multiple benchmark datasets
taken from the KEEL repository. The comparison is done using multiple quality
criteria. The statistical analysis of the obtained results is also performed.
The experimental study shows that, under certain conditions, some improvement
may be obtained.

    

### [[2109.07818] Learning logic programs through divide, constrain, and conquer](http://arxiv.org/abs/2109.07818)


  We introduce an inductive logic programming approach that combines classical
divide-and-conquer search with modern constraint-driven search. Our anytime
approach can learn optimal, recursive, and large programs and supports
predicate invention. Our experiments on three domains (classification,
inductive general game playing, and program synthesis) show that our approach
can increase predictive accuracies and reduce learning times.

    

### [[2109.07826] Directed degree corrected mixed membership model and estimating community memberships in directed networks](http://arxiv.org/abs/2109.07826)


  This paper considers the problem of modeling and estimating community
memberships of nodes in a directed network where every row (column) node is
associated with a vector determining its membership in each row (column)
community. To model such directed network, we propose directed degree corrected
mixed membership (DiDCMM) model by considering degree heterogeneity. DiDCMM is
identifiable under popular conditions for mixed membership network when
considering degree heterogeneity. Based on the cone structure inherent in the
normalized version of the left singular vectors and the simplex structure
inherent in the right singular vectors of the population adjacency matrix, we
build an efficient algorithm called DiMSC to infer the community membership
vectors for both row nodes and column nodes. By taking the advantage of DiMSC's
equivalence algorithm which returns same estimations as DiMSC and the recent
development on row-wise singular vector deviation, we show that the proposed
algorithm is asymptotically consistent under mild conditions by providing error
bounds for the inferred membership vectors of each row node and each column
node under DiDCMM. The theory is supplemented by a simulation study.

    

### [[2109.07830] Reframing Instructional Prompts to GPTk's Language](http://arxiv.org/abs/2109.07830)


  How can model designers turn task instructions into effective prompts for
language models? Backed by extensive empirical analysis on GPT3, we observe
important features for successful instructional prompts, and propose several
reframing techniques for model designers to create such prompts. For example, a
complex task can be decomposed into multiple simpler tasks. We experiment over
12 NLP tasks across 6 diverse categories (question generation, classification,
etc.). Our results show that reframing improves few-shot learning performance
by 14\% while reducing sample complexity over existing few-shot baselines. The
performance gains are particularly important on large language models, such as
GPT3 where tuning models or prompts on large datasets is not feasible.
Furthermore, we observe that such gains are not limited to GPT3; the reframed
tasks remain superior over raw instructions across different model
architectures, underscoring the cross-model generality of these guidelines. We
hope these empirical-driven techniques will pave way for more effective ways to
prompt LMs in future.

    

### [[2109.07835] Incentives in Two-sided Matching Markets with Prediction-enhanced Preference-formation](http://arxiv.org/abs/2109.07835)


  Two-sided matching markets have long existed to pair agents in the absence of
regulated exchanges. A common example is school choice, where a matching
mechanism uses student and school preferences to assign students to schools. In
such settings, forming preferences is both difficult and critical. Prior work
has suggested various prediction mechanisms that help agents make decisions
about their preferences. Although often deployed together, these matching and
prediction mechanisms are almost always analyzed separately. The present work
shows that at the intersection of the two lies a previously unexplored type of
strategic behavior: agents returning to the market (e.g., schools) can attack
future predictions by interacting short-term non-optimally with their matches.
Here, we first introduce this type of strategic behavior, which we call an
`adversarial interaction attack'. Next, we construct a formal economic model
that captures the feedback loop between prediction mechanisms designed to
assist agents and the matching mechanism used to pair them. This economic model
allows us to analyze adversarial interaction attacks. Finally, using school
choice as an example, we build a simulation to show that, as the trust in and
accuracy of predictions increases, schools gain progressively more by
initiating an adversarial interaction attack. We also show that this attack
increases inequality in the student population.

    

### [[2109.07839] Self-supervised Contrastive Learning for EEG-based Sleep Staging](http://arxiv.org/abs/2109.07839)


  EEG signals are usually simple to obtain but expensive to label. Although
supervised learning has been widely used in the field of EEG signal analysis,
its generalization performance is limited by the amount of annotated data.
Self-supervised learning (SSL), as a popular learning paradigm in computer
vision (CV) and natural language processing (NLP), can employ unlabeled data to
make up for the data shortage of supervised learning. In this paper, we propose
a self-supervised contrastive learning method of EEG signals for sleep stage
classification. During the training process, we set up a pretext task for the
network in order to match the right transformation pairs generated from EEG
signals. In this way, the network improves the representation ability by
learning the general features of EEG signals. The robustness of the network
also gets improved in dealing with diverse data, that is, extracting constant
features from changing data. In detail, the network's performance depends on
the choice of transformations and the amount of unlabeled data used in the
training process of self-supervised learning. Empirical evaluations on the
Sleep-edf dataset demonstrate the competitive performance of our method on
sleep staging (88.16% accuracy and 81.96% F1 score) and verify the
effectiveness of SSL strategy for EEG signal analysis in limited labeled data
regimes. All codes are provided publicly online.

    

### [[2109.07846] Telehealthcare and Covid-19: A Noninvasive & Low Cost Invasive, Scalable and Multimodal Real-Time Smartphone Application for Early Diagnosis of SARS-CoV-2 Infection](http://arxiv.org/abs/2109.07846)


  The global coronavirus pandemic overwhelmed many health care systems,
enforcing lockdown and encouraged work from home to control the spread of the
virus and prevent overrunning of hospitalized patients. This prompted a sharp
widespread use of telehealth to provide low-risk care for patients.
Nevertheless, a continuous mutation into new variants and widespread
unavailability of test kits, especially in developing countries, possess the
challenge to control future potential waves of infection. In this paper, we
propose a novel Smartphone application-based platform for early diagnosis of
possible Covid-19 infected patients. The application provides three modes of
diagnosis from possible symptoms, cough sound, and specific blood biomarkers.
When a user chooses a particular setting and provides the necessary
information, it sends the data to a trained machine learning (ML) model
deployed in a remote server using the internet. The ML algorithm then predicts
the possibility of contracting Covid-19 and sends the feedback to the user. The
entire procedure takes place in real-time. Our machine learning models can
identify Covid-19 patients with an accuracy of 100%, 95.65%, and 77.59% from
blood parameters, cough sound, and symptoms respectively. Moreover, the ML
sensitivity for blood and sound is 100%, which indicates correct identification
of Covid positive patients. This is significant in limiting the spread of the
virus. The multimodality offers multiplex diagnostic methods to better classify
possible infectees and together with the instantaneous nature of our technique,
demonstrates the power of telehealthcare as an easy and widespread low-cost
scalable diagnostic solution for future pandemics.

    

### [[2109.07852] OpenFed: An Open-Source Security and Privacy Guaranteed Federated Learning Framework](http://arxiv.org/abs/2109.07852)


  The broad application of artificial intelligence techniques ranging from
self-driving vehicles to advanced medical diagnostics afford many benefits.
Federated learning is a new breed of artificial intelligence, offering
techniques to help bridge the gap between personal data protection and
utilization for research and commercial deployment, especially in the use-cases
where security and privacy are the key concerns. Here, we present OpenFed, an
open-source software framework to simultaneously address the demands for data
protection and utilization. In practice, OpenFed enables state-of-the-art model
development in low-trust environments despite limited local data availability,
which lays the groundwork for sustainable collaborative model development and
commercial deployment by alleviating concerns of asset protection. In addition,
OpenFed also provides an end-to-end toolkit to facilitate federated learning
algorithm development as well as several benchmarks to fair performance
comparison under diverse computing paradigms and configurations.

    

### [[2109.07857] Soft Confusion Matrix Classifier for Stream Classification](http://arxiv.org/abs/2109.07857)


  In this paper, the issue of tailoring the soft confusion matrix (SCM) based
classifier to deal with stream learning task is addressed. The main goal of the
work is to develop a wrapping-classifier that allows incremental learning to
classifiers that are unable to learn incrementally. The goal is achieved by
making two improvements in the previously developed SCM classifier. The first
one is aimed at reducing the computational cost of the SCM classifier. To do
so, the definition of the fuzzy neighborhood of an object is changed. The
second one is aimed at effective dealing with the concept drift. This is done
by employing the ADWIN-driven concept drift detector that is not only used to
detect the drift but also to control the size of the neighbourhood. The
obtained experimental results show that the proposed approach significantly
outperforms the reference methods.

    

### [[2109.07861] Building an Ensemble of Classifiers via Randomized Models of Ensemble Members](http://arxiv.org/abs/2109.07861)


  Many dynamic ensemble selection (DES) methods are known in the literature. A
previously-developed by the authors, method consists in building a randomized
classifier which is treated as a model of the base classifier. The model is
equivalent to the base classifier in a certain probabilistic sense. Next, the
probability of correct classification of randomized classifier is taken as the
competence of the evaluated classifier.
In this paper, a novel randomized model of base classifier is developed. In
the proposed method, the random operation of the model results from a random
selection of the learning set from the family of learning sets of a fixed size.
The paper presents the mathematical foundations of this approach and shows how,
for a practical application when learning and validation sets are given, one
can determine the measure of competence and build a MC system with the DES
scheme.
The DES scheme with the proposed model of competence was experimentally
evaluated on the collection of 67 benchmark datasets and compared in terms of
eight quality criteria with two ensemble classifiers which use the
previously-proposed concepts of randomized model. The proposed approach
achieved the lowest ranks for almost all investigated quality criteria.

    

### [[2109.07865] OMPQ: Orthogonal Mixed Precision Quantization](http://arxiv.org/abs/2109.07865)


  To bridge the ever increasing gap between deep neural networks' complexity
and hardware capability, network quantization has attracted more and more
research attention. The latest trend of mixed precision quantization takes
advantage of hardware's multiple bit-width arithmetic operations to unleash the
full potential of network quantization. However, this also results in a
difficult integer programming formulation, and forces most existing approaches
to use an extremely time-consuming search process even with various
relaxations. Instead of solving a problem of the original integer programming,
we propose to optimize a proxy metric, the concept of network orthogonality,
which is highly correlated with the loss of the integer programming but also
easy to optimize with linear programming. This approach reduces the search time
and required data amount by orders of magnitude, with little compromise on
quantization accuracy. Specifically, on post-training quantization, we achieve
71.27% Top-1 accuracy on MobileNetV2, which only takes 9 seconds for searching
and 1.4 GPU hours for finetuning on ImageNet. Our codes are avaliable at
this https URL.

    

### [[2109.07867] Humanly Certifying Superhuman Classifiers](http://arxiv.org/abs/2109.07867)


  Estimating the performance of a machine learning system is a longstanding
challenge in artificial intelligence research. Today, this challenge is
especially relevant given the emergence of systems which appear to increasingly
outperform human beings. In some cases, this "superhuman" performance is
readily demonstrated; for example by defeating legendary human players in
traditional two player games. On the other hand, it can be challenging to
evaluate classification models that potentially surpass human performance.
Indeed, human annotations are often treated as a ground truth, which implicitly
assumes the superiority of the human over any models trained on human
annotations. In reality, human annotators can make mistakes and be subjective.
Evaluating the performance with respect to a genuine oracle may be more
objective and reliable, even when querying the oracle is expensive or
impossible. In this paper, we first raise the challenge of evaluating the
performance of both humans and models with respect to an oracle which is
unobserved. We develop a theory for estimating the accuracy compared to the
oracle, using only imperfect human annotations for reference. Our analysis
provides a simple recipe for detecting and certifying superhuman performance in
this setting, which we believe will assist in understanding the stage of
current research on classification. We validate the convergence of the bounds
and the assumptions of our theory on carefully designed toy experiments with
known oracles. Moreover, we demonstrate the utility of our theory by
meta-analyzing large-scale natural language processing tasks, for which an
oracle does not exist, and show that under our assumptions a number of models
from recent years are with high probability superhuman.

    

### [[2109.07869] Explainability Requires Interactivity](http://arxiv.org/abs/2109.07869)


  When explaining the decisions of deep neural networks, simple stories are
tempting but dangerous. Especially in computer vision, the most popular
explanation approaches give a false sense of comprehension to its users and
provide an overly simplistic picture. We introduce an interactive framework to
understand the highly complex decision boundaries of modern vision models. It
allows the user to exhaustively inspect, probe, and test a network's decisions.
Across a range of case studies, we compare the power of our interactive
approach to static explanation methods, showing how these can lead a user
astray, with potentially severe consequences.

    

### [[2109.07882] A Quadratic Time Locally Optimal Algorithm for NP-hard Equal Cardinality Partition Optimization](http://arxiv.org/abs/2109.07882)


  We study the optimization version of the equal cardinality set partition
problem (where the absolute difference between the equal sized partitions' sums
are minimized). While this problem is NP-hard and requires exponential
complexity to solve in general, we have formulated a weaker version of this
NP-hard problem, where the goal is to find a locally optimal solution. The
local optimality considered in our work is under any swap between the opposing
partitions' element pairs. To this end, we designed an algorithm which can
produce such a locally optimal solution in $O(N^2)$ time and $O(N)$ space. Our
approach does not require positive or integer inputs and works equally well
under arbitrary input precisions. Thus, it is widely applicable in different
problem scenarios.

    

### [[2109.07893] Efficient Scaling of Dynamic Graph Neural Networks](http://arxiv.org/abs/2109.07893)


  We present distributed algorithms for training dynamic Graph Neural Networks
(GNN) on large scale graphs spanning multi-node, multi-GPU systems. To the best
of our knowledge, this is the first scaling study on dynamic GNN. We devise
mechanisms for reducing the GPU memory usage and identify two execution time
bottlenecks: CPU-GPU data transfer; and communication volume. Exploiting
properties of dynamic graphs, we design a graph difference-based strategy to
significantly reduce the transfer time. We develop a simple, but effective data
distribution technique under which the communication volume remains fixed and
linear in the input size, for any number of GPUs. Our experiments using
billion-size graphs on a system of 128 GPUs shows that: (i) the distribution
scheme achieves up to 30x speedup on 128 GPUs; (ii) the graph-difference
technique reduces the transfer time by a factor of up to 4.1x and the overall
execution time by up to 40%

    

### [[2109.07903] Predicting students' performance in online courses using multiple data sources](http://arxiv.org/abs/2109.07903)


  Data-driven decision making is serving and transforming education. We
approached the problem of predicting students' performance by using multiple
data sources which came from online courses, including one we created.
Experimental results show preliminary conclusions towards which data are to be
considered for the task.

    

### [[2109.07904] A literature survey on student feedback assessment tools and their usage in sentiment analysis](http://arxiv.org/abs/2109.07904)


  Online learning is becoming increasingly popular, whether for convenience, to
accommodate work hours, or simply to have the freedom to study from anywhere.
Especially, during the Covid-19 pandemic, it has become the only viable option
for learning. The effectiveness of teaching various hard-core programming
courses with a mix of theoretical content is determined by the student
interaction and responses. In contrast to a digital lecture through Zoom or
Teams, a lecturer may rapidly acquire such responses from students' facial
expressions, behavior, and attitude in a physical session, even if the listener
is largely idle and non-interactive. However, student assessment in virtual
learning is a challenging task. Despite the challenges, different technologies
are progressively being integrated into teaching environments to boost student
engagement and motivation. In this paper, we evaluate the effectiveness of
various in-class feedback assessment methods such as Kahoot!, Mentimeter,
Padlet, and polling to assist a lecturer in obtaining real-time feedback from
students throughout a session and adapting the teaching style accordingly.
Furthermore, some of the topics covered by student suggestions include tutor
suggestions, enhancing teaching style, course content, and other subjects. Any
input gives the instructor valuable insight into how to improve the student's
learning experience, however, manually going through all of the qualitative
comments and extracting the ideas is tedious. Thus, in this paper, we propose a
sentiment analysis model for extracting the explicit suggestions from the
students' qualitative feedback comments.

    

### [[2109.07908] Auditing Fairness and Imputation Impact in Predictive Analytics for Higher Education](http://arxiv.org/abs/2109.07908)


  Nowadays, colleges and universities use predictive analytics in a variety of
ways to increase student success rates. Despite the potentials for predictive
analytics, there exist two major barriers to their adoption in higher
education: (a) the lack of democratization in deployment, and (b) the potential
to exacerbate inequalities. Education researchers and policymakers encounter
numerous challenges in deploying predictive modeling in practice. These
challenges present in different steps of modeling including data preparation,
model development, and evaluation. Nevertheless, each of these steps can
introduce additional bias to the system if not appropriately performed. Most
large-scale and nationally representative education data sets suffer from a
significant number of incomplete responses from the research participants.
Missing Values are the frequent latent causes behind many data analysis
challenges. While many education-related studies addressed the challenges of
missing data, little is known about the impact of handling missing values on
the fairness of predictive outcomes in practice.
In this paper, we set out to first assess the disparities in predictive
modeling outcome for college-student success, then investigate the impact of
imputation techniques on the model performance and fairness using a
comprehensive set of common metrics. The comprehensive analysis of a real
large-scale education dataset reveals key insights on the modeling disparity
and how different imputation techniques fundamentally compare to one another in
terms of their impact on the fairness of the student-success predictive
outcome.

    

### [[2109.07920] On the inductive biases of deep domain adaptation](http://arxiv.org/abs/2109.07920)


  Domain alignment is currently the most prevalent solution to unsupervised
domain-adaptation tasks and are often being presented as minimizers of some
theoretical upper-bounds on risk in the target domain. However, further works
revealed severe inadequacies between theory and practice: we consolidate this
analysis and confirm that imposing domain invariance on features is neither
necessary nor sufficient to obtain low target risk. We instead argue that
successful deep domain adaptation rely largely on hidden inductive biases found
in the common practice, such as model pre-training or design of encoder
architecture. We perform various ablation experiments on popular benchmarks and
our own synthetic transfers to illustrate their role in prototypical
situations. To conclude our analysis, we propose to meta-learn parametric
inductive biases to solve specific transfers and show their superior
performance over handcrafted heuristics.

    

### [[2109.07925] PDBench: Evaluating Computational Methods for Protein Sequence Design](http://arxiv.org/abs/2109.07925)


  Proteins perform critical processes in all living systems: converting solar
energy into chemical energy, replicating DNA, as the basis of highly performant
materials, sensing and much more. While an incredible range of functionality
has been sampled in nature, it accounts for a tiny fraction of the possible
protein universe. If we could tap into this pool of unexplored protein
structures, we could search for novel proteins with useful properties that we
could apply to tackle the environmental and medical challenges facing humanity.
This is the purpose of protein design.
Sequence design is an important aspect of protein design, and many successful
methods to do this have been developed. Recently, deep-learning methods that
frame it as a classification problem have emerged as a powerful approach.
Beyond their reported improvement in performance, their primary advantage over
physics-based methods is that the computational burden is shifted from the user
to the developers, thereby increasing accessibility to the design method.
Despite this trend, the tools for assessment and comparison of such models
remain quite generic. The goal of this paper is to both address the timely
problem of evaluation and to shine a spotlight, within the Machine Learning
community, on specific assessment criteria that will accelerate impact.
We present a carefully curated benchmark set of proteins and propose a number
of standard tests to assess the performance of deep learning based methods. Our
robust benchmark provides biological insight into the behaviour of design
methods, which is essential for evaluating their performance and utility. We
compare five existing models with two novel models for sequence prediction.
Finally, we test the designs produced by these models with AlphaFold2, a
state-of-the-art structure-prediction algorithm, to determine if they are
likely to fold into the intended 3D shapes.

    

### [[2109.07930] Behavior of Keyword Spotting Networks Under Noisy Conditions](http://arxiv.org/abs/2109.07930)


  Keyword spotting (KWS) is becoming a ubiquitous need with the advancement in
artificial intelligence and smart devices. Recent work in this field have
focused on several different architectures to achieve good results on datasets
with low to moderate noise. However, the performance of these models
deteriorates under high noise conditions as shown by our experiments. In our
paper, we present an extensive comparison between state-of-the-art KWS networks
under various noisy conditions. We also suggest adaptive batch normalization as
a technique to improve the performance of the networks when the noise files are
unknown during the training phase. The results of such high noise
characterization enable future work in developing models that perform better in
the aforementioned conditions.

    

### [[2109.07945] Lifting 2D Object Locations to 3D by Discounting LiDAR Outliers across Objects and Views](http://arxiv.org/abs/2109.07945)


  We present a system for automatic converting of 2D mask object predictions
and raw LiDAR point clouds into full 3D bounding boxes of objects. Because the
LiDAR point clouds are partial, directly fitting bounding boxes to the point
clouds is meaningless. Instead, we suggest that obtaining good results requires
sharing information between \emph{all} objects in the dataset jointly, over
multiple frames. We then make three improvements to the baseline. First, we
address ambiguities in predicting the object rotations via direct optimization
in this space while still backpropagating rotation prediction through the
model. Second, we explicitly model outliers and task the network with learning
their typical patterns, thus better discounting them. Third, we enforce
temporal consistency when video data is available. With these contributions,
our method significantly outperforms previous work despite the fact that those
methods use significantly more complex pipelines, 3D models and additional
human-annotated external sources of prior information.

    

### [[2109.07955] Quality-aware Cine Cardiac MRI Reconstruction and Analysis from Undersampled k-space Data](http://arxiv.org/abs/2109.07955)


  Cine cardiac MRI is routinely acquired for the assessment of cardiac health,
but the imaging process is slow and typically requires several breath-holds to
acquire sufficient k-space profiles to ensure good image quality. Several
undersampling-based reconstruction techniques have been proposed during the
last decades to speed up cine cardiac MRI acquisition. However, the
undersampling factor is commonly fixed to conservative values before
acquisition to ensure diagnostic image quality, potentially leading to
unnecessarily long scan times. In this paper, we propose an end-to-end
quality-aware cine short-axis cardiac MRI framework that combines image
acquisition and reconstruction with downstream tasks such as segmentation,
volume curve analysis and estimation of cardiac functional parameters. The goal
is to reduce scan time by acquiring only a fraction of k-space data to enable
the reconstruction of images that can pass quality control checks and produce
reliable estimates of cardiac functional parameters. The framework consists of
a deep learning model for the reconstruction of 2D+t cardiac cine MRI images
from undersampled data, an image quality-control step to detect good quality
reconstructions, followed by a deep learning model for bi-ventricular
segmentation, a quality-control step to detect good quality segmentations and
automated calculation of cardiac functional parameters. To demonstrate the
feasibility of the proposed approach, we perform simulations using a cohort of
selected participants from the UK Biobank (n=270), 200 healthy subjects and 70
patients with cardiomyopathies. Our results show that we can produce
quality-controlled images in a scan time reduced from 12 to 4 seconds per
slice, enabling reliable estimates of cardiac functional parameters such as
ejection fraction within 5% mean absolute error.

    

### [[2109.07958] TruthfulQA: Measuring How Models Mimic Human Falsehoods](http://arxiv.org/abs/2109.07958)


  We propose a benchmark to measure whether a language model is truthful in
generating answers to questions. The benchmark comprises 817 questions that
span 38 categories, including health, law, finance and politics. We crafted
questions that some humans would answer falsely due to a false belief or
misconception. To perform well, models must avoid generating false answers
learned from imitating human texts. We tested GPT-3, GPT-Neo/J, GPT-2 and a
T5-based model. The best model was truthful on 58% of questions, while human
performance was 94%. Models generated many false answers that mimic popular
misconceptions and have the potential to deceive humans. The largest models
were generally the least truthful. For example, the 6B-parameter GPT-J model
was 17% less truthful than its 125M-parameter counterpart. This contrasts with
other NLP tasks, where performance improves with model size. However, this
result is expected if false answers are learned from the training distribution.
We suggest that scaling up models alone is less promising for improving
truthfulness than fine-tuning using training objectives other than imitation of
text from the web.

    

### [[2109.07991] ObjectFolder: A Dataset of Objects with Implicit Visual, Auditory, and Tactile Representations](http://arxiv.org/abs/2109.07991)


  Multisensory object-centric perception, reasoning, and interaction have been
a key research topic in recent years. However, the progress in these directions
is limited by the small set of objects available -- synthetic objects are not
realistic enough and are mostly centered around geometry, while real object
datasets such as YCB are often practically challenging and unstable to acquire
due to international shipping, inventory, and financial cost. We present
ObjectFolder, a dataset of 100 virtualized objects that addresses both
challenges with two key innovations. First, ObjectFolder encodes the visual,
auditory, and tactile sensory data for all objects, enabling a number of
multisensory object recognition tasks, beyond existing datasets that focus
purely on object geometry. Second, ObjectFolder employs a uniform,
object-centric, and implicit representation for each object's visual textures,
acoustic simulations, and tactile readings, making the dataset flexible to use
and easy to share. We demonstrate the usefulness of our dataset as a testbed
for multisensory perception and control by evaluating it on a variety of
benchmark tasks, including instance recognition, cross-sensory retrieval, 3D
reconstruction, and robotic grasping.

    

### [[2109.07994] KnowMAN: Weakly Supervised Multinomial Adversarial Networks](http://arxiv.org/abs/2109.07994)


  The absence of labeled data for training neural models is often addressed by
leveraging knowledge about the specific task, resulting in heuristic but noisy
labels. The knowledge is captured in labeling functions, which detect certain
regularities or patterns in the training samples and annotate corresponding
labels for training. This process of weakly supervised training may result in
an over-reliance on the signals captured by the labeling functions and hinder
models to exploit other signals or to generalize well. We propose KnowMAN, an
adversarial scheme that enables to control influence of signals associated with
specific labeling functions. KnowMAN forces the network to learn
representations that are invariant to those signals and to pick up other
signals that are more generally associated with an output label. KnowMAN
strongly improves results compared to direct weakly supervised learning with a
pre-trained transformer language model and a feature-based baseline.

    

### [[2109.08002] SAFRAN: An interpretable, rule-based link prediction method outperforming embedding models](http://arxiv.org/abs/2109.08002)


  Neural embedding-based machine learning models have shown promise for
predicting novel links in knowledge graphs. Unfortunately, their practical
utility is diminished by their lack of interpretability. Recently, the fully
interpretable, rule-based algorithm AnyBURL yielded highly competitive results
on many general-purpose link prediction benchmarks. However, current approaches
for aggregating predictions made by multiple rules are affected by
redundancies. We improve upon AnyBURL by introducing the SAFRAN rule
application framework, which uses a novel aggregation approach called
Non-redundant Noisy-OR that detects and clusters redundant rules prior to
aggregation. SAFRAN yields new state-of-the-art results for fully interpretable
link prediction on the established general-purpose benchmarks FB15K-237, WN18RR
and YAGO3-10. Furthermore, it exceeds the results of multiple established
embedding-based algorithms on FB15K-237 and WN18RR and narrows the gap between
rule-based and embedding-based algorithms on YAGO3-10.

    

### [[2109.08010] WildWood: a new Random Forest algorithm](http://arxiv.org/abs/2109.08010)


  We introduce WildWood (WW), a new ensemble algorithm for supervised learning
of Random Forest (RF) type. While standard RF algorithms use bootstrap
out-of-bag samples to compute out-of-bag scores, WW uses these samples to
produce improved predictions given by an aggregation of the predictions of all
possible subtrees of each fully grown tree in the forest. This is achieved by
aggregation with exponential weights computed over out-of-bag samples, that are
computed exactly and very efficiently thanks to an algorithm called context
tree weighting. This improvement, combined with a histogram strategy to
accelerate split finding, makes WW fast and competitive compared with other
well-established ensemble methods, such as standard RF and extreme gradient
boosting algorithms.

    

### [[2109.08013] Detecting Propaganda Techniques in Memes](http://arxiv.org/abs/2109.08013)


  Propaganda can be defined as a form of communication that aims to influence
the opinions or the actions of people towards a specific goal; this is achieved
by means of well-defined rhetorical and psychological devices. Propaganda, in
the form we know it today, can be dated back to the beginning of the 17th
century. However, it is with the advent of the Internet and the social media
that it has started to spread on a much larger scale than before, thus becoming
major societal and political issue. Nowadays, a large fraction of propaganda in
social media is multimodal, mixing textual with visual content. With this in
mind, here we propose a new multi-label multimodal task: detecting the type of
propaganda techniques used in memes. We further create and release a new corpus
of 950 memes, carefully annotated with 22 propaganda techniques, which can
appear in the text, in the image, or in both. Our analysis of the corpus shows
that understanding both modalities together is essential for detecting these
techniques. This is further confirmed in our experiments with several
state-of-the-art multimodal models.

    

### [[2109.08017] Super-resolution data assimilation](http://arxiv.org/abs/2109.08017)


  Increasing the resolution of a model can improve the performance of a data
assimilation system: first because model field are in better agreement with
high resolution observations, then the corrections are better sustained and,
with ensemble data assimilation, the forecast error covariances are improved.
However, resolution increase is associated with a cubical increase of the
computational costs. Here we are testing an approach inspired from images
super-resolution techniques and called "Super-resolution data assimilation"
(SRDA). Starting from a low-resolution forecast, a neural network (NN) emulates
a high-resolution field that is then used to assimilate high-resolution
observations. We apply the SRDA to a quasi-geostrophic model representing
simplified surface ocean dynamics, with a model resolution up to four times
lower than the reference high-resolution and we use the Ensemble Kalman Filter
data assimilation method. We show that SRDA outperforms the low-resolution data
assimilation approach and a SRDA version with cubic spline interpolation
instead of NN. The NN's ability to anticipate the systematic differences
between low and high resolution model dynamics explains the enhanced
performance, for example by correcting the difference of propagation speed of
eddies. Increasing the computational cost by 55\% above the LR data
assimilation system (using a 25-members ensemble), the SRDA reduces the errors
by 40\% making the performance very close to the HR system (16\% larger,
compared to 92\% larger for the LR EnKF). The reliability of the ensemble
system is not degraded by SRDA.

    

### [[2109.08020] The pitfalls of using open data to develop deep learning solutions for COVID-19 detection in chest X-rays](http://arxiv.org/abs/2109.08020)


  Since the emergence of COVID-19, deep learning models have been developed to
identify COVID-19 from chest X-rays. With little to no direct access to
hospital data, the AI community relies heavily on public data comprising
numerous data sources. Model performance results have been exceptional when
training and testing on open-source data, surpassing the reported capabilities
of AI in pneumonia-detection prior to the COVID-19 outbreak. In this study
impactful models are trained on a widely used open-source data and tested on an
external test set and a hospital dataset, for the task of classifying chest
X-rays into one of three classes: COVID-19, non-COVID pneumonia and
no-pneumonia. Classification performance of the models investigated is
evaluated through ROC curves, confusion matrices and standard classification
metrics. Explainability modules are implemented to explore the image features
most important to classification. Data analysis and model evaluations show that
the popular open-source dataset COVIDx is not representative of the real
clinical problem and that results from testing on this are inflated. Dependence
on open-source data can leave models vulnerable to bias and confounding
variables, requiring careful analysis to develop clinically useful/viable AI
tools for COVID-19 detection in chest X-rays.

    

### [[2109.08021] Predicting Users' Value Changes by the Friends' Influence from Social Media Usage](http://arxiv.org/abs/2109.08021)


  Basic human values represent a set of values such as security, independence,
success, kindness, and pleasure, which we deem important to our lives. Each of
us holds different values with different degrees of significance. Existing
studies show that values of a person can be identified from their social
network usage. However, the value priority of a person may change over time due
to different factors such as life experiences, influence, social structure and
technology. Existing studies do not conduct any analysis regarding the change
of users' value from the social influence, i.e., group persuasion, form the
social media usage. In our research, first, we predict users' value score by
the influence of friends from their social media usage. We propose a Bounded
Confidence Model (BCM) based value dynamics model from 275 different ego
networks in Facebook that predicts how social influence may persuade a person
to change their value over time. Then, to predict better, we use particle swarm
optimization based hyperparameter tuning technique. We observe that these
optimized hyperparameters produce accurate future value score. We also run our
approach with different machine learning based methods and find support vector
regression (SVR) outperforms other regressor models. By using SVR with the best
hyperparameters of BCM model, we find the lowest Mean Squared Error (MSE) score
0.00347.

    

### [[2109.08031] Accurately Modeling Biased Random Walks on Weighted Graphs Using $\textit{Node2vec+}$](http://arxiv.org/abs/2109.08031)


  Node embedding is a powerful approach for representing the structural role of
each node in a graph. $\textit{Node2vec}$ is a widely used method for node
embedding that works by exploring the local neighborhoods via biased random
walks on the graph. However, $\textit{node2vec}$ does not consider edge weights
when computing walk biases. This intrinsic limitation prevents
$\textit{node2vec}$ from leveraging all the information in weighted graphs and,
in turn, limits its application to many real-world networks that are weighted
and dense. Here, we naturally extend $\textit{node2vec}$ to
$\textit{node2vec+}$ in a way that accounts for edge weights when calculating
walk biases, but which reduces to $\textit{node2vec}$ in the cases of
unweighted graphs or unbiased walks. We empirically show that
$\textit{node2vec+}$ is more robust to additive noise than $\textit{node2vec}$
in weighted graphs using two synthetic datasets. We also demonstrate that
$\textit{node2vec+}$ significantly outperforms $\textit{node2vec}$ on a
commonly benchmarked multi-label dataset (Wikipedia). Furthermore, we test
$\textit{node2vec+}$ against GCN and GraphSAGE using various challenging gene
classification tasks on two protein-protein interaction networks. Despite some
clear advantages of GCN and GraphSAGE, they show comparable performance with
$\textit{node2vec+}$. Finally, $\textit{node2vec+}$ can be used as a general
approach for generating biased random walks, benefiting all existing methods
built on top of $\textit{node2vec}$. $\textit{Node2vec+}$ is implemented as
part of $\texttt{PecanPy}$, which is available at
this https URL .

    

### [[2109.08045] Membership Inference Attacks Against Recommender Systems](http://arxiv.org/abs/2109.08045)


  Recently, recommender systems have achieved promising performances and become
one of the most widely used web applications. However, recommender systems are
often trained on highly sensitive user data, thus potential data leakage from
recommender systems may lead to severe privacy problems.
In this paper, we make the first attempt on quantifying the privacy leakage
of recommender systems through the lens of membership inference. In contrast
with traditional membership inference against machine learning classifiers, our
attack faces two main differences. First, our attack is on the user-level but
not on the data sample-level. Second, the adversary can only observe the
ordered recommended items from a recommender system instead of prediction
results in the form of posterior probabilities. To address the above
challenges, we propose a novel method by representing users from relevant
items. Moreover, a shadow recommender is established to derive the labeled
training data for training the attack model. Extensive experimental results
show that our attack framework achieves a strong performance. In addition, we
design a defense mechanism to effectively mitigate the membership inference
threat of recommender systems.

    

### [[2109.08049] A Machine Learning Framework for Automatic Prediction of Human Semen Motility](http://arxiv.org/abs/2109.08049)


  In the field of reproductive health, a vital aspect for the detection of male
fertility issues is the analysis of human semen quality. Two factors of
importance are the morphology and motility of the sperm cells. While the former
describes defects in different parts of a spermatozoon, the latter measures the
efficient movement of cells. For many non-human species, so-called
Computer-Aided Sperm Analysis systems work well for assessing these
characteristics from microscopic video recordings but struggle with human sperm
samples which generally show higher degrees of debris and dead spermatozoa, as
well as lower overall sperm motility. Here, machine learning methods that
harness large amounts of training data to extract salient features could
support physicians with the detection of fertility issues or in vitro
fertilisation procedures. In this work, the overall motility of given sperm
samples is predicted with the help of a machine learning framework integrating
unsupervised methods for feature extraction with downstream regression models.
The models evaluated herein improve on the state-of-the-art for video-based
sperm-motility prediction.

    

### [[2109.08051] Frame by frame completion probability of an NFL pass](http://arxiv.org/abs/2109.08051)


  American football is an increasingly popular sport, with a growing audience
in many countries in the world. The most watched American football league in
the world is the United States' National Football League (NFL), where every
offensive play can be either a run or a pass, and in this work we focus on
passes. Many factors can affect the probability of pass completion, such as
receiver separation from the nearest defender, distance from receiver to
passer, offense formation, among many others. When predicting the completion
probability of a pass, it is essential to know who the target of the pass is.
By using distance measures between players and the ball, it is possible to
calculate empirical probabilities and predict very accurately who the target
will be. The big question is: how likely is it for a pass to be completed in an
NFL match while the ball is in the air? We developed a machine learning
algorithm to answer this based on several predictors. Using data from the 2018
NFL season, we obtained conditional and marginal predictions for pass
completion probability based on a random forest model. This is based on a
two-stage procedure: first, we calculate the probability of each offensive
player being the pass target, then, conditional on the target, we predict
completion probability based on the random forest model. Finally, the general
completion probability can be calculated using the law of total probability. We
present animations for selected plays and show the pass completion probability
evolution.

    

### [[2109.08052] Semi-Supervised Visual Representation Learning for Fashion Compatibility](http://arxiv.org/abs/2109.08052)


  We consider the problem of complementary fashion prediction. Existing
approaches focus on learning an embedding space where fashion items from
different categories that are visually compatible are closer to each other.
However, creating such labeled outfits is intensive and also not feasible to
generate all possible outfit combinations, especially with large fashion
catalogs. In this work, we propose a semi-supervised learning approach where we
leverage large unlabeled fashion corpus to create pseudo-positive and
pseudo-negative outfits on the fly during training. For each labeled outfit in
a training batch, we obtain a pseudo-outfit by matching each item in the
labeled outfit with unlabeled items. Additionally, we introduce consistency
regularization to ensure that representation of the original images and their
transformations are consistent to implicitly incorporate colour and other
important attributes through self-supervision. We conduct extensive experiments
on Polyvore, Polyvore-D and our newly created large-scale Fashion Outfits
datasets, and show that our approach with only a fraction of labeled examples
performs on-par with completely supervised methods.

    

### [[2109.08060] Urdu text in natural scene images: a new dataset and preliminary text detection](http://arxiv.org/abs/2109.08060)


  Text detection in natural scene images for content analysis is an interesting
task. The research community has seen some great developments for
English/Mandarin text detection. However, Urdu text extraction in natural scene
images is a task not well addressed. In this work, firstly, a new dataset is
introduced for Urdu text in natural scene images. The dataset comprises of 500
standalone images acquired from real scenes. Secondly, the channel enhanced
Maximally Stable Extremal Region (MSER) method is applied to extract Urdu text
regions as candidates in an image. Two-stage filtering mechanism is applied to
eliminate non-candidate regions. In the first stage, text and noise are
classified based on their geometric properties. In the second stage, a support
vector machine classifier is trained to discard non-text candidate regions.
After this, text candidate regions are linked using centroid-based vertical and
horizontal distances. Text lines are further analyzed by a different classifier
based on HOG features to remove non-text regions. Extensive experimentation is
performed on the locally developed dataset to evaluate the performance. The
experimental results show good performance on test set images. The dataset will
be made available for research use. To the best of our knowledge, the work is
the first of its kind for the Urdu language and would provide a good dataset
for free research use and serve as a baseline performance on the task of Urdu
text extraction.

    

### [[2109.08063] Associative Memories via Predictive Coding](http://arxiv.org/abs/2109.08063)


  Associative memories in the brain receive and store patterns of activity
registered by the sensory neurons, and are able to retrieve them when
necessary. Due to their importance in human intelligence, computational models
of associative memories have been developed for several decades now. They
include autoassociative memories, which allow for storing data points and
retrieving a stored data point $s$ when provided with a noisy or partial
variant of $s$, and heteroassociative memories, able to store and recall
multi-modal data. In this paper, we present a novel neural model for realizing
associative memories, based on a hierarchical generative network that receives
external stimuli via sensory neurons. This model is trained using predictive
coding, an error-based learning algorithm inspired by information processing in
the cortex. To test the capabilities of this model, we perform multiple
retrieval experiments from both corrupted and incomplete data points. In an
extensive comparison, we show that this new model outperforms in retrieval
accuracy and robustness popular associative memory models, such as autoencoders
trained via backpropagation, and modern Hopfield networks. In particular, in
completing partial data points, our model achieves remarkable results on
natural image datasets, such as ImageNet, with a surprisingly high accuracy,
even when only a tiny fraction of pixels of the original images is presented.
Furthermore, we show that this method is able to handle multi-modal data,
retrieving images from descriptions, and vice versa. We conclude by discussing
the possible impact of this work in the neuroscience community, by showing that
our model provides a plausible framework to study learning and retrieval of
memories in the brain, as it closely mimics the behavior of the hippocampus as
a memory index and generative model.

    

### [[2109.08075] Field Study in Deploying Restless Multi-Armed Bandits: Assisting Non-Profits in Improving Maternal and Child Health](http://arxiv.org/abs/2109.08075)


  The widespread availability of cell phones has enabled non-profits to deliver
critical health information to their beneficiaries in a timely manner. This
paper describes our work to assist non-profits that employ automated messaging
programs to deliver timely preventive care information to beneficiaries (new
and expecting mothers) during pregnancy and after delivery. Unfortunately, a
key challenge in such information delivery programs is that a significant
fraction of beneficiaries drop out of the program. Yet, non-profits often have
limited health-worker resources (time) to place crucial service calls for live
interaction with beneficiaries to prevent such engagement drops. To assist
non-profits in optimizing this limited resource, we developed a Restless
Multi-Armed Bandits (RMABs) system. One key technical contribution in this
system is a novel clustering method of offline historical data to infer unknown
RMAB parameters. Our second major contribution is evaluation of our RMAB system
in collaboration with an NGO, via a real-world service quality improvement
study. The study compared strategies for optimizing service calls to 23003
participants over a period of 7 weeks to reduce engagement drops. We show that
the RMAB group provides statistically significant improvement over other
comparison groups, reducing ~ 30% engagement drops. To the best of our
knowledge, this is the first study demonstrating the utility of RMABs in real
world public health settings. We are transitioning our RMAB system to the NGO
for real-world use.

    

### [[2109.08078] Weighted Graph-Based Signal Temporal Logic Inference Using Neural Networks](http://arxiv.org/abs/2109.08078)


  Extracting spatial-temporal knowledge from data is useful in many
applications. It is important that the obtained knowledge is
human-interpretable and amenable to formal analysis. In this paper, we propose
a method that trains neural networks to learn spatial-temporal properties in
the form of weighted graph-based signal temporal logic (wGSTL) formulas. For
learning wGSTL formulas, we introduce a flexible wGSTL formula structure in
which the user's preference can be applied in the inferred wGSTL formulas. In
the proposed framework, each neuron of the neural networks corresponds to a
subformula in a flexible wGSTL formula structure. We initially train a neural
network to learn the wGSTL operators and then train a second neural network to
learn the parameters in a flexible wGSTL formula structure. We use a COVID-19
dataset and a rain prediction dataset to evaluate the performance of the
proposed framework and algorithms. We compare the performance of the proposed
framework with three baseline classification methods including K-nearest
neighbors, decision trees, and artificial neural networks. The classification
accuracy obtained by the proposed framework is comparable with the baseline
classification methods.

    

### [[2109.08079] Zero-Shot Open Information Extraction using Question Generation and Reading Comprehension](http://arxiv.org/abs/2109.08079)


  Typically, Open Information Extraction (OpenIE) focuses on extracting
triples, representing a subject, a relation, and the object of the relation.
However, most of the existing techniques are based on a predefined set of
relations in each domain which limits their applicability to newer domains
where these relations may be unknown such as financial documents. This paper
presents a zero-shot open information extraction technique that extracts the
entities (value) and their descriptions (key) from a sentence, using off the
shelf machine reading comprehension (MRC) Model. The input questions to this
model are created using a novel noun phrase generation method. This method
takes the context of the sentence into account and can create a wide variety of
questions making our technique domain independent. Given the questions and the
sentence, our technique uses the MRC model to extract entities (value). The
noun phrase corresponding to the question, with the highest confidence, is
taken as the description (key).
This paper also introduces the EDGAR10-Q dataset which is based on publicly
available financial documents from corporations listed in US securities and
exchange commission (SEC). The dataset consists of paragraphs, tagged values
(entities), and their keys (descriptions) and is one of the largest among
entity extraction datasets. This dataset will be a valuable addition to the
research community, especially in the financial domain. Finally, the paper
demonstrates the efficacy of the proposed technique on the EDGAR10-Q and Ade
corpus drug dosage datasets, where it obtained 86.84 % and 97% accuracy,
respectively.

    

### [[2109.08090] DisUnknown: Distilling Unknown Factors for Disentanglement Learning](http://arxiv.org/abs/2109.08090)


  Disentangling data into interpretable and independent factors is critical for
controllable generation tasks. With the availability of labeled data,
supervision can help enforce the separation of specific factors as expected.
However, it is often expensive or even impossible to label every single factor
to achieve fully-supervised disentanglement. In this paper, we adopt a general
setting where all factors that are hard to label or identify are encapsulated
as a single unknown factor. Under this setting, we propose a flexible
weakly-supervised multi-factor disentanglement framework DisUnknown, which
Distills Unknown factors for enabling multi-conditional generation regarding
both labeled and unknown factors. Specifically, a two-stage training approach
is adopted to first disentangle the unknown factor with an effective and robust
training method, and then train the final generator with the proper
disentanglement of all labeled factors utilizing the unknown distillation. To
demonstrate the generalization capacity and scalability of our method, we
evaluate it on multiple benchmark datasets qualitatively and quantitatively and
further apply it to various real-world applications on complicated datasets.

    

### [[2109.08098] MOFSimplify: Machine Learning Models with Extracted Stability Data of Three Thousand Metal-Organic Frameworks](http://arxiv.org/abs/2109.08098)


  We report a workflow and the output of a natural language processing
(NLP)-based procedure to mine the extant metal-organic framework (MOF)
literature describing structurally characterized MOFs and their solvent removal
and thermal stabilities. We obtain over 2,000 solvent removal stability
measures from text mining and 3,000 thermal decomposition temperatures from
thermogravimetric analysis data. We assess the validity of our NLP methods and
the accuracy of our extracted data by comparing to a hand-labeled subset.
Machine learning (ML, i.e. artificial neural network) models trained on this
data using graph- and pore-geometry-based representations enable prediction of
stability on new MOFs with quantified uncertainty. Our web interface,
MOFSimplify, provides users access to our curated data and enables them to
harness that data for predictions on new MOFs. MOFSimplify also encourages
community feedback on existing data and on ML model predictions for
community-based active learning for improved MOF stability models.

    

### [[2109.08119] Personalized Federated Learning for Heterogeneous Clients with Clustered Knowledge Transfer](http://arxiv.org/abs/2109.08119)


  Personalized federated learning (FL) aims to train model(s) that can perform
well for individual clients that are highly data and system heterogeneous. Most
work in personalized FL, however, assumes using the same model architecture at
all clients and increases the communication cost by sending/receiving models.
This may not be feasible for realistic scenarios of FL. In practice, clients
have highly heterogeneous system-capabilities and limited communication
resources. In our work, we propose a personalized FL framework, PerFed-CKT,
where clients can use heterogeneous model architectures and do not directly
communicate their model parameters. PerFed-CKT uses clustered co-distillation,
where clients use logits to transfer their knowledge to other clients that have
similar data-distributions. We theoretically show the convergence and
generalization properties of PerFed-CKT and empirically show that PerFed-CKT
achieves high test accuracy with several orders of magnitude lower
communication cost compared to the state-of-the-art personalized FL schemes.

    

### [[2109.08128] Conservative Data Sharing for Multi-Task Offline Reinforcement Learning](http://arxiv.org/abs/2109.08128)


  Offline reinforcement learning (RL) algorithms have shown promising results
in domains where abundant pre-collected data is available. However, prior
methods focus on solving individual problems from scratch with an offline
dataset without considering how an offline RL agent can acquire multiple
skills. We argue that a natural use case of offline RL is in settings where we
can pool large amounts of data collected in various scenarios for solving
different tasks, and utilize all of this data to learn behaviors for all the
tasks more effectively rather than training each one in isolation. However,
sharing data across all tasks in multi-task offline RL performs surprisingly
poorly in practice. Thorough empirical analysis, we find that sharing data can
actually exacerbate the distributional shift between the learned policy and the
dataset, which in turn can lead to divergence of the learned policy and poor
performance. To address this challenge, we develop a simple technique for
data-sharing in multi-task offline RL that routes data based on the improvement
over the task-specific data. We call this approach conservative data sharing
(CDS), and it can be applied with multiple single-task offline RL methods. On a
range of challenging multi-task locomotion, navigation, and vision-based
robotic manipulation problems, CDS achieves the best or comparable performance
compared to prior offline multi-task RL methods and previous data sharing
approaches.

    

### [[2109.08131] Studying Up Machine Learning Data: Why Talk About Bias When We Mean Power?](http://arxiv.org/abs/2109.08131)


  Research in machine learning (ML) has primarily argued that models trained on
incomplete or biased datasets can lead to discriminatory outputs. In this
commentary, we propose moving the research focus beyond bias-oriented framings
by adopting a power-aware perspective to "study up" ML datasets. This means
accounting for historical inequities, labor conditions, and epistemological
standpoints inscribed in data. We draw on HCI and CSCW work to support our
argument, critically analyze previous research, and point at two co-existing
lines of work within our community -- one bias-oriented, the other power-aware.
This way, we highlight the need for dialogue and cooperation in three areas:
data quality, data work, and data documentation. In the first area, we argue
that reducing societal problems to "bias" misses the context-based nature of
data. In the second one, we highlight the corporate forces and market
imperatives involved in the labor of data workers that subsequently shape ML
datasets. Finally, we propose expanding current transparency-oriented efforts
in dataset documentation to reflect the social contexts of data design and
production.

    

### [[2109.08134] Comparison and Unification of Three Regularization Methods in Batch Reinforcement Learning](http://arxiv.org/abs/2109.08134)


  In batch reinforcement learning, there can be poorly explored state-action
pairs resulting in poorly learned, inaccurate models and poorly performing
associated policies. Various regularization methods can mitigate the problem of
learning overly-complex models in Markov decision processes (MDPs), however
they operate in technically and intuitively distinct ways and lack a common
form in which to compare them. This paper unifies three regularization methods
in a common framework -- a weighted average transition matrix. Considering
regularization methods in this common form illuminates how the MDP structure
and the state-action pair distribution of the batch data set influence the
relative performance of regularization methods. We confirm intuitions generated
from the common framework by empirical evaluation across a range of MDPs and
data collection policies.

    

### [[2109.08141] An End-to-End Transformer Model for 3D Object Detection](http://arxiv.org/abs/2109.08141)


  We propose 3DETR, an end-to-end Transformer based object detection model for
3D point clouds. Compared to existing detection methods that employ a number of
3D-specific inductive biases, 3DETR requires minimal modifications to the
vanilla Transformer block. Specifically, we find that a standard Transformer
with non-parametric queries and Fourier positional embeddings is competitive
with specialized architectures that employ libraries of 3D-specific operators
with hand-tuned hyperparameters. Nevertheless, 3DETR is conceptually simple and
easy to implement, enabling further improvements by incorporating 3D domain
knowledge. Through extensive experiments, we show 3DETR outperforms the
well-established and highly optimized VoteNet baselines on the challenging
ScanNetV2 dataset by 9.5%. Furthermore, we show 3DETR is applicable to 3D tasks
beyond detection, and can serve as a building block for future research.

    

### [[1905.12686] Learning Representations by Humans, for Humans](http://arxiv.org/abs/1905.12686)


  When machine predictors can achieve higher performance than the human
decision-makers they support, improving the performance of human
decision-makers is often conflated with improving machine accuracy. Here we
propose a framework to directly support human decision-making, in which the
role of machines is to reframe problems rather than to prescribe actions
through prediction. Inspired by the success of representation learning in
improving performance of machine predictors, our framework learns human-facing
representations optimized for human performance. This "Mind Composed with
Machine" framework incorporates a human decision-making model directly into the
representation learning paradigm and is trained with a novel human-in-the-loop
training procedure. We empirically demonstrate the successful application of
the framework to various tasks and representational forms.

    

### [[2002.03328] Towards Out-of-Distribution Detection with Divergence Guarantee in Deep Generative Models](http://arxiv.org/abs/2002.03328)


  Recent research has revealed that deep generative models including flow-based
models and Variational autoencoders may assign higher likelihood to
out-of-distribution (OOD) data than in-distribution (ID) data. However, we
cannot sample out OOD data from the model. This counterintuitive phenomenon has
not been satisfactorily explained. In this paper, we prove theorems to
investigate the divergences in flow-based model and give two explanations to
the above phenomenon from divergence and geometric perspectives, respectively.
Based on our analysis, we propose two group anomaly detection methods.
Furthermore, we decompose the KL divergence and propose a point-wise anomaly
detection method. We have conducted extensive experiments on prevalent
benchmarks to evaluate our methods. For group anomaly detection (GAD), our
method can achieve near 100\% AUROC on all problems and has robustness against
data manipulations. On the contrary, the state-of-the-art (SOTA) GAD method
performs not better than random guessing for challenging problems and can be
attacked by data manipulation in almost all cases. For point-wise anomaly
detection (PAD), our method is comparable to the SOTA PAD method on one
category of problems and outperforms the baseline significantly on another
category of problems.

    

### [[2003.08414] Scattering GCN: Overcoming Oversmoothness in Graph Convolutional Networks](http://arxiv.org/abs/2003.08414)


  Graph convolutional networks (GCNs) have shown promising results in
processing graph data by extracting structure-aware features. This gave rise to
extensive work in geometric deep learning, focusing on designing network
architectures that ensure neuron activations conform to regularity patterns
within the input graph. However, in most cases the graph structure is only
accounted for by considering the similarity of activations between adjacent
nodes, which limits the capabilities of such methods to discriminate between
nodes in a graph. Here, we propose to augment conventional GCNs with geometric
scattering transforms and residual convolutions. The former enables band-pass
filtering of graph signals, thus alleviating the so-called oversmoothing often
encountered in GCNs, while the latter is introduced to clear the resulting
features of high-frequency noise. We establish the advantages of the presented
Scattering GCN with both theoretical results establishing the complementary
benefits of scattering and GCN features, as well as experimental results
showing the benefits of our method compared to leading graph neural networks
for semi-supervised node classification, including the recently proposed GAT
network that typically alleviates oversmoothing using graph attention
mechanisms.

    

### [[2003.13221] Planning as Inference in Epidemiological Models](http://arxiv.org/abs/2003.13221)


  In this work we demonstrate how to automate parts of the infectious
disease-control policy-making process via performing inference in existing
epidemiological models. The kind of inference tasks undertaken include
computing the posterior distribution over controllable, via direct
policy-making choices, simulation model parameters that give rise to acceptable
disease progression outcomes. Among other things, we illustrate the use of a
probabilistic programming language that automates inference in existing
simulators. Neither the full capabilities of this tool for automating inference
nor its utility for planning is widely disseminated at the current time. Timely
gains in understanding about how such simulation-based models and inference
automation tools applied in support of policymaking could lead to less
economically damaging policy prescriptions, particularly during the current
COVID-19 pandemic.

    

### [[2004.12088] SplitFed: When Federated Learning Meets Split Learning](http://arxiv.org/abs/2004.12088)


  Federated learning (FL) and split learning (SL) are two popular distributed
machine learning approaches. Both follow a model-to-data scenario; clients
train and test machine learning models without sharing raw data. SL provides
better model privacy than FL due to the machine learning model architecture
split between clients and the server. Moreover, the split model makes SL a
better option for resource-constrained environments. However, SL performs
slower than FL due to the relay-based training across multiple clients. In this
regard, this paper presents a novel approach, named splitfed learning (SFL),
that amalgamates the two approaches eliminating their inherent drawbacks, along
with a refined architectural configuration incorporating differential privacy
and PixelDP to enhance data privacy and model robustness. Our analysis and
empirical results demonstrate that (pure) SFL provides similar test accuracy
and communication efficiency as SL while significantly decreasing its
computation time per global epoch than in SL for multiple clients. Furthermore,
as in SL, its communication efficiency over FL improves with the number of
clients. Besides, the performance of SFL with privacy and robustness measures
is further evaluated under extended experimental settings.

    

### [[2005.00480] Regex Queries over Incomplete Knowledge Bases](http://arxiv.org/abs/2005.00480)


  We propose the novel task of answering regular expression queries (containing
disjunction ($\vee$) and Kleene plus ($+$) operators) over incomplete KBs. The
answer set of these queries potentially has a large number of entities, hence
previous works for single-hop queries in KBC that model a query as a point in
high-dimensional space are not as effective. In response, we develop RotatE-Box
-- a novel combination of RotatE and box embeddings. It can model more
relational inference patterns compared to existing embedding based models.
Furthermore, we define baseline approaches for embedding based KBC models to
handle regex operators. We demonstrate performance of RotatE-Box on two new
regex-query datasets introduced in this paper, including one where the queries
are harvested based on actual user query logs. We find that our final
RotatE-Box model significantly outperforms models based on just RotatE and just
box embeddings.

    

### [[2005.04288] Incremental Learning for End-to-End Automatic Speech Recognition](http://arxiv.org/abs/2005.04288)


  In this paper, we propose an incremental learning method for end-to-end
Automatic Speech Recognition (ASR) which enables an ASR system to perform well
on new tasks while maintaining the performance on its originally learned ones.
To mitigate catastrophic forgetting during incremental learning, we design a
novel explainability-based knowledge distillation for ASR models, which is
combined with a response-based knowledge distillation to maintain the original
model's predictions and the "reason" for the predictions. Our method works
without access to the training data of original tasks, which addresses the
cases where the previous data is no longer available or joint training is
costly. Results on a multi-stage sequential training task show that our method
outperforms existing ones in mitigating forgetting. Furthermore, in two
practical scenarios, compared to the target-reference joint training method,
the performance drop of our method is 0.02% Character Error Rate (CER), which
is 97% smaller than the drops of the baseline methods.

    

### [[2006.14028] Class-Similarity Based Label Smoothing for Confidence Calibration](http://arxiv.org/abs/2006.14028)


  Generating confidence calibrated outputs is of utmost importance for the
applications of deep neural networks in safety-critical decision-making
systems. The output of a neural network is a probability distribution where the
scores are estimated confidences of the input belonging to the corresponding
classes, and hence they represent a complete estimate of the output likelihood
relative to all classes. In this paper, we propose a novel form of label
smoothing to improve confidence calibration. Since different classes are of
different intrinsic similarities, more similar classes should result in closer
probability values in the final output. This motivates the development of a new
smooth label where the label values are based on similarities with the
reference class. We adopt different similarity measurements, including those
that capture feature-based similarities or semantic similarity. We demonstrate
through extensive experiments, on various datasets and network architectures,
that our approach consistently outperforms state-of-the-art calibration
techniques including uniform label smoothing.

    

### [[2007.11026] Spectral estimation from simulations via sketching](http://arxiv.org/abs/2007.11026)


  Sketching is a stochastic dimension reduction method that preserves geometric
structures of data and has applications in high-dimensional regression, low
rank approximation and graph sparsification. In this work, we show that
sketching can be used to compress simulation data and still accurately estimate
time autocorrelation and power spectral density. For a given compression ratio,
the accuracy is much higher than using previously known methods. In addition to
providing theoretical guarantees, we apply sketching to a molecular dynamics
simulation of methanol and find that the estimate of spectral density is 90%
accurate using only 10% of the data.

    

### [[2009.00647] Lifelong Graph Learning](http://arxiv.org/abs/2009.00647)


  Graph neural networks (GNNs) are powerful models for many graph-structured
tasks. Existing models often assume that a complete structure of a graph is
available during training, however, in practice, graph-structured data is
usually formed in a streaming fashion, so that learning a graph continuously is
often necessary. In this paper, we aim to bridge GNN to lifelong learning by
converting a graph problem to a regular learning problem, so that GNN is able
to inherit the lifelong learning techniques developed for convolutional neural
networks (CNNs). To this end, we propose a new graph topology based on feature
cross-correlation, called the feature graph. It takes features as new nodes and
turns nodes into independent graphs. This successfully converts the original
problem of node classification to graph classification, in which the increasing
nodes are turned into independent training samples. In the experiments, we
demonstrate the efficiency and effectiveness of feature graph networks (FGN) by
continuously learning a sequence of classical graph datasets. We also show that
FGN achieves superior performance in human action recognition with distributed
streaming signals for wearable devices.

    

### [[2010.05796] Pedestrian Trajectory Prediction with Convolutional Neural Networks](http://arxiv.org/abs/2010.05796)


  Predicting the future trajectories of pedestrians is a challenging problem
that has a range of application, from crowd surveillance to autonomous driving.
In literature, methods to approach pedestrian trajectory prediction have
evolved, transitioning from physics-based models to data-driven models based on
recurrent neural networks. In this work, we propose a new approach to
pedestrian trajectory prediction, with the introduction of a novel 2D
convolutional model. This new model outperforms recurrent models, and it
achieves state-of-the-art results on the ETH and TrajNet datasets. We also
present an effective system to represent pedestrian positions and powerful data
augmentation techniques, such as the addition of Gaussian noise and the use of
random rotations, which can be applied to any model. As an additional
exploratory analysis, we present experimental results on the inclusion of
occupancy methods to model social information, which empirically show that
these methods are ineffective in capturing social interaction.

    

### [[2012.15498] An Online Algorithm for Maximum-Likelihood Quantum State Tomography](http://arxiv.org/abs/2012.15498)


  We propose, to the best of our knowledge, the first online algorithm to
compute the maximum-likelihood estimate in quantum state tomography. Suppose
the quantum state to be estimated corresponds to a $D$-by-$D$ density matrix.
The per-iteration computational complexity of the algorithm is $O ( D ^ 3 )$,
independent of the data size. The expected optimization error of the algorithm
is $O(\sqrt{ ( 1 / T ) D \log D })$, where $T$ denotes the number of
iterations. The algorithm can be viewed as a quantum extension of Soft-Bayes, a
recent algorithm for online portfolio selection (Orseau et al. Soft-Bayes: Prod
for mixtures of experts with log-loss. Int. Conf. Algorithmic Learning Theory.
2017).

    

### [[2101.05796] DeFlow: Learning Complex Image Degradations from Unpaired Data with Conditional Flows](http://arxiv.org/abs/2101.05796)


  The difficulty of obtaining paired data remains a major bottleneck for
learning image restoration and enhancement models for real-world applications.
Current strategies aim to synthesize realistic training data by modeling noise
and degradations that appear in real-world settings. We propose DeFlow, a
method for learning stochastic image degradations from unpaired data. Our
approach is based on a novel unpaired learning formulation for conditional
normalizing flows. We model the degradation process in the latent space of a
shared flow encoder-decoder network. This allows us to learn the conditional
distribution of a noisy image given the clean input by solely minimizing the
negative log-likelihood of the marginal distributions. We validate our DeFlow
formulation on the task of joint image restoration and super-resolution. The
models trained with the synthetic data generated by DeFlow outperform previous
learnable approaches on three recent datasets. Code and trained models are
available at: this https URL


### [[2102.01659] Capacity and quantum geometry of parametrized quantum circuits](http://arxiv.org/abs/2102.01659)


  To harness the potential of noisy intermediate-scale quantum devices, it is
paramount to find the best type of circuits to run hybrid quantum-classical
algorithms. Key candidates are parametrized quantum circuits that can be
effectively implemented on current devices. Here, we evaluate the capacity and
trainability of these circuits using the geometric structure of the parameter
space via the effective quantum dimension, which reveals the expressive power
of circuits in general as well as of particular initialization strategies. We
assess the expressive power of various popular circuit types and find striking
differences depending on the type of entangling gates used. Particular circuits
are characterized by scaling laws in their expressiveness. We identify a
transition in the quantum geometry of the parameter space, which leads to a
decay of the quantum natural gradient for deep circuits. For shallow circuits,
the quantum natural gradient can be orders of magnitude larger in value
compared to the regular gradient; however, both of them can suffer from
vanishing gradients. By tuning a fixed set of circuit parameters to randomized
ones, we find a region where the circuit is expressive, but does not suffer
from barren plateaus, hinting at a good way to initialize circuits. We show an
algorithm that prunes redundant parameters of a circuit without affecting its
effective dimension. Our results enhance the understanding of parametrized
quantum circuits and can be immediately applied to improve variational quantum
algorithms.

    

### [[2102.03353] Cross-domain Activity Recognition via Substructural Optimal Transport](http://arxiv.org/abs/2102.03353)


  It is expensive and time-consuming to collect sufficient labeled data for
human activity recognition (HAR). Domain adaptation is a promising approach for
cross-domain activity recognition. Existing methods mainly focus on adapting
cross-domain representations via domain-level, class-level, or sample-level
distribution matching. However, they might fail to capture the fine-grained
locality information in activity data. The domain- and class-level matching are
too coarse that may result in under-adaptation, while sample-level matching may
be affected by the noise seriously and eventually cause over-adaptation. In
this paper, we propose substructure-level matching for domain adaptation (SSDA)
to better utilize the locality information of activity data for accurate and
efficient knowledge transfer. Based on SSDA, we propose an optimal
transport-based implementation, Substructural Optimal Transport (SOT), for
cross-domain HAR. We obtain the substructures of activities via clustering
methods and seeks the coupling of the weighted substructures between different
domains. We conduct comprehensive experiments on four public activity
recognition datasets (i.e. UCI-DSADS, UCI-HAR, USC-HAD, PAMAP2), which
demonstrates that SOT significantly outperforms other state-of-the-art methods
w.r.t classification accuracy (9%+ improvement). In addition, our mehtod is 5x
faster than traditional OT-based DA methods with the same hyper-parameters.

    

### [[2103.02554] Enabling Visual Action Planning for Object Manipulation through Latent Space Roadmap](http://arxiv.org/abs/2103.02554)


  We present a framework for visual action planning of complex manipulation
tasks with high-dimensional state spaces, focusing on manipulation of
deformable objects. We propose a Latent Space Roadmap (LSR) for task planning
which is a graph-based structure globally capturing the system dynamics in a
low-dimensional latent space. Our framework consists of three parts: (1) a
Mapping Module (MM) that maps observations given in the form of images into a
structured latent space extracting the respective states as well as generates
observations from the latent states, (2) the LSR which builds and connects
clusters containing similar states in order to find the latent plans between
start and goal states extracted by MM, and (3) the Action Proposal Module that
complements the latent plan found by the LSR with the corresponding actions. We
present a thorough investigation of our framework on simulated box stacking and
rope/box manipulation tasks, and a folding task executed on a real robot.

    

### [[2103.04689] Reverse Differentiation via Predictive Coding](http://arxiv.org/abs/2103.04689)


  Deep learning has redefined the field of artificial intelligence (AI) thanks
to the rise of artificial neural networks, which are architectures inspired by
their neurological counterpart in the brain. Through the years, this dualism
between AI and neuroscience has brought immense benefits to both fields,
allowing neural networks to be used in dozens of applications. These networks
use an efficient implementation of reverse differentiation, called
backpropagation (BP). This algorithm, however, is often criticized for its
biological implausibility (e.g., lack of local update rules for the
parameters). Therefore, biologically plausible learning methods that rely on
predictive coding (PC), a framework for describing information processing in
the brain, are increasingly studied. Recent works prove that these methods can
approximate BP up to a certain margin on multilayer perceptrons (MLPs), and
asymptotically on any other complex model, and that zero-divergence inference
learning (Z-IL), a variant of PC, is able to exactly implement BP on MLPs.
However, the recent literature shows also that there is no biologically
plausible method yet that can exactly replicate the weight update of BP on
complex models. To fill this gap, in this paper, we generalize (PC and) Z-IL by
directly defining them on computational graphs, and show that it can perform
exact reverse differentiation. What results is the first biologically plausible
algorithm that is equivalent to BP in the way of updating parameters on any
neural network, providing a bridge between the interdisciplinary research of
neuroscience and deep learning.

    

### [[2103.10547] Data driven algorithms for limited labeled data learning](http://arxiv.org/abs/2103.10547)


  We consider a novel data driven approach for designing learning algorithms
that can effectively learn with only a small number of labeled examples. This
is crucial for modern machine learning applications where labels are scarce or
expensive to obtain. We focus on graph-based techniques, where the unlabeled
examples are connected in a graph under the implicit assumption that similar
nodes likely have similar labels. Over the past decades, several elegant
graph-based semi-supervised and active learning algorithms for how to infer the
labels of the unlabeled examples given the graph and a few labeled examples
have been proposed. However, the problem of how to create the graph (which
impacts the practical usefulness of these methods significantly) has been
relegated to domain-specific art and heuristics and no general principles have
been proposed. In this work we present a novel data driven approach for
learning the graph and provide strong formal guarantees in both the
distributional and online learning formalizations.
We show how to leverage problem instances coming from an underlying problem
domain to learn the graph hyperparameters from commonly used parametric
families of graphs that perform well on new instances coming from the same
domain. We obtain low regret and efficient algorithms in the online setting,
and generalization guarantees in the distributional setting. We also show how
to combine several very different similarity metrics and learn multiple
hyperparameters, providing general techniques to apply to large classes of
problems. We expect some of the tools and techniques we develop along the way
to be of interest beyond semi-supervised and active learning, for data driven
algorithms for combinatorial problems more generally.

    

### [[2104.03916] Field Convolutions for Surface CNNs](http://arxiv.org/abs/2104.03916)


  We present a novel surface convolution operator acting on vector fields that
is based on a simple observation: instead of combining neighboring features
with respect to a single coordinate parameterization defined at a given point,
we have every neighbor describe the position of the point within its own
coordinate frame. This formulation combines intrinsic spatial convolution with
parallel transport in a scattering operation while placing no constraints on
the filters themselves, providing a definition of convolution that commutes
with the action of isometries, has increased descriptive potential, and is
robust to noise and other nuisance factors. The result is a rich notion of
convolution which we call field convolution, well-suited for CNNs on surfaces.
Field convolutions are flexible, straight-forward to incorporate into surface
learning frameworks, and their highly discriminating nature has cascading
effects throughout the learning pipeline. Using simple networks constructed
from residual field convolution blocks, we achieve state-of-the-art results on
standard benchmarks in fundamental geometry processing tasks, such as shape
classification, segmentation, correspondence, and sparse matching.

    

### [[2104.04704] DuRIN: A Deep-unfolded Sparse Seismic Reflectivity Inversion Network](http://arxiv.org/abs/2104.04704)


  We consider the reflection seismology problem of recovering the locations of
interfaces and the amplitudes of reflection coefficients from seismic data,
which are vital for estimating the subsurface structure. The reflectivity
inversion problem is typically solved using greedy algorithms and iterative
techniques. Sparse Bayesian learning framework, and more recently, deep
learning techniques have shown the potential of data-driven approaches to solve
the problem. In this paper, we propose a weighted minimax-concave
penalty-regularized reflectivity inversion formulation and solve it through a
model-based neural network. The network is referred to as deep-unfolded
reflectivity inversion network (DuRIN). We demonstrate the efficacy of the
proposed approach over the benchmark techniques by testing on synthetic 1-D
seismic traces and 2-D wedge models and validation with the simulated 2-D
Marmousi2 model and real data from the Penobscot 3D survey off the coast of
Nova Scotia, Canada.

    

### [[2104.06893] I Wish I Would Have Loved This One, But I Didn't -- A Multilingual Dataset for Counterfactual Detection in Product Reviews](http://arxiv.org/abs/2104.06893)


  Counterfactual statements describe events that did not or cannot take place.
We consider the problem of counterfactual detection (CFD) in product reviews.
For this purpose, we annotate a multilingual CFD dataset from Amazon product
reviews covering counterfactual statements written in English, German, and
Japanese languages. The dataset is unique as it contains counterfactuals in
multiple languages, covers a new application area of e-commerce reviews, and
provides high quality professional annotations. We train CFD models using
different text representation methods and classifiers. We find that these
models are robust against the selectional biases introduced due to cue
phrase-based sentence selection. Moreover, our CFD dataset is compatible with
prior datasets and can be merged to learn accurate CFD models. Applying machine
translation on English counterfactual examples to create multilingual data
performs poorly, demonstrating the language-specificity of this problem, which
has been ignored so far.

    

### [[2104.08438] Bayesian graph convolutional neural networks via tempered MCMC](http://arxiv.org/abs/2104.08438)


  Deep learning models, such as convolutional neural networks, have long been
applied to image and multi-media tasks, particularly those with structured
data. More recently, there has been more attention to unstructured data that
can be represented via graphs. These types of data are often found in health
and medicine, social networks, and research data repositories. Graph
convolutional neural networks have recently gained attention in the field of
deep learning that takes advantage of graph-based data representation with
automatic feature extraction via convolutions. Given the popularity of these
methods in a wide range of applications, robust uncertainty quantification is
vital. This remains a challenge for large models and unstructured datasets.
Bayesian inference provides a principled approach to uncertainty quantification
of model parameters for deep learning models. Although Bayesian inference has
been used extensively elsewhere, its application to deep learning remains
limited due to the computational requirements of the Markov Chain Monte Carlo
(MCMC) methods. Recent advances in parallel computing and advanced proposal
schemes in MCMC sampling methods has opened the path for Bayesian deep
learning. In this paper, we present Bayesian graph convolutional neural
networks that employ tempered MCMC sampling with Langevin-gradient proposal
distribution implemented via parallel computing. Our results show that the
proposed method can provide accuracy similar to advanced optimisers while
providing uncertainty quantification for key benchmark problems.

    

### [[2104.08678] Improving Question Answering Model Robustness with Synthetic Adversarial Data Generation](http://arxiv.org/abs/2104.08678)


  Despite recent progress, state-of-the-art question answering models remain
vulnerable to a variety of adversarial attacks. While dynamic adversarial data
collection, in which a human annotator tries to write examples that fool a
model-in-the-loop, can improve model robustness, this process is expensive
which limits the scale of the collected data. In this work, we are the first to
use synthetic adversarial data generation to make question answering models
more robust to human adversaries. We develop a data generation pipeline that
selects source passages, identifies candidate answers, generates questions,
then finally filters or re-labels them to improve quality. Using this approach,
we amplify a smaller human-written adversarial dataset to a much larger set of
synthetic question-answer pairs. By incorporating our synthetic data, we
improve the state-of-the-art on the AdversarialQA dataset by 3.7F1 and improve
model generalisation on nine of the twelve MRQA datasets. We further conduct a
novel human-in-the-loop evaluation to show that our models are considerably
more robust to new human-written adversarial examples: crowdworkers can fool
our model only 8.8% of the time on average, compared to 17.6% for a model
trained without synthetic data.

    

### [[2105.00003] NuSPAN: A Proximal Average Network for Nonuniform Sparse Model -- Application to Seismic Reflectivity Inversion](http://arxiv.org/abs/2105.00003)


  We solve the problem of sparse signal deconvolution in the context of seismic
reflectivity inversion, which pertains to high-resolution recovery of the
subsurface reflection coefficients. Our formulation employs a nonuniform,
non-convex synthesis sparse model comprising a combination of convex and
non-convex regularizers, which results in accurate approximations of the l0
pseudo-norm. The resulting iterative algorithm requires the proximal average
strategy. When unfolded, the iterations give rise to a learnable proximal
average network architecture that can be optimized in a data-driven fashion. We
demonstrate the efficacy of the proposed approach through numerical experiments
on synthetic 1-D seismic traces and 2-D wedge models in comparison with the
benchmark techniques. We also present validations considering the simulated
Marmousi2 model as well as real 3-D seismic volume data acquired from the
Penobscot 3D survey off the coast of Nova Scotia, Canada.

    

### [[2105.05582] Discrete representations in neural models of spoken language](http://arxiv.org/abs/2105.05582)


  The distributed and continuous representations used by neural networks are at
odds with representations employed in linguistics, which are typically
symbolic. Vector quantization has been proposed as a way to induce discrete
neural representations that are closer in nature to their linguistic
counterparts. However, it is not clear which metrics are the best-suited to
analyze such discrete representations. We compare the merits of four commonly
used metrics in the context of weakly supervised models of spoken language. We
compare the results they show when applied to two different models, while
systematically studying the effect of the placement and size of the
discretization layer. We find that different evaluation regimes can give
inconsistent results. While we can attribute them to the properties of the
different metrics in most cases, one point of concern remains: the use of
minimal pairs of phoneme triples as stimuli disadvantages larger discrete unit
inventories, unlike metrics applied to complete utterances. Furthermore, while
in general vector quantization induces representations that correlate with
units posited in linguistics, the strength of this correlation is only
moderate.

    

### [[2105.11363] Learning Security Classifiers with Verified Global Robustness Properties](http://arxiv.org/abs/2105.11363)


  Many recent works have proposed methods to train classifiers with local
robustness properties, which can provably eliminate classes of evasion attacks
for most inputs, but not all inputs. Since data distribution shift is very
common in security applications, e.g., often observed for malware detection,
local robustness cannot guarantee that the property holds for unseen inputs at
the time of deploying the classifier. Therefore, it is more desirable to
enforce global robustness properties that hold for all inputs, which is
strictly stronger than local robustness.
In this paper, we present a framework and tools for training classifiers that
satisfy global robustness properties. We define new notions of global
robustness that are more suitable for security classifiers. We design a novel
booster-fixer training framework to enforce global robustness properties. We
structure our classifier as an ensemble of logic rules and design a new
verifier to verify the properties. In our training algorithm, the booster
increases the classifier's capacity, and the fixer enforces verified global
robustness properties following counterexample guided inductive synthesis.
We show that we can train classifiers to satisfy different global robustness
properties for three security datasets, and even multiple properties at the
same time, with modest impact on the classifier's performance. For example, we
train a Twitter spam account classifier to satisfy five global robustness
properties, with 5.4% decrease in true positive rate, and 0.1% increase in
false positive rate, compared to a baseline XGBoost model that doesn't satisfy
any property.

    

### [[2106.05997] Verifying Quantized Neural Networks using SMT-Based Model Checking](http://arxiv.org/abs/2106.05997)


  Artificial Neural Networks (ANNs) are being deployed for an increasing number
of safety-critical applications, including autonomous cars and medical
diagnosis. However, concerns about their reliability have been raised due to
their black-box nature and apparent fragility to adversarial attacks. These
concerns are amplified when ANNs are deployed on restricted system, which limit
the precision of mathematical operations and thus introduce additional
quantization errors. Here, we develop and evaluate a novel symbolic
verification framework using software model checking (SMC) and satisfiability
modulo theories (SMT) to check for vulnerabilities in ANNs. More specifically,
we propose several ANN-related optimizations for SMC, including invariant
inference via interval analysis, slicing, expression simplifications, and
discretization of non-linear activation functions. With this verification
framework, we can provide formal guarantees on the safe behavior of ANNs
implemented both in floating- and fixed-point arithmetic. In this regard, our
verification approach was able to verify and produce adversarial examples for
$52$ test cases spanning image classification and general machine learning
applications. Furthermore, for small- to medium-sized ANN, our approach
completes most of its verification runs in minutes. Moreover, in contrast to
most state-of-the-art methods, our approach is not restricted to specific
choices regarding activation functions and non-quantized representations. Our
experiments show that our approach can analyze larger ANN implementations and
substantially reduce the verification time compared to state-of-the-art
techniques that use SMT solving.

    

### [[2106.10744] On the Cryptographic Hardness of Learning Single Periodic Neurons](http://arxiv.org/abs/2106.10744)


  We show a simple reduction which demonstrates the cryptographic hardness of
learning a single periodic neuron over isotropic Gaussian distributions in the
presence of noise. More precisely, our reduction shows that any polynomial-time
algorithm (not necessarily gradient-based) for learning such functions under
small noise implies a polynomial-time quantum algorithm for solving worst-case
lattice problems, whose hardness form the foundation of lattice-based
cryptography. Our core hard family of functions, which are well-approximated by
one-layer neural networks, take the general form of a univariate periodic
function applied to an affine projection of the data. These functions have
appeared in previous seminal works which demonstrate their hardness against
gradient-based (Shamir'18), and Statistical Query (SQ) algorithms (Song et
al.'17). We show that if (polynomially) small noise is added to the labels, the
intractability of learning these functions applies to all polynomial-time
algorithms, beyond gradient-based and SQ algorithms, under the aforementioned
cryptographic assumptions. Moreover, we demonstrate the necessity of noise in
the hardness result by designing a polynomial-time algorithm for learning
certain families of such functions under exponentially small adversarial noise.
Our proposed algorithm is not a gradient-based or an SQ algorithm, but is
rather based on the celebrated Lenstra-Lenstra-Lovász (LLL) lattice basis
reduction algorithm. Furthermore, in the absence of noise, this algorithm can
be directly applied to solve CLWE detection (Bruna et al.'21) and phase
retrieval with an optimal sample complexity of $d+1$ samples. In the former
case, this improves upon the quadratic-in-$d$ sample complexity required in
(Bruna et al.'21).

    

### [[2106.16194] DNN-Based Decentralized Hybrid Beamforming for Cell-Free Massive MIMO](http://arxiv.org/abs/2106.16194)


  Cell-free massive MIMO (CF-mMIMO) systems represent a promising approach to
increase the spectral efficiency of wireless communication systems. However,
near-optimal hybrid beamforming solutions require a large amount of signaling
exchange between access points (APs) and the network controller (NC). In this
letter, we propose two unsupervised deep neural networks (DNN) architectures,
fully and partially distributed, that can perform decentralized coordinated
hybrid beamforming with zero or limited communication overhead between APs and
NC, while achieving near-optimal sum-rate with a reduced computational
complexity compared to conventional near-optimal solutions.

    

### [[2107.00593] Disaggregated Interventions to Reduce Inequality](http://arxiv.org/abs/2107.00593)


  A significant body of research in the data sciences considers unfair
discrimination against social categories such as race or gender that could
occur or be amplified as a result of algorithmic decisions. Simultaneously,
real-world disparities continue to exist, even before algorithmic decisions are
made. In this work, we draw on insights from the social sciences brought into
the realm of causal modeling and constrained optimization, and develop a novel
algorithmic framework for tackling pre-existing real-world disparities. The
purpose of our framework, which we call the "impact remediation framework," is
to measure real-world disparities and discover the optimal intervention
policies that could help improve equity or access to opportunity for those who
are underserved with respect to an outcome of interest. We develop a
disaggregated approach to tackling pre-existing disparities that relaxes the
typical set of assumptions required for the use of social categories in
structural causal models. Our approach flexibly incorporates counterfactuals
and is compatible with various ontological assumptions about the nature of
social categories. We demonstrate impact remediation with a hypothetical case
study and compare our disaggregated approach to an existing state-of-the-art
approach, comparing its structure and resulting policy recommendations. In
contrast to most work on optimal policy learning, we explore disparity
reduction itself as an objective, explicitly focusing the power of algorithms
on reducing inequality.

    

### [[2108.00833] Adversarial Attacks Against Deep Reinforcement Learning Framework in Internet of Vehicles](http://arxiv.org/abs/2108.00833)


  Machine learning (ML) has made incredible impacts and transformations in a
wide range of vehicular applications. As the use of ML in Internet of Vehicles
(IoV) continues to advance, adversarial threats and their impact have become an
important subject of research worth exploring. In this paper, we focus on
Sybil-based adversarial threats against a deep reinforcement learning
(DRL)-assisted IoV framework and more specifically, DRL-based dynamic service
placement in IoV. We carry out an experimental study with real vehicle
trajectories to analyze the impact on service delay and resource congestion
under different attack scenarios for the DRL-based dynamic service placement
application. We further investigate the impact of the proportion of
Sybil-attacked vehicles in the network. The results demonstrate that the
performance is significantly affected by Sybil-based data poisoning attacks
when compared to adversary-free healthy network scenario.

    

### [[1907.11184] HEIDL: Learning Linguistic Expressions with Deep Learning and Human-in-the-Loop](http://arxiv.org/abs/1907.11184)


  While the role of humans is increasingly recognized in machine learning
community, representation of and interaction with models in current
human-in-the-loop machine learning (HITL-ML) approaches are too low-level and
far-removed from human's conceptual models. We demonstrate HEIDL, a prototype
HITL-ML system that exposes the machine-learned model through high-level,
explainable linguistic expressions formed of predicates representing semantic
structure of text. In HEIDL, human's role is elevated from simply evaluating
model predictions to interpreting and even updating the model logic directly by
enabling interaction with rule predicates themselves. Raising the currency of
interaction to such semantic levels calls for new interaction paradigms between
humans and machines that result in improved productivity for text analytics
model development process. Moreover, by involving humans in the process, the
human-machine co-created models generalize better to unseen data as domain
experts are able to instill their expertise by extrapolating from what has been
learned by automated algorithms from few labelled data.

    

### [[2003.07678] An Overview and Case Study of the Clinical AI Model Development Life Cycle for Healthcare Systems](http://arxiv.org/abs/2003.07678)


  Healthcare is one of the most promising areas for machine learning models to
make a positive impact. However, successful adoption of AI-based systems in
healthcare depends on engaging and educating stakeholders from diverse
backgrounds about the development process of AI models. We present a broadly
accessible overview of the development life cycle of clinical AI models that is
general enough to be adapted to most machine learning projects, and then give
an in-depth case study of the development process of a deep learning based
system to detect aortic aneurysms in Computed Tomography (CT) exams. We hope
other healthcare institutions and clinical practitioners find the insights we
share about the development process useful in informing their own model
development efforts and to increase the likelihood of successful deployment and
integration of AI in healthcare.

    

### [[2109.05666] AMI-FML: A Privacy-Preserving Federated Machine Learning Framework for AMI](http://arxiv.org/abs/2109.05666)


  Machine learning (ML) based smart meter data analytics is very promising for
energy management and demand-response applications in the advanced metering
infrastructure(AMI). A key challenge in developing distributed ML applications
for AMI is to preserve user privacy while allowing active end-users
participation. This paper addresses this challenge and proposes a
privacy-preserving federated learning framework for ML applications in the AMI.
We consider each smart meter as a federated edge device hosting an ML
application that exchanges information with a central aggregator or a data
concentrator, periodically. Instead of transferring the raw data sensed by the
smart meters, the ML model weights are transferred to the aggregator to
preserve privacy. The aggregator processes these parameters to devise a robust
ML model that can be substituted at each edge device. We also discuss
strategies to enhance privacy and improve communication efficiency while
sharing the ML model parameters, suited for relatively slow network connections
in the AMI. We demonstrate the proposed framework on a use case federated ML
(FML) application that improves short-term load forecasting (STLF). We use a
long short-term memory(LSTM) recurrent neural network (RNN) model for STLF. In
our architecture, we assume that there is an aggregator connected to a group of
smart meters. The aggregator uses the learned model gradients received from the
federated smart meters to generate an aggregate, robust RNN model which
improves the forecasting accuracy for individual and aggregated STLF. Our
results indicate that with FML, forecasting accuracy is increased while
preserving the data privacy of the end-users.

    

### [[2109.07358] Fermion Sampling Made More Efficient](http://arxiv.org/abs/2109.07358)


  Fermion sampling is to generate probability distribution of a many-body
Slater-determinant wavefunction, which is termed "determinantal point process"
in statistical analysis. For its inherently-embedded Pauli exclusion principle,
its application reaches beyond simulating fermionic quantum many-body physics
to constructing machine learning models for diversified datasets. Here we
propose a fermion sampling algorithm, which has a polynomial time-complexity --
quadratic in the fermion number and linear in the system size. This algorithm
is about 100% more efficient in computation time than the best known
algorithms. In sampling the corresponding marginal distribution, our algorithm
has a more drastic improvement, achieving a scaling advantage. We demonstrate
its power on several test applications, including sampling fermions in a
many-body system and a machine learning task of text summarization, and confirm
its improved computation efficiency over other methods by counting
floating-point operations.

    

### [[2109.07929] A Fast Method for Steady-State Memristor Crossbar Array Circuit Simulation](http://arxiv.org/abs/2109.07929)


  In this work we propose an effective preconditioning technique to accelerate
the steady-state simulation of large-scale memristor crossbar arrays (MCAs). We
exploit the structural regularity of MCAs to develop a specially-crafted
preconditioner that can be efficiently evaluated utilizing tensor products and
block matrix inversion. Numerical experiments demonstrate the efficacy of the
proposed technique compared to mainstream preconditioners.

    

### [[2109.07541] Dala: A Simple Capability-Based Dynamic Language Design For Data Race-Freedom](http://arxiv.org/abs/2109.07541)


  Dynamic languages like Erlang, Clojure, JavaScript, and E adopted data-race
freedom by design. To enforce data-race freedom, these languages either deep
copy objects during actor (thread) communication or proxy back to their owning
thread. We present Dala, a simple programming model that ensures data-race
freedom while supporting efficient inter-thread communication. Dala is a
dynamic, concurrent, capability-based language that relies on three core
capabilities: immutable values can be shared freely; isolated mutable objects
can be transferred between threads but not aliased; local objects can be
aliased within their owning thread but not dereferenced by other threads.
Objects with capabilities can co-exist with unsafe objects, that are unchecked
and may suffer data races, without compromising the safety of safe objects. We
present a formal model of Dala, prove data race-freedom and state and prove a
dynamic gradual guarantee. These theorems guarantee data race-freedom when
using safe capabilities and show that the addition of capabilities is semantics
preserving modulo permission and cast errors.

    

### [[2109.07670] Evaluating OptChain with Bitcoin Transactions](http://arxiv.org/abs/2109.07670)


  While many researchers adopt a sharding approach to design scaling
blockchains, few works have studied the transaction placement problem incurred
by sharding protocols. The widely-used hashing placement algorithm renders an
overwhelming portion of transactions as cross-shard. In this paper, we analyze
the high cost of cross-shard transactions and reveal that most Bitcoin
transactions have simple dependencies and can become single-shard under a
placement algorithm taking transaction dependencies into account. In addition,
we perform a case study of OptChain, which is the state-of-the-art transaction
placement algorithm for sharded blockchains, and find a defect of it. A fix is
proposed, and our evaluation results demonstrate that the fix helps OptChain
improve the system throughput by 4x.

    

### [[2109.07721] Blockchain for Trust and Reputation Management in Cyber-physical Systems](http://arxiv.org/abs/2109.07721)


  The salient features of blockchain, such as decentralisation and
transparency, have allowed the development of Decentralised Trust and
Reputation Management Systems (DTRMS), which mainly aim to quantitatively
assess the trustworthiness of the network participants and help to protect the
network from adversaries. In the literature, proposals of DTRMS have been
applied to various Cyber-physical Systems (CPS) applications, including supply
chains, smart cities and distributed energy trading. In this chapter, we
outline the building blocks of a generic DTRMS and discuss how it can benefit
from blockchain. To highlight the significance of DTRMS, we present the
state-of-the-art of DTRMS in various field of CPS applications. In addition, we
also outline challenges and future directions in developing DTRMS for CPS.

    

### [[2109.07744] Disaggregating and Consolidating Network Functionalities with SuperNIC](http://arxiv.org/abs/2109.07744)


  Resource disaggregation has gained huge popularity in recent years. Existing
works demonstrate how to disaggregate compute, memory, and storage resources.
We, for the first time, demonstrate how to disaggregate network resources by
proposing a new distributed hardware framework called SuperNIC. Each SuperNIC
connects a small set of endpoints and consolidates network functionalities for
these endpoints. We prototyped SuperNIC with FPGA and demonstrate its
performance and cost benefits with real network functions and customized
disaggregated applications.

    

### [[2109.07771] Quantifying and Generalizing the CAP Theorem](http://arxiv.org/abs/2109.07771)


  In distributed applications, Brewer's CAP theorem tells us that when networks
become partitioned, there is a tradeoff between consistency and availability.
Consistency is agreement on the values of shared variables across a system, and
availability is the ability to respond to reads and writes accessing those
shared variables. We quantify these concepts, giving numerical values to
inconsistency and unavailability. Recognizing that network partitioning is not
an all-or-nothing proposition, we replace the P in CAP with L, a numerical
measure of apparent latency, and derive the CAL theorem, an algebraic relation
between inconsistency, unavailability, and apparent latency. This relation
shows that if latency becomes unbounded (e.g., the network becomes
partitioned), then one of inconsistency and unavailability must also become
unbounded, and hence the CAP theorem is a special case of the CAL theorem. We
describe two distributed coordination mechanisms, which we have implemented as
an extension of the Lingua Franca coordination language, that support arbitrary
tradeoffs between consistency and availability as apparent latency varies. With
centralized coordination, inconsistency remains bounded by a chosen numerical
value at the cost that unavailability becomes unbounded under network
partitioning. With decentralized coordination, unavailability remains bounded
by a chosen numerical quantity at the cost that inconsistency becomes unbounded
under network partitioning. Our centralized coordination mechanism is an
extension of techniques that have historically been used for distributed
simulation, an application where consistency is paramount. Our decentralized
coordination mechanism is an extension of techniques that have been used in
distributed databases when availability is paramount.

    

### [[2109.08053] Northlight: Declarative and Optimized Analysis of Atmospheric Datasets in SparkSQL](http://arxiv.org/abs/2109.08053)


  Performing data-intensive analytics is an essential part of modern Earth
science. As such, research in atmospheric physics and meteorology frequently
requires the processing of very large observational and/or modeled datasets.
Typically, these datasets (a) have high dimensionality, i.e. contain various
measurements per spatiotemporal point, (b) are extremely large, containing
observations over a long time period. Additionally, (c) the analytical tasks
being performed on these datasets are structurally complex. Over the years, the
binary format NetCDF has been established as a de-facto standard in
distributing and exchanging such multi-dimensional datasets in the Earth
science community -- along with tools and APIs to visualize, process, and
generate them. Unfortunately, these access methods typically lack either (1) an
easy-to-use but rich query interface or (2) an automatic optimization pipeline
tailored towards the specialities of these datasets. As such, researchers from
the field of Earth sciences (which are typically not computer scientists)
unnecessarily struggle in efficiently working with these datasets on a daily
basis. Consequently, in this work, we aim at resolving the aforementioned
issues. Instead of proposing yet another specialized tool and interface to work
with atmospheric datasets, we integrate sophisticated NetCDF processing
capabilities into the established SparkSQL dataflow engine -- resulting in our
system Northlight. In contrast to comparable systems, Northlight introduces a
set of fully automatic optimizations specifically tailored towards NetCDF
processing. We experimentally show that Northlight scales gracefully with the
selectivity of the analysis tasks and outperforms the comparable
state-of-the-art pipeline by up to a factor of 6x.

    

### [[1905.08563] Optimal Space Lower Bound for Deterministic Self-Stabilizing Leader Election Algorithms](http://arxiv.org/abs/1905.08563)


  Given a boolean predicate $\Pi$ on labeled networks (e.g., proper coloring,
leader election, etc.), a self-stabilizing algorithm for $\Pi$ is a distributed
algorithm that can start from any initial configuration of the network (i.e.,
every node has an arbitrary value assigned to each of its variables), and
eventually converge to a configuration satisfying $\Pi$. It is known that
leader election does not have a deterministic self-stabilizing algorithm using
a constant-size register at each node, i.e., for some networks, some of their
nodes must have registers whose sizes grow with the size $n$ of the networks.
On the other hand, it is also known that leader election can be solved by a
deterministic self-stabilizing algorithm using registers of $O(\log \log n)$
bits per node in any $n$-node bounded-degree network. We show that this latter
space complexity is optimal. Specifically, we prove that every deterministic
self-stabilizing algorithm solving leader election must use $\Omega(\log \log
n)$-bit per node registers in some $n$-node networks. In addition, we show that
our lower bounds go beyond leader election, and apply to all problems that
cannot be solved by anonymous algorithms.

    

### [[1910.09017] Demystifying Graph Databases: Analysis and Taxonomy of Data Organization, System Designs, and Graph Queries](http://arxiv.org/abs/1910.09017)


  Graph processing has become an important part of multiple areas of computer
science, such as machine learning, computational sciences, medical
applications, social network analysis, and many others. Numerous graphs such as
web or social networks may contain up to trillions of edges. Often, these
graphs are also dynamic (their structure changes over time) and have
domain-specific rich data associated with vertices and edges. Graph database
systems such as Neo4j enable storing, processing, and analyzing such large,
evolving, and rich datasets. Due to the sheer size of such datasets, combined
with the irregular nature of graph processing, these systems face unique design
challenges. To facilitate the understanding of this emerging domain, we present
the first survey and taxonomy of graph database systems. We focus on
identifying and analyzing fundamental categories of these systems (e.g., triple
stores, tuple stores, native graph database systems, or object-oriented
systems), the associated graph models (e.g., RDF or Labeled Property Graph),
data organization techniques (e.g., storing graph data in indexing structures
or dividing data into records), and different aspects of data distribution and
query execution (e.g., support for sharding and ACID). 45 graph database
systems are presented and compared, including Neo4j, OrientDB, or Virtuoso. We
outline graph database queries and relationships with associated domains (NoSQL
stores, graph streaming, and dynamic graph algorithms). Finally, we describe
research and engineering challenges to outline the future of graph databases.

    

### [[2109.07483] Cross-Register Projection for Headline Part of Speech Tagging](http://arxiv.org/abs/2109.07483)


  Part of speech (POS) tagging is a familiar NLP task. State of the art taggers
routinely achieve token-level accuracies of over 97% on news body text,
evidence that the problem is well understood. However, the register of English
news headlines, "headlinese", is very different from the register of long-form
text, causing POS tagging models to underperform on headlines. In this work, we
automatically annotate news headlines with POS tags by projecting predicted
tags from corresponding sentences in news bodies. We train a multi-domain POS
tagger on both long-form and headline text and show that joint training on both
registers improves over training on just one or naively concatenating training
sets. We evaluate on a newly-annotated corpus of over 5,248 English news
headlines from the Google sentence compression corpus, and show that our model
yields a 23% relative error reduction per token and 19% per headline. In
addition, we demonstrate that better headline POS tags can improve the
performance of a syntax-based open information extraction system. We make POSH,
the POS-tagged Headline corpus, available to encourage research in improved NLP
models for news headlines.

    

### [[2109.07514] DeepMetis: Augmenting a Deep Learning Test Set to Increase its Mutation Score](http://arxiv.org/abs/2109.07514)


  Deep Learning (DL) components are routinely integrated into software systems
that need to perform complex tasks such as image or natural language
processing. The adequacy of the test data used to test such systems can be
assessed by their ability to expose artificially injected faults (mutations)
that simulate real DL faults. In this paper, we describe an approach to
automatically generate new test inputs that can be used to augment the existing
test set so that its capability to detect DL mutations increases. Our tool
DeepMetis implements a search based input generation strategy. To account for
the non-determinism of the training and the mutation processes, our fitness
function involves multiple instances of the DL model under test. Experimental
results show that \tool is effective at augmenting the given test set,
increasing its capability to detect mutants by 63% on average. A leave-one-out
experiment shows that the augmented test set is capable of exposing unseen
mutants, which simulate the occurrence of yet undetected faults.

    

### [[2109.07556] Unit Selection with Causal Diagram](http://arxiv.org/abs/2109.07556)


  The unit selection problem aims to identify a set of individuals who are most
likely to exhibit a desired mode of behavior, for example, selecting
individuals who would respond one way if encouraged and a different way if not
encouraged. Using a combination of experimental and observational data, Li and
Pearl derived tight bounds on the "benefit function" - the payoff/cost
associated with selecting an individual with given characteristics. This paper
shows that these bounds can be narrowed significantly (enough to change
decisions) when structural information is available in the form of a causal
model. We address the problem of estimating the benefit function using
observational and experimental data when specific graphical criteria are
assumed to hold.

    

### [[2109.07576] "It doesn't look good for a date": Transforming Critiques into Preferences for Conversational Recommendation Systems](http://arxiv.org/abs/2109.07576)


  Conversations aimed at determining good recommendations are iterative in
nature. People often express their preferences in terms of a critique of the
current recommendation (e.g., "It doesn't look good for a date"), requiring
some degree of common sense for a preference to be inferred. In this work, we
present a method for transforming a user critique into a positive preference
(e.g., "I prefer more romantic") in order to retrieve reviews pertaining to
potentially better recommendations (e.g., "Perfect for a romantic dinner"). We
leverage a large neural language model (LM) in a few-shot setting to perform
critique-to-preference transformation, and we test two methods for retrieving
recommendations: one that matches embeddings, and another that fine-tunes an LM
for the task. We instantiate this approach in the restaurant domain and
evaluate it using a new dataset of restaurant critiques. In an ablation study,
we show that utilizing critique-to-preference transformation improves
recommendations, and that there are at least three general cases that explain
this improved performance.

    

### [[2109.07648] METEOR: A Massive Dense & Heterogeneous Behavior Dataset for Autonomous Driving](http://arxiv.org/abs/2109.07648)


  We present a new and complex traffic dataset, METEOR, which captures traffic
patterns in unstructured scenarios in India. METEOR consists of more than 1000
one-minute video clips, over 2 million annotated frames with ego-vehicle
trajectories, and more than 13 million bounding boxes for surrounding vehicles
or traffic agents. METEOR is a unique dataset in terms of capturing the
heterogeneity of microscopic and macroscopic traffic characteristics.
Furthermore, we provide annotations for rare and interesting driving behaviors
such as cut-ins, yielding, overtaking, overspeeding, zigzagging, sudden lane
changing, running traffic signals, driving in the wrong lanes, taking wrong
turns, lack of right-of-way rules at intersections, etc. We also present
diverse traffic scenarios corresponding to rainy weather, nighttime driving,
driving in rural areas with unmarked roads, and high-density traffic scenarios.
We use our novel dataset to evaluate the performance of object detection and
behavior prediction algorithms. We show that state-of-the-art object detectors
fail in these challenging conditions and also propose a new benchmark test:
action-behavior prediction with a baseline mAP score of 70.74.

    

### [[2109.07672] An Ontology-Based Information Extraction System for Residential Land Use Suitability Analysis](http://arxiv.org/abs/2109.07672)


  We propose an Ontology-Based Information Extraction (OBIE) system to automate
the extraction of the criteria and values applied in Land Use Suitability
Analysis (LUSA) from bylaw and regulation documents related to the geographic
area of interest. The results obtained by our proposed LUSA OBIE system (land
use suitability criteria and their values) are presented as an ontology
populated with instances of the extracted criteria and property values. This
latter output ontology is incorporated into a Multi-Criteria Decision Making
(MCDM) model applied for constructing suitability maps for different kinds of
land uses. The resulting maps may be the final desired product or can be
incorporated into the cellular automata urban modeling and simulation for
predicting future urban growth. A case study has been conducted where the
output from LUSA OBIE is applied to help produce a suitability map for the City
of Regina, Saskatchewan, to assist in the identification of suitable areas for
residential development. A set of Saskatchewan bylaw and regulation documents
were downloaded and input to the LUSA OBIE system. We accessed the extracted
information using both the populated LUSA ontology and the set of annotated
documents. In this regard, the LUSA OBIE system was effective in producing a
final suitability map.

    

### [[2109.07679] Benchmarking Commonsense Knowledge Base Population with an Effective Evaluation Dataset](http://arxiv.org/abs/2109.07679)


  Reasoning over commonsense knowledge bases (CSKB) whose elements are in the
form of free-text is an important yet hard task in NLP. While CSKB completion
only fills the missing links within the domain of the CSKB, CSKB population is
alternatively proposed with the goal of reasoning unseen assertions from
external resources. In this task, CSKBs are grounded to a large-scale
eventuality (activity, state, and event) graph to discriminate whether novel
triples from the eventuality graph are plausible or not. However, existing
evaluations on the population task are either not accurate (automatic
evaluation with randomly sampled negative examples) or of small scale (human
annotation). In this paper, we benchmark the CSKB population task with a new
large-scale dataset by first aligning four popular CSKBs, and then presenting a
high-quality human-annotated evaluation set to probe neural models' commonsense
reasoning ability. We also propose a novel inductive commonsense reasoning
model that reasons over graphs. Experimental results show that generalizing
commonsense reasoning on unseen assertions is inherently a hard task. Models
achieving high accuracy during training perform poorly on the evaluation set,
with a large gap between human performance. We will make the data publicly
available for future contributions. Codes and data are available at
this https URL.

    

### [[2109.07680] Jointly Modeling Aspect and Polarity for Aspect-based Sentiment Analysis in Persian Reviews](http://arxiv.org/abs/2109.07680)


  Identification of user's opinions from natural language text has become an
exciting field of research due to its growing applications in the real world.
The research field is known as sentiment analysis and classification, where
aspect category detection (ACD) and aspect category polarity (ACP) are two
important sub-tasks of aspect-based sentiment analysis. The goal in ACD is to
specify which aspect of the entity comes up in opinion while ACP aims to
specify the polarity of each aspect category from the ACD task. The previous
works mostly propose separate solutions for these two sub-tasks. This paper
focuses on the ACD and ACP sub-tasks to solve both problems simultaneously. The
proposed method carries out multi-label classification where four different
deep models were employed and comparatively evaluated to examine their
performance. A dataset of Persian reviews was collected from CinemaTicket
website including 2200 samples from 14 categories. The developed models were
evaluated using the collected dataset in terms of example-based and label-based
metrics. The results indicate the high applicability and preference of the CNN
and GRU models in comparison to LSTM and Bi-LSTM.

    

### [[2109.07684] Language Models are Few-shot Multilingual Learners](http://arxiv.org/abs/2109.07684)


  General-purpose language models have demonstrated impressive capabilities,
performing on par with state-of-the-art approaches on a range of downstream
natural language processing (NLP) tasks and benchmarks when inferring
instructions from very few examples. Here, we evaluate the multilingual skills
of the GPT and T5 models in conducting multi-class classification on
non-English languages without any parameter updates. We show that, given a few
English examples as context, pre-trained language models can predict not only
English test samples but also non-English ones. Finally, we find the in-context
few-shot cross-lingual prediction results of language models are significantly
better than random prediction, and they are competitive compared to the
existing state-of-the-art cross-lingual models.

    

### [[2109.07788] Marginal MAP Estimation for Inverse RL under Occlusion with Observer Noise](http://arxiv.org/abs/2109.07788)


  We consider the problem of learning the behavioral preferences of an expert
engaged in a task from noisy and partially-observable demonstrations. This is
motivated by real-world applications such as a line robot learning from
observing a human worker, where some observations are occluded by environmental
objects that cannot be removed. Furthermore, robotic perception tends to be
imperfect and noisy. Previous techniques for inverse reinforcement learning
(IRL) take the approach of either omitting the missing portions or inferring it
as part of expectation-maximization, which tends to be slow and prone to local
optima. We present a new method that generalizes the well-known Bayesian
maximum-a-posteriori (MAP) IRL method by marginalizing the occluded portions of
the trajectory. This is additionally extended with an observation model to
account for perception noise. We show that the marginal MAP (MMAP) approach
significantly improves on the previous IRL technique under occlusion in both
formative evaluations on a toy problem and in a summative evaluation on an
onion sorting line task by a robot.

    

### [[2109.07799] Label-Attention Transformer with Geometrically Coherent Objects for Image Captioning](http://arxiv.org/abs/2109.07799)


  Automatic transcription of scene understanding in images and videos is a step
towards artificial general intelligence. Image captioning is a nomenclature for
describing meaningful information in an image using computer vision techniques.
Automated image captioning techniques utilize encoder and decoder architecture,
where the encoder extracts features from an image and the decoder generates a
transcript. In this work, we investigate two unexplored ideas for image
captioning using transformers: First, we demonstrate the enforcement of using
objects' relevance in the surrounding environment. Second, learning an explicit
association between labels and language constructs. We propose label-attention
Transformer with geometrically coherent objects (LATGeO). The proposed
technique acquires a proposal of geometrically coherent objects using a deep
neural network (DNN) and generates captions by investigating their
relationships using a label-attention module. Object coherence is defined using
the localized ratio of the geometrical properties of the proposals. The
label-attention module associates the extracted objects classes to the
available dictionary using self-attention layers. The experimentation results
show that objects' relevance in surroundings and binding of their visual
feature with their geometrically localized ratios combined with its associated
labels help in defining meaningful captions. The proposed framework is tested
on the MSCOCO dataset, and a thorough evaluation resulting in overall better
quantitative scores pronounces its superiority.

    

### [[2109.07802] Compact Binary Fingerprint for Image Copy Re-Ranking](http://arxiv.org/abs/2109.07802)


  Image copy detection is challenging and appealing topic in computer vision
and signal processing. Recent advancements in multimedia have made distribution
of image across the global easy and fast: that leads to many other issues such
as forgery and image copy retrieval.
Local keypoint descriptors such as SIFT are used to represent the images, and
based on those descriptors matching, images are matched and retrieved. Features
are quantized so that searching/matching may be made feasible for large
databases at the cost of accuracy loss. In this paper, we propose binary
feature that is obtained by quantizing the SIFT into binary, and rank list is
re-examined to remove the false positives. Experiments on challenging dataset
shows the gain in accuracy and time.

    

### [[2109.07827] Enabling risk-aware Reinforcement Learning for medical interventions through uncertainty decomposition](http://arxiv.org/abs/2109.07827)


  Reinforcement Learning (RL) is emerging as tool for tackling complex control
and decision-making problems. However, in high-risk environments such as
healthcare, manufacturing, automotive or aerospace, it is often challenging to
bridge the gap between an apparently optimal policy learnt by an agent and its
real-world deployment, due to the uncertainties and risk associated with it.
Broadly speaking RL agents face two kinds of uncertainty, 1. aleatoric
uncertainty, which reflects randomness or noise in the dynamics of the world,
and 2. epistemic uncertainty, which reflects the bounded knowledge of the agent
due to model limitations and finite amount of information/data the agent has
acquired about the world. These two types of uncertainty carry fundamentally
different implications for the evaluation of performance and the level of risk
or trust. Yet these aleatoric and epistemic uncertainties are generally
confounded as standard and even distributional RL is agnostic to this
difference. Here we propose how a distributional approach (UA-DQN) can be
recast to render uncertainties by decomposing the net effects of each
uncertainty. We demonstrate the operation of this method in grid world examples
to build intuition and then show a proof of concept application for an RL agent
operating as a clinical decision support system in critical care

    

### [[2109.07843] Label Assignment Distillation for Object Detection](http://arxiv.org/abs/2109.07843)


  Knowledge distillation methods are proved to be promising in improving the
performance of neural networks and no additional computational expenses are
required during the inference time. For the sake of boosting the accuracy of
object detection, a great number of knowledge distillation methods have been
proposed particularly designed for object detection. However, most of these
methods only focus on feature-level distillation and label-level distillation,
leaving the label assignment step, a unique and paramount procedure for object
detection, by the wayside. In this work, we come up with a simple but effective
knowledge distillation approach focusing on label assignment in object
detection, in which the positive and negative samples of student network are
selected in accordance with the predictions of teacher network. Our method
shows encouraging results on the MSCOCO2017 benchmark, and can not only be
applied to both one-stage detectors and two-stage detectors but also be
utilized orthogonally with other knowledge distillation methods.

    

### [[2109.07844] Frequent Itemset Mining with Multiple Minimum Supports: a Constraint-based Approach](http://arxiv.org/abs/2109.07844)


  The problem of discovering frequent itemsets including rare ones has received
a great deal of attention. The mining process needs to be flexible enough to
extract frequent and rare regularities at once. On the other hand, it has
recently been shown that constraint programming is a flexible way to tackle
data mining tasks. In this paper, we propose a constraint programming approach
for mining itemsets with multiple minimum supports. Our approach provides the
user with the possibility to express any kind of constraints on the minimum
item supports. An experimental analysis shows the practical effectiveness of
our approach compared to the state of the art.

    

### [[2109.07871] Resolution based Feature Distillation for Cross Resolution Person Re-Identification](http://arxiv.org/abs/2109.07871)


  Person re-identification (re-id) aims to retrieve images of same identities
across different camera views. Resolution mismatch occurs due to varying
distances between person of interest and cameras, this significantly degrades
the performance of re-id in real world scenarios. Most of the existing
approaches resolve the re-id task as low resolution problem in which a low
resolution query image is searched in a high resolution images gallery. Several
approaches apply image super resolution techniques to produce high resolution
images but ignore the multiple resolutions of gallery images which is a better
realistic scenario. In this paper, we introduce channel correlations to improve
the learning of features from the degraded data. In addition, to overcome the
problem of multiple resolutions we propose a Resolution based Feature
Distillation (RFD) approach. Such an approach learns resolution invariant
features by filtering the resolution related features from the final feature
vectors that are used to compute the distance matrix. We tested the proposed
approach on two synthetically created datasets and on one original multi
resolution dataset with real degradation. Our approach improves the performance
when multiple resolutions occur in the gallery and have comparable results in
case of single resolution (low resolution re-id).

    

### [[2109.07906] Ethics of AI: A Systematic Literature Review of Principles and Challenges](http://arxiv.org/abs/2109.07906)


  Ethics in AI becomes a global topic of interest for both policymakers and
academic researchers. In the last few years, various research organizations,
lawyers, think tankers and regulatory bodies get involved in developing AI
ethics guidelines and principles. However, there is still debate about the
implications of these principles. We conducted a systematic literature review
(SLR) study to investigate the agreement on the significance of AI principles
and identify the challenging factors that could negatively impact the adoption
of AI ethics principles. The results reveal that the global convergence set
consists of 22 ethical principles and 15 challenges. Transparency, privacy,
accountability and fairness are identified as the most common AI ethics
principles. Similarly, lack of ethical knowledge and vague principles are
reported as the significant challenges for considering ethics in AI. The
findings of this study are the preliminary inputs for proposing a maturity
model that assess the ethical capabilities of AI systems and provide best
practices for further improvements.

    

### [[2109.07914] Proceedings 37th International Conference on Logic Programming (Technical Communications)](http://arxiv.org/abs/2109.07914)


  ICLP is the premier international event for presenting research in logic
programming.
Contributions to ICLP 2021 were sought in all areas of logic programming,
including but not limited to:
Foundations: Semantics, Formalisms, Nonmonotonic reasoning, Knowledge
representation.
Languages issues: Concurrency, Objects, Coordination, Mobility, Higher order,
Types, Modes, Assertions, Modules, Meta-programming, Logic-based
domain-specific languages, Programming techniques.
Programming support: Program analysis, Transformation, Validation,
Verification, Debugging, Profiling, Testing, Execution visualization.
Implementation: Compilation, Virtual machines, Memory management, Parallel
and Distributed execution, Constraint handling rules, Tabling, Foreign
interfaces, User interfaces.
Related Paradigms and Synergies: Inductive and coinductive logic programming,
Constraint logic programming, Answer set programming, Interaction with SAT, SMT
and CSP solvers, Theorem proving, Argumentation, Probabilistic programming,
Machine learning.
Applications: Databases, Big data, Data integration and federation, Software
engineering, Natural language processing, Web and semantic web, Agents,
Artificial intelligence, Computational life sciences, Cyber-security, Robotics,
Education.

    

### [[2109.07960] Efficient and Effective Generation of Test Cases for Pedestrian Detection -- Search-based Software Testing of Baidu Apollo in SVL](http://arxiv.org/abs/2109.07960)


  With the growing capabilities of autonomous vehicles, there is a higher
demand for sophisticated and pragmatic quality assurance approaches for machine
learning-enabled systems in the automotive AI context. The use of
simulation-based prototyping platforms provides the possibility for early-stage
testing, enabling inexpensive testing and the ability to capture critical
corner-case test scenarios. Simulation-based testing properly complements
conventional on-road testing. However, due to the large space of test input
parameters in these systems, the efficient generation of effective test
scenarios leading to the unveiling of failures is a challenge. This paper
presents a study on testing pedestrian detection and emergency braking system
of the Baidu Apollo autonomous driving platform within the SVL simulator. We
propose an evolutionary automated test generation technique that generates
failure-revealing scenarios for Apollo in the SVL environment. Our approach
models the input space using a generic and flexible data structure and benefits
a multi-criteria safety-based heuristic for the objective function targeted for
optimization. This paper presents the results of our proposed test generation
technique in the 2021 IEEE Autonomous Driving AI Test Challenge. In order to
demonstrate the efficiency and effectiveness of our approach, we also report
the results from a baseline random generation technique. Our evaluation shows
that the proposed evolutionary test case generator is more effective at
generating failure-revealing test cases and provides higher diversity between
the generated failures than the random baseline.

    

### [[2109.07968] Alquist 4.0: Towards Social Intelligence Using Generative Models and Dialogue Personalization](http://arxiv.org/abs/2109.07968)


  The open domain-dialogue system Alquist has a goal to conduct a coherent and
engaging conversation that can be considered as one of the benchmarks of social
intelligence. The fourth version of the system, developed within the Alexa
Prize Socialbot Grand Challenge 4, brings two main innovations. The first
addresses coherence, and the second addresses the engagingness of the
conversation. For innovations regarding coherence, we propose a novel hybrid
approach combining hand-designed responses and a generative model. The proposed
approach utilizes hand-designed dialogues, out-of-domain detection, and a
neural response generator. Hand-designed dialogues walk the user through
high-quality conversational flows. The out-of-domain detection recognizes that
the user diverges from the predefined flow and prevents the system from
producing a scripted response that might not make sense for unexpected user
input. Finally, the neural response generator generates a response based on the
context of the dialogue that correctly reacts to the unexpected user input and
returns the dialogue to the boundaries of hand-designed dialogues. The
innovations for engagement that we propose are mostly inspired by the famous
exploration-exploitation dilemma. To conduct an engaging conversation with the
dialogue partners, one has to learn their preferences and interests --
exploration. Moreover, to engage the partner, we have to utilize the knowledge
we have already learned -- exploitation. In this work, we present the
principles and inner workings of individual components of the open-domain
dialogue system Alquist developed within the Alexa Prize Socialbot Grand
Challenge 4 and the experiments we have conducted to evaluate them.

    

### [[2109.07983] Let the CAT out of the bag: Contrastive Attributed explanations for Text](http://arxiv.org/abs/2109.07983)


  Contrastive explanations for understanding the behavior of black box models
has gained a lot of attention recently as they provide potential for recourse.
In this paper, we propose a method Contrastive Attributed explanations for Text
(CAT) which provides contrastive explanations for natural language text data
with a novel twist as we build and exploit attribute classifiers leading to
more semantically meaningful explanations. To ensure that our contrastive
generated text has the fewest possible edits with respect to the original text,
while also being fluent and close to a human generated contrastive, we resort
to a minimal perturbation approach regularized using a BERT language model and
attribute classifiers trained on available attributes. We show through
qualitative examples and a user study that our method not only conveys more
insight because of these attributes, but also leads to better quality
(contrastive) text. Moreover, quantitatively we show that our method is more
efficient than other state-of-the-art methods with it also scoring higher on
benchmark metrics such as flip rate, (normalized) Levenstein distance, fluency
and content preservation.

    

### [[2109.08006] Deep Algorithmic Question Answering: Towards a Compositionally Hybrid AI for Algorithmic Reasoning](http://arxiv.org/abs/2109.08006)


  An important aspect of artificial intelligence (AI) is the ability to reason
in a step-by-step "algorithmic" manner that can be inspected and verified for
its correctness. This is especially important in the domain of question
answering (QA). We argue that the challenge of algorithmic reasoning in QA can
be effectively tackled with a "systems" approach to AI which features a hybrid
use of symbolic and sub-symbolic methods including deep neural networks.
Additionally, we argue that while neural network models with end-to-end
training pipelines perform well in narrow applications such as image
classification and language modelling, they cannot, on their own, successfully
perform algorithmic reasoning, especially if the task spans multiple domains.
We discuss a few notable exceptions and point out how they are still limited
when the QA problem is widened to include other intelligence-requiring tasks.
However, deep learning, and machine learning in general, do play important
roles as components in the reasoning process. We propose an approach to
algorithm reasoning for QA, Deep Algorithmic Question Answering (DAQA), based
on three desirable properties: interpretability, generalizability and
robustness which such an AI system should possess and conclude that they are
best achieved with a combination of hybrid and compositional AI.

    

### [[2109.08022] Hetero-SCAN: Towards Social Context Aware Fake News Detection via Heterogeneous Graph Neural Network](http://arxiv.org/abs/2109.08022)


  Fake news, false or misleading information presented as news, has a great
impact on many aspects of society, such as politics and healthcare. To handle
this emerging problem, many fake news detection methods have been proposed,
applying Natural Language Processing (NLP) techniques on the article text.
Considering that even people cannot easily distinguish fake news by news
content, these text-based solutions are insufficient. To further improve fake
news detection, researchers suggested graph-based solutions, utilizing the
social context information such as user engagement or publishers information.
However, existing graph-based methods still suffer from the following four
major drawbacks: 1) expensive computational cost due to a large number of user
nodes in the graph, 2) the error in sub-tasks, such as textual encoding or
stance detection, 3) loss of rich social context due to homogeneous
representation of news graphs, and 4) the absence of temporal information
utilization. In order to overcome the aforementioned issues, we propose a novel
social context aware fake news detection method, Hetero-SCAN, based on a
heterogeneous graph neural network. Hetero-SCAN learns the news representation
from the heterogeneous graph of news in an end-to-end manner. We demonstrate
that Hetero-SCAN yields significant improvement over state-of-the-art
text-based and graph-based fake news detection methods in terms of performance
and efficiency.

    

### [[2109.08029] Image Captioning for Effective Use of Language Models in Knowledge-Based Visual Question Answering](http://arxiv.org/abs/2109.08029)


  Integrating outside knowledge for reasoning in visio-linguistic tasks such as
visual question answering (VQA) is an open problem. Given that pretrained
language models have been shown to include world knowledge, we propose to use a
unimodal (text-only) train and inference procedure based on automatic
off-the-shelf captioning of images and pretrained language models. Our results
on a visual question answering task which requires external knowledge (OK-VQA)
show that our text-only model outperforms pretrained multimodal (image-text)
models of comparable number of parameters. In contrast, our model is less
effective in a standard VQA task (VQA 2.0) confirming that our text-only method
is specially effective for tasks requiring external knowledge. In addition, we
show that our unimodal model is complementary to multimodal models in both
OK-VQA and VQA 2.0, and yield the best result to date in OK-VQA among systems
not using external knowledge graphs, and comparable to systems that do use
them. Our qualitative analysis on OK-VQA reveals that automatic captions often
fail to capture relevant information in the images, which seems to be balanced
by the better inference ability of the text-only language models. Our work
opens up possibilities to further improve inference in visio-linguistic tasks.

    

### [[2109.08039] A Survey on Temporal Sentence Grounding in Videos](http://arxiv.org/abs/2109.08039)


  Temporal sentence grounding in videos~(TSGV), which aims to localize one
target segment from an untrimmed video with respect to a given sentence query,
has drawn increasing attentions in the research community over the past few
years. Different from the task of temporal action localization, TSGV is more
flexible since it can locate complicated activities via natural languages,
without restrictions from predefined action categories. Meanwhile, TSGV is more
challenging since it requires both textual and visual understanding for
semantic alignment between two modalities~(i.e., text and video). In this
survey, we give a comprehensive overview for TSGV, which i) summarizes the
taxonomy of existing methods, ii) provides a detailed description of the
evaluation protocols~(i.e., datasets and metrics) to be used in TSGV, and iii)
in-depth discusses potential problems of current benchmarking designs and
research directions for further investigations. To the best of our knowledge,
this is the first systematic survey on temporal sentence grounding. More
specifically, we first discuss existing TSGV approaches by grouping them into
four categories, i.e., two-stage methods, end-to-end methods, reinforcement
learning-based methods, and weakly supervised methods. Then we present the
benchmark datasets and evaluation metrics to assess current research progress.
Finally, we discuss some limitations in TSGV through pointing out potential
problems improperly resolved in the current evaluation protocols, which may
push forwards more cutting edge research in TSGV. Besides, we also share our
insights on several promising directions, including three typical tasks with
new and practical settings based on TSGV.

    

### [[2109.08048] Raising context awareness in motion forecasting](http://arxiv.org/abs/2109.08048)


  Learning-based trajectory prediction models have encountered great success,
with the promise of leveraging contextual information in addition to motion
history. Yet, we find that state-of-the-art forecasting methods tend to overly
rely on the agent's dynamics, failing to exploit the semantic cues provided at
its input. To alleviate this issue, we introduce CAB, a motion forecasting
model equipped with a training procedure designed to promote the use of
semantic contextual information. We also introduce two novel metrics --
dispersion and convergence-to-range -- to measure the temporal consistency of
successive forecasts, which we found missing in standard metrics. Our method is
evaluated on the widely adopted nuScenes Prediction benchmark.

    

### [[2103.04077] Show Me What You Can Do: Capability Calibration on Reachable Workspace for Human-Robot Collaboration](http://arxiv.org/abs/2103.04077)


  Aligning humans' assessment of what a robot can do with its true capability
is crucial for establishing a common ground between human and robot partners
when they collaborate on a joint task. In this work, we propose an approach to
calibrate humans' estimate of a robot's reachable workspace through a small
number of demonstrations before collaboration. We develop a novel motion
planning method, REMP (Reachability-Expressive Motion Planning), which jointly
optimizes the physical cost and the expressiveness of robot motion to reveal
the robot's motion capability to a human observer. Our experiments with human
participants demonstrate that a short calibration using REMP can effectively
bridge the gap between what a non-expert user thinks a robot can reach and the
ground-truth. We show that this calibration procedure not only results in
better user perception, but also promotes more efficient human-robot
collaborations in a subsequent joint task.

    

### [[2105.01652] The Pursuit of Knowledge: Discovering and Localizing Novel Categories using Dual Memory](http://arxiv.org/abs/2105.01652)


  We tackle object category discovery, which is the problem of discovering and
localizing novel objects in a large unlabeled dataset. While existing methods
show results on datasets with less cluttered scenes and fewer object instances
per image, we present our results on the challenging COCO dataset. Moreover, we
argue that, rather than discovering new categories from scratch, discovery
algorithms can benefit from identifying what is already known and focusing
their attention on the unknown. We propose a method that exploits prior
knowledge about certain object types to discover new categories by leveraging
two memory modules, namely Working and Semantic memory. We show the performance
of our detector on the COCO minival dataset to demonstrate its in-the-wild
capabilities.

    

### [[2106.05430] Separating Boundary Points via Structural Regularization for Very Compact Clusters](http://arxiv.org/abs/2106.05430)


  Clustering algorithms have significantly improved along with Deep Neural
Networks which provide effective representation of data. Existing methods are
built upon deep autoencoder and self-training process that leverages the
distribution of cluster assignments of samples. However, as the fundamental
objective of the autoencoder is focused on efficient data reconstruction, the
learnt space may be sub-optimal for clustering. Moreover, it requires highly
effective codes (i.e., representation) of data, otherwise the initial cluster
centers often cause stability issues during self-training. Many
state-of-the-art clustering algorithms use convolution operation to extract
efficient codes but their applications are limited to image data. In this
regard, we propose an end-to-end deep clustering algorithm, i.e., Very Compact
Clusters (VCC). VCC takes advantage of distributions of local relationships of
samples near the boundary of clusters, so that they can be properly separated
and pulled to cluster centers to form compact clusters. Experimental results on
various datasets illustrate that our proposed approach achieves competitive
clustering performance against most of the state-of-the-art clustering methods
for both image and non-image data, and its results can be easily qualitatively
seen in the learnt low-dimensional space.

    

### [[2109.07863] Trillium: Unifying Refinement and Higher-Order Distributed Separation Logic](http://arxiv.org/abs/2109.07863)


  We present a unification of refinement and Hoare-style reasoning in a
foundational mechanized higher-order distributed separation logic. This
unification enables us to prove formally in the Coq proof assistant that
concrete implementations of challenging distributed systems refine more
abstract models and to combine refinement-style reasoning with Hoare-style
program verification. We use our logic to prove correctness of concrete
implementations of two-phase commit and single-decree Paxos by showing that
they refine their abstract TLA+ specifications. We further use our notion of
refinement to transfer fairness assumptions on program executions to model
traces and then transfer liveness properties of fair model traces back to
program executions, which enables us to prove liveness properties such as
strong eventual consistency of a concrete implementation of a Conflict-Free
Replicated Data Type and fair termination of a concurrent program.

    

### [[2109.07923] Efficient Path-Sensitive Data-Dependence Analysis](http://arxiv.org/abs/2109.07923)


  This paper presents a scalable path- and context-sensitive data-dependence
analysis. The key is to address the aliasing-path-explosion problem via a
sparse, demand-driven, and fused approach that piggybacks the computation of
pointer information with the resolution of data dependence. Specifically, our
approach decomposes the computational efforts of disjunctive reasoning into 1)
a context- and semi-path-sensitive analysis that concisely summarizes data
dependence as the symbolic and storeless value-flow graphs, and 2) a
demand-driven phase that resolves transitive data dependence over the graphs.
We have applied the approach to two clients, namely thin slicing and value flow
analysis. Using a suite of 16 programs ranging from 13 KLoC to 8 MLoC, we
compare our techniques against a diverse group of state-of-the-art analyses,
illustrating significant precision and scalability advantages of our approach.

    

### [[2001.07063] Modular coinduction up-to for higher-order languages via first-order transition systems](http://arxiv.org/abs/2001.07063)


  The bisimulation proof method can be enhanced by employing `bisimulations
up-to' techniques. A comprehensive theory of such enhancements has been
developed for first-order (i.e., CCS-like) labelled transition systems (LTSs)
and bisimilarity, based on abstract fixed-point theory and compatible
functions.
We transport this theory onto languages whose bisimilarity and LTS go beyond
those of first-order models. The approach consists in exhibiting fully abstract
translations of the more sophisticated LTSs and bisimilarities onto the
first-order ones. This allows us to reuse directly the large corpus of up-to
techniques that are available on first-order LTSs. The only ingredient that has
to be manually supplied is the compatibility of basic up-to techniques that are
specific to the new languages. We investigate the method on the pi-calculus,
the lambda-calculus, and a (call-by-value) lambda-calculus with references.

    

### [[2103.06127] Linear Constraints](http://arxiv.org/abs/2103.06127)


  A linear argument must be consumed exactly once in the body of its function.
A linear type system can verify the correct usage of resources such as file
handles and manually managed memory. But this verification requires
bureaucracy. This paper presents linear constraints, a front-end feature for
linear typing that decreases the bureaucracy of working with linear types.
Linear constraints are implicit linear arguments that are to be filled in
automatically by the compiler. Linear constraints are presented as a qualified
type system,together with an inference algorithm which extends GHC's existing
constraint solver algorithm. Soundness of linear constraints is ensured by the
fact that they desugar into Linear Haskell.

    