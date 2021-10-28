
## 2021-10-28

### [[2110.13941] Rapid IoT Device Identification at the Edge](http://arxiv.org/abs/2110.13941)


  Consumer Internet of Things (IoT) devices are increasingly common in everyday
homes, from smart speakers to security cameras. Along with their benefits come
potential privacy and security threats. To limit these threats we must
implement solutions to filter IoT traffic at the edge. To this end the
identification of the IoT device is the first natural step.
In this paper we demonstrate a novel method of rapid IoT device
identification that uses neural networks trained on device DNS traffic that can
be captured from a DNS server on the local network. The method identifies
devices by fitting a model to the first seconds of DNS second-level-domain
traffic following their first connection. Since security and privacy threat
detection often operate at a device specific level, rapid identification allows
these strategies to be implemented immediately. Through a total of 51,000
rigorous automated experiments, we classify 30 consumer IoT devices from 27
different manufacturers with 82% and 93% accuracy for product type and device
manufacturers respectively.

    

### [[2110.14211] Beamforming Feedback-based Model-driven Angle of Departure Estimation Toward Firmware-Agnostic WiFi Sensing](http://arxiv.org/abs/2110.14211)


  This paper proves that the angle of departure (AoD) estimation using the
multiple signal classification (MUSIC) with only WiFi control frames for
beamforming feedback (BFF), defined in IEEE 802.11ac/ax, is possible. Although
channel state information (CSI) enables model-driven AoD estimation, most
BFF-based sensing techniques are data-driven because they only contain the
right singular vectors of CSI and subcarrier-averaged stream gain.
Specifically, we find that right singular vectors with a subcarrier-averaged
stream gain of zero have the same role as the noise subspace vectors in the
CSI-based MUSIC algorithm. Numerical evaluations confirm that the proposed
BFF-based MUSIC successfully estimates the AoDs and gains for all propagation
paths. Meanwhile, this result implies a potential privacy risk; a malicious
sniffer can carry out AoD estimation only with unencrypted BFF frames.

    

### [[2110.14224] SOAR: Minimizing Network Utilization with Bounded In-network Computing](http://arxiv.org/abs/2110.14224)


  In-network computing via smart networking devices is a recent trend for
modern datacenter networks. State-of-the-art switches with near line rate
computing and aggregation capabilities are developed to enable, e.g.,
acceleration and better utilization for modern applications like big data
analytics, and large-scale distributed and federated machine learning. We
formulate and study the problem of activating a limited number of in-network
computing devices within a network, aiming at reducing the overall network
utilization for a given workload. Such limitations on the number of in-network
computing elements per workload arise, e.g., in incremental upgrades of network
infrastructure, and are also due to requiring specialized middleboxes, or
FPGAs, that should support heterogeneous workloads, and multiple tenants.
We present an optimal and efficient algorithm for placing such devices in
tree networks with arbitrary link rates, and further evaluate our proposed
solution in various scenarios and for various tasks. Our results show that
having merely a small fraction of network devices support in-network
aggregation can lead to a significant reduction in network utilization.
Furthermore, we show that various intuitive strategies for performing such
placements exhibit significantly inferior performance compared to our solution,
for varying workloads, tasks, and link rates.

    

### [[2110.14239] Sharding and HTTP/2 Connection Reuse Revisited: Why Are There Still Redundant Connections?](http://arxiv.org/abs/2110.14239)


  HTTP/2 and HTTP/3 avoid concurrent connections but instead multiplex requests
over a single connection. Besides enabling new features, this reduces overhead
and enables fair bandwidth sharing. Redundant connections should hence be a
story of the past with HTTP/2. However, they still exist, potentially hindering
innovation and performance. Thus, we measure their spread and analyze their
causes in this paper. We find that 36% - 72% of the 6.24M HTTP Archive and 78%
of the Alexa Top 100k websites cause Chromium-based webbrowsers to open
superfluous connections. We mainly attribute these to domain sharding, despite
HTTP/2 efforts to revert it, and DNS load balancing, but also the Fetch
Standard.

    

### [[2110.14279] SiWa: See into Walls via Deep UWB Radar](http://arxiv.org/abs/2110.14279)


  Being able to see into walls is crucial for diagnostics of building health;
it enables inspections of wall structure without undermining the structural
integrity. However, existing sensing devices do not seem to offer a full
capability in mapping the in-wall structure while identifying their status
(e.g., seepage and corrosion). In this paper, we design and implement SiWa as a
low-cost and portable system for wall inspections. Built upon a customized
IR-UWB radar, SiWa scans a wall as a user swipes its probe along the wall
surface; it then analyzes the reflected signals to synthesize an image and also
to identify the material status. Although conventional schemes exist to handle
these problems individually, they require troublesome calibrations that largely
prevent them from practical adoptions. To this end, we equip SiWa with a deep
learning pipeline to parse the rich sensory data. With an ingenious
construction and innovative training, the deep learning modules perform
structural imaging and the subsequent analysis on material status, without the
need for parameter tuning and calibrations. We build SiWa as a prototype and
evaluate its performance via extensive experiments and field studies; results
confirm that SiWa accurately maps in-wall structures, identifies their
materials, and detects possible failures, suggesting a promising solution for
diagnosing building health with lower effort and cost.

    

### [[2110.14307] RF-Based Human Activity Recognition Using Signal Adapted Convolutional Neural Network](http://arxiv.org/abs/2110.14307)


  Human Activity Recognition (HAR) plays a critical role in a wide range of
real-world applications, and it is traditionally achieved via wearable sensing.
Recently, to avoid the burden and discomfort caused by wearable devices,
device-free approaches exploiting RF signals arise as a promising alternative
for HAR. Most of the latest device-free approaches require training a large
deep neural network model in either time or frequency domain, entailing
extensive storage to contain the model and intensive computations to infer
activities. Consequently, even with some major advances on device-free HAR,
current device-free approaches are still far from practical in real-world
scenarios where the computation and storage resources possessed by, for
example, edge devices, are limited. Therefore, we introduce HAR-SAnet which is
a novel RF-based HAR framework. It adopts an original signal adapted
convolutional neural network architecture: instead of feeding the handcraft
features of RF signals into a classifier, HAR-SAnet fuses them adaptively from
both time and frequency domains to design an end-to-end neural network model.
We apply point-wise grouped convolution and depth-wise separable convolutions
to confine the model scale and to speed up the inference execution time. The
experiment results show that the recognition accuracy of HAR-SAnet outperforms
state-of-the-art algorithms and systems.

    

### [[2110.14389] Charon: Load-Aware Load-Balancing in P4](http://arxiv.org/abs/2110.14389)


  Load-Balancers play an important role in data centers as they distribute
network flows across application servers and guarantee per-connection
consistency. It is hard however to make fair load balancing decisions so that
all resources are efficiently occupied yet not overloaded. Tracking connection
states allows to infer server load states and make informed decisions, but at
the cost of additional memory space consumption. This makes it hard to
implement on programmable hardware, which has constrained memory but offers
line-rate performance. This paper presents Charon, a stateless load-aware load
balancer that has line-rate performance implemented in P4-NetFPGA. Charon
passively collects load states from application servers and employs the
power-of-2-choices scheme to make data-driven load balancing decisions and
improve resource utilization. Perconnection consistency is preserved
statelessly by encoding server ID in a covert channel. The prototype design and
implementation details are described in this paper. Simulation results show
performance gains in terms of load distribution fairness, quality of service,
throughput and processing latency.

    

### [[2110.14578] Spatio-Temporal Federated Learning for Massive Wireless Edge Networks](http://arxiv.org/abs/2110.14578)


  This paper presents a novel approach to conduct highly efficient federated
learning (FL) over a massive wireless edge network, where an edge server and
numerous mobile devices (clients) jointly learn a global model without
transporting the huge amount of data collected by the mobile devices to the
edge server. The proposed FL approach is referred to as spatio-temporal FL
(STFL), which jointly exploits the spatial and temporal correlations between
the learning updates from different mobile devices scheduled to join STFL in
various training epochs. The STFL model not only represents the realistic
intermittent learning behavior from the edge server to the mobile devices due
to data delivery outage, but also features a mechanism of compensating loss
learning updates in order to mitigate the impacts of intermittent learning. An
analytical framework of STFL is proposed and employed to study the learning
capability of STFL via its convergence performance. In particular, we have
assessed the impact of data delivery outage, intermittent learning mitigation,
and statistical heterogeneity of datasets on the convergence performance of
STFL. The results provide crucial insights into the design and analysis of STFL
based wireless networks.

    

### [[2110.14596] Efficient and Secure TSA for the Tangle](http://arxiv.org/abs/2110.14596)


  The Tangle is the data structure used to store transactions in the IOTA
cryptocurrency. In the Tangle, each block has two parents. As a result, the
blocks do not form a chain, but a directed acyclic graph. In traditional
Blockchain, a new block is appended to the heaviest chain in case of fork. In
the Tangle, the parent selection is done by the Tip Selection Algorithm (TSA).
In this paper, we make some important observations about the security of
existing TSAs. We then propose a new TSA that has low complexity and is more
secure than previous TSAs.

    

### [[2105.01308] Defeating Super-Reactive Jammers With Deception Strategy: Modeling, Signal Detection, and Performance Analysis](http://arxiv.org/abs/2105.01308)


  This paper develops a novel framework to defeat a super-reactive jammer, one
of the most difficult jamming attacks to deal with in practice. Specifically,
the jammer has an unlimited power budget and is equipped with the
self-interference suppression capability to simultaneously attack and listen to
the transmitter's activities. Consequently, dealing with super-reactive jammers
is very challenging. Thus, we introduce a smart deception mechanism to attract
the jammer to continuously attack the channel and then leverage jamming signals
to transmit data based on the ambient backscatter communication technology. To
detect the backscattered signals, the maximum likelihood detector can be
adopted. However, this method is notorious for its high computational
complexity and requires the model of the current propagation environment as
well as channel state information. Hence, we propose a deep learning-based
detector that can dynamically adapt to any channels and noise distributions.
With a Long Short-Term Memory network, our detector can learn the received
signals' dependencies to achieve a performance close to that of the optimal
maximum likelihood detector. Through simulation and theoretical results, we
demonstrate that with our approaches, the more power the jammer uses to attack
the channel, the better bit error rate performance the transmitter can achieve.

    

### [[2105.11868] Detection and blind channel estimation for UAV-aided wireless sensor networks in smart cities under mobile jamming attack](http://arxiv.org/abs/2105.11868)


  Unmanned aerial vehicles (UAVs) can be integrated into wireless sensor
networks (WSNs) for smart city applications in several ways. Among them, a UAV
can be employed as a relay in a "store-carry and forward" fashion by uploading
data from ground sensors and metering devices and, then, downloading it to a
central unit. However, both the uploading and downloading phases can be prone
to potential threats and attacks. As a legacy from traditional wireless
networks, the jamming attack is still one of the major and serious threats to
UAV-aided communications, especially when also the jammer is mobile, e.g., it
is mounted on a UAV or inside a terrestrial vehicle. In this paper, we
investigate anti-jamming communications for UAV-aided WSNs operating over
doubly-selective channels in the downloading phase. In such a scenario, the
signals transmitted by the UAV and the malicious mobile jammer undergo both
time dispersion due to multipath propagation effects and frequency dispersion
caused by their mobility. To suppress high-power jamming signals, we propose a
blind physical-layer technique that jointly detects the UAV and jammer symbols
through serial disturbance cancellation based on symbol-level post-sorting of
the detector output. Amplitudes, phases, time delays, and Doppler shifts -
required to implement the proposed detection strategy - are blindly estimated
from data through the use of algorithms that exploit the
almost-cyclostationarity properties of the received signal and the detailed
structure of multicarrier modulation format. Simulation results corroborate the
anti-jamming capabilities of the proposed method, for different mobility
scenarios of the jammer.

    

### [[2110.13911] Modeling Category-Selective Cortical Regions with Topographic Variational Autoencoders](http://arxiv.org/abs/2110.13911)


  Category-selectivity in the brain describes the observation that certain
spatially localized areas of the cerebral cortex tend to respond robustly and
selectively to stimuli from specific limited categories. One of the most well
known examples of category-selectivity is the Fusiform Face Area (FFA), an area
of the inferior temporal cortex in primates which responds preferentially to
images of faces when compared with objects or other generic stimuli. In this
work, we leverage the newly introduced Topographic Variational Autoencoder to
model of the emergence of such localized category-selectivity in an
unsupervised manner. Experimentally, we demonstrate our model yields spatially
dense neural clusters selective to faces, bodies, and places through visualized
maps of Cohen's d metric. We compare our model with related supervised
approaches, namely the TDANN, and discuss both theoretical and empirical
similarities. Finally, we show preliminary results suggesting that our model
yields a nested spatial hierarchy of increasingly abstract categories,
analogous to observations from the human ventral temporal cortex.

    

### [[2110.13937] Provably Robust Model-Centric Explanations for Critical Decision-Making](http://arxiv.org/abs/2110.13937)


  We recommend using a model-centric, Boolean Satisfiability (SAT) formalism to
obtain useful explanations of trained model behavior, different and
complementary to what can be gleaned from LIME and SHAP, popular data-centric
explanation tools in Artificial Intelligence (AI). We compare and contrast
these methods, and show that data-centric methods may yield brittle
explanations of limited practical utility. The model-centric framework,
however, can offer actionable insights into risks of using AI models in
practice. For critical applications of AI, split-second decision making is best
informed by robust explanations that are invariant to properties of data, the
capability offered by model-centric frameworks.

    

### [[2110.13939] CausalAF: Causal Autoregressive Flow for Goal-Directed Safety-Critical Scenes Generation](http://arxiv.org/abs/2110.13939)


  Goal-directed generation, aiming for solving downstream tasks by generating
diverse data, has a potentially wide range of applications in the real world.
Previous works tend to formulate goal-directed generation as a purely
data-driven problem, which directly searches or approximates the distribution
of samples satisfying the goal. However, the generation ability of preexisting
work is heavily restricted by inefficient sampling, especially for sparse goals
that rarely show up in off-the-shelf datasets. For instance, generating
safety-critical traffic scenes with the goal of increasing the risk of
collision is critical to evaluate autonomous vehicles, but the rareness of such
scenes is the biggest resistance. In this paper, we integrate causality as a
prior into the safety-critical scene generation process and propose a
flow-based generative framework - Causal Autoregressive Flow (CausalAF).
CausalAF encourages the generative model to uncover and follow the causal
relationship among generated objects via novel causal masking operations
instead of searching the sample only from observational data. By learning the
cause-and-effect mechanism of how the generated scene achieves the goal rather
than just learning correlations from data, CausalAF significantly improves the
learning efficiency. Extensive experiments on three heterogeneous traffic
scenes illustrate that CausalAF requires much fewer optimization resources to
effectively generate goal-directed scenes for safety evaluation tasks.

    

### [[2110.13947] Collaborative Uncertainty in Multi-Agent Trajectory Forecasting](http://arxiv.org/abs/2110.13947)


  Uncertainty modeling is critical in trajectory forecasting systems for both
interpretation and safety reasons. To better predict the future trajectories of
multiple agents, recent works have introduced interaction modules to capture
interactions among agents. This approach leads to correlations among the
predicted trajectories. However, the uncertainty brought by such correlations
is neglected. To fill this gap, we propose a novel concept, collaborative
uncertainty(CU), which models the uncertainty resulting from the interaction
module. We build a general CU-based framework to make a prediction model to
learn the future trajectory and the corresponding uncertainty. The CU-based
framework is integrated as a plugin module to current state-of-the-art (SOTA)
systems and deployed in two special cases based on multivariate Gaussian and
Laplace distributions. In each case, we conduct extensive experiments on two
synthetic datasets and two public, large-scale benchmarks of trajectory
forecasting. The results are promising: 1) The results of synthetic datasets
show that CU-based framework allows the model to appropriately approximate the
ground-truth distribution. 2) The results of trajectory forecasting benchmarks
demonstrate that the CU-based framework steadily helps SOTA systems improve
their performances. Especially, the proposed CU-based framework helps VectorNet
improve by 57cm regarding Final Displacement Error on nuScenes dataset. 3) The
visualization results of CU illustrate that the value of CU is highly related
to the amount of the interactive information among agents.

    

### [[2110.13948] Boosted CVaR Classification](http://arxiv.org/abs/2110.13948)


  Many modern machine learning tasks require models with high tail performance,
i.e. high performance over the worst-off samples in the dataset. This problem
has been widely studied in fields such as algorithmic fairness, class
imbalance, and risk-sensitive decision making. A popular approach to maximize
the model's tail performance is to minimize the CVaR (Conditional Value at
Risk) loss, which computes the average risk over the tails of the loss.
However, for classification tasks where models are evaluated by the zero-one
loss, we show that if the classifiers are deterministic, then the minimizer of
the average zero-one loss also minimizes the CVaR zero-one loss, suggesting
that CVaR loss minimization is not helpful without additional assumptions. We
circumvent this negative result by minimizing the CVaR loss over randomized
classifiers, for which the minimizers of the average zero-one loss and the CVaR
zero-one loss are no longer the same, so minimizing the latter can lead to
better tail performance. To learn such randomized classifiers, we propose the
Boosted CVaR Classification framework which is motivated by a direct
relationship between CVaR and a classical boosting algorithm called LPBoost.
Based on this framework, we design an algorithm called $\alpha$-AdaLPBoost. We
empirically evaluate our proposed algorithm on four benchmark datasets and show
that it achieves higher tail performance than deterministic model training
methods.

    

### [[2110.13950] Can't Fool Me: Adversarially Robust Transformer for Video Understanding](http://arxiv.org/abs/2110.13950)


  Deep neural networks have been shown to perform poorly on adversarial
examples. To address this, several techniques have been proposed to increase
robustness of a model for image classification tasks. However, in video
understanding tasks, developing adversarially robust models is still
unexplored. In this paper, we aim to bridge this gap. We first show that simple
extensions of image based adversarially robust models slightly improve the
worst-case performance. Further, we propose a temporal attention regularization
scheme in Transformer to improve the robustness of attention modules to
adversarial examples. We illustrate using a large-scale video data set
YouTube-8M that the final model (A-ART) achieves close to non-adversarial
performance on its adversarial example set. We achieve 91% GAP on adversarial
examples, whereas baseline Transformer and simple adversarial extensions
achieve 72.9% and 82% respectively, showing significant improvement in
robustness over the state-of-the-art.

    

### [[2110.13953] On sensitivity of meta-learning to support data](http://arxiv.org/abs/2110.13953)


  Meta-learning algorithms are widely used for few-shot learning. For example,
image recognition systems that readily adapt to unseen classes after seeing
only a few labeled examples. Despite their success, we show that modern
meta-learning algorithms are extremely sensitive to the data used for
adaptation, i.e. support data. In particular, we demonstrate the existence of
(unaltered, in-distribution, natural) images that, when used for adaptation,
yield accuracy as low as 4\% or as high as 95\% on standard few-shot image
classification benchmarks. We explain our empirical findings in terms of class
margins, which in turn suggests that robust and safe meta-learning requires
larger margins than supervised learning.

    

### [[2110.13957] Unbiased Graph Embedding with Biased Graph Observations](http://arxiv.org/abs/2110.13957)


  Graph embedding techniques have been increasingly employed in real-world
machine learning tasks on graph-structured data, such as social recommendations
and protein structure modeling. Since the generation of a graph is inevitably
affected by some sensitive node attributes (such as gender and age of users in
a social network), the learned graph representations can inherit such sensitive
information and introduce undesirable biases in downstream tasks. Most existing
works on debiasing graph representations add ad-hoc constraints on the learned
embeddings to restrict their distributions, which however compromise the
utility of resulting graph representations in downstream tasks.
In this paper, we propose a principled new way for obtaining unbiased
representations by learning from an underlying bias-free graph that is not
influenced by sensitive attributes. Based on this new perspective, we propose
two complementary methods for uncovering such an underlying graph with the goal
of introducing minimum impact on the utility of learned representations in
downstream tasks. Both our theoretical justification and extensive experiment
comparisons against state-of-the-art solutions demonstrate the effectiveness of
our proposed methods.

    

### [[2110.13968] On the Effects of Data Distortion on Model Analysis and Training](http://arxiv.org/abs/2110.13968)


  Data modification can introduce artificial information. It is often assumed
that the resulting artefacts are detrimental to training, whilst being
negligible when analysing models. We investigate these assumptions and conclude
that in some cases they are unfounded and lead to incorrect results.
Specifically, we show current shape bias identification methods and occlusion
robustness measures are biased and propose a fairer alternative for the latter.
Subsequently, through a series of experiments we seek to correct and strengthen
the community's perception of how distorting data affects learning. Based on
our empirical results we argue that the impact of the artefacts must be
understood and exploited rather than eliminated.

    

### [[2110.13969] Nonparametric Matrix Estimation with One-Sided Covariates](http://arxiv.org/abs/2110.13969)


  Consider the task of matrix estimation in which a dataset $X \in
\mathbb{R}^{n\times m}$ is observed with sparsity $p$, and we would like to
estimate $\mathbb{E}[X]$, where $\mathbb{E}[X_{ui}] = f(\alpha_u, \beta_i)$ for
some Holder smooth function $f$. We consider the setting where the row
covariates $\alpha$ are unobserved yet the column covariates $\beta$ are
observed. We provide an algorithm and accompanying analysis which shows that
our algorithm improves upon naively estimating each row separately when the
number of rows is not too small. Furthermore when the matrix is moderately
proportioned, our algorithm achieves the minimax optimal nonparametric rate of
an oracle algorithm that knows the row covariates. In simulated experiments we
show our algorithm outperforms other baselines in low data regimes.

    

### [[2110.13970] Rademacher Random Projections with Tensor Networks](http://arxiv.org/abs/2110.13970)


  Random projection (RP) have recently emerged as popular techniques in
themachine learning community for their ability in reducing the dimension of
veryhigh-dimensional tensors. Following the work in [29], we consider a
tensorizedrandom projection relying on Tensor Train (TT) decomposition where
each elementof the core tensors is drawn from a Rademacher distribution. Our
theoreticalresults reveal that the Gaussian low-rank tensor represented in
compressed formin TT format in [29] can be replaced by a TT tensor with core
elements drawnfrom a Rademacher distribution with the same embedding size.
Experiments onsynthetic data demonstrate that tensorized Rademacher RP can
outperform thetensorized Gaussian RP studied in [29]. In addition, we show both
theoreticallyand experimentally, that the tensorized RP in the Matrix Product
Operator (MPO)format proposed in [5] for performing SVD on large matrices is
not a Johnson-Lindenstrauss transform (JLT) and therefore not a well-suited
random projectionmap

    

### [[2110.13972] Video-based fully automatic assessment of open surgery suturing skills](http://arxiv.org/abs/2110.13972)


  The goal of this study was to develop new reliable open surgery suturing
simulation system for training medical students in situation where resources
are limited or in the domestic setup. Namely, we developed an algorithm for
tools and hands localization as well as identifying the interactions between
them based on simple webcam video data, calculating motion metrics for
assessment of surgical skill. Twenty-five participants performed multiple
suturing tasks using our simulator. The YOLO network has been modified to a
multi-task network, for the purpose of tool localization and tool-hand
interaction detection. This was accomplished by splitting the YOLO detection
heads so that they supported both tasks with minimal addition to computer
run-time. Furthermore, based on the outcome of the system, motion metrics were
calculated. These metrics included traditional metrics such as time and path
length as well as new metrics assessing the technique participants use for
holding the tools. The dual-task network performance was similar to that of two
networks, while computational load was only slightly bigger than one network.
In addition, the motion metrics showed significant differences between experts
and novices. While video capture is an essential part of minimally invasive
surgery, it is not an integral component of open surgery. Thus, new algorithms,
focusing on the unique challenges open surgery videos present, are required. In
this study, a dual-task network was developed to solve both a localization task
and a hand-tool interaction task. The dual network may be easily expanded to a
multi-task network, which may be useful for images with multiple layers and for
evaluating the interaction between these different layers.

    

### [[2110.13973] The Value of Information When Deciding What to Learn](http://arxiv.org/abs/2110.13973)


  All sequential decision-making agents explore so as to acquire knowledge
about a particular target. It is often the responsibility of the agent designer
to construct this target which, in rich and complex environments, constitutes a
onerous burden; without full knowledge of the environment itself, a designer
may forge a sub-optimal learning target that poorly balances the amount of
information an agent must acquire to identify the target against the target's
associated performance shortfall. While recent work has developed a connection
between learning targets and rate-distortion theory to address this challenge
and empower agents that decide what to learn in an automated fashion, the
proposed algorithm does not optimally tackle the equally important challenge of
efficient information acquisition. In this work, building upon the seminal
design principle of information-directed sampling (Russo & Van Roy, 2014), we
address this shortcoming directly to couple optimal information acquisition
with the optimal design of learning targets. Along the way, we offer new
insights into learning targets from the literature on rate-distortion theory
before turning to empirical results that confirm the value of information when
deciding what to learn.

    

### [[2110.13981] CHIP: CHannel Independence-based Pruning for Compact Neural Networks](http://arxiv.org/abs/2110.13981)


  Filter pruning has been widely used for neural network compression because of
its enabled practical acceleration. To date, most of the existing filter
pruning works explore the importance of filters via using intra-channel
information. In this paper, starting from an inter-channel perspective, we
propose to perform efficient filter pruning using Channel Independence, a
metric that measures the correlations among different feature maps. The less
independent feature map is interpreted as containing less useful
information$/$knowledge, and hence its corresponding filter can be pruned
without affecting model capacity. We systematically investigate the
quantification metric, measuring scheme and sensitiveness$/$reliability of
channel independence in the context of filter pruning. Our evaluation results
for different models on various datasets show the superior performance of our
approach. Notably, on CIFAR-10 dataset our solution can bring $0.75\%$ and
$0.94\%$ accuracy increase over baseline ResNet-56 and ResNet-110 models,
respectively, and meanwhile the model size and FLOPs are reduced by $42.8\%$
and $47.4\%$ (for ResNet-56) and $48.3\%$ and $52.1\%$ (for ResNet-110),
respectively. On ImageNet dataset, our approach can achieve $40.8\%$ and
$44.8\%$ storage and computation reductions, respectively, with $0.15\%$
accuracy increase over the baseline ResNet-50 model. The code is available at
this https URL.

    

### [[2110.13985] Combining Recurrent, Convolutional, and Continuous-time Models with Linear State-Space Layers](http://arxiv.org/abs/2110.13985)


  Recurrent neural networks (RNNs), temporal convolutions, and neural
differential equations (NDEs) are popular families of deep learning models for
time-series data, each with unique strengths and tradeoffs in modeling power
and computational efficiency. We introduce a simple sequence model inspired by
control systems that generalizes these approaches while addressing their
shortcomings. The Linear State-Space Layer (LSSL) maps a sequence $u \mapsto y$
by simply simulating a linear continuous-time state-space representation
$\dot{x} = Ax + Bu, y = Cx + Du$. Theoretically, we show that LSSL models are
closely related to the three aforementioned families of models and inherit
their strengths. For example, they generalize convolutions to continuous-time,
explain common RNN heuristics, and share features of NDEs such as time-scale
adaptation. We then incorporate and generalize recent theory on continuous-time
memorization to introduce a trainable subset of structured matrices $A$ that
endow LSSLs with long-range memory. Empirically, stacking LSSL layers into a
simple deep neural network obtains state-of-the-art results across time series
benchmarks for long dependencies in sequential image classification, real-world
healthcare regression tasks, and speech. On a difficult speech classification
task with length-16000 sequences, LSSL outperforms prior approaches by 24
accuracy points, and even outperforms baselines that use hand-crafted features
on 100x shorter sequences.

    

### [[2110.13986] Fair Sequential Selection Using Supervised Learning Models](http://arxiv.org/abs/2110.13986)


  We consider a selection problem where sequentially arrived applicants apply
for a limited number of positions/jobs. At each time step, a decision maker
accepts or rejects the given applicant using a pre-trained supervised learning
model until all the vacant positions are filled. In this paper, we discuss
whether the fairness notions (e.g., equal opportunity, statistical parity,
etc.) that are commonly used in classification problems are suitable for the
sequential selection problems. In particular, we show that even with a
pre-trained model that satisfies the common fairness notions, the selection
outcomes may still be biased against certain demographic groups. This
observation implies that the fairness notions used in classification problems
are not suitable for a selection problem where the applicants compete for a
limited number of positions. We introduce a new fairness notion, ``Equal
Selection (ES),'' suitable for sequential selection problems and propose a
post-processing approach to satisfy the ES fairness notion. We also consider a
setting where the applicants have privacy concerns, and the decision maker only
has access to the noisy version of sensitive attributes. In this setting, we
can show that the perfect ES fairness can still be attained under certain
conditions.

    

### [[2110.13987] Learning Collaborative Policies to Solve NP-hard Routing Problems](http://arxiv.org/abs/2110.13987)


  Recently, deep reinforcement learning (DRL) frameworks have shown potential
for solving NP-hard routing problems such as the traveling salesman problem
(TSP) without problem-specific expert knowledge. Although DRL can be used to
solve complex problems, DRL frameworks still struggle to compete with
state-of-the-art heuristics showing a substantial performance gap. This paper
proposes a novel hierarchical problem-solving strategy, termed learning
collaborative policies (LCP), which can effectively find the near-optimum
solution using two iterative DRL policies: the seeder and reviser. The seeder
generates as diversified candidate solutions as possible (seeds) while being
dedicated to exploring over the full combinatorial action space (i.e., sequence
of assignment action). To this end, we train the seeder's policy using a simple
yet effective entropy regularization reward to encourage the seeder to find
diverse solutions. On the other hand, the reviser modifies each candidate
solution generated by the seeder; it partitions the full trajectory into
sub-tours and simultaneously revises each sub-tour to minimize its traveling
distance. Thus, the reviser is trained to improve the candidate solution's
quality, focusing on the reduced solution space (which is beneficial for
exploitation). Extensive experiments demonstrate that the proposed two-policies
collaboration scheme improves over single-policy DRL framework on various
NP-hard routing problems, including TSP, prize collecting TSP (PCTSP), and
capacitated vehicle routing problem (CVRP).

    

### [[2110.13992] Leveraging Local Temporal Information for Multimodal Scene Classification](http://arxiv.org/abs/2110.13992)


  Robust video scene classification models should capture the spatial
(pixel-wise) and temporal (frame-wise) characteristics of a video effectively.
Transformer models with self-attention which are designed to get contextualized
representations for individual tokens given a sequence of tokens, are becoming
increasingly popular in many computer vision tasks. However, the use of
Transformer based models for video understanding is still relatively
unexplored. Moreover, these models fail to exploit the strong temporal
relationships between the neighboring video frames to get potent frame-level
representations. In this paper, we propose a novel self-attention block that
leverages both local and global temporal relationships between the video frames
to obtain better contextualized representations for the individual frames. This
enables the model to understand the video at various granularities. We
illustrate the performance of our models on the large scale YoutTube-8M data
set on the task of video categorization and further analyze the results to
showcase improvement.

    

### [[2110.13996] Controllable Data Augmentation Through Deep Relighting](http://arxiv.org/abs/2110.13996)


  At the heart of the success of deep learning is the quality of the data.
Through data augmentation, one can train models with better generalization
capabilities and thus achieve greater results in their field of interest. In
this work, we explore how to augment a varied set of image datasets through
relighting so as to improve the ability of existing models to be invariant to
illumination changes, namely for learned descriptors. We develop a tool, based
on an encoder-decoder network, that is able to quickly generate multiple
variations of the illumination of various input scenes whilst also allowing the
user to define parameters such as the angle of incidence and intensity. We
demonstrate that by training models on datasets that have been augmented with
our pipeline, it is possible to achieve higher performance on localization
benchmarks.

    

### [[2110.13998] Efficient Learning and Decoding of the Continuous-Time Hidden Markov Model for Disease Progression Modeling](http://arxiv.org/abs/2110.13998)


  The Continuous-Time Hidden Markov Model (CT-HMM) is an attractive approach to
modeling disease progression due to its ability to describe noisy observations
arriving irregularly in time. However, the lack of an efficient parameter
learning algorithm for CT-HMM restricts its use to very small models or
requires unrealistic constraints on the state transitions. In this paper, we
present the first complete characterization of efficient EM-based learning
methods for CT-HMM models, as well as the first solution to decoding the
optimal state transition sequence and the corresponding state dwelling time. We
show that EM-based learning consists of two challenges: the estimation of
posterior state probabilities and the computation of end-state conditioned
statistics. We solve the first challenge by reformulating the estimation
problem as an equivalent discrete time-inhomogeneous hidden Markov model. The
second challenge is addressed by adapting three distinct approaches from the
continuous time Markov chain (CTMC) literature to the CT-HMM domain.
Additionally, we further improve the efficiency of the most efficient method by
a factor of the number of states. Then, for decoding, we incorporate a
state-of-the-art method from the (CTMC) literature, and extend the end-state
conditioned optimal state sequence decoding to the CT-HMM case with the
computation of the expected state dwelling time. We demonstrate the use of
CT-HMMs with more than 100 states to visualize and predict disease progression
using a glaucoma dataset and an Alzheimer's disease dataset, and to decode and
visualize the most probable state transition trajectory for individuals on the
glaucoma dataset, which helps to identify progressing phenotypes in a
comprehensive way. Finally, we apply the CT-HMM modeling and decoding strategy
to investigate the progression of language acquisition and development.

    

### [[2110.14000] Towards Hyperparameter-free Policy Selection for Offline Reinforcement Learning](http://arxiv.org/abs/2110.14000)


  How to select between policies and value functions produced by different
training algorithms in offline reinforcement learning (RL) -- which is crucial
for hyperpa-rameter tuning -- is an important open question. Existing
approaches based on off-policy evaluation (OPE) often require additional
function approximation and hence hyperparameters, creating a chicken-and-egg
situation. In this paper, we design hyperparameter-free algorithms for policy
selection based on BVFT [XJ21], a recent theoretical advance in value-function
selection, and demonstrate their effectiveness in discrete-action benchmarks
such as Atari. To address performance degradation due to poor critics in
continuous-action domains, we further combine BVFT with OPE to get the best of
both worlds, and obtain a hyperparameter-tuning method for Q-function based OPE
with theoretical guarantees as a side product.

    

### [[2110.14001] SurvITE: Learning Heterogeneous Treatment Effects from Time-to-Event Data](http://arxiv.org/abs/2110.14001)


  We study the problem of inferring heterogeneous treatment effects from
time-to-event data. While both the related problems of (i) estimating treatment
effects for binary or continuous outcomes and (ii) predicting survival outcomes
have been well studied in the recent machine learning literature, their
combination -- albeit of high practical relevance -- has received considerably
less attention. With the ultimate goal of reliably estimating the effects of
treatments on instantaneous risk and survival probabilities, we focus on the
problem of learning (discrete-time) treatment-specific conditional hazard
functions. We find that unique challenges arise in this context due to a
variety of covariate shift issues that go beyond a mere combination of
well-studied confounding and censoring biases. We theoretically analyse their
effects by adapting recent generalization bounds from domain adaptation and
treatment effect estimation to our setting and discuss implications for model
design. We use the resulting insights to propose a novel deep learning method
for treatment-specific hazard estimation based on balancing representations. We
investigate performance across a range of experimental settings and empirically
confirm that our method outperforms baselines by addressing covariate shifts
from various sources.

    

### [[2110.14002] CARMS: Categorical-Antithetic-REINFORCE Multi-Sample Gradient Estimator](http://arxiv.org/abs/2110.14002)


  Accurately backpropagating the gradient through categorical variables is a
challenging task that arises in various domains, such as training discrete
latent variable models. To this end, we propose CARMS, an unbiased estimator
for categorical random variables based on multiple mutually negatively
correlated (jointly antithetic) samples. CARMS combines REINFORCE with copula
based sampling to avoid duplicate samples and reduce its variance, while
keeping the estimator unbiased using importance sampling. It generalizes both
the ARMS antithetic estimator for binary variables, which is CARMS for two
categories, as well as LOORF/VarGrad, the leave-one-out REINFORCE estimator,
which is CARMS with independent samples. We evaluate CARMS on several benchmark
datasets on a generative modeling task, as well as a structured output
prediction task, and find it to outperform competing methods including a strong
self-control baseline. The code is publicly available.

    

### [[2110.14007] TOD: Tensor-based Outlier Detection](http://arxiv.org/abs/2110.14007)


  To scale outlier detection (OD) to large-scale, high-dimensional datasets, we
propose TOD, a novel system that abstracts OD algorithms into basic tensor
operations for efficient GPU acceleration. To make TOD highly efficient in both
time and space, we leverage recent advances in deep learning infrastructure in
both hardware and software. To deploy large OD applications on GPUs with
limited memory, we introduce two key techniques. First, provable quantization
accelerates OD computation and reduces the memory requirement by performing
specific OD computations in lower precision while provably guaranteeing no
accuracy loss. Second, to exploit the aggregated compute resources and memory
capacity of multiple GPUs, we introduce automatic batching, which decomposes OD
computations into small batches that can be executed on multiple GPUs in
parallel.
TOD supports a comprehensive set of OD algorithms and utility functions.
Extensive evaluation on both real and synthetic OD datasets shows that TOD is
on average 11.9X faster than the state-of-the-art comprehensive OD system PyOD,
and takes less than an hour to detect outliers within a million samples. TOD
enables straightforward integration for additional OD algorithms and provides a
unified framework for combining classical OD algorithms with deep learning
methods. These combinations result in an infinite number of OD methods, many of
which are novel and can be easily prototyped in TOD.

    

### [[2110.14010] MisConv: Convolutional Neural Networks for Missing Data](http://arxiv.org/abs/2110.14010)


  Processing of missing data by modern neural networks, such as CNNs, remains a
fundamental, yet unsolved challenge, which naturally arises in many practical
applications, like image inpainting or autonomous vehicles and robots. While
imputation-based techniques are still one of the most popular solutions, they
frequently introduce unreliable information to the data and do not take into
account the uncertainty of estimation, which may be destructive for a machine
learning model. In this paper, we present MisConv, a general mechanism, for
adapting various CNN architectures to process incomplete images. By modeling
the distribution of missing values by the Mixture of Factor Analyzers, we cover
the spectrum of possible replacements and find an analytical formula for the
expected value of convolution operator applied to the incomplete image. The
whole framework is realized by matrix operations, which makes MisConv extremely
efficient in practice. Experiments performed on various image processing tasks
demonstrate that MisConv achieves superior or comparable performance to the
state-of-the-art methods.

    

### [[2110.14011] Cluster-and-Conquer: A Framework For Time-Series Forecasting](http://arxiv.org/abs/2110.14011)


  We propose a three-stage framework for forecasting high-dimensional
time-series data. Our method first estimates parameters for each univariate
time series. Next, we use these parameters to cluster the time series. These
clusters can be viewed as multivariate time series, for which we then compute
parameters. The forecasted values of a single time series can depend on the
history of other time series in the same cluster, accounting for intra-cluster
similarity while minimizing potential noise in predictions by ignoring
inter-cluster effects. Our framework -- which we refer to as
"cluster-and-conquer" -- is highly general, allowing for any time-series
forecasting and clustering method to be used in each step. It is
computationally efficient and embarrassingly parallel. We motivate our
framework with a theoretical analysis in an idealized mixed linear regression
setting, where we provide guarantees on the quality of the estimates. We
accompany these guarantees with experimental results that demonstrate the
advantages of our framework: when instantiated with simple linear
autoregressive models, we are able to achieve state-of-the-art results on
several benchmark datasets, sometimes outperforming deep-learning-based
approaches.

    

### [[2110.14012] Graph Posterior Network: Bayesian Predictive Uncertainty for Node Classification](http://arxiv.org/abs/2110.14012)


  The interdependence between nodes in graphs is key to improve class
predictions on nodes and utilized in approaches like Label Propagation (LP) or
in Graph Neural Networks (GNN). Nonetheless, uncertainty estimation for
non-independent node-level predictions is under-explored. In this work, we
explore uncertainty quantification for node classification in three ways: (1)
We derive three axioms explicitly characterizing the expected predictive
uncertainty behavior in homophilic attributed graphs. (2) We propose a new
model Graph Posterior Network (GPN) which explicitly performs Bayesian
posterior updates for predictions on interdependent nodes. GPN provably obeys
the proposed axioms. (3) We extensively evaluate GPN and a strong set of
baselines on semi-supervised node classification including detection of
anomalous features, and detection of left-out classes. GPN outperforms existing
approaches for uncertainty estimation in the experiments.

    

### [[2110.14013] Deep Integrated Pipeline of Segmentation Leading to Classification for Automated Detection of Breast Cancer from Breast Ultrasound Images](http://arxiv.org/abs/2110.14013)


  Breast cancer has become a symbol of tremendous concern in the modern world,
as it is one of the major causes of cancer mortality worldwide. In this
concern, many people are frequently screening for breast cancer in order to be
identified early and avert mortality from the disease by receiving treatment.
Breast Ultrasonography Images are frequently utilized by doctors to diagnose
breast cancer at an early stage. However, the complex artifacts and heavily
noised Breast Ultrasonography Images make detecting Breast Cancer a tough
challenge. Furthermore, the ever-increasing number of patients being screened
for Breast Cancer necessitates the use of automated Computer Aided Technology
for high accuracy diagnosis at a cheap cost and in a short period of time. The
current progress of Artificial Intelligence (AI) in the fields of Medical Image
Analysis and Health Care is a boon to humanity. In this study, we have proposed
a compact integrated automated pipelining framework which integrates
ultrasonography image preprocessing with Simple Linear Iterative Clustering
(SLIC) to tackle the complex artifact of Breast Ultrasonography Images
complementing semantic segmentation with Modified U-Net leading to Breast Tumor
classification with robust feature extraction using a transfer learning
approach with pretrained VGG 16 model and densely connected neural network
architecture. The proposed automated pipeline can be effectively implemented to
assist medical practitioners in making more accurate and timely diagnoses of
breast cancer.

    

### [[2110.14019] Reliable and Trustworthy Machine Learning for Health Using Dataset Shift Detection](http://arxiv.org/abs/2110.14019)


  Unpredictable ML model behavior on unseen data, especially in the health
domain, raises serious concerns about its safety as repercussions for mistakes
can be fatal. In this paper, we explore the feasibility of using
state-of-the-art out-of-distribution detectors for reliable and trustworthy
diagnostic predictions. We select publicly available deep learning models
relating to various health conditions (e.g., skin cancer, lung sound, and
Parkinson's disease) using various input data types (e.g., image, audio, and
motion data). We demonstrate that these models show unreasonable predictions on
out-of-distribution datasets. We show that Mahalanobis distance- and Gram
matrices-based out-of-distribution detection methods are able to detect
out-of-distribution data with high accuracy for the health models that operate
on different modalities. We then translate the out-of-distribution score into a
human interpretable CONFIDENCE SCORE to investigate its effect on the users'
interaction with health ML applications. Our user study shows that the
\textsc{confidence score} helped the participants only trust the results with a
high score to make a medical decision and disregard results with a low score.
Through this work, we demonstrate that dataset shift is a critical piece of
information for high-stake ML applications, such as medical diagnosis and
healthcare, to provide reliable and trustworthy predictions to the users.

    

### [[2110.14020] The Difficulty of Passive Learning in Deep Reinforcement Learning](http://arxiv.org/abs/2110.14020)


  Learning to act from observational data without active environmental
interaction is a well-known challenge in Reinforcement Learning (RL). Recent
approaches involve constraints on the learned policy or conservative updates,
preventing strong deviations from the state-action distribution of the dataset.
Although these methods are evaluated using non-linear function approximation,
theoretical justifications are mostly limited to the tabular or linear cases.
Given the impressive results of deep reinforcement learning, we argue for a
need to more clearly understand the challenges in this setting.
In the vein of Held & Hein's classic 1963 experiment, we propose the "tandem
learning" experimental paradigm which facilitates our empirical analysis of the
difficulties in offline reinforcement learning. We identify function
approximation in conjunction with fixed data distributions as the strongest
factors, thereby extending but also challenging hypotheses stated in past work.
Our results provide relevant insights for offline deep reinforcement learning,
while also shedding new light on phenomena observed in the online case of
learning control.

    

### [[2110.14030] Improving Local Effectiveness for Global robust training](http://arxiv.org/abs/2110.14030)


  Despite its popularity, deep neural networks are easily fooled. To alleviate
this deficiency, researchers are actively developing new training strategies,
which encourage models that are robust to small input perturbations. Several
successful robust training methods have been proposed. However, many of them
rely on strong adversaries, which can be prohibitively expensive to generate
when the input dimension is high and the model structure is complicated. We
adopt a new perspective on robustness and propose a novel training algorithm
that allows a more effective use of adversaries. Our method improves the model
robustness at each local ball centered around an adversary and then, by
combining these local balls through a global term, achieves overall robustness.
We demonstrate that, by maximizing the use of adversaries via focusing on local
balls, we achieve high robust accuracy with weak adversaries. Specifically, our
method reaches a similar robust accuracy level to the state of the art
approaches trained on strong adversaries on MNIST, CIFAR-10 and CIFAR-100. As a
result, the overall training time is reduced. Furthermore, when trained with
strong adversaries, our method matches with the current state of the art on
MNIST and outperforms them on CIFAR-10 and CIFAR-100.

    

### [[2110.14031] Surrogate Regret Bounds for Polyhedral Losses](http://arxiv.org/abs/2110.14031)


  Surrogate risk minimization is an ubiquitous paradigm in supervised machine
learning, wherein a target problem is solved by minimizing a surrogate loss on
a dataset. Surrogate regret bounds, also called excess risk bounds, are a
common tool to prove generalization rates for surrogate risk minimization.
While surrogate regret bounds have been developed for certain classes of loss
functions, such as proper losses, general results are relatively sparse. We
provide two general results. The first gives a linear surrogate regret bound
for any polyhedral (piecewise-linear and convex) surrogate, meaning that
surrogate generalization rates translate directly to target rates. The second
shows that for sufficiently non-polyhedral surrogates, the regret bound is a
square root, meaning fast surrogate generalization rates translate to slow
rates for the target. Together, these results suggest polyhedral surrogates are
optimal in many cases.

    

### [[2110.14032] MEST: Accurate and Fast Memory-Economic Sparse Training Framework on the Edge](http://arxiv.org/abs/2110.14032)


  Recently, a new trend of exploring sparsity for accelerating neural network
training has emerged, embracing the paradigm of training on the edge. This
paper proposes a novel Memory-Economic Sparse Training (MEST) framework
targeting for accurate and fast execution on edge devices. The proposed MEST
framework consists of enhancements by Elastic Mutation (EM) and Soft Memory
Bound (&S) that ensure superior accuracy at high sparsity ratios. Different
from the existing works for sparse training, this current work reveals the
importance of sparsity schemes on the performance of sparse training in terms
of accuracy as well as training speed on real edge devices. On top of that, the
paper proposes to employ data efficiency for further acceleration of sparse
training. Our results suggest that unforgettable examples can be identified
in-situ even during the dynamic exploration of sparsity masks in the sparse
training process, and therefore can be removed for further training speedup on
edge devices. Comparing with state-of-the-art (SOTA) works on accuracy, our
MEST increases Top-1 accuracy significantly on ImageNet when using the same
unstructured sparsity scheme. Systematical evaluation on accuracy, training
speed, and memory footprint are conducted, where the proposed MEST framework
consistently outperforms representative SOTA works. A reviewer strongly against
our work based on his false assumptions and misunderstandings. On top of the
previous submission, we employ data efficiency for further acceleration of
sparse training. And we explore the impact of model sparsity, sparsity schemes,
and sparse training algorithms on the number of removable training examples.
Our codes are publicly available at: this https URL.

    

### [[2110.14037] Revisiting the Performance of iALS on Item Recommendation Benchmarks](http://arxiv.org/abs/2110.14037)


  Matrix factorization learned by implicit alternating least squares (iALS) is
a popular baseline in recommender system research publications. iALS is known
to be one of the most computationally efficient and scalable collaborative
filtering methods. However, recent studies suggest that its prediction quality
is not competitive with the current state of the art, in particular
autoencoders and other item-based collaborative filtering methods. In this
work, we revisit the iALS algorithm and present a bag of tricks that we found
useful when applying iALS. We revisit four well-studied benchmarks where iALS
was reported to perform poorly and show that with proper tuning, iALS is highly
competitive and outperforms any method on at least half of the comparisons. We
hope that these high quality results together with iALS's known scalability
spark new interest in applying and further improving this decade old technique.

    

### [[2110.14038] Robustness of Graph Neural Networks at Scale](http://arxiv.org/abs/2110.14038)


  Graph Neural Networks (GNNs) are increasingly important given their
popularity and the diversity of applications. Yet, existing studies of their
vulnerability to adversarial attacks rely on relatively small graphs. We
address this gap and study how to attack and defend GNNs at scale. We propose
two sparsity-aware first-order optimization attacks that maintain an efficient
representation despite optimizing over a number of parameters which is
quadratic in the number of nodes. We show that common surrogate losses are not
well-suited for global attacks on GNNs. Our alternatives can double the attack
strength. Moreover, to improve GNNs' reliability we design a robust aggregation
function, Soft Median, resulting in an effective defense at all scales. We
evaluate our attacks and defense with standard GNNs on graphs more than 100
times larger compared to previous work. We even scale one order of magnitude
further by extending our techniques to a scalable GNN.

    

### [[2110.14044] iALS++: Speeding up Matrix Factorization with Subspace Optimization](http://arxiv.org/abs/2110.14044)


  iALS is a popular algorithm for learning matrix factorization models from
implicit feedback with alternating least squares. This algorithm was invented
over a decade ago but still shows competitive quality compared to recent
approaches like VAE, EASE, SLIM, or NCF. Due to a computational trick that
avoids negative sampling, iALS is very efficient especially for large item
catalogues. However, iALS does not scale well with large embedding dimensions,
d, due to its cubic runtime dependency on d. Coordinate descent variations,
iCD, have been proposed to lower the complexity to quadratic in d. In this
work, we show that iCD approaches are not well suited for modern processors and
can be an order of magnitude slower than a careful iALS implementation for
small to mid scale embedding sizes (d ~ 100) and only perform better than iALS
on large embeddings d ~ 1000. We propose a new solver iALS++ that combines the
advantages of iALS in terms of vector processing with a low computational
complexity as in iCD. iALS++ is an order of magnitude faster than iCD both for
small and large embedding dimensions. It can solve benchmark problems like
Movielens 20M or Million Song Dataset even for 1000 dimensional embedding
vectors in a few minutes.

    

### [[2110.14048] Conflict-Averse Gradient Descent for Multi-task Learning](http://arxiv.org/abs/2110.14048)


  The goal of multi-task learning is to enable more efficient learning than
single task learning by sharing model structures for a diverse set of tasks. A
standard multi-task learning objective is to minimize the average loss across
all tasks. While straightforward, using this objective often results in much
worse final performance for each task than learning them independently. A major
challenge in optimizing a multi-task model is the conflicting gradients, where
gradients of different task objectives are not well aligned so that following
the average gradient direction can be detrimental to specific tasks'
performance. Previous work has proposed several heuristics to manipulate the
task gradients for mitigating this problem. But most of them lack convergence
guarantee and/or could converge to any Pareto-stationary point. In this paper,
we introduce Conflict-Averse Gradient descent (CAGrad) which minimizes the
average loss function, while leveraging the worst local improvement of
individual tasks to regularize the algorithm trajectory. CAGrad balances the
objectives automatically and still provably converges to a minimum over the
average loss. It includes the regular gradient descent (GD) and the multiple
gradient descent algorithm (MGDA) in the multi-objective optimization (MOO)
literature as special cases. On a series of challenging multi-task supervised
learning and reinforcement learning tasks, CAGrad achieves improved performance
over prior state-of-the-art multi-objective gradient manipulation methods.

    

### [[2110.14049] Beta Shapley: a Unified and Noise-reduced Data Valuation Framework for Machine Learning](http://arxiv.org/abs/2110.14049)


  Data Shapley has recently been proposed as a principled framework to quantify
the contribution of individual datum in machine learning. It can effectively
identify helpful or harmful data points for a learning algorithm. In this
paper, we propose Beta Shapley, which is a substantial generalization of Data
Shapley. Beta Shapley arises naturally by relaxing the efficiency axiom of the
Shapley value, which is not critical for machine learning settings. Beta
Shapley unifies several popular data valuation methods and includes data
Shapley as a special case. Moreover, we prove that Beta Shapley has several
desirable statistical properties and propose efficient algorithms to estimate
it. We demonstrate that Beta Shapley outperforms state-of-the-art data
valuation methods on several downstream ML tasks such as: 1) detecting
mislabeled training data; 2) learning with subsamples; and 3) identifying
points whose addition or removal have the largest positive or negative impact
on the model.

    

### [[2110.14051] A Unified Survey on Anomaly, Novelty, Open-Set, and Out-of-Distribution Detection: Solutions and Future Challenges](http://arxiv.org/abs/2110.14051)


  Machine learning models often encounter samples that are diverged from the
training distribution. Failure to recognize an out-of-distribution (OOD)
sample, and consequently assign that sample to an in-class label significantly
compromises the reliability of a model. The problem has gained significant
attention due to its importance for safety deploying models in open-world
settings. Detecting OOD samples is challenging due to the intractability of
modeling all possible unknown distributions. To date, several research domains
tackle the problem of detecting unfamiliar samples, including anomaly
detection, novelty detection, one-class learning, open set recognition, and
out-of-distribution detection. Despite having similar and shared concepts,
out-of-distribution, open-set, and anomaly detection have been investigated
independently. Accordingly, these research avenues have not cross-pollinated,
creating research barriers. While some surveys intend to provide an overview of
these approaches, they seem to only focus on a specific domain without
examining the relationship between different domains. This survey aims to
provide a cross-domain and comprehensive review of numerous eminent works in
respective areas while identifying their commonalities. Researchers can benefit
from the overview of research advances in different fields and develop future
methodology synergistically. Furthermore, to the best of our knowledge, while
there are surveys in anomaly detection or one-class learning, there is no
comprehensive or up-to-date survey on out-of-distribution detection, which our
survey covers extensively. Finally, having a unified cross-domain perspective,
we discuss and shed light on future lines of research, intending to bring these
fields closer together.

    

### [[2110.14053] Improving SAT Solving with Graph Neural Networks](http://arxiv.org/abs/2110.14053)


  Propositional satisfiability (SAT) is an NP-complete problem that impacts
many research fields, such as planning, verification, and security. Despite the
remarkable success of modern SAT solvers, scalability still remains a
challenge. Main stream modern SAT solvers are based on the Conflict-Driven
Clause Learning (CDCL) algorithm. Recent work aimed to enhance CDCL SAT solvers
by improving its variable branching heuristics through predictions generated by
Graph Neural Networks (GNNs). However, so far this approach either has not made
solving more effective, or has required frequent online accesses to substantial
GPU resources. Aiming to make GNN improvements practical, this paper proposes
an approach called NeuroComb, which builds on two insights: (1) predictions of
important variables and clauses can be combined with dynamic branching into a
more effective hybrid branching strategy, and (2) it is sufficient to query the
neural model only once for the predictions before the SAT solving starts.
Implemented as an enhancement to the classic MiniSat solver, NeuroComb allowed
it to solve 18.5% more problems on the recent SATCOMP-2020 competition problem
set. NeuroComb is therefore a practical approach to improving SAT solving
through modern machine learning.

    

### [[2110.14055] Polynomial-Spline Neural Networks with Exact Integrals](http://arxiv.org/abs/2110.14055)


  Using neural networks to solve variational problems, and other scientific
machine learning tasks, has been limited by a lack of consistency and an
inability to exactly integrate expressions involving neural network
architectures. We address these limitations by formulating a novel neural
network architecture that combines a polynomial mixture-of-experts model with
free knot B1-spline basis functions. Effectively, our architecture performs
piecewise polynomial approximation on each cell of a trainable partition of
unity. Our architecture exhibits both $h$- and $p$- refinement for regression
problems at the convergence rates expected from approximation theory, allowing
for consistency in solving variational problems. Moreover, this architecture,
its moments, and its partial derivatives can all be integrated exactly,
obviating a reliance on sampling or quadrature and enabling error-free
computation of variational forms. We demonstrate the success of our network on
a range of regression and variational problems that illustrate the consistency
and exact integrability of our network architecture.

    

### [[2110.14056] How to transfer algorithmic reasoning knowledge to learn new algorithms?](http://arxiv.org/abs/2110.14056)


  Learning to execute algorithms is a fundamental problem that has been widely
studied. Prior work~\cite{veli19neural} has shown that to enable systematic
generalisation on graph algorithms it is critical to have access to the
intermediate steps of the program/algorithm. In many reasoning tasks, where
algorithmic-style reasoning is important, we only have access to the input and
output examples. Thus, inspired by the success of pre-training on similar tasks
or data in Natural Language Processing (NLP) and Computer Vision, we set out to
study how we can transfer algorithmic reasoning knowledge. Specifically, we
investigate how we can use algorithms for which we have access to the execution
trace to learn to solve similar tasks for which we do not. We investigate two
major classes of graph algorithms, parallel algorithms such as breadth-first
search and Bellman-Ford and sequential greedy algorithms such as Prim and
Dijkstra. Due to the fundamental differences between algorithmic reasoning
knowledge and feature extractors such as used in Computer Vision or NLP, we
hypothesise that standard transfer techniques will not be sufficient to achieve
systematic generalisation. To investigate this empirically we create a dataset
including 9 algorithms and 3 different graph types. We validate this
empirically and show how instead multi-task learning can be used to achieve the
transfer of algorithmic reasoning knowledge.

    

### [[2110.14057] Meta-learning with an Adaptive Task Scheduler](http://arxiv.org/abs/2110.14057)


  To benefit the learning of a new task, meta-learning has been proposed to
transfer a well-generalized meta-model learned from various meta-training
tasks. Existing meta-learning algorithms randomly sample meta-training tasks
with a uniform probability, under the assumption that tasks are of equal
importance. However, it is likely that tasks are detrimental with noise or
imbalanced given a limited number of meta-training tasks. To prevent the
meta-model from being corrupted by such detrimental tasks or dominated by tasks
in the majority, in this paper, we propose an adaptive task scheduler (ATS) for
the meta-training process. In ATS, for the first time, we design a neural
scheduler to decide which meta-training tasks to use next by predicting the
probability being sampled for each candidate task, and train the scheduler to
optimize the generalization capacity of the meta-model to unseen tasks. We
identify two meta-model-related factors as the input of the neural scheduler,
which characterize the difficulty of a candidate task to the meta-model.
Theoretically, we show that a scheduler taking the two factors into account
improves the meta-training loss and also the optimization landscape. Under the
setting of meta-learning with noise and limited budgets, ATS improves the
performance on both miniImageNet and a real-world drug discovery benchmark by
up to 13% and 18%, respectively, compared to state-of-the-art task schedulers.

    

### [[2110.14066] Model Reduction of Swing Equations with Physics Informed PDE](http://arxiv.org/abs/2110.14066)


  This manuscript is the first step towards building a robust and efficient
model reduction methodology to capture transient dynamics in a transmission
level electric power system. Such dynamics is normally modeled on
seconds-to-tens-of-seconds time scales by the so-called swing equations, which
are ordinary differential equations defined on a spatially discrete model of
the power grid. We suggest, following Seymlyen (1974) and Thorpe, Seyler and
Phadke (1999), to map the swing equations onto a linear, inhomogeneous Partial
Differential Equation (PDE) of parabolic type in two space and one time
dimensions with time-independent coefficients and properly defined boundary
conditions. The continuous two-dimensional spatial domain is defined by a
geographical map of the area served by the power grid, and associated with the
PDE coefficients derived from smoothed graph-Laplacian of susceptances, machine
inertia and damping. Inhomogeneous source terms represent spatially distributed
injection/consumption of power. We illustrate our method on PanTaGruEl
(Pan-European Transmission Grid and ELectricity generation model). We show
that, when properly coarse-grained, i.e. with the PDE coefficients and source
terms extracted from a spatial convolution procedure of the respective discrete
coefficients in the swing equations, the resulting PDE reproduces faithfully
and efficiently the original swing dynamics. We finally discuss future
extensions of this work, where the presented PDE-based reduced modeling will
initialize a physics-informed machine learning approach for real-time modeling,
$n-1$ feasibility assessment and transient stability analysis of power systems.

    

### [[2110.14068] Drawing Robust Scratch Tickets: Subnetworks with Inborn Robustness Are Found within Randomly Initialized Networks](http://arxiv.org/abs/2110.14068)


  Deep Neural Networks (DNNs) are known to be vulnerable to adversarial
attacks, i.e., an imperceptible perturbation to the input can mislead DNNs
trained on clean images into making erroneous predictions. To tackle this,
adversarial training is currently the most effective defense method, by
augmenting the training set with adversarial samples generated on the fly.
Interestingly, we discover for the first time that there exist subnetworks with
inborn robustness, matching or surpassing the robust accuracy of the
adversarially trained networks with comparable model sizes, within randomly
initialized networks without any model training, indicating that adversarial
training on model weights is not indispensable towards adversarial robustness.
We name such subnetworks Robust Scratch Tickets (RSTs), which are also by
nature efficient. Distinct from the popular lottery ticket hypothesis, neither
the original dense networks nor the identified RSTs need to be trained. To
validate and understand this fascinating finding, we further conduct extensive
experiments to study the existence and properties of RSTs under different
models, datasets, sparsity patterns, and attacks, drawing insights regarding
the relationship between DNNs' robustness and their
initialization/overparameterization. Furthermore, we identify the poor
adversarial transferability between RSTs of different sparsity ratios drawn
from the same randomly initialized dense network, and propose a Random RST
Switch (R2S) technique, which randomly switches between different RSTs, as a
novel defense method built on top of RSTs. We believe our findings about RSTs
have opened up a new perspective to study model robustness and extend the
lottery ticket hypothesis.

    

### [[2110.14074] Fault-Tolerant Federated Reinforcement Learning with Theoretical Guarantee](http://arxiv.org/abs/2110.14074)


  The growing literature of Federated Learning (FL) has recently inspired
Federated Reinforcement Learning (FRL) to encourage multiple agents to
federatively build a better decision-making policy without sharing raw
trajectories. Despite its promising applications, existing works on FRL fail to
I) provide theoretical analysis on its convergence, and II) account for random
system failures and adversarial attacks. Towards this end, we propose the first
FRL framework the convergence of which is guaranteed and tolerant to less than
half of the participating agents being random system failures or adversarial
attackers. We prove that the sample efficiency of the proposed framework is
guaranteed to improve with the number of agents and is able to account for such
potential failures or attacks. All theoretical results are empirically verified
on various RL benchmark tasks.

    

### [[2110.14078] Adversarial Online Learning with Variable Plays in the Pursuit-Evasion Game: Theoretical Foundations and Application in Connected and Automated Vehicle Cybersecurity](http://arxiv.org/abs/2110.14078)


  We extend the adversarial/non-stochastic multi-play multi-armed bandit
(MPMAB) to the case where the number of arms to play is variable. The work is
motivated by the fact that the resources allocated to scan different critical
locations in an interconnected transportation system change dynamically over
time and depending on the environment. By modeling the malicious hacker and the
intrusion monitoring system as the attacker and the defender, respectively, we
formulate the problem for the two players as a sequential pursuit-evasion game.
We derive the condition under which a Nash equilibrium of the strategic game
exists. For the defender side, we provide an exponential-weighted based
algorithm with sublinear pseudo-regret. We further extend our model to
heterogeneous rewards for both players, and obtain lower and upper bounds on
the average reward for the attacker. We provide numerical experiments to
demonstrate the effectiveness of a variable-arm play.

    

### [[2110.14092] BioGrad: Biologically Plausible Gradient-Based Learning for Spiking Neural Networks](http://arxiv.org/abs/2110.14092)


  Spiking neural networks (SNN) are delivering energy-efficient, massively
parallel, and low-latency solutions to AI problems, facilitated by the emerging
neuromorphic chips. To harness these computational benefits, SNN need to be
trained by learning algorithms that adhere to brain-inspired neuromorphic
principles, namely event-based, local, and online computations. Yet, the
state-of-the-art SNN training algorithms are based on backprop that does not
follow the above principles. Due to its limited biological plausibility, the
application of backprop to SNN requires non-local feedback pathways for
transmitting continuous-valued errors, and relies on gradients from future
timesteps. The introduction of biologically plausible modifications to backprop
has helped overcome several of its limitations, but limits the degree to which
backprop is approximated, which hinders its performance. We propose a
biologically plausible gradient-based learning algorithm for SNN that is
functionally equivalent to backprop, while adhering to all three neuromorphic
principles. We introduced multi-compartment spiking neurons with local
eligibility traces to compute the gradients required for learning, and a
periodic "sleep" phase to further improve the approximation to backprop during
which a local Hebbian rule aligns the feedback and feedforward weights. Our
method achieved the same level of performance as backprop with multi-layer
fully connected SNN on MNIST (98.13%) and the event-based N-MNIST (97.59%)
datasets. We deployed our learning algorithm on Intel's Loihi to train a
1-hidden-layer network for MNIST, and obtained 93.32% test accuracy while
consuming 400 times less energy per training sample than BioGrad on GPU. Our
work shows that optimal learning is feasible in neuromorphic computing, and
further pursuing its biological plausibility can better capture the benefits of
this emerging computing paradigm.

    

### [[2110.14094] Learning-Augmented $k$-means Clustering](http://arxiv.org/abs/2110.14094)


  $k$-means clustering is a well-studied problem due to its wide applicability.
Unfortunately, there exist strong theoretical limits on the performance of any
algorithm for the $k$-means problem on worst-case inputs. To overcome this
barrier, we consider a scenario where "advice" is provided to help perform
clustering. Specifically, we consider the $k$-means problem augmented with a
predictor that, given any point, returns its cluster label in an approximately
optimal clustering up to some, possibly adversarial, error. We present an
algorithm whose performance improves along with the accuracy of the predictor,
even though naïvely following the accurate predictor can still lead to a
high clustering cost. Thus if the predictor is sufficiently accurate, we can
retrieve a close to optimal clustering with nearly optimal runtime, breaking
known computational barriers for algorithms that do not have access to such
advice. We evaluate our algorithms on real datasets and show significant
improvements in the quality of clustering.

    

### [[2110.14096] Towards Robust Bisimulation Metric Learning](http://arxiv.org/abs/2110.14096)


  Learned representations in deep reinforcement learning (DRL) have to extract
task-relevant information from complex observations, balancing between
robustness to distraction and informativeness to the policy. Such stable and
rich representations, often learned via modern function approximation
techniques, can enable practical application of the policy improvement theorem,
even in high-dimensional continuous state-action spaces. Bisimulation metrics
offer one solution to this representation learning problem, by collapsing
functionally similar states together in representation space, which promotes
invariance to noise and distractors. In this work, we generalize value function
approximation bounds for on-policy bisimulation metrics to non-optimal policies
and approximate environment dynamics. Our theoretical results help us identify
embedding pathologies that may occur in practical use. In particular, we find
that these issues stem from an underconstrained dynamics model and an unstable
dependence of the embedding norm on the reward signal in environments with
sparse rewards. Further, we propose a set of practical remedies: (i) a norm
constraint on the representation space, and (ii) an extension of prior
approaches with intrinsic rewards and latent space regularization. Finally, we
provide evidence that the resulting method is not only more robust to sparse
reward functions, but also able to solve challenging continuous control tasks
with observational distractions, where prior methods fail.

    

### [[2110.14098] Provable Lifelong Learning of Representations](http://arxiv.org/abs/2110.14098)


  In lifelong learning, the tasks (or classes) to be learned arrive
sequentially over time in arbitrary order. During training, knowledge from
previous tasks can be captured and transferred to subsequent ones to improve
sample efficiency. We consider the setting where all target tasks can be
represented in the span of a small number of unknown linear or nonlinear
features of the input data. We propose a provable lifelong learning algorithm
that maintains and refines the internal feature representation. We prove that
for any desired accuracy on all tasks, the dimension of the representation
remains close to that of the underlying representation. The resulting sample
complexity improves significantly on existing bounds. In the setting of linear
features, our algorithm is provably efficient and the sample complexity for
input dimension $d$, $m$ tasks with $k$ features up to error $\epsilon$ is
$\tilde{O}(dk^{1.5}/\epsilon+km/\epsilon)$. We also prove a matching lower
bound for any lifelong learning algorithm that uses a single task learner as a
black box. Finally, we complement our analysis with an empirical study.

    

### [[2110.14099] Tight Concentrations and Confidence Sequences from the Regret of Universal Portfolio](http://arxiv.org/abs/2110.14099)


  A classic problem in statistics is the estimation of the expectation of
random variables from samples. This gives rise to the tightly connected
problems of deriving concentration inequalities and confidence sequences, that
is confidence intervals that hold uniformly over time. Jun and Orabona
[COLT'19] have shown how to easily convert the regret guarantee of an online
betting algorithm into a time-uniform concentration inequality. Here, we show
that we can go even further: We show that the regret of a minimax betting
algorithm gives rise to a new implicit empirical time-uniform concentration. In
particular, we use a new data-dependent regret guarantee of the universal
portfolio algorithm. We then show how to invert the new concentration in two
different ways: in an exact way with a numerical algorithm and symbolically in
an approximate way. Finally, we show empirically that our algorithms have
state-of-the-art performance in terms of the width of the confidence sequences
up to a moderately large amount of samples. In particular, our numerically
obtained confidence sequences are never vacuous, even with a single sample.

    

### [[2110.14109] Eigencurve: Optimal Learning Rate Schedule for SGD on Quadratic Objectives with Skewed Hessian Spectrums](http://arxiv.org/abs/2110.14109)


  Learning rate schedulers have been widely adopted in training deep neural
networks. Despite their practical importance, there is a discrepancy between
its practice and its theoretical analysis. For instance, it is not known what
schedules of SGD achieve best convergence, even for simple problems such as
optimizing quadratic objectives. So far, step decay has been one of the
strongest candidates under this setup, which is proved to be nearly optimal
with a $\cO(\log T)$ gap. However, according to our analysis, this gap turns
out to be $\Omega(\log T)$ in a wide range of settings, which throws the
schedule optimality problem into an open question again. Towards answering this
reopened question, in this paper, we propose Eigencurve, the first family of
learning rate schedules that can achieve minimax optimal convergence rates (up
to a constant) for SGD on quadratic objectives when the eigenvalue distribution
of the underlying Hessian matrix is skewed. The condition is quite common in
practice. Experimental results show that Eigencurve can significantly
outperform step decay in image classification tasks on CIFAR-10, especially
when the number of epochs is small. Moreover, the theory inspires two simple
learning rate schedulers for practical applications that can approximate
Eigencurve. For some problems, the optimal shape of the proposed schedulers
resembles that of cosine decay, which sheds light to the success of cosine
decay for such situations. For other situations, the proposed schedulers are
superior to cosine decay.

    

### [[2110.14110] Mining frequency-based sequential trajectory co-clusters](http://arxiv.org/abs/2110.14110)


  Co-clustering is a specific type of clustering that addresses the problem of
finding groups of objects without necessarily considering all attributes. This
technique has shown to have more consistent results in high-dimensional sparse
data than traditional clustering. In trajectory co-clustering, the methods
found in the literature have two main limitations: first, the space and time
dimensions have to be constrained by user-defined thresholds; second, elements
(trajectory points) are clustered ignoring the trajectory sequence, assuming
that the points are independent among them. To address the limitations above,
we propose a new trajectory co-clustering method for mining semantic trajectory
co-clusters. It simultaneously clusters the trajectories and their elements
taking into account the order in which they appear. This new method uses the
element frequency to identify candidate co-clusters. Besides, it uses an
objective cost function that automatically drives the co-clustering process,
avoiding the need for constraining dimensions. We evaluate the proposed
approach using real-world a publicly available dataset. The experimental
results show that our proposal finds frequent and meaningful contiguous
sequences revealing mobility patterns, thereby the most relevant elements.

    

### [[2110.14116] Data-driven decomposition of brain dynamics with principal component analysis in different types of head impacts](http://arxiv.org/abs/2110.14116)


  Strain and strain rate are effective traumatic brain injury predictors.
Kinematics-based models estimating these metrics suffer from significant
different distributions of both kinematics and the injury metrics across head
impact types. To address this, previous studies focus on the kinematics but not
the injury metrics. We have previously shown the kinematic features vary
largely across head impact types, resulting in different patterns of brain
deformation. This study analyzes the spatial distribution of brain deformation
and applies principal component analysis (PCA) to extract the representative
patterns of injury metrics (maximum principal strain (MPS), MPS rate (MPSR) and
MPSXMPSR) in four impact types (simulation, football, mixed martial arts and
car crashes). We apply PCA to decompose the patterns of the injury metrics for
all impacts in each impact type, and investigate the distributions among brain
regions using the first principal component (PC1). Furthermore, we developed a
deep learning head model (DLHM) to predict PC1 and then inverse-transform to
predict for all brain elements. PC1 explained >80% variance on the datasets.
Based on PC1 coefficients, the corpus callosum and midbrain exhibit high
variance on all datasets. We found MPSXMPSR the most sensitive metric on which
the top 5% of severe impacts further deviates from the mean and there is a
higher variance among the severe impacts. Finally, the DLHM reached mean
absolute errors of <0.018 for MPS, <3.7 (1/s) for MPSR and <1.1 (1/s) for
MPSXMPSR, much smaller than the injury thresholds. The brain injury metric in a
dataset can be decomposed into mean components and PC1 with high explained
variance. The brain dynamics decomposition enables better interpretation of the
patterns in brain injury metrics and the sensitivity of brain injury metrics
across impact types. The decomposition also reduces the dimensionality of DLHM.

    

### [[2110.14118] Object-Aware Regularization for Addressing Causal Confusion in Imitation Learning](http://arxiv.org/abs/2110.14118)


  Behavioral cloning has proven to be effective for learning sequential
decision-making policies from expert demonstrations. However, behavioral
cloning often suffers from the causal confusion problem where a policy relies
on the noticeable effect of expert actions due to the strong correlation but
not the cause we desire. This paper presents Object-aware REgularizatiOn
(OREO), a simple technique that regularizes an imitation policy in an
object-aware manner. Our main idea is to encourage a policy to uniformly attend
to all semantic objects, in order to prevent the policy from exploiting
nuisance variables strongly correlated with expert actions. To this end, we
introduce a two-stage approach: (a) we extract semantic objects from images by
utilizing discrete codes from a vector-quantized variational autoencoder, and
(b) we randomly drop the units that share the same discrete code together,
i.e., masking out semantic objects. Our experiments demonstrate that OREO
significantly improves the performance of behavioral cloning, outperforming
various other regularization and causality-based methods on a variety of Atari
environments and a self-driving CARLA environment. We also show that our method
even outperforms inverse reinforcement learning methods trained with a
considerable amount of environment interaction.

    

### [[2110.14120] ScaleCert: Scalable Certified Defense against Adversarial Patches with Sparse Superficial Layers](http://arxiv.org/abs/2110.14120)


  Adversarial patch attacks that craft the pixels in a confined region of the
input images show their powerful attack effectiveness in physical environments
even with noises or deformations. Existing certified defenses towards
adversarial patch attacks work well on small images like MNIST and CIFAR-10
datasets, but achieve very poor certified accuracy on higher-resolution images
like ImageNet. It is urgent to design both robust and effective defenses
against such a practical and harmful attack in industry-level larger images. In
this work, we propose the certified defense methodology that achieves high
provable robustness for high-resolution images and largely improves the
practicality for real adoption of the certified defense. The basic insight of
our work is that the adversarial patch intends to leverage localized
superficial important neurons (SIN) to manipulate the prediction results.
Hence, we leverage the SIN-based DNN compression techniques to significantly
improve the certified accuracy, by reducing the adversarial region searching
overhead and filtering the prediction noises. Our experimental results show
that the certified accuracy is increased from 36.3% (the state-of-the-art
certified detection) to 60.4% on the ImageNet dataset, largely pushing the
certified defenses for practical use.

    

### [[2110.14121] On Computing the Hyperparameter of Extreme Learning Machines: Algorithm and Application to Computational PDEs, and Comparison with Classical and High-Order Finite Elements](http://arxiv.org/abs/2110.14121)


  We consider the use of extreme learning machines (ELM) for computational
partial differential equations (PDE). In ELM the hidden-layer coefficients in
the neural network are assigned to random values generated on $[-R_m,R_m]$ and
fixed, where $R_m$ is a user-provided constant, and the output-layer
coefficients are trained by a linear or nonlinear least squares computation. We
present a method for computing the optimal value of $R_m$ based on the
differential evolution algorithm. The presented method enables us to illuminate
the characteristics of the optimal $R_m$ for two types of ELM configurations:
(i) Single-Rm-ELM, in which a single $R_m$ is used for generating the random
coefficients in all the hidden layers, and (ii) Multi-Rm-ELM, in which multiple
$R_m$ constants are involved with each used for generating the random
coefficients of a different hidden layer. We adopt the optimal $R_m$ from this
method and also incorporate other improvements into the ELM implementation. In
particular, here we compute all the differential operators involving the output
fields of the last hidden layer by a forward-mode auto-differentiation, as
opposed to the reverse-mode auto-differentiation in a previous work. These
improvements significantly reduce the network training time and enhance the ELM
performance. We systematically compare the computational performance of the
current improved ELM with that of the finite element method (FEM), both the
classical second-order FEM and the high-order FEM with Lagrange elements of
higher degrees, for solving a number of linear and nonlinear PDEs. It is shown
that the current improved ELM far outperforms the classical FEM. Its
computational performance is comparable to that of the high-order FEM for
smaller problem sizes, and for larger problem sizes the ELM markedly
outperforms the high-order FEM.

    

### [[2110.14122] Data-Driven Representations for Testing Independence: Modeling, Analysis and Connection with Mutual Information Estimation](http://arxiv.org/abs/2110.14122)


  This work addresses testing the independence of two continuous and
finite-dimensional random variables from the design of a data-driven partition.
The empirical log-likelihood statistic is adopted to approximate the sufficient
statistics of an oracle test against independence (that knows the two
hypotheses). It is shown that approximating the sufficient statistics of the
oracle test offers a learning criterion for designing a data-driven partition
that connects with the problem of mutual information estimation. Applying these
ideas in the context of a data-dependent tree-structured partition (TSP), we
derive conditions on the TSP's parameters to achieve a strongly consistent
distribution-free test of independence over the family of probabilities
equipped with a density. Complementing this result, we present finite-length
results that show our TSP scheme's capacity to detect the scenario of
independence structurally with the data-driven partition as well as new
sampling complexity bounds for this detection. Finally, some experimental
analyses provide evidence regarding our scheme's advantage for testing
independence compared with some strategies that do not use data-driven
representations.

    

### [[2110.14127] Constrained Optimization Involving Nonconvex $\ell_p$ Norms: Optimality Conditions, Algorithm and Convergence](http://arxiv.org/abs/2110.14127)


  This paper investigates the optimality conditions for characterizing the
local minimizers of the constrained optimization problems involving an $\ell_p$
norm ($0<p<1$) of the variables, which may appear in either the objective or
the constraint. This kind of problems have strong applicability to a wide range
of areas since usually the $\ell_p$ norm can promote sparse solutions. However,
the nonsmooth and non-Lipschtiz nature of the $\ell_p$ norm often cause these
problems difficult to analyze and solve. We provide the calculation of the
subgradients of the $\ell_p$ norm and the normal cones of the $\ell_p$ ball.
For both problems, we derive the first-order necessary conditions under various
constraint qualifications. We also derive the sequential optimality conditions
for both problems and study the conditions under which these conditions imply
the first-order necessary conditions. We point out that the sequential
optimality conditions can be easily satisfied for iteratively reweighted
algorithms and show that the global convergence can be easily derived using
sequential optimality conditions.

    

### [[2110.14131] Temporal Knowledge Distillation for On-device Audio Classification](http://arxiv.org/abs/2110.14131)


  Improving the performance of on-device audio classification models remains a
challenge given the computational limits of the mobile environment. Many
studies leverage knowledge distillation to boost predictive performance by
transferring the knowledge from large models to on-device models. However, most
lack the essence of the temporal information which is crucial to audio
classification tasks, or similar architecture is often required. In this paper,
we propose a new knowledge distillation method designed to incorporate the
temporal knowledge embedded in attention weights of large models to on-device
models. Our distillation method is applicable to various types of
architectures, including the non-attention-based architectures such as CNNs or
RNNs, without any architectural change during inference. Through extensive
experiments on both an audio event detection dataset and a noisy keyword
spotting dataset, we show that our proposed method improves the predictive
performance across diverse on-device architectures.

    

### [[2110.14148] Uniform Concentration Bounds toward a Unified Framework for Robust Clustering](http://arxiv.org/abs/2110.14148)


  Recent advances in center-based clustering continue to improve upon the
drawbacks of Lloyd's celebrated $k$-means algorithm over $60$ years after its
introduction. Various methods seek to address poor local minima, sensitivity to
outliers, and data that are not well-suited to Euclidean measures of fit, but
many are supported largely empirically. Moreover, combining such approaches in
a piecemeal manner can result in ad hoc methods, and the limited theoretical
results supporting each individual contribution may no longer hold. Toward
addressing these issues in a principled way, this paper proposes a cohesive
robust framework for center-based clustering under a general class of
dissimilarity measures. In particular, we present a rigorous theoretical
treatment within a Median-of-Means (MoM) estimation framework, showing that it
subsumes several popular $k$-means variants. In addition to unifying existing
methods, we derive uniform concentration bounds that complete their analyses,
and bridge these results to the MoM framework via Dudley's chaining arguments.
Importantly, we neither require any assumptions on the distribution of the
outlying observations nor on the relative number of observations $n$ to
features $p$. We establish strong consistency and an error rate of
$O(n^{-1/2})$ under mild conditions, surpassing the best-known results in the
literature. The methods are empirically validated thoroughly on real and
synthetic datasets.

    

### [[2110.14149] Diversity Matters When Learning From Ensembles](http://arxiv.org/abs/2110.14149)


  Deep ensembles excel in large-scale image classification tasks both in terms
of prediction accuracy and calibration. Despite being simple to train, the
computation and memory cost of deep ensembles limits their practicability.
While some recent works propose to distill an ensemble model into a single
model to reduce such costs, there is still a performance gap between the
ensemble and distilled models. We propose a simple approach for reducing this
gap, i.e., making the distilled performance close to the full ensemble. Our key
assumption is that a distilled model should absorb as much function diversity
inside the ensemble as possible. We first empirically show that the typical
distillation procedure does not effectively transfer such diversity, especially
for complex models that achieve near-zero training error. To fix this, we
propose a perturbation strategy for distillation that reveals diversity by
seeking inputs for which ensemble member outputs disagree. We empirically show
that a model distilled with such perturbed samples indeed exhibits enhanced
diversity, leading to improved performance.

    

### [[2110.14150] Training Wasserstein GANs without gradient penalties](http://arxiv.org/abs/2110.14150)


  We propose a stable method to train Wasserstein generative adversarial
networks. In order to enhance stability, we consider two objective functions
using the $c$-transform based on Kantorovich duality which arises in the theory
of optimal transport. We experimentally show that this algorithm can
effectively enforce the Lipschitz constraint on the discriminator while other
standard methods fail to do so. As a consequence, our method yields an accurate
estimation for the optimal discriminator and also for the Wasserstein distance
between the true distribution and the generated one. Our method requires no
gradient penalties nor corresponding hyperparameter tuning and is
computationally more efficient than other methods. At the same time, it yields
competitive generators of synthetic images based on the MNIST, F-MNIST, and
CIFAR-10 datasets.

    

### [[2110.14153] Differentially Private Federated Bayesian Optimization with Distributed Exploration](http://arxiv.org/abs/2110.14153)


  Bayesian optimization (BO) has recently been extended to the federated
learning (FL) setting by the federated Thompson sampling (FTS) algorithm, which
has promising applications such as federated hyperparameter tuning. However,
FTS is not equipped with a rigorous privacy guarantee which is an important
consideration in FL. Recent works have incorporated differential privacy (DP)
into the training of deep neural networks through a general framework for
adding DP to iterative algorithms. Following this general DP framework, our
work here integrates DP into FTS to preserve user-level privacy. We also
leverage the ability of this general DP framework to handle different parameter
vectors, as well as the technique of local modeling for BO, to further improve
the utility of our algorithm through distributed exploration (DE). The
resulting differentially private FTS with DE (DP-FTS-DE) algorithm is endowed
with theoretical guarantees for both the privacy and utility and is amenable to
interesting theoretical insights about the privacy-utility trade-off. We also
use real-world experiments to show that DP-FTS-DE achieves high utility
(competitive performance) with a strong privacy guarantee (small privacy loss)
and induces a trade-off between privacy and utility.

    

### [[2110.14157] Dream to Explore: Adaptive Simulations for Autonomous Systems](http://arxiv.org/abs/2110.14157)


  One's ability to learn a generative model of the world without supervision
depends on the extent to which one can construct abstract knowledge
representations that generalize across experiences. To this end, capturing an
accurate statistical structure from observational data provides useful
inductive biases that can be transferred to novel environments. Here, we tackle
the problem of learning to control dynamical systems by applying Bayesian
nonparametric methods, which is applied to solve visual servoing tasks. This is
accomplished by first learning a state space representation, then inferring
environmental dynamics and improving the policies through imagined future
trajectories. Bayesian nonparametric models provide automatic model adaptation,
which not only combats underfitting and overfitting, but also allows the
model's unbounded dimension to be both flexible and computationally tractable.
By employing Gaussian processes to discover latent world dynamics, we mitigate
common data efficiency issues observed in reinforcement learning and avoid
introducing explicit model bias by describing the system's dynamics. Our
algorithm jointly learns a world model and policy by optimizing a variational
lower bound of a log-likelihood with respect to the expected free energy
minimization objective function. Finally, we compare the performance of our
model with the state-of-the-art alternatives for continuous control tasks in
simulated environments.

    

### [[2110.14163] Does the Data Induce Capacity Control in Deep Learning?](http://arxiv.org/abs/2110.14163)


  This paper studies how the dataset may be the cause of the anomalous
generalization performance of deep networks. We show that the data correlation
matrix of typical classification datasets has an eigenspectrum where, after a
sharp initial drop, a large number of small eigenvalues are distributed
uniformly over an exponentially large range. This structure is mirrored in a
network trained on this data: we show that the Hessian and the Fisher
Information Matrix (FIM) have eigenvalues that are spread uniformly over
exponentially large ranges. We call such eigenspectra "sloppy" because sets of
weights corresponding to small eigenvalues can be changed by large magnitudes
without affecting the loss. Networks trained on atypical, non-sloppy synthetic
data do not share these traits. We show how this structure in the data can give
to non-vacuous PAC-Bayes generalization bounds analytically; we also construct
data-distribution dependent priors that lead to accurate bounds using numerical
optimization.

    

### [[2110.14168] Training Verifiers to Solve Math Word Problems](http://arxiv.org/abs/2110.14168)


  State-of-the-art language models can match human performance on many tasks,
but they still struggle to robustly perform multi-step mathematical reasoning.
To diagnose the failures of current models and support research, we introduce
GSM8K, a dataset of 8.5K high quality linguistically diverse grade school math
word problems. We find that even the largest transformer models fail to achieve
high test performance, despite the conceptual simplicity of this problem
distribution. To increase performance, we propose training verifiers to judge
the correctness of model completions. At test time, we generate many candidate
solutions and select the one ranked highest by the verifier. We demonstrate
that verification significantly improves performance on GSM8K, and we provide
strong empirical evidence that verification scales more effectively with
increased data than a finetuning baseline.

    

### [[2110.14170] Standing on the Shoulders of Predecessors: Meta-Knowledge Transfer for Knowledge Graphs](http://arxiv.org/abs/2110.14170)


  Knowledge graphs (KGs) have become widespread, and various knowledge graphs
are constructed incessantly to support many in-KG and out-of-KG applications.
During the construction of KGs, although new KGs may contain new entities with
respect to constructed KGs, some entity-independent knowledge can be
transferred from constructed KGs to new KGs. We call such knowledge
meta-knowledge, and refer to the problem of transferring meta-knowledge from
constructed (source) KGs to new (target) KGs to improve the performance of
tasks on target KGs as meta-knowledge transfer for knowledge graphs. However,
there is no available general framework that can tackle meta-knowledge transfer
for both in-KG and out-of-KG tasks uniformly. Therefore, in this paper, we
propose a framework, MorsE, which means conducting Meta-Learning for
Meta-Knowledge Transfer via Knowledge Graph Embedding. MorsE represents the
meta-knowledge via Knowledge Graph Embedding and learns the meta-knowledge by
Meta-Learning. Specifically, MorsE uses an entity initializer and a Graph
Neural Network (GNN) modulator to entity-independently obtain entity embeddings
given a KG and is trained following the meta-learning setting to gain the
ability of effectively obtaining embeddings. Experimental results on
meta-knowledge transfer for both in-KG and out-of-KG tasks show that MorsE is
able to learn and transfer meta-knowledge between KGs effectively, and
outperforms existing state-of-the-art models.

    

### [[2110.14171] Diversity Enhanced Active Learning with Strictly Proper Scoring Rules](http://arxiv.org/abs/2110.14171)


  We study acquisition functions for active learning (AL) for text
classification. The Expected Loss Reduction (ELR) method focuses on a Bayesian
estimate of the reduction in classification error, recently updated with Mean
Objective Cost of Uncertainty (MOCU). We convert the ELR framework to estimate
the increase in (strictly proper) scores like log probability or negative mean
square error, which we call Bayesian Estimate of Mean Proper Scores (BEMPS). We
also prove convergence results borrowing techniques used with MOCU. In order to
allow better experimentation with the new acquisition functions, we develop a
complementary batch AL algorithm, which encourages diversity in the vector of
expected changes in scores for unlabelled data. To allow high performance text
classifiers, we combine ensembling and dynamic validation set construction on
pretrained language models. Extensive experimental evaluation then explores how
these different acquisition functions perform. The results show that the use of
mean square error and log probability with BEMPS yields robust acquisition
functions, which consistently outperform the others tested.

    

### [[2110.14177] Federated Linear Contextual Bandits](http://arxiv.org/abs/2110.14177)


  This paper presents a novel federated linear contextual bandits model, where
individual clients face different $K$-armed stochastic bandits coupled through
common global parameters. By leveraging the geometric structure of the linear
rewards, a collaborative algorithm called Fed-PE is proposed to cope with the
heterogeneity across clients without exchanging local feature vectors or raw
data. Fed-PE relies on a novel multi-client G-optimal design, and achieves
near-optimal regrets for both disjoint and shared parameter cases with
logarithmic communication costs. In addition, a new concept called
collinearly-dependent policies is introduced, based on which a tight minimax
regret lower bound for the disjoint parameter case is derived. Experiments
demonstrate the effectiveness of the proposed algorithms on both synthetic and
real-world datasets.

    

### [[2110.14181] QU-net++: Image Quality Detection Framework for Segmentation of 3D Medical Image Stacks](http://arxiv.org/abs/2110.14181)


  Automated segmentation of pathological regions of interest has been shown to
aid prognosis and follow up treatment. However, accurate pathological
segmentations require high quality of annotated data that can be both cost and
time intensive to generate. In this work, we propose an automated two-step
method that evaluates the quality of medical images from 3D image stacks using
a U-net++ model, such that images that can aid further training of the U-net++
model can be detected based on the disagreement in segmentations produced from
the final two layers. Images thus detected can then be used to further fine
tune the U-net++ model for semantic segmentation. The proposed QU-net++ model
isolates around 10\% of images per 3D stack and can scale across imaging
modalities to segment cysts in OCT images and ground glass opacity in Lung CT
images with Dice cores in the range 0.56-0.72. Thus, the proposed method can be
applied for multi-modal binary segmentation of pathology.

    

### [[2110.14182] Evidential Softmax for Sparse Multimodal Distributions in Deep Generative Models](http://arxiv.org/abs/2110.14182)


  Many applications of generative models rely on the marginalization of their
high-dimensional output probability distributions. Normalization functions that
yield sparse probability distributions can make exact marginalization more
computationally tractable. However, sparse normalization functions usually
require alternative loss functions for training since the log-likelihood is
undefined for sparse probability distributions. Furthermore, many sparse
normalization functions often collapse the multimodality of distributions. In
this work, we present $\textit{ev-softmax}$, a sparse normalization function
that preserves the multimodality of probability distributions. We derive its
properties, including its gradient in closed-form, and introduce a continuous
family of approximations to $\textit{ev-softmax}$ that have full support and
can be trained with probabilistic loss functions such as negative
log-likelihood and Kullback-Leibler divergence. We evaluate our method on a
variety of generative models, including variational autoencoders and
auto-regressive architectures. Our method outperforms existing dense and sparse
normalization techniques in distributional accuracy. We demonstrate that
$\textit{ev-softmax}$ successfully reduces the dimensionality of probability
distributions while maintaining multimodality.

    

### [[2110.14184] OpeNPDN: A Neural-network-based Framework for Power Delivery Network Synthesis](http://arxiv.org/abs/2110.14184)


  Power delivery network (PDN) design is a nontrivial, time-intensive, and
iterative task. Correct PDN design must account for considerations related to
power bumps, currents, blockages, and signal congestion distribution patterns.
This work proposes a machine learning-based methodology that employs a set of
predefined PDN templates. At the floorplan stage, coarse estimates of current,
congestion, macro/blockages, and C4 bump distributions are used to synthesize a
grid for early design. At the placement stage, the grid is incrementally
refined based on more accurate and fine-grained distributions of current and
congestion. At each stage, a convolutional neural network (CNN) selects an
appropriate PDN template for each region on the chip, building a
safe-by-construction PDN that meets IR drop and electromigration (EM)
specifications. The CNN is initially trained using a large
synthetically-created dataset, following which transfer learning is leveraged
to bridge the gap between real-circuit data (with a limited dataset size) and
synthetically-generated data. On average, the optimization of the PDN frees
thousands of routing tracks in congestion-critical regions, when compared to a
globally uniform PDN, while staying within the IR drop and EM limits.

    

### [[2110.14188] RoMA: Robust Model Adaptation for Offline Model-based Optimization](http://arxiv.org/abs/2110.14188)


  We consider the problem of searching an input maximizing a black-box
objective function given a static dataset of input-output queries. A popular
approach to solving this problem is maintaining a proxy model, e.g., a deep
neural network (DNN), that approximates the true objective function. Here, the
main challenge is how to avoid adversarially optimized inputs during the
search, i.e., the inputs where the DNN highly overestimates the true objective
function. To handle the issue, we propose a new framework, coined robust model
adaptation (RoMA), based on gradient-based optimization of inputs over the DNN.
Specifically, it consists of two steps: (a) a pre-training strategy to robustly
train the proxy model and (b) a novel adaptation procedure of the proxy model
to have robust estimates for a specific set of candidate solutions. At a high
level, our scheme utilizes the local smoothness prior to overcome the
brittleness of the DNN. Experiments under various tasks show the effectiveness
of RoMA compared with previous methods, obtaining state-of-the-art results,
e.g., RoMA outperforms all at 4 out of 6 tasks and achieves runner-up results
at the remaining tasks.

    

### [[2110.14189] Robust Contrastive Learning Using Negative Samples with Diminished Semantics](http://arxiv.org/abs/2110.14189)


  Unsupervised learning has recently made exceptional progress because of the
development of more effective contrastive learning methods. However, CNNs are
prone to depend on low-level features that humans deem non-semantic. This
dependency has been conjectured to induce a lack of robustness to image
perturbations or domain shift. In this paper, we show that by generating
carefully designed negative samples, contrastive learning can learn more robust
representations with less dependence on such features. Contrastive learning
utilizes positive pairs that preserve semantic information while perturbing
superficial features in the training images. Similarly, we propose to generate
negative samples in a reversed way, where only the superfluous instead of the
semantic features are preserved. We develop two methods, texture-based and
patch-based augmentations, to generate negative samples. These samples achieve
better generalization, especially under out-of-domain settings. We also analyze
our method and the generated texture-based samples, showing that texture
features are indispensable in classifying particular ImageNet classes and
especially finer classes. We also show that model bias favors texture and shape
features differently under different test settings. Our code, trained models,
and ImageNet-Texture dataset can be found at
this https URL.

    

### [[2110.14197] Encoder-Decoder Networks for Analyzing Thermal and Power Delivery Networks](http://arxiv.org/abs/2110.14197)


  Power delivery network (PDN) analysis and thermal analysis are
computationally expensive tasks that are essential for successful IC design.
Algorithmically, both these analyses have similar computational structure and
complexity as they involve the solution to a partial differential equation of
the same form. This paper converts these analyses into image-to-image and
sequence-to-sequence translation tasks, which allows leveraging a class of
machine learning models with an encoder-decoder-based generative (EDGe)
architecture to address the time-intensive nature of these tasks. For PDN
analysis, we propose two networks: (i) IREDGe: a full-chip static and dynamic
IR drop predictor and (ii) EMEDGe: electromigration (EM) hotspot classifier
based on input power, power grid distribution, and power pad distribution
patterns. For thermal analysis, we propose ThermEDGe, a full-chip static and
dynamic temperature estimator based on input power distribution patterns for
thermal analysis. These networks are transferable across designs synthesized
within the same technology and packing solution. The networks predict on-chip
IR drop, EM hotspot locations, and temperature in milliseconds with negligibly
small errors against commercial tools requiring several hours.

    

### [[2110.14202] Revisit Multimodal Meta-Learning through the Lens of Multi-Task Learning](http://arxiv.org/abs/2110.14202)


  Multimodal meta-learning is a recent problem that extends conventional
few-shot meta-learning by generalizing its setup to diverse multimodal task
distributions. This setup makes a step towards mimicking how humans make use of
a diverse set of prior skills to learn new skills. Previous work has achieved
encouraging performance. In particular, in spite of the diversity of the
multimodal tasks, previous work claims that a single meta-learner trained on a
multimodal distribution can sometimes outperform multiple specialized
meta-learners trained on individual unimodal distributions. The improvement is
attributed to knowledge transfer between different modes of task distributions.
However, there is no deep investigation to verify and understand the knowledge
transfer between multimodal tasks. Our work makes two contributions to
multimodal meta-learning. First, we propose a method to quantify knowledge
transfer between tasks of different modes at a micro-level. Our quantitative,
task-level analysis is inspired by the recent transference idea from multi-task
learning. Second, inspired by hard parameter sharing in multi-task learning and
a new interpretation of related work, we propose a new multimodal meta-learner
that outperforms existing work by considerable margins. While the major focus
is on multimodal meta-learning, our work also attempts to shed light on task
interaction in conventional meta-learning. The code for this project is
available at this https URL.

    

### [[2110.14203] Syllabic Quantity Patterns as Rhythmic Features for Latin Authorship Attribution](http://arxiv.org/abs/2110.14203)


  It is well known that, within the Latin production of written text, peculiar
metric schemes were followed not only in poetic compositions, but also in many
prose works. Such metric patterns were based on so-called syllabic quantity,
i.e., on the length of the involved syllables, and there is substantial
evidence suggesting that certain authors had a preference for certain metric
patterns over others. In this research we investigate the possibility to employ
syllabic quantity as a base for deriving rhythmic features for the task of
computational authorship attribution of Latin prose texts. We test the impact
of these features on the authorship attribution task when combined with other
topic-agnostic features. Our experiments, carried out on three different
datasets, using two different machine learning methods, show that rhythmic
features based on syllabic quantity are beneficial in discriminating among
Latin prose authors.

    

### [[2110.14205] FedPrune: Towards Inclusive Federated Learning](http://arxiv.org/abs/2110.14205)


  Federated learning (FL) is a distributed learning technique that trains a
shared model over distributed data in a privacy-preserving manner.
Unfortunately, FL's performance degrades when there is (i) variability in
client characteristics in terms of computational and memory resources (system
heterogeneity) and (ii) non-IID data distribution across clients (statistical
heterogeneity). For example, slow clients get dropped in FL schemes, such as
Federated Averaging (FedAvg), which not only limits overall learning but also
biases results towards fast clients. We propose FedPrune; a system that tackles
this challenge by pruning the global model for slow clients based on their
device characteristics. By doing so, slow clients can train a small model
quickly and participate in FL which increases test accuracy as well as
fairness. By using insights from Central Limit Theorem, FedPrune incorporates a
new aggregation technique that achieves robust performance over non-IID data.
Experimental evaluation shows that Fed- Prune provides robust convergence and
better fairness compared to Federated Averaging.

    

### [[2110.14215] Beyond Classification: Knowledge Distillation using Multi-Object Impressions](http://arxiv.org/abs/2110.14215)


  Knowledge Distillation (KD) utilizes training data as a transfer set to
transfer knowledge from a complex network (Teacher) to a smaller network
(Student). Several works have recently identified many scenarios where the
training data may not be available due to data privacy or sensitivity concerns
and have proposed solutions under this restrictive constraint for the
classification task. Unlike existing works, we, for the first time, solve a
much more challenging problem, i.e., "KD for object detection with zero
knowledge about the training data and its statistics". Our proposed approach
prepares pseudo-targets and synthesizes corresponding samples (termed as
"Multi-Object Impressions"), using only the pretrained Faster RCNN Teacher
network. We use this pseudo-dataset as a transfer set to conduct zero-shot KD
for object detection. We demonstrate the efficacy of our proposed method
through several ablations and extensive experiments on benchmark datasets like
KITTI, Pascal and COCO. Our approach with no training samples, achieves a
respectable mAP of 64.2% and 55.5% on the student with same and half capacity
while performing distillation from a Resnet-18 Teacher of 73.3% mAP on KITTI.

    

### [[2110.14216] What Do We Mean by Generalization in Federated Learning?](http://arxiv.org/abs/2110.14216)


  Federated learning data is drawn from a distribution of distributions:
clients are drawn from a meta-distribution, and their data are drawn from local
data distributions. Thus generalization studies in federated learning should
separate performance gaps from unseen client data (out-of-sample gap) from
performance gaps from unseen client distributions (participation gap). In this
work, we propose a framework for disentangling these performance gaps. Using
this framework, we observe and explain differences in behavior across natural
and synthetic federated datasets, indicating that dataset synthesis strategy
can be important for realistic simulations of generalization in federated
learning. We propose a semantic synthesis strategy that enables realistic
simulation without naturally-partitioned data. Informed by our findings, we
call out community suggestions for future federated learning works.

    

### [[2110.14221] Learning Diverse Policies in MOBA Games via Macro-Goals](http://arxiv.org/abs/2110.14221)


  Recently, many researchers have made successful progress in building the AI
systems for MOBA-game-playing with deep reinforcement learning, such as on Dota
2 and Honor of Kings. Even though these AI systems have achieved or even
exceeded human-level performance, they still suffer from the lack of policy
diversity. In this paper, we propose a novel Macro-Goals Guided framework,
called MGG, to learn diverse policies in MOBA games. MGG abstracts strategies
as macro-goals from human demonstrations and trains a Meta-Controller to
predict these macro-goals. To enhance policy diversity, MGG samples macro-goals
from the Meta-Controller prediction and guides the training process towards
these goals. Experimental results on the typical MOBA game Honor of Kings
demonstrate that MGG can execute diverse policies in different matches and
lineups, and also outperform the state-of-the-art methods over 102 heroes.

    

### [[2110.14222] Sample Selection for Fair and Robust Training](http://arxiv.org/abs/2110.14222)


  Fairness and robustness are critical elements of Trustworthy AI that need to
be addressed together. Fairness is about learning an unbiased model while
robustness is about learning from corrupted data, and it is known that
addressing only one of them may have an adverse affect on the other. In this
work, we propose a sample selection-based algorithm for fair and robust
training. To this end, we formulate a combinatorial optimization problem for
the unbiased selection of samples in the presence of data corruption. Observing
that solving this optimization problem is strongly NP-hard, we propose a greedy
algorithm that is efficient and effective in practice. Experiments show that
our algorithm obtains fairness and robustness that are better than or
comparable to the state-of-the-art technique, both on synthetic and benchmark
real datasets. Moreover, unlike other fair and robust training baselines, our
algorithm can be used by only modifying the sampling step in batch selection
without changing the training algorithm or leveraging additional clean data.

    

### [[2110.14237] Learning Graph Cellular Automata](http://arxiv.org/abs/2110.14237)


  Cellular automata (CA) are a class of computational models that exhibit rich
dynamics emerging from the local interaction of cells arranged in a regular
lattice. In this work we focus on a generalised version of typical CA, called
graph cellular automata (GCA), in which the lattice structure is replaced by an
arbitrary graph. In particular, we extend previous work that used convolutional
neural networks to learn the transition rule of conventional CA and we use
graph neural networks to learn a variety of transition rules for GCA. First, we
present a general-purpose architecture for learning GCA, and we show that it
can represent any arbitrary GCA with finite and discrete state space. Then, we
test our approach on three different tasks: 1) learning the transition rule of
a GCA on a Voronoi tessellation; 2) imitating the behaviour of a group of
flocking agents; 3) learning a rule that converges to a desired target state.

    

### [[2110.14241] Dynamic population-based meta-learning for multi-agent communication with natural language](http://arxiv.org/abs/2110.14241)


  In this work, our goal is to train agents that can coordinate with seen,
unseen as well as human partners in a multi-agent communication environment
involving natural language. Previous work using a single set of agents has
shown great progress in generalizing to known partners, however it struggles
when coordinating with unfamiliar agents. To mitigate that, recent work
explored the use of population-based approaches, where multiple agents interact
with each other with the goal of learning more generic protocols. These
methods, while able to result in good coordination between unseen partners,
still only achieve so in cases of simple languages, thus failing to adapt to
human partners using natural language. We attribute this to the use of static
populations and instead propose a dynamic population-based meta-learning
approach that builds such a population in an iterative manner. We perform a
holistic evaluation of our method on two different referential games, and show
that our agents outperform all prior work when communicating with seen partners
and humans. Furthermore, we analyze the natural language generation skills of
our agents, where we find that our agents also outperform strong baselines.
Finally, we test the robustness of our agents when communicating with
out-of-population agents and carefully test the importance of each component of
our method through ablation studies.

    

### [[2110.14242] Tight FPT Approximation for Constrained k-Center and k-Supplier](http://arxiv.org/abs/2110.14242)


  In this work, we study a range of constrained versions of the $k$-supplier
and $k$-center problems such as: capacitated, fault-tolerant, fair, etc. These
problems fall under a broad framework of constrained clustering. A unified
framework for constrained clustering was proposed by Ding and Xu [SODA 2015] in
context of the $k$-median and $k$-means objectives. In this work, we extend
this framework to the $k$-supplier and $k$-center objectives. This unified
framework allows us to obtain results simultaneously for the following
constrained versions of the $k$-supplier problem: $r$-gather, $r$-capacity,
balanced, chromatic, fault-tolerant, strongly private, $\ell$-diversity, and
fair $k$-supplier problems, with and without outliers. We obtain the following
results: We give $3$ and $2$ approximation algorithms for the constrained
$k$-supplier and $k$-center problems, respectively, with $\mathsf{FPT}$ running
time $k^{O(k)} \cdot n^{O(1)}$, where $n = |C \cup L|$. Moreover, these
approximation guarantees are tight; that is, for any constant $\epsilon>0$, no
algorithm can achieve $(3-\epsilon)$ and $(2-\epsilon)$ approximation
guarantees for the constrained $k$-supplier and $k$-center problems in
$\mathsf{FPT}$ time, assuming $\mathsf{FPT} \neq \mathsf{W}[2]$. Furthermore,
we study these constrained problems in outlier setting. Our algorithm gives $3$
and $2$ approximation guarantees for the constrained outlier $k$-supplier and
$k$-center problems, respectively, with $\mathsf{FPT}$ running time
$(k+m)^{O(k)} \cdot n^{O(1)}$, where $n = |C \cup L|$ and $m$ is the number of
outliers.

    

### [[2110.14243] Online Selective Classification with Limited Feedback](http://arxiv.org/abs/2110.14243)


  Motivated by applications to resource-limited and safety-critical domains, we
study selective classification in the online learning model, wherein a
predictor may abstain from classifying an instance. For example, this may model
an adaptive decision to invoke more resources on this instance. Two salient
aspects of the setting we consider are that the data may be non-realisable, due
to which abstention may be a valid long-term action, and that feedback is only
received when the learner abstains, which models the fact that reliable labels
are only available when the resource intensive processing is invoked.
Within this framework, we explore strategies that make few mistakes, while
not abstaining too many times more than the best-in-hindsight error-free
classifier from a given class. That is, the one that makes no mistakes, while
abstaining the fewest number of times. We construct simple versioning-based
schemes for any $\mu \in (0,1],$ that make most $T^\mu$ mistakes while
incurring \smash{$\tilde{O}(T^{1-\mu})$} excess abstention against adaptive
adversaries. We further show that this dependence on $T$ is tight, and provide
illustrative experiments on realistic datasets.

    

### [[2110.14248] Learning Domain Invariant Representations in Goal-conditioned Block MDPs](http://arxiv.org/abs/2110.14248)


  Deep Reinforcement Learning (RL) is successful in solving many complex Markov
Decision Processes (MDPs) problems. However, agents often face unanticipated
environmental changes after deployment in the real world. These changes are
often spurious and unrelated to the underlying problem, such as background
shifts for visual input agents. Unfortunately, deep RL policies are usually
sensitive to these changes and fail to act robustly against them. This
resembles the problem of domain generalization in supervised learning. In this
work, we study this problem for goal-conditioned RL agents. We propose a
theoretical framework in the Block MDP setting that characterizes the
generalizability of goal-conditioned policies to new environments. Under this
framework, we develop a practical method PA-SkewFit that enhances domain
generalization. The empirical evaluation shows that our goal-conditioned RL
agent can perform well in various unseen test environments, improving by 50%
over baselines.

    

### [[2110.14254] Multilayer Lookahead: a Nested Version of Lookahead](http://arxiv.org/abs/2110.14254)


  In recent years, SGD and its variants have become the standard tool to train
Deep Neural Networks. In this paper, we focus on the recently proposed variant
Lookahead, which improves upon SGD in a wide range of applications. Following
this success, we study an extension of this algorithm, the \emph{Multilayer
Lookahead} optimizer, which recursively wraps Lookahead around itself. We prove
the convergence of Multilayer Lookahead with two layers to a stationary point
of smooth non-convex functions with $O(\frac{1}{\sqrt{T}})$ rate. We also
justify the improved generalization of both Lookahead over SGD, and of
Multilayer Lookahead over Lookahead, by showing how they amplify the implicit
regularization effect of SGD. We empirically verify our results and show that
Multilayer Lookahead outperforms Lookahead on CIFAR-10 and CIFAR-100
classification tasks, and on GANs training on the MNIST dataset.

    

### [[2110.14256] Cascaded Classifier for Pareto-Optimal Accuracy-Cost Trade-Off Using off-the-Shelf ANNs](http://arxiv.org/abs/2110.14256)


  Machine-learning classifiers provide high quality of service in
classification tasks. Research now targets cost reduction measured in terms of
average processing time or energy per solution. Revisiting the concept of
cascaded classifiers, we present a first of its kind analysis of optimal
pass-on criteria between the classifier stages. Based on this analysis, we
derive a methodology to maximize accuracy and efficiency of cascaded
classifiers. On the one hand, our methodology allows cost reduction of 1.32x
while preserving reference classifier's accuracy. On the other hand, it allows
to scale cost over two orders while gracefully degrading accuracy. Thereby, the
final classifier stage sets the top accuracy. Hence, the multi-stage
realization can be employed to optimize any state-of-the-art classifier.

    

### [[2110.14266] SQALER: Scaling Question Answering by Decoupling Multi-Hop and Logical Reasoning](http://arxiv.org/abs/2110.14266)


  State-of-the-art approaches to reasoning and question answering over
knowledge graphs (KGs) usually scale with the number of edges and can only be
applied effectively on small instance-dependent subgraphs. In this paper, we
address this issue by showing that multi-hop and more complex logical reasoning
can be accomplished separately without losing expressive power. Motivated by
this insight, we propose an approach to multi-hop reasoning that scales
linearly with the number of relation types in the graph, which is usually
significantly smaller than the number of edges or nodes. This produces a set of
candidate solutions that can be provably refined to recover the solution to the
original problem. Our experiments on knowledge-based question answering show
that our approach solves the multi-hop MetaQA dataset, achieves a new
state-of-the-art on the more challenging WebQuestionsSP, is orders of magnitude
more scalable than competitive approaches, and can achieve compositional
generalization out of the training distribution.

    

### [[2110.14270] Counterfactual Shapley Additive Explanations](http://arxiv.org/abs/2110.14270)


  Feature attributions are a common paradigm for model explanations due to
their simplicity in assigning a single numeric score for each input feature to
a model. In the actionable recourse setting, wherein the goal of the
explanations is to improve outcomes for model consumers, it is often unclear
how feature attributions should be correctly used. With this work, we aim to
strengthen and clarify the link between actionable recourse and feature
attributions. Concretely, we propose a variant of SHAP, CoSHAP, that uses
counterfactual generation techniques to produce a background dataset for use
within the marginal (a.k.a. interventional) Shapley value framework. We
motivate the need within the actionable recourse setting for careful
consideration of background datasets when using Shapley values for feature
attributions, alongside the requirement for monotonicity, with numerous
synthetic examples. Moreover, we demonstrate the efficacy of CoSHAP by
proposing and justifying a quantitative score for feature attributions,
counterfactual-ability, showing that as measured by this metric, CoSHAP is
superior to existing methods when evaluated on public datasets using monotone
tree ensembles.

    

### [[2110.14284] MIRA: Multihop Relation Prediction in Temporal Knowledge Graphs](http://arxiv.org/abs/2110.14284)


  In knowledge graph reasoning, we observe a trend to analyze temporal data
evolving over time. The additional temporal dimension is attached to facts in a
knowledge base resulting in quadruples between entities such as (Nintendo,
released, Super Mario, Sep-13-1985), where the relation between two entities is
associated to a specific time interval or point in time. Multi-hop reasoning on
inferred subgraphs connecting entities within a knowledge graph can be
formulated as a reinforcement learning task where the agent sequentially
performs inference upon the explored subgraph. The task in this work is to
infer the predicate between a subject and an object entity, i.e., (subject, ?,
object, time), being valid at a certain timestamp or time interval. Given query
entities, our agent starts to gather temporal relevant information about the
neighborhood of the subject and object. The encoding of information about the
explored graph structures is referred to as fingerprints. Subsequently, we use
the two fingerprints as input to a Q-Network. Our agent decides sequentially
which relational type needs to be explored next expanding the local subgraphs
of the query entities in order to find promising paths between them. The
evaluation shows that the proposed method not only yields results being in line
with state-of-the-art embedding algorithms for temporal Knowledge Graphs (tKG),
but we also gain information about the relevant structures between subjects and
objects.

    

### [[2110.14286] TopicNet: Semantic Graph-Guided Topic Discovery](http://arxiv.org/abs/2110.14286)


  Existing deep hierarchical topic models are able to extract semantically
meaningful topics from a text corpus in an unsupervised manner and
automatically organize them into a topic hierarchy. However, it is unclear how
to incorporate prior beliefs such as knowledge graph to guide the learning of
the topic hierarchy. To address this issue, we introduce TopicNet as a deep
hierarchical topic model that can inject prior structural knowledge as an
inductive bias to influence learning. TopicNet represents each topic as a
Gaussian-distributed embedding vector, projects the topics of all layers into a
shared embedding space, and explores both the symmetric and asymmetric
similarities between Gaussian embedding vectors to incorporate prior semantic
hierarchies. With an auto-encoding variational inference network, the model
parameters are optimized by minimizing the evidence lower bound and a
regularization term via stochastic gradient descent. Experiments on widely used
benchmarks show that TopicNet outperforms related deep topic models on
discovering deeper interpretable topics and mining better
document~representations.

    

### [[2110.14295] A Subgame Perfect Equilibrium Reinforcement Learning Approach to Time-inconsistent Problems](http://arxiv.org/abs/2110.14295)


  In this paper, we establish a subgame perfect equilibrium reinforcement
learning (SPERL) framework for time-inconsistent (TIC) problems. In the context
of RL, TIC problems are known to face two main challenges: the non-existence of
natural recursive relationships between value functions at different time
points and the violation of Bellman's principle of optimality that raises
questions on the applicability of standard policy iteration algorithms for
unprovable policy improvement theorems. We adapt an extended dynamic
programming theory and propose a new class of algorithms, called backward
policy iteration (BPI), that solves SPERL and addresses both challenges. To
demonstrate the practical usage of BPI as a training framework, we adapt
standard RL simulation methods and derive two BPI-based training algorithms. We
examine our derived training frameworks on a mean-variance portfolio selection
problem and evaluate some performance metrics including convergence and model
identifiability.

    

### [[2110.14296] Learning Stable Deep Dynamics Models for Partially Observed or Delayed Dynamical Systems](http://arxiv.org/abs/2110.14296)


  Learning how complex dynamical systems evolve over time is a key challenge in
system identification. For safety critical systems, it is often crucial that
the learned model is guaranteed to converge to some equilibrium point. To this
end, neural ODEs regularized with neural Lyapunov functions are a promising
approach when states are fully observed. For practical applications however,
partial observations are the norm. As we will demonstrate, initialization of
unobserved augmented states can become a key problem for neural ODEs. To
alleviate this issue, we propose to augment the system's state with its
history. Inspired by state augmentation in discrete-time systems, we thus
obtain neural delay differential equations. Based on classical time delay
stability analysis, we then show how to ensure stability of the learned models,
and theoretically analyze our approach. Our experiments demonstrate its
applicability to stable system identification of partially observed systems and
learning a stabilizing feedback policy in delayed feedback control.

    

### [[2110.14297] Revisiting Sanity Checks for Saliency Maps](http://arxiv.org/abs/2110.14297)


  Saliency methods are a popular approach for model debugging and
explainability. However, in the absence of ground-truth data for what the
correct maps should be, evaluating and comparing different approaches remains a
long-standing challenge. The sanity checks methodology of Adebayo et al
[Neurips 2018] has sought to address this challenge. They argue that some
popular saliency methods should not be used for explainability purposes since
the maps they produce are not sensitive to the underlying model that is to be
explained. Through a causal re-framing of their objective, we argue that their
empirical evaluation does not fully establish these conclusions, due to a form
of confounding introduced by the tasks they evaluate on. Through various
experiments on simple custom tasks we demonstrate that some of their
conclusions may indeed be artifacts of the tasks more than a criticism of the
saliency methods themselves. More broadly, our work challenges the utility of
the sanity check methodology, and further highlights that saliency map
evaluation beyond ad-hoc visual examination remains a fundamental challenge.

    

### [[2110.14300] Multi-Agent Reinforcement Learning for Active Voltage Control on Power Distribution Networks](http://arxiv.org/abs/2110.14300)


  This paper presents a problem in power networks that creates an exciting and
yet challenging real-world scenario for application of multi-agent
reinforcement learning (MARL). The emerging trend of decarbonisation is placing
excessive stress on power distribution networks. Active voltage control is seen
as a promising solution to relieve power congestion and improve voltage quality
without extra hardware investment, taking advantage of the controllable
apparatuses in the network, such as roof-top photovoltaics (PVs) and static var
compensators (SVCs). These controllable apparatuses appear in a vast number and
are distributed in a wide geographic area, making MARL a natural candidate.
This paper formulates the active voltage control problem in the framework of
Dec-POMDP and establishes an open-source environment. It aims to bridge the gap
between the power community and the MARL community and be a drive force towards
real-world applications of MARL algorithms. Finally, we analyse the special
characteristics of the active voltage control problems that cause challenges
for state-of-the-art MARL approaches, and summarise the potential directions.

    

### [[2110.14317] Ask "Who", Not "What": Bitcoin Volatility Forecasting with Twitter Data](http://arxiv.org/abs/2110.14317)


  Understanding the variations in trading price (volatility), and its response
to external information is a well-studied topic in finance. In this study, we
focus on volatility predictions for a relatively new asset class of
cryptocurrencies (in particular, Bitcoin) using deep learning representations
of public social media data from Twitter. For the field work, we extracted
semantic information and user interaction statistics from over 30 million
Bitcoin-related tweets, in conjunction with 15-minute intraday price data over
a 144-day horizon. Using this data, we built several deep learning
architectures that utilized a combination of the gathered information. For all
architectures, we conducted ablation studies to assess the influence of each
component and feature set in our model. We found statistical evidences for the
hypotheses that: (i) temporal convolutional networks perform significantly
better than both autoregressive and other deep learning-based models in the
literature, and (ii) the tweet author meta-information, even detached from the
tweet itself, is a better predictor than the semantic content and tweet volume
statistics.

    

### [[2110.14322] Node-wise Localization of Graph Neural Networks](http://arxiv.org/abs/2110.14322)


  Graph neural networks (GNNs) emerge as a powerful family of representation
learning models on graphs. To derive node representations, they utilize a
global model that recursively aggregates information from the neighboring
nodes. However, different nodes reside at different parts of the graph in
different local contexts, making their distributions vary across the graph.
Ideally, how a node receives its neighborhood information should be a function
of its local context, to diverge from the global GNN model shared by all nodes.
To utilize node locality without overfitting, we propose a node-wise
localization of GNNs by accounting for both global and local aspects of the
graph. Globally, all nodes on the graph depend on an underlying global GNN to
encode the general patterns across the graph; locally, each node is localized
into a unique model as a function of the global model and its local context.
Finally, we conduct extensive experiments on four benchmark graphs, and
consistently obtain promising performance surpassing the state-of-the-art GNNs.

    

### [[2110.14329] Feature selection revisited in the single-cell era](http://arxiv.org/abs/2110.14329)


  Feature selection techniques are essential for high-dimensional data
analysis. In the last two decades, their popularity has been fuelled by the
increasing availability of high-throughput biomolecular data where
high-dimensionality is a common data property. Recent advances in
biotechnologies enable global profiling of various molecular and cellular
features at single-cell resolution, resulting in large-scale datasets with
increased complexity. These technological developments have led to a resurgence
in feature selection research and application in the single-cell field. Here,
we revisit feature selection techniques and summarise recent developments. We
review their versatile application to a range of single-cell data types
including those generated from traditional cytometry and imaging technologies
and the latest array of single-cell omics technologies. We highlight some of
the challenges and future directions on which feature selection could have a
significant impact. Finally, we consider the scalability and make general
recommendations on the utility of each type of feature selection method. We
hope this review serves as a reference point to stimulate future research and
application of feature selection in the single-cell era.

    

### [[2110.14331] GACAN: Graph Attention-Convolution-Attention Networks for Traffic Forecasting Based on Multi-granularity Time Series](http://arxiv.org/abs/2110.14331)


  Traffic forecasting is an integral part of intelligent transportation systems
(ITS). Achieving a high prediction accuracy is a challenging task due to a high
level of dynamics and complex spatial-temporal dependency of road networks. For
this task, we propose Graph Attention-Convolution-Attention Networks (GACAN).
The model uses a novel Att-Conv-Att (ACA) block which contains two graph
attention layers and one spectral-based GCN layer sandwiched in between. The
graph attention layers are meant to capture temporal features while the
spectral-based GCN layer is meant to capture spatial features. The main novelty
of the model is the integration of time series of four different time
granularities: the original time series, together with hourly, daily, and
weekly time series. Unlike previous work that used multi-granularity time
series by handling every time series separately, GACAN combines the outcome of
processing all time series after each graph attention layer. Thus, the effects
of different time granularities are integrated throughout the model. We perform
a series of experiments on three real-world datasets. The experimental results
verify the advantage of using multi-granularity time series and that the
proposed GACAN model outperforms the state-of-the-art baselines.

    

### [[2110.14341] Active-LATHE: An Active Learning Algorithm for Boosting the Error Exponent for Learning Homogeneous Ising Trees](http://arxiv.org/abs/2110.14341)


  The Chow-Liu algorithm (IEEE Trans.~Inform.~Theory, 1968) has been a mainstay
for the learning of tree-structured graphical models from i.i.d.\ sampled data
vectors. Its theoretical properties have been well-studied and are
well-understood. In this paper, we focus on the class of trees that are
arguably even more fundamental, namely {\em homogeneous} trees in which each
pair of nodes that forms an edge has the same correlation $\rho$. We ask
whether we are able to further reduce the error probability of learning the
structure of the homogeneous tree model when {\em active learning} or {\em
active sampling of nodes or variables} is allowed. Our figure of merit is the
{\em error exponent}, which quantifies the exponential rate of decay of the
error probability with an increasing number of data samples. At first sight, an
improvement in the error exponent seems impossible, as all the edges are
statistically identical. We design and analyze an algorithm Active Learning
Algorithm for Trees with Homogeneous Edge (Active-LATHE), which surprisingly
boosts the error exponent by at least 40\% when $\rho$ is at least $0.8$. For
all other values of $\rho$, we also observe commensurate, but more modest,
improvements in the error exponent. Our analysis hinges on judiciously
exploiting the minute but detectable statistical variation of the samples to
allocate more data to parts of the graph in which we are less confident of
being correct.

    

### [[2110.14343] Comprehensive learning particle swarm optimization enabled modeling framework for multi-step-ahead influenza prediction](http://arxiv.org/abs/2110.14343)


  Epidemics of influenza are major public health concerns. Since influenza
prediction always relies on the weekly clinical or laboratory surveillance
data, typically the weekly Influenza-like illness (ILI) rate series, accurate
multi-step-ahead influenza predictions using ILI series is of great importance,
especially, to the potential coming influenza outbreaks. This study proposes
Comprehensive Learning Particle Swarm Optimization based Machine Learning
(CLPSO-ML) framework incorporating support vector regression (SVR) and
multilayer perceptron (MLP) for multi-step-ahead influenza prediction. A
comprehensive examination and comparison of the performance and potential of
three commonly used multi-step-ahead prediction modeling strategies, including
iterated strategy, direct strategy and multiple-input multiple-output (MIMO)
strategy, was conducted using the weekly ILI rate series from both the Southern
and Northern China. The results show that: (1) The MIMO strategy achieves the
best multi-step-ahead prediction, and is potentially more adaptive for longer
horizon; (2) The iterated strategy demonstrates special potentials for deriving
the least time difference between the occurrence of the predicted peak value
and the true peak value of an influenza outbreak; (3) For ILI in the Northern
China, SVR model implemented with MIMO strategy performs best, and SVR with
iterated strategy also shows remarkable performance especially during outbreak
periods; while for ILI in the Southern China, both SVR and MLP models with MIMO
strategy have competitive prediction performance

    

### [[2110.14346] A Scalable Inference Method For Large Dynamic Economic Systems](http://arxiv.org/abs/2110.14346)


  The nature of available economic data has changed fundamentally in the last
decade due to the economy's digitisation. With the prevalence of often black
box data-driven machine learning methods, there is a necessity to develop
interpretable machine learning methods that can conduct econometric inference,
helping policymakers leverage the new nature of economic data. We therefore
present a novel Variational Bayesian Inference approach to incorporate a
time-varying parameter auto-regressive model which is scalable for big data.
Our model is applied to a large blockchain dataset containing prices,
transactions of individual actors, analyzing transactional flows and price
movements on a very granular level. The model is extendable to any dataset
which can be modelled as a dynamical system. We further improve the simple
state-space modelling by introducing non-linearities in the forward model with
the help of machine learning architectures.

    

### [[2110.14350] Enhancing Reinforcement Learning with discrete interfaces to learn the Dyck Language](http://arxiv.org/abs/2110.14350)


  Even though most interfaces in the real world are discrete, no efficient way
exists to train neural networks to make use of them, yet. We enhance an
Interaction Network (a Reinforcement Learning architecture) with discrete
interfaces and train it on the generalized Dyck language. This task requires an
understanding of hierarchical structures to solve, and has long proven
difficult for neural networks. We provide the first solution based on learning
to use discrete data structures. We encountered unexpected anomalous behavior
during training, and utilized pre-training based on execution traces to
overcome them. The resulting model is very small and fast, and generalizes to
sequences that are an entire order of magnitude longer than the training data.

    

### [[2110.14354] MixSeq: Connecting Macroscopic Time Series Forecasting with Microscopic Time Series Data](http://arxiv.org/abs/2110.14354)


  Time series forecasting is widely used in business intelligence, e.g.,
forecast stock market price, sales, and help the analysis of data trend. Most
time series of interest are macroscopic time series that are aggregated from
microscopic data. However, instead of directly modeling the macroscopic time
series, rare literature studied the forecasting of macroscopic time series by
leveraging data on the microscopic level. In this paper, we assume that the
microscopic time series follow some unknown mixture probabilistic
distributions. We theoretically show that as we identify the ground truth
latent mixture components, the estimation of time series from each component
could be improved because of lower variance, thus benefitting the estimation of
macroscopic time series as well. Inspired by the power of Seq2seq and its
variants on the modeling of time series data, we propose Mixture of Seq2seq
(MixSeq), an end2end mixture model to cluster microscopic time series, where
all the components come from a family of Seq2seq models parameterized by
different parameters. Extensive experiments on both synthetic and real-world
data show the superiority of our approach.

    

### [[2110.14355] Transfer learning with causal counterfactual reasoning in Decision Transformers](http://arxiv.org/abs/2110.14355)


  The ability to adapt to changes in environmental contingencies is an
important challenge in reinforcement learning. Indeed, transferring previously
acquired knowledge to environments with unseen structural properties can
greatly enhance the flexibility and efficiency by which novel optimal policies
may be constructed. In this work, we study the problem of transfer learning
under changes in the environment dynamics. In this study, we apply causal
reasoning in the offline reinforcement learning setting to transfer a learned
policy to new environments. Specifically, we use the Decision Transformer (DT)
architecture to distill a new policy on the new environment. The DT is trained
on data collected by performing policy rollouts on factual and counterfactual
simulations from the source environment. We show that this mechanism can
bootstrap a successful policy on the target environment while retaining most of
the reward.

    

### [[2110.14363] VQ-GNN: A Universal Framework to Scale up Graph Neural Networks using Vector Quantization](http://arxiv.org/abs/2110.14363)


  Most state-of-the-art Graph Neural Networks (GNNs) can be defined as a form
of graph convolution which can be realized by message passing between direct
neighbors or beyond. To scale such GNNs to large graphs, various neighbor-,
layer-, or subgraph-sampling techniques are proposed to alleviate the "neighbor
explosion" problem by considering only a small subset of messages passed to the
nodes in a mini-batch. However, sampling-based methods are difficult to apply
to GNNs that utilize many-hops-away or global context each layer, show unstable
performance for different tasks and datasets, and do not speed up model
inference. We propose a principled and fundamentally different approach,
VQ-GNN, a universal framework to scale up any convolution-based GNNs using
Vector Quantization (VQ) without compromising the performance. In contrast to
sampling-based techniques, our approach can effectively preserve all the
messages passed to a mini-batch of nodes by learning and updating a small
number of quantized reference vectors of global node representations, using VQ
within each GNN layer. Our framework avoids the "neighbor explosion" problem of
GNNs using quantized representations combined with a low-rank version of the
graph convolution matrix. We show that such a compact low-rank version of the
gigantic convolution matrix is sufficient both theoretically and
experimentally. In company with VQ, we design a novel approximated message
passing algorithm and a nontrivial back-propagation rule for our framework.
Experiments on various types of GNN backbones demonstrate the scalability and
competitive performance of our framework on large-graph node classification and
link prediction benchmarks.

    

### [[2110.14373] Neural-PIL: Neural Pre-Integrated Lighting for Reflectance Decomposition](http://arxiv.org/abs/2110.14373)


  Decomposing a scene into its shape, reflectance and illumination is a
fundamental problem in computer vision and graphics. Neural approaches such as
NeRF have achieved remarkable success in view synthesis, but do not explicitly
perform decomposition and instead operate exclusively on radiance (the product
of reflectance and illumination). Extensions to NeRF, such as NeRD, can perform
decomposition but struggle to accurately recover detailed illumination, thereby
significantly limiting realism. We propose a novel reflectance decomposition
network that can estimate shape, BRDF, and per-image illumination given a set
of object images captured under varying illumination. Our key technique is a
novel illumination integration network called Neural-PIL that replaces a costly
illumination integral operation in the rendering with a simple network query.
In addition, we also learn deep low-dimensional priors on BRDF and illumination
representations using novel smooth manifold auto-encoders. Our decompositions
can result in considerably better BRDF and light estimates enabling more
accurate novel view-synthesis and relighting compared to prior art. Project
page: this https URL


### [[2110.14375] Perceptual Score: What Data Modalities Does Your Model Perceive?](http://arxiv.org/abs/2110.14375)


  Machine learning advances in the last decade have relied significantly on
large-scale datasets that continue to grow in size. Increasingly, those
datasets also contain different data modalities. However, large multi-modal
datasets are hard to annotate, and annotations may contain biases that we are
often unaware of. Deep-net-based classifiers, in turn, are prone to exploit
those biases and to find shortcuts. To study and quantify this concern, we
introduce the perceptual score, a metric that assesses the degree to which a
model relies on the different subsets of the input features, i.e., modalities.
Using the perceptual score, we find a surprisingly consistent trend across four
popular datasets: recent, more accurate state-of-the-art multi-modal models for
visual question-answering or visual dialog tend to perceive the visual data
less than their predecessors. This trend is concerning as answers are hence
increasingly inferred from textual cues only. Using the perceptual score also
helps to analyze model biases by decomposing the score into data subset
contributions. We hope to spur a discussion on the perceptiveness of
multi-modal models and also hope to encourage the community working on
multi-modal classifiers to start quantifying perceptiveness via the proposed
perceptual score.

    

### [[2110.14377] Node Dependent Local Smoothing for Scalable Graph Learning](http://arxiv.org/abs/2110.14377)


  Recent works reveal that feature or label smoothing lies at the core of Graph
Neural Networks (GNNs). Concretely, they show feature smoothing combined with
simple linear regression achieves comparable performance with the carefully
designed GNNs, and a simple MLP model with label smoothing of its prediction
can outperform the vanilla GCN. Though an interesting finding, smoothing has
not been well understood, especially regarding how to control the extent of
smoothness. Intuitively, too small or too large smoothing iterations may cause
under-smoothing or over-smoothing and can lead to sub-optimal performance.
Moreover, the extent of smoothness is node-specific, depending on its degree
and local structure. To this end, we propose a novel algorithm called
node-dependent local smoothing (NDLS), which aims to control the smoothness of
every node by setting a node-specific smoothing iteration. Specifically, NDLS
computes influence scores based on the adjacency matrix and selects the
iteration number by setting a threshold on the scores. Once selected, the
iteration number can be applied to both feature smoothing and label smoothing.
Experimental results demonstrate that NDLS enjoys high accuracy --
state-of-the-art performance on node classifications tasks, flexibility -- can
be incorporated with any models, scalability and efficiency -- can support
large scale graphs with fast training.

    

### [[2110.14381] Temporal-attentive Covariance Pooling Networks for Video Recognition](http://arxiv.org/abs/2110.14381)


  For video recognition task, a global representation summarizing the whole
contents of the video snippets plays an important role for the final
performance. However, existing video architectures usually generate it by using
a simple, global average pooling (GAP) method, which has limited ability to
capture complex dynamics of videos. For image recognition task, there exist
evidences showing that covariance pooling has stronger representation ability
than GAP. Unfortunately, such plain covariance pooling used in image
recognition is an orderless representative, which cannot model spatio-temporal
structure inherent in videos. Therefore, this paper proposes a
Temporal-attentive Covariance Pooling(TCP), inserted at the end of deep
architectures, to produce powerful video representations. Specifically, our TCP
first develops a temporal attention module to adaptively calibrate
spatio-temporal features for the succeeding covariance pooling, approximatively
producing attentive covariance representations. Then, a temporal covariance
pooling performs temporal pooling of the attentive covariance representations
to characterize both intra-frame correlations and inter-frame
cross-correlations of the calibrated features. As such, the proposed TCP can
capture complex temporal dynamics. Finally, a fast matrix power normalization
is introduced to exploit geometry of covariance representations. Note that our
TCP is model-agnostic and can be flexibly integrated into any video
architectures, resulting in TCPNet for effective video recognition. The
extensive experiments on six benchmarks using various video architectures show
our TCPNet is clearly superior to its counterparts, while having strong
generalization
ability.$\href{this https URL}{\textit{The
source code is publicly available.}}$

    

### [[2110.14383] Traffic Forecasting on Traffic Moving Snippets](http://arxiv.org/abs/2110.14383)


  Advances in traffic forecasting technology can greatly impact urban mobility.
In the traffic4cast competition, the task of short-term traffic prediction is
tackled in unprecedented detail, with traffic volume and speed information
available at 5 minute intervals and high spatial resolution. To improve
generalization to unknown cities, as required in the 2021 extended challenge,
we propose to predict small quadratic city sections, rather than processing a
full-city-raster at once. At test time, breaking down the test data into
spatially-cropped overlapping snippets improves stability and robustness of the
final predictions, since multiple patches covering one cell can be processed
independently. With the performance on the traffic4cast test data and further
experiments on a validation set it is shown that patch-wise prediction indeed
improves accuracy. Further advantages can be gained with a Unet++ architecture
and with an increasing number of patches per sample processed at test time. We
conclude that our snippet-based method, combined with other successful network
architectures proposed in the competition, can leverage performance, in
particular on unseen cities. All source code is available at
this https URL.

    

### [[2110.14402] Learning where to learn: Gradient sparsity in meta and continual learning](http://arxiv.org/abs/2110.14402)


  Finding neural network weights that generalize well from small datasets is
difficult. A promising approach is to learn a weight initialization such that a
small number of weight changes results in low generalization error. We show
that this form of meta-learning can be improved by letting the learning
algorithm decide which weights to change, i.e., by learning where to learn. We
find that patterned sparsity emerges from this process, with the pattern of
sparsity varying on a problem-by-problem basis. This selective sparsity results
in better generalization and less interference in a range of few-shot and
continual learning problems. Moreover, we find that sparse learning also
emerges in a more expressive model where learning rates are meta-learned. Our
results shed light on an ongoing debate on whether meta-learning can discover
adaptable features and suggest that learning by sparse gradient descent is a
powerful inductive bias for meta-learning systems.

    

### [[2110.14413] Localized Super Resolution for Foreground Images using U-Net and MR-CNN](http://arxiv.org/abs/2110.14413)


  Images play a vital role in understanding data through visual representation.
It gives a clear representation of the object in context. But if this image is
not clear it might not be of much use. Thus, the topic of Image Super
Resolution arose and many researchers have been working towards applying
Computer Vision and Deep Learning Techniques to increase the quality of images.
One of the applications of Super Resolution is to increase the quality of
Portrait Images. Portrait Images are images which mainly focus on capturing the
essence of the main object in the frame, where the object in context is
highlighted whereas the background is occluded. When performing Super
Resolution the model tries to increase the overall resolution of the image. But
in portrait images the foreground resolution is more important than that of the
background. In this paper, the performance of a Convolutional Neural Network
(CNN) architecture known as U-Net for Super Resolution combined with Mask
Region Based CNN (MR-CNN) for foreground super resolution is analysed. This
analysis is carried out based on Localized Super Resolution i.e. We pass the LR
Images to a pre-trained Image Segmentation model (MR-CNN) and perform super
resolution inference on the foreground or Segmented Images and compute the
Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR)
metrics for comparisons.

    

### [[2110.14416] Transformers Generalize DeepSets and Can be Extended to Graphs and Hypergraphs](http://arxiv.org/abs/2110.14416)


  We present a generalization of Transformers to any-order permutation
invariant data (sets, graphs, and hypergraphs). We begin by observing that
Transformers generalize DeepSets, or first-order (set-input) permutation
invariant MLPs. Then, based on recently characterized higher-order invariant
MLPs, we extend the concept of self-attention to higher orders and propose
higher-order Transformers for order-$k$ data ($k=2$ for graphs and $k>2$ for
hypergraphs). Unfortunately, higher-order Transformers turn out to have
prohibitive complexity $\mathcal{O}(n^{2k})$ to the number of input nodes $n$.
To address this problem, we present sparse higher-order Transformers that have
quadratic complexity to the number of input hyperedges, and further adopt the
kernel attention approach to reduce the complexity to linear. In particular, we
show that the sparse second-order Transformers with kernel attention are
theoretically more expressive than message passing operations while having an
asymptotically identical complexity. Our models achieve significant performance
improvement over invariant MLPs and message-passing graph neural networks in
large-scale graph regression and set-to-(hyper)graph prediction tasks. Our
implementation is available at this https URL.

    

### [[2110.14423] Vector-valued Gaussian Processes on Riemannian Manifolds via Gauge Equivariant Projected Kernels](http://arxiv.org/abs/2110.14423)


  Gaussian processes are machine learning models capable of learning unknown
functions in a way that represents uncertainty, thereby facilitating
construction of optimal decision-making systems. Motivated by a desire to
deploy Gaussian processes in novel areas of science, a rapidly-growing line of
research has focused on constructively extending these models to handle
non-Euclidean domains, including Riemannian manifolds, such as spheres and
tori. We propose techniques that generalize this class to model vector fields
on Riemannian manifolds, which are important in a number of application areas
in the physical sciences. To do so, we present a general recipe for
constructing gauge equivariant kernels, which induce Gaussian vector fields,
i.e. vector-valued Gaussian processes coherent with geometry, from
scalar-valued Riemannian kernels. We extend standard Gaussian process training
methods, such as variational inference, to this setting. This enables
vector-valued Gaussian processes on Riemannian manifolds to be trained using
standard methods and makes them accessible to machine learning practitioners.

    

### [[2110.14425] Generalizing AUC Optimization to Multiclass Classification for Audio Segmentation With Limited Training Data](http://arxiv.org/abs/2110.14425)


  Area under the ROC curve (AUC) optimisation techniques developed for neural
networks have recently demonstrated their capabilities in different audio and
speech related tasks. However, due to its intrinsic nature, AUC optimisation
has focused only on binary tasks so far. In this paper, we introduce an
extension to the AUC optimisation framework so that it can be easily applied to
an arbitrary number of classes, aiming to overcome the issues derived from
training data limitations in deep learning solutions. Building upon the
multiclass definitions of the AUC metric found in the literature, we define two
new training objectives using a one-versus-one and a one-versus-rest approach.
In order to demonstrate its potential, we apply them in an audio segmentation
task with limited training data that aims to differentiate 3 classes:
foreground music, background music and no music. Experimental results show that
our proposal can improve the performance of audio segmentation systems
significantly compared to traditional training criteria such as cross entropy.

    

### [[2110.14426] Locally Differentially Private Bayesian Inference](http://arxiv.org/abs/2110.14426)


  In recent years, local differential privacy (LDP) has emerged as a technique
of choice for privacy-preserving data collection in several scenarios when the
aggregator is not trustworthy. LDP provides client-side privacy by adding noise
at the user's end. Thus, clients need not rely on the trustworthiness of the
aggregator.
In this work, we provide a noise-aware probabilistic modeling framework,
which allows Bayesian inference to take into account the noise added for
privacy under LDP, conditioned on locally perturbed observations. Stronger
privacy protection (compared to the central model) provided by LDP protocols
comes at a much harsher privacy-utility trade-off. Our framework tackles
several computational and statistical challenges posed by LDP for accurate
uncertainty quantification under Bayesian settings. We demonstrate the efficacy
of our framework in parameter estimation for univariate and multi-variate
distributions as well as logistic and linear regression.

    

### [[2110.14427] The ODE Method for Asymptotic Statistics in Stochastic Approximation and Reinforcement Learning](http://arxiv.org/abs/2110.14427)


  The paper concerns convergence and asymptotic statistics for stochastic
approximation driven by Markovian noise: $$ \theta_{n+1}= \theta_n + \alpha_{n
+ 1} f(\theta_n, \Phi_{n+1}) \,,\quad n\ge 0, $$ in which each
$\theta_n\in\Re^d$, $ \{ \Phi_n \}$ is a Markov chain on a general state space
X with stationary distribution $\pi$, and $f:\Re^d\times \text{X} \to\Re^d$. In
addition to standard Lipschitz bounds on $f$, and conditions on the vanishing
step-size sequence $\{\alpha_n\}$, it is assumed that the associated ODE is
globally asymptotically stable with stationary point denoted $\theta^*$, where
$\bar f(\theta)=E[f(\theta,\Phi)]$ with $\Phi\sim\pi$. Moreover, the
ODE@$\infty$ defined with respect to the vector field, $$ \bar
f_\infty(\theta):= \lim_{r\to\infty} r^{-1} \bar f(r\theta) \,,\qquad
\theta\in\Re^d, $$ is asymptotically stable. The main contributions are
summarized as follows:
(i) The sequence $\theta$ is convergent if $\Phi$ is geometrically ergodic,
and subject to compatible bounds on $f$.
The remaining results are established under a stronger assumption on the
Markov chain: A slightly weaker version of the Donsker-Varadhan Lyapunov drift
condition known as (DV3).
(ii) A Lyapunov function is constructed for the joint process
$\{\theta_n,\Phi_n\}$ that implies convergence of $\{ \theta_n\}$ in $L_4$.
(iii) A functional CLT is established, as well as the usual one-dimensional
CLT for the normalized error $z_n:= (\theta_n-\theta^*)/\sqrt{\alpha_n}$.
Moment bounds combined with the CLT imply convergence of the normalized
covariance, $$ \lim_{n \to \infty} E [ z_n z_n^T ] = \Sigma_\theta, $$ where
$\Sigma_\theta$ is the asymptotic covariance appearing in the CLT.
(iv) An example is provided where the Markov chain $\Phi$ is geometrically
ergodic but it does not satisfy (DV3). While the algorithm is convergent, the
second moment is unbounded.

    

### [[2110.14430] Adversarial Neuron Pruning Purifies Backdoored Deep Models](http://arxiv.org/abs/2110.14430)


  As deep neural networks (DNNs) are growing larger, their requirements for
computational resources become huge, which makes outsourcing training more
popular. Training in a third-party platform, however, may introduce potential
risks that a malicious trainer will return backdoored DNNs, which behave
normally on clean samples but output targeted misclassifications whenever a
trigger appears at the test time. Without any knowledge of the trigger, it is
difficult to distinguish or recover benign DNNs from backdoored ones. In this
paper, we first identify an unexpected sensitivity of backdoored DNNs, that is,
they are much easier to collapse and tend to predict the target label on clean
samples when their neurons are adversarially perturbed. Based on these
observations, we propose a novel model repairing method, termed Adversarial
Neuron Pruning (ANP), which prunes some sensitive neurons to purify the
injected backdoor. Experiments show, even with only an extremely small amount
of clean data (e.g., 1%), ANP effectively removes the injected backdoor without
causing obvious performance degradation.

    

### [[2110.14432] Iterative Teaching by Label Synthesis](http://arxiv.org/abs/2110.14432)


  In this paper, we consider the problem of iterative machine teaching, where a
teacher provides examples sequentially based on the current iterative learner.
In contrast to previous methods that have to scan over the entire pool and
select teaching examples from it in each iteration, we propose a label
synthesis teaching framework where the teacher randomly selects input teaching
examples (e.g., images) and then synthesizes suitable outputs (e.g., labels)
for them. We show that this framework can avoid costly example selection while
still provably achieving exponential teachability. We propose multiple novel
teaching algorithms in this framework. Finally, we empirically demonstrate the
value of our framework.

    

### [[2110.14434] Nonnegative Tucker Decomposition with Beta-divergence for Music Structure Analysis of audio signals](http://arxiv.org/abs/2110.14434)


  Nonnegative Tucker Decomposition (NTD), a tensor decomposition model, has
received increased interest in the recent years because of its ability to
blindly extract meaningful patterns in tensor data. Nevertheless, existing
algorithms to compute NTD are mostly designed for the Euclidean loss. On the
other hand, NTD has recently proven to be a powerful tool in Music Information
Retrieval. This work proposes a Multiplicative Updates algorithm to compute NTD
with the beta-divergence loss, often considered a better loss for audio
processing. We notably show how to implement efficiently the multiplicative
rules using tensor algebra, a naive approach being intractable. Finally, we
show on a Music Structure Analysis task that unsupervised NTD fitted with
beta-divergence loss outperforms earlier results obtained with the Euclidean
loss.

    

### [[2110.14437] Exploring single-song autoencoding schemes for audio-based music structure analysis](http://arxiv.org/abs/2110.14437)


  The ability of deep neural networks to learn complex data relations and
representations is established nowadays, but it generally relies on large sets
of training data. This work explores a "piece-specific" autoencoding scheme, in
which a low-dimensional autoencoder is trained to learn a latent/compressed
representation specific to a given song, which can then be used to infer the
song structure. Such a model does not rely on supervision nor annotations,
which are well-known to be tedious to collect and often ambiguous in Music
Structure Analysis. We report that the proposed unsupervised auto-encoding
scheme achieves the level of performance of supervised state-of-the-art methods
with 3 seconds tolerance when using a Log Mel spectrogram representation on the
RWC-Pop dataset.

    

### [[2110.14443] Failure-averse Active Learning for Physics-constrained Systems](http://arxiv.org/abs/2110.14443)


  Active learning is a subfield of machine learning that is devised for design
and modeling of systems with highly expensive sampling costs. Industrial and
engineering systems are generally subject to physics constraints that may
induce fatal failures when they are violated, while such constraints are
frequently underestimated in active learning. In this paper, we develop a novel
active learning method that avoids failures considering implicit physics
constraints that govern the system. The proposed approach is driven by two
tasks: the safe variance reduction explores the safe region to reduce the
variance of the target model, and the safe region expansion aims to extend the
explorable region exploiting the probabilistic model of constraints. The global
acquisition function is devised to judiciously optimize acquisition functions
of two tasks, and its theoretical properties are provided. The proposed method
is applied to the composite fuselage assembly process with consideration of
material failure using the Tsai-wu criterion, and it is able to achieve
zero-failure without the knowledge of explicit failure regions.

    

### [[2110.14446] Large Scale Learning on Non-Homophilous Graphs: New Benchmarks and Strong Simple Methods](http://arxiv.org/abs/2110.14446)


  Many widely used datasets for graph machine learning tasks have generally
been homophilous, where nodes with similar labels connect to each other.
Recently, new Graph Neural Networks (GNNs) have been developed that move beyond
the homophily regime; however, their evaluation has often been conducted on
small graphs with limited application domains. We collect and introduce diverse
non-homophilous datasets from a variety of application areas that have up to
384x more nodes and 1398x more edges than prior datasets. We further show that
existing scalable graph learning and graph minibatching techniques lead to
performance degradation on these non-homophilous datasets, thus highlighting
the need for further work on scalable non-homophilous methods. To address these
concerns, we introduce LINKX -- a strong simple method that admits
straightforward minibatch training and inference. Extensive experimental
results with representative simple methods and GNNs across our proposed
datasets show that LINKX achieves state-of-the-art performance for learning on
non-homophilous graphs. Our codes and data are available at
this https URL.

    

### [[2110.14451] Validation Methods for Energy Time Series Scenarios from Deep Generative Models](http://arxiv.org/abs/2110.14451)


  The design and operation of modern energy systems are heavily influenced by
time-dependent and uncertain parameters, e.g., renewable electricity
generation, load-demand, and electricity prices. These are typically
represented by a set of discrete realizations known as scenarios. A popular
scenario generation approach uses deep generative models (DGM) that allow
scenario generation without prior assumptions about the data distribution.
However, the validation of generated scenarios is difficult, and a
comprehensive discussion about appropriate validation methods is currently
lacking. To start this discussion, we provide a critical assessment of the
currently used validation methods in the energy scenario generation literature.
In particular, we assess validation methods based on probability density,
auto-correlation, and power spectral density. Furthermore, we propose using the
multifractal detrended fluctuation analysis (MFDFA) as an additional validation
method for non-trivial features like peaks, bursts, and plateaus. As
representative examples, we train generative adversarial networks (GANs),
Wasserstein GANs (WGANs), and variational autoencoders (VAEs) on two renewable
power generation time series (photovoltaic and wind from Germany in 2013 to
2015) and an intra-day electricity price time series form the European Energy
Exchange in 2017 to 2019. We apply the four validation methods to both the
historical and the generated data and discuss the interpretation of validation
results as well as common mistakes, pitfalls, and limitations of the validation
methods. Our assessment shows that no single method sufficiently characterizes
a scenario but ideally validation should include multiple methods and be
interpreted carefully in the context of scenarios over short time periods.

    

### [[2110.14455] CBIR using Pre-Trained Neural Networks](http://arxiv.org/abs/2110.14455)


  Much of the recent research work in image retrieval, has been focused around
using Neural Networks as the core component. Many of the papers in other domain
have shown that training multiple models, and then combining their outcomes,
provide good results. This is since, a single Neural Network model, may not
extract sufficient information from the input. In this paper, we aim to follow
a different approach. Instead of the using a single model, we use a pretrained
Inception V3 model, and extract activation of its last fully connected layer,
which forms a low dimensional representation of the image. This feature matrix,
is then divided into branches and separate feature extraction is done for each
branch, to obtain multiple features flattened into a vector. Such individual
vectors are then combined, to get a single combined feature. We make use of
CUB200-2011 Dataset, which comprises of 200 birds classes to train the model
on. We achieved a training accuracy of 99.46% and validation accuracy of 84.56%
for the same. On further use of 3 branched global descriptors, we improve the
validation accuracy to 88.89%. For this, we made use of MS-RMAC feature
extraction method.

    

### [[2110.14457] Direct then Diffuse: Incremental Unsupervised Skill Discovery for State Covering and Goal Reaching](http://arxiv.org/abs/2110.14457)


  Learning meaningful behaviors in the absence of reward is a difficult problem
in reinforcement learning. A desirable and challenging unsupervised objective
is to learn a set of diverse skills that provide a thorough coverage of the
state space while being directed, i.e., reliably reaching distinct regions of
the environment. In this paper, we build on the mutual information framework
for skill discovery and introduce UPSIDE, which addresses the
coverage-directedness trade-off in the following ways: 1) We design policies
with a decoupled structure of a directed skill, trained to reach a specific
region, followed by a diffusing part that induces a local coverage. 2) We
optimize policies by maximizing their number under the constraint that each of
them reaches distinct regions of the environment (i.e., they are sufficiently
discriminable) and prove that this serves as a lower bound to the original
mutual information objective. 3) Finally, we compose the learned directed
skills into a growing tree that adaptively covers the environment. We
illustrate in several navigation and control environments how the skills
learned by UPSIDE solve sparse-reward downstream tasks better than existing
baselines.

    

### [[2110.14459] Accelerating Gradient-based Meta Learner](http://arxiv.org/abs/2110.14459)


  Meta Learning has been in focus in recent years due to the meta-learner
model's ability to adapt well and generalize to new tasks, thus, reducing both
the time and data requirements for learning. However, a major drawback of meta
learner is that, to reach to a state from where learning new tasks becomes
feasible with less data, it requires a large number of iterations and a lot of
time. We address this issue by proposing various acceleration techniques to
speed up meta learning algorithms such as MAML (Model Agnostic Meta Learning).
We present 3.73X acceleration on a well known RNN optimizer based meta learner
proposed in literature [11]. We introduce a novel method of training tasks in
clusters, which not only accelerates the meta learning process but also
improves model accuracy performance.
Keywords: Meta learning, RNN optimizer, AGI, Performance optimization

    

### [[2110.14464] Learning from demonstrations with SACR2: Soft Actor-Critic with Reward Relabeling](http://arxiv.org/abs/2110.14464)


  During recent years, deep reinforcement learning (DRL) has made successful
incursions into complex decision-making applications such as robotics,
autonomous driving or video games. However, a well-known caveat of DRL
algorithms is their inefficiency, requiring huge amounts of data to converge.
Off-policy algorithms tend to be more sample-efficient, and can additionally
benefit from any off-policy data stored in the replay buffer. Expert
demonstrations are a popular source for such data: the agent is exposed to
successful states and actions early on, which can accelerate the learning
process and improve performance. In the past, multiple ideas have been proposed
to make good use of the demonstrations in the buffer, such as pretraining on
demonstrations only or minimizing additional cost functions. We carry on a
study to evaluate several of these ideas in isolation, to see which of them
have the most significant impact. We also present a new method, based on a
reward bonus given to demonstrations and successful episodes. First, we give a
reward bonus to the transitions coming from demonstrations to encourage the
agent to match the demonstrated behaviour. Then, upon collecting a successful
episode, we relabel its transitions with the same bonus before adding them to
the replay buffer, encouraging the agent to also match its previous successes.
The base algorithm for our experiments is the popular Soft Actor-Critic (SAC),
a state-of-the-art off-policy algorithm for continuous action spaces. Our
experiments focus on robotics, specifically on a reaching task for a robotic
arm in simulation. We show that our method SACR2 based on reward relabeling
improves the performance on this task, even in the absence of demonstrations.

    

### [[2110.14468] DESTA: A Framework for Safe Reinforcement Learning with Markov Games of Intervention](http://arxiv.org/abs/2110.14468)


  Exploring in an unknown system can place an agent in dangerous situations,
exposing to potentially catastrophic hazards. Many current approaches for
tackling safe learning in reinforcement learning (RL) lead to a trade-off
between safe exploration and fulfilling the task. Though these methods possibly
incur fewer safety violations, they often also lead to reduced task
performance. In this paper, we take the first step in introducing a generation
of RL solvers that learn to minimise safety violations while maximising the
task reward to the extend that can be tolerated by safe policies. Our approach
uses a new two-player framework for safe RL called Distributive Exploration
Safety Training Algorithm (DESTA). The core of DESTA is a novel game between
two RL agents: SAFETY AGENT that is delegated the task of minimising safety
violations and TASK AGENT whose goal is to maximise the reward set by the
environment task. SAFETY AGENT can selectively take control of the system at
any given point to prevent safety violations while TASK AGENT is free to
execute its actions at all other states. This framework enables SAFETY AGENT to
learn to take actions that minimise future safety violations (during and after
training) by performing safe actions at certain states while TASK AGENT
performs actions that maximise the task performance everywhere else. We
demonstrate DESTA's ability to tackle challenging tasks and compare against
state-of-the-art RL methods in Safety Gym Benchmarks which simulate real-world
physical systems and OpenAI's Lunar Lander.

    

### [[2110.14484] PL-Net: Progressive Learning Network for Medical Image Segmentation](http://arxiv.org/abs/2110.14484)


  In recent years, segmentation methods based on deep convolutional neural
networks (CNNs) have made state-of-the-art achievements for many medical
analysis tasks. However, most of these approaches improve performance by
optimizing the structure or adding new functional modules of the U-Net, which
ignoring the complementation and fusion of the coarse-grained and fine-grained
semantic information. To solve the above problems, we propose a medical image
segmentation framework called progressive learning network (PL-Net), which
includes internal progressive learning (IPL) and external progressive learning
(EPL). PL-Net has the following advantages: (1) IPL divides feature extraction
into two "steps", which can mix different size receptive fields and capture
semantic information from coarse to fine granularity without introducing
additional parameters; (2) EPL divides the training process into two "stages"
to optimize parameters, and realizes the fusion of coarse-grained information
in the previous stage and fine-grained information in the latter stage. We
evaluate our method in different medical image analysis tasks, and the results
show that the segmentation performance of PL-Net is better than the
state-of-the-art methods of U-Net and its variants.

    

### [[2110.14503] Simple data balancing achieves competitive worst-group-accuracy](http://arxiv.org/abs/2110.14503)


  We study the problem of learning classifiers that perform well across (known
or unknown) groups of data. After observing that common worst-group-accuracy
datasets suffer from substantial imbalances, we set out to compare
state-of-the-art methods to simple balancing of classes and groups by either
subsampling or reweighting data. Our results show that these data balancing
baselines achieve state-of-the-art-accuracy, while being faster to train and
requiring no additional hyper-parameters. In addition, we highlight that access
to group information is most critical for model selection purposes, and not so
much during training. All in all, our findings beg closer examination of
benchmarks and methods for research in worst-group-accuracy optimization.

    

### [[2110.14508] Finding Regions of Heterogeneity in Decision-Making via Expected Conditional Covariance](http://arxiv.org/abs/2110.14508)


  Individuals often make different decisions when faced with the same context,
due to personal preferences and background. For instance, judges may vary in
their leniency towards certain drug-related offenses, and doctors may vary in
their preference for how to start treatment for certain types of patients. With
these examples in mind, we present an algorithm for identifying types of
contexts (e.g., types of cases or patients) with high inter-decision-maker
disagreement. We formalize this as a causal inference problem, seeking a region
where the assignment of decision-maker has a large causal effect on the
decision. Our algorithm finds such a region by maximizing an empirical
objective, and we give a generalization bound for its performance. In a
semi-synthetic experiment, we show that our algorithm recovers the correct
region of heterogeneity accurately compared to baselines. Finally, we apply our
algorithm to real-world healthcare datasets, recovering variation that aligns
with existing clinical knowledge.

    

### [[2110.14509] Deep Transfer Learning for Multi-source Entity Linkage via Domain Adaptation](http://arxiv.org/abs/2110.14509)


  Multi-source entity linkage focuses on integrating knowledge from multiple
sources by linking the records that represent the same real world entity. This
is critical in high-impact applications such as data cleaning and user
stitching. The state-of-the-art entity linkage pipelines mainly depend on
supervised learning that requires abundant amounts of training data. However,
collecting well-labeled training data becomes expensive when the data from many
sources arrives incrementally over time. Moreover, the trained models can
easily overfit to specific data sources, and thus fail to generalize to new
sources due to significant differences in data and label distributions. To
address these challenges, we present AdaMEL, a deep transfer learning framework
that learns generic high-level knowledge to perform multi-source entity
linkage. AdaMEL models the attribute importance that is used to match entities
through an attribute-level self-attention mechanism, and leverages the massive
unlabeled data from new data sources through domain adaptation to make it
generic and data-source agnostic. In addition, AdaMEL is capable of
incorporating an additional set of labeled data to more accurately integrate
data sources with different attribute importance. Extensive experiments show
that our framework achieves state-of-the-art results with 8.21% improvement on
average over methods based on supervised learning. Besides, it is more stable
in handling different sets of data sources in less runtime.

    

### [[2110.14514] Streaming Generalized Canonical Polyadic Tensor Decompositions](http://arxiv.org/abs/2110.14514)


  In this paper, we develop a method which we call OnlineGCP for computing the
Generalized Canonical Polyadic (GCP) tensor decomposition of streaming data.
GCP differs from traditional canonical polyadic (CP) tensor decompositions as
it allows for arbitrary objective functions which the CP model attempts to
minimize. This approach can provide better fits and more interpretable models
when the observed tensor data is strongly non-Gaussian. In the streaming case,
tensor data is gradually observed over time and the algorithm must
incrementally update a GCP factorization with limited access to prior data. In
this work, we extend the GCP formalism to the streaming context by deriving a
GCP optimization problem to be solved as new tensor data is observed, formulate
a tunable history term to balance reconstruction of recently observed data with
data observed in the past, develop a scalable solution strategy based on
segregated solves using stochastic gradient descent methods, describe a
software implementation that provides performance and portability to
contemporary CPU and GPU architectures and integrates with Matlab for enhanced
useability, and demonstrate the utility and performance of the approach and
software on several synthetic and real tensor data sets.

    

### [[2110.14518] NIDA-CLIFGAN: Natural Infrastructure Damage Assessment through Efficient Classification Combining Contrastive Learning, Information Fusion and Generative Adversarial Networks](http://arxiv.org/abs/2110.14518)


  During natural disasters, aircraft and satellites are used to survey the
impacted regions. Usually human experts are needed to manually label the
degrees of the building damage so that proper humanitarian assistance and
disaster response (HADR) can be achieved, which is labor-intensive and
time-consuming. Expecting human labeling of major disasters over a wide area
gravely slows down the HADR efforts. It is thus of crucial interest to take
advantage of the cutting-edge Artificial Intelligence and Machine Learning
techniques to speed up the natural infrastructure damage assessment process to
achieve effective HADR. Accordingly, the paper demonstrates a systematic effort
to achieve efficient building damage classification. First, two novel
generative adversarial nets (GANs) are designed to augment data used to train
the deep-learning-based classifier. Second, a contrastive learning based method
using novel data structures is developed to achieve great performance. Third,
by using information fusion, the classifier is effectively trained with very
few training data samples for transfer learning. All the classifiers are small
enough to be loaded in a smart phone or simple laptop for first responders.
Based on the available overhead imagery dataset, results demonstrate data and
computational efficiency with 10% of the collected data combined with a GAN
reducing the time of computation from roughly half a day to about 1 hour with
roughly similar classification performances.

    

### [[2110.14524] Model based Multi-agent Reinforcement Learning with Tensor Decompositions](http://arxiv.org/abs/2110.14524)


  A challenge in multi-agent reinforcement learning is to be able to generalize
over intractable state-action spaces. Inspired from Tesseract [Mahajan et al.,
2021], this position paper investigates generalisation in state-action space
over unexplored state-action pairs by modelling the transition and reward
functions as tensors of low CP-rank. Initial experiments on synthetic MDPs show
that using tensor decompositions in a model-based reinforcement learning
algorithm can lead to much faster convergence if the true transition and reward
functions are indeed of low rank.

    

### [[2110.14529] HSVI fo zs-POSGs using Concavity, Convexity and Lipschitz Properties](http://arxiv.org/abs/2110.14529)


  Dynamic programming and heuristic search are at the core of state-of-the-art
solvers for sequential decision-making problems. In partially observable or
collaborative settings (\eg, POMDPs and Dec-POMDPs), this requires introducing
an appropriate statistic that induces a fully observable problem as well as
bounding (convex) approximators of the optimal value function. This approach
has succeeded in some subclasses of 2-player zero-sum partially observable
stochastic games (zs-POSGs) as well, but failed in the general case despite
known concavity and convexity properties, which only led to heuristic
algorithms with poor convergence guarantees. We overcome this issue, leveraging
on these properties to derive bounding approximators and efficient update and
selection operators, before deriving a prototypical solver inspired by HSVI
that provably converges to an $\epsilon$-optimal solution in finite time, and
which we empirically evaluate. This opens the door to a novel family of
promising approaches complementing those relying on linear programming or
iterative methods.

    

### [[2110.14538] Reinforcement Learning in Factored Action Spaces using Tensor Decompositions](http://arxiv.org/abs/2110.14538)


  We present an extended abstract for the previously published work TESSERACT
[Mahajan et al., 2021], which proposes a novel solution for Reinforcement
Learning (RL) in large, factored action spaces using tensor decompositions. The
goal of this abstract is twofold: (1) To garner greater interest amongst the
tensor research community for creating methods and analysis for approximate RL,
(2) To elucidate the generalised setting of factored action spaces where tensor
decompositions can be used. We use cooperative multi-agent reinforcement
learning scenario as the exemplary setting where the action space is naturally
factored across agents and learning becomes intractable without resorting to
approximation on the underlying hypothesis space for candidate solutions.

    

### [[2110.14541] Deep Reinforcement Learning for Simultaneous Sensing and Channel Access in Cognitive Networks](http://arxiv.org/abs/2110.14541)


  We consider the problem of dynamic spectrum access (DSA) in cognitive
wireless networks, where only partial observations are available to the users
due to narrowband sensing and transmissions. The cognitive network consists of
primary users (PUs) and a secondary user (SU), which operate in a time
duplexing regime. The traffic pattern for each PU is assumed to be unknown to
the SU and is modeled as a finite-memory Markov chain. Since observations are
partial, then both channel sensing and access actions affect the throughput.
The objective is to maximize the SU's long-term throughput. To achieve this
goal, we develop a novel algorithm that learns both access and sensing policies
via deep Q-learning, dubbed Double Deep Q-network for Sensing and Access
(DDQSA). To the best of our knowledge, this is the first paper that solves both
sensing and access policies for DSA via deep Q-learning. Second, we analyze the
optimal policy theoretically to validate the performance of DDQSA. Although the
general DSA problem is P-SPACE hard, we derive the optimal policy explicitly
for a common model of a cyclic user dynamics. Our results show that DDQSA
learns a policy that implements both sensing and channel access, and
significantly outperforms existing approaches.

    

### [[2110.14549] Latent Equilibrium: A unified learning theory for arbitrarily fast computation with arbitrarily slow neurons](http://arxiv.org/abs/2110.14549)


  The response time of physical computational elements is finite, and neurons
are no exception. In hierarchical models of cortical networks each layer thus
introduces a response lag. This inherent property of physical dynamical systems
results in delayed processing of stimuli and causes a timing mismatch between
network output and instructive signals, thus afflicting not only inference, but
also learning. We introduce Latent Equilibrium, a new framework for inference
and learning in networks of slow components which avoids these issues by
harnessing the ability of biological neurons to phase-advance their output with
respect to their membrane potential. This principle enables quasi-instantaneous
inference independent of network depth and avoids the need for phased
plasticity or computationally expensive network relaxation phases. We jointly
derive disentangled neuron and synapse dynamics from a prospective energy
function that depends on a network's generalized position and momentum. The
resulting model can be interpreted as a biologically plausible approximation of
error backpropagation in deep cortical networks with continuous-time, leaky
neuronal dynamics and continuously active, local plasticity. We demonstrate
successful learning of standard benchmark datasets, achieving competitive
performance using both fully-connected and convolutional architectures, and
show how our principle can be applied to detailed models of cortical
microcircuitry. Furthermore, we study the robustness of our model to
spatio-temporal substrate imperfections to demonstrate its feasibility for
physical realization, be it in vivo or in silico.

    

### [[2110.14553] GenURL: A General Framework for Unsupervised Representation Learning](http://arxiv.org/abs/2110.14553)


  Recently unsupervised representation learning (URL) has achieved remarkable
progress in various scenarios. However, most methods are specifically designed
based on specific data characters or task assumptions. Based on the manifold
assumption, we regard most URL problems as an embedding problem that seeks an
optimal low-dimensional representation of the given high-dimensional data. We
split the embedding process into two steps, data structural modeling and
low-dimensional embedding, and propose a general similarity-based framework
called GenURL. Specifically, we provide a general method to model data
structures by adaptively combining graph distances on the feature space and
predefined graphs, then propose robust loss functions to learn the
low-dimensional embedding. Combining with a specific pretext task, we can adapt
GenURL to various URL tasks in a unified manner and achieve state-of-the-art
performance, including self-supervised visual representation learning,
unsupervised knowledge distillation, graph embeddings, and dimension reduction.
Moreover, ablation studies of loss functions and basic hyper-parameter settings
in GenURL illustrate the data characters of various tasks.

    

### [[2110.14555] V-Learning -- A Simple, Efficient, Decentralized Algorithm for Multiagent RL](http://arxiv.org/abs/2110.14555)


  A major challenge of multiagent reinforcement learning (MARL) is the curse of
multiagents, where the size of the joint action space scales exponentially with
the number of agents. This remains to be a bottleneck for designing efficient
MARL algorithms even in a basic scenario with finitely many states and actions.
This paper resolves this challenge for the model of episodic Markov games. We
design a new class of fully decentralized algorithms -- V-learning, which
provably learns Nash equilibria (in the two-player zero-sum setting),
correlated equilibria and coarse correlated equilibria (in the multiplayer
general-sum setting) in a number of samples that only scales with
$\max_{i\in[m]} A_i$, where $A_i$ is the number of actions for the $i^{\rm th}$
player. This is in sharp contrast to the size of the joint action space which
is $\prod_{i=1}^m A_i$. V-learning (in its basic form) is a new class of
single-agent RL algorithms that convert any adversarial bandit algorithm with
suitable regret guarantees into a RL algorithm. Similar to the classical
Q-learning algorithm, it performs incremental updates to the value functions.
Different from Q-learning, it only maintains the estimates of V-values instead
of Q-values. This key difference allows V-learning to achieve the claimed
guarantees in the MARL setting by simply letting all agents run V-learning
independently.

    

### [[2110.14565] DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations](http://arxiv.org/abs/2110.14565)


  Top-performing Model-Based Reinforcement Learning (MBRL) agents, such as
Dreamer, learn the world model by reconstructing the image observations. Hence,
they often fail to discard task-irrelevant details and struggle to handle
visual distractions. To address this issue, previous work has proposed to
contrastively learn the world model, but the performance tends to be inferior
in the absence of distractions. In this paper, we seek to enhance robustness to
distractions for MBRL agents. Specifically, we consider incorporating
prototypical representations, which have yielded more accurate and robust
results than contrastive approaches in computer vision. However, it remains
elusive how prototypical representations can benefit temporal dynamics learning
in MBRL, since they treat each image independently without capturing temporal
structures. To this end, we propose to learn the prototypes from the recurrent
states of the world model, thereby distilling temporal structures from past
observations and actions into the prototypes. The resulting model, DreamerPro,
successfully combines Dreamer with prototypes, making large performance gains
on the DeepMind Control suite both in the standard setting and when there are
complex background distractions. Code available at
this https URL .

    

### [[2110.14577] A Geometric Perspective towards Neural Calibration via Sensitivity Decomposition](http://arxiv.org/abs/2110.14577)


  It is well known that vision classification models suffer from poor
calibration in the face of data distribution shifts. In this paper, we take a
geometric approach to this problem. We propose Geometric Sensitivity
Decomposition (GSD) which decomposes the norm of a sample feature embedding and
the angular similarity to a target classifier into an instance-dependent and an
instance-independent component. The instance-dependent component captures the
sensitive information about changes in the input while the instance-independent
component represents the insensitive information serving solely to minimize the
loss on the training dataset. Inspired by the decomposition, we analytically
derive a simple extension to current softmax-linear models, which learns to
disentangle the two components during training. On several common vision
models, the disentangled model outperforms other calibration methods on
standard calibration metrics in the face of out-of-distribution (OOD) data and
corruption with significantly less complexity. Specifically, we surpass the
current state of the art by 30.8% relative improvement on corrupted CIFAR100 in
Expected Calibration Error. Code available at
this https URL.

    

### [[2110.14583] Deep learning via message passing algorithms based on belief propagation](http://arxiv.org/abs/2110.14583)


  Message-passing algorithms based on the Belief Propagation (BP) equations
constitute a well-known distributed computational scheme. It is exact on
tree-like graphical models and has also proven to be effective in many problems
defined on graphs with loops (from inference to optimization, from signal
processing to clustering). The BP-based scheme is fundamentally different from
stochastic gradient descent (SGD), on which the current success of deep
networks is based. In this paper, we present and adapt to mini-batch training
on GPUs a family of BP-based message-passing algorithms with a reinforcement
field that biases distributions towards locally entropic solutions. These
algorithms are capable of training multi-layer neural networks with discrete
weights and activations with performance comparable to SGD-inspired heuristics
(BinaryNet) and are naturally well-adapted to continual learning. Furthermore,
using these algorithms to estimate the marginals of the weights allows us to
make approximate Bayesian predictions that have higher accuracy than point-wise
solutions.

    

### [[2110.14588] Fuzzy Generative Adversarial Networks](http://arxiv.org/abs/2110.14588)


  Generative Adversarial Networks (GANs) are well-known tools for data
generation and semi-supervised classification. GANs, with less labeled data,
outperform Deep Neural Networks (DNNs) and Convolutional Neural Networks (CNNs)
in classification across various tasks, this shows promise for developing GANs
capable of trespassing into the domain of semi-supervised regression. However,
developing GANs for regression introduce two major challenges: (1) inherent
instability in the GAN formulation and (2) performing regression and achieving
stability simultaneously. This paper introduces techniques that show
improvement in the GANs' regression capability through mean absolute error
(MAE) and mean squared error (MSE). We bake a differentiable fuzzy logic system
at multiple locations in a GAN because fuzzy logic systems have demonstrated
high efficacy in classification and regression settings. The fuzzy logic takes
the output of either or both the generator and the discriminator to either or
both predict the output, $y$, and evaluate the generator's performance. We
outline the results of applying the fuzzy logic system to CGAN and summarize
each approach's efficacy. This paper shows that adding a fuzzy logic layer can
enhance GAN's ability to perform regression; the most desirable injection
location is problem-specific, and we show this through experiments over various
datasets. Besides, we demonstrate empirically that the fuzzy-infused GAN is
competitive with DNNs.

    

### [[2110.14590] TMBuD: A dataset for urban scene building detection](http://arxiv.org/abs/2110.14590)


  Building recognition and 3D reconstruction of human made structures in urban
scenarios has become an interesting and actual topic in the image processing
domain. For this research topic the Computer Vision and Augmented Reality areas
intersect for creating a better understanding of the urban scenario for various
topics. In this paper we aim to introduce a dataset solution, the TMBuD, that
is better fitted for image processing on human made structures for urban scene
scenarios. The proposed dataset will allow proper evaluation of salient edges
and semantic segmentation of images focusing on the street view perspective of
buildings. The images that form our dataset offer various street view
perspectives of buildings from urban scenarios, which allows for evaluating
complex algorithms. The dataset features 160 images of buildings from
Timisoara, Romania, with a resolution of 768 x 1024 pixels each.

    

### [[2110.14594] End-to-end LSTM based estimation of volcano event epicenter localization](http://arxiv.org/abs/2110.14594)


  In this paper, an end-to-end based LSTM scheme is proposed to address the
problem of volcano event localization without any a priori model relating phase
picking with localization estimation. It is worth emphasizing that automatic
phase picking in volcano signals is highly inaccurate because of the short
distances between the event epicenters and the seismograph stations. LSTM was
chosen due to its capability to capture the dynamics of time varying signals,
and to remove or add information within the memory cell state and model
long-term dependencies. A brief insight into LSTM is also discussed here. The
results presented in this paper show that the LSTM based architecture provided
a success rate, i.e., an error smaller than 1.0Km, equal to 48.5%, which in
turn is dramatically superior to the one delivered by automatic phase picking.
Moreover, the proposed end-to-end LSTM based method gave a success rate 18%
higher than CNN.

    

### [[2110.14597] Evaluating Deep Learning Models and Adversarial Attacks on Accelerometer-Based Gesture Authentication](http://arxiv.org/abs/2110.14597)


  Gesture-based authentication has emerged as a non-intrusive, effective means
of authenticating users on mobile devices. Typically, such authentication
techniques have relied on classical machine learning techniques, but recently,
deep learning techniques have been applied this problem. Although prior
research has shown that deep learning models are vulnerable to adversarial
attacks, relatively little research has been done in the adversarial domain for
behavioral biometrics. In this research, we collect tri-axial accelerometer
gesture data (TAGD) from 46 users and perform classification experiments with
both classical machine learning and deep learning models. Specifically, we
train and test support vector machines (SVM) and convolutional neural networks
(CNN). We then consider a realistic adversarial attack, where we assume the
attacker has access to real users' TAGD data, but not the authentication model.
We use a deep convolutional generative adversarial network (DC-GAN) to create
adversarial samples, and we show that our deep learning model is surprisingly
robust to such an attack scenario.

    

### [[2110.14602] Towards a Theory of Evolution as Multilevel Learning](http://arxiv.org/abs/2110.14602)


  We apply the theory of learning to physically renormalizable systems in an
attempt to develop a theory of biological evolution, including the origin of
life, as multilevel learning. We formulate seven fundamental principles of
evolution that appear to be necessary and sufficient to render a universe
observable and show that they entail the major features of biological
evolution, including replication and natural selection. These principles also
follow naturally from the theory of learning. We formulate the theory of
evolution using the mathematical framework of neural networks, which provides
for detailed analysis of evolutionary phenomena. To demonstrate the potential
of the proposed theoretical framework, we derive a generalized version of the
Central Dogma of molecular biology by analyzing the flow of information during
learning (back-propagation) and predicting (forward-propagation) the
environment by evolving organisms. The more complex evolutionary phenomena,
such as major transitions in evolution, in particular, the origin of life, have
to be analyzed in the thermodynamic limit, which is described in detail in the
accompanying paper.

    

### [[2110.14615] Play to Grade: Testing Coding Games as Classifying Markov Decision Process](http://arxiv.org/abs/2110.14615)


  Contemporary coding education often presents students with the task of
developing programs that have user interaction and complex dynamic systems,
such as mouse based games. While pedagogically compelling, there are no
contemporary autonomous methods for providing feedback. Notably, interactive
programs are impossible to grade by traditional unit tests. In this paper we
formalize the challenge of providing feedback to interactive programs as a task
of classifying Markov Decision Processes (MDPs). Each student's program fully
specifies an MDP where the agent needs to operate and decide, under reasonable
generalization, if the dynamics and reward model of the input MDP should be
categorized as correct or broken. We demonstrate that by designing a
cooperative objective between an agent and an autoregressive model, we can use
the agent to sample differential trajectories from the input MDP that allows a
classifier to determine membership: Play to Grade. Our method enables an
automatic feedback system for interactive code assignments. We release a
dataset of 711,274 anonymized student submissions to a single assignment with
hand-coded bug labels to support future research.

    

### [[2110.14621] Fairer LP-based Online Allocation](http://arxiv.org/abs/2110.14621)


  In this paper, we consider a Linear Program (LP)-based online resource
allocation problem where a decision maker accepts or rejects incoming customer
requests irrevocably in order to maximize expected revenue given limited
resources. At each time, a new order/customer/bid is revealed with a request of
some resource(s) and a reward. We consider a stochastic setting where all the
orders are i.i.d. sampled from an unknown distribution. Such formulation
contains many classic applications such as the canonical (quantity-based)
network revenue management problem and the Adwords problem. Specifically, we
study the objective of providing fairness guarantees while maintaining low
regret. Our definition of fairness is that a fair online algorithm should treat
similar agents/customers similarly and the decision made for similar
individuals should be consistent over time. We define a fair offline solution
as the analytic center of the offline optimal solution set, and propose a fair
algorithm that uses an interior-point LP solver and dynamically detects unfair
resource spending. Our algorithm can control cumulative unfairness (the
cumulative deviation from the online solutions to the offline fair solution) on
the scale of order $O(\log(T))$, while maintaining the regret to be bounded
with dependency on $T$. Our approach do not formulate the fairness requirement
as a constrain in the optimization instance, and instead we address the problem
from the perspective of algorithm design. We get the desirable fairness
guarantee without imposing any fairness constraint, and our regret result is
strong for the reason that we evaluate the regret by comparing to the original
objective value.

    

### [[2110.14622] Heterogeneous Multi-player Multi-armed Bandits: Closing the Gap and Generalization](http://arxiv.org/abs/2110.14622)


  Despite the significant interests and many progresses in decentralized
multi-player multi-armed bandits (MP-MAB) problems in recent years, the regret
gap to the natural centralized lower bound in the heterogeneous MP-MAB setting
remains open. In this paper, we propose BEACON -- Batched Exploration with
Adaptive COmmunicatioN -- that closes this gap. BEACON accomplishes this goal
with novel contributions in implicit communication and efficient exploration.
For the former, we propose a novel adaptive differential communication (ADC)
design that significantly improves the implicit communication efficiency. For
the latter, a carefully crafted batched exploration scheme is developed to
enable incorporation of the combinatorial upper confidence bound (CUCB)
principle. We then generalize the existing linear-reward MP-MAB problems, where
the system reward is always the sum of individually collected rewards, to a new
MP-MAB problem where the system reward is a general (nonlinear) function of
individual rewards. We extend BEACON to solve this problem and prove a
logarithmic regret. BEACON bridges the algorithm design and regret analysis of
combinatorial MAB (CMAB) and MP-MAB, two largely disjointed areas in MAB, and
the results in this paper suggest that this previously ignored connection is
worth further investigation. Supplementary Material: pdf

    

### [[2110.14626] Scalable Bayesian Network Structure Learning with Splines](http://arxiv.org/abs/2110.14626)


  A Bayesian Network (BN) is a probabilistic graphical model consisting of a
directed acyclic graph (DAG), where each node is a random variable represented
as a function of its parents. We present a novel approach capable of learning
the global DAG structure of a BN and modelling linear and non-linear local
relationships between variables. We achieve this by a combination of feature
selection to reduce the search space for local relationships, and extending the
widely used score-and-search approach to support modelling relationships
between variables as Multivariate Adaptive Regression Splines (MARS). MARS are
polynomial regression models represented as piecewise spline functions - this
lets us model non-linear relationships without the risk of overfitting that a
single polynomial regression model would bring. The combination allows us to
learn relationships in all bnlearn benchmark instances within minutes and
enables us to scale to networks of over a thousand nodes

    

### [[2110.14628] (Almost) Free Incentivized Exploration from Decentralized Learning Agents](http://arxiv.org/abs/2110.14628)


  Incentivized exploration in multi-armed bandits (MAB) has witnessed
increasing interests and many progresses in recent years, where a principal
offers bonuses to agents to do explorations on her behalf. However, almost all
existing studies are confined to temporary myopic agents. In this work, we
break this barrier and study incentivized exploration with multiple and
long-term strategic agents, who have more complicated behaviors that often
appear in real-world applications. An important observation of this work is
that strategic agents' intrinsic needs of learning benefit (instead of harming)
the principal's explorations by providing "free pulls". Moreover, it turns out
that increasing the population of agents significantly lowers the principal's
burden of incentivizing. The key and somewhat surprising insight revealed from
our results is that when there are sufficiently many learning agents involved,
the exploration process of the principal can be (almost) free. Our main results
are built upon three novel components which may be of independent interest: (1)
a simple yet provably effective incentive-provision strategy; (2) a carefully
crafted best arm identification algorithm for rewards aggregated under unequal
confidences; (3) a high-probability finite-time lower bound of UCB algorithms.
Experimental results are provided to complement the theoretical analysis.

    

### [[2110.14633] Similarity and Matching of Neural Network Representations](http://arxiv.org/abs/2110.14633)


  We employ a toolset -- dubbed Dr. Frankenstein -- to analyse the similarity
of representations in deep neural networks. With this toolset, we aim to match
the activations on given layers of two trained neural networks by joining them
with a stitching layer. We demonstrate that the inner representations emerging
in deep convolutional neural networks with the same architecture but different
initializations can be matched with a surprisingly high degree of accuracy even
with a single, affine stitching layer. We choose the stitching layer from
several possible classes of linear transformations and investigate their
performance and properties. The task of matching representations is closely
related to notions of similarity. Using this toolset, we also provide a novel
viewpoint on the current line of research regarding similarity indices of
neural network representations: the perspective of the performance on a task.

    

### [[1901.05639] Machine learning with neural networks](http://arxiv.org/abs/1901.05639)


  These are lecture notes for a course on machine learning with neural networks
for scientists and engineers that I have given at Gothenburg University and
Chalmers Technical University in Gothenburg, Sweden. The material is organised
into three parts: Hopfield networks, supervised learning of labeled data, and
learning algorithms for unlabeled data sets. Part I introduces stochastic
recurrent networks: Hopfield networks and Boltzmann machines. The analysis of
their learning rules sets the scene for the later parts. Part II describes
supervised learning with multilayer perceptrons and convolutional neural
networks. This part starts with a simple geometrical interpretation of the
learning rule and leads to the recent successes of convolutional networks in
object recognition, recurrent networks in language processing, and reservoir
computers in time-series analysis. Part III explains what neural networks can
learn about data that is not labeled. This part begins with a description of
unsupervised learning techniques for clustering of data, non-linear
projections, and embeddings. A section on autoencoders explains how to learn
without labels using convolutional networks, and the last chapter is dedicated
to reinforcement learning. The overall goal of the course is to explain the
fundamental principles that allow neural networks to learn, emphasising ideas
and concepts that are common to all three parts.
The present version does not contain exercises (copyright owned by Cambridge
University Press). The complete book is available at
this https URL.

    

### [[1909.04497] Equity2Vec: End-to-end Deep Learning Framework for Cross-sectional Asset Pricing](http://arxiv.org/abs/1909.04497)


  Pricing assets has attracted significant attention from the financial
technology community. We observe that the existing solutions overlook the
cross-sectional effects and not fully leveraged the heterogeneous data sets,
leading to sub-optimal performance.
To this end, we propose an end-to-end deep learning framework to price the
assets. Our framework possesses two main properties: 1) We propose Equity2Vec,
a graph-based component that effectively captures both long-term and evolving
cross-sectional interactions. 2) The framework simultaneously leverages all the
available heterogeneous alpha sources including technical indicators, financial
news signals, and cross-sectional signals. Experimental results on datasets
from the real-world stock market show that our approach outperforms the
existing state-of-the-art approaches. Furthermore, market trading simulations
demonstrate that our framework monetizes the signals effectively.

    

### [[1911.11815] Local Model Poisoning Attacks to Byzantine-Robust Federated Learning](http://arxiv.org/abs/1911.11815)


  In federated learning, multiple client devices jointly learn a machine
learning model: each client device maintains a local model for its local
training dataset, while a master device maintains a global model via
aggregating the local models from the client devices. The machine learning
community recently proposed several federated learning methods that were
claimed to be robust against Byzantine failures (e.g., system failures,
adversarial manipulations) of certain client devices. In this work, we perform
the first systematic study on local model poisoning attacks to federated
learning. We assume an attacker has compromised some client devices, and the
attacker manipulates the local model parameters on the compromised client
devices during the learning process such that the global model has a large
testing error rate. We formulate our attacks as optimization problems and apply
our attacks to four recent Byzantine-robust federated learning methods. Our
empirical results on four real-world datasets show that our attacks can
substantially increase the error rates of the models learnt by the federated
learning methods that were claimed to be robust against Byzantine failures of
some client devices. We generalize two defenses for data poisoning attacks to
defend against our local model poisoning attacks. Our evaluation results show
that one defense can effectively defend against our attacks in some cases, but
the defenses are not effective enough in other cases, highlighting the need for
new defenses against our local model poisoning attacks to federated learning.

    

### [[1912.02143] Landscape Complexity for the Empirical Risk of Generalized Linear Models](http://arxiv.org/abs/1912.02143)


  We present a method to obtain the average and the typical value of the number
of critical points of the empirical risk landscape for generalized linear
estimation problems and variants. This represents a substantial extension of
previous applications of the Kac-Rice method since it allows to analyze the
critical points of high dimensional non-Gaussian random functions. We obtain a
rigorous explicit variational formula for the annealed complexity, which is the
logarithm of the average number of critical points at fixed value of the
empirical risk. This result is simplified, and extended, using the non-rigorous
Kac-Rice replicated method from theoretical physics. In this way we find an
explicit variational formula for the quenched complexity, which is generally
different from its annealed counterpart, and allows to obtain the number of
critical points for typical instances up to exponential accuracy.

    

### [[2001.11279] Goal-directed graph construction using reinforcement learning](http://arxiv.org/abs/2001.11279)


  Graphs can be used to represent and reason about systems and a variety of
metrics have been devised to quantify their global characteristics. However,
little is currently known about how to construct a graph or improve an existing
one given a target objective. In this work, we formulate the construction of a
graph as a decision-making process in which a central agent creates topologies
by trial and error and receives rewards proportional to the value of the target
objective. By means of this conceptual framework, we propose an algorithm based
on reinforcement learning and graph neural networks to learn graph construction
and improvement strategies. Our core case study focuses on robustness to
failures and attacks, a property relevant for the infrastructure and
communication networks that power modern society. Experiments on synthetic and
real-world graphs show that this approach can outperform existing methods while
being cheaper to evaluate. It also allows generalization to out-of-sample
graphs, as well as to larger out-of-distribution graphs in some cases. The
approach is applicable to the optimization of other global structural
properties of graphs.

    

### [[2006.05057] Towards More Practical Adversarial Attacks on Graph Neural Networks](http://arxiv.org/abs/2006.05057)


  We study the black-box attacks on graph neural networks (GNNs) under a novel
and realistic constraint: attackers have access to only a subset of nodes in
the network, and they can only attack a small number of them. A node selection
step is essential under this setup. We demonstrate that the structural
inductive biases of GNN models can be an effective source for this type of
attacks. Specifically, by exploiting the connection between the backward
propagation of GNNs and random walks, we show that the common gradient-based
white-box attacks can be generalized to the black-box setting via the
connection between the gradient and an importance score similar to PageRank. In
practice, we find attacks based on this importance score indeed increase the
classification loss by a large margin, but they fail to significantly increase
the mis-classification rate. Our theoretical and empirical analyses suggest
that there is a discrepancy between the loss and mis-classification rate, as
the latter presents a diminishing-return pattern when the number of attacked
nodes increases. Therefore, we propose a greedy procedure to correct the
importance score that takes into account of the diminishing-return pattern.
Experimental results show that the proposed procedure can significantly
increase the mis-classification rate of common GNNs on real-world data without
access to model parameters nor predictions.

    

### [[2008.01645] A Visual Analytics Framework for Reviewing Multivariate Time-Series Data with Dimensionality Reduction](http://arxiv.org/abs/2008.01645)


  Data-driven problem solving in many real-world applications involves analysis
of time-dependent multivariate data, for which dimensionality reduction (DR)
methods are often used to uncover the intrinsic structure and features of the
data. However, DR is usually applied to a subset of data that is either
single-time-point multivariate or univariate time-series, resulting in the need
to manually examine and correlate the DR results out of different data subsets.
When the number of dimensions is large either in terms of the number of time
points or attributes, this manual task becomes too tedious and infeasible. In
this paper, we present MulTiDR, a new DR framework that enables processing of
time-dependent multivariate data as a whole to provide a comprehensive overview
of the data. With the framework, we employ DR in two steps. When treating the
instances, time points, and attributes of the data as a 3D array, the first DR
step reduces the three axes of the array to two, and the second DR step
visualizes the data in a lower-dimensional space. In addition, by coupling with
a contrastive learning method and interactive visualizations, our framework
enhances analysts' ability to interpret DR results. We demonstrate the
effectiveness of our framework with four case studies using real-world
datasets.

    

### [[2010.01051] Neural Bootstrapper](http://arxiv.org/abs/2010.01051)


  Bootstrapping has been a primary tool for ensemble and uncertainty
quantification in machine learning and statistics. However, due to its nature
of multiple training and resampling, bootstrapping deep neural networks is
computationally burdensome; hence it has difficulties in practical application
to the uncertainty estimation and related tasks. To overcome this computational
bottleneck, we propose a novel approach called \emph{Neural Bootstrapper}
(NeuBoots), which learns to generate bootstrapped neural networks through
single model training. NeuBoots injects the bootstrap weights into the
high-level feature layers of the backbone network and outputs the bootstrapped
predictions of the target, without additional parameters and the repetitive
computations from scratch. We apply NeuBoots to various machine learning tasks
related to uncertainty quantification, including prediction calibrations in
image classification and semantic segmentation, active learning, and detection
of out-of-distribution samples. Our empirical results show that NeuBoots
outperforms other bagging based methods under a much lower computational cost
without losing the validity of bootstrapping.

    

### [[2010.07778] Local Differential Privacy for Regret Minimization in Reinforcement Learning](http://arxiv.org/abs/2010.07778)


  Reinforcement learning algorithms are widely used in domains where it is
desirable to provide a personalized service. In these domains it is common that
user data contains sensitive information that needs to be protected from third
parties. Motivated by this, we study privacy in the context of finite-horizon
Markov Decision Processes (MDPs) by requiring information to be obfuscated on
the user side. We formulate this notion of privacy for RL by leveraging the
local differential privacy (LDP) framework. We establish a lower bound for
regret minimization in finite-horizon MDPs with LDP guarantees which shows that
guaranteeing privacy has a multiplicative effect on the regret. This result
shows that while LDP is an appealing notion of privacy, it makes the learning
problem significantly more complex. Finally, we present an optimistic algorithm
that simultaneously satisfies $\varepsilon$-LDP requirements, and achieves
$\sqrt{K}/\varepsilon$ regret in any finite-horizon MDP after $K$ episodes,
matching the lower bound dependency on the number of episodes $K$.

    

### [[2010.09063] Enabling Fast Differentially Private SGD via Just-in-Time Compilation and Vectorization](http://arxiv.org/abs/2010.09063)


  A common pain point in differentially private machine learning is the
significant runtime overhead incurred when executing Differentially Private
Stochastic Gradient Descent (DPSGD), which may be as large as two orders of
magnitude. We thoroughly demonstrate that by exploiting powerful language
primitives, including vectorization, just-in-time compilation, and static graph
optimization, one can dramatically reduce these overheads, in many cases nearly
matching the best non-private running times. These gains are realized in two
frameworks: JAX and TensorFlow. JAX provides rich support for these primitives
as core features of the language through the XLA compiler. We also rebuild core
parts of TensorFlow Privacy, integrating features from TensorFlow 2 as well as
XLA compilation, granting significant memory and runtime improvements over the
current release version. These approaches allow us to achieve up to 50x
speedups in comparison to the best alternatives. Our code is available at
this https URL.

    

### [[2010.11171] How Data Augmentation affects Optimization for Linear Regression](http://arxiv.org/abs/2010.11171)


  Though data augmentation has rapidly emerged as a key tool for optimization
in modern machine learning, a clear picture of how augmentation schedules
affect optimization and interact with optimization hyperparameters such as
learning rate is nascent. In the spirit of classical convex optimization and
recent work on implicit bias, the present work analyzes the effect of
augmentation on optimization in the simple convex setting of linear regression
with MSE loss.
We find joint schedules for learning rate and data augmentation scheme under
which augmented gradient descent provably converges and characterize the
resulting minimum. Our results apply to arbitrary augmentation schemes,
revealing complex interactions between learning rates and augmentations even in
the convex setting. Our approach interprets augmented (S)GD as a stochastic
optimization method for a time-varying sequence of proxy losses. This gives a
unified way to analyze learning rate, batch size, and augmentations ranging
from additive noise to random projections. From this perspective, our results,
which also give rates of convergence, can be viewed as Monro-Robbins type
conditions for augmented (S)GD.

    

### [[2010.12866] Optimal Algorithms for Stochastic Multi-Armed Bandits with Heavy Tailed Rewards](http://arxiv.org/abs/2010.12866)


  In this paper, we consider stochastic multi-armed bandits (MABs) with
heavy-tailed rewards, whose $p$-th moment is bounded by a constant $\nu_{p}$
for $1<p\leq2$. First, we propose a novel robust estimator which does not
require $\nu_{p}$ as prior information, while other existing robust estimators
demand prior knowledge about $\nu_{p}$. We show that an error probability of
the proposed estimator decays exponentially fast. Using this estimator, we
propose a perturbation-based exploration strategy and develop a generalized
regret analysis scheme that provides upper and lower regret bounds by revealing
the relationship between the regret and the cumulative density function of the
perturbation. From the proposed analysis scheme, we obtain gap-dependent and
gap-independent upper and lower regret bounds of various perturbations. We also
find the optimal hyperparameters for each perturbation, which can achieve the
minimax optimal regret bound with respect to total rounds. In simulation, the
proposed estimator shows favorable performance compared to existing robust
estimators for various $p$ values and, for MAB problems, the proposed
perturbation strategy outperforms existing exploration methods.

    

### [[2010.15206] Rosella: A Self-Driving Distributed Scheduler for Heterogeneous Clusters](http://arxiv.org/abs/2010.15206)


  Large-scale interactive web services and advanced AI applications make
sophisticated decisions in real-time, based on executing a massive amount of
computation tasks on thousands of servers. Task schedulers, which often operate
in heterogeneous and volatile environments, require high throughput, i.e.,
scheduling millions of tasks per second, and low latency, i.e., incurring
minimal scheduling delays for millisecond-level tasks. Scheduling is further
complicated by other users' workloads in a shared system, other background
activities, and the diverse hardware configurations inside datacenters.
We present Rosella, a new self-driving, distributed approach for task
scheduling in heterogeneous clusters. Rosella automatically learns the compute
environment and adjusts its scheduling policy in real-time. The solution
provides high throughput and low latency simultaneously because it runs in
parallel on multiple machines with minimum coordination and only performs
simple operations for each scheduling decision. Our learning module monitors
total system load and uses the information to dynamically determine optimal
estimation strategy for the backends' compute-power. Rosella generalizes
power-of-two-choice algorithms to handle heterogeneous workers, reducing the
max queue length of O(log n) obtained by prior algorithms to O(log log n). We
evaluate Rosella with a variety of workloads on a 32-node AWS cluster.
Experimental results show that Rosella significantly reduces task response
time, and adapts to environment changes quickly.

    

### [[2010.16103] Labeling Trick: A Theory of Using Graph Neural Networks for Multi-Node Representation Learning](http://arxiv.org/abs/2010.16103)


  In this paper, we provide a theory of using graph neural networks (GNNs) for
multi-node representation learning (where we are interested in learning a
representation for a set of more than one node). We know that GNN is designed
to learn single-node representations. When we want to learn a node set
representation involving multiple nodes, a common practice in previous works is
to directly aggregate the multiple node representations learned by a GNN into a
joint representation of the node set. In this paper, we show a fundamental
constraint of such an approach, namely the inability to capture the dependence
between nodes in the node set, and argue that directly aggregating individual
node representations does not lead to an effective joint representation for
multiple nodes. Then, we notice that a few previous successful works for
multi-node representation learning, including SEAL, Distance Encoding, and
ID-GNN, all used node labeling. These methods first label nodes in the graph
according to their relationships with the target node set before applying a
GNN. Then, the node representations obtained in the labeled graph are
aggregated into a node set representation. By investigating their inner
mechanisms, we unify these node labeling techniques into a single and most
basic form, namely labeling trick. We prove that with labeling trick a
sufficiently expressive GNN learns the most expressive node set
representations, thus in principle can solve any joint learning tasks over node
sets. Experiments on one important two-node representation learning task, link
prediction, verified our theory. Our work establishes a theoretical foundation
of using GNNs for joint prediction tasks over node sets.

    

### [[2011.03173] Does enforcing fairness mitigate biases caused by subpopulation shift?](http://arxiv.org/abs/2011.03173)


  Many instances of algorithmic bias are caused by subpopulation shifts. For
example, ML models often perform worse on demographic groups that are
underrepresented in the training data. In this paper, we study whether
enforcing algorithmic fairness during training improves the performance of the
trained model in the \emph{target domain}. On one hand, we conceive scenarios
in which enforcing fairness does not improve performance in the target domain.
In fact, it may even harm performance. On the other hand, we derive necessary
and sufficient conditions under which enforcing algorithmic fairness leads to
the Bayes model in the target domain. We also illustrate the practical
implications of our theoretical results in simulations and on real data.

    

### [[2011.03902] Learning Neural Event Functions for Ordinary Differential Equations](http://arxiv.org/abs/2011.03902)


  The existing Neural ODE formulation relies on an explicit knowledge of the
termination time. We extend Neural ODEs to implicitly defined termination
criteria modeled by neural event functions, which can be chained together and
differentiated through. Neural Event ODEs are capable of modeling discrete and
instantaneous changes in a continuous-time system, without prior knowledge of
when these changes should occur or how many such changes should exist. We test
our approach in modeling hybrid discrete- and continuous- systems such as
switching dynamical systems and collision in multi-body systems, and we propose
simulation-based training of point processes with applications in discrete
control.

    

### [[2011.06391] FusedMM: A Unified SDDMM-SpMM Kernel for Graph Embedding and Graph Neural Networks](http://arxiv.org/abs/2011.06391)


  We develop a fused matrix multiplication kernel that unifies sampled
dense-dense matrix multiplication and sparse-dense matrix multiplication under
a single operation called FusedMM. By using user-defined functions, FusedMM can
capture almost all computational patterns needed by popular graph embedding and
GNN approaches. FusedMM is an order of magnitude faster than its equivalent
kernels in Deep Graph Library. The superior performance of FusedMM comes from
the low-level vectorized kernels, a suitable load balancing scheme and an
efficient utilization of the memory bandwidth. FusedMM can tune its performance
using a code generator and perform equally well on Intel, AMD and ARM
processors. FusedMM speeds up an end-to-end graph embedding algorithm by up to
28x on different processors.

    

### [[2011.06741] Rebounding Bandits for Modeling Satiation Effects](http://arxiv.org/abs/2011.06741)


  Psychological research shows that enjoyment of many goods is subject to
satiation, with short-term satisfaction declining after repeated exposures to
the same item. Nevertheless, proposed algorithms for powering recommender
systems seldom model these dynamics, instead proceeding as though user
preferences were fixed in time. In this work, we introduce rebounding bandits,
a multi-armed bandit setup, where satiation dynamics are modeled as
time-invariant linear dynamical systems. Expected rewards for each arm decline
monotonically with consecutive exposures to it and rebound towards the initial
reward whenever that arm is not pulled. Unlike classical bandit settings,
methods for tackling rebounding bandits must plan ahead and model-based methods
rely on estimating the parameters of the satiation dynamics. We characterize
the planning problem, showing that the greedy policy is optimal when the arms
exhibit identical deterministic dynamics. To address stochastic satiation
dynamics with unknown parameters, we propose Explore-Estimate-Plan (EEP), an
algorithm that pulls arms methodically, estimates the system dynamics, and then
plans accordingly.

    

### [[2011.10510] Seismic Facies Analysis: A Deep Domain Adaptation Approach](http://arxiv.org/abs/2011.10510)


  Deep neural networks (DNNs) can learn accurately from large quantities of
labeled input data, but often fail to do so when labelled data are scarce. DNNs
sometimes fail to generalize ontest data sampled from different input
distributions. Unsupervised Deep Domain Adaptation (DDA)techniques have been
proven useful when no labels are available, and when distribution shifts are
observed in the target domain (TD). In the present study, experiments are
performed on seismic images of the F3 block 3D dataset from offshore
Netherlands (source domain; SD) and Penobscot 3D survey data from Canada
(target domain; TD). Three geological classes from SD and TD that have similar
reflection patterns are considered. A deep neural network architecture named
EarthAdaptNet (EAN) is proposed to semantically segment the seismic images when
few classes have data scarcity, and we use a transposed residual unit to
replace the traditional dilated convolution in the decoder block. The EAN
achieved a pixel-level accuracy >84% and an accuracy of ~70% for the minority
classes, showing improved performance compared to existing architectures. In
addition, we introduce the CORAL (Correlation Alignment) method to the EAN to
create an unsupervised deep domain adaptation network (EAN-DDA) for the
classification of seismic reflections from F3 and Penobscot, to demonstrate
possible approaches when labelled data are unavailable. Maximum class accuracy
achieved was ~99% for class 2 of Penobscot, with an overall accuracy>50%. Taken
together, the EAN-DDA has the potential to classify target domain seismic
facies classes with high accuracy.

    

### [[2012.08101] Detecting and Adapting to Irregular Distribution Shifts in Bayesian Online Learning](http://arxiv.org/abs/2012.08101)


  We consider the problem of online learning in the presence of distribution
shifts that occur at an unknown rate and of unknown intensity. We derive a new
Bayesian online inference approach to simultaneously infer these distribution
shifts and adapt the model to the detected changes by integrating ideas from
change point detection, switching dynamical systems, and Bayesian online
learning. Using a binary 'change variable,' we construct an informative prior
such that--if a change is detected--the model partially erases the information
of past model updates by tempering to facilitate adaptation to the new data
distribution. Furthermore, the approach uses beam search to track multiple
change-point hypotheses and selects the most probable one in hindsight. Our
proposed method is model-agnostic, applicable in both supervised and
unsupervised learning settings, suitable for an environment of concept drifts
or covariate drifts, and yields improvements over state-of-the-art Bayesian
online learning approaches.

    

### [[2012.11207] On Success and Simplicity: A Second Look at Transferable Targeted Attacks](http://arxiv.org/abs/2012.11207)


  Achieving transferability of targeted attacks is reputed to be remarkably
difficult. Currently, state-of-the-art approaches are resource-intensive
because they necessitate training model(s) for each target class with
additional data. In our investigation, we find, however, that simple
transferable attacks which require neither additional data nor model training
can achieve surprisingly high targeted transferability. This insight has been
overlooked until now, mainly due to the widespread practice of unreasonably
restricting attack optimization to a limited number of iterations. In
particular, we, for the first time, identify that a simple logit loss can yield
competitive results with the state of the arts. Our analysis spans a variety of
transfer settings, especially including three new, realistic settings: an
ensemble transfer setting with little model similarity, a worse-case setting
with low-ranked target classes, and also a real-world attack against the Google
Cloud Vision API. Results in these new settings demonstrate that the commonly
adopted, easy settings cannot fully reveal the actual properties of different
attacks and may cause misleading comparisons. We also show the usefulness of
the simple logit loss for generating targeted universal adversarial
perturbations in a data-free and training-free manner. Overall, the aim of our
analysis is to inspire a more meaningful evaluation on targeted
transferability. Code is available at
this https URL


### [[2102.01854] Provably Secure Federated Learning against Malicious Clients](http://arxiv.org/abs/2102.01854)


  Federated learning enables clients to collaboratively learn a shared global
model without sharing their local training data with a cloud server. However,
malicious clients can corrupt the global model to predict incorrect labels for
testing examples. Existing defenses against malicious clients leverage
Byzantine-robust federated learning methods. However, these methods cannot
provably guarantee that the predicted label for a testing example is not
affected by malicious clients. We bridge this gap via ensemble federated
learning. In particular, given any base federated learning algorithm, we use
the algorithm to learn multiple global models, each of which is learnt using a
randomly selected subset of clients. When predicting the label of a testing
example, we take majority vote among the global models. We show that our
ensemble federated learning with any base federated learning algorithm is
provably secure against malicious clients. Specifically, the label predicted by
our ensemble global model for a testing example is provably not affected by a
bounded number of malicious clients. Moreover, we show that our derived bound
is tight. We evaluate our method on MNIST and Human Activity Recognition
datasets. For instance, our method can achieve a certified accuracy of 88% on
MNIST when 20 out of 1,000 clients are malicious.

    

### [[2102.02956] DetectorGuard: Provably Securing Object Detectors against Localized Patch Hiding Attacks](http://arxiv.org/abs/2102.02956)


  State-of-the-art object detectors are vulnerable to localized patch hiding
attacks, where an adversary introduces a small adversarial patch to make
detectors miss the detection of salient objects. The patch attacker can carry
out a physical-world attack by printing and attaching an adversarial patch to
the victim object. In this paper, we propose DetectorGuard as the first general
framework for building provably robust object detectors against localized patch
hiding attacks. DetectorGuard is inspired by recent advancements in robust
image classification research; we ask: can we adapt robust image classifiers
for robust object detection? Unfortunately, due to their task difference, an
object detector naively adapted from a robust image classifier 1) may not
necessarily be robust in the adversarial setting or 2) even maintain decent
performance in the clean setting. To build a high-performance robust object
detector, we propose an objectness explaining strategy: we adapt a robust image
classifier to predict objectness for every image location and then explain each
objectness using the bounding boxes predicted by a conventional object
detector. If all objectness is well explained, we output the predictions made
by the conventional object detector; otherwise, we issue an attack alert.
Notably, 1) in the adversarial setting, we formally prove the end-to-end
robustness of DetectorGuard on certified objects, i.e., it either detects the
object or triggers an alert, against any patch hiding attacker within our
threat model; 2) in the clean setting, we have almost the same performance as
state-of-the-art object detectors. Our evaluation on the PASCAL VOC, MS COCO,
and KITTI datasets further demonstrates that DetectorGuard achieves the first
provable robustness against localized patch hiding attacks at a negligible cost
(<1%) of clean performance.

    

### [[2102.04259] Concentration of Non-Isotropic Random Tensors with Applications to Learning and Empirical Risk Minimization](http://arxiv.org/abs/2102.04259)


  Dimension is an inherent bottleneck to some modern learning tasks, where
optimization methods suffer from the size of the data. In this paper, we study
non-isotropic distributions of data and develop tools that aim at reducing
these dimensional costs by a dependency on an effective dimension rather than
the ambient one. Based on non-asymptotic estimates of the metric entropy of
ellipsoids -- that prove to generalize to infinite dimensions -- and on a
chaining argument, our uniform concentration bounds involve an effective
dimension instead of the global dimension, improving over existing results. We
show the importance of taking advantage of non-isotropic properties in learning
problems with the following applications: i) we improve state-of-the-art
results in statistical preconditioning for communication-efficient distributed
optimization, ii) we introduce a non-isotropic randomized smoothing for
non-smooth optimization. Both applications cover a class of functions that
encompasses empirical risk minization (ERM) for linear models.

    

### [[2102.04426] Arbitrary Conditional Distributions with Energy](http://arxiv.org/abs/2102.04426)


  Modeling distributions of covariates, or density estimation, is a core
challenge in unsupervised learning. However, the majority of work only
considers the joint distribution, which has limited utility in practical
situations. A more general and useful problem is arbitrary conditional density
estimation, which aims to model any possible conditional distribution over a
set of covariates, reflecting the more realistic setting of inference based on
prior knowledge. We propose a novel method, Arbitrary Conditioning with Energy
(ACE), that can simultaneously estimate the distribution $p(\mathbf{x}_u \mid
\mathbf{x}_o)$ for all possible subsets of unobserved features $\mathbf{x}_u$
and observed features $\mathbf{x}_o$. ACE is designed to avoid unnecessary bias
and complexity -- we specify densities with a highly expressive energy function
and reduce the problem to only learning one-dimensional conditionals (from
which more complex distributions can be recovered during inference). This
results in an approach that is both simpler and higher-performing than prior
methods. We show that ACE achieves state-of-the-art for arbitrary conditional
likelihood estimation and data imputation on standard benchmarks.

    

### [[2102.06062] Deep Learning with Label Differential Privacy](http://arxiv.org/abs/2102.06062)


  The Randomized Response (RR) algorithm is a classical technique to improve
robustness in survey aggregation, and has been widely adopted in applications
with differential privacy guarantees. We propose a novel algorithm, Randomized
Response with Prior (RRWithPrior), which can provide more accurate results
while maintaining the same level of privacy guaranteed by RR. We then apply
RRWithPrior to learn neural networks with label differential privacy (LabelDP),
and show that when only the label needs to be protected, the model performance
can be significantly improved over the previous state-of-the-art private
baselines. Moreover, we study different ways to obtain priors, which when used
with RRWithPrior can additionally improve the model performance, further
reducing the accuracy gap between private and non-private models. We complement
the empirical results with theoretical analysis showing that LabelDP is
provably easier than protecting both the inputs and labels.

    

### [[2102.08098] GradInit: Learning to Initialize Neural Networks for Stable and Efficient Training](http://arxiv.org/abs/2102.08098)


  Innovations in neural architectures have fostered significant breakthroughs
in language modeling and computer vision. Unfortunately, novel architectures
often result in challenging hyper-parameter choices and training instability if
the network parameters are not properly initialized. A number of
architecture-specific initialization schemes have been proposed, but these
schemes are not always portable to new architectures. This paper presents
GradInit, an automated and architecture agnostic method for initializing neural
networks. GradInit is based on a simple heuristic; the norm of each network
layer is adjusted so that a single step of SGD or Adam with prescribed
hyperparameters results in the smallest possible loss value. This adjustment is
done by introducing a scalar multiplier variable in front of each parameter
block, and then optimizing these variables using a simple numerical scheme.
GradInit accelerates the convergence and test performance of many convolutional
architectures, both with or without skip connections, and even without
normalization layers. It also improves the stability of the original
Transformer architecture for machine translation, enabling training it without
learning rate warmup using either Adam or SGD under a wide range of learning
rates and momentum coefficients. Code is available at
this https URL.

    

### [[2102.08473] COCO-LM: Correcting and Contrasting Text Sequences for Language Model Pretraining](http://arxiv.org/abs/2102.08473)


  We present a self-supervised learning framework, COCO-LM, that pretrains
Language Models by COrrecting and COntrasting corrupted text sequences.
Following ELECTRA-style pretraining, COCO-LM employs an auxiliary language
model to corrupt text sequences, upon which it constructs two new tasks for
pretraining the main model. The first token-level task, Corrective Language
Modeling, is to detect and correct tokens replaced by the auxiliary model, in
order to better capture token-level semantics. The second sequence-level task,
Sequence Contrastive Learning, is to align text sequences originated from the
same source input while ensuring uniformity in the representation space.
Experiments on GLUE and SQuAD demonstrate that COCO-LM not only outperforms
recent state-of-the-art pretrained models in accuracy, but also improves
pretraining efficiency. It achieves the MNLI accuracy of ELECTRA with 50% of
its pretraining GPU hours. With the same pretraining steps of standard
base/large-sized models, COCO-LM outperforms the previous best models by 1+
GLUE average points.

    

### [[2102.11069] A PAC-Bayes Analysis of Adversarial Robustness](http://arxiv.org/abs/2102.11069)


  We propose the first general PAC-Bayesian generalization bounds for
adversarial robustness, that estimate, at test time, how much a model will be
invariant to imperceptible perturbations in the input. Instead of deriving a
worst-case analysis of the risk of a hypothesis over all the possible
perturbations, we leverage the PAC-Bayesian framework to bound the averaged
risk on the perturbations for majority votes (over the whole class of
hypotheses). Our theoretically founded analysis has the advantage to provide
general bounds (i) that are valid for any kind of attacks (i.e., the
adversarial attacks), (ii) that are tight thanks to the PAC-Bayesian framework,
(iii) that can be directly minimized during the learning phase to obtain a
robust model on different attacks at test time.

    

### [[2102.11860] Automated Discovery of Adaptive Attacks on Adversarial Defenses](http://arxiv.org/abs/2102.11860)


  Reliable evaluation of adversarial defenses is a challenging task, currently
limited to an expert who manually crafts attacks that exploit the defense's
inner workings or approaches based on an ensemble of fixed attacks, none of
which may be effective for the specific defense at hand. Our key observation is
that adaptive attacks are composed of reusable building blocks that can be
formalized in a search space and used to automatically discover attacks for
unknown defenses. We evaluated our approach on 24 adversarial defenses and show
that it outperforms AutoAttack, the current state-of-the-art tool for reliable
evaluation of adversarial defenses: our tool discovered significantly stronger
attacks by producing 3.0\%-50.8\% additional adversarial examples for 10
models, while obtaining attacks with slightly stronger or similar strength for
the remaining models.

    

### [[2102.12033] Self-Diagnosing GAN: Diagnosing Underrepresented Samples in Generative Adversarial Networks](http://arxiv.org/abs/2102.12033)


  Despite remarkable performance in producing realistic samples, Generative
Adversarial Networks (GANs) often produce low-quality samples near low-density
regions of the data manifold, e.g., samples of minor groups. Many techniques
have been developed to improve the quality of generated samples, either by
post-processing generated samples or by pre-processing the empirical data
distribution, but at the cost of reduced diversity. To promote diversity in
sample generation without degrading the overall quality, we propose a simple
yet effective method to diagnose and emphasize underrepresented samples during
training of a GAN. The main idea is to use the statistics of the discrepancy
between the data distribution and the model distribution at each data instance.
Based on the observation that the underrepresented samples have a high average
discrepancy or high variability in discrepancy, we propose a method to
emphasize those samples during training of a GAN. Our experimental results
demonstrate that the proposed method improves GAN performance on various
datasets, and it is especially effective in improving the quality and diversity
of sample generation for minor groups.

    

### [[2102.13156] Physics-Integrated Variational Autoencoders for Robust and Interpretable Generative Modeling](http://arxiv.org/abs/2102.13156)


  Integrating physics models within machine learning models holds considerable
promise toward learning robust models with improved interpretability and
abilities to extrapolate. In this work, we focus on the integration of
incomplete physics models into deep generative models. In particular, we
introduce an architecture of variational autoencoders (VAEs) in which a part of
the latent space is grounded by physics. A key technical challenge is to strike
a balance between the incomplete physics and trainable components such as
neural networks for ensuring that the physics part is used in a meaningful
manner. To this end, we propose a regularized learning method that controls the
effect of the trainable components and preserves the semantics of the
physics-based latent variables as intended. We not only demonstrate generative
performance improvements over a set of synthetic and real-world datasets, but
we also show that we learn robust models that can consistently extrapolate
beyond the training distribution in a meaningful manner. Moreover, we show that
we can control the generative process in an interpretable manner.

    

### [[2102.13380] A novel notion of barycenter for probability distributions based on optimal weak mass transport](http://arxiv.org/abs/2102.13380)


  We introduce weak barycenters of a family of probability distributions, based
on the recently developed notion of optimal weak transport of mass by Gozlanet
al. (2017) and Backhoff-Veraguas et al. (2020). We provide a theoretical
analysis of this object and discuss its interpretation in the light of convex
ordering between probability measures. In particular, we show that, rather than
averaging the input distributions in a geometric way (as the Wasserstein
barycenter based on classic optimal transport does) weak barycenters extract
common geometric information shared by all the input distributions, encoded as
a latent random variable that underlies all of them. We also provide an
iterative algorithm to compute a weak barycenter for a finite family of input
distributions, and a stochastic algorithm that computes them for arbitrary
populations of laws. The latter approach is particularly well suited for the
streaming setting, i.e., when distributions are observed sequentially. The
notion of weak barycenter and our approaches to compute it are illustrated on
synthetic examples, validated on 2D real-world data and compared to standard
Wasserstein barycenters.

    

### [[2102.13647] Beware of the Simulated DAG! Causal Discovery Benchmarks May Be Easy To Game](http://arxiv.org/abs/2102.13647)


  Simulated DAG models may exhibit properties that, perhaps inadvertently,
render their structure identifiable and unexpectedly affect structure learning
algorithms. Here, we show that marginal variance tends to increase along the
causal order for generically sampled additive noise models. We introduce
varsortability as a measure of the agreement between the order of increasing
marginal variance and the causal order. For commonly sampled graphs and model
parameters, we show that the remarkable performance of some continuous
structure learning algorithms can be explained by high varsortability and
matched by a simple baseline method. Yet, this performance may not transfer to
real-world data where varsortability may be moderate or dependent on the choice
of measurement scales. On standardized data, the same algorithms fail to
identify the ground-truth DAG or its Markov equivalence class. While
standardization removes the pattern in marginal variance, we show that data
generating processes that incur high varsortability also leave a distinct
covariance pattern that may be exploited even after standardization. Our
findings challenge the significance of generic benchmarks with independently
drawn parameters. The code is available at
this https URL.

    

### [[2103.08902] Differentiable Learning Under Triage](http://arxiv.org/abs/2103.08902)


  Multiple lines of evidence suggest that predictive models may benefit from
algorithmic triage. Under algorithmic triage, a predictive model does not
predict all instances but instead defers some of them to human experts.
However, the interplay between the prediction accuracy of the model and the
human experts under algorithmic triage is not well understood. In this work, we
start by formally characterizing under which circumstances a predictive model
may benefit from algorithmic triage. In doing so, we also demonstrate that
models trained for full automation may be suboptimal under triage. Then, given
any model and desired level of triage, we show that the optimal triage policy
is a deterministic threshold rule in which triage decisions are derived
deterministically by thresholding the difference between the model and human
errors on a per-instance level. Building upon these results, we introduce a
practical gradient-based algorithm that is guaranteed to find a sequence of
triage policies and predictive models of increasing performance. Experiments on
a wide variety of supervised learning tasks using synthetic and real data from
two important applications -- content moderation and scientific discovery --
illustrate our theoretical results and show that the models and triage policies
provided by our gradient-based algorithm outperform those provided by several
competitive baselines.

    

### [[2103.12513] On gray-box modeling for virtual flow metering](http://arxiv.org/abs/2103.12513)


  A virtual flow meter (VFM) enables continuous prediction of flow rates in
petroleum production systems. The predicted flow rates may aid the daily
control and optimization of a petroleum asset. Gray-box modeling is an approach
that combines mechanistic and data-driven modeling. The objective is to create
a computationally feasible VFM for use in real-time applications, with high
prediction accuracy and scientifically consistent behavior. This article
investigates five different gray-box model types in an industrial case study
using real, historical production data from 10 petroleum wells, spanning at
most four years of production. The results are diverse with an oil flow rate
prediction error in the range of 1.8%-40.6%. Further, the study casts light
upon the nontrivial task of balancing learning from both physics and data.
Consequently, providing general recommendations towards the suitability of
different hybrid models is challenging. Nevertheless, the results are promising
and indicate that gray-box VFMs may reduce the prediction error of a
mechanistic VFM while remaining scientifically consistent. The findings
motivate further experimentation with gray-box VFM models and suggest several
future research directions to improve upon the performance and scientific
consistency.

    

### [[2103.14077] Nearly Horizon-Free Offline Reinforcement Learning](http://arxiv.org/abs/2103.14077)


  We revisit offline reinforcement learning on episodic time-homogeneous Markov
Decision Processes (MDP). For tabular MDP with $S$ states and $A$ actions, or
linear MDP with anchor points and feature dimension $d$, given the collected
$K$ episodes data with minimum visiting probability of (anchor) state-action
pairs $d_m$, we obtain nearly horizon $H$-free sample complexity bounds for
offline reinforcement learning when the total reward is upper bounded by $1$.
Specifically: 1. For offline policy evaluation, we obtain an
$\tilde{O}\left(\sqrt{\frac{1}{Kd_m}} \right)$ error bound for the plug-in
estimator, which matches the lower bound up to logarithmic factors and does not
have additional dependency on $\mathrm{poly}\left(H, S, A, d\right)$ in
higher-order term. 2.For offline policy optimization, we obtain an
$\tilde{O}\left(\sqrt{\frac{1}{Kd_m}} + \frac{\min(S, d)}{Kd_m}\right)$
sub-optimality gap for the empirical optimal policy, which approaches the lower
bound up to logarithmic factors and a high-order term, improving upon the best
known result by \cite{cui2020plug} that has additional $\mathrm{poly}\left(H,
S, d\right)$ factors in the main term. To the best of our knowledge, these are
the \emph{first} set of nearly horizon-free bounds for episodic
time-homogeneous offline tabular MDP and linear MDP with anchor points. Central
to our analysis is a simple yet effective recursion based method to bound a
``total variance'' term in the offline scenarios, which could be of individual
interest.

    

### [[2104.04646] DeepSITH: Efficient Learning via Decomposition of What and When Across Time Scales](http://arxiv.org/abs/2104.04646)


  Extracting temporal relationships over a range of scales is a hallmark of
human perception and cognition -- and thus it is a critical feature of machine
learning applied to real-world problems. Neural networks are either plagued by
the exploding/vanishing gradient problem in recurrent neural networks (RNNs) or
must adjust their parameters to learn the relevant time scales (e.g., in
LSTMs). This paper introduces DeepSITH, a network comprising
biologically-inspired Scale-Invariant Temporal History (SITH) modules in series
with dense connections between layers. SITH modules respond to their inputs
with a geometrically-spaced set of time constants, enabling the DeepSITH
network to learn problems along a continuum of time-scales. We compare DeepSITH
to LSTMs and other recent RNNs on several time series prediction and decoding
tasks. DeepSITH achieves state-of-the-art performance on these problems.

    

### [[2104.12112] Improved Analysis and Rates for Variance Reduction under Without-replacement Sampling Orders](http://arxiv.org/abs/2104.12112)


  When applying a stochastic algorithm, one must choose an order to draw
samples. The practical choices are without-replacement sampling orders, which
are empirically faster and more cache-friendly than uniform-iid-sampling but
often have inferior theoretical guarantees. Without-replacement sampling is
well understood only for SGD without variance reduction. In this paper, we will
improve the convergence analysis and rates of variance reduction under
without-replacement sampling orders for composite finite-sum minimization.
Our results are in two-folds. First, we develop a damped variant of Finito
called Prox-DFinito and establish its convergence rates with random
reshuffling, cyclic sampling, and shuffling-once, under both convex and
strongly convex scenarios. These rates match full-batch gradient descent and
are state-of-the-art compared to the existing results for without-replacement
sampling with variance-reduction. Second, our analysis can gauge how the cyclic
order will influence the rate of cyclic sampling and, thus, allows us to derive
the optimal fixed ordering. In the highly data-heterogeneous scenario,
Prox-DFinito with optimal cyclic sampling can attain a sample-size-independent
convergence rate, which, to our knowledge, is the first result that can match
with uniform-iid-sampling with variance reduction. We also propose a practical
method to discover the optimal cyclic ordering numerically.

    

### [[2104.14113] Regret Bounds for Gaussian-Process Optimization in Large Domains](http://arxiv.org/abs/2104.14113)


  The goal of this paper is to characterize Gaussian-Process optimization in
the setting where the function domain is large relative to the number of
admissible function evaluations, i.e., where it is impossible to find the
global optimum. We provide upper bounds on the suboptimality (Bayesian simple
regret) of the solution found by optimization strategies that are closely
related to the widely used expected improvement (EI) and upper confidence bound
(UCB) algorithms. These regret bounds illuminate the relationship between the
number of evaluations, the domain size (i.e. cardinality of finite domains /
Lipschitz constant of the covariance function in continuous domains), and the
optimality of the retrieved function value. In particular, we show that even
when the number of evaluations is far too small to find the global optimum, we
can find nontrivial function values (e.g. values that achieve a certain ratio
with the optimal value).

    

### [[2105.08195] Parallel Bayesian Optimization of Multiple Noisy Objectives with Expected Hypervolume Improvement](http://arxiv.org/abs/2105.08195)


  Optimizing multiple competing black-box objectives is a challenging problem
in many fields, including science, engineering, and machine learning.
Multi-objective Bayesian optimization (MOBO) is a sample-efficient approach for
identifying the optimal trade-offs between the objectives. However, many
existing methods perform poorly when the observations are corrupted by noise.
We propose a novel acquisition function, NEHVI, that overcomes this important
practical limitation by applying a Bayesian treatment to the popular expected
hypervolume improvement (EHVI) criterion and integrating over this uncertainty
in the Pareto frontier. We argue that, even in the noiseless setting,
generating multiple candidates in parallel is an incarnation of EHVI with
uncertainty in the Pareto frontier and therefore can be addressed using the
same underlying technique. Through this lens, we derive a natural parallel
variant, $q$NEHVI, that reduces computational complexity of parallel EHVI from
exponential to polynomial with respect to the batch size. $q$NEHVI is one-step
Bayes-optimal for hypervolume maximization in both noisy and noiseless
environments, and we show that it can be optimized effectively with
gradient-based methods via sample average approximation. Empirically, we
demonstrate not only that $q$NEHVI is substantially more robust to observation
noise than existing MOBO approaches, but also that it achieves state-of-the-art
optimization performance and competitive wall-times in large-batch
environments.

    

### [[2105.13504] Lattice partition recovery with dyadic CART](http://arxiv.org/abs/2105.13504)


  We study piece-wise constant signals corrupted by additive Gaussian noise
over a $d$-dimensional lattice. Data of this form naturally arise in a host of
applications, and the tasks of signal detection or testing, de-noising and
estimation have been studied extensively in the statistical and signal
processing literature. In this paper we consider instead the problem of
partition recovery, i.e.~of estimating the partition of the lattice induced by
the constancy regions of the unknown signal, using the
computationally-efficient dyadic classification and regression tree (DCART)
methodology proposed by \citep{donoho1997cart}. We prove that, under
appropriate regularity conditions on the shape of the partition elements, a
DCART-based procedure consistently estimates the underlying partition at a rate
of order $\sigma^2 k^* \log (N)/\kappa^2$, where $k^*$ is the minimal number of
rectangular sub-graphs obtained using recursive dyadic partitions supporting
the signal partition, $\sigma^2$ is the noise variance, $\kappa$ is the minimal
magnitude of the signal difference among contiguous elements of the partition
and $N$ is the size of the lattice. Furthermore, under stronger assumptions,
our method attains a sharper estimation error of order
$\sigma^2\log(N)/\kappa^2$, independent of $k^*$, which we show to be minimax
rate optimal. Our theoretical guarantees further extend to the partition
estimator based on the optimal regression tree estimator (ORT) of
\cite{chatterjee2019adaptive} and to the one obtained through an NP-hard
exhaustive search method. We corroborate our theoretical findings and the
effectiveness of DCART for partition recovery in simulations.

    

### [[2105.13655] Scheduling Jobs with Stochastic Holding Costs](http://arxiv.org/abs/2105.13655)


  This paper proposes a learning and scheduling algorithm to minimize the
expected cumulative holding cost incurred by jobs, where statistical parameters
defining their individual holding costs are unknown a priori. In each time
slot, the server can process a job while receiving the realized random holding
costs of the jobs remaining in the system. Our algorithm is a learning-based
variant of the $c\mu$ rule for scheduling: it starts with a preemption period
of fixed length which serves as a learning phase, and after accumulating enough
data about individual jobs, it switches to nonpreemptive scheduling mode. The
algorithm is designed to handle instances with large or small gaps in jobs'
parameters and achieves near-optimal performance guarantees. The performance of
our algorithm is captured by its regret, where the benchmark is the minimum
possible cost attained when the statistical parameters of jobs are fully known.
We prove upper bounds on the regret of our algorithm, and we derive a regret
lower bound that is almost matching the proposed upper bounds. Our numerical
results demonstrate the effectiveness of our algorithm and show that our
theoretical regret analysis is nearly tight.

    

### [[2105.13831] Implicit Regularization in Matrix Sensing via Mirror Descent](http://arxiv.org/abs/2105.13831)


  We study discrete-time mirror descent applied to the unregularized empirical
risk in matrix sensing. In both the general case of rectangular matrices and
the particular case of positive semidefinite matrices, a simple potential-based
analysis in terms of the Bregman divergence allows us to establish convergence
of mirror descent -- with different choices of the mirror maps -- to a matrix
that, among all global minimizers of the empirical risk, minimizes a quantity
explicitly related to the nuclear norm, the Frobenius norm, and the von Neumann
entropy. In both cases, this characterization implies that mirror descent, a
first-order algorithm minimizing the unregularized empirical risk, recovers
low-rank matrices under the same set of assumptions that are sufficient to
guarantee recovery for nuclear-norm minimization. When the sensing matrices are
symmetric and commute, we show that gradient descent with full-rank factorized
parametrization is a first-order approximation to mirror descent, in which case
we obtain an explicit characterization of the implicit bias of gradient flow as
a by-product.

    

### [[2105.14084] Support vector machines and linear regression coincide with very high-dimensional features](http://arxiv.org/abs/2105.14084)


  The support vector machine (SVM) and minimum Euclidean norm least squares
regression are two fundamentally different approaches to fitting linear models,
but they have recently been connected in models for very high-dimensional data
through a phenomenon of support vector proliferation, where every training
example used to fit an SVM becomes a support vector. In this paper, we explore
the generality of this phenomenon and make the following contributions. First,
we prove a super-linear lower bound on the dimension (in terms of sample size)
required for support vector proliferation in independent feature models,
matching the upper bounds from previous works. We further identify a sharp
phase transition in Gaussian feature models, bound the width of this
transition, and give experimental support for its universality. Finally, we
hypothesize that this phase transition occurs only in much higher-dimensional
settings in the $\ell_1$ variant of the SVM, and we present a new geometric
characterization of the problem that may elucidate this phenomenon for the
general $\ell_p$ case.

    

### [[2106.00661] Reward is enough for convex MDPs](http://arxiv.org/abs/2106.00661)


  Maximising a cumulative reward function that is Markov and stationary, i.e.,
defined over state-action pairs and independent of time, is sufficient to
capture many kinds of goals in a Markov decision process (MDP). However, not
all goals can be captured in this manner. In this paper we study convex MDPs in
which goals are expressed as convex functions of the stationary distribution
and show that they cannot be formulated using stationary reward functions.
Convex MDPs generalize the standard reinforcement learning (RL) problem
formulation to a larger framework that includes many supervised and
unsupervised RL problems, such as apprenticeship learning, constrained MDPs,
and so-called `pure exploration'. Our approach is to reformulate the convex MDP
problem as a min-max game involving policy and cost (negative reward)
`players', using Fenchel duality. We propose a meta-algorithm for solving this
problem and show that it unifies many existing algorithms in the literature.

    

### [[2106.00666] You Only Look at One Sequence: Rethinking Transformer in Vision through Object Detection](http://arxiv.org/abs/2106.00666)


  Can Transformer perform 2D object- and region-level recognition from a pure
sequence-to-sequence perspective with minimal knowledge about the 2D spatial
structure? To answer this question, we present You Only Look at One Sequence
(YOLOS), a series of object detection models based on the vanilla Vision
Transformer with the fewest possible modifications, region priors, as well as
inductive biases of the target task. We find that YOLOS pre-trained on the
mid-sized ImageNet-1k dataset only can already achieve quite competitive
performance on the challenging COCO object detection benchmark, e.g.,
YOLOS-Base directly adopted from BERT-Base architecture can obtain 42.0 box AP
on COCO val. We also discuss the impacts as well as limitations of current
pre-train schemes and model scaling strategies for Transformer in vision
through YOLOS. Code and pre-trained models are available at
this https URL.

    

### [[2106.01798] Implicit MLE: Backpropagating Through Discrete Exponential Family Distributions](http://arxiv.org/abs/2106.01798)


  Combining discrete probability distributions and combinatorial optimization
problems with neural network components has numerous applications but poses
several challenges. We propose Implicit Maximum Likelihood Estimation (I-MLE),
a framework for end-to-end learning of models combining discrete exponential
family distributions and differentiable neural components. I-MLE is widely
applicable as it only requires the ability to compute the most probable states
and does not rely on smooth relaxations. The framework encompasses several
approaches such as perturbation-based implicit differentiation and recent
methods to differentiate through black-box combinatorial solvers. We introduce
a novel class of noise distributions for approximating marginals via
perturb-and-MAP. Moreover, we show that I-MLE simplifies to maximum likelihood
estimation when used in some recently studied learning settings that involve
combinatorial solvers. Experiments on several datasets suggest that I-MLE is
competitive with and often outperforms existing approaches which rely on
problem-specific relaxations.

    

### [[2106.02997] Causal Abstractions of Neural Networks](http://arxiv.org/abs/2106.02997)


  Structural analysis methods (e.g., probing and feature attribution) are
increasingly important tools for neural network analysis. We propose a new
structural analysis method grounded in a formal theory of causal abstraction
that provides rich characterizations of model-internal representations and
their roles in input/output behavior. In this method, neural representations
are aligned with variables in interpretable causal models, and then interchange
interventions are used to experimentally verify that the neural representations
have the causal properties of their aligned variables. We apply this method in
a case study to analyze neural models trained on Multiply Quantified Natural
Language Inference (MQNLI) corpus, a highly complex NLI dataset that was
constructed with a tree-structured natural logic causal model. We discover that
a BERT-based model with state-of-the-art performance successfully realizes
parts of the natural logic model's causal structure, whereas a simpler baseline
model fails to show any such structure, demonstrating that BERT representations
encode the compositional structure of MQNLI.

    

### [[2106.03257] Structured Reordering for Modeling Latent Alignments in Sequence Transduction](http://arxiv.org/abs/2106.03257)


  Despite success in many domains, neural models struggle in settings where
train and test examples are drawn from different distributions. In particular,
in contrast to humans, conventional sequence-to-sequence (seq2seq) models fail
to generalize systematically, i.e., interpret sentences representing novel
combinations of concepts (e.g., text segments) seen in training. Traditional
grammar formalisms excel in such settings by implicitly encoding alignments
between input and output segments, but are hard to scale and maintain. Instead
of engineering a grammar, we directly model segment-to-segment alignments as
discrete structured latent variables within a neural seq2seq model. To
efficiently explore the large space of alignments, we introduce a reorder-first
align-later framework whose central component is a neural reordering module
producing {\it separable} permutations. We present an efficient dynamic
programming algorithm performing exact marginal inference of separable
permutations, and, thus, enabling end-to-end differentiable training of our
model. The resulting seq2seq model exhibits better systematic generalization
than standard models on synthetic problems and NLP tasks (i.e., semantic
parsing and machine translation).

    

### [[2106.03542] How Tight Can PAC-Bayes be in the Small Data Regime?](http://arxiv.org/abs/2106.03542)


  In this paper, we investigate the question: Given a small number of
datapoints, for example N = 30, how tight can PAC-Bayes and test set bounds be
made? For such small datasets, test set bounds adversely affect generalisation
performance by withholding data from the training procedure. In this setting,
PAC-Bayes bounds are especially attractive, due to their ability to use all the
data to simultaneously learn a posterior and bound its generalisation risk. We
focus on the case of i.i.d. data with a bounded loss and consider the generic
PAC-Bayes theorem of Germain et al. While their theorem is known to recover
many existing PAC-Bayes bounds, it is unclear what the tightest bound derivable
from their framework is. For a fixed learning algorithm and dataset, we show
that the tightest possible bound coincides with a bound considered by Catoni;
and, in the more natural case of distributions over datasets, we establish a
lower bound on the best bound achievable in expectation. Interestingly, this
lower bound recovers the Chernoff test set bound if the posterior is equal to
the prior. Moreover, to illustrate how tight these bounds can be, we study
synthetic one-dimensional classification tasks in which it is feasible to
meta-learn both the prior and the form of the bound to numerically optimise for
the tightest bounds possible. We find that in this simple, controlled scenario,
PAC-Bayes bounds are competitive with comparable, commonly used Chernoff test
set bounds. However, the sharpest test set bounds still lead to better
guarantees on the generalisation error than the PAC-Bayes bounds we consider.

    

### [[2106.03721] OoD-Bench: Benchmarking and Understanding Out-of-Distribution Generalization Datasets and Algorithms](http://arxiv.org/abs/2106.03721)


  Deep learning has achieved tremendous success with independent and
identically distributed (i.i.d.) data. However, the performance of neural
networks often degenerates drastically when encountering out-of-distribution
(OoD) data, i.e., training and test data are sampled from different
distributions. While a plethora of algorithms has been proposed to deal with
OoD generalization, our understanding of the data used to train and evaluate
these algorithms remains stagnant. In this work, we position existing datasets
and algorithms from various research areas (e.g., domain generalization, stable
learning, invariant risk minimization) seemingly unconnected into the same
coherent picture. First, we identify and measure two distinct kinds of
distribution shifts that are ubiquitous in various datasets. Next, we compare
various OoD generalization algorithms with a new benchmark dominated by the two
distribution shifts. Through extensive experiments, we show that existing OoD
algorithms that outperform empirical risk minimization on one distribution
shift usually have limitations on the other distribution shift. The new
benchmark may serve as a strong foothold that can be resorted to by future OoD
generalization research.

    

### [[2106.03827] Stateful Strategic Regression](http://arxiv.org/abs/2106.03827)


  Automated decision-making tools increasingly assess individuals to determine
if they qualify for high-stakes opportunities. A recent line of research
investigates how strategic agents may respond to such scoring tools to receive
favorable assessments. While prior work has focused on the short-term strategic
interactions between a decision-making institution (modeled as a principal) and
individual decision-subjects (modeled as agents), we investigate interactions
spanning multiple time-steps. In particular, we consider settings in which the
agent's effort investment today can accumulate over time in the form of an
internal state - impacting both his future rewards and that of the principal.
We characterize the Stackelberg equilibrium of the resulting game and provide
novel algorithms for computing it. Our analysis reveals several intriguing
insights about the role of multiple interactions in shaping the game's outcome:
First, we establish that in our stateful setting, the class of all linear
assessment policies remains as powerful as the larger class of all monotonic
assessment policies. While recovering the principal's optimal policy requires
solving a non-convex optimization problem, we provide polynomial-time
algorithms for recovering both the principal and agent's optimal policies under
common assumptions about the process by which effort investments convert to
observable features. Most importantly, we show that with multiple rounds of
interaction at her disposal, the principal is more effective at incentivizing
the agent to accumulate effort in her desired direction. Our work addresses
several critical gaps in the growing literature on the societal impacts of
automated decision-making - by focusing on longer time horizons and accounting
for the compounding nature of decisions individuals receive over time.

    

### [[2106.03893] Rethinking Graph Transformers with Spectral Attention](http://arxiv.org/abs/2106.03893)


  In recent years, the Transformer architecture has proven to be very
successful in sequence processing, but its application to other data
structures, such as graphs, has remained limited due to the difficulty of
properly defining positions. Here, we present the $\textit{Spectral Attention
Network}$ (SAN), which uses a learned positional encoding (LPE) that can take
advantage of the full Laplacian spectrum to learn the position of each node in
a given graph. This LPE is then added to the node features of the graph and
passed to a fully-connected Transformer. By leveraging the full spectrum of the
Laplacian, our model is theoretically powerful in distinguishing graphs, and
can better detect similar sub-structures from their resonance. Further, by
fully connecting the graph, the Transformer does not suffer from
over-squashing, an information bottleneck of most GNNs, and enables better
modeling of physical phenomenons such as heat transfer and electric
interaction. When tested empirically on a set of 4 standard datasets, our model
performs on par or better than state-of-the-art GNNs, and outperforms any
attention-based model by a wide margin, becoming the first fully-connected
architecture to perform well on graph benchmarks.

    

### [[2106.04013] The Future is Log-Gaussian: ResNets and Their Infinite-Depth-and-Width Limit at Initialization](http://arxiv.org/abs/2106.04013)


  Theoretical results show that neural networks can be approximated by Gaussian
processes in the infinite-width limit. However, for fully connected networks,
it has been previously shown that for any fixed network width, $n$, the
Gaussian approximation gets worse as the network depth, $d$, increases. Given
that modern networks are deep, this raises the question of how well modern
architectures, like ResNets, are captured by the infinite-width limit. To
provide a better approximation, we study ReLU ResNets in the
infinite-depth-and-width limit, where both depth and width tend to infinity as
their ratio, $d/n$, remains constant. In contrast to the Gaussian
infinite-width limit, we show theoretically that the network exhibits
log-Gaussian behaviour at initialization in the infinite-depth-and-width limit,
with parameters depending on the ratio $d/n$. Using Monte Carlo simulations, we
demonstrate that even basic properties of standard ResNet architectures are
poorly captured by the Gaussian limit, but remarkably well captured by our
log-Gaussian limit. Moreover, our analysis reveals that ReLU ResNets at
initialization are hypoactivated: fewer than half of the ReLUs are activated.
Additionally, we calculate the interlayer correlations, which have the effect
of exponentially increasing the variance of the network output. Based on our
analysis, we introduce Balanced ResNets, a simple architecture modification,
which eliminates hypoactivation and interlayer correlations and is more
amenable to theoretical analysis.

    

### [[2106.04152] PlayVirtual: Augmenting Cycle-Consistent Virtual Trajectories for Reinforcement Learning](http://arxiv.org/abs/2106.04152)


  Learning good feature representations is important for deep reinforcement
learning (RL). However, with limited experience, RL often suffers from data
inefficiency for training. For un-experienced or less-experienced trajectories
(i.e., state-action sequences), the lack of data limits the use of them for
better feature learning. In this work, we propose a novel method, dubbed
PlayVirtual, which augments cycle-consistent virtual trajectories to enhance
the data efficiency for RL feature representation learning. Specifically,
PlayVirtual predicts future states in the latent space based on the current
state and action by a dynamics model and then predicts the previous states by a
backward dynamics model, which forms a trajectory cycle. Based on this, we
augment the actions to generate a large amount of virtual state-action
trajectories. Being free of groudtruth state supervision, we enforce a
trajectory to meet the cycle consistency constraint, which can significantly
enhance the data efficiency. We validate the effectiveness of our designs on
the Atari and DeepMind Control Suite benchmarks. Our method achieves the
state-of-the-art performance on both benchmarks.

    

### [[2106.04243] Parameter Inference with Bifurcation Diagrams](http://arxiv.org/abs/2106.04243)


  Estimation of parameters in differential equation models can be achieved by
applying learning algorithms to quantitative time-series data. However,
sometimes it is only possible to measure qualitative changes of a system in
response to a controlled condition. In dynamical systems theory, such change
points are known as bifurcations and lie on a function of the controlled
condition called the bifurcation diagram. In this work, we propose a
gradient-based approach for inferring the parameters of differential equations
that produce a user-specified bifurcation diagram. The cost function contains
an error term that is minimal when the model bifurcations match the specified
targets and a bifurcation measure which has gradients that push optimisers
towards bifurcating parameter regimes. The gradients can be computed without
the need to differentiate through the operations of the solver that was used to
compute the diagram. We demonstrate parameter inference with minimal models
which explore the space of saddle-node and pitchfork diagrams and the genetic
toggle switch from synthetic biology. Furthermore, the cost landscape allows us
to organise models in terms of topological and geometric equivalence.

    

### [[2106.04379] Learning Markov State Abstractions for Deep Reinforcement Learning](http://arxiv.org/abs/2106.04379)


  A fundamental assumption of reinforcement learning in Markov decision
processes (MDPs) is that the relevant decision process is, in fact, Markov.
However, when MDPs have rich observations, agents typically learn by way of an
abstract state representation, and such representations are not guaranteed to
preserve the Markov property. We introduce a novel set of conditions and prove
that they are sufficient for learning a Markov abstract state representation.
We then describe a practical training procedure that combines inverse model
estimation and temporal contrastive learning to learn an abstraction that
approximately satisfies these conditions. Our novel training objective is
compatible with both online and offline training: it does not require a reward
signal, but agents can capitalize on reward information when available. We
empirically evaluate our approach on a visual gridworld domain and a set of
continuous control benchmarks. Our approach learns representations that capture
the underlying structure of the domain and lead to improved sample efficiency
over state-of-the-art deep reinforcement learning with visual features -- often
matching or exceeding the performance achieved with hand-designed compact state
information.

    

### [[2106.04443] Robust Generalization despite Distribution Shift via Minimum Discriminating Information](http://arxiv.org/abs/2106.04443)


  Training models that perform well under distribution shifts is a central
challenge in machine learning. In this paper, we introduce a modeling framework
where, in addition to training data, we have partial structural knowledge of
the shifted test distribution. We employ the principle of minimum
discriminating information to embed the available prior knowledge, and use
distributionally robust optimization to account for uncertainty due to the
limited samples. By leveraging large deviation results, we obtain explicit
generalization bounds with respect to the unknown shifted distribution. Lastly,
we demonstrate the versatility of our framework by demonstrating it on two
rather distinct applications: (1) training classifiers on systematically biased
data and (2) off-policy evaluation in Markov Decision Processes.

    

### [[2106.04759] Communication-efficient SGD: From Local SGD to One-Shot Averaging](http://arxiv.org/abs/2106.04759)


  We consider speeding up stochastic gradient descent (SGD) by parallelizing it
across multiple workers. We assume the same data set is shared among $N$
workers, who can take SGD steps and coordinate with a central server. While it
is possible to obtain a linear reduction in the variance by averaging all the
stochastic gradients at every step, this requires a lot of communication
between the workers and the server, which can dramatically reduce the gains
from parallelism. The Local SGD method, proposed and analyzed in the earlier
literature, suggests machines should make many local steps between such
communications. While the initial analysis of Local SGD showed it needs $\Omega
( \sqrt{T} )$ communications for $T$ local gradient steps in order for the
error to scale proportionately to $1/(NT)$, this has been successively improved
in a string of papers, with the state of the art requiring $\Omega \left( N
\left( \mbox{ poly} (\log T) \right) \right)$ communications. In this paper, we
suggest a Local SGD scheme that communicates less overall by communicating less
frequently as the number of iterations grows. Our analysis shows that this can
achieve an error that scales as $1/(NT)$ with a number of communications that
is completely independent of $T$. In particular, we show that $\Omega(N)$
communications are sufficient. Empirical evidence suggests this bound is close
to tight as we further show that $\sqrt{N}$ or $N^{3/4}$ communications fail to
achieve linear speed-up in simulations. Moreover, we show that under mild
assumptions, the main of which is twice differentiability on any neighborhood
of the optimal solution, one-shot averaging which only uses a single round of
communication can also achieve the optimal convergence rate asymptotically.

    

### [[2106.04765] Predicting Deep Neural Network Generalization with Perturbation Response Curves](http://arxiv.org/abs/2106.04765)


  The field of Deep Learning is rich with empirical evidence of human-like
performance on a variety of prediction tasks. However, despite these successes,
the recent Predicting Generalization in Deep Learning (PGDL) NeurIPS 2020
competition suggests that there is a need for more robust and efficient
measures of network generalization. In this work, we propose a new framework
for evaluating the generalization capabilities of trained networks. We use
perturbation response (PR) curves that capture the accuracy change of a given
network as a function of varying levels of training sample perturbation. From
these PR curves, we derive novel statistics that capture generalization
capability. Specifically, we introduce two new measures for accurately
predicting generalization gaps: the Gi-score and Pal-score, which are inspired
by the Gini coefficient and Palma ratio (measures of income inequality), that
accurately predict generalization gaps. Using our framework applied to intra
and inter-class sample mixup, we attain better predictive scores than the
current state-of-the-art measures on a majority of tasks in the PGDL
competition. In addition, we show that our framework and the proposed
statistics can be used to capture to what extent a trained network is invariant
to a given parametric input transformation, such as rotation or translation.
Therefore, these generalization gap prediction statistics also provide a useful
means for selecting optimal network architectures and hyperparameters that are
invariant to a certain perturbation.

    

### [[2106.05445] Exploiting Local Convergence of Quasi-Newton Methods Globally: Adaptive Sample Size Approach](http://arxiv.org/abs/2106.05445)


  In this paper, we study the application of quasi-Newton methods for solving
empirical risk minimization (ERM) problems defined over a large dataset.
Traditional deterministic and stochastic quasi-Newton methods can be executed
to solve such problems; however, it is known that their global convergence rate
may not be better than first-order methods, and their local superlinear
convergence only appears towards the end of the learning process. In this
paper, we use an adaptive sample size scheme that exploits the superlinear
convergence of quasi-Newton methods globally and throughout the entire learning
process. The main idea of the proposed adaptive sample size algorithms is to
start with a small subset of data points and solve their corresponding ERM
problem within its statistical accuracy, and then enlarge the sample size
geometrically and use the optimal solution of the problem corresponding to the
smaller set as an initial point for solving the subsequent ERM problem with
more samples. We show that if the initial sample size is sufficiently large and
we use quasi-Newton methods to solve each subproblem, the subproblems can be
solved superlinearly fast (after at most three iterations), as we guarantee
that the iterates always stay within a neighborhood that quasi-Newton methods
converge superlinearly. Numerical experiments on various datasets confirm our
theoretical results and demonstrate the computational advantages of our method.

    

### [[2106.05480] Lower Bounds on Metropolized Sampling Methods for Well-Conditioned Distributions](http://arxiv.org/abs/2106.05480)


  We give lower bounds on the performance of two of the most popular sampling
methods in practice, the Metropolis-adjusted Langevin algorithm (MALA) and
multi-step Hamiltonian Monte Carlo (HMC) with a leapfrog integrator, when
applied to well-conditioned distributions. Our main result is a nearly-tight
lower bound of $\widetilde{\Omega}(\kappa d)$ on the mixing time of MALA from
an exponentially warm start, matching a line of algorithmic results up to
logarithmic factors and answering an open question of Chewi et. al. We also
show that a polynomial dependence on dimension is necessary for the relaxation
time of HMC under any number of leapfrog steps, and bound the gains achievable
by changing the step count. Our HMC analysis draws upon a novel connection
between leapfrog integration and Chebyshev polynomials, which may be of
independent interest.

    

### [[2106.05931] Score-based Generative Modeling in Latent Space](http://arxiv.org/abs/2106.05931)


  Score-based generative models (SGMs) have recently demonstrated impressive
results in terms of both sample quality and distribution coverage. However,
they are usually applied directly in data space and often require thousands of
network evaluations for sampling. Here, we propose the Latent Score-based
Generative Model (LSGM), a novel approach that trains SGMs in a latent space,
relying on the variational autoencoder framework. Moving from data to latent
space allows us to train more expressive generative models, apply SGMs to
non-continuous data, and learn smoother SGMs in a smaller space, resulting in
fewer network evaluations and faster sampling. To enable training LSGMs
end-to-end in a scalable and stable manner, we (i) introduce a new
score-matching objective suitable to the LSGM setting, (ii) propose a novel
parameterization of the score function that allows SGM to focus on the mismatch
of the target distribution with respect to a simple Normal one, and (iii)
analytically derive multiple techniques for variance reduction of the training
objective. LSGM obtains a state-of-the-art FID score of 2.10 on CIFAR-10,
outperforming all existing generative results on this dataset. On
CelebA-HQ-256, LSGM is on a par with previous SGMs in sample quality while
outperforming them in sampling time by two orders of magnitude. In modeling
binary images, LSGM achieves state-of-the-art likelihood on the binarized
OMNIGLOT dataset.

    

### [[2106.06098] Meta-Adaptive Nonlinear Control: Theory and Algorithms](http://arxiv.org/abs/2106.06098)


  We present an online multi-task learning approach for adaptive nonlinear
control, which we call Online Meta-Adaptive Control (OMAC). The goal is to
control a nonlinear system subject to adversarial disturbance and unknown
$\textit{environment-dependent}$ nonlinear dynamics, under the assumption that
the environment-dependent dynamics can be well captured with some shared
representation. Our approach is motivated by robot control, where a robotic
system encounters a sequence of new environmental conditions that it must
quickly adapt to. A key emphasis is to integrate online representation learning
with established methods from control theory, in order to arrive at a unified
framework that yields both control-theoretic and learning-theoretic guarantees.
We provide instantiations of our approach under varying conditions, leading to
the first non-asymptotic end-to-end convergence guarantee for multi-task
nonlinear control. OMAC can also be integrated with deep representation
learning. Experiments show that OMAC significantly outperforms conventional
adaptive control approaches which do not learn the shared representation, in
inverted pendulum and 6-DoF drone control tasks under varying wind conditions.

    

### [[2106.06295] Going Beyond Linear Transformers with Recurrent Fast Weight Programmers](http://arxiv.org/abs/2106.06295)


  Transformers with linearised attention (''linear Transformers'') have
demonstrated the practical scalability and effectiveness of outer product-based
Fast Weight Programmers (FWPs) from the '90s. However, the original FWP
formulation is more general than the one of linear Transformers: a slow neural
network (NN) continually reprograms the weights of a fast NN with arbitrary
architecture. In existing linear Transformers, both NNs are feedforward and
consist of a single layer. Here we explore new variations by adding recurrence
to the slow and fast nets. We evaluate our novel recurrent FWPs (RFWPs) on two
synthetic algorithmic tasks (code execution and sequential ListOps),
Wikitext-103 language models, and on the Atari 2600 2D game environment. Our
models exhibit properties of Transformers and RNNs. In the reinforcement
learning setting, we report large improvements over LSTM in several Atari
games. Our code is public.

    

### [[2106.07009] Noise2Score: Tweedie's Approach to Self-Supervised Image Denoising without Clean Images](http://arxiv.org/abs/2106.07009)


  Recently, there has been extensive research interest in training deep
networks to denoise images without clean reference. However, the representative
approaches such as Noise2Noise, Noise2Void, Stein's unbiased risk estimator
(SURE), etc. seem to differ from one another and it is difficult to find the
coherent mathematical structure. To address this, here we present a novel
approach, called Noise2Score, which reveals a missing link in order to unite
these seemingly different approaches. Specifically, we show that image
denoising problems without clean images can be addressed by finding the mode of
the posterior distribution and that the Tweedie's formula offers an explicit
solution through the score function (i.e. the gradient of log likelihood). Our
method then uses the recent finding that the score function can be stably
estimated from the noisy images using the amortized residual denoising
autoencoder, the method of which is closely related to Noise2Noise or
Nose2Void. Our Noise2Score approach is so universal that the same network
training can be used to remove noises from images that are corrupted by any
exponential family distributions and noise parameters. Using extensive
experiments with Gaussian, Poisson, and Gamma noises, we show that Noise2Score
significantly outperforms the state-of-the-art self-supervised denoising
methods in the benchmark data set such as (C)BSD68, Set12, and Kodak, etc.

    

### [[2106.07428] Audio Attacks and Defenses against AED Systems -- A Practical Study](http://arxiv.org/abs/2106.07428)


  In this paper, we evaluate deep learning-enabled AED systems against evasion
attacks based on adversarial examples. We test the robustness of multiple
security critical AED tasks, implemented as CNNs classifiers, as well as
existing third-party Nest devices, manufactured by Google, which run their own
black-box deep learning models. Our adversarial examples use audio
perturbations made of white and background noises. Such disturbances are easy
to create, to perform and to reproduce, and can be accessible to a large number
of potential attackers, even non-technically savvy ones.
We show that an adversary can focus on audio adversarial inputs to cause AED
systems to misclassify, achieving high success rates, even when we use small
levels of a given type of noisy disturbance. For instance, on the case of the
gunshot sound class, we achieve nearly 100% success rate when employing as
little as 0.05 white noise level. Similarly to what has been previously done by
works focusing on adversarial examples from the image domain as well as on the
speech recognition domain. We then, seek to improve classifiers' robustness
through countermeasures. We employ adversarial training and audio denoising. We
show that these countermeasures, when applied to audio input, can be
successful, either in isolation or in combination, generating relevant
increases of nearly fifty percent in the performance of the classifiers when
these are under attack.

    

### [[2106.07644] A Continuized View on Nesterov Acceleration for Stochastic Gradient Descent and Randomized Gossip](http://arxiv.org/abs/2106.07644)


  We introduce the continuized Nesterov acceleration, a close variant of
Nesterov acceleration whose variables are indexed by a continuous time
parameter. The two variables continuously mix following a linear ordinary
differential equation and take gradient steps at random times. This continuized
variant benefits from the best of the continuous and the discrete frameworks:
as a continuous process, one can use differential calculus to analyze
convergence and obtain analytical expressions for the parameters; and a
discretization of the continuized process can be computed exactly with
convergence rates similar to those of Nesterov original acceleration. We show
that the discretization has the same structure as Nesterov acceleration, but
with random parameters. We provide continuized Nesterov acceleration under
deterministic as well as stochastic gradients, with either additive or
multiplicative noise. Finally, using our continuized framework and expressing
the gossip averaging problem as the stochastic minimization of a certain energy
function, we provide the first rigorous acceleration of asynchronous gossip
algorithms.

    

### [[2106.07914] Control Variates for Slate Off-Policy Evaluation](http://arxiv.org/abs/2106.07914)


  We study the problem of off-policy evaluation from batched contextual bandit
data with multidimensional actions, often termed slates. The problem is common
to recommender systems and user-interface optimization, and it is particularly
challenging because of the combinatorially-sized action space. Swaminathan et
al. (2017) have proposed the pseudoinverse (PI) estimator under the assumption
that the conditional mean rewards are additive in actions. Using control
variates, we consider a large class of unbiased estimators that includes as
specific cases the PI estimator and (asymptotically) its self-normalized
variant. By optimizing over this class, we obtain new estimators with risk
improvement guarantees over both the PI and the self-normalized PI estimators.
Experiments with real-world recommender data as well as synthetic data validate
these improvements in practice.

    

### [[2106.08769] Knowledge-Adaptation Priors](http://arxiv.org/abs/2106.08769)


  Humans and animals have a natural ability to quickly adapt to their
surroundings, but machine-learning models, when subjected to changes, often
require a complete retraining from scratch. We present Knowledge-adaptation
priors (K-priors) to reduce the cost of retraining by enabling quick and
accurate adaptation for a wide-variety of tasks and models. This is made
possible by a combination of weight and function-space priors to reconstruct
the gradients of the past, which recovers and generalizes many existing, but
seemingly-unrelated, adaptation strategies. Training with simple first-order
gradient methods can often recover the exact retrained model to an arbitrary
accuracy by choosing a sufficiently large memory of the past data. Empirical
results show that adaptation with K-priors achieves performance similar to full
retraining, but only requires training on a handful of past examples.

    

### [[2106.09526] Exploring the Properties and Evolution of Neural Network Eigenspaces during Training](http://arxiv.org/abs/2106.09526)


  In this work we explore the information processing inside neural networks
using logistic regression probes \cite{probes} and the saturation metric
\cite{featurespace_saturation}. We show that problem difficulty and neural
network capacity affect the predictive performance in an antagonistic manner,
opening the possibility of detecting over- and under-parameterization of neural
networks for a given task. We further show that the observed effects are
independent from previously reported pathological patterns like the ``tail
pattern'' described in \cite{featurespace_saturation}. Finally we are able to
show that saturation patterns converge early during training, allowing for a
quicker cycle time during analysis

    

### [[2106.09620] Disentangling Identifiable Features from Noisy Data with Structured Nonlinear ICA](http://arxiv.org/abs/2106.09620)


  We introduce a new general identifiable framework for principled
disentanglement referred to as Structured Nonlinear Independent Component
Analysis (SNICA). Our contribution is to extend the identifiability theory of
deep generative models for a very broad class of structured models. While
previous works have shown identifiability for specific classes of time-series
models, our theorems extend this to more general temporal structures as well as
to models with more complex structures such as spatial dependencies. In
particular, we establish the major result that identifiability for this
framework holds even in the presence of noise of unknown distribution. Finally,
as an example of our framework's flexibility, we introduce the first nonlinear
ICA model for time-series that combines the following very useful properties:
it accounts for both nonstationarity and autocorrelation in a fully
unsupervised setting; performs dimensionality reduction; models hidden states;
and enables principled estimation and inference by variational
maximum-likelihood.

    

### [[2106.10575] EvoGrad: Efficient Gradient-Based Meta-Learning and Hyperparameter Optimization](http://arxiv.org/abs/2106.10575)


  Gradient-based meta-learning and hyperparameter optimization have seen
significant progress recently, enabling practical end-to-end training of neural
networks together with many hyperparameters. Nevertheless, existing approaches
are relatively expensive as they need to compute second-order derivatives and
store a longer computational graph. This cost prevents scaling them to larger
network architectures. We present EvoGrad, a new approach to meta-learning that
draws upon evolutionary techniques to more efficiently compute hypergradients.
EvoGrad estimates hypergradient with respect to hyperparameters without
calculating second-order gradients, or storing a longer computational graph,
leading to significant improvements in efficiency. We evaluate EvoGrad on three
substantial recent meta-learning applications, namely cross-domain few-shot
learning with feature-wise transformations, noisy label learning with
Meta-Weight-Net and low-resource cross-lingual learning with meta
representation transformation. The results show that EvoGrad significantly
improves efficiency and enables scaling meta-learning to bigger architectures
such as from ResNet10 to ResNet34.

    

### [[2106.12674] Fairness via Representation Neutralization](http://arxiv.org/abs/2106.12674)


  Existing bias mitigation methods for DNN models primarily work on learning
debiased encoders. This process not only requires a lot of instance-level
annotations for sensitive attributes, it also does not guarantee that all
fairness sensitive information has been removed from the encoder. To address
these limitations, we explore the following research question: Can we reduce
the discrimination of DNN models by only debiasing the classification head,
even with biased representations as inputs? To this end, we propose a new
mitigation technique, namely, Representation Neutralization for Fairness (RNF)
that achieves fairness by debiasing only the task-specific classification head
of DNN models. To this end, we leverage samples with the same ground-truth
label but different sensitive attributes, and use their neutralized
representations to train the classification head of the DNN model. The key idea
of RNF is to discourage the classification head from capturing spurious
correlation between fairness sensitive information in encoder representations
with specific class labels. To address low-resource settings with no access to
sensitive attribute annotations, we leverage a bias-amplified model to generate
proxy annotations for sensitive attributes. Experimental results over several
benchmark datasets demonstrate our RNF framework to effectively reduce
discrimination of DNN models with minimal degradation in task-specific
performance.

    

### [[2106.14326] Last-iterate Convergence in Extensive-Form Games](http://arxiv.org/abs/2106.14326)


  Regret-based algorithms are highly efficient at finding approximate Nash
equilibria in sequential games such as poker games. However, most regret-based
algorithms, including counterfactual regret minimization (CFR) and its
variants, rely on iterate averaging to achieve convergence. Inspired by recent
advances on last-iterate convergence of optimistic algorithms in zero-sum
normal-form games, we study this phenomenon in sequential games, and provide a
comprehensive study of last-iterate convergence for zero-sum extensive-form
games with perfect recall (EFGs), using various optimistic regret-minimization
algorithms over treeplexes. This includes algorithms using the vanilla entropy
or squared Euclidean norm regularizers, as well as their dilated versions which
admit more efficient implementation. In contrast to CFR, we show that all of
these algorithms enjoy last-iterate convergence, with some of them even
converging exponentially fast. We also provide experiments to further support
our theoretical results.

    

### [[2106.14942] Fast Training of Neural Lumigraph Representations using Meta Learning](http://arxiv.org/abs/2106.14942)


  Novel view synthesis is a long-standing problem in machine learning and
computer vision. Significant progress has recently been made in developing
neural scene representations and rendering techniques that synthesize
photorealistic images from arbitrary views. These representations, however, are
extremely slow to train and often also slow to render. Inspired by neural
variants of image-based rendering, we develop a new neural rendering approach
with the goal of quickly learning a high-quality representation which can also
be rendered in real-time. Our approach, MetaNLR++, accomplishes this by using a
unique combination of a neural shape representation and 2D CNN-based image
feature extraction, aggregation, and re-projection. To push representation
convergence times down to minutes, we leverage meta learning to learn neural
shape and image feature priors which accelerate training. The optimized shape
and image features can then be extracted using traditional graphics techniques
and rendered in real time. We show that MetaNLR++ achieves similar or better
novel view synthesis results in a fraction of the time that competing methods
require.

    

### [[2106.15577] As easy as APC: overcoming missing data and class imbalance in time series with self-supervised learning](http://arxiv.org/abs/2106.15577)


  High levels of missing data and strong class imbalance are ubiquitous
challenges that are often presented simultaneously in real-world time series
data. Existing methods approach these problems separately, frequently making
significant assumptions about the underlying data generation process in order
to lessen the impact of missing information. In this work, we instead
demonstrate how a general self-supervised training method, namely
Autoregressive Predictive Coding (APC), can be leveraged to overcome both
missing data and class imbalance simultaneously without strong assumptions.
Specifically, on a synthetic dataset, we show that standard baselines are
substantially improved upon through the use of APC, yielding the greatest gains
in the combined setting of high missingness and severe class imbalance. We
further apply APC on two real-world medical time-series datasets, and show that
APC improves the classification performance in all settings, ultimately
achieving state-of-the-art AUPRC results on the Physionet benchmark.

    

### [[2106.15842] Dual Aspect Self-Attention based on Transformer for Remaining Useful Life Prediction](http://arxiv.org/abs/2106.15842)


  Remaining useful life prediction (RUL) is one of the key technologies of
condition-based maintenance, which is important to maintain the reliability and
safety of industrial equipments. While deep learning has achieved great success
in RUL prediction, existing methods have difficulties in processing long
sequences and extracting information from the sensor and time step aspects. In
this paper, we propose Dual Aspect Self-attention based on Transformer (DAST),
a novel deep RUL prediction method. DAST consists of two encoders, which work
in parallel to simultaneously extract features of different sensors and time
steps. Solely based on self-attention, the DAST encoders are more effective in
processing long data sequences, and are capable of adaptively learning to focus
on more important parts of input. Moreover, the parallel feature extraction
design avoids mutual influence of information from two aspects. Experimental
results on two real turbofan engine datasets show that our method significantly
outperforms state-of-the-art methods.

    

### [[2109.09314] Investigating the Relationship Between World Development Indicators and the Occurrence of Disease Outbreaks in the 21st Century: A Case Study](http://arxiv.org/abs/2109.09314)


  The timely identification of socio-economic sectors vulnerable to a disease
outbreak presents an important challenge to the civic authorities and
healthcare workers interested in outbreak mitigation measures. This problem was
traditionally solved by studying the aberrances in small-scale healthcare data.
In this paper, we leverage data driven models to determine the relationship
between the trends of World Development Indicators and occurrence of disease
outbreaks using worldwide historical data from 2000-2019, and treat it as a
classic supervised classification problem. CART based feature selection was
employed in an unorthodox fashion to determine the covariates getting affected
by the disease outbreak, thus giving the most vulnerable sectors. The result
involves a comprehensive analysis of different classification algorithms and is
indicative of the relationship between the disease outbreak occurrence and the
magnitudes of various development indicators.

    

### [[2110.08949] Real-time Mortality Prediction Using MIMIC-IV ICU Data Via Boosted Nonparametric Hazards](http://arxiv.org/abs/2110.08949)


  Electronic Health Record (EHR) systems provide critical, rich and valuable
information at high frequency. One of the most exciting applications of EHR
data is in developing a real-time mortality warning system with tools from
survival analysis. However, most of the survival analysis methods used recently
are based on (semi)parametric models using static covariates. These models do
not take advantage of the information conveyed by the time-varying EHR data. In
this work, we present an application of a highly scalable survival analysis
method, BoXHED 2.0 to develop a real-time in-ICU mortality warning indicator
based on the MIMIC IV data set. Importantly, BoXHED can incorporate
time-dependent covariates in a fully nonparametric manner and is backed by
theory. Our in-ICU mortality model achieves an AUC-PRC of 0.41 and AUC-ROC of
0.83 out of sample, demonstrating the benefit of real-time monitoring.

    

### [[2110.11216] User-friendly introduction to PAC-Bayes bounds](http://arxiv.org/abs/2110.11216)


  Aggregated predictors are obtained by making a set of basic predictors vote
according to some weights, that is, to some probability distribution.
Randomized predictors are obtained by sampling in a set of basic predictors,
according to some prescribed probability distribution.
Thus, aggregated and randomized predictors have in common that they are not
defined by a minimization problem, but by a probability distribution on the set
of predictors. In statistical learning theory, there is a set of tools designed
to understand the generalization ability of such procedures: PAC-Bayesian or
PAC-Bayes bounds.
Since the original PAC-Bayes bounds of D. McAllester, these tools have been
considerably improved in many directions (we will for example describe a
simplified version of the localization technique of O. Catoni that was missed
by the community, and later rediscovered as "mutual information bounds"). Very
recently, PAC-Bayes bounds received a considerable attention: for example there
was workshop on PAC-Bayes at NIPS 2017, "(Almost) 50 Shades of Bayesian
Learning: PAC-Bayesian trends and insights", organized by B. Guedj, F. Bach and
P. Germain. One of the reason of this recent success is the successful
application of these bounds to neural networks by G. Dziugaite and D. Roy.
An elementary introduction to PAC-Bayes theory is still missing. This is an
attempt to provide such an introduction.

    

### [[2110.13005] Myelin: An asynchronous, message-driven parallel framework for extreme-scale deep learning](http://arxiv.org/abs/2110.13005)


  In the last few years, the memory requirements to train state-of-the-art
neural networks have far exceeded the DRAM capacities of modern hardware
accelerators. This has necessitated the development of efficient algorithms to
train these neural networks in parallel on large-scale GPU-based clusters.
Since computation is relatively inexpensive on modern GPUs, designing and
implementing extremely efficient communication in these parallel training
algorithms is critical for extracting the maximum performance. This paper
presents Myelin, a parallel deep learning framework that exploits asynchrony
and message-driven execution to schedule neural network operations on each GPU,
thereby reducing GPU idle time and maximizing hardware efficiency. By using the
CPU memory as a scratch space for offloading data periodically during training,
Myelin is able to reduce GPU memory consumption by four times. This allows us
to increase the number of parameters per GPU by four times, thus reducing the
amount of communication and increasing performance by over 13%. When tested
against large transformer models with 12-100 billion parameters on 48-384
NVIDIA Tesla V100 GPUs, Myelin achieves a per-GPU throughput of 49.4-54.78% of
theoretical peak and reduces the training time by 22-37 days (15-25% speedup)
as compared to the state-of-the-art.

    

### [[2110.12628] Recurrent Off-policy Baselines for Memory-based Continuous Control](http://arxiv.org/abs/2110.12628)


  When the environment is partially observable (PO), a deep reinforcement
learning (RL) agent must learn a suitable temporal representation of the entire
history in addition to a strategy to control. This problem is not novel, and
there have been model-free and model-based algorithms proposed for this
problem. However, inspired by recent success in model-free image-based RL, we
noticed the absence of a model-free baseline for history-based RL that (1) uses
full history and (2) incorporates recent advances in off-policy continuous
control. Therefore, we implement recurrent versions of DDPG, TD3, and SAC
(RDPG, RTD3, and RSAC) in this work, evaluate them on short-term and long-term
PO domains, and investigate key design choices. Our experiments show that RDPG
and RTD3 can surprisingly fail on some domains and that RSAC is the most
reliable, reaching near-optimal performance on nearly all domains. However, one
task that requires systematic exploration still proved to be difficult, even
for RSAC. These results show that model-free RL can learn good temporal
representation using only reward signals; the primary difficulty seems to be
computational cost and exploration. To facilitate future research, we have made
our PyTorch implementation publicly available at
this https URL.

    

### [[2006.00364] CLARINET: A RISC-V Based Framework for Posit Arithmetic Empiricism](http://arxiv.org/abs/2006.00364)


  Many engineering and scientific applications require high precision
arithmetic. IEEE~754-2008 compliant (floating-point) arithmetic is the de facto
standard for performing these computations. Recently, posit arithmetic has been
proposed as a drop-in replacement for floating-point arithmetic. The
posit\texttrademark data representation and arithmetic claim several absolute
advantages over the floating-point format and arithmetic, including higher
dynamic range, better accuracy, and superior performance-area trade-offs.
However, there does not exist any accessible, holistic framework that
facilitates the validation of these claims of posit arithmetic, especially when
the claims involve long accumulations (quire).
In this paper, we present a consolidated general-purpose processor-based
framework to support posit arithmetic empiricism. The end-users of the
framework have the liberty to seamlessly experiment with their applications
using posit and floating-point arithmetic since the framework is designed for
the two number systems to coexist. Melodica is a posit arithmetic core that
implements parametric fused operations that uniquely involve the quire data
type. Clarinet is a Melodica-enabled processor based on the RISC-V ISA. To the
best of our knowledge, this is the first-ever integration of quire with a
RISC-V core. To show the effectiveness of the Clarinet platform, we perform an
extensive application study and benchmark some of the common linear algebra and
computer vision kernels. We emulate Clarinet on a Xilinx FPGA and present
utilization and timing data. Clarinet and Melodica remain actively under
development and is available in open-source for posit arithmetic empiricism.

    

### [[2106.04772] HyCA: A Hybrid Computing Architecture for Fault Tolerant Deep Learning](http://arxiv.org/abs/2106.04772)


  Hardware faults on the regular 2-D computing array of a typical deep learning
accelerator (DLA) can lead to dramatic prediction accuracy loss. Prior
redundancy design approaches typically have each homogeneous redundant
processing element (PE) to mitigate faulty PEs for a limited region of the 2-D
computing array rather than the entire computing array to avoid the excessive
hardware overhead. However, they fail to recover the computing array when the
number of faulty PEs in any region exceeds the number of redundant PEs in the
same region. The mismatch problem deteriorates when the fault injection rate
rises and the faults are unevenly distributed. To address the problem, we
propose a hybrid computing architecture (HyCA) for fault-tolerant DLAs. It has
a set of dot-production processing units (DPPUs) to recompute all the
operations that are mapped to the faulty PEs despite the faulty PE locations.
According to our experiments, HyCA shows significantly higher reliability,
scalability, and performance with less chip area penalty when compared to the
conventional redundancy approaches. Moreover, by taking advantage of the
flexible recomputing, HyCA can also be utilized to scan the entire 2-D
computing array and detect the faulty PEs effectively at runtime.

    

### [[2110.13967] Evaluating Serverless Architecture for Big Data Enterprise Applications](http://arxiv.org/abs/2110.13967)


  In this paper, we investigate serverless computing for performing large scale
data processing with cloudnative primitives.

    

### [[2110.13999] Exploring the Role of Machine Learning in Scientific Workflows: Opportunities and Challenges](http://arxiv.org/abs/2110.13999)


  In this survey, we discuss the challenges of executing scientific workflows
as well as existing Machine Learning (ML) techniques to alleviate those
challenges. We provide the context and motivation for applying ML to each step
of the execution of these workflows. Furthermore, we provide recommendations on
how to extend ML techniques to unresolved challenges in the execution of
scientific workflows. Moreover, we discuss the possibility of using ML
techniques for in-situ operations. We explore the challenges of in-situ
workflows and provide suggestions for improving the performance of their
execution using ML techniques.

    

### [[2110.14340] JACC: An OpenACC Runtime Framework with Kernel-Level and Multi-GPU Parallelization](http://arxiv.org/abs/2110.14340)


  The rapid development in computing technology has paved the way for
directive-based programming models towards a principal role in maintaining
software portability of performance-critical applications. Efforts on such
models involve a least engineering cost for enabling computational acceleration
on multiple architectures while programmers are only required to add meta
information upon sequential code. Optimizations for obtaining the best possible
efficiency, however, are often challenging. The insertions of directives by the
programmer can lead to side-effects that limit the available compiler
optimization possible, which could result in performance degradation. This is
exacerbated when targeting multi-GPU systems, as pragmas do not automatically
adapt to such systems, and require expensive and time consuming code adjustment
by programmers.
This paper introduces JACC, an OpenACC runtime framework which enables the
dynamic extension of OpenACC programs by serving as a transparent layer between
the program and the compiler. We add a versatile code-translation method for
multi-device utilization by which manually-optimized applications can be
distributed automatically while keeping original code structure and
parallelism. We show in some cases nearly linear scaling on the part of kernel
execution with the NVIDIA V100 GPUs. While adaptively using multi-GPUs, the
resulting performance improvements amortize the latency of GPU-to-GPU
communications.

    

### [[2110.14391] Distributed Principal Component Analysis with Limited Communication](http://arxiv.org/abs/2110.14391)


  We study efficient distributed algorithms for the fundamental problem of
principal component analysis and leading eigenvector computation on the sphere,
when the data are randomly distributed among a set of computational nodes. We
propose a new quantized variant of Riemannian gradient descent to solve this
problem, and prove that the algorithm converges with high probability under a
set of necessary spherical-convexity properties. We give bounds on the number
of bits transmitted by the algorithm under common initialization schemes, and
investigate the dependency on the problem dimension in each case.

    

### [[2110.14502] Closing the "Quantum Supremacy" Gap: Achieving Real-Time Simulation of a Random Quantum Circuit Using a New Sunway Supercomputer](http://arxiv.org/abs/2110.14502)


  We develop a high-performance tensor-based simulator for random quantum
circuits(RQCs) on the new Sunway supercomputer. Our major innovations include:
(1) a near-optimal slicing scheme, and a path-optimization strategy that
considers both complexity and compute density; (2) a three-level
parallelization scheme that scales to about 42 million cores; (3) a fused
permutation and multiplication design that improves the compute efficiency for
a wide range of tensor contraction scenarios; and (4) a mixed-precision scheme
to further improve the performance. Our simulator effectively expands the scope
of simulatable RQCs to include the 10*10(qubits)*(1+40+1)(depth) circuit, with
a sustained performance of 1.2 Eflops (single-precision), or 4.4 Eflops
(mixed-precision)as a new milestone for classical simulation of quantum
circuits; and reduces the simulation sampling time of Google Sycamore to 304
seconds, from the previously claimed 10,000 years.

    

### [[2110.14609] Paving the Way for Consensus: Convergence of Block Gossip Algorithms](http://arxiv.org/abs/2110.14609)


  Gossip protocols are popular methods for average consensus problems in
distributed computing. We prove new convergence guarantees for a variety of
such protocols, including path, clique, and synchronous pairwise gossip. These
arise by exploiting the connection between these protocols and the block
randomized Kaczmarz method for solving linear systems. Moreover, we extend
existing convergence results for block randomized Kaczmarz to allow for a more
general choice of blocks, rank-deficient systems, and provide a tighter
convergence rate guarantee. We furthermore apply this analysis to inconsistent
consensus models and obtain similar guarantees. An extensive empirical analysis
of these methods is provided for a variety of synthetic networks.

    

### [[1912.12740] Practice of Streaming Processing of Dynamic Graphs: Concepts, Models, and Systems](http://arxiv.org/abs/1912.12740)


  Graph processing has become an important part of various areas of computing,
including machine learning, medical applications, social network analysis,
computational sciences, and others. A growing amount of the associated graph
processing workloads are dynamic, with millions of edges added or removed per
second. Graph streaming frameworks are specifically crafted to enable the
processing of such highly dynamic workloads. Recent years have seen the
development of many such frameworks. However, they differ in their general
architectures (with key details such as the support for the concurrent
execution of graph updates and queries, or the incorporated graph data
organization), the types of updates and workloads allowed, and many others. To
facilitate the understanding of this growing field, we provide the first
analysis and taxonomy of dynamic and streaming graph processing. We focus on
identifying the fundamental system designs and on understanding their support
for concurrency, and for different graph updates as well as analytics
workloads. We also crystallize the meaning of different concepts associated
with streaming graph processing, such as dynamic, temporal, online, and
time-evolving graphs, edge-centric processing, models for the maintenance of
updates, and graph databases. Moreover, we provide a bridge with the very rich
landscape of graph streaming theory by giving a broad overview of recent
theoretical related advances, and by discussing which graph streaming models
and settings could be helpful in developing more powerful streaming frameworks
and designs. We also outline graph streaming workloads and research challenges.

    

### [[2012.13995] FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping](http://arxiv.org/abs/2012.13995)


  Byzantine-robust federated learning aims to enable a service provider to
learn an accurate global model when a bounded number of clients are malicious.
The key idea of existing Byzantine-robust federated learning methods is that
the service provider performs statistical analysis among the clients' local
model updates and removes suspicious ones, before aggregating them to update
the global model. However, malicious clients can still corrupt the global
models in these methods via sending carefully crafted local model updates to
the service provider. The fundamental reason is that there is no root of trust
in existing federated learning methods.
In this work, we bridge the gap via proposing FLTrust, a new federated
learning method in which the service provider itself bootstraps trust. In
particular, the service provider itself collects a clean small training dataset
(called root dataset) for the learning task and the service provider maintains
a model (called server model) based on it to bootstrap trust. In each
iteration, the service provider first assigns a trust score to each local model
update from the clients, where a local model update has a lower trust score if
its direction deviates more from the direction of the server model update.
Then, the service provider normalizes the magnitudes of the local model updates
such that they lie in the same hyper-sphere as the server model update in the
vector space. Our normalization limits the impact of malicious local model
updates with large magnitudes. Finally, the service provider computes the
average of the normalized local model updates weighted by their trust scores as
a global model update, which is used to update the global model. Our extensive
evaluations on six datasets from different domains show that our FLTrust is
secure against both existing attacks and strong adaptive attacks.

    

### [[2110.14076] CoFiNet: Reliable Coarse-to-fine Correspondences for Robust Point Cloud Registration](http://arxiv.org/abs/2110.14076)


  We study the problem of extracting correspondences between a pair of point
clouds for registration. For correspondence retrieval, existing works benefit
from matching sparse keypoints detected from dense points but usually struggle
to guarantee their repeatability. To address this issue, we present CoFiNet -
Coarse-to-Fine Network which extracts hierarchical correspondences from coarse
to fine without keypoint detection. On a coarse scale and guided by a weighting
scheme, our model firstly learns to match down-sampled nodes whose vicinity
points share more overlap, which significantly shrinks the search space of a
consecutive stage. On a finer scale, node proposals are consecutively expanded
to patches that consist of groups of points together with associated
descriptors. Point correspondences are then refined from the overlap areas of
corresponding patches, by a density-adaptive matching module capable to deal
with varying point density. Extensive evaluation of CoFiNet on both indoor and
outdoor standard benchmarks shows our superiority over existing methods.
Especially on 3DLoMatch where point clouds share less overlap, CoFiNet
significantly outperforms state-of-the-art approaches by at least 5% on
Registration Recall, with at most two-third of their parameters.

    

### [[2110.14091] Connect-the-Dots: Bridging Semantics between Words and Definitions via Aligning Word Sense Inventories](http://arxiv.org/abs/2110.14091)


  Word Sense Disambiguation (WSD) aims to automatically identify the exact
meaning of one word according to its context. Existing supervised models
struggle to make correct predictions on rare word senses due to limited
training data and can only select the best definition sentence from one
predefined word sense inventory (e.g., WordNet). To address the data sparsity
problem and generalize the model to be independent of one predefined inventory,
we propose a gloss alignment algorithm that can align definition sentences
(glosses) with the same meaning from different sense inventories to collect
rich lexical knowledge. We then train a model to identify semantic equivalence
between a target word in context and one of its glosses using these aligned
inventories, which exhibits strong transfer capability to many WSD tasks.
Experiments on benchmark datasets show that the proposed method improves
predictions on both frequent and rare word senses, outperforming prior work by
1.2% on the All-Words WSD Task and 4.3% on the Low-Shot WSD Task. Evaluation on
WiC Task also indicates that our method can better capture word meanings in
context.

    

### [[2110.14187] An Experimental Study of Permanently Stored Learned Clauses](http://arxiv.org/abs/2110.14187)


  Modern CDCL SAT solvers learn clauses rapidly, and an important heuristic is
the clause deletion scheme. Most current solvers have two (or more) stores of
clauses. One has ``valuable'' clauses which are never deleted. Most learned
clauses are added to the other, with an aggressive deletion strategy to
restrict its size. Recent solvers in the MapleSAT family, have comparatively
complex deletion scheme, and perform well. Many solvers store only binary
clauses permanently, but MapleLCMDistChronoBT stores clauses with small LBD
permanently. We report an experimental study of the permanent clause store in
MapleLCMDistChronoBT. We observe that this store can get quite large, but
several methods for limiting its size reduced performance. We also show that
alternate size and LBD based criteria improve performance, while still having
large permanent stores. In particular, saving clauses up to size 8, and adding
small numbers of high-centrality clauses, both improved performance, with the
best improvement using both methods.

    

### [[2110.14196] From Image to Imuge: Immunized Image Generation](http://arxiv.org/abs/2110.14196)


  We introduce Imuge, an image tamper resilient generative scheme for image
self-recovery. The traditional manner of concealing image content within the
image are inflexible and fragile to diverse digital attack, i.e. image cropping
and JPEG compression. To address this issue, we jointly train a U-Net backboned
encoder, a tamper localization network and a decoder for image recovery. Given
an original image, the encoder produces a visually indistinguishable immunized
image. At the recipient's side, the verifying network localizes the malicious
modifications, and the original content can be approximately recovered by the
decoder, despite the presence of the attacks. Several strategies are proposed
to boost the training efficiency. We demonstrate that our method can recover
the details of the tampered regions with a high quality despite the presence of
various kinds of attacks. Comprehensive ablation studies are conducted to
validate our network designs.

    

### [[2110.14207] How Much Coffee Was Consumed During EMNLP 2019? Fermi Problems: A New Reasoning Challenge for AI](http://arxiv.org/abs/2110.14207)


  Many real-world problems require the combined application of multiple
reasoning abilities employing suitable abstractions, commonsense knowledge, and
creative synthesis of problem-solving strategies. To help advance AI systems
towards such capabilities, we propose a new reasoning challenge, namely Fermi
Problems (FPs), which are questions whose answers can only be approximately
estimated because their precise computation is either impractical or
impossible. For example, "How much would the sea level rise if all ice in the
world melted?" FPs are commonly used in quizzes and interviews to bring out and
evaluate the creative reasoning abilities of humans. To do the same for AI
systems, we present two datasets: 1) A collection of 1k real-world FPs sourced
from quizzes and olympiads; and 2) a bank of 10k synthetic FPs of intermediate
complexity to serve as a sandbox for the harder real-world challenge. In
addition to question answer pairs, the datasets contain detailed solutions in
the form of an executable program and supporting facts, helping in supervision
and evaluation of intermediate steps. We demonstrate that even extensively
fine-tuned large scale language models perform poorly on these datasets, on
average making estimates that are off by two orders of magnitude. Our
contribution is thus the crystallization of several unsolved AI problems into a
single, new challenge that we hope will spur further advances in building
systems that can reason.

    

### [[2110.14227] Emoji-based Co-attention Network for Microblog Sentiment Analysis](http://arxiv.org/abs/2110.14227)


  Emojis are widely used in online social networks to express emotions,
attitudes, and opinions. As emotional-oriented characters, emojis can be
modeled as important features of emotions towards the recipient or subject for
sentiment analysis. However, existing methods mainly take emojis as heuristic
information that fails to resolve the problem of ambiguity noise. Recent
researches have utilized emojis as an independent input to classify text
sentiment but they ignore the emotional impact of the interaction between text
and emojis. It results that the emotional semantics of emojis cannot be fully
explored. In this paper, we propose an emoji-based co-attention network that
learns the mutual emotional semantics between text and emojis on microblogs.
Our model adopts the co-attention mechanism based on bidirectional long
short-term memory incorporating the text and emojis, and integrates a
squeeze-and-excitation block in a convolutional neural network classifier to
increase its sensitivity to emotional semantic features. Experimental results
show that the proposed method can significantly outperform several baselines
for sentiment analysis on short texts of social media.

    

### [[2110.14357] Binarized ResNet: Enabling Automatic Modulation Classification at the resource-constrained Edge](http://arxiv.org/abs/2110.14357)


  In this paper, we propose a ResNet based neural architecture to solve the
problem of Automatic Modulation Classification. We showed that our architecture
outperforms the state-of-the-art (SOTA) architectures. We further propose to
binarize the network to deploy it in the Edge network where the devices are
resource-constrained i.e. have limited memory and computing power. Instead of
simple binarization, rotated binarization is applied to the network which helps
to close the significant performance gap between the real and the binarized
network. Because of the immense representation capability or the real network,
its rotated binarized version achieves $85.33\%$ accuracy compared to $95.76\%$
accuracy of the proposed real network with $2.33$ and $16$ times lesser
computing power than two of the SOTA architectures, MCNet and RMLResNet
respectively, and approximately $16$ times less memory than both. The
performance can be improved further to $87.74\%$ by taking an ensemble of four
such rotated binarized networks.

    

### [[2110.14378] WenLan 2.0: Make AI Imagine via a Multimodal Foundation Model](http://arxiv.org/abs/2110.14378)


  The fundamental goal of artificial intelligence (AI) is to mimic the core
cognitive activities of human including perception, memory, and reasoning.
Although tremendous success has been achieved in various AI research fields
(e.g., computer vision and natural language processing), the majority of
existing works only focus on acquiring single cognitive ability (e.g., image
classification, reading comprehension, or visual commonsense reasoning). To
overcome this limitation and take a solid step to artificial general
intelligence (AGI), we develop a novel foundation model pre-trained with huge
multimodal (visual and textual) data, which is able to be quickly adapted for a
broad class of downstream cognitive tasks. Such a model is fundamentally
different from the multimodal foundation models recently proposed in the
literature that typically make strong semantic correlation assumption and
expect exact alignment between image and text modalities in their pre-training
data, which is often hard to satisfy in practice thus limiting their
generalization abilities. To resolve this issue, we propose to pre-train our
foundation model by self-supervised learning with weak semantic correlation
data crawled from the Internet and show that state-of-the-art results can be
obtained on a wide range of downstream tasks (both single-modal and
cross-modal). Particularly, with novel model-interpretability tools developed
in this work, we demonstrate that strong imagination ability (even with hints
of commonsense) is now possessed by our foundation model. We believe our work
makes a transformative stride towards AGI and will have broad impact on various
AI+ fields (e.g., neuroscience and healthcare).

    

### [[2110.14397] A Preliminary Case Study of Planning With Complex Transitions: Plotting](http://arxiv.org/abs/2110.14397)


  Plotting is a tile-matching puzzle video game published by Taito in 1989. Its
objective is to reduce a given grid of coloured blocks down to a goal number or
fewer. This is achieved by the avatar character repeatedly shooting the block
it holds into the grid. Plotting is an example of a planning problem: given a
model of the environment, a planning problem asks us to find a sequence of
actions that can lead from an initial state of the environment to a given goal
state while respecting some constraints. The key difficulty in modelling
Plotting is in capturing the way the puzzle state changes after each shot. A
single shot can affect multiple tiles directly, and the grid is affected by
gravity so numerous other tiles can be affected indirectly. We present and
evaluate a constraint model of the Plotting problem that captures this
complexity. We also discuss the difficulties and inefficiencies of modelling
Plotting in PDDL, the standard language used for input to specialised AI
planners. We conclude by arguing that AI planning could benefit from a richer
modelling language.

    

### [[2110.14422] Zero-shot Voice Conversion via Self-supervised Prosody Representation Learning](http://arxiv.org/abs/2110.14422)


  Voice Conversion (VC) for unseen speakers, also known as zero-shot VC, is an
attractive topic due to its usefulness in real use-case scenarios. Recent work
in this area made progress with disentanglement methods that separate utterance
content and speaker characteristics. Although crucial, extracting disentangled
prosody characteristics for unseen speakers remains an open issue. In this
paper, we propose a novel self-supervised approach to effectively learn the
prosody characteristics. Then, we use the learned prosodic representations to
train our VC model for zero-shot conversion. Our evaluation demonstrates that
we can efficiently extract disentangled prosody representation. Moreover, we
show improved performance compared to the state-of-the-art zero-shot VC models.

    

### [[2110.14440] Predictive Geological Mapping with Convolution Neural Network Using Statistical Data Augmentation on a 3D Model](http://arxiv.org/abs/2110.14440)


  Airborne magnetic data are commonly used to produce preliminary geological
maps. Machine learning has the potential to partly fulfill this task rapidly
and objectively, as geological mapping is comparable to a semantic segmentation
problem. Because this method requires a high-quality dataset, we developed a
data augmentation workflow that uses a 3D geological and magnetic
susceptibility model as input. The workflow uses soft-constrained Multi-Point
Statistics, to create many synthetic 3D geological models, and Sequential
Gaussian Simulation algorithms, to populate the models with the appropriate
magnetic distribution. Then, forward modeling is used to compute the airborne
magnetic responses of the synthetic models, which are associated with their
counterpart surficial lithologies. A Gated Shape Convolutional Neural Network
algorithm was trained on a generated synthetic dataset to perform geological
mapping of airborne magnetic data and detect lithological contacts. The
algorithm also provides attention maps highlighting the structures at different
scales, and clustering was applied to its high-level features to do a
semi-supervised segmentation of the area. The validation conducted on a portion
of the synthetic dataset and data from adjacent areas shows that the
methodology is suitable to segment the surficial geology using airborne
magnetic data. Especially, the clustering shows a good segmentation of the
magnetic anomalies into a pertinent geological map. Moreover, the first
attention map isolates the structures at low scales and shows a pertinent
representation of the original data. Thus, our method can be used to produce
preliminary geological maps of good quality and new representations of any area
where a geological and petrophysical 3D model exists, or in areas sharing the
same geological context, using airborne magnetic data only.

    

### [[2110.14450] Rot-Pro: Modeling Transitivity by Projection in Knowledge Graph Embedding](http://arxiv.org/abs/2110.14450)


  Knowledge graph embedding models learn the representations of entities and
relations in the knowledge graphs for predicting missing links (relations)
between entities. Their effectiveness are deeply affected by the ability of
modeling and inferring different relation patterns such as symmetry, asymmetry,
inversion, composition and transitivity. Although existing models are already
able to model many of these relations patterns, transitivity, a very common
relation pattern, is still not been fully supported. In this paper, we first
theoretically show that the transitive relations can be modeled with
projections. We then propose the Rot-Pro model which combines the projection
and relational rotation together. We prove that Rot-Pro can infer all the above
relation patterns. Experimental results show that the proposed Rot-Pro model
effectively learns the transitivity pattern and achieves the state-of-the-art
results on the link prediction task in the datasets containing transitive
relations.

    

### [[2110.14461] Hand gesture detection in the hand movement test for the early diagnosis of dementia](http://arxiv.org/abs/2110.14461)


  Collecting hands data is important for many cognitive studies, especially for
senior participants who has no IT background. For example, alternating hand
movements and imitation of gestures are formal cognitive assessment in the
early detection of dementia. During data collection process, one of the key
steps is to detect whether the participants is following the instruction
correctly to do the correct gestures. Meanwhile, re-searchers found a lot of
problems in TAS Test hand movement data collection process, where is
challenging to detect similar gestures and guarantee the quality of the
collect-ed images. We have implemented a hand gesture detector to detect the
gestures per-formed in the hand movement tests, which enables us to monitor if
the participants are following the instructions correctly. In this research, we
have processed 20,000 images collected from TAS Test and labelled 6,450 images
to detect different hand poses in the hand movement tests. This paper has the
following three contributions. Firstly, we compared the performance of
different network structures for hand poses detection. Secondly, we introduced
a transformer block in the state of art network and increased the
classification performance of the similar gestures. Thirdly, we have created
two datasets and included 20 percent of blurred images in the dataset to
investigate how different network structures were impacted by noisy data, then
we proposed a novel net-work to increase the detection accuracy to mediate the
influence of the noisy data.

    

### [[2110.14491] Training Lightweight CNNs for Human-Nanodrone Proximity Interaction from Small Datasets using Background Randomization](http://arxiv.org/abs/2110.14491)


  We consider the task of visually estimating the pose of a human from images
acquired by a nearby nano-drone; in this context, we propose a data
augmentation approach based on synthetic background substitution to learn a
lightweight CNN model from a small real-world training set. Experimental
results on data from two different labs proves that the approach improves
generalization to unseen environments.

    

### [[2110.14513] Neural Analysis and Synthesis: Reconstructing Speech from Self-Supervised Representations](http://arxiv.org/abs/2110.14513)


  We present a neural analysis and synthesis (NANSY) framework that can
manipulate voice, pitch, and speed of an arbitrary speech signal. Most of the
previous works have focused on using information bottleneck to disentangle
analysis features for controllable synthesis, which usually results in poor
reconstruction quality. We address this issue by proposing a novel training
strategy based on information perturbation. The idea is to perturb information
in the original input signal (e.g., formant, pitch, and frequency response),
thereby letting synthesis networks selectively take essential attributes to
reconstruct the input signal. Because NANSY does not need any bottleneck
structures, it enjoys both high reconstruction quality and controllability.
Furthermore, NANSY does not require any labels associated with speech data such
as text and speaker information, but rather uses a new set of analysis
features, i.e., wav2vec feature and newly proposed pitch feature, Yingram,
which allows for fully self-supervised training. Taking advantage of fully
self-supervised training, NANSY can be easily extended to a multilingual
setting by simply training it with a multilingual dataset. The experiments show
that NANSY can achieve significant improvement in performance in several
applications such as zero-shot voice conversion, pitch shift, and time-scale
modification.

    

### [[2110.14521] Active clustering for labeling training data](http://arxiv.org/abs/2110.14521)


  Gathering training data is a key step of any supervised learning task, and it
is both critical and expensive. Critical, because the quantity and quality of
the training data has a high impact on the performance of the learned function.
Expensive, because most practical cases rely on humans-in-the-loop to label the
data. The process of determining the correct labels is much more expensive than
comparing two items to see whether they belong to the same class. Thus
motivated, we propose a setting for training data gathering where the human
experts perform the comparatively cheap task of answering pairwise queries, and
the computer groups the items into classes (which can be labeled cheaply at the
very end of the process). Given the items, we consider two random models for
the classes: one where the set partition they form is drawn uniformly, the
other one where each item chooses its class independently following a fixed
distribution. In the first model, we characterize the algorithms that minimize
the average number of queries required to cluster the items and analyze their
complexity. In the second model, we analyze a specific algorithm family,
propose as a conjecture that they reach the minimum average number of queries
and compare their performance to a random approach. We also propose solutions
to handle errors or inconsistencies in the experts' answers.

    

### [[2110.14535] Comparing Heuristics, Constraint Optimization, and Reinforcement Learning for an Industrial 2D Packing Problem](http://arxiv.org/abs/2110.14535)


  Cutting and Packing problems are occurring in different industries with a
direct impact on the revenue of businesses. Generally, the goal in Cutting and
Packing is to assign a set of smaller objects to a set of larger objects. To
solve Cutting and Packing problems, practitioners can resort to heuristic and
exact methodologies. Lately, machine learning is increasingly used for solving
such problems. This paper considers a 2D packing problem from the furniture
industry, where a set of wooden workpieces must be assigned to different
modules of a trolley in the most space-saving way. We present an experimental
setup to compare heuristics, constraint optimization, and deep reinforcement
learning for the given problem. The used methodologies and their results get
collated in terms of their solution quality and runtime. In the given use case
a greedy heuristic produces optimal results and outperforms the other
approaches in terms of runtime. Constraint optimization also produces optimal
results but requires more time to perform. The deep reinforcement learning
approach did not always produce optimal or even feasible solutions. While we
assume this could be remedied with more training, considering the good results
with the heuristic, deep reinforcement learning seems to be a bad fit for the
given use case.

    

### [[2110.14613] International Workshop on Continual Semi-Supervised Learning: Introduction, Benchmarks and Baselines](http://arxiv.org/abs/2110.14613)


  The aim of this paper is to formalize a new continual semi-supervised
learning (CSSL) paradigm, proposed to the attention of the machine learning
community via the IJCAI 2021 International Workshop on Continual
Semi-Supervised Learning (CSSL-IJCAI), with the aim of raising field awareness
about this problem and mobilizing its effort in this direction. After a formal
definition of continual semi-supervised learning and the appropriate training
and testing protocols, the paper introduces two new benchmarks specifically
designed to assess CSSL on two important computer vision tasks: activity
recognition and crowd counting. We describe the Continual Activity Recognition
(CAR) and Continual Crowd Counting (CCC) challenges built upon those
benchmarks, the baseline models proposed for the challenges, and describe a
simple CSSL baseline which consists in applying batch self-training in temporal
sessions, for a limited number of rounds. The results show that learning from
unlabelled data streams is extremely challenging, and stimulate the search for
methods that can encode the dynamics of the data stream.

    

### [[1705.09231] Neural Attribute Machines for Program Generation](http://arxiv.org/abs/1705.09231)


  Recurrent neural networks have achieved remarkable success at generating
sequences with complex structures, thanks to advances that include richer
embeddings of input and cures for vanishing gradients. Trained only on
sequences from a known grammar, though, they can still struggle to learn rules
and constraints of the grammar. Neural Attribute Machines (NAMs) are equipped
with a logical machine that represents the underlying grammar, which is used to
teach the constraints to the neural machine by (i) augmenting the input
sequence, and (ii) optimizing a custom loss function. Unlike traditional RNNs,
NAMs are exposed to the grammar, as well as samples from the language of the
grammar. During generation, NAMs make significantly fewer violations of the
constraints of the underlying grammar than RNNs trained only on samples from
the language of the grammar.

    

### [[1908.02962] CRIC: A VQA Dataset for Compositional Reasoning on Vision and Commonsense](http://arxiv.org/abs/1908.02962)


  Alternatively inferring on the visual facts and commonsense is fundamental
for an advanced VQA system. This ability requires models to go beyond the
literal understanding of commonsense. The system should not just treat objects
as the entrance to query background knowledge, but fully ground commonsense to
the visual world and imagine the possible relationships between objects, e.g.,
"fork, can lift, food". To comprehensively evaluate such abilities, we propose
a VQA benchmark, CRIC, which introduces new types of questions about
Compositional Reasoning on vIsion and Commonsense, and an evaluation metric
integrating the correctness of answering and commonsense grounding. To collect
such questions and rich additional annotations to support the metric, we also
propose an automatic algorithm to generate question samples from the scene
graph associated with the images and the relevant knowledge graph. We further
analyze several representative types of VQA models on the CRIC dataset.
Experimental results show that grounding the commonsense to the image region
and joint reasoning on vision and commonsense are still challenging for current
approaches. The dataset is available at this https URL.

    

### [[2007.01647] Learning intuitive physics and one-shot imitation using state-action-prediction self-organizing maps](http://arxiv.org/abs/2007.01647)


  Human learning and intelligence work differently from the supervised pattern
recognition approach adopted in most deep learning architectures. Humans seem
to learn rich representations by exploration and imitation, build causal models
of the world, and use both to flexibly solve new tasks. We suggest a simple but
effective unsupervised model which develops such characteristics. The agent
learns to represent the dynamical physical properties of its environment by
intrinsically motivated exploration, and performs inference on this
representation to reach goals. For this, a set of self-organizing maps which
represent state-action pairs is combined with a causal model for sequence
prediction. The proposed system is evaluated in the cartpole environment. After
an initial phase of playful exploration, the agent can execute kinematic
simulations of the environment's future, and use those for action planning. We
demonstrate its performance on a set of several related, but different one-shot
imitation tasks, which the agent flexibly solves in an active inference style.

    

### [[2009.03561] Local and Central Differential Privacy for Robustness and Privacy in Federated Learning](http://arxiv.org/abs/2009.03561)


  Federated Learning (FL) allows multiple participants to train machine
learning models collaboratively by keeping their datasets local while only
exchanging model updates. Alas, this is not necessarily free from privacy and
robustness vulnerabilities, e.g., via membership, property, and backdoor
attacks. This paper investigates whether and to what extent one can use
differential Privacy (DP) to protect both privacy and robustness in FL. To this
end, we present a first-of-its-kind evaluation of Local and Central
Differential Privacy (LDP/CDP) techniques in FL, assessing their feasibility
and effectiveness. Our experiments show that both DP variants do d fend against
backdoor attacks, albeit with varying levels of protection-utility trade-offs,
but anyway more effectively than other robustness defenses. DP also mitigates
white-box membership inference attacks in FL, and our work is the first to show
it empirically. Neither LDP nor CDP, however, defend against property
inference. Overall, our work provides a comprehensive, re-usable measurement
methodology to quantify the trade-offs between robustness/privacy and utility
in differentially private FL.

    

### [[2104.05857] From partners to populations: A hierarchical Bayesian account of coordination and convention](http://arxiv.org/abs/2104.05857)


  Languages are powerful solutions to coordination problems: they provide
stable, shared expectations about how the words we say correspond to the
beliefs and intentions in our heads. Yet language use in a variable and
non-stationary social environment requires linguistic representations to be
flexible: old words acquire new ad hoc or partner-specific meanings on the fly.
In this paper, we introduce CHAI (Continual Hierarchical Adaptation through
Inference), a hierarchical Bayesian theory of coordination and convention
formation that aims to reconcile the long-standing tension between these two
basic observations. We argue that the central computational problem of
communication is not simply transmission, as in classical formulations, but
continual learning and adaptation over multiple timescales. Partner-specific
common ground quickly emerges from social inferences within dyadic
interactions, while community-wide social conventions are stable priors that
have been abstracted away from interactions with multiple partners. We present
new empirical data alongside simulations showing how our model provides a
computational foundation for several phenomena that have posed a challenge for
previous accounts: (1) the convergence to more efficient referring expressions
across repeated interaction with the same partner, (2) the gradual transfer of
partner-specific common ground to strangers, and (3) the influence of
communicative context on which conventions eventually form.

    

### [[2105.00642] One Model to Rule them All: Towards Zero-Shot Learning for Databases](http://arxiv.org/abs/2105.00642)


  In this paper, we present our vision of so called zero-shot learning for
databases which is a new learning approach for database components. Zero-shot
learning for databases is inspired by recent advances in transfer learning of
models such as GPT-3 and can support a new database out-of-the box without the
need to train a new model. As a first concrete contribution in this paper, we
show the feasibility of zero-shot learning for the task of physical cost
estimation and present very promising initial results. Moreover, as a second
contribution we discuss the core challenges related to zero-shot learning for
databases and present a roadmap to extend zero-shot learning towards many other
tasks beyond cost estimation or even beyond classical database systems and
workloads.

    

### [[2110.12509] Per-Pixel Lung Thickness and Lung Capacity Estimation on Chest X-Rays using Convolutional Neural Networks](http://arxiv.org/abs/2110.12509)


  Estimating the lung depth on x-ray images could provide both an accurate
opportunistic lung volume estimation during clinical routine and improve image
contrast in modern structural chest imaging techniques like x-ray dark-field
imaging. We present a method based on a convolutional neural network that
allows a per-pixel lung thickness estimation and subsequent total lung capacity
estimation. The network was trained and validated using 5250 simulated
radiographs generated from 525 real CT scans. Furthermore, we are able to infer
the model trained with simulation data on real radiographs.
For 35 patients, quantitative and qualitative evaluation was performed on
standard clinical radiographs. The ground-truth for each patient's total lung
volume was defined based on the patients' corresponding CT scan. The
mean-absolute error between the estimated lung volume on the 35 real
radiographs and groundtruth volume was 0.73 liter. Additionally, we predicted
the lung thicknesses on a synthetic dataset of 131 radiographs, where the
mean-absolute error was 0.27 liter. The results show, that it is possible to
transfer the knowledge obtained in a simulation model to real x-ray images.

    

### [[2110.14560] Bugs in Quantum Computing Platforms: An Empirical Study](http://arxiv.org/abs/2110.14560)


  The interest in quantum computing is growing, and with it, the importance of
software platforms to develop quantum programs. Ensuring the correctness of
such platforms is important, and it requires a thorough understanding of the
bugs they typically suffer from. To address this need, this paper presents the
first in-depth study of bugs in quantum computing platforms. We gather and
inspect a set of 223 real-world bugs from 18 open-source quantum computing
platforms. Our study shows that a significant fraction of these bugs (39.9%)
are quantum-specific, calling for dedicated approaches to prevent and find
them. The bugs are spread across various components, but quantum-specific bugs
occur particularly often in components that represent, compile, and optimize
quantum programming abstractions. Many quantum-specific bugs manifest through
unexpected outputs, rather than more obvious signs of misbehavior, such as
crashes. Finally, we present a hierarchy of recurrent bug patterns, including
ten novel, quantum-specific patterns. Our findings not only show the importance
and prevalence bugs in quantum computing platforms, but they help developers to
avoid common mistakes and tool builders to tackle the challenge of preventing,
finding, and fixing these bugs.

    

### [<title data-react-helmet="true">从ICCV 2021看域泛化与域自适应最新研究进展 - 知乎</title>](https://zhuanlan.zhihu.com/p/426728622)