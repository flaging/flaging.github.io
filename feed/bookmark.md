
## 2021-12-10

### [[2112.04602] Adaptive packet transmission in response to anomaly detection in software defined smart meter networks](http://arxiv.org/abs/2112.04602)


  In this paper, we examine a basic smart meter network topology in mininet and
address the issue of congestion over a commodity network, proposing an adaptive
algorithm to cope with varying grid data delivery latencies.

    

### [[2112.04698] Explainable AI for B5G/6G: Technical Aspects, Use Cases, and Research Challenges](http://arxiv.org/abs/2112.04698)


  When 5G began its commercialisation journey around 2020, the discussion on
the vision of 6G also surfaced. Researchers expect 6G to have higher bandwidth,
coverage, reliability, energy efficiency, lower latency, and, more importantly,
an integrated "human-centric" network system powered by artificial intelligence
(AI). Such a 6G network will lead to an excessive number of automated decisions
made every second. These decisions can range widely, from network resource
allocation to collision avoidance for self-driving cars. However, the risk of
losing control over decision-making may increase due to high-speed
data-intensive AI decision-making beyond designers and users' comprehension.
The promising explainable AI (XAI) methods can mitigate such risks by enhancing
the transparency of the black box AI decision-making process. This survey paper
highlights the need for XAI towards the upcoming 6G age in every aspect,
including 6G technologies (e.g., intelligent radio, zero-touch network
management) and 6G use cases (e.g., industry 5.0). Moreover, we summarised the
lessons learned from the recent attempts and outlined important research
challenges in applying XAI for building 6G systems. This research aligns with
goals 9, 11, 16, and 17 of the United Nations Sustainable Development Goals
(UN-SDG), promoting innovation and building infrastructure, sustainable and
inclusive human settlement, advancing justice and strong institutions, and
fostering partnership at the global level.

    

### [[2112.04703] Modelling and Optimization of OAM-MIMO Communication Systems with Unaligned Antennas](http://arxiv.org/abs/2112.04703)


  The orbital angular momentum (OAM) wireless communication technique is
emerging as one of potential techniques for the Sixth generation (6G) wireless
communication system. The most advantage of OAM wireless communication
technique is the natural orthogonality among different OAM states. However, one
of the most disadvantages is the crosstalk among different OAM states which is
widely caused by the atmospheric turbulence and misalignment between
transmitting and receiving antennas. Considering the OAM-based multiple-input
multiple-output (OAM-MIMO) transmission system with unaligned antennas, a new
channel model is proposed for performance analysis. Moreover, a purity model of
the OAM-MIMO transmission system with unaligned antennas is derived for the
non-Kolmogorov turbulence. Furthermore, error probability and capacity models
are derived for OAM-MIMO transmission systems with unaligned antennas. To
overcome the disadvantage caused by unaligned antennas and non-Kolmogorov
turbulence, a new optimization algorithm of OAM state interval is proposed to
improve the capacity of OAM-MIMO transmission system. Numerical results
indicate that the capacity of OAM-MIMO transmission system is improved by the
optimization algorithm. Specifically, the capacity increment of OAM-MIMO
transmission system adopting the optimization algorithm is up to 28.7% and
320.3% when the angle of deflection between transmitting and receiving antennas
is -24 dB and -5 dB, respectively.

    

### [[2112.04737] Asynchronous Semi-Decentralized Federated Edge Learning for Heterogeneous Clients](http://arxiv.org/abs/2112.04737)


  Federated edge learning (FEEL) has drawn much attention as a
privacy-preserving distributed learning framework for mobile edge networks. In
this work, we investigate a novel semi-decentralized FEEL (SD-FEEL)
architecture where multiple edge servers collaborate to incorporate more data
from edge devices in training. Despite the low training latency enabled by fast
edge aggregation, the device heterogeneity in computational resources
deteriorates the efficiency. This paper proposes an asynchronous training
algorithm for SD-FEEL to overcome this issue, where edge servers can
independently set deadlines for the associated client nodes and trigger the
model aggregation. To deal with different levels of staleness, we design a
staleness-aware aggregation scheme and analyze its convergence performance.
Simulation results demonstrate the effectiveness of our proposed algorithm in
achieving faster convergence and better learning performance.

    

### [[2112.05008] Millimeter Wave Localization with Imperfect Training Data using Shallow Neural Networks](http://arxiv.org/abs/2112.05008)


  Millimeter wave (mmWave) localization algorithms exploit the quasi-optical
propagation of mmWave signals, which yields sparse angular spectra at the
receiver. Geometric approaches to angle-based localization typically require to
know the map of the environment and the location of the access points. Thus,
several works have resorted to automated learning in order to infer a device's
location from the properties of the received mmWave signals. However,
collecting training data for such models is a significant burden. In this work,
we propose a shallow neural network model to localize mmWave devices indoors.
This model requires significantly fewer weights than those proposed in the
literature. Therefore, it is amenable for implementation in
resource-constrained hardware, and needs fewer training samples to converge. We
also propose to relieve training data collection efforts by retrieving
(inherently imperfect) location estimates from geometry-based mmWave
localization algorithms. Even in this case, our results show that the proposed
neural networks perform as good as or better than state-of-the-art algorithms.

    

### [[2008.04088] mpNet: variable depth unfolded neural network for massive MIMO channel estimation](http://arxiv.org/abs/2008.04088)


  Massive multiple-input multiple-output (MIMO) communication systems have a
huge potential both in terms of data rate and energy efficiency, although
channel estimation becomes challenging for a large number of antennas. Using a
physical model allows to ease the problem by injecting a priori information
based on the physics of propagation. However, such a model rests on simplifying
assumptions and requires to know precisely the configuration of the system,
which is unrealistic in this http URL this paper we present mpNet, an unfolded
neural network specifically designed for massive MIMO channel estimation. It is
trained online in an unsupervised way. Moreover, mpNet is computationally
efficient and automatically adapts its depth to the signal-to-noise ratio
(SNR). The method we propose adds flexibility to physical channel models by
allowing a base station (BS) to automatically correct its channel estimation
algorithm based on incoming data, without the need for a separate offline
training this http URL is applied to realistic millimeter wave channels and shows
great performance, achieving a channel estimation error almost as low as one
would get with a perfectly calibrated system. It also allows incident detection
and automatic correction, making the BS resilient and able to automatically
adapt to changes in its environment.

    

### [[2103.15924] How Far Can We Go in Compute-less Networking: Computation Correctness and Accuracy](http://arxiv.org/abs/2103.15924)


  Emerging applications such as augmented reality and tactile Internet are
compute-intensive and latency-sensitive, which hampers their running in
constrained end devices alone or in the distant cloud. The stringent
requirements of such application drove to the realization of Edge computing in
which computation is offloaded near to users. Compute-less networking is an
extension of edge computing that aims at reducing computation and abridging
communication by adopting in-network computing and computation reuse.
Computation reuse aims to cache the result of computations and use them to
perform similar tasks in the future and, therefore, avoid redundant
calculations and optimize the use of resources. In this paper, we focus on the
correctness of the final output produced by computation reuse. Since the input
might not be identical but similar, the reuse of previous computation raises
questions about the accuracy of the final results. To this end, we implement a
proof of concept to study and gauge the effectiveness and efficiency of
computation reuse. We are able to reduce task completion time by up to 80%
while ensuring high correctness. We further discuss open challenges and
highlight future research directions.

    

### [[2106.04906] Engineering-Economic Evaluation of Diffractive Non-Line-Of-Sight Backhaul (e3nb): A Techno-economic Model for 3D Wireless Backhaul Assessment](http://arxiv.org/abs/2106.04906)


  Developing ways to affordably deliver broadband connectivity is one of the
major issues of our time. In challenging deployment locations with irregular
terrain, traditional Clear-Line-Of-Sight (CLOS) wireless links can be
uneconomical to deploy, as the number of required towers make infrastructure
investment unviable. With new research focusing on developing wireless
diffractive backhaul technologies to provide Non-Line-Of-Sight (NLOS) links,
this paper evaluates the engineering-economic implications. A Three-Dimensional
(3D) techno-economic assessment framework is developed, utilizing a combination
of remote sensing and viewshed geospatial techniques, in order to quantify the
impact of different wireless backhaul strategies. This framework is applied to
assess both Clear-Line-Of-Sight and diffractive Non-Line-Of-Sight strategies
for deployment in Peru, as well as the islands of Kalimantan and Papua, in
Indonesia. The results find that a hybrid strategy combining the use of
Clear-Line-Of-Sight and diffractive Non-Line-Of-Sight links produces a 9-45
percent cost-efficiency saving, relative to only using traditional
Clear-Line-Of-Sight wireless backhaul links.

    

### [[2112.04492] Daily peak electrical load forecasting with a multi-resolution approach](http://arxiv.org/abs/2112.04492)


  In the context of smart grids and load balancing, daily peak load forecasting
has become a critical activity for stakeholders of the energy industry. An
understanding of peak magnitude and timing is paramount for the implementation
of smart grid strategies such as peak shaving. The modelling approach proposed
in this paper leverages high-resolution and low-resolution information to
forecast daily peak demand size and timing. The resulting multi-resolution
modelling framework can be adapted to different model classes. The key
contributions of this paper are a) a general and formal introduction to the
multi-resolution modelling approach, b) a discussion on modelling approaches at
different resolutions implemented via Generalised Additive Models and Neural
Networks and c) experimental results on real data from the UK electricity
market. The results confirm that the predictive performance of the proposed
modelling approach is competitive with that of low- and high-resolution
alternatives.

    

### [[2112.04494] Deep Q-Learning Market Makers in a Multi-Agent Simulated Stock Market](http://arxiv.org/abs/2112.04494)


  Market makers play a key role in financial markets by providing liquidity.
They usually fill order books with buy and sell limit orders in order to
provide traders alternative price levels to operate. This paper focuses
precisely on the study of these markets makers strategies from an agent-based
perspective. In particular, we propose the application of Reinforcement
Learning (RL) for the creation of intelligent market markers in simulated stock
markets. This research analyzes how RL market maker agents behaves in
non-competitive (only one RL market maker learning at the same time) and
competitive scenarios (multiple RL market markers learning at the same time),
and how they adapt their strategies in a Sim2Real scope with interesting
results. Furthermore, it covers the application of policy transfer between
different experiments, describing the impact of competing environments on RL
agents performance. RL and deep RL techniques are proven as profitable market
maker approaches, leading to a better understanding of their behavior in stock
markets.

    

### [[2112.04527] Building Quantum Field Theories Out of Neurons](http://arxiv.org/abs/2112.04527)


  An approach to field theory is studied in which fields are comprised of $N$
constituent random neurons. Gaussian theories arise in the infinite-$N$ limit
when neurons are independently distributed, via the Central Limit Theorem,
while interactions arise due to finite-$N$ effects or non-independently
distributed neurons. Euclidean-invariant ensembles of neurons are engineered,
with tunable two-point function, yielding families of Euclidean-invariant field
theories. Some Gaussian, Euclidean invariant theories are reflection positive,
which allows for analytic continuation to a Lorentz-invariant quantum field
theory. Examples are presented that yield dual theories at infinite-$N$, but
have different symmetries at finite-$N$. Landscapes of classical field
configurations are determined by local maxima of parameter distributions.
Predictions arise from mixed field-neuron correlators. Near-Gaussianity is
exhibited at large-$N$, potentially explaining a feature of field theories in
Nature.

    

### [[2112.04552] PATO: Producibility-Aware Topology Optimization using Deep Learning for Metal Additive Manufacturing](http://arxiv.org/abs/2112.04552)


  In this paper, we propose PATO-a producibility-aware topology optimization
(TO) framework to help efficiently explore the design space of components
fabricated using metal additive manufacturing (AM), while ensuring
manufacturability with respect to cracking. Specifically, parts fabricated
through Laser Powder Bed Fusion are prone to defects such as warpage or
cracking due to high residual stress values generated from the steep thermal
gradients produced during the build process. Maturing the design for such parts
and planning their fabrication can span months to years, often involving
multiple handoffs between design and manufacturing engineers. PATO is based on
the a priori discovery of crack-free designs, so that the optimized part can be
built defect-free at the outset. To ensure that the design is crack free during
optimization, producibility is explicitly encoded within the standard
formulation of TO, using a crack index. Multiple crack indices are explored and
using experimental validation, maximum shear strain index (MSSI) is shown to be
an accurate crack index. Simulating the build process is a coupled,
multi-physics computation and incorporating it in the TO loop can be
computationally prohibitive. We leverage the current advances in deep
convolutional neural networks and present a high-fidelity surrogate model based
on an Attention-based U-Net architecture to predict the MSSI values as a
spatially varying field over the part's domain. Further, we employ automatic
differentiation to directly compute the gradient of maximum MSSI with respect
to the input design variables and augment it with the performance-based
sensitivity field to optimize the design while considering the trade-off
between weight, manufacturability, and functionality. We demonstrate the
effectiveness of the proposed method through benchmark studies in 3D as well as
experimental validation.

    

### [[2112.04553] Recent Advances in Reinforcement Learning in Finance](http://arxiv.org/abs/2112.04553)


  The rapid changes in the finance industry due to the increasing amount of
data have revolutionized the techniques on data processing and data analysis
and brought new theoretical and computational challenges. In contrast to
classical stochastic control theory and other analytical approaches for solving
financial decision-making problems that heavily reply on model assumptions, new
developments from reinforcement learning (RL) are able to make full use of the
large amount of financial data with fewer model assumptions and to improve
decisions in complex financial environments. This survey paper aims to review
the recent developments and use of RL approaches in finance. We give an
introduction to Markov decision processes, which is the setting for many of the
commonly used RL approaches. Various algorithms are then introduced with a
focus on value and policy based methods that do not require any model
assumptions. Connections are made with neural networks to extend the framework
to encompass deep RL algorithms. Our survey concludes by discussing the
application of these RL algorithms in a variety of decision-making problems in
finance, including optimal execution, portfolio optimization, option pricing
and hedging, market making, smart order routing, and robo-advising.

    

### [[2112.04554] Whose Ground Truth? Accounting for Individual and Collective Identities Underlying Dataset Annotation](http://arxiv.org/abs/2112.04554)


  Human annotations play a crucial role in machine learning (ML) research and
development. However, the ethical considerations around the processes and
decisions that go into building ML datasets has not received nearly enough
attention. In this paper, we survey an array of literature that provides
insights into ethical considerations around crowdsourced dataset annotation. We
synthesize these insights, and lay out the challenges in this space along two
layers: (1) who the annotator is, and how the annotators' lived experiences can
impact their annotations, and (2) the relationship between the annotators and
the crowdsourcing platforms and what that relationship affords them. Finally,
we put forth a concrete set of recommendations and considerations for dataset
developers at various stages of the ML data pipeline: task formulation,
selection of annotators, platform and infrastructure choices, dataset analysis
and evaluation, and dataset documentation and release.

    

### [[2112.04558] SoK: Anti-Facial Recognition Technology](http://arxiv.org/abs/2112.04558)


  The rapid adoption of facial recognition (FR) technology by both government
and commercial entities in recent years has raised concerns about civil
liberties and privacy. In response, a broad suite of so-called "anti-facial
recognition" (AFR) tools has been developed to help users avoid unwanted facial
recognition. The set of AFR tools proposed in the last few years is
wide-ranging and rapidly evolving, necessitating a step back to consider the
broader design space of AFR systems and long-term challenges. This paper aims
to fill that gap and provides the first comprehensive analysis of the AFR
research landscape. Using the operational stages of FR systems as a starting
point, we create a systematic framework for analyzing the benefits and
tradeoffs of different AFR approaches. We then consider both technical and
social challenges facing AFR tools and propose directions for future research
in this field.

    

### [[2112.04564] CoSSL: Co-Learning of Representation and Classifier for Imbalanced Semi-Supervised Learning](http://arxiv.org/abs/2112.04564)


  In this paper, we propose a novel co-learning framework (CoSSL) with
decoupled representation learning and classifier learning for imbalanced SSL.
To handle the data imbalance, we devise Tail-class Feature Enhancement (TFE)
for classifier learning. Furthermore, the current evaluation protocol for
imbalanced SSL focuses only on balanced test sets, which has limited
practicality in real-world scenarios. Therefore, we further conduct a
comprehensive evaluation under various shifted test distributions. In
experiments, we show that our approach outperforms other methods over a large
range of shifted distributions, achieving state-of-the-art performance on
benchmark datasets ranging from CIFAR-10, CIFAR-100, ImageNet, to Food-101. Our
code will be made publicly available.

    

### [[2112.04571] Ambiguous Dynamic Treatment Regimes: A Reinforcement Learning Approach](http://arxiv.org/abs/2112.04571)


  A main research goal in various studies is to use an observational data set
and provide a new set of counterfactual guidelines that can yield causal
improvements. Dynamic Treatment Regimes (DTRs) are widely studied to formalize
this process. However, available methods in finding optimal DTRs often rely on
assumptions that are violated in real-world applications (e.g., medical
decision-making or public policy), especially when (a) the existence of
unobserved confounders cannot be ignored, and (b) the unobserved confounders
are time-varying (e.g., affected by previous actions). When such assumptions
are violated, one often faces ambiguity regarding the underlying causal model
that is needed to be assumed to obtain an optimal DTR. This ambiguity is
inevitable, since the dynamics of unobserved confounders and their causal
impact on the observed part of the data cannot be understood from the observed
data. Motivated by a case study of finding superior treatment regimes for
patients who underwent transplantation in our partner hospital and faced a
medical condition known as New Onset Diabetes After Transplantation (NODAT), we
extend DTRs to a new class termed Ambiguous Dynamic Treatment Regimes (ADTRs),
in which the casual impact of treatment regimes is evaluated based on a "cloud"
of potential causal models. We then connect ADTRs to Ambiguous Partially
Observable Mark Decision Processes (APOMDPs) proposed by Saghafian (2018), and
develop two Reinforcement Learning methods termed Direct Augmented V-Learning
(DAV-Learning) and Safe Augmented V-Learning (SAV-Learning), which enable using
the observed data to efficiently learn an optimal treatment regime. We
establish theoretical results for these learning methods, including (weak)
consistency and asymptotic normality. We further evaluate the performance of
these learning methods both in our case study and in simulation experiments.

    

### [[2112.04572] Merging Subject Matter Expertise and Deep Convolutional Neural Network for State-Based Online Machine-Part Interaction Classification](http://arxiv.org/abs/2112.04572)


  Machine-part interaction classification is a key capability required by
Cyber-Physical Systems (CPS), a pivotal enabler of Smart Manufacturing (SM).
While previous relevant studies on the subject have primarily focused on time
series classification, change point detection is equally important because it
provides temporal information on changes in behavior of the machine. In this
work, we address point detection and time series classification for
machine-part interactions with a deep Convolutional Neural Network (CNN) based
framework. The CNN in this framework utilizes a two-stage encoder-classifier
structure for efficient feature representation and convenient deployment
customization for CPS. Though data-driven, the design and optimization of the
framework are Subject Matter Expertise (SME) guided. An SME defined Finite
State Machine (FSM) is incorporated into the framework to prohibit intermittent
misclassifications. In the case study, we implement the framework to perform
machine-part interaction classification on a milling machine, and the
performance is evaluated using a testing dataset and deployment simulations.
The implementation achieved an average F1-Score of 0.946 across classes on the
testing dataset and an average delay of 0.24 seconds on the deployment
simulations.

    

### [[2112.04573] Application of Artificial Intelligence and Machine Learning in Libraries: A Systematic Review](http://arxiv.org/abs/2112.04573)


  As the concept and implementation of cutting-edge technologies like
artificial intelligence and machine learning has become relevant, academics,
researchers and information professionals involve research in this area. The
objective of this systematic literature review is to provide a synthesis of
empirical studies exploring application of artificial intelligence and machine
learning in libraries. To achieve the objectives of the study, a systematic
literature review was conducted based on the original guidelines proposed by
Kitchenham et al. (2009). Data was collected from Web of Science, Scopus, LISA
and LISTA databases. Following the rigorous/ established selection process, a
total of thirty-two articles were finally selected, reviewed and analyzed to
summarize on the application of AI and ML domain and techniques which are most
often used in libraries. Findings show that the current state of the AI and ML
research that is relevant with the LIS domain mainly focuses on theoretical
works. However, some researchers also emphasized on implementation projects or
case studies. This study will provide a panoramic view of AI and ML in
libraries for researchers, practitioners and educators for furthering the more
technology-oriented approaches, and anticipating future innovation pathways.

    

### [[2112.04575] Adaptive Kernel Graph Neural Network](http://arxiv.org/abs/2112.04575)


  Graph neural networks (GNNs) have demonstrated great success in
representation learning for graph-structured data. The layer-wise graph
convolution in GNNs is shown to be powerful at capturing graph topology. During
this process, GNNs are usually guided by pre-defined kernels such as Laplacian
matrix, adjacency matrix, or their variants. However, the adoptions of
pre-defined kernels may restrain the generalities to different graphs: mismatch
between graph and kernel would entail sub-optimal performance. For example,
GNNs that focus on low-frequency information may not achieve satisfactory
performance when high-frequency information is significant for the graphs, and
vice versa. To solve this problem, in this paper, we propose a novel framework
- i.e., namely Adaptive Kernel Graph Neural Network (AKGNN) - which learns to
adapt to the optimal graph kernel in a unified manner at the first attempt. In
the proposed AKGNN, we first design a data-driven graph kernel learning
mechanism, which adaptively modulates the balance between all-pass and low-pass
filters by modifying the maximal eigenvalue of the graph Laplacian. Through
this process, AKGNN learns the optimal threshold between high and low frequency
signals to relieve the generality problem. Later, we further reduce the number
of parameters by a parameterization trick and enhance the expressive power by a
global readout function. Extensive experiments are conducted on acknowledged
benchmark datasets and promising results demonstrate the outstanding
performance of our proposed AKGNN by comparison with state-of-the-art GNNs. The
source code is publicly available at: this https URL.

    

### [[2112.04583] Estimating Divergences in High Dimensions](http://arxiv.org/abs/2112.04583)


  The problem of estimating the divergence between 2 high dimensional
distributions with limited samples is an important problem in various fields
such as machine learning. Although previous methods perform well with moderate
dimensional data, their accuracy starts to degrade in situations with 100s of
binary variables. Therefore, we propose the use of decomposable models for
estimating divergences in high dimensional data. These allow us to factorize
the estimated density of the high-dimensional distribution into a product of
lower dimensional functions. We conduct formal and experimental analyses to
explore the properties of using decomposable models in the context of
divergence estimation. To this end, we show empirically that estimating the
Kullback-Leibler divergence using decomposable models from a maximum likelihood
estimator outperforms existing methods for divergence estimation in situations
where dimensionality is high and useful decomposable models can be learnt from
the available data.

    

### [[2112.04585] STAF: A Spatio-Temporal Attention Fusion Network for Few-shot Video Classification](http://arxiv.org/abs/2112.04585)


  We propose STAF, a Spatio-Temporal Attention Fusion network for few-shot
video classification. STAF first extracts coarse-grained spatial and temporal
features of videos by applying a 3D Convolution Neural Networks embedding
network. It then fine-tunes the extracted features using self-attention and
cross-attention networks. Last, STAF applies a lightweight fusion network and a
nearest neighbor classifier to classify each query video. To evaluate STAF, we
conduct extensive experiments on three benchmarks (UCF101, HMDB51, and
Something-Something-V2). The experimental results show that STAF improves
state-of-the-art accuracy by a large margin, e.g., STAF increases the five-way
one-shot accuracy by 5.3% and 7.0% for UCF101 and HMDB51, respectively.

    

### [[2112.04590] The perils of being unhinged: On the accuracy of classifiers minimizing a noise-robust convex loss](http://arxiv.org/abs/2112.04590)


  van Rooyen et al. introduced a notion of convex loss functions being robust
to random classification noise, and established that the "unhinged" loss
function is robust in this sense. In this note we study the accuracy of binary
classifiers obtained by minimizing the unhinged loss, and observe that even for
simple linearly separable data distributions, minimizing the unhinged loss may
only yield a binary classifier with accuracy no better than random guessing.

    

### [[2112.04591] Variational Regularization in Inverse Problems and Machine Learning](http://arxiv.org/abs/2112.04591)


  This paper discusses basic results and recent developments on variational
regularization methods, as developed for inverse problems. In a typical setup
we review basic properties needed to obtain a convergent regularization scheme
and further discuss the derivation of quantitative estimates respectively
needed ingredients such as Bregman distances for convex functionals.
In addition to the approach developed for inverse problems we will also
discuss variational regularization in machine learning and work out some
connections to the classical regularization theory. In particular we will
discuss a reinterpretation of machine learning problems in the framework of
regularization theory and a reinterpretation of variational methods for inverse
problems in the framework of risk minimization. Moreover, we establish some
previously unknown connections between error estimates in Bregman distances and
generalization errors.

    

### [[2112.04598] InvGAN: Invertable GANs](http://arxiv.org/abs/2112.04598)


  Generation of photo-realistic images, semantic editing and representation
learning are a few of many potential applications of high resolution generative
models. Recent progress in GANs have established them as an excellent choice
for such tasks. However, since they do not provide an inference model, image
editing or downstream tasks such as classification can not be done on real
images using the GAN latent space. Despite numerous efforts to train an
inference model or design an iterative method to invert a pre-trained
generator, previous methods are dataset (e.g. human face images) and
architecture (e.g. StyleGAN) specific. These methods are nontrivial to extend
to novel datasets or architectures. We propose a general framework that is
agnostic to architecture and datasets. Our key insight is that, by training the
inference and the generative model together, we allow them to adapt to each
other and to converge to a better quality model. Our \textbf{InvGAN}, short for
Invertable GAN, successfully embeds real images to the latent space of a high
quality generative model. This allows us to perform image inpainting, merging,
interpolation and online data augmentation. We demonstrate this with extensive
qualitative and quantitative experiments.

    

### [[2112.04604] Regularization methods for the short-term forecasting of the Italian electric load](http://arxiv.org/abs/2112.04604)


  The problem of forecasting the whole 24 profile of the Italian electric load
is addressed as a multitask learning problem, whose complexity is kept under
control via alternative regularization methods. In view of the quarter-hourly
samplings, 96 predictors are used, each of which linearly depends on 96
regressors. The 96x96 matrix weights form a 96x96 matrix, that can be seen and
displayed as a surface sampled on a square domain. Different regularization and
sparsity approaches to reduce the degrees of freedom of the surface were
explored, comparing the obtained forecasts with those of the Italian
Transmission System Operator Terna. Besides outperforming Terna in terms of
quarter-hourly mean absolute percentage error and mean absolute error, the
prediction residuals turned out to be weakly correlated with Terna, which
suggests that further improvement could ensue from forecasts aggregation. In
fact, the aggregated forecasts yielded further relevant drops in terms of
quarter-hourly and daily mean absolute percentage error, mean absolute error
and root mean square error (up to 30%) over the three test years considered.

    

### [[2112.04605] Prediction of Adverse Biological Effects of Chemicals Using Knowledge Graph Embeddings](http://arxiv.org/abs/2112.04605)


  We have created a knowledge graph based on major data sources used in
ecotoxicological risk assessment. We have applied this knowledge graph to an
important task in risk assessment, namely chemical effect prediction. We have
evaluated nine knowledge graph embedding models from a selection of geometric,
decomposition, and convolutional models on this prediction task. We show that
using knowledge graph embeddings can increase the accuracy of effect prediction
with neural networks. Furthermore, we have implemented a fine-tuning
architecture which adapts the knowledge graph embeddings to the effect
prediction task and leads to a better performance. Finally, we evaluate certain
characteristics of the knowledge graph embedding models to shed light on the
individual model performance.

    

### [[2112.04608] Enhancing Food Intake Tracking in Long-Term Care with Automated Food Imaging and Nutrient Intake Tracking (AFINI-T) Technology](http://arxiv.org/abs/2112.04608)


  Half of long-term care (LTC) residents are malnourished increasing
hospitalization, mortality, morbidity, with lower quality of life. Current
tracking methods are subjective and time consuming. This paper presents the
automated food imaging and nutrient intake tracking (AFINI-T) technology
designed for LTC. We propose a novel convolutional autoencoder for food
classification, trained on an augmented UNIMIB2016 dataset and tested on our
simulated LTC food intake dataset (12 meal scenarios; up to 15 classes each;
top-1 classification accuracy: 88.9%; mean intake error: -0.4 mL$\pm$36.7 mL).
Nutrient intake estimation by volume was strongly linearly correlated with
nutrient estimates from mass ($r^2$ 0.92 to 0.99) with good agreement between
methods ($\sigma$= -2.7 to -0.01; zero within each of the limits of agreement).
The AFINI-T approach is a deep-learning powered computational nutrient sensing
system that may provide a novel means for more accurately and objectively
tracking LTC resident food intake to support and prevent malnutrition tracking
strategies.

    

### [[2112.04612] Gaussian Process Constraint Learning for Scalable Chance-Constrained Motion Planning from Demonstrations](http://arxiv.org/abs/2112.04612)


  We propose a method for learning constraints represented as Gaussian
processes (GPs) from locally-optimal demonstrations. Our approach uses the
Karush-Kuhn-Tucker (KKT) optimality conditions to determine where on the
demonstrations the constraint is tight, and a scaling of the constraint
gradient at those states. We then train a GP representation of the constraint
which is consistent with and which generalizes this information. We further
show that the GP uncertainty can be used within a kinodynamic RRT to plan
probabilistically-safe trajectories, and that we can exploit the GP structure
within the planner to exactly achieve a specified safety probability. We
demonstrate our method can learn complex, nonlinear constraints demonstrated on
a 5D nonholonomic car, a 12D quadrotor, and a 3-link planar arm, all while
requiring minimal prior information on the constraint. Our results suggest the
learned GP constraint is accurate, outperforming previous constraint learning
methods that require more a priori knowledge.

    

### [[2112.04620] Calibration Improves Bayesian Optimization](http://arxiv.org/abs/2112.04620)


  Bayesian optimization is a procedure that allows obtaining the global optimum
of black-box functions and that is useful in applications such as
hyper-parameter optimization. Uncertainty estimates over the shape of the
objective function are instrumental in guiding the optimization process.
However, these estimates can be inaccurate if the objective function violates
assumptions made within the underlying model (e.g., Gaussianity). We propose a
simple algorithm to calibrate the uncertainty of posterior distributions over
the objective function as part of the Bayesian optimization process. We show
that by improving the uncertainty estimates of the posterior distribution with
calibration, Bayesian optimization makes better decisions and arrives at the
global optimum in fewer steps. We show that this technique improves the
performance of Bayesian optimization on standard benchmark functions and
hyperparameter optimization tasks.

    

### [[2112.04624] Deep Molecular Representation Learning via Fusing Physical and Chemical Information](http://arxiv.org/abs/2112.04624)


  Molecular representation learning is the first yet vital step in combining
deep learning and molecular science. To push the boundaries of molecular
representation learning, we present PhysChem, a novel neural architecture that
learns molecular representations via fusing physical and chemical information
of molecules. PhysChem is composed of a physicist network (PhysNet) and a
chemist network (ChemNet). PhysNet is a neural physical engine that learns
molecular conformations through simulating molecular dynamics with
parameterized forces; ChemNet implements geometry-aware deep message-passing to
learn chemical / biomedical properties of molecules. Two networks specialize in
their own tasks and cooperate by providing expertise to each other. By fusing
physical and chemical information, PhysChem achieved state-of-the-art
performances on MoleculeNet, a standard molecular machine learning benchmark.
The effectiveness of PhysChem was further corroborated on cutting-edge datasets
of SARS-CoV-2.

    

### [[2112.04629] Transferability Properties of Graph Neural Networks](http://arxiv.org/abs/2112.04629)


  Graph neural networks (GNNs) are deep convolutional architectures consisting
of layers composed by graph convolutions and pointwise nonlinearities. Due to
their invariance and stability properties, GNNs are provably successful at
learning representations from network data. However, training them requires
matrix computations which can be expensive for large graphs. To address this
limitation, we investigate the ability of GNNs to be transferred across graphs.
We consider graphons, which are both graph limits and generative models for
weighted and stochastic graphs, to define limit objects of graph convolutions
and GNNs -- graphon convolutions and graphon neural networks (WNNs) -- which we
use as generative models for graph convolutions and GNNs. We show that these
graphon filters and WNNs can be approximated by graph filters and GNNs sampled
from them on weighted and stochastic graphs. Using these results, we then
derive error bounds for transferring graph filters and GNNs across such graphs.
These bounds show that transferability increases with the graph size, and
reveal a tradeoff between transferability and spectral discriminability which
in GNNs is alleviated by the pointwise nonlinearities. These findings are
further verified empirically in numerical experiments in movie recommendation
and decentralized robot control.

    

### [[2112.04640] Differentially Private Ensemble Classifiers for Data Streams](http://arxiv.org/abs/2112.04640)


  Learning from continuous data streams via classification/regression is
prevalent in many domains. Adapting to evolving data characteristics (concept
drift) while protecting data owners' private information is an open challenge.
We present a differentially private ensemble solution to this problem with two
distinguishing features: it allows an \textit{unbounded} number of ensemble
updates to deal with the potentially never-ending data streams under a fixed
privacy budget, and it is \textit{model agnostic}, in that it treats any
pre-trained differentially private classification/regression model as a
black-box. Our method outperforms competitors on real-world and simulated
datasets for varying settings of privacy, concept drift, and data distribution.

    

### [[2112.04643] Autoregressive Quantile Flows for Predictive Uncertainty Estimation](http://arxiv.org/abs/2112.04643)


  Numerous applications of machine learning involve predicting flexible
probability distributions over model outputs. We propose Autoregressive
Quantile Flows, a flexible class of probabilistic models over high-dimensional
variables that can be used to accurately capture predictive aleatoric
uncertainties. These models are instances of autoregressive flows trained using
a novel objective based on proper scoring rules, which simplifies the
calculation of computationally expensive determinants of Jacobians during
training and supports new types of neural architectures. We demonstrate that
these models can be used to parameterize predictive conditional distributions
and improve the quality of probabilistic predictions on time series forecasting
and object detection.

    

### [[2112.04645] BACON: Band-limited Coordinate Networks for Multiscale Scene Representation](http://arxiv.org/abs/2112.04645)


  Coordinate-based networks have emerged as a powerful tool for 3D
representation and scene reconstruction. These networks are trained to map
continuous input coordinates to the value of a signal at each point. Still,
current architectures are black boxes: their spectral characteristics cannot be
easily analyzed, and their behavior at unsupervised points is difficult to
predict. Moreover, these networks are typically trained to represent a signal
at a single scale, and so naive downsampling or upsampling results in
artifacts. We introduce band-limited coordinate networks (BACON), a network
architecture with an analytical Fourier spectrum. BACON has predictable
behavior at unsupervised points, can be designed based on the spectral
characteristics of the represented signal, and can represent signals at
multiple scales without explicit supervision. We demonstrate BACON for
multiscale neural representation of images, radiance fields, and 3D scenes
using signed distance functions and show that it outperforms conventional
single-scale coordinate networks in terms of interpretability and quality.

    

### [[2112.04660] A Fully Single Loop Algorithm for Bilevel Optimization without Hessian Inverse](http://arxiv.org/abs/2112.04660)


  In this paper, we propose a new Hessian inverse free Fully Single Loop
Algorithm (FSLA) for bilevel optimization problems. Classic algorithms for
bilevel optimization admit a double loop structure which is computationally
expensive. Recently, several single loop algorithms have been proposed with
optimizing the inner and outer variable alternatively. However, these
algorithms not yet achieve fully single loop. As they overlook the loop needed
to evaluate the hyper-gradient for a given inner and outer state. In order to
develop a fully single loop algorithm, we first study the structure of the
hyper-gradient and identify a general approximation formulation of
hyper-gradient computation that encompasses several previous common approaches,
e.g. back-propagation through time, conjugate gradient, \emph{etc.} Based on
this formulation, we introduce a new state variable to maintain the historical
hyper-gradient information. Combining our new formulation with the alternative
update of the inner and outer variables, we propose an efficient fully single
loop algorithm. We theoretically show that the error generated by the new state
can be bounded and our algorithm converges with the rate of $O(\epsilon^{-2})$.
Finally, we verify the efficacy our algorithm empirically through multiple
bilevel optimization based machine learning tasks.

    

### [[2112.04677] A Note on Comparison of F-measures](http://arxiv.org/abs/2112.04677)


  We comment on a recent TKDE paper "Linear Approximation of F-measure for the
Performance Evaluation of Classification Algorithms on Imbalanced Data Sets",
and make two improvements related to comparison of F-measures for two
prediction rules.

    

### [[2112.04682] Clairvoyance: Intelligent Route Planning for Electric Buses Based on Urban Big Data](http://arxiv.org/abs/2112.04682)


  Nowadays many cities around the world have introduced electric buses to
optimize urban traffic and reduce local carbon emissions. In order to cut
carbon emissions and maximize the utility of electric buses, it is important to
choose suitable routes for them. Traditionally, route selection is on the basis
of dedicated surveys, which are costly in time and labor. In this paper, we
mainly focus attention on planning electric bus routes intelligently, depending
on the unique needs of each region throughout the city. We propose
Clairvoyance, a route planning system that leverages a deep neural network and
a multilayer perceptron to predict the future people's trips and the future
transportation carbon emission in the whole city, respectively. Given the
future information of people's trips and transportation carbon emission, we
utilize a greedy mechanism to recommend bus routes for electric buses that will
depart in an ideal state. Furthermore, representative features of the two
neural networks are extracted from the heterogeneous urban datasets. We
evaluate our approach through extensive experiments on real-world data sources
in Zhuhai, China. The results show that our designed neural network-based
algorithms are consistently superior to the typical baselines. Additionally,
the recommended routes for electric buses are helpful in reducing the peak
value of carbon emissions and making full use of electric buses in the city.

    

### [[2112.04684] Trajectory-Constrained Deep Latent Visual Attention for Improved Local Planning in Presence of Heterogeneous Terrain](http://arxiv.org/abs/2112.04684)


  We present a reward-predictive, model-based deep learning method featuring
trajectory-constrained visual attention for use in mapless, local visual
navigation tasks. Our method learns to place visual attention at locations in
latent image space which follow trajectories caused by vehicle control actions
to enhance predictive accuracy during planning. The attention model is jointly
optimized by the task-specific loss and an additional trajectory-constraint
loss, allowing adaptability yet encouraging a regularized structure for
improved generalization and reliability. Importantly, visual attention is
applied in latent feature map space instead of raw image space to promote
efficient planning. We validated our model in visual navigation tasks of
planning low turbulence, collision-free trajectories in off-road settings and
hill climbing with locking differentials in the presence of slippery terrain.
Experiments involved randomized procedural generated simulation and real-world
environments. We found our method improved generalization and learning
efficiency when compared to no-attention and self-attention alternatives.

    

### [[2112.04704] Ymir: A Supervised Ensemble Framework for Multivariate Time Series Anomaly Detection](http://arxiv.org/abs/2112.04704)


  We proposed a multivariate time series anomaly detection frame-work Ymir,
which leverages ensemble learning and supervisedlearning technology to
efficiently learn and adapt to anomaliesin real-world system applications. Ymir
integrates several currentlywidely used unsupervised anomaly detection models
through anensemble learning method, and thus can provide robust frontalanomaly
detection results in unsupervised scenarios. In a super-vised setting, domain
experts and system users discuss and providelabels (anomalous or not) for the
training data, which reflects theiranomaly detection criteria for the specific
system. Ymir leveragesthe aforementioned unsupervised methods to extract rich
and usefulfeature representations from the raw multivariate time series
data,then combines the features and labels with a supervised classifier todo
anomaly detection. We evaluated Ymir on internal multivariatetime series
datasets from large monitoring systems and achievedgood anomaly detection
performance.

    

### [[2112.04716] DR3: Value-Based Deep Reinforcement Learning Requires Explicit Regularization](http://arxiv.org/abs/2112.04716)


  Despite overparameterization, deep networks trained via supervised learning
are easy to optimize and exhibit excellent generalization. One hypothesis to
explain this is that overparameterized deep networks enjoy the benefits of
implicit regularization induced by stochastic gradient descent, which favors
parsimonious solutions that generalize well on test inputs. It is reasonable to
surmise that deep reinforcement learning (RL) methods could also benefit from
this effect. In this paper, we discuss how the implicit regularization effect
of SGD seen in supervised learning could in fact be harmful in the offline deep
RL setting, leading to poor generalization and degenerate feature
representations. Our theoretical analysis shows that when existing models of
implicit regularization are applied to temporal difference learning, the
resulting derived regularizer favors degenerate solutions with excessive
"aliasing", in stark contrast to the supervised learning case. We back up these
findings empirically, showing that feature representations learned by a deep
network value function trained via bootstrapping can indeed become degenerate,
aliasing the representations for state-action pairs that appear on either side
of the Bellman backup. To address this issue, we derive the form of this
implicit regularizer and, inspired by this derivation, propose a simple and
effective explicit regularizer, called DR3, that counteracts the undesirable
effects of this implicit regularizer. When combined with existing offline RL
methods, DR3 substantially improves performance and stability, alleviating
unlearning in Atari 2600 games, D4RL domains and robotic manipulation from
images.

    

### [[2112.04720] Amicable Aid: Turning Adversarial Attack to Benefit Classification](http://arxiv.org/abs/2112.04720)


  While adversarial attacks on deep image classification models pose serious
security concerns in practice, this paper suggests a novel paradigm where the
concept of adversarial attacks can benefit classification performance, which we
call amicable aid. We show that by taking the opposite search direction of
perturbation, an image can be converted to another yielding higher confidence
by the classification model and even a wrongly classified image can be made to
be correctly classified. Furthermore, with a large amount of perturbation, an
image can be made unrecognizable by human eyes, while it is correctly
recognized by the model. The mechanism of the amicable aid is explained in the
viewpoint of the underlying natural image manifold. We also consider universal
amicable perturbations, i.e., a fixed perturbation can be applied to multiple
images to improve their classification results. While it is challenging to find
such perturbations, we show that making the decision boundary as perpendicular
to the image manifold as possible via training with modified data is effective
to obtain a model for which universal amicable perturbations are more easily
found. Finally, we discuss several application scenarios where the amicable aid
can be useful, including secure image communication, privacy-preserving image
communication, and protection against adversarial attacks.

    

### [[2112.04728] Reducing Catastrophic Forgetting in Self Organizing Maps with Internally-Induced Generative Replay](http://arxiv.org/abs/2112.04728)


  A lifelong learning agent is able to continually learn from potentially
infinite streams of pattern sensory data. One major historic difficulty in
building agents that adapt in this way is that neural systems struggle to
retain previously-acquired knowledge when learning from new samples. This
problem is known as catastrophic forgetting (interference) and remains an
unsolved problem in the domain of machine learning to this day. While
forgetting in the context of feedforward networks has been examined extensively
over the decades, far less has been done in the context of alternative
architectures such as the venerable self-organizing map (SOM), an unsupervised
neural model that is often used in tasks such as clustering and dimensionality
reduction. Although the competition among its internal neurons might carry the
potential to improve memory retention, we observe that a fixed-sized SOM
trained on task incremental data, i.e., it receives data points related to
specific classes at certain temporal increments, experiences significant
forgetting. In this study, we propose the continual SOM (c-SOM), a model that
is capable of reducing its own forgetting when processing information.

    

### [[2112.04731] Mimicking the Oracle: An Initial Phase Decorrelation Approach for Class Incremental Learning](http://arxiv.org/abs/2112.04731)


  Class Incremental Learning (CIL) aims at learning a multi-class classifier in
a phase-by-phase manner, in which only data of a subset of the classes are
provided at each phase. Previous works mainly focus on mitigating forgetting in
phases after the initial one. However, we find that improving CIL at its
initial phase is also a promising direction. Specifically, we experimentally
show that directly encouraging CIL Learner at the initial phase to output
similar representations as the model jointly trained on all classes can greatly
boost the CIL performance. Motivated by this, we study the difference between a
naïvely-trained initial-phase model and the oracle model. Specifically, since
one major difference between these two models is the number of training
classes, we investigate how such difference affects the model representations.
We find that, with fewer training classes, the data representations of each
class lie in a long and narrow region; with more training classes, the
representations of each class scatter more uniformly. Inspired by this
observation, we propose Class-wise Decorrelation (CwD) that effectively
regularizes representations of each class to scatter more uniformly, thus
mimicking the model jointly trained with all classes (i.e., the oracle model).
Our CwD is simple to implement and easy to plug into existing methods.
Extensive experiments on various benchmark datasets show that CwD consistently
and significantly improves the performance of existing state-of-the-art methods
by around 1\% to 3\%. Code will be released.

    

### [[2112.04734] New Tight Relaxations of Rank Minimization for Multi-Task Learning](http://arxiv.org/abs/2112.04734)


  Multi-task learning has been observed by many researchers, which supposes
that different tasks can share a low-rank common yet latent subspace. It means
learning multiple tasks jointly is better than learning them independently. In
this paper, we propose two novel multi-task learning formulations based on two
regularization terms, which can learn the optimal shared latent subspace by
minimizing the exactly $k$ minimal singular values. The proposed regularization
terms are the more tight approximations of rank minimization than trace norm.
But it's an NP-hard problem to solve the exact rank minimization problem.
Therefore, we design a novel re-weighted based iterative strategy to solve our
models, which can tactically handle the exact rank minimization problem by
setting a large penalizing parameter. Experimental results on benchmark
datasets demonstrate that our methods can correctly recover the low-rank
structure shared across tasks, and outperform related multi-task learning
methods.

    

### [[2112.04735] From Good to Best: Two-Stage Training for Cross-lingual Machine Reading Comprehension](http://arxiv.org/abs/2112.04735)


  Cross-lingual Machine Reading Comprehension (xMRC) is challenging due to the
lack of training data in low-resource languages. The recent approaches use
training data only in a resource-rich language like English to fine-tune
large-scale cross-lingual pre-trained language models. Due to the big
difference between languages, a model fine-tuned only by a source language may
not perform well for target languages. Interestingly, we observe that while the
top-1 results predicted by the previous approaches may often fail to hit the
ground-truth answers, the correct answers are often contained in the top-k
predicted results. Based on this observation, we develop a two-stage approach
to enhance the model performance. The first stage targets at recall: we design
a hard-learning (HL) algorithm to maximize the likelihood that the top-k
predictions contain the accurate answer. The second stage focuses on precision:
an answer-aware contrastive learning (AA-CL) mechanism is developed to learn
the fine difference between the accurate answer and other candidates. Our
extensive experiments show that our model significantly outperforms a series of
strong baselines on two cross-lingual MRC benchmark datasets.

    

### [[2112.04755] High-Dimensional Stock Portfolio Trading with Deep Reinforcement Learning](http://arxiv.org/abs/2112.04755)


  This paper proposes a Deep Reinforcement Learning algorithm for financial
portfolio trading based on Deep Q-learning. The algorithm is capable of trading
high-dimensional portfolios from cross-sectional datasets of any size which may
include data gaps and non-unique history lengths in the assets. We sequentially
set up environments by sampling one asset for each environment while rewarding
investments with the resulting asset's return and cash reservation with the
average return of the set of assets. This enforces the agent to strategically
assign capital to assets that it predicts to perform above-average. We apply
our methodology in an out-of-sample analysis to 48 US stock portfolio setups,
varying in the number of stocks from ten up to 500 stocks, in the selection
criteria and in the level of transaction costs. The algorithm on average
outperforms all considered passive and active benchmark investment strategies
by a large margin using only one hyperparameter setup for all portfolios.

    

### [[2112.04758] Does Redundancy in AI Perception Systems Help to Test for Super-Human Automated Driving Performance?](http://arxiv.org/abs/2112.04758)


  While automated driving is often advertised with better-than-human driving
performance, this work reviews that it is nearly impossible to provide direct
statistical evidence on the system level that this is actually the case. The
amount of labeled data needed would exceed dimensions of present day technical
and economical capabilities. A commonly used strategy therefore is the use of
redundancy along with the proof of sufficient subsystems' performances. As it
is known, this strategy is efficient especially for the case of subsystems
operating independently, i.e. the occurrence of errors is independent in a
statistical sense. Here, we give some first considerations and experimental
evidence that this strategy is not a free ride as the errors of neural networks
fulfilling the same computer vision task, at least for some cases, show
correlated occurrences of errors. This remains true, if training data,
architecture, and training are kept separate or independence is trained using
special loss functions. Using data from different sensors (realized by up to
five 2D projections of the 3D MNIST data set) in our experiments is more
efficiently reducing correlations, however not to an extent that is realizing
the potential of reduction of testing data that can be obtained for redundant
and statistically independent subsystems.

    

### [[2112.04764] 3D-VField: Learning to Adversarially Deform Point Clouds for Robust 3D Object Detection](http://arxiv.org/abs/2112.04764)


  As 3D object detection on point clouds relies on the geometrical
relationships between the points, non-standard object shapes can hinder a
method's detection capability. However, in safety-critical settings, robustness
on out-of-distribution and long-tail samples is fundamental to circumvent
dangerous issues, such as the misdetection of damaged or rare cars. In this
work, we substantially improve the generalization of 3D object detectors to
out-of-domain data by taking into account deformed point clouds during
training. We achieve this with 3D-VField: a novel method that plausibly deforms
objects via vectors learned in an adversarial fashion. Our approach constrains
3D points to slide along their sensor view rays while neither adding nor
removing any of them. The obtained vectors are transferrable,
sample-independent and preserve shape smoothness and occlusions. By augmenting
normal samples with the deformations produced by these vector fields during
training, we significantly improve robustness against differently shaped
objects, such as damaged/deformed cars, even while training only on KITTI.
Towards this end, we propose and share open source CrashD: a synthetic dataset
of realistic damaged and rare cars, with a variety of crash scenarios.
Extensive experiments on KITTI, Waymo, our CrashD and SUN RGB-D show the high
generalizability of our techniques to out-of-domain data, different models and
sensors, namely LiDAR and ToF cameras, for both indoor and outdoor scenes. Our
CrashD dataset is available at this https URL.

    

### [[2112.04766] Adaptive Methods for Aggregated Domain Generalization](http://arxiv.org/abs/2112.04766)


  Domain generalization involves learning a classifier from a heterogeneous
collection of training sources such that it generalizes to data drawn from
similar unknown target domains, with applications in large-scale learning and
personalized inference. In many settings, privacy concerns prohibit obtaining
domain labels for the training data samples, and instead only have an
aggregated collection of training points. Existing approaches that utilize
domain labels to create domain-invariant feature representations are
inapplicable in this setting, requiring alternative approaches to learn
generalizable classifiers. In this paper, we propose a domain-adaptive approach
to this problem, which operates in two steps: (a) we cluster training data
within a carefully chosen feature space to create pseudo-domains, and (b) using
these pseudo-domains we learn a domain-adaptive classifier that makes
predictions using information about both the input and the pseudo-domain it
belongs to. Our approach achieves state-of-the-art performance on a variety of
domain generalization benchmarks without using domain labels whatsoever.
Furthermore, we provide novel theoretical guarantees on domain generalization
using cluster information. Our approach is amenable to ensemble-based methods
and provides substantial gains even on large-scale benchmark datasets. The code
can be found at: this https URL


### [[2112.04779] Regularized Modal Regression on Markov-dependent Observations: A Theoretical Assessment](http://arxiv.org/abs/2112.04779)


  Modal regression, a widely used regression protocol, has been extensively
investigated in statistical and machine learning communities due to its
robustness to outliers and heavy-tailed noises. Understanding modal
regression's theoretical behavior can be fundamental in learning theory.
Despite significant progress in characterizing its statistical property, the
majority of the results are based on the assumption that samples are
independent and identical distributed (i.i.d.), which is too restrictive for
real-world applications. This paper concerns the statistical property of
regularized modal regression (RMR) within an important dependence structure -
Markov dependent. Specifically, we establish the upper bound for RMR estimator
under moderate conditions and give an explicit learning rate. Our results show
that the Markov dependence impacts on the generalization error in the way that
sample size would be discounted by a multiplicative factor depending on the
spectral gap of underlying Markov chain. This result shed a new light on
characterizing the theoretical underpinning for robust regression.

    

### [[2112.04785] VMAgent: Scheduling Simulator for Reinforcement Learning](http://arxiv.org/abs/2112.04785)


  A novel simulator called VMAgent is introduced to help RL researchers better
explore new methods, especially for virtual machine scheduling. VMAgent is
inspired by practical virtual machine (VM) scheduling tasks and provides an
efficient simulation platform that can reflect the real situations of cloud
computing. Three scenarios (fading, recovering, and expansion) are concluded
from practical cloud computing and corresponds to many reinforcement learning
challenges (high dimensional state and action spaces, high non-stationarity,
and life-long demand). VMAgent provides flexible configurations for RL
researchers to design their customized scheduling environments considering
different problem features. From the VM scheduling perspective, VMAgent also
helps to explore better learning-based scheduling solutions.

    

### [[2112.04796] Detecting Potentially Harmful and Protective Suicide-related Content on Twitter: A Machine Learning Approach](http://arxiv.org/abs/2112.04796)


  Research shows that exposure to suicide-related news media content is
associated with suicide rates, with some content characteristics likely having
harmful and others potentially protective effects. Although good evidence
exists for a few selected characteristics, systematic large scale
investigations are missing in general, and in particular for social media data.
We apply machine learning methods to automatically label large quantities of
Twitter data. We developed a novel annotation scheme that classifies
suicide-related tweets into different message types and problem- vs.
solution-focused perspectives. We then trained a benchmark of machine learning
models including a majority classifier, an approach based on word frequency
(TF-IDF with a linear SVM) and two state-of-the-art deep learning models (BERT,
XLNet). The two deep learning models achieved the best performance in two
classification tasks: First, we classified six main content categories,
including personal stories about either suicidal ideation and attempts or
coping, calls for action intending to spread either problem awareness or
prevention-related information, reportings of suicide cases, and other
suicide-related and off-topic tweets. The deep learning models reach accuracy
scores above 73% on average across the six categories, and F1-scores in between
69% and 85% for all but the suicidal ideation and attempts category (55%).
Second, in separating postings referring to actual suicide from off-topic
tweets, they correctly labelled around 88% of tweets, with BERT achieving
F1-scores of 93% and 74% for the two categories. These classification
performances are comparable to the state-of-the-art on similar tasks. By making
data labeling more efficient, this work enables future large-scale
investigations on harmful and protective effects of various kinds of social
media content on suicide rates and on help-seeking behavior.

    

### [[2112.04800] GPU backed Data Mining on Android Devices](http://arxiv.org/abs/2112.04800)


  Choosing an appropriate programming paradigm for high-performance computing
on low-power devices can be useful to speed up calculations. Many Android
devices have an integrated GPU and - although not officially supported - the
OpenCL framework can be used on Android devices for addressing these GPUs.
OpenCL supports thread and data parallelism. Applications that use the GPU must
account for the fact that they can be suspended by the user or the Android
operating system at any moment. We have created a wrapper library that allows
to use OpenCL on Android devices. Already written OpenCL programs can be
executed with almost no modification. We have used this library to compare the
performance of the DBSCAN and Kmeans algorithms on an integrated GPU of an
Arm-v7 tablet with other single and multithreaded implementations on the same
device. We have investigated which programming paradigm and language allows the
best tradeoff between execution speed and energy consumption. Using the GPU for
HPC on Android devices can help to carry out computationally intensive machine
learning or data mining tasks in remote areas, under harsh environmental
conditions and in areas where energy supply is an issue.

    

### [[2112.04803] Combining Textual Features for the Detection of Hateful and Offensive Language](http://arxiv.org/abs/2112.04803)


  The detection of offensive, hateful and profane language has become a
critical challenge since many users in social networks are exposed to
cyberbullying activities on a daily basis. In this paper, we present an
analysis of combining different textual features for the detection of hateful
or offensive posts on Twitter. We provide a detailed experimental evaluation to
understand the impact of each building block in a neural network architecture.
The proposed architecture is evaluated on the English Subtask 1A: Identifying
Hate, offensive and profane content from the post datasets of HASOC-2021
dataset under the team name TIB-VA. We compared different variants of the
contextual word embeddings combined with the character level embeddings and the
encoding of collected hate terms.

    

### [[2112.04807] Effective dimension of machine learning models](http://arxiv.org/abs/2112.04807)


  Making statements about the performance of trained models on tasks involving
new data is one of the primary goals of machine learning, i.e., to understand
the generalization power of a model. Various capacity measures try to capture
this ability, but usually fall short in explaining important characteristics of
models that we observe in practice. In this study, we propose the local
effective dimension as a capacity measure which seems to correlate well with
generalization error on standard data sets. Importantly, we prove that the
local effective dimension bounds the generalization error and discuss the
aptness of this capacity measure for machine learning models.

    

### [[2112.04809] Next Steps: Learning a Disentangled Gait Representation for Versatile Quadruped Locomotion](http://arxiv.org/abs/2112.04809)


  Quadruped locomotion is rapidly maturing to a degree where robots now
routinely traverse a variety of unstructured terrains. However, while gaits can
be varied typically by selecting from a range of pre-computed styles, current
planners are unable to vary key gait parameters continuously while the robot is
in motion. The synthesis, on-the-fly, of gaits with unexpected operational
characteristics or even the blending of dynamic manoeuvres lies beyond the
capabilities of the current state-of-the-art. In this work we address this
limitation by learning a latent space capturing the key stance phases
constituting a particular gait. This is achieved via a generative model trained
on a single trot style, which encourages disentanglement such that application
of a drive signal to a single dimension of the latent state induces holistic
plans synthesising a continuous variety of trot styles. We demonstrate that
specific properties of the drive signal map directly to gait parameters such as
cadence, foot step height and full stance duration. Due to the nature of our
approach these synthesised gaits are continuously variable online during robot
operation and robustly capture a richness of movement significantly exceeding
the relatively narrow behaviour seen during training. In addition, the use of a
generative model facilitates the detection and mitigation of disturbances to
provide a versatile and robust planning framework. We evaluate our approach on
a real ANYmal quadruped robot and demonstrate that our method achieves a
continuous blend of dynamic trot styles whilst being robust and reactive to
external perturbations.

    

### [[2112.04814] Multimodal Pre-Training Model for Sequence-based Prediction of Protein-Protein Interaction](http://arxiv.org/abs/2112.04814)


  Protein-protein interactions (PPIs) are essentials for many biological
processes where two or more proteins physically bind together to achieve their
functions. Modeling PPIs is useful for many biomedical applications, such as
vaccine design, antibody therapeutics, and peptide drug discovery. Pre-training
a protein model to learn effective representation is critical for PPIs. Most
pre-training models for PPIs are sequence-based, which naively adopt the
language models used in natural language processing to amino acid sequences.
More advanced works utilize the structure-aware pre-training technique, taking
advantage of the contact maps of known protein structures. However, neither
sequences nor contact maps can fully characterize structures and functions of
the proteins, which are closely related to the PPI problem. Inspired by this
insight, we propose a multimodal protein pre-training model with three
modalities: sequence, structure, and function (S2F). Notably, instead of using
contact maps to learn the amino acid-level rigid structures, we encode the
structure feature with the topology complex of point clouds of heavy atoms. It
allows our model to learn structural information about not only the backbones
but also the side chains. Moreover, our model incorporates the knowledge from
the functional description of proteins extracted from literature or manual
annotations. Our experiments show that the S2F learns protein embeddings that
achieve good performances on a variety of PPIs tasks, including cross-species
PPI, antibody-antigen affinity prediction, antibody neutralization prediction
for SARS-CoV-2, and mutation-driven binding affinity change prediction.

    

### [[2112.04828] Evaluation of survival distribution predictions with discrimination measures](http://arxiv.org/abs/2112.04828)


  In this paper we consider how to evaluate survival distribution predictions
with measures of discrimination. This is a non-trivial problem as
discrimination measures are the most commonly used in survival analysis and yet
there is no clear method to derive a risk prediction from a distribution
prediction. We survey methods proposed in literature and software and consider
their respective advantages and disadvantages. Whilst distributions are
frequently evaluated by discrimination measures, we find that the method for
doing so is rarely described in the literature and often leads to unfair
comparisons. We find that the most robust method of reducing a distribution to
a risk is to sum over the predicted cumulative hazard. We recommend that
machine learning survival analysis software implements clear transformations
between distribution and risk predictions in order to allow more transparent
and accessible model evaluation.

    

### [[2112.04842] Siamese Attribute-missing Graph Auto-encoder](http://arxiv.org/abs/2112.04842)


  Graph representation learning (GRL) on attribute-missing graphs, which is a
common yet challenging problem, has recently attracted considerable attention.
We observe that existing literature: 1) isolates the learning of attribute and
structure embedding thus fails to take full advantages of the two types of
information; 2) imposes too strict distribution assumption on the latent space
variables, leading to less discriminative feature representations. In this
paper, based on the idea of introducing intimate information interaction
between the two information sources, we propose our Siamese Attribute-missing
Graph Auto-encoder (SAGA). Specifically, three strategies have been conducted.
First, we entangle the attribute embedding and structure embedding by
introducing a siamese network structure to share the parameters learned by both
processes, which allows the network training to benefit from more abundant and
diverse information. Second, we introduce a K-nearest neighbor (KNN) and
structural constraint enhanced learning mechanism to improve the quality of
latent features of the missing attributes by filtering unreliable connections.
Third, we manually mask the connections on multiple adjacent matrices and force
the structural information embedding sub-network to recover the true adjacent
matrix, thus enforcing the resulting network to be able to selectively exploit
more high-order discriminative features for data completion. Extensive
experiments on six benchmark datasets demonstrate the superiority of our SAGA
against the state-of-the-art methods.

    

### [[2112.04857] A New Measure of Model Redundancy for Compressed Convolutional Neural Networks](http://arxiv.org/abs/2112.04857)


  While recently many designs have been proposed to improve the model
efficiency of convolutional neural networks (CNNs) on a fixed resource budget,
theoretical understanding of these designs is still conspicuously lacking. This
paper aims to provide a new framework for answering the question: Is there
still any remaining model redundancy in a compressed CNN? We begin by
developing a general statistical formulation of CNNs and compressed CNNs via
the tensor decomposition, such that the weights across layers can be summarized
into a single tensor. Then, through a rigorous sample complexity analysis, we
reveal an important discrepancy between the derived sample complexity and the
naive parameter counting, which serves as a direct indicator of the model
redundancy. Motivated by this finding, we introduce a new model redundancy
measure for compressed CNNs, called the $K/R$ ratio, which further allows for
nonlinear activations. The usefulness of this new measure is supported by
ablation studies on popular block designs and datasets.

    

### [[2112.04871] KGE-CL: Contrastive Learning of Knowledge Graph Embeddings](http://arxiv.org/abs/2112.04871)


  Learning the embeddings of knowledge graphs is vital in artificial
intelligence, and can benefit various downstream applications, such as
recommendation and question answering. In recent years, many research efforts
have been proposed for knowledge graph embedding. However, most previous
knowledge graph embedding methods ignore the semantic similarity between the
related entities and entity-relation couples in different triples since they
separately optimize each triple with the scoring function. To address this
problem, we propose a simple yet efficient contrastive learning framework for
knowledge graph embeddings, which can shorten the semantic distance of the
related entities and entity-relation couples in different triples and thus
improve the expressiveness of knowledge graph embeddings. We evaluate our
proposed method on three standard knowledge graph benchmarks. It is noteworthy
that our method can yield some new state-of-the-art results, achieving 51.2%
MRR, 46.8% Hits@1 on the WN18RR dataset, and 59.1% MRR, 51.8% Hits@1 on the
YAGO3-10 dataset.

    

### [[2112.04882] Evaluating saliency methods on artificial data with different background types](http://arxiv.org/abs/2112.04882)


  Over the last years, many 'explainable artificial intelligence' (xAI)
approaches have been developed, but these have not always been objectively
evaluated. To evaluate the quality of heatmaps generated by various saliency
methods, we developed a framework to generate artificial data with synthetic
lesions and a known ground truth map. Using this framework, we evaluated two
data sets with different backgrounds, Perlin noise and 2D brain MRI slices, and
found that the heatmaps vary strongly between saliency methods and backgrounds.
We strongly encourage further evaluation of saliency maps and xAI methods using
this framework before applying these in clinical or other safety-critical
settings.

    

### [[2112.04887] Forecast Evaluation in Large Cross-Sections of Realized Volatility](http://arxiv.org/abs/2112.04887)


  In this paper, we consider the forecast evaluation of realized volatility
measures under cross-section dependence using equal predictive accuracy testing
procedures. We evaluate the predictive accuracy of the model based on the
augmented cross-section when forecasting Realized Volatility. Under the null
hypothesis of equal predictive accuracy the benchmark model employed is a
standard HAR model while under the alternative of non-equal predictive accuracy
the forecast model is an augmented HAR model estimated via the LASSO shrinkage.
We study the sensitivity of forecasts to the model specification by
incorporating a measurement error correction as well as cross-sectional jump
component measures. The out-of-sample forecast evaluation of the models is
assessed with numerical implementations.

    

### [[2112.04891] Multi-Task Learning on Networks](http://arxiv.org/abs/2112.04891)


  The multi-task learning (MTL) paradigm can be traced back to an early paper
of Caruana (1997) in which it was argued that data from multiple tasks can be
used with the aim to obtain a better performance over learning each task
independently. A solution of MTL with conflicting objectives requires modelling
the trade-off among them which is generally beyond what a straight linear
combination can achieve. A theoretically principled and computationally
effective strategy is finding solutions which are not dominated by others as it
is addressed in the Pareto analysis. Multi-objective optimization problems
arising in the multi-task learning context have specific features and require
adhoc methods. The analysis of these features and the proposal of a new
computational approach represent the focus of this work. Multi-objective
evolutionary algorithms (MOEAs) can easily include the concept of dominance and
therefore the Pareto analysis. The major drawback of MOEAs is a low sample
efficiency with respect to function evaluations. The key reason for this
drawback is that most of the evolutionary approaches do not use models for
approximating the objective function. Bayesian Optimization takes a radically
different approach based on a surrogate model, such as a Gaussian Process. In
this thesis the solutions in the Input Space are represented as probability
distributions encapsulating the knowledge contained in the function
evaluations. In this space of probability distributions, endowed with the
metric given by the Wasserstein distance, a new algorithm MOEA/WST can be
designed in which the model is not directly on the objective function but in an
intermediate Information Space where the objects from the input space are
mapped into histograms. Computational results show that the sample efficiency
and the quality of the Pareto set provided by MOEA/WST are significantly better
than in the standard MOEA.

    

### [[2112.04893] Real-World Dexterous Object Manipulation based Deep Reinforcement Learning](http://arxiv.org/abs/2112.04893)


  Deep reinforcement learning has shown its advantages in real-time
decision-making based on the state of the agent. In this stage, we solved the
task of using a real robot to manipulate the cube to a given trajectory. The
task is broken down into different procedures and we propose a hierarchical
structure, the high-level deep reinforcement learning model selects appropriate
contact positions and the low-level control module performs the position
control under the corresponding trajectory. Our framework reduces the
disadvantage of low sample efficiency of deep reinforcement learning and
lacking adaptability of traditional robot control methods. Our algorithm is
trained in simulation and migrated to reality without fine-tuning. The
experimental results show the effectiveness of our method both simulation and
reality. Our code and video can be found at
this https URL and
this https URL.

    

### [[2112.04895] Latent Space Explanation by Intervention](http://arxiv.org/abs/2112.04895)


  The success of deep neural nets heavily relies on their ability to encode
complex relations between their input and their output. While this property
serves to fit the training data well, it also obscures the mechanism that
drives prediction. This study aims to reveal hidden concepts by employing an
intervention mechanism that shifts the predicted class based on discrete
variational autoencoders. An explanatory model then visualizes the encoded
information from any hidden layer and its corresponding intervened
representation. By the assessment of differences between the original
representation and the intervened representation, one can determine the
concepts that can alter the class, hence providing interpretability. We
demonstrate the effectiveness of our approach on CelebA, where we show various
visualizations for bias in the data and suggest different interventions to
reveal and change bias.

    

### [[2112.04899] Assessing Fairness in the Presence of Missing Data](http://arxiv.org/abs/2112.04899)


  Missing data are prevalent and present daunting challenges in real data
analysis. While there is a growing body of literature on fairness in analysis
of fully observed data, there has been little theoretical work on investigating
fairness in analysis of incomplete data. In practice, a popular analytical
approach for dealing with missing data is to use only the set of complete
cases, i.e., observations with all features fully observed to train a
prediction algorithm. However, depending on the missing data mechanism, the
distribution of complete cases and the distribution of the complete data may be
substantially different. When the goal is to develop a fair algorithm in the
complete data domain where there are no missing values, an algorithm that is
fair in the complete case domain may show disproportionate bias towards some
marginalized groups in the complete data domain. To fill this significant gap,
we study the problem of estimating fairness in the complete data domain for an
arbitrary model evaluated merely using complete cases. We provide upper and
lower bounds on the fairness estimation error and conduct numerical experiments
to assess our theoretical results. Our work provides the first known
theoretical results on fairness guarantee in analysis of incomplete data.

    

### [[2112.04902] Learning Personal Representations from fMRIby Predicting Neurofeedback Performance](http://arxiv.org/abs/2112.04902)


  We present a deep neural network method for learning a personal
representation for individuals that are performing a self neuromodulation task,
guided by functional MRI (fMRI). This neurofeedback task (watch vs. regulate)
provides the subjects with a continuous feedback contingent on down regulation
of their Amygdala signal and the learning algorithm focuses on this region's
time-course of activity. The representation is learned by a self-supervised
recurrent neural network, that predicts the Amygdala activity in the next fMRI
frame given recent fMRI frames and is conditioned on the learned individual
representation. It is shown that the individuals' representation improves the
next-frame prediction considerably. Moreover, this personal representation,
learned solely from fMRI images, yields good performance in linear prediction
of psychiatric traits, which is better than performing such a prediction based
on clinical data and personality tests. Our code is attached as supplementary
and the data would be shared subject to ethical approvals.

    

### [[2112.04905] i-SpaSP: Structured Neural Pruning via Sparse Signal Recovery](http://arxiv.org/abs/2112.04905)


  We propose a novel, structured pruning algorithm for neural networks -- the
iterative, Sparse Structured Pruning algorithm, dubbed as i-SpaSP. Inspired by
ideas from sparse signal recovery, i-SpaSP operates by iteratively identifying
a larger set of important parameter groups (e.g., filters or neurons) within a
network that contribute most to the residual between pruned and dense network
output, then thresholding these groups based on a smaller, pre-defined pruning
ratio. For both two-layer and multi-layer network architectures with ReLU
activations, we show the error induced by pruning with i-SpaSP decays
polynomially, where the degree of this polynomial becomes arbitrarily large
based on the sparsity of the dense network's hidden representations. In our
experiments, i-SpaSP is evaluated across a variety of datasets (i.e., MNIST and
ImageNet) and architectures (i.e., feed forward networks, ResNet34, and
MobileNetV2), where it is shown to discover high-performing sub-networks and
improve upon the pruning efficiency of provable baseline methodologies by
several orders of magnitude. Put simply, i-SpaSP is easy to implement with
automatic differentiation, achieves strong empirical results, comes with
theoretical convergence guarantees, and is efficient, thus distinguishing
itself as one of the few computationally efficient, practical, and provable
pruning algorithms.

    

### [[2112.04906] Enhancing Column Generation by a Machine-Learning-Based Pricing Heuristic for Graph Coloring](http://arxiv.org/abs/2112.04906)


  Column Generation (CG) is an effective method for solving large-scale
optimization problems. CG starts by solving a sub-problem with a subset of
columns (i.e., variables) and gradually includes new columns that can improve
the solution of the current subproblem. The new columns are generated as needed
by repeatedly solving a pricing problem, which is often NP-hard and is a
bottleneck of the CG approach. To tackle this, we propose a
Machine-Learning-based Pricing Heuristic (MLPH)that can generate many
high-quality columns efficiently. In each iteration of CG, our MLPH leverages
an ML model to predict the optimal solution of the pricing problem, which is
then used to guide a sampling method to efficiently generate multiple
high-quality columns. Using the graph coloring problem, we empirically show
that MLPH significantly enhancesCG as compared to six state-of-the-art methods,
and the improvement in CG can lead to substantially better performance of the
branch-and-price exact method.

    

### [[2112.04907] JueWu-MC: Playing Minecraft with Sample-efficient Hierarchical Reinforcement Learning](http://arxiv.org/abs/2112.04907)


  Learning rational behaviors in open-world games like Minecraft remains to be
challenging for Reinforcement Learning (RL) research due to the compound
challenge of partial observability, high-dimensional visual perception and
delayed reward. To address this, we propose JueWu-MC, a sample-efficient
hierarchical RL approach equipped with representation learning and imitation
learning to deal with perception and exploration. Specifically, our approach
includes two levels of hierarchy, where the high-level controller learns a
policy to control over options and the low-level workers learn to solve each
sub-task. To boost the learning of sub-tasks, we propose a combination of
techniques including 1) action-aware representation learning which captures
underlying relations between action and representation, 2) discriminator-based
self-imitation learning for efficient exploration, and 3) ensemble behavior
cloning with consistency filtering for policy robustness. Extensive experiments
show that JueWu-MC significantly improves sample efficiency and outperforms a
set of baselines by a large margin. Notably, we won the championship of the
NeurIPS MineRL 2021 research competition and achieved the highest performance
score ever.

    

### [[2112.04912] Scalable and Decentralized Algorithms for Anomaly Detection via Learning-Based Controlled Sensing](http://arxiv.org/abs/2112.04912)


  We address the problem of sequentially selecting and observing processes from
a given set to find the anomalies among them. The decision-maker observes a
subset of the processes at any given time instant and obtains a noisy binary
indicator of whether or not the corresponding process is anomalous. In this
setting, we develop an anomaly detection algorithm that chooses the processes
to be observed at a given time instant, decides when to stop taking
observations, and declares the decision on anomalous processes. The objective
of the detection algorithm is to identify the anomalies with an accuracy
exceeding the desired value while minimizing the delay in decision making. We
devise a centralized algorithm where the processes are jointly selected by a
common agent as well as a decentralized algorithm where the decision of whether
to select a process is made independently for each process. Our algorithms rely
on a Markov decision process defined using the marginal probability of each
process being normal or anomalous, conditioned on the observations. We
implement the detection algorithms using the deep actor-critic reinforcement
learning framework. Unlike prior work on this topic that has exponential
complexity in the number of processes, our algorithms have computational and
memory requirements that are both polynomial in the number of processes. We
demonstrate the efficacy of these algorithms using numerical experiments by
comparing them with state-of-the-art methods.

    

### [[2112.04913] Identification of Twitter Bots based on an Explainable ML Framework: the US 2020 Elections Case Study](http://arxiv.org/abs/2112.04913)


  Twitter is one of the most popular social networks attracting millions of
users, while a considerable proportion of online discourse is captured. It
provides a simple usage framework with short messages and an efficient
application programming interface (API) enabling the research community to
study and analyze several aspects of this social network. However, the Twitter
usage simplicity can lead to malicious handling by various bots. The malicious
handling phenomenon expands in online discourse, especially during the
electoral periods, where except the legitimate bots used for dissemination and
communication purposes, the goal is to manipulate the public opinion and the
electorate towards a certain direction, specific ideology, or political party.
This paper focuses on the design of a novel system for identifying Twitter bots
based on labeled Twitter data. To this end, a supervised machine learning (ML)
framework is adopted using an Extreme Gradient Boosting (XGBoost) algorithm,
where the hyper-parameters are tuned via cross-validation. Our study also
deploys Shapley Additive Explanations (SHAP) for explaining the ML model
predictions by calculating feature importance, using the game theoretic-based
Shapley values. Experimental evaluation on distinct Twitter datasets
demonstrate the superiority of our approach, in terms of bot detection
accuracy, when compared against a recent state-of-the-art Twitter bot detection
method.

    

### [[2112.04914] End-to-end Alexa Device Arbitration](http://arxiv.org/abs/2112.04914)


  We introduce a variant of the speaker localization problem, which we call
device arbitration. In the device arbitration problem, a user utters a keyword
that is detected by multiple distributed microphone arrays (smart home
devices), and we want to determine which device was closest to the user. Rather
than solving the full localization problem, we propose an end-to-end machine
learning system. This system learns a feature embedding that is computed
independently on each device. The embeddings from each device are then
aggregated together to produce the final arbitration decision. We use a
large-scale room simulation to generate training and evaluation data, and
compare our system against a signal processing baseline.

    

### [[2112.04922] A More Stable Accelerated Gradient Method Inspired by Continuous-Time Perspective](http://arxiv.org/abs/2112.04922)


  Nesterov's accelerated gradient method (NAG) is widely used in problems with
machine learning background including deep learning, and is corresponding to a
continuous-time differential equation. From this connection, the property of
the differential equation and its numerical approximation can be investigated
to improve the accelerated gradient method. In this work we present a new
improvement of NAG in terms of stability inspired by numerical analysis. We
give the precise order of NAG as a numerical approximation of its
continuous-time limit and then present a new method with higher order. We show
theoretically that our new method is more stable than NAG for large step size.
Experiments of matrix completion and handwriting digit recognition demonstrate
that the stability of our new method is better. Furthermore, better stability
leads to higher computational speed in experiments.

    

### [[2112.04928] Self-Supervised Image-to-Text and Text-to-Image Synthesis](http://arxiv.org/abs/2112.04928)


  A comprehensive understanding of vision and language and their interrelation
are crucial to realize the underlying similarities and differences between
these modalities and to learn more generalized, meaningful representations. In
recent years, most of the works related to Text-to-Image synthesis and
Image-to-Text generation, focused on supervised generative deep architectures
to solve the problems, where very little interest was placed on learning the
similarities between the embedding spaces across modalities. In this paper, we
propose a novel self-supervised deep learning based approach towards learning
the cross-modal embedding spaces; for both image to text and text to image
generations. In our approach, we first obtain dense vector representations of
images using StackGAN-based autoencoder model and also dense vector
representations on sentence-level utilizing LSTM based text-autoencoder; then
we study the mapping from embedding space of one modality to embedding space of
the other modality utilizing GAN and maximum mean discrepancy based generative
networks. We, also demonstrate that our model learns to generate textual
description from image data as well as images from textual data both
qualitatively and quantitatively.

    

### [[2112.04933] Measuring Wind Turbine Health Using Drifting Concepts](http://arxiv.org/abs/2112.04933)


  Time series processing is an essential aspect of wind turbine health
monitoring. Despite the progress in this field, there is still room for new
methods to improve modeling quality. In this paper, we propose two new
approaches for the analysis of wind turbine health. Both approaches are based
on abstract concepts, implemented using fuzzy sets, which summarize and
aggregate the underlying raw data. By observing the change in concepts, we
infer about the change in the turbine's health. Analyzes are carried out
separately for different external conditions (wind speed and temperature). We
extract concepts that represent relative low, moderate, and high power
production. The first method aims at evaluating the decrease or increase in
relatively high and low power production. This task is performed using a
regression-like model. The second method evaluates the overall drift of the
extracted concepts. Large drift indicates that the power production process
undergoes fluctuations in time. Concepts are labeled using linguistic labels,
thus equipping our model with improved interpretability features. We applied
the proposed approach to process publicly available data describing four wind
turbines. The simulation results have shown that the aging process is not
homogeneous in all wind turbines.

    

### [[2112.04934] Model Doctor: A Simple Gradient Aggregation Strategy for Diagnosing and Treating CNN Classifiers](http://arxiv.org/abs/2112.04934)


  Recently, Convolutional Neural Network (CNN) has achieved excellent
performance in the classification task. It is widely known that CNN is deemed
as a 'black-box', which is hard for understanding the prediction mechanism and
debugging the wrong prediction. Some model debugging and explanation works are
developed for solving the above drawbacks. However, those methods focus on
explanation and diagnosing possible causes for model prediction, based on which
the researchers handle the following optimization of models manually. In this
paper, we propose the first completely automatic model diagnosing and treating
tool, termed as Model Doctor. Based on two discoveries that 1) each category is
only correlated with sparse and specific convolution kernels, and 2)
adversarial samples are isolated while normal samples are successive in the
feature space, a simple aggregate gradient constraint is devised for
effectively diagnosing and optimizing CNN classifiers. The aggregate gradient
strategy is a versatile module for mainstream CNN classifiers. Extensive
experiments demonstrate that the proposed Model Doctor applies to all existing
CNN classifiers, and improves the accuracy of $16$ mainstream CNN classifiers
by 1%-5%.

    

### [[2112.04939] A Training Framework for Stereo-Aware Speech Enhancement using Deep Neural Networks](http://arxiv.org/abs/2112.04939)


  Deep learning-based speech enhancement has shown unprecedented performance in
recent years. The most popular mono speech enhancement frameworks are
end-to-end networks mapping the noisy mixture into an estimate of the clean
speech. With growing computational power and availability of multichannel
microphone recordings, prior works have aimed to incorporate spatial statistics
along with spectral information to boost up performance. Despite an improvement
in enhancement performance of mono output, the spatial image preservation and
subjective evaluations have not gained much attention in the literature. This
paper proposes a novel stereo-aware framework for speech enhancement, i.e., a
training loss for deep learning-based speech enhancement to preserve the
spatial image while enhancing the stereo mixture. The proposed framework is
model independent, hence it can be applied to any deep learning based
architecture. We provide an extensive objective and subjective evaluation of
the trained models through a listening test. We show that by regularizing for
an image preservation loss, the overall performance is improved, and the stereo
aspect of the speech is better preserved.

    

### [[2112.04947] Automated Side Channel Analysis of Media Software with Manifold Learning](http://arxiv.org/abs/2112.04947)


  The prosperous development of cloud computing and machine learning as a
service has led to the widespread use of media software to process confidential
media data. This paper explores an adversary's ability to launch side channel
analyses (SCA) against media software to reconstruct confidential media inputs.
Recent advances in representation learning and perceptual learning inspired us
to consider the reconstruction of media inputs from side channel traces as a
cross-modality manifold learning task that can be addressed in a unified manner
with an autoencoder framework trained to learn the mapping between media inputs
and side channel observations. We further enhance the autoencoder with
attention to localize the program points that make the primary contribution to
SCA, thus automatically pinpointing information-leakage points in media
software. We also propose a novel and highly effective defensive technique
called perception blinding that can perturb media inputs with perception masks
and mitigate manifold learning-based SCA.
Our evaluation exploits three popular media software to reconstruct inputs in
image, audio, and text formats. We analyze three common side channels - cache
bank, cache line, and page tables - and userspace-only cache set accesses
logged by standard Prime+Probe. Our framework successfully reconstructs
high-quality confidential inputs from the assessed media software and
automatically pinpoint their vulnerable program points, many of which are
unknown to the public. We further show that perception blinding can mitigate
manifold learning-based SCA with negligible extra cost.

    

### [[2112.04948] PARL: Enhancing Diversity of Ensemble Networks to Resist Adversarial Attacks via Pairwise Adversarially Robust Loss Function](http://arxiv.org/abs/2112.04948)


  The security of Deep Learning classifiers is a critical field of study
because of the existence of adversarial attacks. Such attacks usually rely on
the principle of transferability, where an adversarial example crafted on a
surrogate classifier tends to mislead the target classifier trained on the same
dataset even if both classifiers have quite different architecture. Ensemble
methods against adversarial attacks demonstrate that an adversarial example is
less likely to mislead multiple classifiers in an ensemble having diverse
decision boundaries. However, recent ensemble methods have either been shown to
be vulnerable to stronger adversaries or shown to lack an end-to-end
evaluation. This paper attempts to develop a new ensemble methodology that
constructs multiple diverse classifiers using a Pairwise Adversarially Robust
Loss (PARL) function during the training procedure. PARL utilizes gradients of
each layer with respect to input in every classifier within the ensemble
simultaneously. The proposed training procedure enables PARL to achieve higher
robustness against black-box transfer attacks compared to previous ensemble
methods without adversely affecting the accuracy of clean examples. We also
evaluate the robustness in the presence of white-box attacks, where adversarial
examples are crafted using parameters of the target classifier. We present
extensive experiments using standard image classification datasets like
CIFAR-10 and CIFAR-100 trained using standard ResNet20 classifier against
state-of-the-art adversarial attacks to demonstrate the robustness of the
proposed ensemble methodology.

    

### [[2112.04953] Machine Learning for Utility Prediction in Argument-Based Computational Persuasion](http://arxiv.org/abs/2112.04953)


  Automated persuasion systems (APS) aim to persuade a user to believe
something by entering into a dialogue in which arguments and counterarguments
are exchanged. To maximize the probability that an APS is successful in
persuading a user, it can identify a global policy that will allow it to select
the best arguments it presents at each stage of the dialogue whatever arguments
the user presents. However, in real applications, such as for healthcare, it is
unlikely the utility of the outcome of the dialogue will be the same, or the
exact opposite, for the APS and user. In order to deal with this situation,
games in extended form have been harnessed for argumentation in Bi-party
Decision Theory. This opens new problems that we address in this paper: (1) How
can we use Machine Learning (ML) methods to predict utility functions for
different subpopulations of users? and (2) How can we identify for a new user
the best utility function from amongst those that we have learned? To this
extent, we develop two ML methods, EAI and EDS, that leverage information
coming from the users to predict their utilities. EAI is restricted to a fixed
amount of information, whereas EDS can choose the information that best detects
the subpopulations of a user. We evaluate EAI and EDS in a simulation setting
and in a realistic case study concerning healthy eating habits. Results are
promising in both cases, but EDS is more effective at predicting useful utility
functions.

    

### [[2112.04963] Model-Agnostic Hybrid Numerical Weather Prediction and Machine Learning Paradigm for Solar Forecasting in the Tropics](http://arxiv.org/abs/2112.04963)


  Numerical weather prediction (NWP) and machine learning (ML) methods are
popular for solar forecasting. However, NWP models have multiple possible
physical parameterizations, which requires site-specific NWP optimization. This
is further complicated when regional NWP models are used with global climate
models with different possible parameterizations. In this study, an alternative
approach is proposed and evaluated for four radiation models. Weather Research
and Forecasting (WRF) model is run in both global and regional mode to provide
an estimate for solar irradiance. This estimate is then post-processed using ML
to provide a final prediction. Normalized root-mean-square error from WRF is
reduced by up to 40-50% with this ML error correction model. Results obtained
using CAM, GFDL, New Goddard and RRTMG radiation models were comparable after
this correction, negating the need for WRF parameterization tuning. Other
models incorporating nearby locations and sensor data are also evaluated, with
the latter being particularly promising.

    

### [[2112.04977] Bringing Atomistic Deep Learning to Prime Time](http://arxiv.org/abs/2112.04977)


  Artificial intelligence has not yet revolutionized the design of materials
and molecules. In this perspective, we identify four barriers preventing the
integration of atomistic deep learning, molecular science, and high-performance
computing. We outline focused research efforts to address the opportunities
presented by these challenges.

    

### [[2112.04979] A fully-differentiable compressible high-order computational fluid dynamics solver](http://arxiv.org/abs/2112.04979)


  Fluid flows are omnipresent in nature and engineering disciplines. The
reliable computation of fluids has been a long-lasting challenge due to
nonlinear interactions over multiple spatio-temporal scales. The compressible
Navier-Stokes equations govern compressible flows and allow for complex
phenomena like turbulence and shocks. Despite tremendous progress in hardware
and software, capturing the smallest length-scales in fluid flows still
introduces prohibitive computational cost for real-life applications. We are
currently witnessing a paradigm shift towards machine learning supported design
of numerical schemes as a means to tackle aforementioned problem. While prior
work has explored differentiable algorithms for one- or two-dimensional
incompressible fluid flows, we present a fully-differentiable three-dimensional
framework for the computation of compressible fluid flows using high-order
state-of-the-art numerical methods. Firstly, we demonstrate the efficiency of
our solver by computing classical two- and three-dimensional test cases,
including strong shocks and transition to turbulence. Secondly, and more
importantly, our framework allows for end-to-end optimization to improve
existing numerical schemes inside computational fluid dynamics algorithms. In
particular, we are using neural networks to substitute a conventional numerical
flux function.

    

### [[2112.04981] PE-former: Pose Estimation Transformer](http://arxiv.org/abs/2112.04981)


  Vision transformer architectures have been demonstrated to work very
effectively for image classification tasks. Efforts to solve more challenging
vision tasks with transformers rely on convolutional backbones for feature
extraction. In this paper we investigate the use of a pure transformer
architecture (i.e., one with no CNN backbone) for the problem of 2D body pose
estimation. We evaluate two ViT architectures on the COCO dataset. We
demonstrate that using an encoder-decoder transformer architecture yields state
of the art results on this estimation problem.

    

### [[2112.04984] Robust Weakly Supervised Learning for COVID-19 Recognition Using Multi-Center CT Images](http://arxiv.org/abs/2112.04984)


  The world is currently experiencing an ongoing pandemic of an infectious
disease named coronavirus disease 2019 (i.e., COVID-19), which is caused by the
severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2). Computed
Tomography (CT) plays an important role in assessing the severity of the
infection and can also be used to identify those symptomatic and asymptomatic
COVID-19 carriers. With a surge of the cumulative number of COVID-19 patients,
radiologists are increasingly stressed to examine the CT scans manually.
Therefore, an automated 3D CT scan recognition tool is highly in demand since
the manual analysis is time-consuming for radiologists and their fatigue can
cause possible misjudgment. However, due to various technical specifications of
CT scanners located in different hospitals, the appearance of CT images can be
significantly different leading to the failure of many automated image
recognition approaches. The multi-domain shift problem for the multi-center and
multi-scanner studies is therefore nontrivial that is also crucial for a
dependable recognition and critical for reproducible and objective diagnosis
and prognosis. In this paper, we proposed a COVID-19 CT scan recognition model
namely coronavirus information fusion and diagnosis network (CIFD-Net) that can
efficiently handle the multi-domain shift problem via a new robust weakly
supervised learning paradigm. Our model can resolve the problem of different
appearance in CT scan images reliably and efficiently while attaining higher
accuracy compared to other state-of-the-art methods.

    

### [[2112.05000] The Peril of Popular Deep Learning Uncertainty Estimation Methods](http://arxiv.org/abs/2112.05000)


  Uncertainty estimation (UE) techniques -- such as the Gaussian process (GP),
Bayesian neural networks (BNN), Monte Carlo dropout (MCDropout) -- aim to
improve the interpretability of machine learning models by assigning an
estimated uncertainty value to each of their prediction outputs. However, since
too high uncertainty estimates can have fatal consequences in practice, this
paper analyzes the above techniques.
Firstly, we show that GP methods always yield high uncertainty estimates on
out of distribution (OOD) data. Secondly, we show on a 2D toy example that both
BNNs and MCDropout do not give high uncertainty estimates on OOD samples.
Finally, we show empirically that this pitfall of BNNs and MCDropout holds on
real world datasets as well. Our insights (i) raise awareness for the more
cautious use of currently popular UE methods in Deep Learning, (ii) encourage
the development of UE methods that approximate GP-based methods -- instead of
BNNs and MCDropout, and (iii) our empirical setups can be used for verifying
the OOD performances of any other UE method. The source code is available at
this https URL.

    

### [[2112.05003] Wikidated 1.0: An Evolving Knowledge Graph Dataset of Wikidata's Revision History](http://arxiv.org/abs/2112.05003)


  Wikidata is the largest general-interest knowledge base that is openly
available. It is collaboratively edited by thousands of volunteer editors and
has thus evolved considerably since its inception in 2012. In this paper, we
present Wikidated 1.0, a dataset of Wikidata's full revision history, which
encodes changes between Wikidata revisions as sets of deletions and additions
of RDF triples. To the best of our knowledge, it constitutes the first large
dataset of an evolving knowledge graph, a recently emerging research subject in
the Semantic Web community. We introduce the methodology for generating
Wikidated 1.0 from dumps of Wikidata, discuss its implementation and
limitations, and present statistical characteristics of the dataset.

    

### [[2112.05005] Mutual Adversarial Training: Learning together is better than going alone](http://arxiv.org/abs/2112.05005)


  Recent studies have shown that robustness to adversarial attacks can be
transferred across networks. In other words, we can make a weak model more
robust with the help of a strong teacher model. We ask if instead of learning
from a static teacher, can models "learn together" and "teach each other" to
achieve better robustness? In this paper, we study how interactions among
models affect robustness via knowledge distillation. We propose mutual
adversarial training (MAT), in which multiple models are trained together and
share the knowledge of adversarial examples to achieve improved robustness. MAT
allows robust models to explore a larger space of adversarial samples, and find
more robust feature spaces and decision boundaries. Through extensive
experiments on CIFAR-10 and CIFAR-100, we demonstrate that MAT can effectively
improve model robustness and outperform state-of-the-art methods under
white-box attacks, bringing $\sim$8% accuracy gain to vanilla adversarial
training (AT) under PGD-100 attacks. In addition, we show that MAT can also
mitigate the robustness trade-off among different perturbation types, bringing
as much as 13.1% accuracy gain to AT baselines against the union of $l_\infty$,
$l_2$ and $l_1$ attacks. These results show the superiority of the proposed
method and demonstrate that collaborative learning is an effective strategy for
designing robust models.

    

### [[2112.05025] Gradient-matching coresets for continual learning](http://arxiv.org/abs/2112.05025)


  We devise a coreset selection method based on the idea of gradient matching:
The gradients induced by the coreset should match, as closely as possible,
those induced by the original training dataset. We evaluate the method in the
context of continual learning, where it can be used to curate a rehearsal
memory. Our method performs strong competitors such as reservoir sampling
across a range of memory sizes.

    

### [[2112.05056] Opinion Extraction as A Structured Sentiment Analysis using Transformers](http://arxiv.org/abs/2112.05056)


  Relationship extraction and named entity recognition have always been
considered as two distinct tasks that require different input data, labels, and
models. However, both are essential for structured sentiment analysis. We
believe that both tasks can be combined into a single stacked model with the
same input data. We performed different experiments to find the best model to
extract multiple opinion tuples from a single sentence. The opinion tuples will
consist of holders, targets, and expressions. With the opinion tuples, we will
be able to extract the relationship we need.

    

### [[2112.05062] Learning Transferable Motor Skills with Hierarchical Latent Mixture Policies](http://arxiv.org/abs/2112.05062)


  For robots operating in the real world, it is desirable to learn reusable
behaviours that can effectively be transferred and adapted to numerous tasks
and scenarios. We propose an approach to learn abstract motor skills from data
using a hierarchical mixture latent variable model. In contrast to existing
work, our method exploits a three-level hierarchy of both discrete and
continuous latent variables, to capture a set of high-level behaviours while
allowing for variance in how they are executed. We demonstrate in manipulation
domains that the method can effectively cluster offline data into distinct,
executable behaviours, while retaining the flexibility of a continuous latent
variable model. The resulting skills can be transferred and fine-tuned on new
tasks, unseen objects, and from state to vision-based policies, yielding better
sample efficiency and asymptotic performance compared to existing skill- and
imitation-based methods. We further analyse how and when the skills are most
beneficial: they encourage directed exploration to cover large regions of the
state space relevant to the task, making them most effective in challenging
sparse-reward settings.

    

### [[2112.05068] A Bayesian Treatment of Real-to-Sim for Deformable Object Manipulation](http://arxiv.org/abs/2112.05068)


  Deformable object manipulation remains a challenging task in robotics
research. Conventional techniques for parameter inference and state estimation
typically rely on a precise definition of the state space and its dynamics.
While this is appropriate for rigid objects and robot states, it is challenging
to define the state space of a deformable object and how it evolves in time. In
this work, we pose the problem of inferring physical parameters of deformable
objects as a probabilistic inference task defined with a simulator. We propose
a novel methodology for extracting state information from image sequences via a
technique to represent the state of a deformable object as a distribution
embedding. This allows to incorporate noisy state observations directly into
modern Bayesian simulation-based inference tools in a principled manner. Our
experiments confirm that we can estimate posterior distributions of physical
properties, such as elasticity, friction and scale of highly deformable
objects, such as cloth and ropes. Overall, our method addresses the real-to-sim
problem probabilistically and helps to better represent the evolution of the
state of deformable objects.

    

### [[2112.05071] A Novel Tropical Geometry-based Interpretable Machine Learning Method: Application in Prognosis of Advanced Heart Failure](http://arxiv.org/abs/2112.05071)


  A model's interpretability is essential to many practical applications such
as clinical decision support systems. In this paper, a novel interpretable
machine learning method is presented, which can model the relationship between
input variables and responses in humanly understandable rules. The method is
built by applying tropical geometry to fuzzy inference systems, wherein
variable encoding functions and salient rules can be discovered by supervised
learning. Experiments using synthetic datasets were conducted to investigate
the performance and capacity of the proposed algorithm in classification and
rule discovery. Furthermore, the proposed method was applied to a clinical
application that identified heart failure patients that would benefit from
advanced therapies such as heart transplant or durable mechanical circulatory
support. Experimental results show that the proposed network achieved great
performance on the classification tasks. In addition to learning humanly
understandable rules from the dataset, existing fuzzy domain knowledge can be
easily transferred into the network and used to facilitate model training. From
our results, the proposed model and the ability of learning existing domain
knowledge can significantly improve the model generalizability. The
characteristics of the proposed network make it promising in applications
requiring model reliability and justification.

    

### [[2112.05077] Generating Useful Accident-Prone Driving Scenarios via a Learned Traffic Prior](http://arxiv.org/abs/2112.05077)


  Evaluating and improving planning for autonomous vehicles requires scalable
generation of long-tail traffic scenarios. To be useful, these scenarios must
be realistic and challenging, but not impossible to drive through safely. In
this work, we introduce STRIVE, a method to automatically generate challenging
scenarios that cause a given planner to produce undesirable behavior, like
collisions. To maintain scenario plausibility, the key idea is to leverage a
learned model of traffic motion in the form of a graph-based conditional VAE.
Scenario generation is formulated as an optimization in the latent space of
this traffic model, effected by perturbing an initial real-world scene to
produce trajectories that collide with a given planner. A subsequent
optimization is used to find a "solution" to the scenario, ensuring it is
useful to improve the given planner. Further analysis clusters generated
scenarios based on collision type. We attack two planners and show that STRIVE
successfully generates realistic, challenging scenarios in both cases. We
additionally "close the loop" and use these scenarios to optimize
hyperparameters of a rule-based planner.

    

### [[2112.05084] A Survey on Echo Chambers on Social Media: Description, Detection and Mitigation](http://arxiv.org/abs/2112.05084)


  Echo chambers on social media are a significant problem that can elicit a
number of negative consequences, most recently affecting the response to
COVID-19. Echo chambers promote conspiracy theories about the virus and are
found to be linked to vaccine hesitancy, less compliance with mask mandates,
and the practice of social distancing. Moreover, the problem of echo chambers
is connected to other pertinent issues like political polarization and the
spread of misinformation. An echo chamber is defined as a network of users in
which users only interact with opinions that support their pre-existing beliefs
and opinions, and they exclude and discredit other viewpoints. This survey aims
to examine the echo chamber phenomenon on social media from a social computing
perspective and provide a blueprint for possible solutions. We survey the
related literature to understand the attributes of echo chambers and how they
affect the individual and society at large. Additionally, we show the
mechanisms, both algorithmic and psychological, that lead to the formation of
echo chambers. These mechanisms could be manifested in two forms: (1) the bias
of social media's recommender systems and (2) internal biases such as
confirmation bias and homophily. While it is immensely challenging to mitigate
internal biases, there has been great efforts seeking to mitigate the bias of
recommender systems. These recommender systems take advantage of our own biases
to personalize content recommendations to keep us engaged in order to watch
more ads. Therefore, we further investigate different computational approaches
for echo chamber detection and prevention, mainly based around recommender
systems.

    

### [[2112.05090] Extending the WILDS Benchmark for Unsupervised Adaptation](http://arxiv.org/abs/2112.05090)


  Machine learning systems deployed in the wild are often trained on a source
distribution but deployed on a different target distribution. Unlabeled data
can be a powerful point of leverage for mitigating these distribution shifts,
as it is frequently much more available than labeled data. However, existing
distribution shift benchmarks for unlabeled data do not reflect the breadth of
scenarios that arise in real-world applications. In this work, we present the
WILDS 2.0 update, which extends 8 of the 10 datasets in the WILDS benchmark of
distribution shifts to include curated unlabeled data that would be
realistically obtainable in deployment. To maintain consistency, the labeled
training, validation, and test sets, as well as the evaluation metrics, are
exactly the same as in the original WILDS benchmark. These datasets span a wide
range of applications (from histology to wildlife conservation), tasks
(classification, regression, and detection), and modalities (photos, satellite
images, microscope slides, text, molecular graphs). We systematically benchmark
state-of-the-art methods that leverage unlabeled data, including
domain-invariant, self-training, and self-supervised methods, and show that
their success on WILDS 2.0 is limited. To facilitate method development and
evaluation, we provide an open-source package that automates data loading and
contains all of the model architectures and methods used in this paper. Code
and leaderboards are available at this https URL.

    

### [[2112.05095] Provable Continual Learning via Sketched Jacobian Approximations](http://arxiv.org/abs/2112.05095)


  An important problem in machine learning is the ability to learn tasks in a
sequential manner. If trained with standard first-order methods most models
forget previously learned tasks when trained on a new task, which is often
referred to as catastrophic forgetting. A popular approach to overcome
forgetting is to regularize the loss function by penalizing models that perform
poorly on previous tasks. For example, elastic weight consolidation (EWC)
regularizes with a quadratic form involving a diagonal matrix build based on
past data. While EWC works very well for some setups, we show that, even under
otherwise ideal conditions, it can provably suffer catastrophic forgetting if
the diagonal matrix is a poor approximation of the Hessian matrix of previous
tasks. We propose a simple approach to overcome this: Regularizing training of
a new task with sketches of the Jacobian matrix of past data. This provably
enables overcoming catastrophic forgetting for linear models and for wide
neural networks, at the cost of memory. The overarching goal of this paper is
to provided insights on when regularization-based continual learning algorithms
work and under what memory costs.

    

### [[2112.05104] Continuation Path with Linear Convergence Rate](http://arxiv.org/abs/2112.05104)


  Path-following algorithms are frequently used in composite optimization
problems where a series of subproblems, with varying regularization
hyperparameters, are solved sequentially. By reusing the previous solutions as
initialization, better convergence speeds have been observed numerically. This
makes it a rather useful heuristic to speed up the execution of optimization
algorithms in machine learning. We present a primal dual analysis of the
path-following algorithm and explore how to design its hyperparameters as well
as determining how accurately each subproblem should be solved to guarantee a
linear convergence rate on a target problem. Furthermore, considering
optimization with a sparsity-inducing penalty, we analyze the change of the
active sets with respect to the regularization parameter. The latter can then
be adaptively calibrated to finely determine the number of features that will
be selected along the solution path. This leads to simple heuristics for
calibrating hyperparameters of active set approaches to reduce their complexity
and improve their execution time.

    

### [[2112.05120] On Convergence of Federated Averaging Langevin Dynamics](http://arxiv.org/abs/2112.05120)


  We propose a federated averaging Langevin algorithm (FA-LD) for uncertainty
quantification and mean predictions with distributed clients. In particular, we
generalize beyond normal posterior distributions and consider a general class
of models. We develop theoretical guarantees for FA-LD for strongly log-concave
distributions with non-i.i.d data and study how the injected noise and the
stochastic-gradient noise, the heterogeneity of data, and the varying learning
rates affect the convergence. Such an analysis sheds light on the optimal
choice of local updates to minimize communication costs. Important to our
approach is that the communication efficiency does not deteriorate with the
injected noise in the Langevin algorithms. In addition, we examine in our FA-LD
algorithm both independent and correlated noise used over different clients. We
observe that there is also a trade-off between federation and communication
cost there. As local devices may become inactive in the federated network, we
also show convergence results based on different averaging schemes where only
partial device updates are available.

    

### [[2112.05124] Neural Descriptor Fields: SE(3)-Equivariant Object Representations for Manipulation](http://arxiv.org/abs/2112.05124)


  We present Neural Descriptor Fields (NDFs), an object representation that
encodes both points and relative poses between an object and a target (such as
a robot gripper or a rack used for hanging) via category-level descriptors. We
employ this representation for object manipulation, where given a task
demonstration, we want to repeat the same task on a new object instance from
the same category. We propose to achieve this objective by searching (via
optimization) for the pose whose descriptor matches that observed in the
demonstration. NDFs are conveniently trained in a self-supervised fashion via a
3D auto-encoding task that does not rely on expert-labeled keypoints. Further,
NDFs are SE(3)-equivariant, guaranteeing performance that generalizes across
all possible 3D object translations and rotations. We demonstrate learning of
manipulation tasks from few (5-10) demonstrations both in simulation and on a
real robot. Our performance generalizes across both object instances and 6-DoF
object poses, and significantly outperforms a recent baseline that relies on 2D
descriptors. Project website: this https URL.

    

### [[2112.05128] Fair Structure Learning in Heterogeneous Graphical Models](http://arxiv.org/abs/2112.05128)


  Inference of community structure in probabilistic graphical models may not be
consistent with fairness constraints when nodes have demographic attributes.
Certain demographics may be over-represented in some detected communities and
under-represented in others. This paper defines a novel $\ell_1$-regularized
pseudo-likelihood approach for fair graphical model selection. In particular,
we assume there is some community or clustering structure in the true
underlying graph, and we seek to learn a sparse undirected graph and its
communities from the data such that demographic groups are fairly represented
within the communities. Our optimization approach uses the demographic parity
definition of fairness, but the framework is easily extended to other
definitions of fairness. We establish statistical consistency of the proposed
method for both a Gaussian graphical model and an Ising model for,
respectively, continuous and binary data, proving that our method can recover
the graphs and their fair communities with high probability.

    

### [[2112.05135] PixMix: Dreamlike Pictures Comprehensively Improve Safety Measures](http://arxiv.org/abs/2112.05135)


  In real-world applications of machine learning, reliable and safe systems
must consider measures of performance beyond standard test set accuracy. These
other goals include out-of-distribution (OOD) robustness, prediction
consistency, resilience to adversaries, calibrated uncertainty estimates, and
the ability to detect anomalous inputs. However, improving performance towards
these goals is often a balancing act that today's methods cannot achieve
without sacrificing performance on other safety axes. For instance, adversarial
training improves adversarial robustness but sharply degrades other classifier
performance metrics. Similarly, strong data augmentation and regularization
techniques often improve OOD robustness but harm anomaly detection, raising the
question of whether a Pareto improvement on all existing safety measures is
possible. To meet this challenge, we design a new data augmentation strategy
utilizing the natural structural complexity of pictures such as fractals, which
outperforms numerous baselines, is near Pareto-optimal, and roundly improves
safety measures.

    

### [[2112.05136] PTR: A Benchmark for Part-based Conceptual, Relational, and Physical Reasoning](http://arxiv.org/abs/2112.05136)


  A critical aspect of human visual perception is the ability to parse visual
scenes into individual objects and further into object parts, forming
part-whole hierarchies. Such composite structures could induce a rich set of
semantic concepts and relations, thus playing an important role in the
interpretation and organization of visual signals as well as for the
generalization of visual perception and reasoning. However, existing visual
reasoning benchmarks mostly focus on objects rather than parts. Visual
reasoning based on the full part-whole hierarchy is much more challenging than
object-centric reasoning due to finer-grained concepts, richer geometry
relations, and more complex physics. Therefore, to better serve for part-based
conceptual, relational and physical reasoning, we introduce a new large-scale
diagnostic visual reasoning dataset named PTR. PTR contains around 70k RGBD
synthetic images with ground truth object and part level annotations regarding
semantic instance segmentation, color attributes, spatial and geometric
relationships, and certain physical properties such as stability. These images
are paired with 700k machine-generated questions covering various types of
reasoning types, making them a good testbed for visual reasoning models. We
examine several state-of-the-art visual reasoning models on this dataset and
observe that they still make many surprising mistakes in situations where
humans can easily infer the correct answer. We believe this dataset will open
up new opportunities for part-based reasoning.

    

### [[1805.07984] Adversarial Attacks on Neural Networks for Graph Data](http://arxiv.org/abs/1805.07984)


  Deep learning models for graphs have achieved strong performance for the task
of node classification. Despite their proliferation, currently there is no
study of their robustness to adversarial attacks. Yet, in domains where they
are likely to be used, e.g. the web, adversaries are common. Can deep learning
models for graphs be easily fooled? In this work, we introduce the first study
of adversarial attacks on attributed graphs, specifically focusing on models
exploiting ideas of graph convolutions. In addition to attacks at test time, we
tackle the more challenging class of poisoning/causative attacks, which focus
on the training phase of a machine learning model. We generate adversarial
perturbations targeting the node's features and the graph structure, thus,
taking the dependencies between instances in account. Moreover, we ensure that
the perturbations remain unnoticeable by preserving important data
characteristics. To cope with the underlying discrete domain we propose an
efficient algorithm Nettack exploiting incremental computations. Our
experimental study shows that accuracy of node classification significantly
drops even when performing only few perturbations. Even more, our attacks are
transferable: the learned attacks generalize to other state-of-the-art node
classification models and unsupervised approaches, and likewise are successful
even when only limited knowledge about the graph is given.

    

### [[1905.12346] Nyström landmark sampling and regularized Christoffel functions](http://arxiv.org/abs/1905.12346)


  Selecting diverse and important items, called landmarks, from a large set is
a problem of interest in machine learning. As a specific example, in order to
deal with large training sets, kernel methods often rely on low rank matrix
Nyström approximations based on the selection or sampling of landmarks. In
this context, we propose a deterministic and a randomized adaptive algorithm
for selecting landmark points within a training data set. These landmarks are
related to the minima of a sequence of kernelized Christoffel functions. Beyond
the known connection between Christoffel functions and leverage scores, a
connection of our method with finite determinantal point processes (DPPs) is
also explained. Namely, our construction promotes diversity among important
landmark points in a way similar to DPPs. Also, we explain how our randomized
adaptive algorithm can influence the accuracy of Kernel Ridge Regression.

    

### [[1906.06397] Interpretable and Personalized Apprenticeship Scheduling: Learning Interpretable Scheduling Policies from Heterogeneous User Demonstrations](http://arxiv.org/abs/1906.06397)


  Resource scheduling and coordination is an NP-hard optimization requiring an
efficient allocation of agents to a set of tasks with upper- and lower bound
temporal and resource constraints. Due to the large-scale and dynamic nature of
resource coordination in hospitals and factories, human domain experts manually
plan and adjust schedules on the fly. To perform this job, domain experts
leverage heterogeneous strategies and rules-of-thumb honed over years of
apprenticeship. What is critically needed is the ability to extract this domain
knowledge in a heterogeneous and interpretable apprenticeship learning
framework to scale beyond the power of a single human expert, a necessity in
safety-critical domains. We propose a personalized and interpretable
apprenticeship scheduling algorithm that infers an interpretable representation
of all human task demonstrators by extracting decision-making criteria via an
inferred, personalized embedding non-parametric in the number of demonstrator
types. We achieve near-perfect LfD accuracy in synthetic domains and 88.22\%
accuracy on a planning domain with real-world, outperforming baselines.
Finally, our user study showed our methodology produces more interpretable and
easier-to-use models than neural networks ($p < 0.05$).

    

### [[1912.10784] An improper estimator with optimal excess risk in misspecified density estimation and logistic regression](http://arxiv.org/abs/1912.10784)


  We introduce a procedure for conditional density estimation under logarithmic
loss, which we call SMP (Sample Minmax Predictor). This estimator minimizes a
new general excess risk bound for statistical learning. On standard examples,
this bound scales as $d/n$ with $d$ the model dimension and $n$ the sample
size, and critically remains valid under model misspecification. Being an
improper (out-of-model) procedure, SMP improves over within-model estimators
such as the maximum likelihood estimator, whose excess risk degrades under
misspecification. Compared to approaches reducing to the sequential problem,
our bounds remove suboptimal $\log n$ factors and can handle unbounded classes.
For the Gaussian linear model, the predictions and risk bound of SMP are
governed by leverage scores of covariates, nearly matching the optimal risk in
the well-specified case without conditions on the noise variance or
approximation error of the linear model. For logistic regression, SMP provides
a non-Bayesian approach to calibration of probabilistic predictions relying on
virtual samples, and can be computed by solving two logistic regressions. It
achieves a non-asymptotic excess risk of $O((d + B^2R^2)/n)$, where $R$ bounds
the norm of features and $B$ that of the comparison parameter; by contrast, no
within-model estimator can achieve better rate than $\min({B R}/{\sqrt{n}}, {d
e^{BR}}/{n} )$ in general. This provides a more practical alternative to
Bayesian approaches, which require approximate posterior sampling, thereby
partly addressing a question raised by Foster et al. (2018).

    

### [[2007.13437] Energy-based View of Retrosynthesis](http://arxiv.org/abs/2007.13437)


  Retrosynthesis -- the process of identifying a set of reactants to synthesize
a target molecule -- is of vital importance to material design and drug
discovery. Existing machine learning approaches based on language models and
graph neural networks have achieved encouraging results. In this paper, we
propose a framework that unifies sequence- and graph-based methods as
energy-based models (EBMs) with different energy functions. This unified
perspective provides critical insights about EBM variants through a
comprehensive assessment of performance. Additionally, we present a novel dual
variant within the framework that performs consistent training over Bayesian
forward- and backward-prediction by constraining the agreement between the two
directions. This model improves state-of-the-art performance by 9.6% for
template-free approaches where the reaction type is unknown.

    

### [[2008.12552] Probabilistic Random Indexing for Continuous Event Detection](http://arxiv.org/abs/2008.12552)


  The present paper explores a novel variant of Random Indexing (RI) based
representations for encoding language data with a view to using them in a
dynamic scenario where events are happening in a continuous fashion. As the
size of the representations in the general method of onehot encoding grows
linearly with the size of the vocabulary, they become non-scalable for online
purposes with high volumes of dynamic data. On the other hand, existing
pre-trained embedding models are not suitable for detecting happenings of new
events due to the dynamic nature of the text data. The present work addresses
this issue by using a novel RI representation by imposing a probability
distribution on the number of randomized entries which leads to a class of RI
representations. It also provides a rigorous analysis of the goodness of the
representation methods to encode semantic information in terms of the
probability of orthogonality. Building on these ideas we propose an algorithm
that is log-linear with the size of vocabulary to track the semantic
relationship of a query word to other words for suggesting the events that are
relevant to the word in question. We ran simulations using the proposed
algorithm for tweet data specific to three different events and present our
findings. The proposed probabilistic RI representations are found to be much
faster and scalable than Bag of Words (BoW) embeddings while maintaining
accuracy in depicting semantic relationships.

    

### [[2009.01845] Qibo: a framework for quantum simulation with hardware acceleration](http://arxiv.org/abs/2009.01845)


  We present Qibo, a new open-source software for fast evaluation of quantum
circuits and adiabatic evolution which takes full advantage of hardware
accelerators. The growing interest in quantum computing and the recent
developments of quantum hardware devices motivates the development of new
advanced computational tools focused on performance and usage simplicity. In
this work we introduce a new quantum simulation framework that enables
developers to delegate all complicated aspects of hardware or platform
implementation to the library so they can focus on the problem and quantum
algorithms at hand. This software is designed from scratch with simulation
performance, code simplicity and user friendly interface as target goals. It
takes advantage of hardware acceleration such as multi-threading CPU, single
GPU and multi-GPU devices.

    

### [[2009.10007] Learning Realistic Patterns from Unrealistic Stimuli: Generalization and Data Anonymization](http://arxiv.org/abs/2009.10007)


  Good training data is a prerequisite to develop useful ML applications.
However, in many domains existing data sets cannot be shared due to privacy
regulations (e.g., from medical studies). This work investigates a simple yet
unconventional approach for anonymized data synthesis to enable third parties
to benefit from such private data. We explore the feasibility of learning
implicitly from unrealistic, task-relevant stimuli, which are synthesized by
exciting the neurons of a trained deep neural network (DNN). As such, neuronal
excitation serves as a pseudo-generative model. The stimuli data is used to
train new classification models. Furthermore, we extend this framework to
inhibit representations that are associated with specific individuals. We use
sleep monitoring data from both an open and a large closed clinical study and
evaluate whether (1) end-users can create and successfully use customized
classification models for sleep apnea detection, and (2) the identity of
participants in the study is protected. Extensive comparative empirical
investigation shows that different algorithms trained on the stimuli are able
generalize successfully on the same task as the original model. However,
architectural and algorithmic similarity between new and original models play
an important role in performance. For similar architectures, the performance is
close to that of using the true data (e.g., Accuracy difference of 0.56\%,
Kappa coefficient difference of 0.03-0.04). Further experiments show that the
stimuli can to a large extent successfully anonymize participants of the
clinical studies.

    

### [[2010.00145] Entropy Regularization for Mean Field Games with Learning](http://arxiv.org/abs/2010.00145)


  Entropy regularization has been extensively adopted to improve the
efficiency, the stability, and the convergence of algorithms in reinforcement
learning. This paper analyzes both quantitatively and qualitatively the impact
of entropy regularization for Mean Field Game (MFG) with learning in a finite
time horizon. Our study provides a theoretical justification that entropy
regularization yields time-dependent policies and, furthermore, helps
stabilizing and accelerating convergence to the game equilibrium. In addition,
this study leads to a policy-gradient algorithm for exploration in MFG. Under
this algorithm, agents are able to learn the optimal exploration scheduling,
with stable and fast convergence to the game equilibrium.

    

### [[2011.04843] Multi-document Summarization via Deep Learning Techniques: A Survey](http://arxiv.org/abs/2011.04843)


  Multi-document summarization (MDS) is an effective tool for information
aggregation that generates an informative and concise summary from a cluster of
topic-related documents. Our survey, the first of its kind, systematically
overviews the recent deep learning based MDS models. We propose a novel
taxonomy to summarize the design strategies of neural networks and conduct a
comprehensive summary of the state-of-the-art. We highlight the differences
between various objective functions that are rarely discussed in the existing
literature. Finally, we propose several future directions pertaining to this
new and exciting field.

    

### [[2011.09588] Beyond Pinball Loss: Quantile Methods for Calibrated Uncertainty Quantification](http://arxiv.org/abs/2011.09588)


  Among the many ways of quantifying uncertainty in a regression setting,
specifying the full quantile function is attractive, as quantiles are amenable
to interpretation and evaluation. A model that predicts the true conditional
quantiles for each input, at all quantile levels, presents a correct and
efficient representation of the underlying uncertainty. To achieve this, many
current quantile-based methods focus on optimizing the so-called pinball loss.
However, this loss restricts the scope of applicable regression models, limits
the ability to target many desirable properties (e.g. calibration, sharpness,
centered intervals), and may produce poor conditional quantiles. In this work,
we develop new quantile methods that address these shortcomings. In particular,
we propose methods that can apply to any class of regression model, allow for
selecting a trade-off between calibration and sharpness, optimize for
calibration of centered intervals, and produce more accurate conditional
quantiles. We provide a thorough experimental evaluation of our methods, which
includes a high dimensional uncertainty quantification task in nuclear fusion.

    

### [[2011.14473] Kinetics-Informed Neural Networks](http://arxiv.org/abs/2011.14473)


  Chemical kinetics and reaction engineering consists of the phenomenological
framework for the disentanglement of reaction mechanisms, optimization of
reaction performance and the rational design of chemical processes. Here, we
utilize feed-forward artificial neural networks as basis functions to solve
ordinary differential equations (ODEs) constrained by differential algebraic
equations (DAEs) that describe microkinetic models (MKMs). We present an
algebraic framework for the mathematical description and classification of
reaction networks, types of elementary reaction, and chemical species. Under
this framework, we demonstrate that the simultaneous training of neural nets
and kinetic model parameters in a regularized multi-objective optimization
setting leads to the solution of the inverse problem through the estimation of
kinetic parameters from synthetic experimental data. We analyze a set of
scenarios to establish the extent to which kinetic parameters can be retrieved
from transient kinetic data, and assess the robustness of the methodology with
respect to statistical noise. This approach to inverse kinetic ODEs can assist
in the elucidation of reaction mechanisms based on transient data.

    

### [[2012.04567] Bayesian Image Reconstruction using Deep Generative Models](http://arxiv.org/abs/2012.04567)


  Machine learning models are commonly trained end-to-end and in a supervised
setting, using paired (input, output) data. Examples include recent
super-resolution methods that train on pairs of (low-resolution,
high-resolution) images. However, these end-to-end approaches require
re-training every time there is a distribution shift in the inputs (e.g., night
images vs daylight) or relevant latent variables (e.g., camera blur or hand
motion). In this work, we leverage state-of-the-art (SOTA) generative models
(here StyleGAN2) for building powerful image priors, which enable application
of Bayes' theorem for many downstream reconstruction tasks. Our method,
Bayesian Reconstruction through Generative Models (BRGM), uses a single
pre-trained generator model to solve different image restoration tasks, i.e.,
super-resolution and in-painting, by combining it with different forward
corruption models. We keep the weights of the generator model fixed, and
reconstruct the image by estimating the Bayesian maximum a-posteriori (MAP)
estimate over the input latent vector that generated the reconstructed image.
We further use variational inference to approximate the posterior distribution
over the latent vectors, from which we sample multiple solutions. We
demonstrate BRGM on three large and diverse datasets: (i) 60,000 images from
the Flick Faces High Quality dataset (ii) 240,000 chest X-rays from MIMIC III
and (iii) a combined collection of 5 brain MRI datasets with 7,329 scans.
Across all three datasets and without any dataset-specific hyperparameter
tuning, our simple approach yields performance competitive with current
task-specific state-of-the-art methods on super-resolution and in-painting,
while being more generalisable and without requiring any training. Our source
code and pre-trained models are available online:
this https URL.

    

### [[2102.01606] Structure-preserving Gaussian Process Dynamics](http://arxiv.org/abs/2102.01606)


  Most physical processes posses structural properties such as constant
energies, volumes, and other invariants over time. When learning models of such
dynamical systems, it is critical to respect these invariants to ensure
accurate predictions and physically meaningful behavior. Strikingly,
state-of-the-art methods in Gaussian process (GP) dynamics model learning are
not addressing this issue. On the other hand, classical numerical integrators
are specifically designed to preserve these crucial properties through time. We
propose to combine the advantages of GPs as function approximators with
structure preserving numerical integrators for dynamical systems, such as
Runge-Kutta methods. These integrators assume access to the ground truth
dynamics and require evaluations of intermediate and future time steps that are
unknown in a learning-based scenario. This makes direct inference of the GP
dynamics, with embedded numerical scheme, intractable. Our key technical
contribution is the evaluation of the implicitly defined Runge-Kutta transition
probability. In a nutshell, we introduce an implicit layer for GP regression,
which is embedded into a variational inference-based model learning scheme.

    

### [[2102.04738] End-to-End Deep Learning of Lane Detection and Path Prediction for Real-Time Autonomous Driving](http://arxiv.org/abs/2102.04738)


  Inspired by the UNet architecture of semantic image segmentation, we propose
a lightweight UNet using depthwise separable convolutions (DSUNet) for
end-to-end learning of lane detection and path prediction (PP) in autonomous
driving. We also design and integrate a PP algorithm with convolutional neural
network (CNN) to form a simulation model (CNN-PP) that can be used to assess
CNN's performance qualitatively, quantitatively, and dynamically in a host
agent car driving along with other agents all in a real-time autonomous manner.
DSUNet is 5.16x lighter in model size and 1.61x faster in inference than UNet.
DSUNet-PP outperforms UNet-PP in mean average errors of predicted curvature and
lateral offset for path planning in dynamic simulation. DSUNet-PP outperforms a
modified UNet in lateral error, which is tested in a real car on real road.
These results show that DSUNet is efficient and effective for lane detection
and path prediction in autonomous driving.

    

### [[2102.07148] A New Look and Convergence Rate of Federated Multi-Task Learning with Laplacian Regularization](http://arxiv.org/abs/2102.07148)


  Non-Independent and Identically Distributed (non- IID) data distribution
among clients is considered as the key factor that degrades the performance of
federated learning (FL). Several approaches to handle non-IID data such as
personalized FL and federated multi-task learning (FMTL) are of great interest
to research communities. In this work, first, we formulate the FMTL problem
using Laplacian regularization to explicitly leverage the relationships among
the models of clients for multi-task learning. Then, we introduce a new view of
the FMTL problem, which in the first time shows that the formulated FMTL
problem can be used for conventional FL and personalized FL. We also propose
two algorithms FedU and dFedU to solve the formulated FMTL problem in
communication-centralized and decentralized schemes, respectively.
Theoretically, we prove that the convergence rates of both algorithms achieve
linear speedup for strongly convex and sublinear speedup of order 1/2 for
nonconvex objectives. Experimentally, we show that our algorithms outperform
the conventional algorithm FedAvg in FL settings, MOCHA in FMTL settings, as
well as pFedMe and Per-FedAvg in personalized FL settings.

    

### [[2102.08138] IronMan: GNN-assisted Design Space Exploration in High-Level Synthesis via Reinforcement Learning](http://arxiv.org/abs/2102.08138)


  Despite the great success of High-Level Synthesis (HLS) tools, we observe
several unresolved challenges: 1) the high-level abstraction of programming
styles in HLS sometimes conceals optimization opportunities; 2) existing HLS
tools do not provide flexible trade-off (Pareto) solutions among different
objectives and constraints; 3) the actual quality of the resulting RTL designs
is hard to predict. To address these challenges, we propose an end-to-end
framework, namelyIronMan. The primary goal is to enable a flexible and
automated design space exploration (DSE), to provide either optimal solutions
under user-specified constraints, or various trade-offs among different
objectives (such as different types of resources, area, and latency). Such DSE
either requires tedious manual efforts or is not achievable to attain these
goals through existing HLS tools. There are three components in IronMan: 1)
GPP, a highly accurate graph-neural-network-based performance and resource
predictor; 2) RLMD, a reinforcement-learning-based multi-objective DSE engine
that explores the optimal resource allocation strategy, to provide Pareto
solutions between different objectives; 3) CT, a code transformer to assist
RLMD and GPP, which extracts the data flow graph from original HLS C/C++ and
automatically generates synthesizable code with HLS directives. The
experimental results show that: 1) GPP achieves high prediction accuracy,
reducing prediction errors of HLS tools by 10.9x in resource utilization and
5.7x in timing; 2) RLMD obtains optimal or Pareto solutions that outperform the
genetic algorithm and simulated annealing by 12.7% and 12.9%, respectively; 3)
IronMan is able to find optimized solutions perfectly matching various DSP
constraints, with 2.54x fewer DSPs and up to 6x shorter latency than those of
HLS tools while being up to 400x faster than the heuristic algorithms and HLS
tools.

    

### [[2103.02895] On the privacy-utility trade-off in differentially private hierarchical text classification](http://arxiv.org/abs/2103.02895)


  Hierarchical text classification consists in classifying text documents into
a hierarchy of classes and sub-classes. Although artificial neural networks
have proved useful to perform this task, unfortunately they can leak training
data information to adversaries due to training data memorization. Using
differential privacy during model training can mitigate leakage attacks against
trained models, enabling the models to be shared safely at the cost of reduced
model accuracy. This work investigates the privacy-utility trade-off in
hierarchical text classification with differential privacy guarantees, and
identifies neural network architectures that offer superior trade-offs. To this
end, we use a white-box membership inference attack to empirically assess the
information leakage of three widely used neural network architectures. We show
that large differential privacy parameters already suffice to completely
mitigate membership inference attacks, thus resulting only in a moderate
decrease in model utility. More specifically, for large datasets with long
texts we observed Transformer-based models to achieve an overall favorable
privacy-utility trade-off, while for smaller datasets with shorter texts
convolutional neural networks are preferable.

    

### [[2103.05577] Parametrized quantum policies for reinforcement learning](http://arxiv.org/abs/2103.05577)


  With the advent of real-world quantum computing, the idea that parametrized
quantum computations can be used as hypothesis families in a quantum-classical
machine learning system is gaining increasing traction. Such hybrid systems
have already shown the potential to tackle real-world tasks in supervised and
generative learning, and recent works have established their provable
advantages in special artificial tasks. Yet, in the case of reinforcement
learning, which is arguably most challenging and where learning boosts would be
extremely valuable, no proposal has been successful in solving even standard
benchmarking tasks, nor in showing a theoretical learning advantage over
classical algorithms. In this work, we achieve both. We propose a hybrid
quantum-classical reinforcement learning model using very few qubits, which we
show can be effectively trained to solve several standard benchmarking
environments. Moreover, we demonstrate, and formally prove, the ability of
parametrized quantum circuits to solve certain learning tasks that are
intractable for classical models, including current state-of-art deep neural
networks, under the widely-believed classical hardness of the discrete
logarithm problem.

    

### [[2103.07454] EventGraD: Event-Triggered Communication in Parallel Machine Learning](http://arxiv.org/abs/2103.07454)


  Communication in parallel systems imposes significant overhead which often
turns out to be a bottleneck in parallel machine learning. To relieve some of
this overhead, in this paper, we present EventGraD - an algorithm with
event-triggered communication for stochastic gradient descent in parallel
machine learning. The main idea of this algorithm is to modify the requirement
of communication at every iteration in standard implementations of stochastic
gradient descent in parallel machine learning to communicating only when
necessary at certain iterations. We provide theoretical analysis of convergence
of our proposed algorithm. We also implement the proposed algorithm for
data-parallel training of a popular residual neural network used for training
the CIFAR-10 dataset and show that EventGraD can reduce the communication load
by up to 60% while retaining the same level of accuracy. In addition, EventGraD
can be combined with other approaches such as Top-K sparsification to decrease
communication further while maintaining accuracy.

    

### [[2103.13056] Minimax Regret for Stochastic Shortest Path](http://arxiv.org/abs/2103.13056)


  We study the Stochastic Shortest Path (SSP) problem in which an agent has to
reach a goal state in minimum total expected cost. In the learning formulation
of the problem, the agent has no prior knowledge about the costs and dynamics
of the model. She repeatedly interacts with the model for $K$ episodes, and has
to minimize her regret. In this work we show that the minimax regret for this
setting is $\widetilde O(\sqrt{ (B_\star^2 + B_\star) |S| |A| K})$ where
$B_\star$ is a bound on the expected cost of the optimal policy from any state,
$S$ is the state space, and $A$ is the action space. This matches the $\Omega
(\sqrt{ B_\star^2 |S| |A| K})$ lower bound of Rosenberg et al. [2020] for
$B_\star \ge 1$, and improves their regret bound by a factor of $\sqrt{|S|}$.
For $B_\star < 1$ we prove a matching lower bound of $\Omega (\sqrt{ B_\star
|S| |A| K})$. Our algorithm is based on a novel reduction from SSP to
finite-horizon MDPs. To that end, we provide an algorithm for the
finite-horizon setting whose leading term in the regret depends polynomially on
the expected cost of the optimal policy and only logarithmically on the
horizon.

    

### [[2103.16634] Exploiting Invariance in Training Deep Neural Networks](http://arxiv.org/abs/2103.16634)


  Inspired by two basic mechanisms in animal visual systems, we introduce a
feature transform technique that imposes invariance properties in the training
of deep neural networks. The resulting algorithm requires less parameter
tuning, trains well with an initial learning rate 1.0, and easily generalizes
to different tasks. We enforce scale invariance with local statistics in the
data to align similar samples at diverse scales. To accelerate convergence, we
enforce a GL(n)-invariance property with global statistics extracted from a
batch such that the gradient descent solution should remain invariant under
basis change. Profiling analysis shows our proposed modifications takes 5% of
the computations of the underlying convolution layer. Tested on convolutional
networks and transformer networks, our proposed technique requires fewer
iterations to train, surpasses all baselines by a large margin, seamlessly
works on both small and large batch size training, and applies to different
computer vision and language tasks.

    

### [[2104.04258] Counter-Strike Deathmatch with Large-Scale Behavioural Cloning](http://arxiv.org/abs/2104.04258)


  This paper describes an AI agent that plays the popular first-person-shooter
(FPS) video game `Counter-Strike; Global Offensive' (CSGO) from pixel input.
The agent, a deep neural network, matches the performance of the medium
difficulty built-in AI on the deathmatch game mode, whilst adopting a humanlike
play style. Unlike much prior work in games, no API is available for CSGO, so
algorithms must train and run in real-time. This limits the quantity of
on-policy data that can be generated, precluding many reinforcement learning
algorithms. Our solution uses behavioural cloning - training on a large noisy
dataset scraped from human play on online servers (4 million frames, comparable
in size to ImageNet), and a smaller dataset of high-quality expert
demonstrations. This scale is an order of magnitude larger than prior work on
imitation learning in FPS games.

    

### [[2104.05463] Scalable Power Control/Beamforming in Heterogeneous Wireless Networks with Graph Neural Networks](http://arxiv.org/abs/2104.05463)


  Machine learning (ML) has been widely used for efficient resource allocation
(RA) in wireless networks. Although superb performance is achieved on small and
simple networks, most existing ML-based approaches are confronted with
difficulties when heterogeneity occurs and network size expands. In this paper,
specifically focusing on power control/beamforming (PC/BF) in heterogeneous
device-to-device (D2D) networks, we propose a novel unsupervised learning-based
framework named heterogeneous interference graph neural network (HIGNN) to
handle these challenges. First, we characterize diversified link features and
interference relations with heterogeneous graphs. Then, HIGNN is proposed to
empower each link to obtain its individual transmission scheme after limited
information exchange with neighboring links. It is noteworthy that HIGNN is
scalable to wireless networks of growing sizes with robust performance after
trained on small-sized networks. Numerical results show that compared with
state-of-the-art benchmarks, HIGNN achieves much higher execution efficiency
while providing strong performance.

    

### [[2104.12138] Learning to Address Intra-segment Misclassification in Retinal Imaging](http://arxiv.org/abs/2104.12138)


  Accurate multi-class segmentation is a long-standing challenge in medical
imaging, especially in scenarios where classes share strong similarity.
Segmenting retinal blood vessels in retinal photographs is one such scenario,
in which arteries and veins need to be identified and differentiated from each
other and from the background. Intra-segment misclassification, i.e. veins
classified as arteries or vice versa, frequently occurs when arteries and veins
intersect, whereas in binary retinal vessel segmentation, error rates are much
lower. We thus propose a new approach that decomposes multi-class segmentation
into multiple binary, followed by a binary-to-multi-class fusion network. The
network merges representations of artery, vein, and multi-class feature maps,
each of which are supervised by expert vessel annotation in adversarial
training. A skip-connection based merging process explicitly maintains
class-specific gradients to avoid gradient vanishing in deep layers, to favor
the discriminative features. The results show that, our model respectively
improves F1-score by 4.4\%, 5.1\%, and 4.2\% compared with three
state-of-the-art deep learning based methods on DRIVE-AV, LES-AV, and HRF-AV
data sets. Code: this https URL


### [[2105.04504] Deep Neural Networks as Point Estimates for Deep Gaussian Processes](http://arxiv.org/abs/2105.04504)


  Neural networks and Gaussian processes are complementary in their strengths
and weaknesses. Having a better understanding of their relationship comes with
the promise to make each method benefit from the strengths of the other. In
this work, we establish an equivalence between the forward passes of neural
networks and (deep) sparse Gaussian process models. The theory we develop is
based on interpreting activation functions as interdomain inducing features
through a rigorous analysis of the interplay between activation functions and
kernels. This results in models that can either be seen as neural networks with
improved uncertainty prediction or deep Gaussian processes with increased
prediction accuracy. These claims are supported by experimental results on
regression and classification datasets.

    

### [[2106.03027] Model Zoo: A Growing "Brain" That Learns Continually](http://arxiv.org/abs/2106.03027)


  This paper argues that continual learning methods can benefit by splitting
the capacity of the learner across multiple models. We use statistical learning
theory and experimental analysis to show how multiple tasks can interact with
each other in a non-trivial fashion when a single model is trained on them. The
generalization error on a particular task can improve when it is trained with
synergistic tasks, but can also deteriorate when trained with competing tasks.
This theory motivates our method named Model Zoo which, inspired from the
boosting literature, grows an ensemble of small models, each of which is
trained during one episode of continual learning. We demonstrate that Model Zoo
obtains large gains in accuracy on a variety of continual learning benchmark
problems.

    

### [[2106.05206] Avoiding Traps in Nonconvex Problems](http://arxiv.org/abs/2106.05206)


  Iterative projection methods may become trapped at non-solutions when the
constraint sets are nonconvex. Two kinds of parameters are available to help
avoid this behavior and this study gives examples of both. The first kind of
parameter, called a hyperparameter, includes any kind of parameter that appears
in the definition of the iteration rule itself. The second kind comprises
metric parameters in the definition of the constraint sets, a feature that
arises when the problem to be solved has two or more kinds of variables.
Through examples we show the importance of properly tuning both kinds of
parameters and offer heuristic interpretations of the observed behavior.

    

### [[2106.06168] Generate, Annotate, and Learn: NLP with Synthetic Text](http://arxiv.org/abs/2106.06168)


  Semi-Supervised Learning (SSL) has seen success in many application domains,
but this success often hinges on the availability of task-specific unlabeled
data. Knowledge distillation (KD) has enabled effective optimization of compact
neural nets, achieving the best results when the knowledge of an expensive
network is distilled via fresh task-specific unlabeled data. However,
task-specific unlabeled data can be challenging to find, especially for NLP. We
investigate the use of generative models in synthesizing unlabeled data and
present a simple and general framework called "generate, annotate, and learn
(GAL)". A language model (LM) is used to synthesize in-domain unlabeled data.
Then, a classifier is used to annotate such data. Finally, synthetically
generated and annotated data is used to advance SSL, KD, and few-shot learning
on NLP and tabular tasks. To obtain a strong task-specific LM, we either
fine-tune a large LM on inputs from a specific task, or prompt a large LM with
a few input examples and conditionally generate more unlabeled examples. It
also yields a new state-of-the-art for 6-layer transformers on the GLUE
leaderboard. Finally, self-training with GAL offers large gains on four tabular
tasks from the UCI repository.

    

### [[2106.07153] Iterative Methods for Private Synthetic Data: Unifying Framework and New Methods](http://arxiv.org/abs/2106.07153)


  We study private synthetic data generation for query release, where the goal
is to construct a sanitized version of a sensitive dataset, subject to
differential privacy, that approximately preserves the answers to a large
collection of statistical queries. We first present an algorithmic framework
that unifies a long line of iterative algorithms in the literature. Under this
framework, we propose two new methods. The first method, private entropy
projection (PEP), can be viewed as an advanced variant of MWEM that adaptively
reuses past query measurements to boost accuracy. Our second method, generative
networks with the exponential mechanism (GEM), circumvents computational
bottlenecks in algorithms such as MWEM and PEP by optimizing over generative
models parameterized by neural networks, which capture a rich family of
distributions while enabling fast gradient-based optimization. We demonstrate
that PEP and GEM empirically outperform existing algorithms. Furthermore, we
show that GEM nicely incorporates prior information from public data while
overcoming limitations of PMW^Pub, the existing state-of-the-art method that
also leverages public data.

    

### [[2106.15278] Open-Set Representation Learning through Combinatorial Embedding](http://arxiv.org/abs/2106.15278)


  Visual recognition tasks are often limited to dealing with a small subset of
classes simply because the labels for the remaining classes are unavailable. We
are interested in identifying novel concepts in a dataset through the
representation learning based on both labeled and unlabeled examples, and
extending the horizon of recognition to both known and novel classes. To
address this challenging task, we propose a combinatorial learning approach,
which naturally clusters the examples in unseen classes using the compositional
knowledge given by multiple supervised meta-classifiers on heterogeneous label
spaces. The representations given by the combinatorial embedding are made more
robust by consistency regularization. We also introduce a metric learning
strategy to estimate pairwise pseudo-labels for improving the representations
of unlabeled examples, which preserves semantic relations across known and
novel classes effectively. The proposed algorithm discovers novel concepts via
a joint optimization of enhancing the discrimitiveness of unseen classes as
well as learning the representations of known classes generalizable to novel
ones. Our extensive experiments demonstrate remarkable performance gains by the
proposed approach in multiple image retrieval and novel class discovery
benchmarks.

    

### [[2107.00644] Stabilizing Deep Q-Learning with ConvNets and Vision Transformers under Data Augmentation](http://arxiv.org/abs/2107.00644)


  While agents trained by Reinforcement Learning (RL) can solve increasingly
challenging tasks directly from visual observations, generalizing learned
skills to novel environments remains very challenging. Extensive use of data
augmentation is a promising technique for improving generalization in RL, but
it is often found to decrease sample efficiency and can even lead to
divergence. In this paper, we investigate causes of instability when using data
augmentation in common off-policy RL algorithms. We identify two problems, both
rooted in high-variance Q-targets. Based on our findings, we propose a simple
yet effective technique for stabilizing this class of algorithms under
augmentation. We perform extensive empirical evaluation of image-based RL using
both ConvNets and Vision Transformers (ViT) on a family of benchmarks based on
DeepMind Control Suite, as well as in robotic manipulation tasks. Our method
greatly improves stability and sample efficiency of ConvNets under
augmentation, and achieves generalization results competitive with
state-of-the-art methods for image-based RL in environments with unseen
visuals. We further show that our method scales to RL with ViT-based
architectures, and that data augmentation may be especially important in this
setting.

    

### [[2111.14788] Function Approximation for High-Energy Physics: Comparing Machine Learning and Interpolation Methods](http://arxiv.org/abs/2111.14788)


  The need to approximate functions is ubiquitous in science, either due to
empirical constraints or high computational cost of accessing the function. In
high-energy physics, the precise computation of the scattering cross-section of
a process requires the evaluation of computationally intensive integrals. A
wide variety of methods in machine learning have been used to tackle this
problem, but often the motivation of using one method over another is lacking.
Comparing these methods is typically highly dependent on the problem at hand,
so we specify to the case where we can evaluate the function a large number of
times, after which quick and accurate evaluation can take place. We consider
four interpolation and three machine learning techniques and compare their
performance on three toy functions, the four-point scalar Passarino-Veltman
$D_0$ function, and the two-loop self-energy master integral $M$. We find that
in low dimensions ($d = 3$), traditional interpolation techniques like the
Radial Basis Function perform very well, but in higher dimensions ($d=5, 6, 9$)
we find that multi-layer perceptrons (a.k.a neural networks) do not suffer as
much from the curse of dimensionality and provide the fastest and most accurate
predictions.

    

### [[2112.02864] Autoencoders for Semivisible Jet Detection](http://arxiv.org/abs/2112.02864)


  The production of dark matter particles from confining dark sectors may lead
to many novel experimental signatures. Depending on the details of the theory,
dark quark production in proton-proton collisions could result in semivisible
jets of particles: collimated sprays of dark hadrons of which only some are
detectable by particle collider experiments. The experimental signature is
characterised by the presence of reconstructed missing momentum collinear with
the visible components of the jets. This complex topology is sensitive to
detector inefficiencies and mis-reconstruction that generate artificial missing
momentum. With this work, we propose a signal-agnostic strategy to reject
ordinary jets and identify semivisible jets via anomaly detection techniques. A
deep neural autoencoder network with jet substructure variables as input proves
highly useful for analyzing anomalous jets. The study focuses on the
semivisible jet signature; however, the technique can apply to any new physics
model that predicts signatures with jets from non-SM particles.

    

### [[2112.04715] Scheduling Algorithms for Hierarchical Fog Networks](http://arxiv.org/abs/2112.04715)


  Fog computing brings the functionality of the cloud near the edge of the
network with the help of fog devices/micro data centers ($mdcs$). Job
scheduling in such systems is a complex problem due to the hierarchical and
geo-distributed nature of fog devices. We propose two fog scheduling
algorithms, named $FiFSA$ (Hierarchical $Fi$rst $F$og $S$cheduling $A$lgorithm)
and $EFSA$ ( Hierarchical $E$lected $F$og $S$cheduling $A$lgorithm). We
consider a hierarchical model of fog devices, where the computation power of
fog devices present in higher tiers is greater than those present in lower
tiers. However, the higher tier fog devices are located at greater physical
distance from data generation sources as compared to lower tier fog devices.
Jobs with varying granularity and cpu requirements have been considered. In
general, jobs with modest cpu requirements are scheduled on lower tier fog
devices, and jobs with larger cpu requirements are scheduled on higher tier fog
devices or the cloud data center $(cdc)$. The performance of $FiFSA$ and $EFSA$
has been evaluated using a real life workload trace on various simulated fog
hierarchies as well as on a prototype testbed. Employing $FiFSA$ offers an
average improvement of 27% and 57.9% in total completion time and an
improvement of 32% and 61% in cost as compared to Longest Time First ($LTF$)
and cloud-only ($cdc-only$) scheduling algorithms, respectively. Employing
$EFSA$ offers an average improvement of 48% and 70% in total completion time
and an improvement of 52% and 72% in cost as compared to $LTF$ and $cdc-only$
respectively.

    

### [[2112.04778] Justifying the Dependability and Security of Business-Critical Blockchain-based Applications](http://arxiv.org/abs/2112.04778)


  In the industry, blockchains are increasingly used as the backbone of product
and process traceability. Blockchain-based traceability participates in the
demonstration of product and/or process compliance with existing safety
standards or quality criteria. In this perspective, services and applications
built on top of blockchains are business-critical applications, because an
intended failure or corruption of the system can lead to an important
reputation loss regarding the products or the processes involved. The
development of a blockchain-based business-critical application must be then
conducted carefully, requiring a thorough justification of its dependability
and security. To this end, this paper encourages an engineering perspective
rooted in well-understood tools and concepts borrowed from the engineering of
safety-critical systems. Concretely, we use a justification framework, called
CAE (Claim, Argument, Evidence), by following an approach based on assurance
cases, in order to provide convincing arguments that a business-critical
blockchain-based application is dependable and secure. The application of this
approach is sketched with a case study based on the blockchain HYPERLEDGER
FABRIC.

    

### [[2112.04845] High performance computing on Android devices -- a case study](http://arxiv.org/abs/2112.04845)


  High performance computing for low power devices can be useful to speed up
calculations on processors that use a lower clock rate than computers for which
energy efficiency is not an issue. In this trial, different high performance
techniques for Android devices have been compared, with a special focus on the
use of the GPU. Although not officially supported, the OpenCL framework can be
used on Android tablets. For the comparison of the different parallel
programming paradigms, a benchmark was chosen that could be implemented easily
with all frameworks. The Mandelbrot algorithm is computationally intensive and
has very few input and output operations. The algorithm has been implemented in
Java, C, C with assembler, C with SIMD assembler, C with OpenCL and scalar
instructions and C with OpenCL and vector instructions. The implementations
have been tested for all architectures currently supported by Android. High
speedups can be achieved using SIMD and OpenCL, although the implementation is
not straightforward for either one. Apps that use the GPU must account for the
fact that they can be suspended by the user at any moment. In using the OpenCL
framework on the GPU of Android devices, a computational power comparable to
those of modern high speed CPUs can be made available to the software
developer.

    

### [[2011.12984] Enabling GPU Accelerated Computing in the SUNDIALS Time Integration Library](http://arxiv.org/abs/2011.12984)


  As part of the Exascale Computing Project (ECP), a recent focus of
development efforts for the SUite of Nonlinear and DIfferential/ALgebraic
equation Solvers (SUNDIALS) has been to enable GPU-accelerated time integration
in scientific applications at extreme scales. This effort has resulted in
several new GPU-enabled implementations of core SUNDIALS data structures,
support for programming paradigms which are aware of the heterogeneous
architectures, and the introduction of utilities to provide new points of
flexibility. In this paper, we discuss our considerations, both internal and
external, when designing these new features and present the features
themselves. We also present performance results for several of the features on
the Summit supercomputer and early access hardware for the Frontier
supercomputer, which demonstrate negligible performance overhead resulting from
the additional infrastructure and significant speedups when using both NVIDIA
and AMD GPUs.

    

### [[2112.04596] Refined Commonsense Knowledge from Large-Scale Web Contents](http://arxiv.org/abs/2112.04596)


  Commonsense knowledge (CSK) about concepts and their properties is useful for
AI applications. Prior works like ConceptNet, COMET and others compiled large
CSK collections, but are restricted in their expressiveness to
subject-predicate-object (SPO) triples with simple concepts for S and strings
for P and O. This paper presents a method, called ASCENT++, to automatically
build a large-scale knowledge base (KB) of CSK assertions, with refined
expressiveness and both better precision and recall than prior works. ASCENT++
goes beyond SPO triples by capturing composite concepts with subgroups and
aspects, and by refining assertions with semantic facets. The latter is
important to express the temporal and spatial validity of assertions and
further qualifiers. ASCENT++ combines open information extraction with
judicious cleaning and ranking by typicality and saliency scores. For high
coverage, our method taps into the large-scale crawl C4 with broad web
contents. The evaluation with human judgements shows the superior quality of
the ASCENT++ KB, and an extrinsic evaluation for QA-support tasks underlines
the benefits of ASCENT++. A web interface, data and code can be accessed at
this https URL.

    

### [[2112.04674] DualFormer: Local-Global Stratified Transformer for Efficient Video Recognition](http://arxiv.org/abs/2112.04674)


  While transformers have shown great potential on video recognition tasks with
their strong capability of capturing long-range dependencies, they often suffer
high computational costs induced by self-attention operation on the huge number
of 3D tokens in a video. In this paper, we propose a new transformer
architecture, termed DualFormer, which can effectively and efficiently perform
space-time attention for video recognition. Specifically, our DualFormer
stratifies the full space-time attention into dual cascaded levels, i.e., to
first learn fine-grained local space-time interactions among nearby 3D tokens,
followed by the capture of coarse-grained global dependencies between the query
token and the coarse-grained global pyramid contexts. Different from existing
methods that apply space-time factorization or restrict attention computations
within local windows for improving efficiency, our local-global stratified
strategy can well capture both short- and long-range spatiotemporal
dependencies, and meanwhile greatly reduces the number of keys and values in
attention computation to boost efficiency. Experimental results show the
superiority of DualFormer on five video benchmarks against existing methods. In
particular, DualFormer sets new state-of-the-art 82.9%/85.2% top-1 accuracy on
Kinetics-400/600 with around 1000G inference FLOPs which is at least 3.2 times
fewer than existing methods with similar performances.

    

### [[2112.04685] CWS-PResUNet: Music Source Separation with Channel-wise Subband Phase-aware ResUNet](http://arxiv.org/abs/2112.04685)


  Music source separation (MSS) shows active progress with deep learning models
in recent years. Many MSS models perform separations on spectrograms by
estimating bounded ratio masks and reusing the phases of the mixture. When
using convolutional neural networks (CNN), weights are usually shared within a
spectrogram during convolution regardless of the different patterns between
frequency bands. In this study, we propose a new MSS model, channel-wise
subband phase-aware ResUNet (CWS-PResUNet), to decompose signals into subbands
and estimate an unbound complex ideal ratio mask (cIRM) for each source.
CWS-PResUNet utilizes a channel-wise subband (CWS) feature to limit unnecessary
global weights sharing on the spectrogram and reduce computational resource
consumptions. The saved computational cost and memory can in turn allow for a
larger architecture. On the MUSDB18HQ test set, we propose a 276-layer
CWS-PResUNet and achieve state-of-the-art (SoTA) performance on vocals with an
8.92 signal-to-distortion ratio (SDR) score. By combining CWS-PResUNet and
Demucs, our ByteMSS system ranks the 2nd on vocals score and 5th on average
score in the 2021 ISMIR Music Demixing (MDX) Challenge limited training data
track (leaderboard A). Our code and pre-trained models are publicly available
at: this https URL


### [[2112.04721] One-dimensional Deep Low-rank and Sparse Network for Accelerated MRI](http://arxiv.org/abs/2112.04721)


  Deep learning has shown astonishing performance in accelerated magnetic
resonance imaging (MRI). Most state-of-the-art deep learning reconstructions
adopt the powerful convolutional neural network and perform 2D convolution
since many magnetic resonance images or their corresponding k-space are in 2D.
In this work, we present a new approach that explores the 1D convolution,
making the deep network much easier to be trained and generalized. We further
integrate the 1D convolution into the proposed deep network, named as
One-dimensional Deep Low-rank and Sparse network (ODLS), which unrolls the
iteration procedure of a low-rank and sparse reconstruction model. Extensive
results on in vivo knee and brain datasets demonstrate that, the proposed ODLS
is very suitable for the case of limited training subjects and provides
improved reconstruction performance than state-of-the-art methods both visually
and quantitatively. Additionally, ODLS also shows nice robustness to different
undersampling scenarios and some mismatches between the training and test data.
In summary, our work demonstrates that the 1D deep learning scheme is
memory-efficient and robust in fast MRI.

    

### [[2112.04741] Learning multiple gaits of quadruped robot using hierarchical reinforcement learning](http://arxiv.org/abs/2112.04741)


  There is a growing interest in learning a velocity command tracking
controller of quadruped robot using reinforcement learning due to its
robustness and scalability. However, a single policy, trained end-to-end,
usually shows a single gait regardless of the command velocity. This could be a
suboptimal solution considering the existence of optimal gait according to the
velocity for quadruped animals. In this work, we propose a hierarchical
controller for quadruped robot that could generate multiple gaits (i.e. pace,
trot, bound) while tracking velocity command. Our controller is composed of two
policies, each working as a central pattern generator and local feedback
controller, and trained with hierarchical reinforcement learning. Experiment
results show 1) the existence of optimal gait for specific velocity range 2)
the efficiency of our hierarchical controller compared to a controller composed
of a single policy, which usually shows a single gait. Codes are publicly
available.

    

### [[2112.04748] LipSound2: Self-Supervised Pre-Training for Lip-to-Speech Reconstruction and Lip Reading](http://arxiv.org/abs/2112.04748)


  The aim of this work is to investigate the impact of crossmodal
self-supervised pre-training for speech reconstruction (video-to-audio) by
leveraging the natural co-occurrence of audio and visual streams in videos. We
propose LipSound2 which consists of an encoder-decoder architecture and
location-aware attention mechanism to map face image sequences to mel-scale
spectrograms directly without requiring any human annotations. The proposed
LipSound2 model is firstly pre-trained on $\sim$2400h multi-lingual (e.g.
English and German) audio-visual data (VoxCeleb2). To verify the
generalizability of the proposed method, we then fine-tune the pre-trained
model on domain-specific datasets (GRID, TCD-TIMIT) for English speech
reconstruction and achieve a significant improvement on speech quality and
intelligibility compared to previous approaches in speaker-dependent and
-independent settings. In addition to English, we conduct Chinese speech
reconstruction on the CMLR dataset to verify the impact on transferability.
Lastly, we train the cascaded lip reading (video-to-text) system by fine-tuning
the generated audios on a pre-trained speech recognition system and achieve
state-of-the-art performance on both English and Chinese benchmark datasets.

    

### [[2112.04751] Co-evolutionary hybrid intelligence](http://arxiv.org/abs/2112.04751)


  Artificial intelligence is one of the drivers of modern technological
development. The current approach to the development of intelligent systems is
data-centric. It has several limitations: it is fundamentally impossible to
collect data for modeling complex objects and processes; training neural
networks requires huge computational and energy resources; solutions are not
explainable. The article discusses an alternative approach to the development
of artificial intelligence systems based on human-machine hybridization and
their co-evolution.

    

### [[2112.04797] Complexity assessments for decidable fragments of Set Theory. III: A quadratic reduction of constraints over nested sets to Boolean formulae](http://arxiv.org/abs/2112.04797)


  As a contribution to quantitative set-theoretic inferencing, a translation is
proposed of conjunctions of literals of the forms $x=y\setminus z$, $x \neq
y\setminus z$, and $z =\{x\}$, where $x,y,z$ stand for variables ranging over
the von Neumann universe of sets, into unquantified Boolean formulae of a
rather simple conjunctive normal form. The formulae in the target language
involve variables ranging over a Boolean ring of sets, along with a difference
operator and relators designating equality, non-disjointness and inclusion.
Moreover, the result of each translation is a conjunction of literals of the
forms $x=y\setminus z$, $x\neq y\setminus z$ and of implications whose
antecedents are isolated literals and whose consequents are either inclusions
(strict or non-strict) between variables, or equalities between variables.
Besides reflecting a simple and natural semantics, which ensures
satisfiability-preservation, the proposed translation has quadratic algorithmic
time-complexity, and bridges two languages both of which are known to have an
NP-complete satisfiability problem.

    

### [[2112.04827] Explainability of the Implications of Supervised and Unsupervised Face Image Quality Estimations Through Activation Map Variation Analyses in Face Recognition Models](http://arxiv.org/abs/2112.04827)


  It is challenging to derive explainability for unsupervised or
statistical-based face image quality assessment (FIQA) methods. In this work,
we propose a novel set of explainability tools to derive reasoning for
different FIQA decisions and their face recognition (FR) performance
implications. We avoid limiting the deployment of our tools to certain FIQA
methods by basing our analyses on the behavior of FR models when processing
samples with different FIQA decisions. This leads to explainability tools that
can be applied for any FIQA method with any CNN-based FR solution using
activation mapping to exhibit the network's activation derived from the face
embedding. To avoid the low discrimination between the general spatial
activation mapping of low and high-quality images in FR models, we build our
explainability tools in a higher derivative space by analyzing the variation of
the FR activation maps of image sets with different quality decisions. We
demonstrate our tools and analyze the findings on four FIQA methods, by
presenting inter and intra-FIQA method analyses. Our proposed tools and the
analyses based on them point out, among other conclusions, that high-quality
images typically cause consistent low activation on the areas outside of the
central face region, while low-quality images, despite general low activation,
have high variations of activation in such areas. Our explainability tools also
extend to analyzing single images where we show that low-quality images tend to
have an FR model spatial activation that strongly differs from what is expected
from a high-quality image where this difference also tends to appear more in
areas outside of the central face region and does correspond to issues like
extreme poses and facial occlusions. The implementation of the proposed tools
is accessible here [link].

    

### [[2112.04889] Artificial Intelligence and Design of Experiments for Assessing Security of Electricity Supply: A Review and Strategic Outlook](http://arxiv.org/abs/2112.04889)


  Assessing the effects of the energy transition and liberalization of energy
markets on resource adequacy is an increasingly important and demanding task.
The rising complexity in energy systems requires adequate methods for energy
system modeling leading to increased computational requirements. Furthermore,
with complexity, uncertainty increases likewise calling for probabilistic
assessments and scenario analyses. To adequately and efficiently address these
various requirements, new methods from the field of data science are needed to
accelerate current methods. With our systematic literature review, we want to
close the gap between the three disciplines (1) assessment of security of
electricity supply, (2) artificial intelligence, and (3) design of experiments.
For this, we conduct a large-scale quantitative review on selected fields of
application and methods and make a synthesis that relates the different
disciplines to each other. Among other findings, we identify metamodeling of
complex security of electricity supply models using AI methods and applications
of AI-based methods for forecasts of storage dispatch and (non-)availabilities
as promising fields of application that have not sufficiently been covered,
yet. We end with deriving a new methodological pipeline for adequately and
efficiently addressing the present and upcoming challenges in the assessment of
security of electricity supply.

    

### [[2112.04937] DVHN: A Deep Hashing Framework for Large-scale Vehicle Re-identification](http://arxiv.org/abs/2112.04937)


  In this paper, we make the very first attempt to investigate the integration
of deep hash learning with vehicle re-identification. We propose a deep
hash-based vehicle re-identification framework, dubbed DVHN, which
substantially reduces memory usage and promotes retrieval efficiency while
reserving nearest neighbor search accuracy. Concretely,~DVHN directly learns
discrete compact binary hash codes for each image by jointly optimizing the
feature learning network and the hash code generating module. Specifically, we
directly constrain the output from the convolutional neural network to be
discrete binary codes and ensure the learned binary codes are optimal for
classification. To optimize the deep discrete hashing framework, we further
propose an alternating minimization method for learning binary
similarity-preserved hashing codes. Extensive experiments on two widely-studied
vehicle re-identification datasets- \textbf{VehicleID} and \textbf{VeRi}-~have
demonstrated the superiority of our method against the state-of-the-art deep
hash methods. \textbf{DVHN} of $2048$ bits can achieve 13.94\% and 10.21\%
accuracy improvement in terms of \textbf{mAP} and \textbf{Rank@1} for
\textbf{VehicleID (800)} dataset. For \textbf{VeRi}, we achieve 35.45\% and
32.72\% performance gains for \textbf{Rank@1} and \textbf{mAP}, respectively.

    

### [[2112.04957] Smart Support for Mission Success](http://arxiv.org/abs/2112.04957)


  Today's battlefield environment is complex, dynamic and uncertain, and
requires efficient support to ensure mission success. This relies on a proper
support strategy to provide supported equipment able to fulfill the mission. In
the context of defense where both systems and organization are complex, having
a holistic approach is challenging by nature, forces and support agencies need
to rely on an efficient decision support system. Logistics, readiness and
sustainability are critical factors for asset management, which can benefit
from AI to reach "Smart In Service" level relying especially on predictive and
prescriptive approaches and on effective management of operational re-sources.
Smart Support capacities can be then monitored by appropriate metrics and
improved by multi-criteria decision support and knowledge management system.
Depending on the operational context in terms of information and the objective,
different AI paradigms (data-driven AI, knowledge-based AI) are suitable even a
combination through hybrid AI.

    

### [[2112.05050] End-to-End Learning of Joint Geometric and Probabilistic Constellation Shaping](http://arxiv.org/abs/2112.05050)


  We present a novel autoencoder-based learning of joint geometric and
probabilistic constellation shaping for coded-modulation systems. It can
maximize either the mutual information (for symbol-metric decoding) or the
generalized mutual information (for bit-metric decoding).

    

### [[2112.05080] Locally Shifted Attention With Early Global Integration](http://arxiv.org/abs/2112.05080)


  Recent work has shown the potential of transformers for computer vision
applications. An image is first partitioned into patches, which are then used
as input tokens for the attention mechanism. Due to the expensive quadratic
cost of the attention mechanism, either a large patch size is used, resulting
in coarse-grained global interactions, or alternatively, attention is applied
only on a local region of the image, at the expense of long-range interactions.
In this work, we propose an approach that allows for both coarse global
interactions and fine-grained local interactions already at early layers of a
vision transformer.
At the core of our method is the application of local and global attention
layers. In the local attention layer, we apply attention to each patch and its
local shifts, resulting in virtually located local patches, which are not bound
to a single, specific location. These virtually located patches are then used
in a global attention layer. The separation of the attention layer into local
and global counterparts allows for a low computational cost in the number of
patches, while still supporting data-dependent localization already at the
first layer, as opposed to the static positioning in other visual transformers.
Our method is shown to be superior to both convolutional and transformer-based
methods for image classification on CIFAR10, CIFAR100, and ImageNet. Code is
available at: this https URL.

    

### [[2012.13577] LOREN: Logic-Regularized Reasoning for Interpretable Fact Verification](http://arxiv.org/abs/2012.13577)


  Given a natural language statement, how to verify its veracity against a
large-scale textual knowledge source like Wikipedia? Most existing neural
models make predictions without giving clues about which part of a false claim
goes wrong. In this paper, we propose LOREN, an approach for interpretable fact
verification. We decompose the verification of the whole claim at phrase-level,
where the veracity of the phrases serves as explanations and can be aggregated
into the final verdict according to logical rules. The key insight of LOREN is
to represent claim phrase veracity as three-valued latent variables, which are
regularized by aggregation logical rules. The final claim verification is based
on all latent variables. Thus, LOREN enjoys the additional benefit of
interpretability -- it is easy to explain how it reaches certain results with
claim phrase veracity. Experiments on a public fact verification benchmark show
that LOREN is competitive against previous approaches while enjoying the merit
of faithful and accurate interpretability. The resources of LOREN are available
at: this https URL.

    

### [[2112.04630] Towards Neural Functional Program Evaluation](http://arxiv.org/abs/2112.04630)


  This paper explores the capabilities of current transformer-based language
models for program evaluation of simple functional programming languages. We
introduce a new program generation mechanism that allows control over syntactic
sugar for semantically equivalent programs. T5 experiments reveal that neural
functional program evaluation performs surprisingly well, achieving high 90%
exact program match scores for most in-distribution and out-of-distribution
tests. Using pretrained T5 weights has significant advantages over random
initialization. We present and evaluate on three datasets to study
generalization abilities that are specific to functional programs based on:
type, function composition, and reduction steps. Code and data are publicly
available at this https URL.

    