
## 2021-10-14

### [<title>NaN values with early stopping - XGBoost</title>](https://discuss.xgboost.ai/t/nan-values-with-early-stopping/2494/4)

### [<title>NaN values with early stopping - XGBoost</title>](https://discuss.xgboost.ai/t/nan-values-with-early-stopping/2494/3)

### [[2110.06292] UCX Programming Interface for Remote Function Injection and Invocation](http://arxiv.org/abs/2110.06292)


  Network library APIs have historically been developed with the emphasis on
data movement, placement, and communication semantics. Many communication
semantics are available across a large variety of network libraries, such as
send-receive, data streaming, put/get/atomic, RPC, active messages, collective
communication, etc. In this work we introduce new compute and data movement
APIs that overcome the constraints of the single-program, multiple-data (SPMD)
programming model by allowing users to send binary executable code between
processing elements. Our proof-of-concept implementation of the API is based on
the UCX communication framework and leverages the RDMA network for fast compute
migration. We envision the API being used to dispatch user functions from a
host CPU to a SmartNIC (DPU), computational storage drive (CSD), or remote
servers. In addition, the API can be used by large-scale irregular applications
(such as semantic graph analysis), composed of many coordinating tasks
operating on a data set so big that it has to be stored on many physical
devices. In such cases, it may be more efficient to dynamically choose where
code runs as the applications progresses.

    

### [[2110.06508] Efficiency in the Serverless Cloud Computing Paradigm: A Survey Study](http://arxiv.org/abs/2110.06508)


  Serverless computing along with Function-as-a-Service (FaaS) are forming a
new computing paradigm that is anticipated to found the next generation of
cloud systems. The popularity of this paradigm is due to offering a highly
transparent infrastructure that enables user applications to scale in the
granularity of their functions. Since these often small and single-purpose
functions are managed on shared computing resources behind the scene, a great
potential for computational reuse and approximate computing emerges that if
unleashed, can remarkably improve the efficiency of serverless cloud systems --
both from the user's QoS and system's (energy consumption and incurred cost)
perspectives. Accordingly, the goal of this survey study is to, first, unfold
the internal mechanics of the serverless computing and, second, explore the
scope for efficiency within this paradigm via studying function reuse and
approximation approaches and discussing the pros and cons of each one. Next, we
outline potential future research directions within this paradigm that can
either unlock new use cases or make the paradigm more efficient.

    

### [[2110.06838] Full-stack Comparison of Channel Models for Networks Above 100 GHz in an Indoor Scenario](http://arxiv.org/abs/2110.06838)


  The Sixth Generation (6G) of mobile networks is expected to use carrier
frequencies in the spectrum above 100 GHz, to satisfy the demands for higher
data rates and bandwidth of future digital applications. The development of
networking solutions at such high frequencies is challenged by the harsh
propagation environment, and by the need for directional communications and
signal processing at high data rates. A fundamental step in defining and
developing wireless networks above 100 GHz is given by an accurate performance
evaluation. For simulations, this strongly depends on the accuracy of the
modeling of the channel and of the interaction with the higher layers of the
stack. This paper introduces the implementation of two recently proposed
channel models (based on ray tracing and on a fully stochastic model) for the
140 GHz band for the ns-3 TeraSim module, which enables simulation of macro
wireless networks in the sub-terahertz and terahertz spectrum. We also compare
the two channel models with full-stack simulations in an indoor scenario,
highlighting differences and similarities in how they interact with the
protocol stack and antenna model of TeraSim.

    

### [[2101.12704] Federated Learning over Wireless Device-to-Device Networks: Algorithms and Convergence Analysis](http://arxiv.org/abs/2101.12704)


  The proliferation of Internet-of-Things (IoT) devices and cloud-computing
applications over siloed data centers is motivating renewed interest in the
collaborative training of a shared model by multiple individual clients via
federated learning (FL). To improve the communication efficiency of FL
implementations in wireless systems, recent works have proposed compression and
dimension reduction mechanisms, along with digital and analog transmission
schemes that account for channel noise, fading, and interference. The prior art
has mainly focused on star topologies consisting of distributed clients and a
central server. In contrast, this paper studies FL over wireless
device-to-device (D2D) networks by providing theoretical insights into the
performance of digital and analog implementations of decentralized stochastic
gradient descent (DSGD). First, we introduce generic digital and analog
wireless implementations of communication-efficient DSGD algorithms, leveraging
random linear coding (RLC) for compression and over-the-air computation
(AirComp) for simultaneous analog transmissions. Next, under the assumptions of
convexity and connectivity, we provide convergence bounds for both
implementations. The results demonstrate the dependence of the optimality gap
on the connectivity and on the signal-to-noise ratio (SNR) levels in the
network. The analysis is corroborated by experiments on an image-classification
task.

    

### [[2103.14699] SkyQuery: An Aerial Drone Video Sensing Platform](http://arxiv.org/abs/2103.14699)


  Video-based sensing from aerial drones, especially small multirotor drones,
can provide rich data for numerous applications, including traffic analysis
(computing traffic flow volumes), precision agriculture (periodically
evaluating plant health), and wildlife population management (estimating
population sizes). However, aerial drone video sensing applications must handle
a surprisingly wide range of tasks: video frames must be aligned so that we can
equate coordinates of objects that appear in different frames, video data must
be analyzed to extract application-specific insights, and drone routes must be
computed that maximize the value of newly captured video. To address these
challenges, we built SkyQuery, a novel aerial drone video sensing platform that
provides an expressive, high-level programming language to make it
straightforward for users to develop complex long-running sensing applications.
SkyQuery combines novel methods for fast video frame alignment and detection of
small objects in top-down aerial drone video to efficiently execute
applications with diverse video analysis workflows and data distributions,
thereby allowing application developers to focus on the unique qualities of
their particular application rather than general video processing, data
analysis, and drone routing tasks. We conduct diverse case studies using
SkyQuery in parking monitoring, pedestrian activity mapping, and traffic hazard
detection scenarios to demonstrate the generalizability and effectiveness of
our system.

    

### [[2105.13389] GPS-Based Geolocation of Consumer IP Addresses](http://arxiv.org/abs/2105.13389)


  This paper uses two commercial datasets of IP addresses from smartphones,
geolocated through the Global Positioning System (GPS), to characterize the
geography of IP address subnets from mobile and broadband ISPs. Datasets that
ge olocate IP addresses based on GPS offer superlative accuracy and precision
for IP geolocation and thus provide an unprecedented opportunity to understand
both the accuracy of existing geolocation databases as well as other properties
of IP addresses, such as mobility and churn. We focus our analysis on large
cities in the United States.
After evaluating the accuracy of existing geolocation databases, we analyze
the circumstances under which IP geolocation databases may be more or less
accurate. We find that geolocation databases are more accurate on fixed-line
than mobile networks, that IP addresses on university networks can be more
accurately located than those from consumer or business networks, and that
often the paid versions of these databases are not significantly more accurate
than the free versions. We then characterize how quickly subnets associated
with fixed-line networks change geographic locations, and how long residential
broadband ISP subscribers retain individual IP addresses. We find, generally,
that most IP address assignments are stable over two months, although stability
does vary across ISPs. Finally, we evaluate the suitability of existing IP
geolocation databases for understanding Internet access and performance in
human populations within specific geographies and demographics. Although the
median accuracy of IP geolocation is better than 3 km in some contexts, we
conclude that relying on IP geolocation databases to understand Internet access
in densely populated regions such as cities is premature.

    

### [[2110.06209] An Introduction to Automatic Differentiation forMachine Learning](http://arxiv.org/abs/2110.06209)


  Machine learning and neural network models in particular have been improving
the state of the art performance on many artificial intelligence related tasks.
Neural network models are typically implemented using frameworks that perform
gradient based optimization methods to fit a model to a dataset. These
frameworks use a technique of calculating derivatives called automatic
differentiation (AD) which removes the burden of performing derivative
calculations from the model designer. In this report we describe AD, its
motivations, and different implementation approaches. We briefly describe
dataflow programming as it relates to AD. Lastly, we present example programs
that are implemented with Tensorflow and PyTorch, which are two commonly used
AD frameworks.

    

### [[2110.06241] Molecular Graph Generation via Geometric Scattering](http://arxiv.org/abs/2110.06241)


  Graph neural networks (GNNs) have been used extensively for addressing
problems in drug design and discovery. Both ligand and target molecules are
represented as graphs with node and edge features encoding information about
atomic elements and bonds respectively. Although existing deep learning models
perform remarkably well at predicting physicochemical properties and binding
affinities, the generation of new molecules with optimized properties remains
challenging. Inherently, most GNNs perform poorly in whole-graph representation
due to the limitations of the message-passing paradigm. Furthermore,
step-by-step graph generation frameworks that use reinforcement learning or
other sequential processing can be slow and result in a high proportion of
invalid molecules with substantial post-processing needed in order to satisfy
the principles of stoichiometry. To address these issues, we propose a
representation-first approach to molecular graph generation. We guide the
latent representation of an autoencoder by capturing graph structure
information with the geometric scattering transform and apply penalties that
structure the representation also by molecular properties. We show that this
highly structured latent space can be directly used for molecular graph
generation by the use of a GAN. We demonstrate that our architecture learns
meaningful representations of drug datasets and provides a platform for
goal-directed drug synthesis.

    

### [[2110.06255] Not all noise is accounted equally: How differentially private learning benefits from large sampling rates](http://arxiv.org/abs/2110.06255)


  Learning often involves sensitive data and as such, privacy preserving
extensions to Stochastic Gradient Descent (SGD) and other machine learning
algorithms have been developed using the definitions of Differential Privacy
(DP). In differentially private SGD, the gradients computed at each training
iteration are subject to two different types of noise. Firstly, inherent
sampling noise arising from the use of minibatches. Secondly, additive Gaussian
noise from the underlying mechanisms that introduce privacy. In this study, we
show that these two types of noise are equivalent in their effect on the
utility of private neural networks, however they are not accounted for equally
in the privacy budget. Given this observation, we propose a training paradigm
that shifts the proportions of noise towards less inherent and more additive
noise, such that more of the overall noise can be accounted for in the privacy
budget. With this paradigm, we are able to improve on the state-of-the-art in
the privacy/utility tradeoff of private end-to-end CNNs.

    

### [[2110.06256] On Convergence of Training Loss Without Reaching Stationary Points](http://arxiv.org/abs/2110.06256)


  It is a well-known fact that nonconvex optimization is computationally
intractable in the worst case. As a result, theoretical analysis of
optimization algorithms such as gradient descent often focuses on local
convergence to stationary points where the gradient norm is zero or negligible.
In this work, we examine the disconnect between the existing theoretical
analysis of gradient-based algorithms and actual practice. Specifically, we
provide numerical evidence that in large-scale neural network training, such as
in ImageNet, ResNet, and WT103 + TransformerXL models, the Neural Network
weight variables do not converge to stationary points where the gradient of the
loss function vanishes. Remarkably, however, we observe that while weights do
not converge to stationary points, the value of the loss function converges.
Inspired by this observation, we propose a new perspective based on ergodic
theory of dynamical systems. We prove convergence of the distribution of weight
values to an approximate invariant measure (without smoothness assumptions)
that explains this phenomenon. We further discuss how this perspective can
better align the theory with empirical observations.

    

### [[2110.06257] Causal discovery from conditionally stationary time-series](http://arxiv.org/abs/2110.06257)


  Causal discovery, i.e., inferring underlying cause-effect relationships from
observations of a scene or system, is an inherent mechanism in human cognition,
but has been shown to be highly challenging to automate. The majority of
approaches in the literature aiming for this task consider constrained
scenarios with fully observed variables or data from stationary time-series. In
this work we aim for causal discovery in a more general class of scenarios,
scenes with non-stationary behavior over time. For our purposes we here regard
a scene as a composition objects interacting with each other over time.
Non-stationarity is modeled as stationarity conditioned on an underlying
variable, a state, which can be of varying dimension, more or less hidden given
observations of the scene, and also depend more or less directly on these
observations. We propose a probabilistic deep learning approach called
State-Dependent Causal Inference (SDCI) for causal discovery in such
conditionally stationary time-series data. Results in two different synthetic
scenarios show that this method is able to recover the underlying causal
dependencies with high accuracy even in cases with hidden states.

    

### [[2110.06267] Twice regularized MDPs and the equivalence between robustness and regularization](http://arxiv.org/abs/2110.06267)


  Robust Markov decision processes (MDPs) aim to handle changing or partially
known system dynamics. To solve them, one typically resorts to robust
optimization methods. However, this significantly increases computational
complexity and limits scalability in both learning and planning. On the other
hand, regularized MDPs show more stability in policy learning without impairing
time complexity. Yet, they generally do not encompass uncertainty in the model
dynamics. In this work, we aim to learn robust MDPs using regularization. We
first show that regularized MDPs are a particular instance of robust MDPs with
uncertain reward. We thus establish that policy iteration on reward-robust MDPs
can have the same time complexity as on regularized MDPs. We further extend
this relationship to MDPs with uncertain transitions: this leads to a
regularization term with an additional dependence on the value function. We
finally generalize regularized MDPs to twice regularized MDPs (R${}^2$ MDPs),
i.e., MDPs with $\textit{both}$ value and policy regularization. The
corresponding Bellman operators enable developing policy iteration schemes with
convergence and robustness guarantees. It also reduces planning and learning in
robust MDPs to regularized MDPs.

    

### [[2110.06273] Småprat: DialoGPT for Natural Language Generation of Swedish Dialogue by Transfer Learning](http://arxiv.org/abs/2110.06273)


  Building open-domain conversational systems (or chatbots) that produce
convincing responses is a recognized challenge. Recent state-of-the-art (SoTA)
transformer-based models for the generation of natural language dialogue have
demonstrated impressive performance in simulating human-like, single-turn
conversations in English. This work investigates, by an empirical study, the
potential for transfer learning of such models to Swedish language. DialoGPT,
an English language pre-trained model, is adapted by training on three
different Swedish language conversational datasets obtained from publicly
available sources. Perplexity score (an automated intrinsic language model
metric) and surveys by human evaluation were used to assess the performances of
the fine-tuned models, with results that indicate that the capacity for
transfer learning can be exploited with considerable success. Human evaluators
asked to score the simulated dialogue judged over 57% of the chatbot responses
to be human-like for the model trained on the largest (Swedish) dataset. We
provide the demos and model checkpoints of our English and Swedish chatbots on
the HuggingFace platform for public use.

    

### [[2110.06280] S3PRL-VC: Open-source Voice Conversion Framework with Self-supervised Speech Representations](http://arxiv.org/abs/2110.06280)


  This paper introduces S3PRL-VC, an open-source voice conversion (VC)
framework based on the S3PRL toolkit. In the context of recognition-synthesis
VC, self-supervised speech representation (S3R) is valuable in its potential to
replace the expensive supervised representation adopted by state-of-the-art VC
systems. Moreover, we claim that VC is a good probing task for S3R analysis. In
this work, we provide a series of in-depth analyses by benchmarking on the two
tasks in VCC2020, namely intra-/cross-lingual any-to-one (A2O) VC, as well as
an any-to-any (A2A) setting. We also provide comparisons between not only
different S3Rs but also top systems in VCC2020 with supervised representations.
Systematic objective and subjective evaluation were conducted, and we show that
S3R is comparable with VCC2020 top systems in the A2O setting in terms of
similarity, and achieves state-of-the-art in S3R-based A2A VC. We believe the
extensive analysis, as well as the toolkit itself, contribute to not only the
S3R community but also the VC community. The codebase is now open-sourced.

    

### [[2110.06282] The Rich Get Richer: Disparate Impact of Semi-Supervised Learning](http://arxiv.org/abs/2110.06282)


  Semi-supervised learning (SSL) has demonstrated its potential to improve the
model accuracy for a variety of learning tasks when the high-quality supervised
data is severely limited. Although it is often established that the average
accuracy for the entire population of data is improved, it is unclear how SSL
fares with different sub-populations. Understanding the above question has
substantial fairness implications when these different sub-populations are
defined by the demographic groups we aim to treat fairly. In this paper, we
reveal the disparate impacts of deploying SSL: the sub-population who has a
higher baseline accuracy without using SSL (the ``rich" sub-population) tends
to benefit more from SSL; while the sub-population who suffers from a low
baseline accuracy (the ``poor" sub-population) might even observe a performance
drop after adding the SSL module. We theoretically and empirically establish
the above observation for a broad family of SSL algorithms, which either
explicitly or implicitly use an auxiliary ``pseudo-label". Our experiments on a
set of image and text classification tasks confirm our claims. We discuss how
this disparate impact can be mitigated and hope that our paper will alarm the
potential pitfall of using SSL and encourage a multifaceted evaluation of
future SSL algorithms. Code is available at this http URL.

    

### [[2110.06283] A Good Representation Detects Noisy Labels](http://arxiv.org/abs/2110.06283)


  Label noise is pervasive in real-world datasets, which encodes wrong
correlation patterns and impairs the generalization of deep neural networks
(DNNs). It is critical to find efficient ways to detect the corrupted patterns.
Current methods primarily focus on designing robust training techniques to
prevent DNNs from memorizing corrupted patterns. This approach has two
outstanding caveats: 1) applying this approach to each individual dataset would
often require customized training processes; 2) as long as the model is trained
with noisy supervisions, overfitting to corrupted patterns is often hard to
avoid, leading to performance drop in detection. In this paper, given good
representations, we propose a universally applicable and training-free solution
to detect noisy labels. Intuitively, good representations help define
``neighbors'' of each training instance, and closer instances are more likely
to share the same clean label. Based on the neighborhood information, we
propose two methods: the first one uses ``local voting" via checking the noisy
label consensuses of nearby representations. The second one is a ranking-based
approach that scores each instance and filters out a guaranteed number of
instances that are likely to be corrupted, again using only representations.
Given good (but possibly imperfect) representations that are commonly available
in practice, we theoretically analyze how they affect the local voting and
provide guidelines for tuning neighborhood size. We also prove the worst-case
error bound for the ranking-based method. Experiments with both synthetic and
real-world label noise demonstrate our training-free solutions are consistently
and significantly improving over most of the training-based baselines. Code is
available at this http URL.

    

### [[2110.06287] Real-Time Learning from An Expert in Deep Recommendation Systems with Marginal Distance Probability Distribution](http://arxiv.org/abs/2110.06287)


  Recommendation systems play an important role in today's digital world. They
have found applications in various applications such as music platforms, e.g.,
Spotify, and movie streaming services, e.g., Netflix. Less research effort has
been devoted to physical exercise recommendation systems. Sedentary lifestyles
have become the major driver of several diseases as well as healthcare costs.
In this paper, we develop a recommendation system for daily exercise activities
to users based on their history, profile and similar users. The developed
recommendation system uses a deep recurrent neural network with user-profile
attention and temporal attention mechanisms.
Moreover, exercise recommendation systems are significantly different from
streaming recommendation systems in that we are not able to collect click
feedback from the participants in exercise recommendation systems. Thus, we
propose a real-time, expert-in-the-loop active learning procedure. The active
learners calculate the uncertainty of the recommender at each time step for
each user and ask an expert for a recommendation when the certainty is low. In
this paper, we derive the probability distribution function of marginal
distance, and use it to determine when to ask experts for feedback. Our
experimental results on a mHealth dataset show improved accuracy after
incorporating the real-time active learner with the recommendation system.

    

### [[2110.06290] Scalable Consistency Training for Graph Neural Networks via Self-Ensemble Self-Distillation](http://arxiv.org/abs/2110.06290)


  Consistency training is a popular method to improve deep learning models in
computer vision and natural language processing. Graph neural networks (GNNs)
have achieved remarkable performance in a variety of network science learning
tasks, but to date no work has studied the effect of consistency training on
large-scale graph problems. GNNs scale to large graphs by minibatch training
and subsample node neighbors to deal with high degree nodes. We utilize the
randomness inherent in the subsampling of neighbors and introduce a novel
consistency training method to improve accuracy. For a target node we generate
different neighborhood expansions, and distill the knowledge of the average of
the predictions to the GNN. Our method approximates the expected prediction of
the possible neighborhood samples and practically only requires a few samples.
We demonstrate that our training method outperforms standard GNN training in
several different settings, and yields the largest gains when label rates are
low.

    

### [[2110.06296] The Role of Permutation Invariance in Linear Mode Connectivity of Neural Networks](http://arxiv.org/abs/2110.06296)


  In this paper, we conjecture that if the permutation invariance of neural
networks is taken into account, SGD solutions will likely have no barrier in
the linear interpolation between them. Although it is a bold conjecture, we
show how extensive empirical attempts fall short of refuting it. We further
provide a preliminary theoretical result to support our conjecture. Our
conjecture has implications for lottery ticket hypothesis, distributed
training, and ensemble methods.

    

### [[2110.06298] Domain Generalization via Domain-based Covariance Minimization](http://arxiv.org/abs/2110.06298)


  Researchers have been facing a difficult problem that data generation
mechanisms could be influenced by internal or external factors leading to the
training and test data with quite different distributions, consequently
traditional classification or regression from the training set is unable to
achieve satisfying results on test data. In this paper, we address this
nontrivial domain generalization problem by finding a central subspace in which
domain-based covariance is minimized while the functional relationship is
simultaneously maximally preserved. We propose a novel variance measurement for
multiple domains so as to minimize the difference between conditional
distributions across domains with solid theoretical demonstration and supports,
meanwhile, the algorithm preserves the functional relationship via maximizing
the variance of conditional expectations given output. Furthermore, we also
provide a fast implementation that requires much less computation and smaller
memory for large-scale matrix operations, suitable for not only domain
generalization but also other kernel-based eigenvalue decompositions. To show
the practicality of the proposed method, we compare our methods against some
well-known dimension reduction and domain generalization techniques on both
synthetic data and real-world applications. We show that for small-scale
datasets, we are able to achieve better quantitative results indicating better
generalization performance over unseen test datasets. For large-scale problems,
the proposed fast implementation maintains the quantitative performance but at
a substantially lower computational cost.

    

### [[2110.06306] Fine-grained style control in Transformer-based Text-to-speech Synthesis](http://arxiv.org/abs/2110.06306)


  In this paper, we present a novel architecture to realize fine-grained style
control on the transformer-based text-to-speech synthesis (TransformerTTS).
Specifically, we model the speaking style by extracting a time sequence of
local style tokens (LST) from the reference speech. The existing content
encoder in TransformerTTS is then replaced by our designed cross-attention
blocks for fusion and alignment between content and style. As the fusion is
performed along with the skip connection, our cross-attention block provides a
good inductive bias to gradually infuse the phoneme representation with a given
style. Additionally, we prevent the style embedding from encoding linguistic
content by randomly truncating LST during training and using wav2vec 2.0
features. Experiments show that with fine-grained style control, our system
performs better in terms of naturalness, intelligibility, and style
transferability. Our code and samples are publicly available.

    

### [[2110.06309] Exploring Wav2vec 2.0 fine-tuning for improved speech emotion recognition](http://arxiv.org/abs/2110.06309)


  While wav2vec 2.0 has been proposed for speech recognition (ASR), it can also
be used for speech emotion recognition (SER); its performance can be
significantly improved using different fine-tuning strategies. Two baseline
methods, vanilla fine-tuning (V-FT) and task adaptive pretraining (TAPT) are
first presented. We show that V-FT is able to outperform state-of-the-art
models on the IEMOCAP dataset. TAPT, an existing NLP fine-tuning strategy,
further improves the performance on SER. We also introduce a novel fine-tuning
method termed P-TAPT, which modifies the TAPT objective to learn contextualized
emotion representations. Experiments show that P-TAPT performs better than TAPT
especially under low-resource settings. Compared to prior works in this
literature, our top-line system achieved a 7.4% absolute improvement on
unweighted accuracy (UA) over the state-of-the-art performance on IEMOCAP. Our
code is publicly available.

    

### [[2110.06311] Incremental Community Detection in Distributed Dynamic Graph](http://arxiv.org/abs/2110.06311)


  Community detection is an important research topic in graph analytics that
has a wide range of applications. A variety of static community detection
algorithms and quality metrics were developed in the past few years. However,
most real-world graphs are not static and often change over time. In the case
of streaming data, communities in the associated graph need to be updated
either continuously or whenever new data streams are added to the graph, which
poses a much greater challenge in devising good community detection algorithms
for maintaining dynamic graphs over streaming data. In this paper, we propose
an incremental community detection algorithm for maintaining a dynamic graph
over streaming data. The contributions of this study include (a) the
implementation of a Distributed Weighted Community Clustering (DWCC) algorithm,
(b) the design and implementation of a novel Incremental Distributed Weighted
Community Clustering (IDWCC) algorithm, and (c) an experimental study to
compare the performance of our IDWCC algorithm with the DWCC algorithm. We
validate the functionality and efficiency of our framework in processing
streaming data and performing large in-memory distributed dynamic graph
analytics. The results demonstrate that our IDWCC algorithm performs up to
three times faster than the DWCC algorithm for a similar accuracy.

    

### [[2110.06324] PSML: A Multi-scale Time-series Dataset for Machine Learning in Decarbonized Energy Grids](http://arxiv.org/abs/2110.06324)


  The electric grid is a key enabling infrastructure for the ambitious
transition towards carbon neutrality as we grapple with climate change. With
deepening penetration of renewable energy resources and electrified
transportation, the reliable and secure operation of the electric grid becomes
increasingly challenging. In this paper, we present PSML, a first-of-its-kind
open-access multi-scale time-series dataset, to aid in the development of
data-driven machine learning (ML) based approaches towards reliable operation
of future electric grids. The dataset is generated through a novel transmission
+ distribution (T+D) co-simulation designed to capture the increasingly
important interactions and uncertainties of the grid dynamics, containing
electric load, renewable generation, weather, voltage and current measurements
at multiple spatio-temporal scales. Using PSML, we provide state-of-the-art ML
baselines on three challenging use cases of critical importance to achieve: (i)
early detection, accurate classification and localization of dynamic
disturbance events; (ii) robust hierarchical forecasting of load and renewable
energy with the presence of uncertainties and extreme events; and (iii)
realistic synthetic generation of physical-law-constrained measurement time
series. We envision that this dataset will enable advances for ML in dynamic
systems, while simultaneously allowing ML researchers to contribute towards
carbon-neutral electricity and mobility.

    

### [[2110.06325] As Easy as ABC: Adaptive Binning Coincidence Test for Uniformity Testing](http://arxiv.org/abs/2110.06325)


  We consider the problem of uniformity testing of Lipschitz continuous
distributions with bounded support. The alternative hypothesis is a composite
set of Lipschitz continuous distributions that are at least $\varepsilon$ away
in $\ell_1$ distance from the uniform distribution. We propose a sequential
test that adapts to the unknown distribution under the alternative hypothesis.
Referred to as the Adaptive Binning Coincidence (ABC) test, the proposed
strategy adapts in two ways. First, it partitions the set of alternative
distributions into layers based on their distances to the uniform distribution.
It then sequentially eliminates the alternative distributions layer by layer in
decreasing distance to the uniform, and subsequently takes advantage of
favorable situations of a distant alternative by exiting early. Second, it
adapts, across layers of the alternative distributions, the resolution level of
the discretization for computing the coincidence statistic. The farther away
the layer is from the uniform, the coarser the discretization is needed for
eliminating/exiting this layer. It thus exits both early in the detection
process and quickly by using a lower resolution to take advantage of favorable
alternative distributions. The ABC test builds on a novel sequential
coincidence test for discrete distributions, which is of independent interest.
We establish the sample complexity of the proposed tests as well as a lower
bound.

    

### [[2110.06340] A novel framework based on deep learning and ANOVA feature selection method for diagnosis of COVID-19 cases from chest X-ray Images](http://arxiv.org/abs/2110.06340)


  The new coronavirus (known as COVID-19) was first identified in Wuhan and
quickly spread worldwide, wreaking havoc on the economy and people's everyday
lives. Fever, cough, sore throat, headache, exhaustion, muscular aches, and
difficulty breathing are all typical symptoms of COVID-19. A reliable detection
technique is needed to identify affected individuals and care for them in the
early stages of COVID-19 and reduce the virus's transmission. The most
accessible method for COVID-19 identification is RT-PCR; however, due to its
time commitment and false-negative results, alternative options must be sought.
Indeed, compared to RT-PCR, chest CT scans and chest X-ray images provide
superior results. Because of the scarcity and high cost of CT scan equipment,
X-ray images are preferable for screening. In this paper, a pre-trained
network, DenseNet169, was employed to extract features from X-ray images.
Features were chosen by a feature selection method (ANOVA) to reduce
computations and time complexity while overcoming the curse of dimensionality
to improve predictive accuracy. Finally, selected features were classified by
XGBoost. The ChestX-ray8 dataset, which was employed to train and evaluate the
proposed method. This method reached 98.72% accuracy for two-class
classification (COVID-19, healthy) and 92% accuracy for three-class
classification (COVID-19, healthy, pneumonia).

    

### [[2110.06354] Tell Me How to Survey: Literature Review Made Simple with Automatic Reading Path Generation](http://arxiv.org/abs/2110.06354)


  Recent years have witnessed the dramatic growth of paper volumes with plenty
of new research papers published every day, especially in the area of computer
science. How to glean papers worth reading from the massive literature to do a
quick survey or keep up with the latest advancement about a specific research
topic has become a challenging task. Existing academic search engines such as
Google Scholar return relevant papers by individually calculating the relevance
between each paper and query. However, such systems usually omit the
prerequisite chains of a research topic and cannot form a meaningful reading
path. In this paper, we introduce a new task named Reading Path Generation
(RPG) which aims at automatically producing a path of papers to read for a
given query. To serve as a research benchmark, we further propose SurveyBank, a
dataset consisting of large quantities of survey papers in the field of
computer science as well as their citation relationships. Each survey paper
contains key phrases extracted from its title and multi-level reading lists
inferred from its references. Furthermore, we propose a
graph-optimization-based approach for reading path generation which takes the
relationship between papers into account. Extensive evaluations demonstrate
that our approach outperforms other baselines. A Real-time Reading Path
Generation System (RePaGer) has been also implemented with our designed model.
To the best of our knowledge, we are the first to target this important
research problem. Our source code of RePaGer system and SurveyBank dataset can
be found on here.

    

### [[2110.06357] Tangent Space and Dimension Estimation with the Wasserstein Distance](http://arxiv.org/abs/2110.06357)


  We provide explicit bounds on the number of sample points required to
estimate tangent spaces and intrinsic dimensions of (smooth, compact) Euclidean
submanifolds via local principal component analysis. Our approach directly
estimates covariance matrices locally, which simultaneously allows estimating
both the tangent spaces and the intrinsic dimension of a manifold. The key
arguments involve a matrix concentration inequality, a Wasserstein bound for
flattening a manifold, and a Lipschitz relation for the covariance matrix with
respect to the Wasserstein distance.

    

### [[2110.06365] Fast Approximations for Job Shop Scheduling: A Lagrangian Dual Deep Learning Method](http://arxiv.org/abs/2110.06365)


  The Jobs shop Scheduling Problem (JSP) is a canonical combinatorial
optimization problem that is routinely solved for a variety of industrial
purposes. It models the optimal scheduling of multiple sequences of tasks, each
under a fixed order of operations, in which individual tasks require exclusive
access to a predetermined resource for a specified processing time. The problem
is NP-hard and computationally challenging even for medium-sized instances.
Motivated by the increased stochasticity in production chains, this paper
explores a deep learning approach to deliver efficient and accurate
approximations to the JSP. In particular, this paper proposes the design of a
deep neural network architecture to exploit the problem structure, its
integration with Lagrangian duality to capture the problem constraints, and a
post-processing optimization to guarantee solution feasibility.The resulting
method, called JSP-DNN, is evaluated on hard JSP instances from the JSPLIB
benchmark library. Computational results show that JSP-DNN can produce JSP
approximations of high quality at negligible computational costs.

    

### [[2110.06367] Voice-assisted Image Labelling for Endoscopic Ultrasound Classification using Neural Networks](http://arxiv.org/abs/2110.06367)


  Ultrasound imaging is a commonly used technology for visualising patient
anatomy in real-time during diagnostic and therapeutic procedures. High
operator dependency and low reproducibility make ultrasound imaging and
interpretation challenging with a steep learning curve. Automatic image
classification using deep learning has the potential to overcome some of these
challenges by supporting ultrasound training in novices, as well as aiding
ultrasound image interpretation in patient with complex pathology for more
experienced practitioners. However, the use of deep learning methods requires a
large amount of data in order to provide accurate results. Labelling large
ultrasound datasets is a challenging task because labels are retrospectively
assigned to 2D images without the 3D spatial context available in vivo or that
would be inferred while visually tracking structures between frames during the
procedure. In this work, we propose a multi-modal convolutional neural network
(CNN) architecture that labels endoscopic ultrasound (EUS) images from raw
verbal comments provided by a clinician during the procedure. We use a CNN
composed of two branches, one for voice data and another for image data, which
are joined to predict image labels from the spoken names of anatomical
landmarks. The network was trained using recorded verbal comments from expert
operators. Our results show a prediction accuracy of 76% at image level on a
dataset with 5 different labels. We conclude that the addition of spoken
commentaries can increase the performance of ultrasound image classification,
and eliminate the burden of manually labelling large EUS datasets necessary for
deep learning applications.

    

### [[2110.06372] Data-driven Leak Localization in Water Distribution Networks via Dictionary Learning and Graph-based Interpolation](http://arxiv.org/abs/2110.06372)


  In this paper, we propose a data-driven leak localization method for water
distribution networks (WDNs) which combines two complementary approaches:
graph-based interpolation and dictionary classification. The former estimates
the complete WDN hydraulic state (i.e., hydraulic heads) from real measurements
at certain nodes and the network graph. Then, these actual measurements,
together with a subset of valuable estimated states, are used to feed and train
the dictionary learning scheme. Thus, the meshing of these two methods is
explored, showing that its performance is superior to either approach alone,
even deriving different mechanisms to increase its resilience to classical
problems (e.g., dimensionality, interpolation errors, etc.). The approach is
validated using the L-TOWN benchmark proposed at BattLeDIM2020.

    

### [[2110.06373] Enabling Level-4 Autonomous Driving on a Single $1k Off-the-Shelf Card](http://arxiv.org/abs/2110.06373)


  Autonomous driving is of great interest in both research and industry. The
high cost has been one of the major roadblocks that slow down the development
and adoption of autonomous driving in practice. This paper, for the first-time,
shows that it is possible to run level-4 (i.e., fully autonomous driving)
software on a single off-the-shelf card (Jetson AGX Xavier) for less than $1k,
an order of magnitude less than the state-of-the-art systems, while meeting all
the requirements of latency. The success comes from the resolution of some
important issues shared by existing practices through a series of measures and
innovations. The study overturns the common perceptions of the computing
resources required by level-4 autonomous driving, points out a promising path
for the industry to lower the cost, and suggests a number of research
opportunities for rethinking the architecture, software design, and
optimizations of autonomous driving.

    

### [[2110.06375] Coupled and Uncoupled Dynamic Mode Decomposition in Multi-Compartmental Systems with Applications to Epidemiological and Additive Manufacturing Problems](http://arxiv.org/abs/2110.06375)


  Dynamic Mode Decomposition (DMD) is an unsupervised machine learning method
that has attracted considerable attention in recent years owing to its
equation-free structure, ability to easily identify coherent spatio-temporal
structures in data, and effectiveness in providing reasonably accurate
predictions for certain problems. Despite these successes, the application of
DMD to certain problems featuring highly nonlinear transient dynamics remains
challenging. In such cases, DMD may not only fail to provide acceptable
predictions but may indeed fail to recreate the data in which it was trained,
restricting its application to diagnostic purposes. For many problems in the
biological and physical sciences, the structure of the system obeys a
compartmental framework, in which the transfer of mass within the system moves
within states. In these cases, the behavior of the system may not be accurately
recreated by applying DMD to a single quantity within the system, as proper
knowledge of the system dynamics, even for a single compartment, requires that
the behavior of other compartments is taken into account in the DMD process. In
this work, we demonstrate, theoretically and numerically, that, when performing
DMD on a fully coupled PDE system with compartmental structure, one may recover
useful predictive behavior, even when DMD performs poorly when acting
compartment-wise. We also establish that important physical quantities, as mass
conservation, are maintained in the coupled-DMD extrapolation. The mathematical
and numerical analysis suggests that DMD may be a powerful tool when applied to
this common class of problems. In particular, we show interesting numerical
applications to a continuous delayed-SIRD model for Covid-19, and to a problem
from additive manufacturing considering a nonlinear temperature field and the
resulting change of material phase from powder, liquid, and solid states.

    

### [[2110.06381] Meta Learning Low Rank Covariance Factors for Energy-Based Deterministic Uncertainty](http://arxiv.org/abs/2110.06381)


  Numerous recent works utilize bi-Lipschitz regularization of neural network
layers to preserve relative distances between data instances in the feature
spaces of each layer. This distance sensitivity with respect to the data aids
in tasks such as uncertainty calibration and out-of-distribution (OOD)
detection. In previous works, features extracted with a distance sensitive
model are used to construct feature covariance matrices which are used in
deterministic uncertainty estimation or OOD detection. However, in cases where
there is a distribution over tasks, these methods result in covariances which
are sub-optimal, as they may not leverage all of the meta information which can
be shared among tasks. With the use of an attentive set encoder, we propose to
meta learn either diagonal or diagonal plus low-rank factors to efficiently
construct task specific covariance matrices. Additionally, we propose an
inference procedure which utilizes scaled energy to achieve a final predictive
distribution which can better separate OOD data, and is well calibrated under a
distributional dataset shift.

    

### [[2110.06383] Real-time Drift Detection on Time-series Data](http://arxiv.org/abs/2110.06383)


  Practical machine learning applications involving time series data, such as
firewall log analysis to proactively detect anomalous behavior, are concerned
with real time analysis of streaming data. Consequently, we need to update the
ML models as the statistical characteristics of such data may shift frequently
with time. One alternative explored in the literature is to retrain models with
updated data whenever the models accuracy is observed to degrade. However,
these methods rely on near real time availability of ground truth, which is
rarely fulfilled. Further, in applications with seasonal data, temporal concept
drift is confounded by seasonal variation. In this work, we propose an approach
called Unsupervised Temporal Drift Detector or UTDD to flexibly account for
seasonal variation, efficiently detect temporal concept drift in time series
data in the absence of ground truth, and subsequently adapt our ML models to
concept drift for better generalization.

    

### [[2110.06384] AutoNLU: Detecting, root-causing, and fixing NLU model errors](http://arxiv.org/abs/2110.06384)


  Improving the quality of Natural Language Understanding (NLU) models, and
more specifically, task-oriented semantic parsing models, in production is a
cumbersome task. In this work, we present a system called AutoNLU, which we
designed to scale the NLU quality improvement process. It adds automation to
three key steps: detection, attribution, and correction of model errors, i.e.,
bugs. We detected four times more failed tasks than with random sampling,
finding that even a simple active learning sampling method on an uncalibrated
model is surprisingly effective for this purpose. The AutoNLU tool empowered
linguists to fix ten times more semantic parsing bugs than with prior manual
processes, auto-correcting 65% of all identified bugs.

    

### [[2110.06388] HETFORMER: Heterogeneous Transformer with Sparse Attention for Long-Text Extractive Summarization](http://arxiv.org/abs/2110.06388)


  To capture the semantic graph structure from raw text, most existing
summarization approaches are built on GNNs with a pre-trained model. However,
these methods suffer from cumbersome procedures and inefficient computations
for long-text documents. To mitigate these issues, this paper proposes
HETFORMER, a Transformer-based pre-trained model with multi-granularity sparse
attentions for long-text extractive summarization. Specifically, we model
different types of semantic nodes in raw text as a potential heterogeneous
graph and directly learn heterogeneous relationships (edges) among nodes by
Transformer. Extensive experiments on both single- and multi-document
summarization tasks show that HETFORMER achieves state-of-the-art performance
in Rouge F1 while using less memory and fewer parameters.

    

### [[2110.06389] Amortized Tree Generation for Bottom-up Synthesis Planning and Synthesizable Molecular Design](http://arxiv.org/abs/2110.06389)


  Molecular design and synthesis planning are two critical steps in the process
of molecular discovery that we propose to formulate as a single shared task of
conditional synthetic pathway generation. We report an amortized approach to
generate synthetic pathways as a Markov decision process conditioned on a
target molecular embedding. This approach allows us to conduct synthesis
planning in a bottom-up manner and design synthesizable molecules by decoding
from optimized conditional codes, demonstrating the potential to solve both
problems of design and synthesis simultaneously. The approach leverages neural
networks to probabilistically model the synthetic trees, one reaction step at a
time, according to reactivity rules encoded in a discrete action space of
reaction templates. We train these networks on hundreds of thousands of
artificial pathways generated from a pool of purchasable compounds and a list
of expert-curated templates. We validate our method with (a) the recovery of
molecules using conditional generation, (b) the identification of synthesizable
structural analogs, and (c) the optimization of molecular structures given
oracle functions relevant to drug discovery.

    

### [[2110.06390] Learning ground states of quantum Hamiltonians with graph networks](http://arxiv.org/abs/2110.06390)


  Solving for the lowest energy eigenstate of the many-body Schrodinger
equation is a cornerstone problem that hinders understanding of a variety of
quantum phenomena. The difficulty arises from the exponential nature of the
Hilbert space which casts the governing equations as an eigenvalue problem of
exponentially large, structured matrices. Variational methods approach this
problem by searching for the best approximation within a lower-dimensional
variational manifold. In this work we use graph neural networks to define a
structured variational manifold and optimize its parameters to find high
quality approximations of the lowest energy solutions on a diverse set of
Heisenberg Hamiltonians. Using graph networks we learn distributed
representations that by construction respect underlying physical symmetries of
the problem and generalize to problems of larger size. Our approach achieves
state-of-the-art results on a set of quantum many-body benchmark problems and
works well on problems whose solutions are not positive-definite. The discussed
techniques hold promise of being a useful tool for studying quantum many-body
systems and providing insights into optimization and implicit modeling of
exponentially-sized objects.

    

### [[2110.06394] Reward-Free Model-Based Reinforcement Learning with Linear Function Approximation](http://arxiv.org/abs/2110.06394)


  We study the model-based reward-free reinforcement learning with linear
function approximation for episodic Markov decision processes (MDPs). In this
setting, the agent works in two phases. In the exploration phase, the agent
interacts with the environment and collects samples without the reward. In the
planning phase, the agent is given a specific reward function and uses samples
collected from the exploration phase to learn a good policy. We propose a new
provably efficient algorithm, called UCRL-RFE under the Linear Mixture MDP
assumption, where the transition probability kernel of the MDP can be
parameterized by a linear function over certain feature mappings defined on the
triplet of state, action, and next state. We show that to obtain an
$\epsilon$-optimal policy for arbitrary reward function, UCRL-RFE needs to
sample at most $\tilde O(H^5d^2\epsilon^{-2})$ episodes during the exploration
phase. Here, $H$ is the length of the episode, $d$ is the dimension of the
feature mapping. We also propose a variant of UCRL-RFE using Bernstein-type
bonus and show that it needs to sample at most $\tilde O(H^4d(H +
d)\epsilon^{-2})$ to achieve an $\epsilon$-optimal policy. By constructing a
special class of linear Mixture MDPs, we also prove that for any reward-free
algorithm, it needs to sample at least $\tilde \Omega(H^2d\epsilon^{-2})$
episodes to obtain an $\epsilon$-optimal policy. Our upper bound matches the
lower bound in terms of the dependence on $\epsilon$ and the dependence on $d$
if $H \ge d$.

    

### [[2110.06395] Robust Neural Regression via Uncertainty Learning](http://arxiv.org/abs/2110.06395)


  Deep neural networks tend to underestimate uncertainty and produce overly
confident predictions. Recently proposed solutions, such as MC Dropout and
SDENet, require complex training and/or auxiliary out-of-distribution data. We
propose a simple solution by extending the time-tested iterative reweighted
least square (IRLS) in generalised linear regression. We use two sub-networks
to parametrise the prediction and uncertainty estimation, enabling easy
handling of complex inputs and nonlinear response. The two sub-networks have
shared representations and are trained via two complementary loss functions for
the prediction and the uncertainty estimates, with interleaving steps as in a
cooperative game. Compared with more complex models such as MC-Dropout or
SDE-Net, our proposed network is simpler to implement and more robust
(insensitive to varying aleatoric and epistemic uncertainty).

    

### [[2110.06399] Dynamic Inference with Neural Interpreters](http://arxiv.org/abs/2110.06399)


  Modern neural network architectures can leverage large amounts of data to
generalize well within the training distribution. However, they are less
capable of systematic generalization to data drawn from unseen but related
distributions, a feat that is hypothesized to require compositional reasoning
and reuse of knowledge. In this work, we present Neural Interpreters, an
architecture that factorizes inference in a self-attention network as a system
of modules, which we call \emph{functions}. Inputs to the model are routed
through a sequence of functions in a way that is end-to-end learned. The
proposed architecture can flexibly compose computation along width and depth,
and lends itself well to capacity extension after training. To demonstrate the
versatility of Neural Interpreters, we evaluate it in two distinct settings:
image classification and visual abstract reasoning on Raven Progressive
Matrices. In the former, we show that Neural Interpreters perform on par with
the vision transformer using fewer parameters, while being transferrable to a
new task in a sample efficient manner. In the latter, we find that Neural
Interpreters are competitive with respect to the state-of-the-art in terms of
systematic generalization

    

### [[2110.06400] CyTran: Cycle-Consistent Transformers for Non-Contrast to Contrast CT Translation](http://arxiv.org/abs/2110.06400)


  We propose a novel approach to translate unpaired contrast computed
tomography (CT) scans to non-contrast CT scans and the other way around.
Solving this task has two important applications: (i) to automatically generate
contrast CT scans for patients for whom injecting contrast substance is not an
option, and (ii) to enhance alignment between contrast and non-contrast CT by
reducing the differences induced by the contrast substance before registration.
Our approach is based on cycle-consistent generative adversarial convolutional
transformers, for short, CyTran. Our neural model can be trained on unpaired
images, due to the integration of a cycle-consistency loss. To deal with
high-resolution images, we design a hybrid architecture based on convolutional
and multi-head attention layers. In addition, we introduce a novel data set,
Coltea-Lung-CT-100W, containing 3D triphasic lung CT scans (with a total of
37,290 images) collected from 100 female patients. Each scan contains three
phases (non-contrast, early portal venous, and late arterial), allowing us to
perform experiments to compare our novel approach with state-of-the-art methods
for image style transfer. Our empirical results show that CyTran outperforms
all competing methods. Moreover, we show that CyTran can be employed as a
preliminary step to improve a state-of-the-art medical image alignment method.
We release our novel model and data set as open source at:
this https URL.

    

### [[2110.06416] MMIU: Dataset for Visual Intent Understanding in Multimodal Assistants](http://arxiv.org/abs/2110.06416)


  In multimodal assistant, where vision is also one of the input modalities,
the identification of user intent becomes a challenging task as visual input
can influence the outcome. Current digital assistants take spoken input and try
to determine the user intent from conversational or device context. So, a
dataset, which includes visual input (i.e. images or videos for the
corresponding questions targeted for multimodal assistant use cases, is not
readily available. The research in visual question answering (VQA) and visual
question generation (VQG) is a great step forward. However, they do not capture
questions that a visually-abled person would ask multimodal assistants.
Moreover, many times questions do not seek information from external knowledge.
In this paper, we provide a new dataset, MMIU (MultiModal Intent
Understanding), that contains questions and corresponding intents provided by
human annotators while looking at images. We, then, use this dataset for intent
classification task in multimodal digital assistant. We also experiment with
various approaches for combining vision and language features including the use
of multimodal transformer for classification of image-question pairs into 14
intents. We provide the benchmark results and discuss the role of visual and
text features for the intent classification task on our dataset.

    

### [[2110.06418] Stabilizing Dynamical Systems via Policy Gradient Methods](http://arxiv.org/abs/2110.06418)


  Stabilizing an unknown control system is one of the most fundamental problems
in control systems engineering. In this paper, we provide a simple, model-free
algorithm for stabilizing fully observed dynamical systems. While model-free
methods have become increasingly popular in practice due to their simplicity
and flexibility, stabilization via direct policy search has received
surprisingly little attention. Our algorithm proceeds by solving a series of
discounted LQR problems, where the discount factor is gradually increased. We
prove that this method efficiently recovers a stabilizing controller for linear
systems, and for smooth, nonlinear systems within a neighborhood of their
equilibria. Our approach overcomes a significant limitation of prior work,
namely the need for a pre-given stabilizing control policy. We empirically
evaluate the effectiveness of our approach on common control benchmarks.

    

### [[2110.06421] Revisiting Latent-Space Interpolation via a Quantitative Evaluation Framework](http://arxiv.org/abs/2110.06421)


  Latent-space interpolation is commonly used to demonstrate the generalization
ability of deep latent variable models. Various algorithms have been proposed
to calculate the best trajectory between two encodings in the latent space. In
this work, we show how data labeled with semantically continuous attributes can
be utilized to conduct a quantitative evaluation of latent-space interpolation
algorithms, for variational autoencoders. Our framework can be used to
complement the standard qualitative comparison, and also enables evaluation for
domains (such as graph) in which the visualization is difficult. Interestingly,
our experiments reveal that the superiority of interpolation algorithms could
be domain-dependent. While normalised interpolation works best for the image
domain, spherical linear interpolation achieves the best performance in the
graph domain. Next, we propose a simple-yet-effective method to restrict the
latent space via a bottleneck structure in the encoder. We find that all
interpolation algorithms evaluated in this work can benefit from this
restriction. Finally, we conduct interpolation-aware training with the labeled
attributes, and show that this explicit supervision can improve the
interpolation performance.

    

### [[2110.06427] Dense Uncertainty Estimation](http://arxiv.org/abs/2110.06427)


  Deep neural networks can be roughly divided into deterministic neural
networks and stochastic neural networks.The former is usually trained to
achieve a mapping from input space to output space via maximum likelihood
estimation for the weights, which leads to deterministic predictions during
testing. In this way, a specific weights set is estimated while ignoring any
uncertainty that may occur in the proper weight space. The latter introduces
randomness into the framework, either by assuming a prior distribution over
model parameters (i.e. Bayesian Neural Networks) or including latent variables
(i.e. generative models) to explore the contribution of latent variables for
model predictions, leading to stochastic predictions during testing. Different
from the former that achieves point estimation, the latter aims to estimate the
prediction distribution, making it possible to estimate uncertainty,
representing model ignorance about its predictions. We claim that conventional
deterministic neural network based dense prediction tasks are prone to
overfitting, leading to over-confident predictions, which is undesirable for
decision making. In this paper, we investigate stochastic neural networks and
uncertainty estimation techniques to achieve both accurate deterministic
prediction and reliable uncertainty estimation. Specifically, we work on two
types of uncertainty estimations solutions, namely ensemble based methods and
generative model based methods, and explain their pros and cons while using
them in fully/semi/weakly-supervised framework. Due to the close connection
between uncertainty estimation and model calibration, we also introduce how
uncertainty estimation can be used for deep model calibration to achieve
well-calibrated models, namely dense model calibration. Code and data are
available at this https URL.

    

### [[2110.06434] DeepA: A Deep Neural Analyzer For Speech And Singing Vocoding](http://arxiv.org/abs/2110.06434)


  Conventional vocoders are commonly used as analysis tools to provide
interpretable features for downstream tasks such as speech synthesis and voice
conversion. They are built under certain assumptions about the signals
following signal processing principle, therefore, not easily generalizable to
different audio, for example, from speech to singing. In this paper, we propose
a deep neural analyzer, denoted as DeepA - a neural vocoder that extracts F0
and timbre/aperiodicity encoding from the input speech that emulate those
defined in conventional vocoders. Therefore, the resulting parameters are more
interpretable than other latent neural representations. At the same time, as
the deep neural analyzer is learnable, it is expected to be more accurate for
signal reconstruction and manipulation, and generalizable from speech to
singing. The proposed neural analyzer is built based on a variational
autoencoder (VAE) architecture. We show that DeepA improves F0 estimation over
the conventional vocoder (WORLD). To our best knowledge, this is the first
study dedicated to the development of a neural framework for extracting
learnable vocoder-like parameters.

    

### [[2110.06435] Dropout Prediction Variation Estimation Using Neuron Activation Strength](http://arxiv.org/abs/2110.06435)


  It is well-known DNNs would generate different prediction results even given
the same model configuration and training dataset. As a result, it becomes more
and more important to study prediction variation, i.e. the variation of the
predictions on a given input example, in neural network models. Dropout has
been commonly used in various applications to quantify prediction variations.
However, using dropout in practice can be expensive as it requires running
dropout inference many times to estimate prediction variation.
In this paper, we study how to estimate dropout prediction variation in a
resource-efficient manner. In particular, we demonstrate that we can use neuron
activation strength to estimate dropout prediction variation under different
dropout settings and on a variety of tasks using three large datasets,
MovieLens, Criteo, and EMNIST. Our approach provides an inference-once
alternative to estimate dropout prediction variation as an auxiliary task when
the main prediction model is served. Moreover, we show that using activation
strength features from a subset of neural network layers can be sufficient to
achieve similar variation estimation performance compared to using activation
features from all layers. This can provide further resource reduction for
variation estimation.

    

### [[2110.06461] Fake News Detection in Spanish Using Deep Learning Techniques](http://arxiv.org/abs/2110.06461)


  This paper addresses the problem of fake news detection in Spanish using
Machine Learning techniques. It is fundamentally the same problem tackled for
the English language; however, there is not a significant amount of publicly
available and adequately labeled fake news in Spanish to effectively train a
Machine Learning model, similarly to those proposed for the English language.
Therefore, this work explores different training strategies and architectures
to establish a baseline for further research in this area. Four datasets were
used, two in English and two in Spanish, and four experimental schemes were
tested, including a baseline with classical Machine Learning models, trained
and validated using a small dataset in Spanish. The remaining schemes include
state-of-the-art Deep Learning models trained (or fine-tuned) and validated in
English, trained and validated in Spanish, and fitted in English and validated
with automatic translated Spanish sentences. The Deep Learning architectures
were built on top of different pre-trained Word Embedding representations,
including GloVe, ELMo, BERT, and BETO (a BERT version trained on a large corpus
in Spanish). According to the results, the best strategy was a combination of a
pre-trained BETO model and a Recurrent Neural Network based on LSTM layers,
yielding an accuracy of up to 80%; nonetheless, a baseline model using a Random
Forest estimator obtained similar outcomes. Additionally, the translation
strategy did not yield acceptable results because of the propagation error;
there was also observed a significant difference in models performance when
trained in English or Spanish, mainly attributable to the number of samples
available for each language.

    

### [[2110.06468] Graph-Fraudster: Adversarial Attacks on Graph Neural Network Based Vertical Federated Learning](http://arxiv.org/abs/2110.06468)


  Graph neural network (GNN) models have achieved great success on graph
representation learning. Challenged by large scale private data collection from
user-side, GNN models may not be able to reflect the excellent performance,
without rich features and complete adjacent relationships. Addressing to the
problem, vertical federated learning (VFL) is proposed to implement local data
protection through training a global model collaboratively. Consequently, for
graph-structured data, it is natural idea to construct VFL framework with GNN
models. However, GNN models are proven to be vulnerable to adversarial attacks.
Whether the vulnerability will be brought into the VFL has not been studied. In
this paper, we devote to study the security issues of GNN based VFL (GVFL),
i.e., robustness against adversarial attacks. Further, we propose an
adversarial attack method, named Graph-Fraudster. It generates adversarial
perturbations based on the noise-added global node embeddings via GVFL's
privacy leakage, and the gradient of pairwise node. First, it steals the global
node embeddings and sets up a shadow server model for attack generator. Second,
noises are added into node embeddings to confuse the shadow server model. At
last, the gradient of pairwise node is used to generate attacks with the
guidance of noise-added node embeddings. To the best of our knowledge, this is
the first study of adversarial attacks on GVFL. The extensive experiments on
five benchmark datasets demonstrate that Graph-Fraudster performs better than
three possible baselines in GVFL. Furthermore, Graph-Fraudster can remain a
threat to GVFL even if two possible defense mechanisms are applied. This paper
reveals that GVFL is vulnerable to adversarial attack similar to centralized
GNN models.

    

### [[2110.06475] SAR-Net: A Scenario-Aware Ranking Network for PersonalizedFair Recommendation in Hundreds of Travel Scenarios](http://arxiv.org/abs/2110.06475)


  The travel marketing platform of Alibaba serves an indispensable role for
hundreds of different travel scenarios from Fliggy, Taobao, Alipay apps, etc.
To provide personalized recommendation service for users visiting different
scenarios, there are two critical issues to be carefully addressed. First,
since the traffic characteristics of different scenarios, it is very
challenging to train a unified model to serve all. Second, during the promotion
period, the exposure of some specific items will be re-weighted due to manual
intervention, resulting in biased logs, which will degrade the ranking model
trained using these biased data. In this paper, we propose a novel
Scenario-Aware Ranking Network (SAR-Net) to address these issues. SAR-Net
harvests the abundant data from different scenarios by learning users'
cross-scenario interests via two specific attention modules, which leverage the
scenario features and item features to modulate the user behavior features,
respectively. Then, taking the encoded features of previous module as input, a
scenario-specific linear transformation layer is adopted to further extract
scenario-specific features, followed by two groups of debias expert networks,
i.e., scenario-specific experts and scenario-shared experts. They output
intermediate results independently, which are further fused into the final
result by a multi-scenario gating module. In addition, to mitigate the data
fairness issue caused by manual intervention, we propose the concept of
Fairness Coefficient (FC) to measures the importance of individual sample and
use it to reweigh the prediction in the debias expert networks. Experiments on
an offline dataset covering over 80 million users and 1.55 million travel items
and an online A/B test demonstrate the effectiveness of our SAR-Net and its
superiority over state-of-the-art methods.

    

### [[2110.06482] Parallel Deep Neural Networks Have Zero Duality Gap](http://arxiv.org/abs/2110.06482)


  Training deep neural networks is a well-known highly non-convex problem. In
recent works, it is shown that there is no duality gap for regularized
two-layer neural networks with ReLU activation, which enables global
optimization via convex programs. For multi-layer linear networks with vector
outputs, we formulate convex dual problems and demonstrate that the duality gap
is non-zero for depth three and deeper networks. However, by modifying the deep
networks to more powerful parallel architectures, we show that the duality gap
is exactly zero. Therefore, strong convex duality holds, and hence there exist
equivalent convex programs that enable training deep networks to global
optimality. We also demonstrate that the weight decay regularization in the
parameters explicitly encourages low-rank solutions via closed-form
expressions. For three-layer non-parallel ReLU networks, we show that strong
duality holds for rank-1 data matrices, however, the duality gap is non-zero
for whitened data matrices. Similarly, by transforming the neural network
architecture into a corresponding parallel version, the duality gap vanishes.

    

### [[2110.06488] The Convex Geometry of Backpropagation: Neural Network Gradient Flows Converge to Extreme Points of the Dual Convex Program](http://arxiv.org/abs/2110.06488)


  We study non-convex subgradient flows for training two-layer ReLU neural
networks from a convex geometry and duality perspective. We characterize the
implicit bias of unregularized non-convex gradient flow as convex
regularization of an equivalent convex model. We then show that the limit
points of non-convex subgradient flows can be identified via primal-dual
correspondence in this convex optimization problem. Moreover, we derive a
sufficient condition on the dual variables which ensures that the stationary
points of the non-convex objective are the KKT points of the convex objective,
thus proving convergence of non-convex gradient flows to the global optimum.
For a class of regular training data distributions such as orthogonal separable
data, we show that this sufficient condition holds. Therefore, non-convex
gradient flows in fact converge to optimal solutions of a convex optimization
problem. We present numerical results verifying the predictions of our theory
for non-convex subgradient descent.

    

### [[2110.06490] Dict-BERT: Enhancing Language Model Pre-training with Dictionary](http://arxiv.org/abs/2110.06490)


  Pre-trained language models (PLMs) aim to learn universal language
representations by conducting self-supervised training tasks on large-scale
corpora. Since PLMs capture word semantics in different contexts, the quality
of word representations highly depends on word frequency, which usually follows
a heavy-tailed distributions in the pre-training corpus. Therefore, the
embeddings of rare words on the tail are usually poorly optimized. In this
work, we focus on enhancing language model pre-training by leveraging
definitions of the rare words in dictionaries (e.g., Wiktionary). To
incorporate a rare word definition as a part of input, we fetch its definition
from the dictionary and append it to the end of the input text sequence. In
addition to training with the masked language modeling objective, we propose
two novel self-supervised pre-training tasks on word and sentence-level
alignment between input text sequence and rare word definitions to enhance
language modeling representation with dictionary. We evaluate the proposed
Dict-BERT model on the language understanding benchmark GLUE and eight
specialized domain benchmark datasets. Extensive experiments demonstrate that
Dict-BERT can significantly improve the understanding of rare words and boost
model performance on various NLP downstream tasks.

    

### [[2110.06500] Differentially Private Fine-tuning of Language Models](http://arxiv.org/abs/2110.06500)


  We give simpler, sparser, and faster algorithms for differentially private
fine-tuning of large-scale pre-trained language models, which achieve the
state-of-the-art privacy versus utility tradeoffs on many standard NLP tasks.
We propose a meta-framework for this problem, inspired by the recent success of
highly parameter-efficient methods for fine-tuning. Our experiments show that
differentially private adaptations of these approaches outperform previous
private algorithms in three important dimensions: utility, privacy, and the
computational and memory cost of private training. On many commonly studied
datasets, the utility of private models approaches that of non-private models.
For example, on the MNLI dataset we achieve an accuracy of $87.8\%$ using
RoBERTa-Large and $83.5\%$ using RoBERTa-Base with a privacy budget of
$\epsilon = 6.7$. In comparison, absent privacy constraints, RoBERTa-Large
achieves an accuracy of $90.2\%$. Our findings are similar for natural language
generation tasks. Privately fine-tuning with DART, GPT-2-Small, GPT-2-Medium,
GPT-2-Large, and GPT-2-XL achieve BLEU scores of 38.5, 42.0, 43.1, and 43.8
respectively (privacy budget of $\epsilon = 6.8,\delta=$ 1e-5) whereas the
non-private baseline is $48.1$. All our experiments suggest that larger models
are better suited for private fine-tuning: while they are well known to achieve
superior accuracy non-privately, we find that they also better maintain their
accuracy when privacy is introduced.

    

### [[2110.06509] Learning Stable Koopman Embeddings](http://arxiv.org/abs/2110.06509)


  In this paper, we present a new data-driven method for learning stable models
of nonlinear systems. Our model lifts the original state space to a
higher-dimensional linear manifold using Koopman embeddings. Interestingly, we
prove that every discrete-time nonlinear contracting model can be learnt in our
framework. Another significant merit of the proposed approach is that it allows
for unconstrained optimization over the Koopman embedding and operator jointly
while enforcing stability of the model, via a direct parameterization of stable
linear systems, greatly simplifying the computations involved. We validate our
method on a simulated system and analyze the advantages of our parameterization
compared to alternatives.

    

### [[2110.06510] The Dawn of Quantum Natural Language Processing](http://arxiv.org/abs/2110.06510)


  In this paper, we discuss the initial attempts at boosting understanding
human language based on deep-learning models with quantum computing. We
successfully train a quantum-enhanced Long Short-Term Memory network to perform
the parts-of-speech tagging task via numerical simulations. Moreover, a
quantum-enhanced Transformer is proposed to perform the sentiment analysis
based on the existing dataset.

    

### [[2110.06512] MedNet: Pre-trained Convolutional Neural Network Model for the Medical Imaging Tasks](http://arxiv.org/abs/2110.06512)


  Deep Learning (DL) requires a large amount of training data to provide
quality outcomes. However, the field of medical imaging suffers from the lack
of sufficient data for properly training DL models because medical images
require manual labelling carried out by clinical experts thus the process is
time-consuming, expensive, and error-prone. Recently, transfer learning (TL)
was introduced to reduce the need for the annotation procedure by means of
transferring the knowledge performed by a previous task and then fine-tuning
the result using a relatively small dataset. Nowadays, multiple classification
methods from medical imaging make use of TL from general-purpose pre-trained
models, e.g., ImageNet, which has been proven to be ineffective due to the
mismatch between the features learned from natural images (ImageNet) and those
more specific from medical images especially medical gray images such as
X-rays. ImageNet does not have grayscale images such as MRI, CT, and X-ray. In
this paper, we propose a novel DL model to be used for addressing
classification tasks of medical imaging, called MedNet. To do so, we aim to
issue two versions of MedNet. The first one is Gray-MedNet which will be
trained on 3M publicly available gray-scale medical images including MRI, CT,
X-ray, ultrasound, and PET. The second version is Color-MedNet which will be
trained on 3M publicly available color medical images including histopathology,
taken images, and many others. To validate the effectiveness MedNet, both
versions will be fine-tuned to train on the target tasks of a more reduced set
of medical images. MedNet performs as the pre-trained model to tackle any
real-world application from medical imaging and achieve the level of
generalization needed for dealing with medical imaging tasks, e.g.
classification. MedNet would serve the research community as a baseline for
future research.

    

### [[2110.06525] Automatic DJ Transitions with Differentiable Audio Effects and Generative Adversarial Networks](http://arxiv.org/abs/2110.06525)


  A central task of a Disc Jockey (DJ) is to create a mixset of mu-sic with
seamless transitions between adjacent tracks. In this paper, we explore a
data-driven approach that uses a generative adversarial network to create the
song transition by learning from real-world DJ mixes. In particular, the
generator of the model uses two differentiable digital signal processing
components, an equalizer (EQ) and a fader, to mix two tracks selected by a data
generation pipeline. The generator has to set the parameters of the EQs and
fader in such away that the resulting mix resembles real mixes created by
humanDJ, as judged by the discriminator counterpart. Result of a listening test
shows that the model can achieve competitive results compared with a number of
baselines.

    

### [[2110.06527] Sub-Setting Algorithm for Training Data Selection in Pattern Recognition](http://arxiv.org/abs/2110.06527)


  Modern pattern recognition tasks use complex algorithms that take advantage
of large datasets to make more accurate predictions than traditional algorithms
such as decision trees or k-nearest-neighbor better suited to describe simple
structures. While increased accuracy is often crucial, less complexity also has
value. This paper proposes a training data selection algorithm that identifies
multiple subsets with simple structures. A learning algorithm trained on such a
subset can classify an instance belonging to the subset with better accuracy
than the traditional learning algorithms. In other words, while existing
pattern recognition algorithms attempt to learn a global mapping function to
represent the entire dataset, we argue that an ensemble of simple local
patterns may better describe the data. Hence the sub-setting algorithm
identifies multiple subsets with simple local patterns by identifying similar
instances in the neighborhood of an instance. This motivation has similarities
to that of gradient boosted trees but focuses on the explainability of the
model that is missing for boosted trees. The proposed algorithm thus balances
accuracy and explainable machine learning by identifying a limited number of
subsets with simple structures. We applied the proposed algorithm to the
international stroke dataset to predict the probability of survival. Our
bottom-up sub-setting algorithm performed on an average 15% better than the
top-down decision tree learned on the entire dataset. The different decision
trees learned on the identified subsets use some of the previously unused
features by the whole dataset decision tree, and each subset represents a
distinct population of data.

    

### [[2110.06530] Reducing Information Bottleneck for Weakly Supervised Semantic Segmentation](http://arxiv.org/abs/2110.06530)


  Weakly supervised semantic segmentation produces pixel-level localization
from class labels; however, a classifier trained on such labels is likely to
focus on a small discriminative region of the target object. We interpret this
phenomenon using the information bottleneck principle: the final layer of a
deep neural network, activated by the sigmoid or softmax activation functions,
causes an information bottleneck, and as a result, only a subset of the
task-relevant information is passed on to the output. We first support this
argument through a simulated toy experiment and then propose a method to reduce
the information bottleneck by removing the last activation function. In
addition, we introduce a new pooling method that further encourages the
transmission of information from non-discriminative regions to the
classification. Our experimental evaluations demonstrate that this simple
modification significantly improves the quality of localization maps on both
the PASCAL VOC 2012 and MS COCO 2014 datasets, exhibiting a new
state-of-the-art performance for weakly supervised semantic segmentation. The
code is available at: this https URL.

    

### [[2110.06532] SMS: An Efficient Source Model Selection Framework for Model Reuse](http://arxiv.org/abs/2110.06532)


  With the explosive increase of big data, training a Machine Learning (ML)
model becomes a computation-intensive workload, which would take days or even
weeks. Thus, model reuse has received attention in the ML community, where it
is called transfer learning. Transfer learning avoids training a new model from
scratch by transferring knowledge from a source task to a target task. Existing
transfer learning methods mostly focus on how to improve the performance of the
target task through a specific source model, but assume that the source model
is given. As many source models are available, it is difficult for data
scientists to select the best source model for the target task manually. Hence,
how to efficiently select a suitable source model for model reuse is still an
unsolved problem.
In this paper, we propose SMS, an effective, efficient and flexible source
model selection framework. SMS is effective even when source and target
datasets have significantly different data labels, is flexible to support
source models with any type of structure, and is efficient to avoid any
training process. For each source model, SMS first vectorizes the samples in
the target dataset into soft labels by directly applying this model to the
target dataset, then uses Gaussian distributions to fit the clusters of soft
labels, and finally measures its distinguishing ability using Gaussian
mixture-based metric. Moreover, we present an improved SMS (I-SMS), which
decreases the output number of source model. I-SMS can significantly reduce the
selection time while retaining the selection performance of SMS. Extensive
experiments on a range of practical model reuse workloads demonstrate the
effectiveness and efficiency of SMS.

    

### [[2110.06537] Well-classified Examples are Underestimated in Classification with Deep Neural Networks](http://arxiv.org/abs/2110.06537)


  The conventional wisdom behind learning deep classification models is to
focus on bad-classified examples and ignore well-classified examples that are
far from the decision boundary. For instance, when training with cross-entropy
loss, examples with higher likelihoods (i.e., well-classified examples)
contribute smaller gradients in back-propagation. However, we theoretically
show that this common practice hinders representation learning, energy
optimization, and the growth of margin. To counteract this deficiency, we
propose to reward well-classified examples with additive bonuses to revive
their contribution to learning. This counterexample theoretically addresses
these three issues. We empirically support this claim by directly verify the
theoretical results or through the significant performance improvement with our
counterexample on diverse tasks, including image classification, graph
classification, and machine translation. Furthermore, this paper shows that
because our idea can solve these three issues, we can deal with complex
scenarios, such as imbalanced classification, OOD detection, and applications
under adversarial attacks.

    

### [[2110.06539] On Covariate Shift of Latent Confounders in Imitation and Reinforcement Learning](http://arxiv.org/abs/2110.06539)


  We consider the problem of using expert data with unobserved confounders for
imitation and reinforcement learning. We begin by defining the problem of
learning from confounded expert data in a contextual MDP setup. We analyze the
limitations of learning from such data with and without external reward, and
propose an adjustment of standard imitation learning algorithms to fit this
setup. We then discuss the problem of distribution shift between the expert
data and the online environment when the data is only partially observable. We
prove possibility and impossibility results for imitation learning under
arbitrary distribution shift of the missing covariates. When additional
external reward is provided, we propose a sampling procedure that addresses the
unknown shift and prove convergence to an optimal solution. Finally, we
validate our claims empirically on challenging assistive healthcare and
recommender system simulation tasks.

    

### [[2110.06543] EIHW-MTG DiCOVA 2021 Challenge System Report](http://arxiv.org/abs/2110.06543)


  This paper aims to automatically detect COVID-19 patients by analysing the
acoustic information embedded in coughs. COVID-19 affects the respiratory
system, and, consequently, respiratory-related signals have the potential to
contain salient information for the task at hand. We focus on analysing the
spectrogram representations of coughing samples with the aim to investigate
whether COVID-19 alters the frequency content of these signals. Furthermore,
this work also assesses the impact of gender in the automatic detection of
COVID-19. To extract deep learnt representations of the spectrograms, we
compare the performance of a cough-specific, and a Resnet18 pre-trained
Convolutional Neural Network (CNN). Additionally, our approach explores the use
of contextual attention, so the model can learn to highlight the most relevant
deep learnt features extracted by the CNN. We conduct our experiments on the
dataset released for the Cough Sound Track of the DiCOVA 2021 Challenge. The
best performance on the test set is obtained using the Resnet18 pre-trained CNN
with contextual attention, which scored an Area Under the Curve (AUC) of 70.91
at 80% sensitivity.

    

### [[2110.06546] A Melody-Unsupervision Model for Singing Voice Synthesis](http://arxiv.org/abs/2110.06546)


  Recent studies in singing voice synthesis have achieved high-quality results
leveraging advances in text-to-speech models based on deep neural networks. One
of the main issues in training singing voice synthesis models is that they
require melody and lyric labels to be temporally aligned with audio data. The
temporal alignment is a time-exhausting manual work in preparing for the
training data. To address the issue, we propose a melody-unsupervision model
that requires only audio-and-lyrics pairs without temporal alignment in
training time but generates singing voice audio given a melody and lyrics input
in inference time. The proposed model is composed of a phoneme classifier and a
singing voice generator jointly trained in an end-to-end manner. The model can
be fine-tuned by adjusting the amount of supervision with temporally aligned
melody labels. Through experiments in melody-unsupervision and semi-supervision
settings, we compare the audio quality of synthesized singing voice. We also
show that the proposed model is capable of being trained with speech audio and
text labels but can generate singing voice in inference time.

    

### [[2110.06556] Communication-Efficient Online Federated Learning Framework for Nonlinear Regression](http://arxiv.org/abs/2110.06556)


  Federated learning (FL) literature typically assumes that each client has a
fixed amount of data, which is unrealistic in many practical applications. Some
recent works introduced a framework for online FL (Online-Fed) wherein clients
perform model learning on streaming data and communicate the model to the
server; however, they do not address the associated communication overhead. As
a solution, this paper presents a partial-sharing-based online federated
learning framework (PSO-Fed) that enables clients to update their local models
using continuous streaming data and share only portions of those updated models
with the server. During a global iteration of PSO-Fed, non-participant clients
have the privilege to update their local models with new data. Here, we
consider a global task of kernel regression, where clients use a random Fourier
features-based kernel LMS on their data for local learning. We examine the mean
convergence of the PSO-Fed for kernel regression. Experimental results show
that PSO-Fed can achieve competitive performance with a significantly lower
communication overhead than Online-Fed.

    

### [[2110.06558] LENS: Localization enhanced by NeRF synthesis](http://arxiv.org/abs/2110.06558)


  Neural Radiance Fields (NeRF) have recently demonstrated photo-realistic
results for the task of novel view synthesis. In this paper, we propose to
apply novel view synthesis to the robot relocalization problem: we demonstrate
improvement of camera pose regression thanks to an additional synthetic dataset
rendered by the NeRF class of algorithm. To avoid spawning novel views in
irrelevant places we selected virtual camera locations from NeRF internal
representation of the 3D geometry of the scene. We further improved
localization accuracy of pose regressors using synthesized realistic and
geometry consistent images as data augmentation during training. At the time of
publication, our approach improved state of the art with a 60% lower error on
Cambridge Landmarks and 7-scenes datasets. Hence, the resulting accuracy
becomes comparable to structure-based methods, without any architecture
modification or domain adaptation constraints. Since our method allows almost
infinite generation of training data, we investigated limitations of camera
pose regression depending on size and distribution of data used for training on
public benchmarks. We concluded that pose regression accuracy is mostly bounded
by relatively small and biased datasets rather than capacity of the pose
regression model to solve the localization task.

    

### [[2110.06559] Infinitely Divisible Noise in the Low Privacy Regime](http://arxiv.org/abs/2110.06559)


  Federated learning, in which training data is distributed among users and
never shared, has emerged as a popular approach to privacy-preserving machine
learning. Cryptographic techniques such as secure aggregation are used to
aggregate contributions, like a model update, from all users. A robust
technique for making such aggregates differentially private is to exploit
infinite divisibility of the Laplace distribution, namely, that a Laplace
distribution can be expressed as a sum of i.i.d. noise shares from a Gamma
distribution, one share added by each user.
However, Laplace noise is known to have suboptimal error in the low privacy
regime for $\varepsilon$-differential privacy, where $\varepsilon > 1$ is a
large constant. In this paper we present the first infinitely divisible noise
distribution for real-valued data that achieves $\varepsilon$-differential
privacy and has expected error that decreases exponentially with $\varepsilon$.

    

### [[2110.06562] Unsupervised Object Learning via Common Fate](http://arxiv.org/abs/2110.06562)


  Learning generative object models from unlabelled videos is a long standing
problem and required for causal scene modeling. We decompose this problem into
three easier subtasks, and provide candidate solutions for each of them.
Inspired by the Common Fate Principle of Gestalt Psychology, we first extract
(noisy) masks of moving objects via unsupervised motion segmentation. Second,
generative models are trained on the masks of the background and the moving
objects, respectively. Third, background and foreground models are combined in
a conditional "dead leaves" scene model to sample novel scene configurations
where occlusions and depth layering arise naturally. To evaluate the individual
stages, we introduce the Fishbowl dataset positioned between complex real-world
scenes and common object-centric benchmarks of simplistic objects. We show that
our approach allows learning generative models that generalize beyond the
occlusions present in the input videos, and represent scenes in a modular
fashion that allows sampling plausible scenes outside the training distribution
by permitting, for instance, object numbers or densities not observed in the
training set.

    

### [[2110.06564] Deep Superpixel-based Network for Blind Image Quality Assessment](http://arxiv.org/abs/2110.06564)


  The goal in a blind image quality assessment (BIQA) model is to simulate the
process of evaluating images by human eyes and accurately assess the quality of
the image. Although many approaches effectively identify degradation, they do
not fully consider the semantic content in images resulting in distortion. In
order to fill this gap, we propose a deep adaptive superpixel-based network,
namely DSN-IQA, to assess the quality of image based on multi-scale and
superpixel segmentation. The DSN-IQA can adaptively accept arbitrary scale
images as input images, making the assessment process similar to human
perception. The network uses two models to extract multi-scale semantic
features and generate a superpixel adjacency map. These two elements are united
together via feature fusion to accurately predict image quality. Experimental
results on different benchmark databases demonstrate that our algorithm is
highly competitive with other approaches when assessing challenging authentic
image databases. Also, due to adaptive deep superpixel-based network, our model
accurately assesses images with complicated distortion, much like the human
eye.

    

### [[2110.06568] One to Multiple Mapping Dual Learning: Learning Multiple Sources from One Mixed Signal](http://arxiv.org/abs/2110.06568)


  Single channel blind source separation (SCBSS) refers to separate multiple
sources from a mixed signal collected by a single sensor. The existing methods
for SCBSS mainly focus on separating two sources and have weak generalization
performance. To address these problems, an algorithm is proposed in this paper
to separate multiple sources from a mixture by designing a parallel dual
generative adversarial Network (PDualGAN) that can build the relationship
between a mixture and the corresponding multiple sources to realize
one-to-multiple cross-domain mapping. This algorithm can be applied to any
mixed model such as linear instantaneous mixed model and convolutional mixed
model. Besides, one-to-multiple datasets are created which including the
mixtures and corresponding sources for this study. The experiment was carried
out on four different datasets and tested with signals mixed in different
proportions. Experimental results show that the proposed algorithm can achieve
high performance in peak signal-to-noise ratio (PSNR) and correlation, which
outperforms state-of-the-art algorithms.

    

### [[2110.06581] Averting A Crisis In Simulation-Based Inference](http://arxiv.org/abs/2110.06581)


  We present extensive empirical evidence showing that current Bayesian
simulation-based inference algorithms are inadequate for the falsificationist
methodology of scientific inquiry. Our results collected through months of
experimental computations show that all benchmarked algorithms -- (S)NPE,
(S)NRE, SNL and variants of ABC -- may produce overconfident posterior
approximations, which makes them demonstrably unreliable and dangerous if one's
scientific goal is to constrain parameters of interest. We believe that failing
to address this issue will lead to a well-founded trust crisis in
simulation-based inference. For this reason, we argue that research efforts
should now consider theoretical and methodological developments of conservative
approximate inference algorithms and present research directions towards this
objective. In this regard, we show empirical evidence that ensembles are
consistently more reliable.

    

### [[2110.06593] Clustering-Based Interpretation of Deep ReLU Network](http://arxiv.org/abs/2110.06593)


  Amongst others, the adoption of Rectified Linear Units (ReLUs) is regarded as
one of the ingredients of the success of deep learning. ReLU activation has
been shown to mitigate the vanishing gradient issue, to encourage sparsity in
the learned parameters, and to allow for efficient backpropagation. In this
paper, we recognize that the non-linear behavior of the ReLU function gives
rise to a natural clustering when the pattern of active neurons is considered.
This observation helps to deepen the learning mechanism of the network; in
fact, we demonstrate that, within each cluster, the network can be fully
represented as an affine map. The consequence is that we are able to recover an
explanation, in the form of feature importance, for the predictions done by the
network to the instances belonging to the cluster. Therefore, the methodology
we propose is able to increase the level of interpretability of a fully
connected feedforward ReLU neural network, downstream from the fitting phase of
the model, without altering the structure of the network. A simulation study
and the empirical application to the Titanic dataset, show the capability of
the method to bridge the gap between the algorithm optimization and the human
understandability of the black box deep ReLU networks.

    

### [[2110.06596] Logic Constraints to Feature Importances](http://arxiv.org/abs/2110.06596)


  In recent years, Artificial Intelligence (AI) algorithms have been proven to
outperform traditional statistical methods in terms of predictivity, especially
when a large amount of data was available. Nevertheless, the "black box" nature
of AI models is often a limit for a reliable application in high-stakes fields
like diagnostic techniques, autonomous guide, etc. Recent works have shown that
an adequate level of interpretability could enforce the more general concept of
model trustworthiness. The basic idea of this paper is to exploit the human
prior knowledge of the features' importance for a specific task, in order to
coherently aid the phase of the model's fitting. This sort of "weighted" AI is
obtained by extending the empirical loss with a regularization term encouraging
the importance of the features to follow predetermined constraints. This
procedure relies on local methods for the feature importance computation, e.g.
LRP, LIME, etc. that are the link between the model weights to be optimized and
the user-defined constraints on feature importance. In the fairness area,
promising experimental results have been obtained for the Adult dataset. Many
other possible applications of this model agnostic theoretical framework are
described.

    

### [[2110.06601] Vibration-Based Condition Monitoring By Ensemble Deep Learning](http://arxiv.org/abs/2110.06601)


  Vibration-based techniques are among the most common condition monitoring
approaches. With the advancement of computers, these approaches have also been
improved such that recently, these approaches in conjunction with deep learning
methods attract attention among researchers. This is mostly due to the nature
of the deep learning method that could facilitate the monitoring procedure by
integrating the feature extraction, feature selection, and classification steps
into one automated step. However, this can be achieved at the expense of
challenges in designing the architecture of a deep learner, tuning its
hyper-parameters. Moreover, it sometimes gives low generalization capability.
As a remedy to these problems, this study proposes a framework based on
ensemble deep learning methodology. The framework was initiated by creating a
pool of Convolutional neural networks (CNN). To create diversity to the CNNs,
they are fed by frequency responses which are passed through different
functions. As the next step, proper CNNs are selected based on an information
criterion to be used for fusion. The fusion is then carried out by improved
Dempster-Shafer theory. The proposed framework is applied to real test data
collected from Equiax Polycrystalline Nickel alloy first-stage turbine blades
with complex geometry.

    

### [[2110.06610] Metaparametric Neural Networks for Survival Analysis](http://arxiv.org/abs/2110.06610)


  Survival analysis is a critical tool for the modelling of time-to-event data,
such as life expectancy after a cancer diagnosis or optimal maintenance
scheduling for complex machinery. However, current neural network models
provide an imperfect solution for survival analysis as they either restrict the
shape of the target probability distribution or restrict the estimation to
pre-determined times. As a consequence, current survival neural networks lack
the ability to estimate a generic function without prior knowledge of its
structure. In this article, we present the metaparametric neural network
framework that encompasses existing survival analysis methods and enables their
extension to solve the aforementioned issues. This framework allows survival
neural networks to satisfy the same independence of generic function estimation
from the underlying data structure that characterizes their regression and
classification counterparts. Further, we demonstrate the application of the
metaparametric framework using both simulated and large real-world datasets and
show that it outperforms the current state-of-the-art methods in (i) capturing
nonlinearities, and (ii) identifying temporal patterns, leading to more
accurate overall estimations whilst placing no restrictions on the underlying
function structure.

    

### [[2110.06620] Maximizing Efficiency of Language Model Pre-training for Learning Representation](http://arxiv.org/abs/2110.06620)


  Pre-trained language models in the past years have shown exponential growth
in model parameters and compute time. ELECTRA is a novel approach for improving
the compute efficiency of pre-trained language models (e.g. BERT) based on
masked language modeling (MLM) by addressing the sample inefficiency problem
with the replaced token detection (RTD) task. Our work proposes adaptive early
exit strategy to maximize the efficiency of the pre-training process by
relieving the model's subsequent layers of the need to process latent features
by leveraging earlier layer representations. Moreover, we evaluate an initial
approach to the problem that has not succeeded in maintaining the accuracy of
the model while showing a promising compute efficiency by thoroughly
investigating the necessity of the generator module of ELECTRA.

    

### [[2110.06624] Identification of Metallic Objects using Spectral Magnetic Polarizability Tensor Signatures: Object Classification](http://arxiv.org/abs/2110.06624)


  The early detection of terrorist threat objects, such as guns and knives,
through improved metal detection, has the potential to reduce the number of
attacks and improve public safety and security. To achieve this, there is
considerable potential to use the fields applied and measured by a metal
detector to discriminate between different shapes and different metals since,
hidden within the field perturbation, is object characterisation information.
The magnetic polarizability tensor (MPT) offers an economical characterisation
of metallic objects and its spectral signature provides additional object
characterisation information. The MPT spectral signature can be determined from
measurements of the induced voltage over a range frequencies in a metal
signature for a hidden object. With classification in mind, it can also be
computed in advance for different threat and non-threat objects. In the
article, we evaluate the performance of probabilistic and non-probabilistic
machine learning algorithms, trained using a dictionary of computed MPT
spectral signatures, to classify objects for metal detection. We discuss the
importances of using appropriate features and selecting an appropriate
algorithm depending on the classification problem being solved and we present
numerical results for a range of practically motivated metal detection
classification problems.

    

### [[2110.06639] When saliency goes off on a tangent: Interpreting Deep Neural Networks with nonlinear saliency maps](http://arxiv.org/abs/2110.06639)


  A fundamental bottleneck in utilising complex machine learning systems for
critical applications has been not knowing why they do and what they do, thus
preventing the development of any crucial safety protocols. To date, no method
exist that can provide full insight into the granularity of the neural
network's decision process. In the past, saliency maps were an early attempt at
resolving this problem through sensitivity calculations, whereby dimensions of
a data point are selected based on how sensitive the output of the system is to
them. However, the success of saliency maps has been at best limited, mainly
due to the fact that they interpret the underlying learning system through a
linear approximation. We present a novel class of methods for generating
nonlinear saliency maps which fully account for the nonlinearity of the
underlying learning system. While agreeing with linear saliency maps on simple
problems where linear saliency maps are correct, they clearly identify more
specific drivers of classification on complex examples where nonlinearities are
more pronounced. This new class of methods significantly aids interpretability
of deep neural networks and related machine learning systems. Crucially, they
provide a starting point for their more broad use in serious applications,
where 'why' is equally important as 'what'.

    

### [[2110.06640] Detecting Slag Formations with Deep Convolutional Neural Networks](http://arxiv.org/abs/2110.06640)


  We investigate the ability to detect slag formations in images from inside a
Grate-Kiln system furnace with two deep convolutional neural networks. The
conditions inside the furnace cause occasional obstructions of the camera view.
Our approach suggests dealing with this problem by introducing a convLSTM-layer
in the deep convolutional neural network. The results show that it is possible
to achieve sufficient performance to automate the decision of timely
countermeasures in the industrial operational setting. Furthermore, the
addition of the convLSTM-layer results in fewer outlying predictions and a
lower running variance of the fraction of detected slag in the image time
series.

    

### [[2110.06641] Dictionary Learning with Convex Update (ROMD)](http://arxiv.org/abs/2110.06641)


  Dictionary learning aims to find a dictionary under which the training data
can be sparsely represented, and it is usually achieved by iteratively applying
two stages: sparse coding and dictionary update. Typical methods for dictionary
update focuses on refining both dictionary atoms and their corresponding sparse
coefficients by using the sparsity patterns obtained from sparse coding stage,
and hence it is a non-convex bilinear inverse problem. In this paper, we
propose a Rank-One Matrix Decomposition (ROMD) algorithm to recast this
challenge into a convex problem by resolving these two variables into a set of
rank-one matrices. Different from methods in the literature, ROMD updates the
whole dictionary at a time using convex programming. The advantages hence
include both convergence guarantees for dictionary update and faster
convergence of the whole dictionary learning. The performance of ROMD is
compared with other benchmark dictionary learning algorithms. The results show
the improvement of ROMD in recovery accuracy, especially in the cases of high
sparsity level and fewer observation data.

    

### [[2110.06650] Multistage linguistic conditioning of convolutional layers for speech emotion recognition](http://arxiv.org/abs/2110.06650)


  In this contribution, we investigate the effectiveness of deep fusion of text
and audio features for categorical and dimensional speech emotion recognition
(SER). We propose a novel, multistage fusion method where the two information
streams are integrated in several layers of a deep neural network (DNN), and
contrast it with a single-stage one where the streams are merged in a single
point. Both methods depend on extracting summary linguistic embeddings from a
pre-trained BERT model, and conditioning one or more intermediate
representations of a convolutional model operating on log-Mel spectrograms.
Experiments on the widely used IEMOCAP and MSP-Podcast databases demonstrate
that the two fusion methods clearly outperform a shallow (late) fusion baseline
and their unimodal constituents, both in terms of quantitative performance and
qualitative behaviour. Our accompanying analysis further reveals a hitherto
unexplored role of the underlying dialogue acts on unimodal and bimodal SER,
with different models showing a biased behaviour across different acts.
Overall, our multistage fusion shows better quantitative performance,
surpassing all alternatives on most of our evaluations. This illustrates the
potential of multistage fusion in better assimilating text and audio
information.

    

### [[2110.06663] Tutorial on Deep Learning for Human Activity Recognition](http://arxiv.org/abs/2110.06663)


  Activity recognition systems that are capable of estimating human activities
from wearable inertial sensors have come a long way in the past decades. Not
only have state-of-the-art methods moved away from feature engineering and have
fully adopted end-to-end deep learning approaches, best practices for setting
up experiments, preparing datasets, and validating activity recognition
approaches have similarly evolved. This tutorial was first held at the 2021 ACM
International Symposium on Wearable Computers (ISWC'21) and International Joint
Conference on Pervasive and Ubiquitous Computing (UbiComp'21). The tutorial,
after a short introduction in the research field of activity recognition,
provides a hands-on and interactive walk-through of the most important steps in
the data pipeline for the deep learning of human activities. All presentation
slides shown during the tutorial, which also contain links to all code
exercises, as well as the link of the GitHub page of the tutorial can be found
on: this https URL


### [[2110.06672] The deep generative decoder: Using MAP estimates of representations](http://arxiv.org/abs/2110.06672)


  A deep generative model is characterized by a representation space, its
distribution, and a neural network mapping the representation to a distribution
over vectors in feature space. Common methods such as variational autoencoders
(VAEs) apply variational inference for training the neural network, but
optimizing these models is often non-trivial. The encoder adds to the
complexity of the model and introduces an amortization gap and the quality of
the variational approximation is usually unknown. Additionally, the balance of
the loss terms of the objective function heavily influences performance.
Therefore, we argue that it is worthwhile to investigate a much simpler
approximation which finds representations and their distribution by maximizing
the model likelihood via back-propagation. In this approach, there is no
encoder, and we therefore call it a Deep Generative Decoder (DGD). Using the
CIFAR10 data set, we show that the DGD is easier and faster to optimize than
the VAE, achieves more consistent low reconstruction errors of test data, and
alleviates the problem of balancing the reconstruction and distribution loss
terms. Although the model in its simple form cannot compete with
state-of-the-art image generation approaches, it obtains better image
generation scores than the variational approach on the CIFAR10 data. We
demonstrate on MNIST data how the use of a Gaussian mixture with priors can
lead to a clear separation of classes in a 2D representation space, and how the
DGD can be used with labels to obtain a supervised representation.

    

### [[2110.06703] Expert-driven Trace Clustering with Instance-level Constraints](http://arxiv.org/abs/2110.06703)


  Within the field of process mining, several different trace clustering
approaches exist for partitioning traces or process instances into similar
groups. Typically, this partitioning is based on certain patterns or similarity
between the traces, or driven by the discovery of a process model for each
cluster. The main drawback of these techniques, however, is that their
solutions are usually hard to evaluate or justify by domain experts. In this
paper, we present two constrained trace clustering techniques that are capable
to leverage expert knowledge in the form of instance-level constraints. In an
extensive experimental evaluation using two real-life datasets, we show that
our novel techniques are indeed capable of producing clustering solutions that
are more justifiable without a substantial negative impact on their quality.

    

### [[2110.06717] On the Parameter Combinations That Matter and on Those That do Not](http://arxiv.org/abs/2110.06717)


  We present a data-driven approach to characterizing nonidentifiability of a
model's parameters and illustrate it through dynamic kinetic models. By
employing Diffusion Maps and their extensions, we discover the minimal
combinations of parameters required to characterize the dynamic output
behavior: a set of effective parameters for the model. Furthermore, we use
Conformal Autoencoder Neural Networks, as well as a kernel-based Jointly Smooth
Function technique, to disentangle the redundant parameter combinations that do
not affect the output behavior from the ones that do. We discuss the
interpretability of our data-driven effective parameters and demonstrate the
utility of the approach both for behavior prediction and parameter estimation.
In the latter task, it becomes important to describe level sets in parameter
space that are consistent with a particular output behavior. We validate our
approach on a model of multisite phosphorylation, where a reduced set of
effective parameters, nonlinear combinations of the physical ones, has
previously been established analytically.

    

### [[2110.06723] The Computerized Classification of Micro-Motions in the Hand using Waveforms from Mobile Phone](http://arxiv.org/abs/2110.06723)


  Our hands reveal important information such as the pulsing of our veins which
help us determine the blood pressure, tremors indicative of motor control, or
neurodegenerative disorders such as Essential Tremor or Parkinson's disease.
The Computerized Classification of Micro-Motions in the hand using waveforms
from mobile phone videos is a novel method that uses Eulerian Video
Magnification, Skeletonization, Heatmapping, and the kNN machine learning model
to detect the micro-motions in the human hand, synthesize their waveforms, and
classify these. The pre-processing is achieved by using Eulerian Video
Magnification, Skeletonization, and Heat-mapping to magnify the micro-motions,
landmark essential features of the hand, and determine the extent of motion,
respectively. Following pre-processing, the visible motions are manually
labeled by appropriately grouping pixels to represent a particular label
correctly. These labeled motions of the pixels are converted into waveforms.
Finally, these waveforms are classified into four categories - hand or finger
movements, vein movement, background motion, and movement of the rest of the
body due to respiration using the kNN model. The final accuracy obtained was
around 92 percent.

    

### [[2110.06726] Scalable Anytime Algorithms for Learning Formulas in Linear Temporal Logic](http://arxiv.org/abs/2110.06726)


  Linear temporal logic (LTL) is a specification language for finite sequences
(called traces) widely used in program verification, motion planning in
robotics, process mining, and many other areas. We consider the problem of
learning LTL formulas for classifying traces; despite a growing interest of the
research community, existing solutions suffer from two limitations: they do not
scale beyond small formulas, and they may exhaust computational resources
without returning any result. We introduce a new algorithm addressing both
issues: our algorithm is able to construct formulas an order of magnitude
larger than previous methods, and it is anytime, meaning that it in most cases
successfully outputs a formula, albeit possibly not of minimal size. We
evaluate the performances of our algorithm using an open source implementation
against publicly available benchmarks.

    

### [[2110.06735] A Time Encoding approach to training Spiking Neural Networks](http://arxiv.org/abs/2110.06735)


  While Spiking Neural Networks (SNNs) have been gaining in popularity, it
seems that the algorithms used to train them are not powerful enough to solve
the same tasks as those tackled by classical Artificial Neural Networks (ANNs).
In this paper, we provide an extra tool to help us understand and train SNNs by
using theory from the field of time encoding. Time encoding machines (TEMs) can
be used to model integrate-and-fire neurons and have well-understood
reconstruction properties. We will see how one can take inspiration from the
field of TEMs to interpret the spike times of SNNs as constraints on the SNNs'
weight matrices. More specifically, we study how to train one-layer SNNs by
solving a set of linear constraints, and how to train two-layer SNNs by
leveraging the all-or-none and asynchronous properties of the spikes emitted by
SNNs. These properties of spikes result in an alternative to backpropagation
which is not possible in the case of simultaneous and graded activations as in
classical ANNs.

    

### [[2110.06740] Transform and Bitstream Domain Image Classification](http://arxiv.org/abs/2110.06740)


  Classification of images within the compressed domain offers significant
benefits. These benefits include reduced memory and computational requirements
of a classification system. This paper proposes two such methods as a proof of
concept: The first classifies within the JPEG image transform domain (i.e. DCT
transform data); the second classifies the JPEG compressed binary bitstream
directly. These two methods are implemented using Residual Network CNNs and an
adapted Vision Transformer. Top-1 accuracy of approximately 70% and 60% were
achieved using these methods respectively when classifying the Caltech C101
database. Although these results are significantly behind the state of the art
for classification for this database (~95%), it illustrates the first time
direct bitstream image classification has been achieved. This work confirms
that direct bitstream image classification is possible and could be utilised in
a first pass database screening of a raw bitstream (within a wired or wireless
network) or where computational, memory and bandwidth requirements are severely
restricted.

    

### [[2110.06741] Dynamical Wasserstein Barycenters for Time-series Modeling](http://arxiv.org/abs/2110.06741)


  Many time series can be modeled as a sequence of segments representing
high-level discrete states, such as running and walking in a human activity
application. Flexible models should describe the system state and observations
in stationary ``pure-state'' periods as well as transition periods between
adjacent segments, such as a gradual slowdown between running and walking.
However, most prior work assumes instantaneous transitions between pure
discrete states. We propose a dynamical Wasserstein barycentric (DWB) model
that estimates the system state over time as well as the data-generating
distributions of pure states in an unsupervised manner. Our model assumes each
pure state generates data from a multivariate normal distribution, and
characterizes transitions between states via displacement-interpolation
specified by the Wasserstein barycenter. The system state is represented by a
barycentric weight vector which evolves over time via a random walk on the
simplex. Parameter learning leverages the natural Riemannian geometry of
Gaussian distributions under the Wasserstein distance, which leads to improved
convergence speeds. Experiments on several human activity datasets show that
our proposed DWB model accurately learns the generating distribution of pure
states while improving state estimation for transition periods compared to the
commonly used linear interpolation mixture models.

    

### [[2110.06742] A Review of the Deep Sea Treasure problem as a Multi-Objective Reinforcement Learning Benchmark](http://arxiv.org/abs/2110.06742)


  In this paper, the authors investigate the Deep Sea Treasure (DST) problem as
proposed by Vamplew et al. Through a number of proofs, the authors show the
original DST problem to be quite basic, and not always representative of
practical Multi-Objective Optimization problems. In an attempt to bring theory
closer to practice, the authors propose an alternative, improved version of the
DST problem, and prove that some of the properties that simplify the original
DST problem no longer hold. The authors also provide a reference implementation
and perform a comparison between their implementation, and other existing
open-source implementations of the problem. Finally, the authors also provide a
complete Pareto-front for their new DST problem.

    

### [[2110.06751] Improving the sample-efficiency of neural architecture search with reinforcement learning](http://arxiv.org/abs/2110.06751)


  Designing complex architectures has been an essential cogwheel in the
revolution deep learning has brought about in the past decade. When solving
difficult problems in a datadriven manner, a well-tried approach is to take an
architecture discovered by renowned deep learning scientists as a basis (e.g.
Inception) and try to apply it to a specific problem. This might be sufficient,
but as of now, achieving very high accuracy on a complex or yet unsolved task
requires the knowledge of highly-trained deep learning experts. In this work,
we would like to contribute to the area of Automated Machine Learning (AutoML),
specifically Neural Architecture Search (NAS), which intends to make deep
learning methods available for a wider range of society by designing neural
topologies automatically. Although several different approaches exist (e.g.
gradient-based or evolutionary algorithms), our focus is on one of the most
promising research directions, reinforcement learning. In this scenario, a
recurrent neural network (controller) is trained to create problem-specific
neural network architectures (child). The validation accuracies of the child
networks serve as a reward signal for training the controller with
reinforcement learning. The basis of our proposed work is Efficient Neural
Architecture Search (ENAS), where parameter sharing is applied among the child
networks. ENAS, like many other RL-based algorithms, emphasize the learning of
child networks as increasing their convergence result in a denser reward signal
for the controller, therefore significantly reducing training times. The
controller was originally trained with REINFORCE. In our research, we propose
to modify this to a more modern and complex algorithm, PPO, which has
demonstrated to be faster and more stable in other environments. Then, we
briefly discuss and evaluate our results.

    

### [[2110.06763] Efficient Estimation in NPIV Models: A Comparison of Various Neural Networks-Based Estimators](http://arxiv.org/abs/2110.06763)


  We investigate the computational performance of Artificial Neural Networks
(ANNs) in semi-nonparametric instrumental variables (NPIV) models of high
dimensional covariates that are relevant to empirical work in economics. We
focus on efficient estimation of and inference on expectation functionals (such
as weighted average derivatives) and use optimal criterion-based procedures
(sieve minimum distance or SMD) and novel efficient score-based procedures
(ES). Both these procedures use ANN to approximate the unknown function. Then,
we provide a detailed practitioner's recipe for implementing these two classes
of estimators. This involves the choice of tuning parameters both for the
unknown functions (that include conditional expectations) but also for the
choice of estimation of the optimal weights in SMD and the Riesz representers
used with the ES estimators. Finally, we conduct a large set of Monte Carlo
experiments that compares the finite-sample performance in complicated designs
that involve a large set of regressors (up to 13 continuous), and various
underlying nonlinearities and covariate correlations. Some of the takeaways
from our results include: 1) tuning and optimization are delicate especially as
the problem is nonconvex; 2) various architectures of the ANNs do not seem to
matter for the designs we consider and given proper tuning, ANN methods perform
well; 3) stable inferences are more difficult to achieve with ANN estimators;
4) optimal SMD based estimators perform adequately; 5) there seems to be a gap
between implementation and approximation theory. Finally, we apply ANN NPIV to
estimate average price elasticity and average derivatives in two demand
examples.

    

### [[2110.06766] Next-Best-View Estimation based on Deep Reinforcement Learning for Active Object Classification](http://arxiv.org/abs/2110.06766)


  The presentation and analysis of image data from a single viewpoint are often
not sufficient to solve a task. Several viewpoints are necessary to obtain more
information. The $\textit{next-best-view}$ problem attempts to find the optimal
viewpoint with the greatest information gain for the underlying task. In this
work, a robot arm holds an object in its end-effector and searches for a
sequence of next-best-view to explicitly identify the object. We use Soft
Actor-Critic (SAC), a method of deep reinforcement learning, to learn these
next-best-views for a specific set of objects. The evaluation shows that an
agent can learn to determine an object pose to which the robot arm should move
an object. This leads to a viewpoint that provides a more accurate prediction
to distinguish such an object from other objects better. We make the code
publicly available for the scientific community and for reproducibility under
$\href{this https URL}{\text{this https link}}$.

    

### [[2110.06773] Leveraging Automated Unit Tests for Unsupervised Code Translation](http://arxiv.org/abs/2110.06773)


  With little to no parallel data available for programming languages,
unsupervised methods are well-suited to source code translation. However, the
majority of unsupervised machine translation approaches rely on
back-translation, a method developed in the context of natural language
translation and one that inherently involves training on noisy inputs.
Unfortunately, source code is highly sensitive to small changes; a single token
can result in compilation failures or erroneous programs, unlike natural
languages where small inaccuracies may not change the meaning of a sentence. To
address this issue, we propose to leverage an automated unit-testing system to
filter out invalid translations, thereby creating a fully tested parallel
corpus. We found that fine-tuning an unsupervised model with this filtered data
set significantly reduces the noise in the translations so-generated,
comfortably outperforming the state-of-the-art for all language pairs studied.
In particular, for Java $\to$ Python and Python $\to$ C++ we outperform the
best previous methods by more than 16% and 24% respectively, reducing the error
rate by more than 35%.

    

### [[2110.06777] Incremental Ensemble Gaussian Processes](http://arxiv.org/abs/2110.06777)


  Belonging to the family of Bayesian nonparametrics, Gaussian process (GP)
based approaches have well-documented merits not only in learning over a rich
class of nonlinear functions, but also in quantifying the associated
uncertainty. However, most GP methods rely on a single preselected kernel
function, which may fall short in characterizing data samples that arrive
sequentially in time-critical applications. To enable {\it online} kernel
adaptation, the present work advocates an incremental ensemble (IE-) GP
framework, where an EGP meta-learner employs an {\it ensemble} of GP learners,
each having a unique kernel belonging to a prescribed kernel dictionary. With
each GP expert leveraging the random feature-based approximation to perform
online prediction and model update with {\it scalability}, the EGP meta-learner
capitalizes on data-adaptive weights to synthesize the per-expert predictions.
Further, the novel IE-GP is generalized to accommodate time-varying functions
by modeling structured dynamics at the EGP meta-learner and within each GP
learner. To benchmark the performance of IE-GP and its dynamic variant in the
adversarial setting where the modeling assumptions are violated, rigorous
performance analysis has been conducted via the notion of regret, as the norm
in online convex optimization. Last but not the least, online unsupervised
learning for dimensionality reduction is explored under the novel IE-GP
framework. Synthetic and real data tests demonstrate the effectiveness of the
proposed schemes.

    

### [[2110.06787] Adapting to Dynamic LEO-B5G Systems: Meta-Critic Learning Based Efficient Resource Scheduling](http://arxiv.org/abs/2110.06787)


  Low earth orbit (LEO) satellite-assisted communications have been considered
as one of key elements in beyond 5G systems to provide wide coverage and
cost-efficient data services. Such dynamic space-terrestrial topologies impose
exponential increase in the degrees of freedom in network management. In this
paper, we address two practical issues for an over-loaded LEO-terrestrial
system. The first challenge is how to efficiently schedule resources to serve
the massive number of connected users, such that more data and users can be
delivered/served. The second challenge is how to make the algorithmic solution
more resilient in adapting to dynamic wireless this http URL address them, we
first propose an iterative suboptimal algorithm to provide an offline
benchmark. To adapt to unforeseen variations, we propose an enhanced
meta-critic learning algorithm (EMCL), where a hybrid neural network for
parameterization and the Wolpertinger policy for action mapping are designed in
EMCL. The results demonstrate EMCL's effectiveness and fast-response
capabilities in over-loaded systems and in adapting to dynamic environments
compare to previous actor-critic and meta-learning methods.

    

### [[2110.06802] Identification of Attack-Specific Signatures in Adversarial Examples](http://arxiv.org/abs/2110.06802)


  The adversarial attack literature contains a myriad of algorithms for
crafting perturbations which yield pathological behavior in neural networks. In
many cases, multiple algorithms target the same tasks and even enforce the same
constraints. In this work, we show that different attack algorithms produce
adversarial examples which are distinct not only in their effectiveness but
also in how they qualitatively affect their victims. We begin by demonstrating
that one can determine the attack algorithm that crafted an adversarial
example. Then, we leverage recent advances in parameter-space saliency maps to
show, both visually and quantitatively, that adversarial attack algorithms
differ in which parts of the network and image they target. Our findings
suggest that prospective adversarial attacks should be compared not only via
their success rates at fooling models but also via deeper downstream effects
they have on victims.

    

### [[2110.06816] A Framework for Verification of Wasserstein Adversarial Robustness](http://arxiv.org/abs/2110.06816)


  Machine learning image classifiers are susceptible to adversarial and
corruption perturbations. Adding imperceptible noise to images can lead to
severe misclassifications of the machine learning model. Using $L_p$-norms for
measuring the size of the noise fails to capture human similarity perception,
which is why optimal transport based distance measures like the Wasserstein
metric are increasingly being used in the field of adversarial robustness.
Verifying the robustness of classifiers using the Wasserstein metric can be
achieved by proving the absence of adversarial examples (certification) or
proving their presence (attack). In this work we present a framework based on
the work by Levine and Feizi, which allows us to transfer existing
certification methods for convex polytopes or $L_1$-balls to the Wasserstein
threat model. The resulting certification can be complete or incomplete,
depending on whether convex polytopes or $L_1$-balls were chosen. Additionally,
we present a new Wasserstein adversarial attack that is projected gradient
descent based and which has a significantly reduced computational burden
compared to existing attack approaches.

    

### [[2110.06821] Leveraging redundancy in attention with Reuse Transformers](http://arxiv.org/abs/2110.06821)


  Pairwise dot product-based attention allows Transformers to exchange
information between tokens in an input-dependent way, and is key to their
success across diverse applications in language and vision. However, a typical
Transformer model computes such pairwise attention scores repeatedly for the
same sequence, in multiple heads in multiple layers. We systematically analyze
the empirical similarity of these scores across heads and layers and find them
to be considerably redundant, especially adjacent layers showing high
similarity. Motivated by these findings, we propose a novel architecture that
reuses attention scores computed in one layer in multiple subsequent layers.
Experiments on a number of standard benchmarks show that reusing attention
delivers performance equivalent to or better than standard transformers, while
reducing both compute and memory usage.

    

### [[2110.06827] NoisyActions2M: A Multimedia Dataset for Video Understanding from Noisy Labels](http://arxiv.org/abs/2110.06827)


  Deep learning has shown remarkable progress in a wide range of problems.
However, efficient training of such models requires large-scale datasets, and
getting annotations for such datasets can be challenging and costly. In this
work, we explore the use of user-generated freely available labels from web
videos for video understanding. We create a benchmark dataset consisting of
around 2 million videos with associated user-generated annotations and other
meta information. We utilize the collected dataset for action classification
and demonstrate its usefulness with existing small-scale annotated datasets,
UCF101 and HMDB51. We study different loss functions and two pretraining
strategies, simple and self-supervised learning. We also show how a network
pretrained on the proposed dataset can help against video corruption and label
noise in downstream datasets. We present this as a benchmark dataset in noisy
learning for video understanding. The dataset, code, and trained models will be
publicly available for future research.

    

### [[2110.06829] Towards a fully RL-based Market Simulator](http://arxiv.org/abs/2110.06829)


  We present a new financial framework where two families of RL-based agents
representing the Liquidity Providers and Liquidity Takers learn simultaneously
to satisfy their objective. Thanks to a parametrized reward formulation and the
use of Deep RL, each group learns a shared policy able to generalize and
interpolate over a wide range of behaviors. This is a step towards a fully
RL-based market simulator replicating complex market conditions particularly
suited to study the dynamics of the financial market under various scenarios.

    

### [[2110.06848] Decoupled Contrastive Learning](http://arxiv.org/abs/2110.06848)


  Contrastive learning (CL) is one of the most successful paradigms for
self-supervised learning (SSL). In a principled way, it considers two augmented
``views'' of the same image as positive to be pulled closer, and all other
images negative to be pushed further apart. However, behind the impressive
success of CL-based techniques, their formulation often relies on
heavy-computation settings, including large sample batches, extensive training
epochs, etc. We are thus motivated to tackle these issues and aim at
establishing a simple, efficient, and yet competitive baseline of contrastive
learning. Specifically, we identify, from theoretical and empirical studies, a
noticeable negative-positive-coupling (NPC) effect in the widely used
cross-entropy (InfoNCE) loss, leading to unsuitable learning efficiency with
respect to the batch size. Indeed the phenomenon tends to be neglected in that
optimizing infoNCE loss with a small-size batch is effective in solving easier
SSL tasks. By properly addressing the NPC effect, we reach a decoupled
contrastive learning (DCL) objective function, significantly improving SSL
efficiency. DCL can achieve competitive performance, requiring neither large
batches in SimCLR, momentum encoding in MoCo, or large epochs. We demonstrate
the usefulness of DCL in various benchmarks, while manifesting its robustness
being much less sensitive to suboptimal hyperparameters. Notably, our approach
achieves $66.9\%$ ImageNet top-1 accuracy using batch size 256 within 200
epochs pre-training, outperforming its baseline SimCLR by $5.1\%$. With further
optimized hyperparameters, DCL can improve the accuracy to $68.2\%$. We believe
DCL provides a valuable baseline for future contrastive learning-based SSL
studies.

    

### [[2110.06850] Boosting the Certified Robustness of L-infinity Distance Nets](http://arxiv.org/abs/2110.06850)


  Recently, Zhang et al. (2021) developed a new neural network architecture
based on $\ell_\infty$-distance functions, which naturally possesses certified
robustness by its construction. Despite the excellent theoretical properties,
the model so far can only achieve comparable performance to conventional
networks. In this paper, we significantly boost the certified robustness of
$\ell_\infty$-distance nets through a careful analysis of its training process.
In particular, we show the $\ell_p$-relaxation, a crucial way to overcome the
non-smoothness of the model, leads to an unexpected large Lipschitz constant at
the early training stage. This makes the optimization insufficient using hinge
loss and produces sub-optimal solutions. Given these findings, we propose a
simple approach to address the issues above by using a novel objective function
that combines a scaled cross-entropy loss with clipped hinge loss. Our
experiments show that using the proposed training strategy, the certified
accuracy of $\ell_\infty$-distance net can be dramatically improved from 33.30%
to 40.06% on CIFAR-10 ($\epsilon=8/255$), meanwhile significantly outperforming
other approaches in this area. Such a result clearly demonstrates the
effectiveness and potential of $\ell_\infty$-distance net for certified
robustness.

    

### [[2110.06851] Fast Posterior Estimation of Cardiac Electrophysiological Model Parameters via Bayesian Active Learning](http://arxiv.org/abs/2110.06851)


  Probabilistic estimation of cardiac electrophysiological model parameters
serves an important step towards model personalization and uncertain
quantification. The expensive computation associated with these model
simulations, however, makes direct Markov Chain Monte Carlo (MCMC) sampling of
the posterior probability density function (pdf) of model parameters
computationally intensive. Approximated posterior pdfs resulting from replacing
the simulation model with a computationally efficient surrogate, on the other
hand, have seen limited accuracy. In this paper, we present a Bayesian active
learning method to directly approximate the posterior pdf function of cardiac
model parameters, in which we intelligently select training points to query the
simulation model in order to learn the posterior pdf using a small number of
samples. We integrate a generative model into Bayesian active learning to allow
approximating posterior pdf of high-dimensional model parameters at the
resolution of the cardiac mesh. We further introduce new acquisition functions
to focus the selection of training points on better approximating the shape
rather than the modes of the posterior pdf of interest. We evaluated the
presented method in estimating tissue excitability in a 3D cardiac
electrophysiological model in a range of synthetic and real-data experiments.
We demonstrated its improved accuracy in approximating the posterior pdf
compared to Bayesian active learning using regular acquisition functions, and
substantially reduced computational cost in comparison to existing standard or
accelerated MCMC sampling.

    

### [[2110.06853] Attentive and Contrastive Learning for Joint Depth and Motion Field Estimation](http://arxiv.org/abs/2110.06853)


  Estimating the motion of the camera together with the 3D structure of the
scene from a monocular vision system is a complex task that often relies on the
so-called scene rigidity assumption. When observing a dynamic environment, this
assumption is violated which leads to an ambiguity between the ego-motion of
the camera and the motion of the objects. To solve this problem, we present a
self-supervised learning framework for 3D object motion field estimation from
monocular videos. Our contributions are two-fold. First, we propose a two-stage
projection pipeline to explicitly disentangle the camera ego-motion and the
object motions with dynamics attention module, called DAM. Specifically, we
design an integrated motion model that estimates the motion of the camera and
object in the first and second warping stages, respectively, controlled by the
attention module through a shared motion encoder. Second, we propose an object
motion field estimation through contrastive sample consensus, called CSAC,
taking advantage of weak semantic prior (bounding box from an object detector)
and geometric constraints (each object respects the rigid body motion model).
Experiments on KITTI, Cityscapes, and Waymo Open Dataset demonstrate the
relevance of our approach and show that our method outperforms state-of-the-art
algorithms for the tasks of self-supervised monocular depth estimation, object
motion segmentation, monocular scene flow estimation, and visual odometry.

    

### [[2110.06866] Bayesian logistic regression for online recalibration and revision of risk prediction models with performance guarantees](http://arxiv.org/abs/2110.06866)


  After deploying a clinical prediction model, subsequently collected data can
be used to fine-tune its predictions and adapt to temporal shifts. Because
model updating carries risks of over-updating/fitting, we study online methods
with performance guarantees. We introduce two procedures for continual
recalibration or revision of an underlying prediction model: Bayesian logistic
regression (BLR) and a Markov variant that explicitly models distribution
shifts (MarBLR). We perform empirical evaluation via simulations and a
real-world study predicting COPD risk. We derive "Type I and II" regret bounds,
which guarantee the procedures are non-inferior to a static model and
competitive with an oracle logistic reviser in terms of the average loss. Both
procedures consistently outperformed the static model and other online logistic
revision methods. In simulations, the average estimated calibration index
(aECI) of the original model was 0.828 (95%CI 0.818-0.938). Online
recalibration using BLR and MarBLR improved the aECI, attaining 0.265 (95%CI
0.230-0.300) and 0.241 (95%CI 0.216-0.266), respectively. When performing more
extensive logistic model revisions, BLR and MarBLR increased the average AUC
(aAUC) from 0.767 (95%CI 0.765-0.769) to 0.800 (95%CI 0.798-0.802) and 0.799
(95%CI 0.797-0.801), respectively, in stationary settings and protected against
substantial model decay. In the COPD study, BLR and MarBLR dynamically combined
the original model with a continually-refitted gradient boosted tree to achieve
aAUCs of 0.924 (95%CI 0.913-0.935) and 0.925 (95%CI 0.914-0.935), compared to
the static model's aAUC of 0.904 (95%CI 0.892-0.916). Despite its simplicity,
BLR is highly competitive with MarBLR. MarBLR outperforms BLR when its prior
better reflects the data. BLR and MarBLR can improve the transportability of
clinical prediction models and maintain their performance over time.

    

### [[2110.06871] Two-argument activation functions learn soft XOR operations like cortical neurons](http://arxiv.org/abs/2110.06871)


  Neurons in the brain are complex machines with distinct functional
compartments that interact nonlinearly. In contrast, neurons in artificial
neural networks abstract away this complexity, typically down to a scalar
activation function of a weighted sum of inputs. Here we emulate more
biologically realistic neurons by learning canonical activation functions with
two input arguments, analogous to basal and apical dendrites. We use a
network-in-network architecture where each neuron is modeled as a multilayer
perceptron with two inputs and a single output. This inner perceptron is shared
by all units in the outer network. Remarkably, the resultant nonlinearities
reliably produce soft XOR functions, consistent with recent experimental
observations about interactions between inputs in human cortical neurons. When
hyperparameters are optimized, networks with these nonlinearities learn faster
and perform better than conventional ReLU nonlinearities with matched parameter
counts, and they are more robust to natural and adversarial perturbations.

    

### [[2110.06880] A Survey of Online Auction Mechanism Design Using Deep Learning Approaches](http://arxiv.org/abs/2110.06880)


  Online auction has been very widespread in the recent years. Platform
administrators are working hard to refine their auction mechanisms that will
generate high profits while maintaining a fair resource allocation. With the
advancement of computing technology and the bottleneck in theoretical
frameworks, researchers are shifting gears towards online auction designs using
deep learning approaches. In this article, we summarized some common deep
learning infrastructures adopted in auction mechanism designs and showed how
these architectures are evolving. We also discussed how researchers are
tackling with the constraints and concerns in the large and dynamic industrial
settings. Finally, we pointed out several currently unresolved issues for
future directions.

    

### [[2110.06890] Extending Environments To Measure Self-Reflection In Reinforcement Learning](http://arxiv.org/abs/2110.06890)


  We consider an extended notion of reinforcement learning in which the
environment can simulate the agent and base its outputs on the agent's
hypothetical behavior. Since good performance usually requires paying attention
to whatever things the environment's outputs are based on, we argue that for an
agent to achieve on-average good performance across many such extended
environments, it is necessary for the agent to self-reflect. Thus, an agent's
self-reflection ability can be numerically estimated by running the agent
through a battery of extended environments. We are simultaneously releasing an
open-source library of extended environments to serve as proof-of-concept of
this technique. As the library is first-of-kind, we have avoided the difficult
problem of optimizing it. Instead we have chosen environments with interesting
properties. Some seem paradoxical, some lead to interesting thought
experiments, some are even suggestive of how self-reflection might have evolved
in nature. We give examples and introduce a simple transformation which
experimentally seems to increase self-reflection.

    

### [[2110.06892] TAG: Toward Accurate Social Media Content Tagging with a Concept Graph](http://arxiv.org/abs/2110.06892)


  Although conceptualization has been widely studied in semantics and knowledge
representation, it is still challenging to find the most accurate concept
phrases to characterize the main idea of a text snippet on the fast-growing
social media. This is partly attributed to the fact that most knowledge bases
contain general terms of the world, such as trees and cars, which do not have
the defining power or are not interesting enough to social media app users.
Another reason is that the intricacy of natural language allows the use of
tense, negation and grammar to change the logic or emphasis of language, thus
conveying completely different meanings. In this paper, we present TAG, a
high-quality concept matching dataset consisting of 10,000 labeled pairs of
fine-grained concepts and web-styled natural language sentences, mined from the
open-domain social media. The concepts we consider represent the trending
interests of online users. Associated with TAG is a concept graph of these
fine-grained concepts and entities to provide the structural context
information. We evaluate a wide range of popular neural text matching models as
well as pre-trained language models on TAG, and point out their insufficiency
to tag social media content with the most appropriate concept. We further
propose a novel graph-graph matching method that demonstrates superior
abstraction and generalization performance by better utilizing both the
structural context in the concept graph and logic interactions between semantic
units in the sentence via syntactic dependency parsing. We open-source both the
TAG dataset and the proposed methods to facilitate further research.

    

### [[2110.06893] Newer is not always better: Rethinking transferability metrics, their peculiarities, stability and performance](http://arxiv.org/abs/2110.06893)


  Fine-tuning of large pre-trained image and language models on small
customized datasets has become increasingly popular for improved prediction and
efficient use of limited resources. Fine-tuning requires identification of best
models to transfer-learn from and quantifying transferability prevents
expensive re-training on all of the candidate models/tasks pairs. We show that
the statistical problems with covariance estimation drive the poor performance
of H-score [Bao et al., 2019] -- a common baseline for newer metrics -- and
propose shrinkage-based estimator. This results in up to 80% absolute gain in
H-score correlation performance, making it competitive with the
state-of-the-art LogME measure by You et al. [2021]. Our shrinkage-based
H-score is 3-55 times faster to compute compared to LogME. Additionally, we
look into a less common setting of target (as opposed to source) task
selection. We identify previously overlooked problems in such settings with
different number of labels, class-imbalance ratios etc. for some recent metrics
e.g., LEEP [Nguyen et al., 2020] that resulted in them being misrepresented as
leading measures. We propose a correction and recommend measuring correlation
performance against relative accuracy in such settings. We also outline the
difficulties of comparing feature-dependent metrics, both supervised (e.g.
H-score) and unsupervised measures (e.g., Maximum Mean Discrepancy [Long et
al., 2015]), across source models/layers with different feature embedding
dimension. We show that dimensionality reduction methods allow for meaningful
comparison across models and improved performance of some of these measures. We
investigate performance of 14 different supervised and unsupervised metrics and
demonstrate that even unsupervised metrics can identify the leading models for
domain adaptation. We support our findings with ~65,000 (fine-tuning trials)
experiments.

    

### [[2110.06897] Machine Learning For Elliptic PDEs: Fast Rate Generalization Bound, Neural Scaling Law and Minimax Optimality](http://arxiv.org/abs/2110.06897)


  In this paper, we study the statistical limits of deep learning techniques
for solving elliptic partial differential equations (PDEs) from random samples
using the Deep Ritz Method (DRM) and Physics-Informed Neural Networks (PINNs).
To simplify the problem, we focus on a prototype elliptic PDE: the
Schrödinger equation on a hypercube with zero Dirichlet boundary condition,
which has wide application in the quantum-mechanical systems. We establish
upper and lower bounds for both methods, which improves upon concurrently
developed upper bounds for this problem via a fast rate generalization bound.
We discover that the current Deep Ritz Methods is sub-optimal and propose a
modified version of it. We also prove that PINN and the modified version of DRM
can achieve minimax optimal bounds over Sobolev spaces. Empirically, following
recent work which has shown that the deep model accuracy will improve with
growing training sets according to a power law, we supply computational
experiments to show a similar behavior of dimension dependent power law for
deep PDE solvers.

    

### [[2110.06906] PER-ETD: A Polynomially Efficient Emphatic Temporal Difference Learning Method](http://arxiv.org/abs/2110.06906)


  Emphatic temporal difference (ETD) learning (Sutton et al., 2016) is a
successful method to conduct the off-policy value function evaluation with
function approximation. Although ETD has been shown to converge asymptotically
to a desirable value function, it is well-known that ETD often encounters a
large variance so that its sample complexity can increase exponentially fast
with the number of iterations. In this work, we propose a new ETD method,
called PER-ETD (i.e., PEriodically Restarted-ETD), which restarts and updates
the follow-on trace only for a finite period for each iteration of the
evaluation parameter. Further, PER-ETD features a design of the logarithmical
increase of the restart period with the number of iterations, which guarantees
the best trade-off between the variance and bias and keeps both vanishing
sublinearly. We show that PER-ETD converges to the same desirable fixed point
as ETD, but improves the exponential sample complexity of ETD to be
polynomials. Our experiments validate the superior performance of PER-ETD and
its advantage over ETD.

    

### [[2110.06909] Reinforcement Learning for Standards Design](http://arxiv.org/abs/2110.06909)


  Communications standards are designed via committees of humans holding
repeated meetings over months or even years until consensus is achieved. This
includes decisions regarding the modulation and coding schemes to be supported
over an air interface. We propose a way to "automate" the selection of the set
of modulation and coding schemes to be supported over a given air interface and
thereby streamline both the standards design process and the ease of extending
the standard to support new modulation schemes applicable to new higher-level
applications and services. Our scheme involves machine learning, whereby a
constructor entity submits proposals to an evaluator entity, which returns a
score for the proposal. The constructor employs reinforcement learning to
iterate on its submitted proposals until a score is achieved that was
previously agreed upon by both constructor and evaluator to be indicative of
satisfying the required design criteria (including performance metrics for
transmissions over the interface).

    

### [[2110.06910] On the Double Descent of Random Features Models Trained with SGD](http://arxiv.org/abs/2110.06910)


  We study generalization properties of random features (RF) regression in high
dimensions optimized by stochastic gradient descent (SGD). In this regime, we
derive precise non-asymptotic error bounds of RF regression under both constant
and adaptive step-size SGD setting, and observe the double descent phenomenon
both theoretically and empirically. Our analysis shows how to cope with
multiple randomness sources of initialization, label noise, and data sampling
(as well as stochastic gradients) with no closed-form solution, and also goes
beyond the commonly-used Gaussian/spherical data assumption. Our theoretical
results demonstrate that, with SGD training, RF regression still generalizes
well for interpolation learning, and is able to characterize the double descent
behavior by the unimodality of variance and monotonic decrease of bias.
Besides, we also prove that the constant step-size SGD setting incurs no loss
in convergence rate when compared to the exact minimal-norm interpolator, as a
theoretical justification of using SGD in practice.

    

### [[2110.06912] OPEn: An Open-ended Physics Environment for Learning Without a Task](http://arxiv.org/abs/2110.06912)


  Humans have mental models that allow them to plan, experiment, and reason in
the physical world. How should an intelligent agent go about learning such
models? In this paper, we will study if models of the world learned in an
open-ended physics environment, without any specific tasks, can be reused for
downstream physics reasoning tasks. To this end, we build a benchmark
Open-ended Physics ENvironment (OPEn) and also design several tasks to test
learning representations in this environment explicitly. This setting reflects
the conditions in which real agents (i.e. rolling robots) find themselves,
where they may be placed in a new kind of environment and must adapt without
any teacher to tell them how this environment works. This setting is
challenging because it requires solving an exploration problem in addition to a
model building and representation learning problem. We test several existing
RL-based exploration methods on this benchmark and find that an agent using
unsupervised contrastive learning for representation learning, and
impact-driven learning for exploration, achieved the best results. However, all
models still fall short in sample efficiency when transferring to the
downstream tasks. We expect that OPEn will encourage the development of novel
rolling robot agents that can build reusable mental models of the world that
facilitate many tasks.

    

### [[2110.06914] What Happens after SGD Reaches Zero Loss? --A Mathematical Framework](http://arxiv.org/abs/2110.06914)


  Understanding the implicit bias of Stochastic Gradient Descent (SGD) is one
of the key challenges in deep learning, especially for overparametrized models,
where the local minimizers of the loss function $L$ can form a manifold.
Intuitively, with a sufficiently small learning rate $\eta$, SGD tracks
Gradient Descent (GD) until it gets close to such manifold, where the gradient
noise prevents further convergence. In such a regime, Blanc et al. (2020)
proved that SGD with label noise locally decreases a regularizer-like term, the
sharpness of loss, $\mathrm{tr}[\nabla^2 L]$. The current paper gives a general
framework for such analysis by adapting ideas from Katzenberger (1991). It
allows in principle a complete characterization for the regularization effect
of SGD around such manifold -- i.e., the "implicit bias" -- using a stochastic
differential equation (SDE) describing the limiting dynamics of the parameters,
which is determined jointly by the loss function and the noise covariance. This
yields some new results: (1) a global analysis of the implicit bias valid for
$\eta^{-2}$ steps, in contrast to the local analysis of Blanc et al. (2020)
that is only valid for $\eta^{-1.6}$ steps and (2) allowing arbitrary noise
covariance. As an application, we show with arbitrary large initialization,
label noise SGD can always escape the kernel regime and only requires
$O(\kappa\ln d)$ samples for learning an $\kappa$-sparse overparametrized
linear model in $\mathbb{R}^d$ (Woodworth et al., 2020), while GD initialized
in the kernel regime requires $\Omega(d)$ samples. This upper bound is minimax
optimal and improves the previous $\tilde{O}(\kappa^2)$ upper bound (HaoChen et
al., 2020).

    

### [[2110.06917] Extracting Dynamical Models from Data](http://arxiv.org/abs/2110.06917)


  The FJet approach is introduced for determining the underlying model of a
dynamical system. It borrows ideas from the fields of Lie symmetries as applied
to differential equations (DEs), and numerical integration (such as
Runge-Kutta). The technique can be considered as a way to use machine learning
(ML) to derive a numerical integration scheme. The technique naturally
overcomes the "extrapolation problem", which is when ML is used to extrapolate
a model beyond the time range of the original training data. It does this by
doing the modeling in the phase space of the system, rather than over the time
domain. When modeled with a type of regression scheme, it's possible to
accurately determine the underlying DE, along with parameter dependencies.
Ideas from the field of Lie symmetries applied to ordinary DEs are used to
determine constants of motion, even for damped and driven systems. These
statements are demonstrated on three examples: a damped harmonic oscillator, a
damped pendulum, and a damped, driven nonlinear oscillator (Duffing
oscillator). In the model for the Duffing oscillator, it's possible to treat
the external force in a manner reminiscent of a Green's function approach.
Also, in the case of the undamped harmonic oscillator, the FJet approach
remains stable approximately $10^9$ times longer than $4$th-order Runge-Kutta.

    

### [[2110.06918] Salient Phrase Aware Dense Retrieval: Can a Dense Retriever Imitate a Sparse One?](http://arxiv.org/abs/2110.06918)


  Despite their recent popularity and well known advantages, dense retrievers
still lag behind sparse methods such as BM25 in their ability to reliably match
salient phrases and rare entities in the query. It has been argued that this is
an inherent limitation of dense models. We disprove this claim by introducing
the Salient Phrase Aware Retriever (SPAR), a dense retriever with the lexical
matching capacity of a sparse model. In particular, we show that a dense
retriever {\Lambda} can be trained to imitate a sparse one, and SPAR is built
by augmenting a standard dense retriever with {\Lambda}. When evaluated on five
open-domain question answering datasets and the MS MARCO passage retrieval
task, SPAR sets a new state of the art for dense and sparse retrievers and can
match or exceed the performance of more complicated dense-sparse hybrid
systems.

    

### [[2110.06922] DETR3D: 3D Object Detection from Multi-view Images via 3D-to-2D Queries](http://arxiv.org/abs/2110.06922)


  We introduce a framework for multi-camera 3D object detection. In contrast to
existing works, which estimate 3D bounding boxes directly from monocular images
or use depth prediction networks to generate input for 3D object detection from
2D information, our method manipulates predictions directly in 3D space. Our
architecture extracts 2D features from multiple camera images and then uses a
sparse set of 3D object queries to index into these 2D features, linking 3D
positions to multi-view images using camera transformation matrices. Finally,
our model makes a bounding box prediction per object query, using a set-to-set
loss to measure the discrepancy between the ground-truth and the prediction.
This top-down approach outperforms its bottom-up counterpart in which object
bounding box prediction follows per-pixel depth estimation, since it does not
suffer from the compounding error introduced by a depth prediction model.
Moreover, our method does not require post-processing such as non-maximum
suppression, dramatically improving inference speed. We achieve
state-of-the-art performance on the nuScenes autonomous driving benchmark.

    

### [[2110.06923] Object DGCNN: 3D Object Detection using Dynamic Graphs](http://arxiv.org/abs/2110.06923)


  3D object detection often involves complicated training and testing
pipelines, which require substantial domain knowledge about individual
datasets. Inspired by recent non-maximum suppression-free 2D object detection
models, we propose a 3D object detection architecture on point clouds. Our
method models 3D object detection as message passing on a dynamic graph,
generalizing the DGCNN framework to predict a set of objects. In our
construction, we remove the necessity of post-processing via object confidence
aggregation or non-maximum suppression. To facilitate object detection from
sparse point clouds, we also propose a set-to-set distillation approach
customized to 3D detection. This approach aligns the outputs of the teacher
model and the student model in a permutation-invariant fashion, significantly
simplifying knowledge distillation for the 3D detection task. Our method
achieves state-of-the-art performance on autonomous driving benchmarks. We also
provide abundant analysis of the detection model and distillation framework.

    

### [[1904.11532] Decision Forest: A Nonparametric Approach to Modeling Irrational Choice](http://arxiv.org/abs/1904.11532)


  Customer behavior is often assumed to follow weak rationality, which implies
that adding a product to an assortment will not increase the choice probability
of another product in that assortment. However, an increasing amount of
research has revealed that customers are not necessarily rational when making
decisions. In this paper, we propose a new nonparametric choice model that
relaxes this assumption and can model a wider range of customer behavior, such
as decoy effects between products. In this model, each customer type is
associated with a binary decision tree, which represents a decision process for
making a purchase based on checking for the existence of specific products in
the assortment. Together with a probability distribution over customer types,
we show that the resulting model -- a decision forest -- is able to represent
any customer choice model, including models that are inconsistent with weak
rationality. We theoretically characterize the depth of the forest needed to
fit a data set of historical assortments and prove that with high probability,
a forest whose depth scales logarithmically in the number of assortments is
sufficient to fit most data sets. We also propose two practical algorithms --
one based on column generation and one based on random sampling -- for
estimating such models from data. Using synthetic data and real transaction
data exhibiting non-rational behavior, we show that the model outperforms both
rational and non-rational benchmark models in out-of-sample predictive ability.

    

### [[1904.11883] Robust Graph Data Learning via Latent Graph Convolutional Representation](http://arxiv.org/abs/1904.11883)


  Graph Convolutional Representation (GCR) has achieved impressive performance
for graph data representation. However, existing GCR is generally defined on
the input fixed graph which may restrict the representation capacity and also
be vulnerable to the structural attacks and noises. To address this issue, we
propose a novel Latent Graph Convolutional Representation (LatGCR) for robust
graph data representation and learning. Our LatGCR is derived based on
reformulating graph convolutional representation from the aspect of graph
neighborhood reconstruction. Given an input graph $\textbf{A}$, LatGCR aims to
generate a flexible latent graph $\widetilde{\textbf{A}}$ for graph
convolutional representation which obviously enhances the representation
capacity and also performs robustly w.r.t graph structural attacks and noises.
Moreover, LatGCR is implemented in a self-supervised manner and thus provides a
basic block for both supervised and unsupervised graph learning tasks.
Experiments on several datasets demonstrate the effectiveness and robustness of
LatGCR.

    

### [[1909.09505] Redirection Controller Using Reinforcement Learning](http://arxiv.org/abs/1909.09505)


  There is a growing demand for redirected walking (RDW) techniques and their
application. To apply appropriate RDW methods and manipulation, the RDW
controllers are predominantly used. There are three types of RDW controllers:
direct scripted controller, generalized controller, and predictive controller.
The scripted controller type pre-scripts the mapping between the real and
virtual environments. The generalized controller type employs the RDW method
and manipulation quantities according to a certain procedure depending on the
user's position in relation to the real space. This approach has the potential
to be reused in any environment; however, it is not fully optimized. The
predictive controller type predicts the user's future path using the user's
behavior and manages RDW techniques. This approach is highly anticipated to be
very effective and versatile; however, it has not been sufficiently developed.
This paper proposes a novel RDW controller using reinforcement learning (RL)
with advanced plannability/versatility. Our simulation experiments indicate
that the proposed method can reduce the number of reset manipulations, which is
one of the indicators of the effectiveness of the RDW controller, compared to
the generalized controller under real environments with many obstacles.
Meanwhile, the experimental results also showed that the gain output by the
proposed method oscillates. The results of a user study conducted showed that
the proposed RDW controller can reduce the number of resets compared to the
conventional generalized controller. Furthermore, no adverse effects such as
cybersickness associated with the oscillation of the output gain were evinced.
The simulation and user studies demonstrate that the proposed RDW controller
with RL outperforms the existing generalized controllers and can be applied to
users.

    

### [[1910.04091] Learning with minibatch Wasserstein : asymptotic and gradient properties](http://arxiv.org/abs/1910.04091)


  Optimal transport distances are powerful tools to compare probability
distributions and have found many applications in machine learning. Yet their
algorithmic complexity prevents their direct use on large scale datasets. To
overcome this challenge, practitioners compute these distances on minibatches
{\em i.e.} they average the outcome of several smaller optimal transport
problems. We propose in this paper an analysis of this practice, which effects
are not well understood so far. We notably argue that it is equivalent to an
implicit regularization of the original problem, with appealing properties such
as unbiased estimators, gradients and a concentration bound around the
expectation, but also with defects such as loss of distance property. Along
with this theoretical analysis, we also conduct empirical experiments on
gradient flows, GANs or color transfer that highlight the practical interest of
this strategy.

    

### [[1910.12802] Model-Free Mean-Field Reinforcement Learning: Mean-Field MDP and Mean-Field Q-Learning](http://arxiv.org/abs/1910.12802)


  We study infinite horizon discounted Mean Field Control (MFC) problems with
common noise through the lens of Mean Field Markov Decision Processes (MFMDP).
We allow the agents to use actions that are randomized not only at the
individual level but also at the level of the population. This common
randomization allows us to establish connections between both closed-loop and
open-loop policies for MFC and Markov policies for the MFMDP. In particular, we
show that there exists an optimal closed-loop policy for the original MFC.
Building on this framework and the notion of state-action value function, we
then propose reinforcement learning (RL) methods for such problems, by adapting
existing tabular and deep RL methods to the mean-field setting. The main
difficulty is the treatment of the population state, which is an input of the
policy and the value function. We provide convergence guarantees for tabular
algorithms based on discretizations of the simplex. Neural network based
algorithms are more suitable for continuous spaces and allow us to avoid
discretizing the mean field state space. Numerical examples are provided.

    

### [[2001.07417] Explaining Data-Driven Decisions made by AI Systems: The Counterfactual Approach](http://arxiv.org/abs/2001.07417)


  We examine counterfactual explanations for explaining the decisions made by
model-based AI systems. The counterfactual approach we consider defines an
explanation as a set of the system's data inputs that causally drives the
decision (i.e., changing the inputs in the set changes the decision) and is
irreducible (i.e., changing any subset of the inputs does not change the
decision). We (1) demonstrate how this framework may be used to provide
explanations for decisions made by general, data-driven AI systems that may
incorporate features with arbitrary data types and multiple predictive models,
and (2) propose a heuristic procedure to find the most useful explanations
depending on the context. We then contrast counterfactual explanations with
methods that explain model predictions by weighting features according to their
importance (e.g., SHAP, LIME) and present two fundamental reasons why we should
carefully consider whether importance-weight explanations are well-suited to
explain system decisions. Specifically, we show that (i) features that have a
large importance weight for a model prediction may not affect the corresponding
decision, and (ii) importance weights are insufficient to communicate whether
and how features influence decisions. We demonstrate this with several concise
examples and three detailed case studies that compare the counterfactual
approach with SHAP to illustrate various conditions under which counterfactual
explanations explain data-driven decisions better than importance weights.

    

### [[2002.11650] Contextual Search in the Presence of Irrational Agents](http://arxiv.org/abs/2002.11650)


  We study contextual search, a generalization of binary search in higher
dimensions, which captures settings such as feature-based dynamic pricing.
Standard game-theoretic formulations of this problem assume that agents act in
accordance with a specific behavioral model. In practice, however, some agents
may not subscribe to the dominant behavioral model or may act in ways that seem
to be arbitrarily irrational. Existing algorithms heavily depend on the
behavioral model being (approximately) accurate for all agents and have poor
performance in the presence of even a few such arbitrarily irrational agents.
We initiate the study of contextual search when some of the agents can behave
in ways inconsistent with the underlying behavioral model. In particular, we
provide two algorithms, one based on multidimensional binary search methods and
one based on gradient descent. We show that these algorithms attain
near-optimal regret guarantees in the absence of irrational agents and their
performance degrades gracefully with the number of such agents, providing the
first results for contextual search in any adversarial noise model. Our
techniques draw inspiration from learning theory, game theory, high-dimensional
geometry, and convex analysis.

    

### [[2003.05554] Linear-time inference for Gaussian Processes on one dimension](http://arxiv.org/abs/2003.05554)


  Gaussian Processes (GPs) provide powerful probabilistic frameworks for
interpolation, forecasting, and smoothing, but have been hampered by
computational scaling issues. Here we investigate data sampled on one dimension
(e.g., a scalar or vector time series sampled at arbitrarily-spaced intervals),
for which state-space models are popular due to their linearly-scaling
computational costs. It has long been conjectured that state-space models are
general, able to approximate any one-dimensional GP. We provide the first
general proof of this conjecture, showing that any stationary GP on one
dimension with vector-valued observations governed by a Lebesgue-integrable
continuous kernel can be approximated to any desired precision using a
specifically-chosen state-space model: the Latent Exponentially Generated (LEG)
family. This new family offers several advantages compared to the general
state-space model: it is always stable (no unbounded growth), the covariance
can be computed in closed form, and its parameter space is unconstrained
(allowing straightforward estimation via gradient descent). The theorem's proof
also draws connections to Spectral Mixture Kernels, providing insight about
this popular family of kernels. We develop parallelized algorithms for
performing inference and learning in the LEG model, test the algorithm on real
and synthetic data, and demonstrate scaling to datasets with billions of
samples.

    

### [[2004.10356] Quantifying With Only Positive Training Data](http://arxiv.org/abs/2004.10356)


  Quantification is the research field that studies methods for counting the
number of data points that belong to each class in an unlabeled sample.
Traditionally, researchers in this field assume the availability of labelled
observations for all classes to induce a quantification model. However, we
often face situations where the number of classes is large or even unknown, or
we have reliable data for a single class. When inducing a multi-class
quantifier is infeasible, we are often concerned with estimates for a specific
class of interest. In this context, we have proposed a novel setting known as
One-class Quantification (OCQ). In contrast, Positive and Unlabeled Learning
(PUL), another branch of Machine Learning, has offered solutions to OCQ,
despite quantification not being the focal point of PUL. This article closes
the gap between PUL and OCQ and brings both areas together under a unified
view. We compare our method, Passive Aggressive Threshold (PAT), against PUL
methods and show that PAT generally is the fastest and most accurate algorithm.
PAT induces quantification models that can be reused to quantify different
samples of data. We additionally introduce Exhaustive TIcE (ExTIcE), an
improved version of the PUL algorithm Tree Induction for c Estimation (TIcE).
We show that ExTIcE quantifies more accurately than PAT and the other assessed
algorithms in scenarios where several negative observations are identical to
the positive ones.

    

### [[2007.05975] A Graph Symmetrisation Bound on Channel Information Leakage under Blowfish Privacy](http://arxiv.org/abs/2007.05975)


  Blowfish privacy is a recent generalisation of differential privacy that
enables improved utility while maintaining privacy policies with semantic
guarantees, a factor that has driven the popularity of differential privacy in
computer science. This paper relates Blowfish privacy to an important measure
of privacy loss of information channels from the communications theory
community: min-entropy leakage. Symmetry in an input data neighbouring relation
is central to known connections between differential privacy and min-entropy
leakage. But while differential privacy exhibits strong symmetry, Blowfish
neighbouring relations correspond to arbitrary simple graphs owing to the
framework's flexible privacy policies. To bound the min-entropy leakage of
Blowfish-private mechanisms we organise our analysis over symmetrical
partitions corresponding to orbits of graph automorphism groups. A construction
meeting our bound with asymptotic equality demonstrates tightness.

    

### [[2010.15963] Deep Jump Learning for Off-Policy Evaluation in Continuous Treatment Settings](http://arxiv.org/abs/2010.15963)


  We consider off-policy evaluation (OPE) in continuous treatment settings,
such as personalized dose-finding. In OPE, one aims to estimate the mean
outcome under a new treatment decision rule using historical data generated by
a different decision rule. Most existing works on OPE focus on discrete
treatment settings. To handle continuous treatments, we develop a novel
estimation method for OPE using deep jump learning. The key ingredient of our
method lies in adaptively discretizing the treatment space using deep
discretization, by leveraging deep learning and multi-scale change point
detection. This allows us to apply existing OPE methods in discrete treatments
to handle continuous treatments. Our method is further justified by theoretical
results, simulations, and a real application to Warfarin Dosing.

    

### [[2011.12720] Omni: Automated Ensemble with Unexpected Models against Adversarial Evasion Attack](http://arxiv.org/abs/2011.12720)


  Background: Machine learning-based security detection models have become
prevalent in modern malware and intrusion detection systems. However, previous
studies show that such models are susceptible to adversarial evasion attacks.
In this type of attack, inputs (i.e., adversarial examples) are specially
crafted by intelligent malicious adversaries, with the aim of being
misclassified by existing state-of-the-art models (e.g., deep neural networks).
Once the attackers can fool a classifier to think that a malicious input is
actually benign, they can render a machine learning-based malware or intrusion
detection system ineffective. Goal: To help security practitioners and
researchers build a more robust model against non-adaptive, white-box, and
non-targeted adversarial evasion attacks through the idea of an ensemble model.
Method: We propose an approach called Omni, the main idea of which is to
explore methods that create an ensemble of "unexpected models"; i.e., models
whose control hyperparameters have a large distance to the hyperparameters of
an adversary's target model, with which we then make an optimized weighted
ensemble prediction. Result: In studies with five types of adversarial evasion
attacks (FGSM, BIM, JSMA, DeepFooland Carlini-Wagner) on five security datasets
(NSL-KDD, CIC-IDS-2017, CSE-CIC-IDS2018, CICAnd-Mal2017, and the Contagio PDF
dataset), we show Omni is a promising approach as a defense strategy against
adversarial attacks when compared with other baseline treatments. Conclusion:
When employing ensemble defense against adversarial evasion attacks, we suggest
creating an ensemble with unexpected models that are distant from the
attacker's expected model (i.e., target model) through methods such as
hyperparameter optimization.

    

### [[2101.03308] A Reconfigurable Convolution-in-Pixel CMOS Image Sensor Architecture](http://arxiv.org/abs/2101.03308)


  The separation of the data capture and analysis in modern vision systems has
led to a massive amount of data transfer between the end devices and cloud
computers, resulting in long latency, slow response, and high power
consumption. Efficient hardware architectures are under focused development to
enable Artificial Intelligence (AI) at the resource-limited end sensing
devices. One of the most promising solutions is to enable Processing-in-Pixel
(PIP) scheme. However, the conventional schemes suffer from the low fill-factor
issue. This paper proposes a PIP based CMOS sensor architecture, which allows
convolution operation before the column readout circuit to significantly
improve the image reading speed with much lower power consumption. The
simulation results show that the proposed architecture could support the
computing efficiency up to 11.65 TOPS/W at the 8-bit weight configuration,
which is three times as high as the conventional schemes. The transistors
required for each pixel are only 2.5T, significantly improving the fill-factor.

    

### [[2101.07140] Specifying and Interpreting Reinforcement Learning Policies through Simulatable Machine Learning](http://arxiv.org/abs/2101.07140)


  Human-AI collaborative policy synthesis is a procedure in which (1) a human
initializes an autonomous agent's behavior, (2) Reinforcement Learning improves
the human specified behavior, and (3) the agent can explain the final optimized
policy to the user. This paradigm leverages human expertise and facilitates a
greater insight into the learned behaviors of an agent. Existing approaches to
enabling collaborative policy specification involve black box methods which are
unintelligible and are not catered towards non-expert end-users. In this paper,
we develop a novel collaborative framework to enable humans to initialize and
interpret an autonomous agent's behavior, rooted in principles of
human-centered design. Through our framework, we enable humans to specify an
initial behavior model in the form of unstructured, natural language, which we
then convert to lexical decision trees. Next, we are able to leverage these
human-specified policies, to warm-start reinforcement learning and further
allow the agent to optimize the policies through reinforcement learning.
Finally, to close the loop on human-specification, we produce explanations of
the final learned policy, in multiple modalities, to provide the user a final
depiction about the learned policy of the agent. We validate our approach by
showing that our model can produce >80% accuracy, and that human-initialized
policies are able to successfully warm-start RL. We then conduct a novel
human-subjects study quantifying the relative subjective and objective benefits
of varying XAI modalities(e.g., Tree, Language, and Program) for explaining
learned policies to end-users, in terms of usability and interpretability and
identify the circumstances that influence these measures. Our findings
emphasize the need for personalized explainable systems that can facilitate
user-centric policy explanations for a variety of end-users.

    

### [[2102.01163] Visual Framing of Science Conspiracy Videos: Integrating Machine Learning with Communication Theories to Study the Use of Color and Brightness](http://arxiv.org/abs/2102.01163)


  Recent years have witnessed an explosion of science conspiracy videos on the
Internet, challenging science epistemology and public understanding of science.
Scholars have started to examine the persuasion techniques used in conspiracy
messages such as uncertainty and fear yet, little is understood about the
visual narratives, especially how visual narratives differ in videos that
debunk conspiracies versus those that propagate conspiracies. This paper
addresses this gap in understanding visual framing in conspiracy videos through
analyzing millions of frames from conspiracy and counter-conspiracy YouTube
videos using computational methods. We found that conspiracy videos tended to
use lower color variance and brightness, especially in thumbnails and earlier
parts of the videos. This paper also demonstrates how researchers can integrate
textual and visual features in machine learning models to study conspiracies on
social media and discusses the implications of computational modeling for
scholars interested in studying visual manipulation in the digital era. The
analysis of visual and textual features presented in this paper could be useful
for future studies focused on designing systems to identify conspiracy content
on the Internet.

    

### [[2102.09583] Encoding Frequency Constraints in Preventive Unit Commitment Using Deep Learning with Region-of-Interest Active Sampling](http://arxiv.org/abs/2102.09583)


  With the increasing penetration of renewable energy, frequency response and
its security are of significant concerns for reliable power system operations.
Frequency-constrained unit commitment (FCUC) is proposed to address this
challenge. Despite existing efforts in modeling frequency characteristics in
unit commitment (UC), current strategies can only handle oversimplified
low-order frequency response models and do not consider wide-range operating
conditions. This paper presents a generic data-driven framework for FCUC under
high renewable penetration. Deep neural networks (DNNs) are trained to predict
the frequency response using real data or high-fidelity simulation data. Next,
the DNN is reformulated as a set of mixed-integer linear constraints to be
incorporated into the ordinary UC formulation. In the data generation phase,
all possible power injections are considered, and a region-of-interests active
sampling is proposed to include power injection samples with frequency nadirs
closer to the UFLC threshold, which significantly enhances the accuracy of
frequency constraints in FCUC. The proposed FCUC is verified on the the IEEE
39-bus system. Then, a full-order dynamic model simulation using PSS/E verifies
the effectiveness of FCUC in frequency-secure generator commitments.

    

### [[2102.10226] ALMA: Alternating Minimization Algorithm for Clustering Mixture Multilayer Network](http://arxiv.org/abs/2102.10226)


  The paper considers a Mixture Multilayer Stochastic Block Model (MMLSBM),
where layers can be partitioned into groups of similar networks, and networks
in each group are equipped with a distinct Stochastic Block Model. The goal is
to partition the multilayer network into clusters of similar layers, and to
identify communities in those layers. Jing et al. (2020) introduced the MMLSBM
and developed a clustering methodology, TWIST, based on regularized tensor
decomposition. The present paper proposes a different technique, an alternating
minimization algorithm (ALMA), that aims at simultaneous recovery of the layer
partition, together with estimation of the matrices of connection probabilities
of the distinct layers. Compared to TWIST, ALMA achieves higher accuracy both
theoretically and numerically.

    

### [[2103.03571] Cycle Self-Training for Domain Adaptation](http://arxiv.org/abs/2103.03571)


  Mainstream approaches for unsupervised domain adaptation (UDA) learn
domain-invariant representations to narrow the domain shift. Recently,
self-training has been gaining momentum in UDA, which exploits unlabeled target
data by training with target pseudo-labels. However, as corroborated in this
work, under distributional shift in UDA, the pseudo-labels can be unreliable in
terms of their large discrepancy from target ground truth. Thereby, we propose
Cycle Self-Training (CST), a principled self-training algorithm that explicitly
enforces pseudo-labels to generalize across domains. CST cycles between a
forward step and a reverse step until convergence. In the forward step, CST
generates target pseudo-labels with a source-trained classifier. In the reverse
step, CST trains a target classifier using target pseudo-labels, and then
updates the shared representations to make the target classifier perform well
on the source data. We introduce the Tsallis entropy as a confidence-friendly
regularization to improve the quality of target pseudo-labels. We analyze CST
theoretically under realistic assumptions, and provide hard cases where CST
recovers target ground truth, while both invariant feature learning and vanilla
self-training fail. Empirical results indicate that CST significantly improves
over the state-of-the-arts on visual recognition and sentiment analysis
benchmarks.

    

### [[2103.03716] Golem: An algorithm for robust experiment and process optimization](http://arxiv.org/abs/2103.03716)


  Numerous challenges in science and engineering can be framed as optimization
tasks, including the maximization of reaction yields, the optimization of
molecular and materials properties, and the fine-tuning of automated hardware
protocols. Design of experiment and optimization algorithms are often adopted
to solve these tasks efficiently. Increasingly, these experiment planning
strategies are coupled with automated hardware to enable autonomous
experimental platforms. The vast majority of the strategies used, however, do
not consider robustness against the variability of experiment and process
conditions. In fact, it is generally assumed that these parameters are exact
and reproducible. Yet some experiments may have considerable noise associated
with some of their conditions, and process parameters optimized under precise
control may be applied in the future under variable operating conditions. In
either scenario, the optimal solutions found might not be robust against input
variability, affecting the reproducibility of results and returning suboptimal
performance in practice. Here, we introduce Golem, an algorithm that is
agnostic to the choice of experiment planning strategy and that enables robust
experiment and process optimization. Golem identifies optimal solutions that
are robust to input uncertainty, thus ensuring the reproducible performance of
optimized experimental protocols and processes. It can be used to analyze the
robustness of past experiments, or to guide experiment planning algorithms
toward robust solutions on the fly. We assess the performance and domain of
applicability of Golem through extensive benchmark studies and demonstrate its
practical relevance by optimizing an analytical chemistry protocol under the
presence of significant noise in its experimental conditions.

    

### [[2103.06376] Functional Collection Programming with Semi-Ring Dictionaries](http://arxiv.org/abs/2103.06376)


  This paper introduces semi-ring dictionaries, a powerful class of
compositional and purely functional collections that subsume other collection
types such as sets, multisets, arrays, vectors, and matrices. We developed
SDQL, a statically typed language that can express relational algebra with
aggregations, linear algebra, and functional collections over data such as
relations and matrices using semi-ring dictionaries. Furthermore, thanks to the
algebraic structure behind these dictionaries, SDQL unifies a wide range of
optimizations commonly used in databases (DB) and linear algebra (LA). As a
result, SDQL enables efficient processing of hybrid DB and LA workloads, by
putting together optimizations that are otherwise confined to either DB systems
or LA frameworks. We show experimentally that a handful of DB and LA workloads
can take advantage of the SDQL language and optimizations. Overall, we observe
that SDQL achieves competitive performance relative to Typer and Tectorwise,
which are state-of-the-art in-memory DB systems for (flat, not nested)
relational data, and achieves an average 2x speedup over SciPy for LA
workloads. For hybrid workloads involving LA processing, SDQL achieves up to
one order of magnitude speedup over Trance, a state-of-the-art nested
relational engine for nested biomedical data, and gives an average 40% speedup
over LMFAO, a state-of-the-art in-DB machine learning engine for two (flat)
relational real-world retail datasets.

    

### [[2103.12692] Benign Overfitting of Constant-Stepsize SGD for Linear Regression](http://arxiv.org/abs/2103.12692)


  There is an increasing realization that algorithmic inductive biases are
central in preventing overfitting; empirically, we often see a benign
overfitting phenomenon in overparameterized settings for natural learning
algorithms, such as stochastic gradient descent (SGD), where little to no
explicit regularization has been employed. This work considers this issue in
arguably the most basic setting: constant-stepsize SGD (with iterate averaging
or tail averaging) for linear regression in the overparameterized regime. Our
main result provides a sharp excess risk bound, stated in terms of the full
eigenspectrum of the data covariance matrix, that reveals a bias-variance
decomposition characterizing when generalization is possible: (i) the variance
bound is characterized in terms of an effective dimension (specific for SGD)
and (ii) the bias bound provides a sharp geometric characterization in terms of
the location of the initial iterate (and how it aligns with the data covariance
matrix). More specifically, for SGD with iterate averaging, we demonstrate the
sharpness of the established excess risk bound by proving a matching lower
bound (up to constant factors). For SGD with tail averaging, we show its
advantage over SGD with iterate averaging by proving a better excess risk bound
together with a nearly matching lower bound. Moreover, we reflect on a number
of notable differences between the algorithmic regularization afforded by
(unregularized) SGD in comparison to ordinary least squares (minimum-norm
interpolation) and ridge regression. Experimental results on synthetic data
corroborate our theoretical findings.

    

### [[2103.15718] von Mises-Fisher Loss: An Exploration of Embedding Geometries for Supervised Learning](http://arxiv.org/abs/2103.15718)


  Recent work has argued that classification losses utilizing softmax
cross-entropy are superior not only for fixed-set classification tasks, but
also by outperforming losses developed specifically for open-set tasks
including few-shot learning and retrieval. Softmax classifiers have been
studied using different embedding geometries -- Euclidean, hyperbolic, and
spherical -- and claims have been made about the superiority of one or another,
but they have not been systematically compared with careful controls. We
conduct an empirical investigation of embedding geometry on softmax losses for
a variety of fixed-set classification and image retrieval tasks. An interesting
property observed for the spherical losses lead us to propose a probabilistic
classifier based on the von Mises-Fisher distribution, and we show that it is
competitive with state-of-the-art methods while producing improved
out-of-the-box calibration. We provide guidance regarding the trade-offs
between losses and how to choose among them.

    

### [[2104.09323] Sequential Deconfounding for Causal Inference with Unobserved Confounders](http://arxiv.org/abs/2104.09323)


  Using observational data to estimate the effect of a treatment is a powerful
tool for decision-making when randomized experiments are infeasible or costly.
However, observational data often yields biased estimates of treatment effects,
since treatment assignment can be confounded by unobserved variables. A remedy
is offered by deconfounding methods that adjust for such unobserved
confounders. In this paper, we develop the Sequential Deconfounder, a method
that enables estimating individualized treatment effects over time in presence
of unobserved confounders. This is the first deconfounding method that can be
used in a general sequential setting (i.e., with one or more treatments
assigned at each timestep). The Sequential Deconfounder uses a novel Gaussian
process latent variable model to infer substitutes for the unobserved
confounders, which are then used in conjunction with an outcome model to
estimate treatment effects over time. We prove that using our method yields
unbiased estimates of individualized treatment responses over time. Using
simulated and real medical data, we demonstrate the efficacy of our method in
deconfounding the estimation of treatment responses over time.

    

### [[2104.11914] EXplainable Neural-Symbolic Learning (X-NeSyL) methodology to fuse deep learning representations with expert knowledge graphs: the MonuMAI cultural heritage use case](http://arxiv.org/abs/2104.11914)


  The latest Deep Learning (DL) models for detection and classification have
achieved an unprecedented performance over classical machine learning
algorithms. However, DL models are black-box methods hard to debug, interpret,
and certify. DL alone cannot provide explanations that can be validated by a
non technical audience. In contrast, symbolic AI systems that convert concepts
into rules or symbols -- such as knowledge graphs -- are easier to explain.
However, they present lower generalisation and scaling capabilities. A very
important challenge is to fuse DL representations with expert knowledge. One
way to address this challenge, as well as the performance-explainability
trade-off is by leveraging the best of both streams without obviating domain
expert knowledge. We tackle such problem by considering the symbolic knowledge
is expressed in form of a domain expert knowledge graph. We present the
eXplainable Neural-symbolic learning (X-NeSyL) methodology, designed to learn
both symbolic and deep representations, together with an explainability metric
to assess the level of alignment of machine and human expert explanations. The
ultimate objective is to fuse DL representations with expert domain knowledge
during the learning process to serve as a sound basis for explainability.
X-NeSyL methodology involves the concrete use of two notions of explanation at
inference and training time respectively: 1) EXPLANet: Expert-aligned
eXplainable Part-based cLAssifier NETwork Architecture, a compositional CNN
that makes use of symbolic representations, and 2) SHAP-Backprop, an
explainable AI-informed training procedure that guides the DL process to align
with such symbolic representations in form of knowledge graphs. We showcase
X-NeSyL methodology using MonuMAI dataset for monument facade image
classification, and demonstrate that our approach improves explainability and
performance.

    

### [[2104.13921] Open-vocabulary Object Detection via Vision and Language Knowledge Distillation](http://arxiv.org/abs/2104.13921)


  We aim at advancing open-vocabulary object detection, which detects objects
described by arbitrary text inputs. The fundamental challenge is the
availability of training data. Existing object detection datasets only contain
hundreds of categories, and it is costly to scale further. To overcome this
challenge, we propose ViLD, a training method via Vision and Language knowledge
Distillation. Our method distills the knowledge from a pretrained
open-vocabulary image classification model (teacher) into a two-stage detector
(student). Specifically, we use the teacher model to encode category texts and
image regions of object proposals. Then we train a student detector, whose
region embeddings of detected boxes are aligned with the text and image
embeddings inferred by the teacher. We benchmark on LVIS by holding out all
rare categories as novel categories not seen during training. ViLD obtains 16.1
mask AP$_r$, even outperforming the supervised counterpart by 3.8 with a
ResNet-50 backbone. The model can directly transfer to other datasets without
finetuning, achieving 72.2 AP$_{50}$, 36.6 AP and 11.8 AP on PASCAL VOC, COCO
and Objects365, respectively. On COCO, ViLD outperforms previous SOTA by 4.8 on
novel AP and 11.4 on overall AP.

    

### [[2105.00108] Explaining a Series of Models by Propagating Shapley Values](http://arxiv.org/abs/2105.00108)


  Local feature attribution methods are increasingly used to explain complex
machine learning models. However, current methods are limited because they are
extremely expensive to compute or are not capable of explaining a distributed
series of models where each model is owned by a separate institution. The
latter is particularly important because it often arises in finance where
explanations are mandated. Here, we present DeepSHAP, a tractable method to
propagate local feature attributions through complex series of models based on
a connection to the Shapley value. We evaluate DeepSHAP across biological,
health, and financial datasets to show that it provides equally salient
explanations an order of magnitude faster than existing model-agnostic
attribution techniques and demonstrate its use in an important distributed
series of models setting.

    

### [[2105.00931] GridToPix: Training Embodied Agents with Minimal Supervision](http://arxiv.org/abs/2105.00931)


  While deep reinforcement learning (RL) promises freedom from hand-labeled
data, great successes, especially for Embodied AI, require significant work to
create supervision via carefully shaped rewards. Indeed, without shaped
rewards, i.e., with only terminal rewards, present-day Embodied AI results
degrade significantly across Embodied AI problems from single-agent
Habitat-based PointGoal Navigation (SPL drops from 55 to 0) and two-agent
AI2-THOR-based Furniture Moving (success drops from 58% to 1%) to three-agent
Google Football-based 3 vs. 1 with Keeper (game score drops from 0.6 to 0.1).
As training from shaped rewards doesn't scale to more realistic tasks, the
community needs to improve the success of training with terminal rewards. For
this we propose GridToPix: 1) train agents with terminal rewards in gridworlds
that generically mirror Embodied AI environments, i.e., they are independent of
the task; 2) distill the learned policy into agents that reside in complex
visual worlds. Despite learning from only terminal rewards with identical
models and RL algorithms, GridToPix significantly improves results across
tasks: from PointGoal Navigation (SPL improves from 0 to 64) and Furniture
Moving (success improves from 1% to 25%) to football gameplay (game score
improves from 0.1 to 0.6). GridToPix even helps to improve the results of
shaped reward training.

    

### [[2105.05991] Improving Code Autocompletion with Transfer Learning](http://arxiv.org/abs/2105.05991)


  Software language models have achieved promising results predicting code
completion usages, and several industry studies have described successful IDE
integrations. Recently, accuracy in autocompletion prediction improved 12.8%
from training on a real-world dataset collected from programmers' IDE activity.
But what if limited examples of IDE autocompletion in the target programming
language are available for model training? In this paper, we investigate the
efficacy of pretraining autocompletion models on non-IDE, non-autocompletion,
and different-language example code sequences. We find that these unsupervised
pretrainings improve model accuracy by over 50% on very small fine-tuning
datasets and over 10% on 50k labeled examples. We confirm the real-world impact
of these pretrainings in an online setting through A/B testing on thousands of
IDE autocompletion users, finding that pretraining is responsible for increases
of up to 6.63% autocompletion usage.

    

### [[2105.09987] Temporal convolutional networks predict dynamic oxygen uptake response from wearable sensors across exercise intensities](http://arxiv.org/abs/2105.09987)


  Oxygen consumption (VO$_2$) provides established clinical and physiological
indicators of cardiorespiratory function and exercise capacity. However, VO$_2$
monitoring is largely limited to specialized laboratory settings, making its
widespread monitoring elusive. Here, we investigate temporal prediction of
VO$_2$ from wearable sensors during cycle ergometer exercise using a temporal
convolutional network (TCN). Cardiorespiratory signals were acquired from a
smart shirt with integrated textile sensors alongside ground-truth VO$_2$ from
a metabolic system on twenty-two young healthy adults. Participants performed
one ramp-incremental and three pseudorandom binary sequence exercise protocols
to assess a range of VO$_2$ dynamics. A TCN model was developed using causal
convolutions across an effective history length to model the time-dependent
nature of VO$_2$. Optimal history length was determined through minimum
validation loss across hyperparameter values. The best performing model encoded
218 s history length (TCN-VO$_2$ A), with 187 s, 97 s, and 76 s yielding less
than 3% deviation from the optimal validation loss. TCN-VO$_2$ A showed strong
prediction accuracy (mean, 95% CI) across all exercise intensities (-22
ml.min$^{-1}$, [-262, 218]), spanning transitions from low-moderate (-23
ml.min$^{-1}$, [-250, 204]), low-high (14 ml.min$^{-1}$, [-252, 280]),
ventilatory threshold-high (-49 ml.min$^{-1}$, [-274, 176]), and maximal (-32
ml.min$^{-1}$, [-261, 197]) exercise. Second-by-second classification of
physical activity across 16090 s of predicted VO$_2$ was able to discern
between vigorous, moderate, and light activity with high accuracy (94.1%). This
system enables quantitative aerobic activity monitoring in non-laboratory
settings across a range of exercise intensities using wearable sensors for
monitoring exercise prescription adherence and personal fitness.

    

### [[2106.02542] Heterogeneous Wasserstein Discrepancy for Incomparable Distributions](http://arxiv.org/abs/2106.02542)


  Optimal Transport (OT) metrics allow for defining discrepancies between two
probability measures. Wasserstein distance is for longer the celebrated
OT-distance frequently-used in the literature, which seeks probability
distributions to be supported on the $\textit{same}$ metric space. Because of
its high computational complexity, several approximate Wasserstein distances
have been proposed based on entropy regularization or on slicing, and
one-dimensional Wassserstein computation. In this paper, we propose a novel
extension of Wasserstein distance to compare two incomparable distributions,
that hinges on the idea of $\textit{distributional slicing}$, embeddings, and
on computing the closed-form Wassertein distance between the sliced
distributions. We provide a theoretical analysis of this new divergence, called
$\textit{heterogeneous Wasserstein discrepancy (HWD)}$, and we show that it
preserves several interesting properties including rotation-invariance. We show
that the embeddings involved in HWD can be efficiently learned. Finally, we
provide a large set of experiments illustrating the behavior of HWD as a
divergence in the context of generative modeling and in query framework.

    

### [[2106.03352] The Power of Exploiter: Provable Multi-Agent RL in Large State Spaces](http://arxiv.org/abs/2106.03352)


  Modern reinforcement learning (RL) commonly engages practical problems with
large state spaces, where function approximation must be deployed to
approximate either the value function or the policy. While recent progresses in
RL theory address a rich set of RL problems with general function
approximation, such successes are mostly restricted to the single-agent
setting. It remains elusive how to extend these results to multi-agent RL,
especially due to the new challenges arising from its game-theoretical nature.
This paper considers two-player zero-sum Markov Games (MGs). We propose a new
algorithm that can provably find the Nash equilibrium policy using a polynomial
number of samples, for any MG with low multi-agent Bellman-Eluder dimension --
a new complexity measure adapted from its single-agent version (Jin et al.,
2021). A key component of our new algorithm is the exploiter, which facilitates
the learning of the main player by deliberately exploiting her weakness. Our
theoretical framework is generic, which applies to a wide range of models
including but not limited to tabular MGs, MGs with linear or kernel function
approximation, and MGs with rich observations.

    

### [[2106.05390] Optimizing Reusable Knowledge for Continual Learning via Metalearning](http://arxiv.org/abs/2106.05390)


  When learning tasks over time, artificial neural networks suffer from a
problem known as Catastrophic Forgetting (CF). This happens when the weights of
a network are overwritten during the training of a new task causing forgetting
of old information. To address this issue, we propose MetA Reusable Knowledge
or MARK, a new method that fosters weight reusability instead of overwriting
when learning a new task. Specifically, MARK keeps a set of shared weights
among tasks. We envision these shared weights as a common Knowledge Base (KB)
that is not only used to learn new tasks, but also enriched with new knowledge
as the model learns new tasks. Key components behind MARK are two-fold. On the
one hand, a metalearning approach provides the key mechanism to incrementally
enrich the KB with new knowledge and to foster weight reusability among tasks.
On the other hand, a set of trainable masks provides the key mechanism to
selectively choose from the KB relevant weights to solve each task. By using
MARK, we achieve state of the art results in several popular benchmarks,
surpassing the best performing methods in terms of average accuracy by over 10%
on the 20-Split-MiniImageNet dataset, while achieving almost zero forgetfulness
using 55% of the number of parameters. Furthermore, an ablation study provides
evidence that, indeed, MARK is learning reusable knowledge that is selectively
used by each task.

    

### [[2106.06770] What can linearized neural networks actually say about generalization?](http://arxiv.org/abs/2106.06770)


  For certain infinitely-wide neural networks, the neural tangent kernel (NTK)
theory fully characterizes generalization, but for the networks used in
practice, the empirical NTK only provides a rough first-order approximation.
Still, a growing body of work keeps leveraging this approximation to
successfully analyze important deep learning phenomena and design algorithms
for new applications. In our work, we provide strong empirical evidence to
determine the practical validity of such approximation by conducting a
systematic comparison of the behavior of different neural networks and their
linear approximations on different tasks. We show that the linear
approximations can indeed rank the learning complexity of certain tasks for
neural networks, even when they achieve very different performances. However,
in contrast to what was previously reported, we discover that neural networks
do not always perform better than their kernel approximations, and reveal that
the performance gap heavily depends on architecture, dataset size and training
task. We discover that networks overfit to these tasks mostly due to the
evolution of their kernel during training, thus, revealing a new type of
implicit bias.

    

### [[2106.06946] Boosting Randomized Smoothing with Variance Reduced Classifiers](http://arxiv.org/abs/2106.06946)


  Randomized Smoothing (RS) is a promising method for obtaining robustness
certificates by evaluating a base model under noise. In this work, we: (i)
theoretically motivate why ensembles are a particularly suitable choice as base
models for RS, and (ii) empirically confirm this choice, obtaining
state-of-the-art results in multiple settings. The key insight of our work is
that the reduced variance of ensembles over the perturbations introduced in RS
leads to significantly more consistent classifications for a given input. This,
in turn, leads to substantially increased certifiable radii for samples close
to the decision boundary. Additionally, we introduce key optimizations which
enable an up to 55-fold decrease in sample complexity of RS, thus drastically
reducing its computational overhead. Experimentally, we show that ensembles of
only 3 to 10 classifiers consistently improve on their strongest constituting
model with respect to their average certified radius (ACR) by 5% to 21% on both
CIFAR10 and ImageNet, achieving a new state-of-the-art ACR of 0.86 and 1.11,
respectively. We release all code and models required to reproduce our results
upon publication.

    

### [[2106.10234] Dual-view Molecule Pre-training](http://arxiv.org/abs/2106.10234)


  Inspired by its success in natural language processing and computer vision,
pre-training has attracted substantial attention in cheminformatics and
bioinformatics, especially for molecule based tasks. A molecule can be
represented by either a graph (where atoms are connected by bonds) or a SMILES
sequence (where depth-first-search is applied to the molecular graph with
specific rules). Existing works on molecule pre-training use either graph
representations only or SMILES representations only. In this work, we propose
to leverage both the representations and design a new pre-training algorithm,
dual-view molecule pre-training (briefly, DMP), that can effectively combine
the strengths of both types of molecule representations. The model of DMP
consists of two branches: a Transformer branch that takes the SMILES sequence
of a molecule as input, and a GNN branch that takes a molecular graph as input.
The training of DMP contains three tasks: (1) predicting masked tokens in a
SMILES sequence by the Transformer branch, (2) predicting masked atoms in a
molecular graph by the GNN branch, and (3) maximizing the consistency between
the two high-level representations output by the Transformer and GNN branches
separately. After pre-training, we can use either the Transformer branch (this
one is recommended according to empirical results), the GNN branch, or both for
downstream tasks. DMP is tested on nine molecular property prediction tasks and
achieves state-of-the-art performances on seven of them. Furthermore, we test
DMP on three retrosynthesis tasks and achieve state-of-the-art results on them.

    

### [[2106.11299] Boundary Graph Neural Networks for 3D Simulations](http://arxiv.org/abs/2106.11299)


  The abundance of data has given machine learning considerable momentum in
natural sciences and engineering. However, the modeling of simulated physical
processes remains difficult. A key problem is the correct handling of geometric
boundaries. While triangularized geometric boundaries are very common in
engineering applications, they are notoriously difficult to model by machine
learning approaches due to their heterogeneity with respect to size and
orientation. In this work, we introduce Boundary Graph Neural Networks (BGNNs),
which dynamically modify graph structures to address boundary conditions.
Boundary graph structures are constructed via modifying edges, augmenting node
features, and dynamically inserting virtual nodes. The new BGNNs are tested on
complex 3D granular flow processes of hoppers and rotating drums which are
standard components of industrial machinery. Using precise simulations that are
obtained by an expensive and complex discrete element method, BGNNs are
evaluated in terms of computational efficiency as well as prediction accuracy
of particle flows and mixing entropies. Even if complex boundaries are present,
BGNNs are able to accurately reproduce 3D granular flows within simulation
uncertainties over hundreds of thousands of simulation timesteps, and most
notably particles completely stay within the geometric objects without using
handcrafted conditions or restrictions.

    

### [[2107.00594] Pretext Tasks selection for multitask self-supervised speech representation learning](http://arxiv.org/abs/2107.00594)


  Through solving pretext tasks, self-supervised learning leverages unlabeled
data to extract useful latent representations replacing traditional input
features in the downstream task. In audio/speech signal processing, a wide
range of features where engineered through decades of research efforts. As it
turns out, learning to predict such features (a.k.a pseudo-labels) has proven
to be a particularly relevant pretext task, leading to useful self-supervised
representations which prove to be effective for downstream tasks. However,
methods and common practices for combining such pretext tasks for better
performance on the downstream task have not been explored and understood
properly. In fact, the process relies almost exclusively on a computationally
heavy experimental procedure, which becomes intractable with the increase of
the number of pretext tasks. This paper introduces a method to select a group
of pretext tasks among a set of candidates. The method we propose estimates
calibrated weights for the partial losses corresponding to the considered
pretext tasks during the self-supervised training process. The experiments
conducted on automatic speech recognition, speaker and emotion recognition
validate our approach, as the groups selected and weighted with our method
perform better than classic baselines, thus facilitating the selection and
combination of relevant pseudo-labels for self-supervised representation
learning.

    

### [[2110.05096] Density-Based Clustering with Kernel Diffusion](http://arxiv.org/abs/2110.05096)


  Finding a suitable density function is essential for density-based clustering
algorithms such as DBSCAN and DPC. A naive density corresponding to the
indicator function of a unit $d$-dimensional Euclidean ball is commonly used in
these algorithms. Such density suffers from capturing local features in complex
datasets. To tackle this issue, we propose a new kernel diffusion density
function, which is adaptive to data of varying local distributional
characteristics and smoothness. Furthermore, we develop a surrogate that can be
efficiently computed in linear time and space and prove that it is
asymptotically equivalent to the kernel diffusion density function. Extensive
empirical experiments on benchmark and large-scale face image datasets show
that the proposed approach not only achieves a significant improvement over
classic density-based clustering algorithms but also outperforms the
state-of-the-art face clustering methods by a large margin.

    

### [[2110.05204] CLIP4Caption ++: Multi-CLIP for Video Caption](http://arxiv.org/abs/2110.05204)


  This report describes our solution to the VALUE Challenge 2021 in the
captioning task. Our solution, named CLIP4Caption++, is built on
X-Linear/X-Transformer, which is an advanced model with encoder-decoder
architecture. We make the following improvements on the proposed
CLIP4Caption++: We employ an advanced encoder-decoder model architecture
X-Transformer as our main framework and make the following improvements: 1) we
utilize three strong pre-trained CLIP models to extract the text-related
appearance visual features. 2) we adopt the TSN sampling strategy for data
enhancement. 3) we involve the video subtitle information to provide richer
semantic information. 3) we introduce the subtitle information, which fuses
with the visual features as guidance. 4) we design word-level and
sentence-level ensemble strategies. Our proposed method achieves 86.5, 148.4,
64.5 CIDEr scores on VATEX, YC2C, and TVC datasets, respectively, which shows
the superior performance of our proposed CLIP4Caption++ on all three datasets.

    

### [[2110.06526] Practice Problems for Hardware Engineers](http://arxiv.org/abs/2110.06526)


  This book is to help undergraduate and graduate students of electrical and
computer engineering disciplines with their job interviews. It may also be used
as a practice resource while taking courses in VLSI, logic and computer
architecture design. The first edition consists of more than 200 problems and
their solutions which the author has used in his VLSI, logic, and architectures
courses while teaching at USC. The author wishes this book to be available to
students and engineers free of charge, subject to the copyright policy on page
3.

    

### [[2110.06407] Efficient Linearizability Checking for Actor-based Systems](http://arxiv.org/abs/2110.06407)


  Recent demand for distributed software had led to a surge in popularity in
actor-based frameworks. However, even with the stylized message passing model
of actors, writing correct distributed software is still difficult. We present
our work on linearizability checking in DS2, an integrated framework for
specifying, synthesizing, and testing distributed actor systems. The key
insight of our approach is that often subcomponents of distributed actor
systems represent common algorithms or data structures (e.g.\ a distributed
hash table or tree) that can be validated against a simple sequential model of
the system. This makes it easy for developers to validate their concurrent
actor systems without complex specifications. DS2 automatically explores the
concurrent schedules that system could arrive at, and it compares observed
output of the system to ensure it is equivalent to what the sequential
implementation could have produced. We describe DS2's linearizability checking
and test it on several concurrent replication algorithms from the literature.
We explore in detail how different algorithms for enumerating the model
schedule space fare in finding bugs in actor systems, and we present our own
refinements on algorithms for exploring actor system schedules that we show are
effective in finding bugs.

    

### [[2110.06780] Spatially constrained direction dependent calibration](http://arxiv.org/abs/2110.06780)


  Direction dependent calibration of widefield radio interferometers estimates
the systematic errors along multiple directions in the sky. This is necessary
because with most systematic errors that are caused by effects such as the
ionosphere or the receiver beam shape, there is significant spatial variation.
Fortunately, there is some deterministic behavior of these variations in most
situations. We enforce this underlying smooth spatial behavior of systematic
errors as an additional constraint onto spectrally constrained direction
dependent calibration. Using both analysis and simulations, we show that this
additional spatial constraint improves the performance of multi-frequency
direction dependent calibration.

    

### [[2110.06870] Architecture of a Junkyard Datacenter](http://arxiv.org/abs/2110.06870)


  It requires significant energy to manufacture and deploy computational
devices. Traditional discussions of the energy-efficiency of compute measure
operational energy, i.e.\ how many FLOPS in a 50\,MW datacenter. However, if we
consider the true lifetime energy use of modern devices, the majority actually
comes not from runtime use but from manufacture and deployment. In this paper,
then, we suggest that perhaps the most climate-impactful action we can take is
to extend the service lifetime of existing compute.
We design two new metrics to measure how to balance continued service of
older devices with the superlinear runtime improvements of newer machines. The
first looks at carbon per raw compute, amortized across the operation and
manufacture of devices. The second considers use of components beyond compute,
such as batteries or radios in smartphone platforms. We use these metrics to
redefine device service lifetime in terms of carbon efficiency. We then realize
a real-world ``junkyard datacenter'' made up of Nexus 4 and Nexus 5 phones,
which are nearly a decade past their official end-of-life dates. This new-old
datacenter is able to nearly match and occasionally exceed modern cloud compute
offerings.

    

### [[2110.06223] Investigating the Effect of Natural Language Explanations on Out-of-Distribution Generalization in Few-shot NLI](http://arxiv.org/abs/2110.06223)


  Although neural models have shown strong performance in datasets such as
SNLI, they lack the ability to generalize out-of-distribution (OOD). In this
work, we formulate a few-shot learning setup and examine the effects of natural
language explanations on OOD generalization. We leverage the templates in the
HANS dataset and construct templated natural language explanations for each
template. Although generated explanations show competitive BLEU scores against
groundtruth explanations, they fail to improve prediction performance. We
further show that generated explanations often hallucinate information and miss
key elements that indicate the label.

    

### [[2110.06263] Speech Summarization using Restricted Self-Attention](http://arxiv.org/abs/2110.06263)


  Speech summarization is typically performed by using a cascade of speech
recognition and text summarization models. End-to-end modeling of speech
summarization models is challenging due to memory and compute constraints
arising from long input audio sequences. Recent work in document summarization
has inspired methods to reduce the complexity of self-attentions, which enables
transformer models to handle long sequences. In this work, we introduce a
single model optimized end-to-end for speech summarization. We apply the
restricted self-attention technique from text-based models to speech models to
address the memory and compute constraints. We demonstrate that the proposed
model learns to directly summarize speech for the How-2 corpus of instructional
videos. The proposed end-to-end model outperforms the previously proposed
cascaded model by 3 points absolute on ROUGE. Further, we consider the spoken
language understanding task of predicting concepts from speech inputs and show
that the proposed end-to-end model outperforms the cascade model by 4 points
absolute F-1.

    

### [[2110.06339] Natural Computational Architectures for Cognitive Info-Communication](http://arxiv.org/abs/2110.06339)


  Recent comprehensive overview of 40 years of research in cognitive
architectures, (Kotseruba and Tsotsos 2020), evaluates modelling of the core
cognitive abilities in humans, but only marginally addresses biologically
plausible approaches based on natural computation. This mini review presents a
set of perspectives and approaches which have shaped the development of
biologically inspired computational models in the recent past that can lead to
the development of biologically more realistic cognitive architectures. For
describing continuum of natural cognitive architectures, from basal cellular to
human-level cognition, we use evolutionary info-computational framework, where
natural/ physical/ morphological computation leads to evolution of increasingly
complex cognitive systems. Forty years ago, when the first cognitive
architectures have been proposed, understanding of cognition, embodiment and
evolution was different. So was the state of the art of information physics,
bioinformatics, information chemistry, computational neuroscience, complexity
theory, self-organization, theory of evolution, information and computation.
Novel developments support a constructive interdisciplinary framework for
cognitive architectures in the context of computing nature, where interactions
between constituents at different levels of organization lead to
complexification of agency and increased cognitive capacities. We identify
several important research questions for further investigation that can
increase understanding of cognition in nature and inspire new developments of
cognitive technologies. Recently, basal cell cognition attracted a lot of
interest for its possible applications in medicine, new computing technologies,
as well as micro- and nanorobotics.

    

### [[2110.06348] Exact and Bounded Collision Probability for Motion Planning under Gaussian Uncertainty](http://arxiv.org/abs/2110.06348)


  Computing collision-free trajectories is of prime importance for safe
navigation. We present an approach for computing the collision probability
under Gaussian distributed motion and sensing uncertainty with the robot and
static obstacle shapes approximated as ellipsoids. The collision condition is
formulated as the distance between ellipsoids and unlike previous approaches we
provide a method for computing the exact collision probability. Furthermore, we
provide a tight upper bound that can be computed much faster during online
planning. Comparison to other state-of-the-art methods is also provided. The
proposed method is evaluated in simulation under varying configuration and
number of obstacles.

    

### [[2110.06396] GridLearn: Multiagent Reinforcement Learning for Grid-Aware Building Energy Management](http://arxiv.org/abs/2110.06396)


  Increasing amounts of distributed generation in distribution networks can
provide both challenges and opportunities for voltage regulation across the
network. Intelligent control of smart inverters and other smart building energy
management systems can be leveraged to alleviate these issues. GridLearn is a
multiagent reinforcement learning platform that incorporates both building
energy models and power flow models to achieve grid level goals, by controlling
behind-the-meter resources. This study demonstrates how multi-agent
reinforcement learning can preserve building owner privacy and comfort while
pursuing grid-level objectives. Building upon the CityLearn framework which
considers RL for building-level goals, this work expands the framework to a
network setting where grid-level goals are additionally considered. As a case
study, we consider voltage regulation on the IEEE-33 bus network using
controllable building loads, energy storage, and smart inverters. The results
show that the RL agents nominally reduce instances of undervoltages and reduce
instances of overvoltages by 34%.

    

### [[2110.06419] Federated Natural Language Generation for Personalized Dialogue System](http://arxiv.org/abs/2110.06419)


  Neural conversational models have long suffered from the problem of
inconsistency and lacking coherent personality. To address the issue,
persona-based models capturing individual characteristics have been proposed,
but they still face the dilemma of model adaption and data privacy. To break
this dilemma, we propose a novel Federated Natural Language Generation (FedNLG)
framework, which learns personalized representations from various dataset on
distributed devices, and thus implements the personalized dialogue system
efficiently and safely. FedNLG first pre-trains parameters of standard neural
conversational model over a large dialogue corpus, and then fine-tune the model
parameters and persona embeddings on specific datasets, in a federated manner.
Thus, the model could simultaneously learn the persona embeddings in local
clients and learn shared model parameters by federated aggregation, which
achieves accuracyprivacy balance. By conducting extensive experiments, we
demonstrate the effectiveness of our model by pre-training model over Cornell
Movie-Dialogs Corpus and fine-tuning the model over two TV series dataset.

    

### [[2110.06443] Harnessing the Conditioning Sensorium for Improved Image Translation](http://arxiv.org/abs/2110.06443)


  Multi-modal domain translation typically refers to synthesizing a novel image
that inherits certain localized attributes from a 'content' image (e.g. layout,
semantics, or geometry), and inherits everything else (e.g. texture, lighting,
sometimes even semantics) from a 'style' image. The dominant approach to this
task is attempting to learn disentangled 'content' and 'style' representations
from scratch. However, this is not only challenging, but ill-posed, as what
users wish to preserve during translation varies depending on their goals.
Motivated by this inherent ambiguity, we define 'content' based on conditioning
information extracted by off-the-shelf pre-trained models. We then train our
style extractor and image decoder with an easy to optimize set of
reconstruction objectives. The wide variety of high-quality pre-trained models
available and simple training procedure makes our approach straightforward to
apply across numerous domains and definitions of 'content'. Additionally it
offers intuitive control over which aspects of 'content' are preserved across
domains. We evaluate our method on traditional, well-aligned, datasets such as
CelebA-HQ, and propose two novel datasets for evaluation on more complex
scenes: ClassicTV and FFHQ-Wild. Our approach, Sensorium, enables higher
quality domain translation for more complex scenes.

    

### [[2110.06459] Learning to Select Historical News Articles for Interaction based Neural News Recommendation](http://arxiv.org/abs/2110.06459)


  The key to personalized news recommendation is to match the user's interests
with the candidate news precisely and efficiently. Most existing approaches
embed user interests into a representation vector then recommend by comparing
it with the candidate news vector. In such a workflow, fine-grained matching
signals may be lost. Recent studies try to cover that by modeling fine-grained
interactions between the candidate news and each browsed news article of the
user. Despite the effectiveness improvement, these models suffer from much
higher computation costs online. Consequently, it remains a tough issue to take
advantage of effective interactions in an efficient way. To address this
problem, we proposed an end-to-end Selective Fine-grained Interaction framework
(SFI) with a learning-to-select mechanism. Instead of feeding all historical
news into interaction, SFI can quickly select informative historical news
w.r.t. the candidate and exclude others from following computations. We empower
the selection to be both sparse and automatic, which guarantees efficiency and
effectiveness respectively. Extensive experiments on the publicly available
dataset MIND validates the superiority of SFI over the state-of-the-art
methods: with only five historical news selected, it can significantly improve
the AUC by 2.17% over the state-of-the-art interaction-based models; at the
same time, it is four times faster.

    

### [[2110.06467] Dual-branch Attention-In-Attention Transformer for single-channel speech enhancement](http://arxiv.org/abs/2110.06467)


  Curriculum learning begins to thrive in the speech enhancement area, which
decouples the original spectrum estimation task into multiple easier sub-tasks
to achieve better performance. Motivated by that, we propose a dual-branch
attention-in-attention transformer dubbed DB-AIAT to handle both coarse- and
fine-grained regions of the spectrum in parallel. From a complementary
perspective, a magnitude masking branch is proposed to coarsely estimate the
overall magnitude spectrum, and simultaneously a complex refining branch is
elaborately designed to compensate for the missing spectral details and
implicitly derive phase information. Within each branch, we propose a novel
attention-in-attention transformer-based module to replace the conventional
RNNs and temporal convolutional networks for temporal sequence modeling.
Specifically, the proposed attention-in-attention transformer consists of
adaptive temporal-frequency attention transformer blocks and an adaptive
hierarchical attention module, aiming to capture long-term temporal-frequency
dependencies and further aggregate global hierarchical contextual information.
Experimental results on Voice Bank + DEMAND demonstrate that DB-AIAT yields
state-of-the-art performance (e.g., 3.31 PESQ, 94.7% STOI and 10.79dB SSNR)
over previous advanced systems with a relatively small model size (2.81M).

    

### [[2110.06474] ActiveEA: Active Learning for Neural Entity Alignment](http://arxiv.org/abs/2110.06474)


  Entity Alignment (EA) aims to match equivalent entities across different
Knowledge Graphs (KGs) and is an essential step of KG fusion. Current
mainstream methods -- neural EA models -- rely on training with seed alignment,
i.e., a set of pre-aligned entity pairs which are very costly to annotate. In
this paper, we devise a novel Active Learning (AL) framework for neural EA,
aiming to create highly informative seed alignment to obtain more effective EA
models with less annotation cost. Our framework tackles two main challenges
encountered when applying AL to EA: (1) How to exploit dependencies between
entities within the AL strategy. Most AL strategies assume that the data
instances to sample are independent and identically distributed. However,
entities in KGs are related. To address this challenge, we propose a
structure-aware uncertainty sampling strategy that can measure the uncertainty
of each entity as well as its impact on its neighbour entities in the KG. (2)
How to recognise entities that appear in one KG but not in the other KG (i.e.,
bachelors). Identifying bachelors would likely save annotation budget. To
address this challenge, we devise a bachelor recognizer paying attention to
alleviate the effect of sampling bias. Empirical results show that our proposed
AL strategy can significantly improve sampling quality with good generality
across different datasets, EA models and amount of bachelors.

    

### [[2110.06477] Feudal Reinforcement Learning by Reading Manuals](http://arxiv.org/abs/2110.06477)


  Reading to act is a prevalent but challenging task which requires the ability
to reason from a concise instruction. However, previous works face the semantic
mismatch between the low-level actions and the high-level language descriptions
and require the human-designed curriculum to work properly. In this paper, we
present a Feudal Reinforcement Learning (FRL) model consisting of a manager
agent and a worker agent. The manager agent is a multi-hop plan generator
dealing with high-level abstract information and generating a series of
sub-goals in a backward manner. The worker agent deals with the low-level
perceptions and actions to achieve the sub-goals one by one. In comparison, our
FRL model effectively alleviate the mismatching between text-level inference
and low-level perceptions and actions; and is general to various forms of
environments, instructions and manuals; and our multi-hop plan generator can
significantly boost for challenging tasks where multi-step reasoning form the
texts is critical to resolve the instructed goals. We showcase our approach
achieves competitive performance on two challenging tasks, Read to Fight
Monsters (RTFM) and Messenger, without human-designed curriculum learning.

    

### [[2110.06516] 2D Multi-Class Model for Gray and White Matter Segmentation of the Cervical Spinal Cord at 7T](http://arxiv.org/abs/2110.06516)


  The spinal cord (SC), which conveys information between the brain and the
peripheral nervous system, plays a key role in various neurological disorders
such as multiple sclerosis (MS) and amyotrophic lateral sclerosis (ALS), in
which both gray matter (GM) and white matter (WM) may be impaired. While
automated methods for WM/GM segmentation are now largely available, these
techniques, developed for conventional systems (3T or lower) do not necessarily
perform well on 7T MRI data, which feature finer details, contrasts, but also
different artifacts or signal dropout.
The primary goal of this study is thus to propose a new deep learning model
that allows robust SC/GM multi-class segmentation based on ultra-high
resolution 7T T2*-w MR images. The second objective is to highlight the
relevance of implementing a specific data augmentation (DA) strategy, in
particular to generate a generic model that could be used for multi-center
studies at 7T.

    

### [[2110.06523] User Experiences Oriented Sightseeing Spot Recommendation](http://arxiv.org/abs/2110.06523)


  POI recommendation is a key task in tourism information systems. However, in
contrast to conventional point of interest (POI) recommender systems, the
available data is extremely sparse; most tourist visit a few sightseeing spots
once and most of these spots have no check-in data from new tourists. Most
conventional systems rank sightseeing spots based on their popularity,
reputations, and category-based similarities with users' preferences. They do
not clarify what users can experience in these spots, which makes it difficult
to meet diverse tourism needs. To this end, in this work, we propose a
mechanism to recommend POIs to tourists. Our mechanism include two components:
one is a probabilistic model that reveals the user behaviors in tourism; the
other is a pseudo rating mechanism to handle the cold-start issue in POIs
recommendations. We carried out extensive experiments with two datasets
collected from Flickr. The experimental results demonstrate that our methods
are superior to the state-of-the-art methods in both the recommendation
performances (precision, recall and F-measure) and fairness. The experimental
results also validate the robustness of the proposed methods, i.e., our methods
can handle well the issue of data sparsity.

    

### [[2110.06536] NeurIPS 2021 Competition IGLU: Interactive Grounded Language Understanding in a Collaborative Environment](http://arxiv.org/abs/2110.06536)


  Human intelligence has the remarkable ability to quickly adapt to new tasks
and environments. Starting from a very young age, humans acquire new skills and
learn how to solve new tasks either by imitating the behavior of others or by
following provided natural language instructions. To facilitate research in
this direction, we propose \emph{IGLU: Interactive Grounded Language
Understanding in a Collaborative Environment}.
The primary goal of the competition is to approach the problem of how to
build interactive agents that learn to solve a task while provided with
grounded natural language instructions in a collaborative environment.
Understanding the complexity of the challenge, we split it into sub-tasks to
make it feasible for participants.
This research challenge is naturally related, but not limited, to two fields
of study that are highly relevant to the NeurIPS community: Natural Language
Understanding and Generation (NLU/G) and Reinforcement Learning (RL).
Therefore, the suggested challenge can bring two communities together to
approach one of the important challenges in AI. Another important aspect of the
challenge is the dedication to perform a human-in-the-loop evaluation as a
final evaluation for the agents developed by contestants.

    

### [[2110.06592] Life is not black and white -- Combining Semi-Supervised Learning with fuzzy labels](http://arxiv.org/abs/2110.06592)


  The required amount of labeled data is one of the biggest issues in deep
learning. Semi-Supervised Learning can potentially solve this issue by using
additional unlabeled data. However, many datasets suffer from variability in
the annotations. The aggregated labels from these annotation are not consistent
between different annotators and thus are considered fuzzy. These fuzzy labels
are often not considered by Semi-Supervised Learning. This leads either to an
inferior performance or to higher initial annotation costs in the complete
machine learning development cycle. We envision the incorporation of fuzzy
labels into Semi-Supervised Learning and give a proof-of-concept of the
potential lower costs and higher consistency in the complete development cycle.
As part of our concept, we discuss current limitations, futures research
opportunities and potential broad impacts.

    

### [[2110.06612] Exploring Dense Retrieval for Dialogue Response Selection](http://arxiv.org/abs/2110.06612)


  Recent research on dialogue response selection has been mainly focused on
selecting a proper response from a pre-defined small set of candidates using
sophisticated neural models. Due to their heavy computational overhead, they
are unable to select responses from a large candidate pool. In this study, we
present a solution to directly select proper responses from a large corpus or
even a nonparallel corpus that only consists of unpaired sentences, using a
dense retrieval model. We extensively test our proposed approach under two
experiment settings: (i) re-rank experiment that aims to rank a small set of
pre-defined candidates; (ii) full-rank experiment where the target is to
directly select proper responses from a full candidate pool that may contain
millions of candidates. For re-rank setting, the superiority is quite
surprising given its simplicity. For full-rank setting, we can emphasize that
we are the first to do such evaluation. Moreover, human evaluation results show
that increasing the size of nonparallel corpus leads to further improvement of
our model performance\footnote{All our source codes, models and other related
resources are publically available at
\url{this https URL}.

    

### [[2110.06623] SSSNET: Semi-Supervised Signed Network Clustering](http://arxiv.org/abs/2110.06623)


  Node embeddings are a powerful tool in the analysis of networks; yet, their
full potential for the important task of node clustering has not been fully
exploited. In particular, most state-of-the-art methods generating node
embeddings of signed networks focus on link sign prediction, and those that
pertain to node clustering are usually not graph neural network (GNN) methods.
Here, we introduce a novel probabilistic balanced normalized cut loss for
training nodes in a GNN framework for semi-supervised signed network
clustering, called SSSNET. The method is end-to-end in combining embedding
generation and clustering without an intermediate step; it has node clustering
as main focus, with an emphasis on polarization effects arising in networks.
The main novelty of our approach is a new take on the role of social balance
theory for signed network embeddings. The standard heuristic for justifying the
criteria for the embeddings hinges on the assumption that "an enemy's enemy is
a friend". Here, instead, a neutral stance is assumed on whether or not the
enemy of an enemy is a friend. Experimental results on various data sets,
including a synthetic signed stochastic block model, a polarized version of it,
and real-world data at different scales, demonstrate that SSSNET can achieve
comparable or better results than state-of-the-art spectral clustering methods,
for a wide range of noise and sparsity levels. SSSNET complements existing
methods through the possibility of including exogenous information, in the form
of node-level features or labels.

    

### [[2110.06630] Fuzzy Overclustering: Semi-Supervised Classification of Fuzzy Labels with Overclustering and Inverse Cross-Entropy](http://arxiv.org/abs/2110.06630)


  Deep learning has been successfully applied to many classification problems
including underwater challenges. However, a long-standing issue with deep
learning is the need for large and consistently labeled datasets. Although
current approaches in semi-supervised learning can decrease the required amount
of annotated data by a factor of 10 or even more, this line of research still
uses distinct classes. For underwater classification, and uncurated real-world
datasets in general, clean class boundaries can often not be given due to a
limited information content in the images and transitional stages of the
depicted objects. This leads to different experts having different opinions and
thus producing fuzzy labels which could also be considered ambiguous or
divergent. We propose a novel framework for handling semi-supervised
classifications of such fuzzy labels. It is based on the idea of overclustering
to detect substructures in these fuzzy labels. We propose a novel loss to
improve the overclustering capability of our framework and show the benefit of
overclustering for fuzzy labels. We show that our framework is superior to
previous state-of-the-art semi-supervised methods when applied to real-world
plankton data with fuzzy labels. Moreover, we acquire 5 to 10\% more consistent
predictions of substructures.

    

### [[2110.06637] Knowledge Graph-enhanced Sampling for Conversational Recommender System](http://arxiv.org/abs/2110.06637)


  The traditional recommendation systems mainly use offline user data to train
offline models, and then recommend items for online users, thus suffering from
the unreliable estimation of user preferences based on sparse and noisy
historical data. Conversational Recommendation System (CRS) uses the
interactive form of the dialogue systems to solve the intrinsic problems of
traditional recommendation systems. However, due to the lack of contextual
information modeling, the existing CRS models are unable to deal with the
exploitation and exploration (E&E) problem well, resulting in the heavy burden
on users. To address the aforementioned issue, this work proposes a contextual
information enhancement model tailored for CRS, called Knowledge Graph-enhanced
Sampling (KGenSam). KGenSam integrates the dynamic graph of user interaction
data with the external knowledge into one heterogeneous Knowledge Graph (KG) as
the contextual information environment. Then, two samplers are designed to
enhance knowledge by sampling fuzzy samples with high uncertainty for obtaining
user preferences and reliable negative samples for updating recommender to
achieve efficient acquisition of user preferences and model updating, and thus
provide a powerful solution for CRS to deal with E&E problem. Experimental
results on two real-world datasets demonstrate the superiority of KGenSam with
significant improvements over state-of-the-art methods.

    

### [[2110.06651] MDERank: A Masked Document Embedding Rank Approach for Unsupervised Keyphrase Extraction](http://arxiv.org/abs/2110.06651)


  Keyphrases are phrases in a document providing a concise summary of core
content, helping readers to understand what the article is talking about in a
minute. However, existing unsupervised works are not robust enough to handle
various types of documents owing to the mismatch of sequence length for
comparison. In this paper, we propose a novel unsupervised keyword extraction
method by leveraging the BERT-based model to select and rank candidate
keyphrases with a MASK strategy. In addition, we further enhance the model,
denoted as Keyphrases Extraction BERT (KPEBERT), via designing a compatible
self-supervised task and conducting a contrast learning. We conducted extensive
experimental evaluation to demonstrate the superiority and robustness of the
proposed method as well as the effectiveness of KPEBERT.

    

### [[2110.06674] Truthful AI: Developing and governing AI that does not lie](http://arxiv.org/abs/2110.06674)


  In many contexts, lying -- the use of verbal falsehoods to deceive -- is
harmful. While lying has traditionally been a human affair, AI systems that
make sophisticated verbal statements are becoming increasingly prevalent. This
raises the question of how we should limit the harm caused by AI "lies" (i.e.
falsehoods that are actively selected for). Human truthfulness is governed by
social norms and by laws (against defamation, perjury, and fraud). Differences
between AI and humans present an opportunity to have more precise standards of
truthfulness for AI, and to have these standards rise over time. This could
provide significant benefits to public epistemics and the economy, and mitigate
risks of worst-case AI futures.
Establishing norms or laws of AI truthfulness will require significant work
to: (1) identify clear truthfulness standards; (2) create institutions that can
judge adherence to those standards; and (3) develop AI systems that are
robustly truthful.
Our initial proposals for these areas include: (1) a standard of avoiding
"negligent falsehoods" (a generalisation of lies that is easier to assess); (2)
institutions to evaluate AI systems before and after real-world deployment; and
(3) explicitly training AI systems to be truthful via curated datasets and
human interaction.
A concerning possibility is that evaluation mechanisms for eventual
truthfulness standards could be captured by political interests, leading to
harmful censorship and propaganda. Avoiding this might take careful attention.
And since the scale of AI speech acts might grow dramatically over the coming
decades, early truthfulness standards might be particularly important because
of the precedents they set.

    

### [[2110.06696] Mengzi: Towards Lightweight yet Ingenious Pre-trained Models for Chinese](http://arxiv.org/abs/2110.06696)


  Although pre-trained models (PLMs) have achieved remarkable improvements in a
wide range of NLP tasks, they are expensive in terms of time and resources.
This calls for the study of training more efficient models with less
computation but still ensures impressive performance. Instead of pursuing a
larger scale, we are committed to developing lightweight yet more powerful
models trained with equal or less computation and friendly to rapid deployment.
This technical report releases our pre-trained model called Mengzi, which
stands for a family of discriminative, generative, domain-specific, and
multimodal pre-trained model variants, capable of a wide range of language and
vision tasks. Compared with public Chinese PLMs, Mengzi is simple but more
powerful. Our lightweight model has achieved new state-of-the-art results on
the widely-used CLUE benchmark with our optimized pre-training and fine-tuning
techniques. Without modifying the model architecture, our model can be easily
employed as an alternative to existing PLMs. Our sources are available at
this https URL.

    

### [[2110.06697] Semantic Image Fusion](http://arxiv.org/abs/2110.06697)


  Image fusion methods and metrics for their evaluation have conventionally
used pixel-based or low-level features. However, for many applications, the aim
of image fusion is to effectively combine the semantic content of the input
images. This paper proposes a novel system for the semantic combination of
visual content using pre-trained CNN network architectures. Our proposed
semantic fusion is initiated through the fusion of the top layer feature map
outputs (for each input image)through gradient updating of the fused image
input (so-called image optimisation). Simple "choose maximum" and "local
majority" filter based fusion rules are utilised for feature map fusion. This
provides a simple method to combine layer outputs and thus a unique framework
to fuse single-channel and colour images within a decomposition pre-trained for
classification and therefore aligned with semantic fusion. Furthermore, class
activation mappings of each input image are used to combine semantic
information at a higher level. The developed methods are able to give
equivalent low-level fusion performance to state of the art methods while
providing a unique architecture to combine semantic information from multiple
images.

    

### [[2110.06758] HEDP: A Method for Early Forecasting Software Defects based on Human Error Mechanisms](http://arxiv.org/abs/2110.06758)


  As the primary cause of software defects, human error is the key to
understanding, and perhaps to predicting and avoiding them. Little research has
been done to predict defects on the basis of the cognitive errors that cause
them. This paper proposes an approach to predicting software defects through
knowledge about the cognitive mechanisms of human errors. Our theory is that
the main process behind a software defect is that an error-prone scenario
triggers human error modes, which psychologists have observed to recur across
diverse activities. Software defects can then be predicted by identifying such
scenarios, guided by this knowledge of typical error modes. The proposed idea
emphasizes predicting the exact location and form of a possible defect. We
conducted two case studies to demonstrate and validate this approach, with 55
programmers in a programming competition and 5 analysts serving as the users of
the approach. We found it impressive that the approach was able to predict, at
the requirement phase, the exact locations and forms of 7 out of the 22 (31.8%)
specific types of defects that were found in the code. The defects predicted
tended to be common defects: their occurrences constituted 75.7% of the total
number of defects in the 55 developed programs; each of them was introduced by
at least two persons. The fraction of the defects introduced by a programmer
that were predicted was on average (over all programmers) 75%. Furthermore,
these predicted defects were highly persistent through the debugging process.
If the prediction had been used to successfully prevent these defects, this
could have saved 46.2% of the debugging iterations. This excellent capability
of forecasting the exact locations and forms of possible defects at the early
phases of software development recommends the approach for substantial benefits
to defect prevention and early detection.

    

### [[2110.06775] Using UAVs for vehicle tracking and collision risk assessment at intersections](http://arxiv.org/abs/2110.06775)


  Assessing collision risk is a critical challenge to effective traffic safety
management. The deployment of unmanned aerial vehicles (UAVs) to address this
issue has shown much promise, given their wide visual field and movement
flexibility. This research demonstrates the application of UAVs and V2X
connectivity to track the movement of road users and assess potential
collisions at intersections. The study uses videos captured by UAVs. The
proposed method combines deep-learning based tracking algorithms and
time-to-collision tasks. The results not only provide beneficial information
for vehicle's recognition of potential crashes and motion planning but also
provided a valuable tool for urban road agencies and safety management
engineers.

    

### [[2110.06794] The Layout Generation Algorithm of Graphic Design Based on Transformer-CVAE](http://arxiv.org/abs/2110.06794)


  Graphic design is ubiquitous in people's daily lives. For graphic design, the
most time-consuming task is laying out various components in the interface.
Repetitive manual layout design will waste a lot of time for professional
graphic designers. Existing templates are usually rudimentary and not suitable
for most designs, reducing efficiency and limiting creativity. This paper
implemented the Transformer model and conditional variational autoencoder
(CVAE) to the graphic design layout generation task. It proposed an end-to-end
graphic design layout generation model named LayoutT-CVAE. We also proposed
element disentanglement and feature-based disentanglement strategies and
introduce new graphic design principles and similarity metrics into the model,
which significantly increased the controllability and interpretability of the
deep model. Compared with the existing state-of-art models, the layout
generated by ours performs better on many metrics.

    

### [[2110.06804] A comprehensive review of Binary Neural Network](http://arxiv.org/abs/2110.06804)


  Binary Neural Network (BNN) method is an extreme application of convolutional
neural network (CNN) parameter quantization. As opposed to the original CNN
methods which employed floating-point computation with full-precision weights
and activations, BBN uses 1-bit activations and weights. With BBNs, a
significant amount of storage, network complexity and energy consumption can be
reduced, and neural networks can be implemented more efficiently in embedded
applications. Unfortunately, binarization causes severe information loss. A gap
still exists between full-precision CNN models and their binarized
counterparts. The recent developments in BNN have led to a lot of algorithms
and solutions that have helped address this issue. This article provides a full
overview of recent developments in BNN. The present paper focuses exclusively
on 1-bit activations and weights networks, as opposed to previous surveys in
which low-bit works are mixed in. In this paper, we conduct a complete
investigation of BNN's development from their predecessors to the latest BNN
algorithms and techniques, presenting a broad design pipeline, and discussing
each module's variants. Along the way, this paper examines BNN (a) purpose:
their early successes and challenges; (b) BNN optimization: selected
representative works that contain key optimization techniques; (c) deployment:
open-source frameworks for BNN modeling and development; (d) terminal:
efficient computing architectures and devices for BNN and (e) applications:
diverse applications with BNN. Moreover, this paper discusses potential
directions and future research opportunities for the latest BNN algorithms and
techniques, presents a broad design pipeline, and discusses each module's
variants.

    

### [[2110.06823] A Speaker-Aware Learning Framework for Improving Multi-turn Dialogue Coherence](http://arxiv.org/abs/2110.06823)


  This paper presents a novel open-domain dialogue generation framework
emphasizing the differentiation of speakers in multi-turn conversations.
Differing from prior work that solely relies on the content of conversation
history to generate a response, we argue that capturing relative social
relations among utterances (i.e., generated by either the same speaker or
different persons) benefits the machine capturing fine-grained context
information from a conversation history to improve context coherence in the
generated response. Given that, we propose a speaker-aware framework, named
Parallel Hierarchical Attentive Encoder-Decoder (PHAED), that aims to model
each utterance with the awareness of its speaker and contextual associations
with the same speaker's previous messages. Specifically, in a conversation
involving two speakers, we regard the utterances from one speaker as responses
and those from the other as queries. After understanding queries via our
encoder with inner-query and inter-query encodings, our decoder reuses the
hidden states of previously generated responses to generate a new response. Our
empirical results show that PHAED outperforms the state-of-the-art in both
automatic and human evaluations. Furthermore, our ablation study shows that
dialogue models with speaker tokens can generally decrease the possibility of
generating non-coherent responses regarding the conversation context.

    

### [[2110.06830] CONetV2: Efficient Auto-Channel Size Optimization for CNNs](http://arxiv.org/abs/2110.06830)


  Neural Architecture Search (NAS) has been pivotal in finding optimal network
configurations for Convolution Neural Networks (CNNs). While many methods
explore NAS from a global search-space perspective, the employed optimization
schemes typically require heavy computational resources. This work introduces a
method that is efficient in computationally constrained environments by
examining the micro-search space of channel size. In tackling channel-size
optimization, we design an automated algorithm to extract the dependencies
within different connected layers of the network. In addition, we introduce the
idea of knowledge distillation, which enables preservation of trained weights,
admist trials where the channel sizes are changing. Further, since the standard
performance indicators (accuracy, loss) fail to capture the performance of
individual network components (providing an overall network evaluation), we
introduce a novel metric that highly correlates with test accuracy and enables
analysis of individual network layers. Combining dependency extraction,
metrics, and knowledge distillation, we introduce an efficient searching
algorithm, with simulated annealing inspired stochasticity, and demonstrate its
effectiveness in finding optimal architectures that outperform baselines by a
large margin.

    

### [[2110.06831] Safe Driving via Expert Guided Policy Optimization](http://arxiv.org/abs/2110.06831)


  When learning common skills like driving, beginners usually have domain
experts standing by to ensure the safety of the learning process. We formulate
such learning scheme under the Expert-in-the-loop Reinforcement Learning where
a guardian is introduced to safeguard the exploration of the learning agent.
While allowing the sufficient exploration in the uncertain environment, the
guardian intervenes under dangerous situations and demonstrates the correct
actions to avoid potential accidents. Thus ERL enables both exploration and
expert's partial demonstration as two training sources. Following such a
setting, we develop a novel Expert Guided Policy Optimization (EGPO) method
which integrates the guardian in the loop of reinforcement learning. The
guardian is composed of an expert policy to generate demonstration and a switch
function to decide when to intervene. Particularly, a constrained optimization
technique is used to tackle the trivial solution that the agent deliberately
behaves dangerously to deceive the expert into taking over. Offline RL
technique is further used to learn from the partial demonstration generated by
the expert. Safe driving experiments show that our method achieves superior
training and test-time safety, outperforms baselines with a substantial margin
in sample efficiency, and preserves the generalizabiliy to unseen environments
in test-time. Demo video and source code are available at:
this https URL


### [[2110.06863] Improving Users' Mental Model with Attention-directed Counterfactual Edits](http://arxiv.org/abs/2110.06863)


  In the domain of Visual Question Answering (VQA), studies have shown
improvement in users' mental model of the VQA system when they are exposed to
examples of how these systems answer certain Image-Question (IQ) pairs. In this
work, we show that showing controlled counterfactual image-question examples
are more effective at improving the mental model of users as compared to simply
showing random examples. We compare a generative approach and a retrieval-based
approach to show counterfactual examples. We use recent advances in generative
adversarial networks (GANs) to generate counterfactual images by deleting and
inpainting certain regions of interest in the image. We then expose users to
changes in the VQA system's answer on those altered images. To select the
region of interest for inpainting, we experiment with using both
human-annotated attention maps and a fully automatic method that uses the VQA
system's attention values. Finally, we test the user's mental model by asking
them to predict the model's performance on a test counterfactual image. We note
an overall improvement in users' accuracy to predict answer change when shown
counterfactual explanations. While realistic retrieved counterfactuals
obviously are the most effective at improving the mental model, we show that a
generative approach can also be equally effective.

    

### [[2110.06884] ConditionalQA: A Complex Reading Comprehension Dataset with Conditional Answers](http://arxiv.org/abs/2110.06884)


  We describe a Question Answering (QA) dataset that contains complex questions
with conditional answers, i.e. the answers are only applicable when certain
conditions apply. We call this dataset ConditionalQA. In addition to
conditional answers, the dataset also features: (1) long context documents with
information that is related in logically complex ways; (2) multi-hop questions
that require compositional logical reasoning; (3) a combination of extractive
questions, yes/no questions, questions with multiple answers, and
not-answerable questions; (4) questions asked without knowing the answers. We
show that ConditionalQA is challenging for many of the existing QA models,
especially in selecting answer conditions. We believe that this dataset will
motivate further research in answering complex questions over long documents.
Data and leaderboard are publicly available at
\url{this https URL}.

    

### [[2110.06898] Representing Matrices Using Algebraic ZX-calculus](http://arxiv.org/abs/2110.06898)


  Elementary matrices play an important role in linear algebra applications. In
this paper, we represent all the elementary matrices of size 2^m\times 2^m
using algebraic ZX-calculus. Then we show their properties on inverses and
transpose using rewriting rules of ZX-calculus. As a consequence, we are able
to depict any matrices of size 2^m\times 2^n by string diagrams without resort
to a diagrammatic normal form for matrices as shown in [Wang 2020]. By doing so
we pave the way towards visualising by string diagrams important matrix
technologies deployed in AI especially machine learning.

    

### [[2110.06904] Traceback of Data Poisoning Attacks in Neural Networks](http://arxiv.org/abs/2110.06904)


  In adversarial machine learning, new defenses against attacks on deep
learning systems are routinely broken soon after their release by more powerful
attacks. In this context, forensic tools can offer a valuable complement to
existing defenses, by tracing back a successful attack to its root cause, and
offering a path forward for mitigation to prevent similar attacks in the
future.
In this paper, we describe our efforts in developing a forensic traceback
tool for poison attacks on deep neural networks. We propose a novel iterative
clustering and pruning solution that trims "innocent" training samples, until
all that remains is the set of poisoned data responsible for the attack. Our
method clusters training samples based on their impact on model parameters,
then uses an efficient data unlearning method to prune innocent clusters. We
empirically demonstrate the efficacy of our system on three types of
dirty-label (backdoor) poison attacks and three types of clean-label poison
attacks, across domains of computer vision and malware classification. Our
system achieves over 98.4% precision and 96.8% recall across all attacks. We
also show that our system is robust against four anti-forensics measures
specifically designed to attack it.

    

### [[2103.03125] Advances in Multi-turn Dialogue Comprehension: A Survey](http://arxiv.org/abs/2103.03125)


  Training machines to understand natural language and interact with humans is
an elusive and essential task of artificial intelligence. A diversity of
dialogue systems has been designed with the rapid development of deep learning
techniques, especially the recent pre-trained language models (PrLMs). Among
these studies, the fundamental yet challenging type of task is dialogue
comprehension whose role is to teach the machines to read and comprehend the
dialogue context before responding. In this paper, we review the previous
methods from the technical perspective of dialogue modeling for the dialogue
comprehension task. We summarize the characteristics and challenges of dialogue
comprehension in contrast to plain-text reading comprehension. Then, we discuss
three typical patterns of dialogue modeling. In addition, we categorize
dialogue-related pre-training techniques which are employed to enhance PrLMs in
dialogue scenarios. Finally, we highlight the technical advances in recent
years and point out the lessons from the empirical analysis and the prospects
towards a new frontier of researches.

    

### [[2105.02544] SGG: Learning to Select, Guide, and Generate for Keyphrase Generation](http://arxiv.org/abs/2105.02544)


  Keyphrases, that concisely summarize the high-level topics discussed in a
document, can be categorized into present keyphrase which explicitly appears in
the source text, and absent keyphrase which does not match any contiguous
subsequence but is highly semantically related to the source. Most existing
keyphrase generation approaches synchronously generate present and absent
keyphrases without explicitly distinguishing these two categories. In this
paper, a Select-Guide-Generate (SGG) approach is proposed to deal with present
and absent keyphrase generation separately with different mechanisms.
Specifically, SGG is a hierarchical neural network which consists of a
pointing-based selector at low layer concentrated on present keyphrase
generation, a selection-guided generator at high layer dedicated to absent
keyphrase generation, and a guider in the middle to transfer information from
selector to generator. Experimental results on four keyphrase generation
benchmarks demonstrate the effectiveness of our model, which significantly
outperforms the strong baselines for both present and absent keyphrases
generation. Furthermore, we extend SGG to a title generation task which
indicates its extensibility in natural language generation tasks.

    

### [[2105.07508] Abstraction, Validation, and Generalization for Explainable Artificial Intelligence](http://arxiv.org/abs/2105.07508)


  Neural network architectures are achieving superhuman performance on an
expanding range of tasks. To effectively and safely deploy these systems, their
decision-making must be understandable to a wide range of stakeholders. Methods
to explain AI have been proposed to answer this challenge, but a lack of theory
impedes the development of systematic abstractions which are necessary for
cumulative knowledge gains. We propose Bayesian Teaching as a framework for
unifying explainable AI (XAI) by integrating machine learning and human
learning. Bayesian Teaching formalizes explanation as a communication act of an
explainer to shift the beliefs of an explainee. This formalization decomposes
any XAI method into four components: (1) the inference to be explained, (2) the
explanatory medium, (3) the explainee model, and (4) the explainer model. The
abstraction afforded by Bayesian Teaching to decompose any XAI method
elucidates the invariances among them. The decomposition of XAI systems enables
modular validation, as each of the first three components listed can be tested
semi-independently. This decomposition also promotes generalization through
recombination of components from different XAI systems, which facilitates the
generation of novel variants. These new variants need not be evaluated one by
one provided that each component has been validated, leading to an exponential
decrease in development time. Finally, by making the goal of explanation
explicit, Bayesian Teaching helps developers to assess how suitable an XAI
system is for its intended real-world use case. Thus, Bayesian Teaching
provides a theoretical framework that encourages systematic, scientific
investigation of XAI.

    

### [[2105.12954] Better Regularization for Sequential Decision Spaces: Fast Convergence Rates for Nash, Correlated, and Team Equilibria](http://arxiv.org/abs/2105.12954)


  We study the application of iterative first-order methods to the problem of
computing equilibria of large-scale two-player extensive-form games.
First-order methods must typically be instantiated with a regularizer that
serves as a distance-generating function for the decision sets of the players.
For the case of two-player zero-sum games, the state-of-the-art theoretical
convergence rate for Nash equilibrium is achieved by using the dilated entropy
function. In this paper, we introduce a new entropy-based distance-generating
function for two-player zero-sum games, and show that this function achieves
significantly better strong convexity properties than the dilated entropy,
while maintaining the same easily-implemented closed-form proximal mapping.
Extensive numerical simulations show that these superior theoretical properties
translate into better numerical performance as well. We then generalize our new
entropy distance function, as well as general dilated distance functions, to
the scaled extension operator. The scaled extension operator is a way to
recursively construct convex sets, which generalizes the decision polytope of
extensive-form games, as well as the convex polytopes corresponding to
correlated and team equilibria. By instantiating first-order methods with our
regularizers, we develop the first accelerated first-order methods for
computing correlated equilibra and ex-ante coordinated team equilibria. Our
methods have a guaranteed $1/T$ rate of convergence, along with linear-time
proximal updates.

    

### [[2109.09425] Clustering in Recurrent Neural Networks for Micro-Segmentation using Spending Personality](http://arxiv.org/abs/2109.09425)


  Customer segmentation has long been a productive field in banking. However,
with new approaches to traditional problems come new opportunities.
Fine-grained customer segments are notoriously elusive and one method of
obtaining them is through feature extraction. It is possible to assign
coefficients of standard personality traits to financial transaction classes
aggregated over time. However, we have found that the clusters formed are not
sufficiently discriminatory for micro-segmentation. In a novel approach, we
extract temporal features with continuous values from the hidden states of
neural networks predicting customers' spending personality from their financial
transactions. We consider both temporal and non-sequential models, using long
short-term memory (LSTM) and feed-forward neural networks, respectively. We
found that recurrent neural networks produce micro-segments where feed-forward
networks produce only coarse segments. Finally, we show that classification
using these extracted features performs at least as well as bespoke models on
two common metrics, namely loan default rate and customer liquidity index.

    

### [[2110.06326] When Does the Gittins Policy Have Asymptotically Optimal Response Time Tail?](http://arxiv.org/abs/2110.06326)


  We consider scheduling in the M/G/1 queue with unknown job sizes. It is known
that the Gittins policy minimizes mean response time in this setting. However,
the behavior of the tail of response time under Gittins is poorly understood,
even in the large-response-time limit. Characterizing Gittins's asymptotic tail
behavior is important because if Gittins has optimal tail asymptotics, then it
simultaneously provides optimal mean response time and good tail performance.
In this work, we give the first comprehensive account of Gittins's asymptotic
tail behavior. For heavy-tailed job sizes, we find that Gittins always has
asymptotically optimal tail. The story for light-tailed job sizes is less
clear-cut: Gittins's tail can be optimal, pessimal, or in between. To remedy
this, we show that a modification of Gittins avoids pessimal tail behavior
while achieving near-optimal mean response time.

    

### [[2110.06303] Orion: Automatic Repair for Network Programs](http://arxiv.org/abs/2110.06303)


  Debugging imperative network programs is a challenging task for developers
because understanding various network modules and complicated data structures
is typically time-consuming. To address the challenge, this paper presents an
automated technique for repairing network programs from unit tests.
Specifically, given as input a faulty network program and a set of unit tests,
our approach localizes the fault through symbolic reasoning, and synthesizes a
patch such that the repaired program can pass all unit tests. It applies
domain-specific abstraction to simplify network data structures and utilizes
modular analysis to facilitate function summary reuse for symbolic analysis. We
implement the proposed techniques in a tool called Orion and evaluate it on 10
benchmarks adapted from real-world software-defined networking controllers. The
evaluation results demonstrate the effectiveness and efficiency of Orion for
repairing network programs.

    

### [[2008.07185] CROW: Code Diversification for WebAssembly](http://arxiv.org/abs/2008.07185)


  The adoption of WebAssembly has rapidly increased in the last few years as it
provides a fast and safe model for program execution. However, WebAssembly is
not exempt from vulnerabilities that could be exploited by side channels
attacks. This class of vulnerabilities that can be addressed by code
diversification. In this paper, we present the first fully automated workflow
for the diversification of WebAssembly binaries. We present CROW, an
open-source tool implementing this workflow. We evaluate CROW's capabilities on
303 C programs and study its use on a real-life security-sensitive program:
libsodium, a cryptographic library. Overall, CROWis able to generate diverse
variants for 239 out of 303,(79%) small programs. Furthermore, our experiments
show that our approach and tool is able to successfully diversify off-the-shelf
cryptographic software (libsodium).

    

### [[2009.04909] Disjunctive Delimited Control](http://arxiv.org/abs/2009.04909)


  Delimited control is a powerful mechanism for programming language extension
which has been recently proposed for Prolog (and implemented in SWI-Prolog). By
manipulating the control flow of a program from inside the language, it enables
the implementation of powerful features, such as tabling, without modifying the
internals of the Prolog engine. However, its current formulation is inadequate:
it does not capture Prolog's unique non-deterministic nature which allows
multiple ways to satisfy a goal. This paper fully embraces Prolog's
non-determinism with a novel interface for disjunctive delimited control, which
gives the programmer not only control over the sequential (conjunctive) control
flow, but also over the non-deterministic control flow. We provide a
meta-interpreter that conservatively extends Prolog with delimited control and
show that it enables a range of typical Prolog features and extensions, now at
the library level: findall, cut, branch-and-bound optimisation, probabilistic
programming, . . .

    