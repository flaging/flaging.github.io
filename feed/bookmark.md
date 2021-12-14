
## 2021-12-14

### [<title>Parameters: { scale_post_weight } might not be used - XGBoost</title>](https://discuss.xgboost.ai/t/parameters-scale-post-weight-might-not-be-used/2597/2)

### [<title data-react-helmet="true">NeurIPS 2021: 多篇顶会文章看UDA新进展 - 知乎</title>](https://zhuanlan.zhihu.com/p/316265317)

### [<title data-react-helmet="true">状态空间法(卡尔曼滤波)解决深度高斯过程问题-札记1 - 知乎</title>](https://zhuanlan.zhihu.com/p/438503498)

### [[2112.05823] A General "Power-of-d'' Dispatching Framework for Heterogeneous Systems](http://arxiv.org/abs/2112.05823)


  Intelligent dispatching is crucial to obtaining low response times in
large-scale systems. One common scalable dispatching paradigm is the
``power-of-$d$,'' in which the dispatcher queries $d$ servers at random and
assigns the job to a server based only on the state of the queried servers. The
bulk of power-of-$d$ policies studied in the literature assume that the system
is homogeneous, meaning that all servers have the same speed; meanwhile
real-world systems often exhibit server speed heterogeneity.
This paper introduces a general framework for describing and analyzing
heterogeneity-aware power-of-$d$ policies. The key idea behind our framework is
that dispatching policies can make use of server speed information at two
decision points: when choosing which $d$ servers to query, and when assigning a
job to one of those servers. Our framework explicitly separates the dispatching
policy into a querying rule and an assignment rule; we consider general
families of both rule types.
While the strongest assignment rules incorporate both detailed queue-length
information and server speed information, these rules typically are difficult
to analyze. We overcome this difficulty by focusing on heterogeneity-aware
assignment rules that ignore queue length information beyond idleness status.
In this setting, we analyze mean response time and formulate novel optimization
problems for the joint optimization of querying and assignment. We build upon
our optimized policies to develop heuristic queue length-aware dispatching
policies. Our heuristic policies perform well in simulation, relative to
policies that have appeared in the literature.

    

### [[2112.06095] Unlocking the Power of Inline Floating-Point Operations on Programmable Switches](http://arxiv.org/abs/2112.06095)


  The advent of switches with programmable dataplanes has enabled the rapid
development of new network functionality, as well as providing a platform for
acceleration of a broad range of application-level functionality. However,
existing switch hardware was not designed with application acceleration in
mind, and thus applications requiring operations or datatypes not used in
traditional network protocols must resort to expensive workarounds.
Applications involving floating point data, including distributed training for
machine learning and distributed query processing, are key examples.
In this paper, we propose FPISA, a floating point representation designed to
work efficiently in programmable switches. We first implement FPISA on an Intel
Tofino switch, but find that it has limitations that impact throughput and
accuracy. We then propose hardware changes to address these limitations based
on the open-source Banzai switch architecture, and synthesize them in a 15-nm
standard-cell library to demonstrate their feasibility. Finally, we use FPISA
to implement accelerators for training for machine learning and for query
processing, and evaluate their performance on a switch implementing our changes
using emulation. We find that FPISA allows distributed training to use 25-75%
fewer CPU cores and provide up to 85.9% better throughput in a CPU-constrained
environment than SwitchML. For distributed query processing with floating point
data, FPISA enables up to 2.7x better throughput than Spark.

    

### [[2112.06224] Joint Sensing, Communication, and Computation Resource Allocation for Cooperative Perception in Fog-Based Vehicular Networks](http://arxiv.org/abs/2112.06224)


  To enlarge the perception range and reliability of individual autonomous
vehicles, cooperative perception has been received much attention. However,
considering the high volume of shared messages, limited bandwidth and
computation resources in vehicular networks become bottlenecks. In this paper,
we investigate how to balance the volume of shared messages and constrained
resources in fog-based vehicular networks. To this end, we first characterize
sum satisfaction of cooperative perception taking account of its
spatial-temporal value and latency performance. Next, the sensing block
message, communication resource block, and computation resource are jointly
allocated to maximize the sum satisfaction of cooperative perception, while
satisfying the maximum latency and sojourn time constraints of vehicles. Owing
to its non-convexity, we decouple the original problem into two separate
sub-problems and devise corresponding solutions. Simulation results demonstrate
that our proposed scheme can effectively boost the sum satisfaction of
cooperative perception compared with existing baselines.

    

### [[1402.3985] Challenges and issues in 4G Networks Mobility Management](http://arxiv.org/abs/1402.3985)


  Wireless broadband technology is now in motion to provide higher data rate,
wider coverage and improved mobility. Towards this the 4G - network is an
integration of various wireless technologies and expected to provide seamless
mobility. Moreover 4G-networks will be entirely packet switched systems based
on IP protocol. One of the research challenges for 4G-Network is the design of
intelligent mobility management techniques that take advantage of IP-based
technologies to achieve global roaming among various access technologies. Hence
Mobile IPv6 is considered to be one of the key technologies for integration of
heterogeneous networks. However the original Mobile IPv6 does not support fast
handover, which is essential function for mobile networks. Number of research
groups working towards this to develop a common protocol to enable seamless
mobility. In this paper we identify and explore the different issues and
challenges related to mobility management in 4G - networks.

    

### [[1705.07524] EDOS: Edge Assisted Offloading System for Mobile Devices](http://arxiv.org/abs/1705.07524)


  Offloading resource-intensive jobs to the cloud and nearby users is a
promising approach to enhance mobile devices. This paper investigates a hybrid
offloading system that takes both infrastructure-based networks and Ad-hoc
networks into the scope. Specifically, we propose EDOS, an edge assisted
offloading system that consists of two major components, an Edge Assistant (EA)
and Offload Agent (OA). EA runs on the routers/towers to manage registered
remote cloud servers and local service providers and OA operates on the users'
devices to discover the services in proximity. We present the system with a
suite of protocols to collect the potential service providers and algorithms to
allocate tasks according to user-specified constraints. To evaluate EDOS, we
prototype it on commercial mobile devices and evaluate it with both experiments
on a small-scale testbed and simulations. The results show that EDOS is
effective and efficient for offloading jobs.

    

### [[2112.05753] Using Machine Learning to Predict Air Quality Index in New Delhi](http://arxiv.org/abs/2112.05753)


  Air quality has a significant impact on human health. Degradation in air
quality leads to a wide range of health issues, especially in children. The
ability to predict air quality enables the government and other concerned
organizations to take necessary steps to shield the most vulnerable, from being
exposed to the air with hazardous quality. Traditional approaches to this task
have very limited success because of a lack of access of such methods to
sufficient longitudinal data. In this paper, we use a Support Vector Regression
(SVR) model to forecast the levels of various pollutants and the air quality
index, using archive pollution data made publicly available by Central
Pollution Control Board and the US Embassy in New Delhi. Among the tested
methods, a Radial Basis Function (RBF) kernel produced the best results with
SVR. According to our experiments, using the whole range of available variables
produced better results than using features selected by principal component
analysis. The model predicts levels of various pollutants, like, sulfur
dioxide, carbon monoxide, nitrogen dioxide, particulate matter 2.5, and
ground-level ozone, as well as the Air Quality Index (AQI), at an accuracy of
93.4 percent.

    

### [[2112.05758] Edge-Enhanced Dual Discriminator Generative Adversarial Network for Fast MRI with Parallel Imaging Using Multi-view Information](http://arxiv.org/abs/2112.05758)


  In clinical medicine, magnetic resonance imaging (MRI) is one of the most
important tools for diagnosis, triage, prognosis, and treatment planning.
However, MRI suffers from an inherent slow data acquisition process because
data is collected sequentially in k-space. In recent years, most MRI
reconstruction methods proposed in the literature focus on holistic image
reconstruction rather than enhancing the edge information. This work steps
aside this general trend by elaborating on the enhancement of edge information.
Specifically, we introduce a novel parallel imaging coupled dual discriminator
generative adversarial network (PIDD-GAN) for fast multi-channel MRI
reconstruction by incorporating multi-view information. The dual discriminator
design aims to improve the edge information in MRI reconstruction. One
discriminator is used for holistic image reconstruction, whereas the other one
is responsible for enhancing edge information. An improved U-Net with local and
global residual learning is proposed for the generator. Frequency channel
attention blocks (FCA Blocks) are embedded in the generator for incorporating
attention mechanisms. Content loss is introduced to train the generator for
better reconstruction quality. We performed comprehensive experiments on
Calgary-Campinas public brain MR dataset and compared our method with
state-of-the-art MRI reconstruction methods. Ablation studies of residual
learning were conducted on the MICCAI13 dataset to validate the proposed
modules. Results show that our PIDD-GAN provides high-quality reconstructed MR
images, with well-preserved edge information. The time of single-image
reconstruction is below 5ms, which meets the demand of faster processing.

    

### [[2112.05760] Learning Representations with Contrastive Self-Supervised Learning for Histopathology Applications](http://arxiv.org/abs/2112.05760)


  Unsupervised learning has made substantial progress over the last few years,
especially by means of contrastive self-supervised learning. The dominating
dataset for benchmarking self-supervised learning has been ImageNet, for which
recent methods are approaching the performance achieved by fully supervised
training. The ImageNet dataset is however largely object-centric, and it is not
clear yet what potential those methods have on widely different datasets and
tasks that are not object-centric, such as in digital pathology. While
self-supervised learning has started to be explored within this area with
encouraging results, there is reason to look closer at how this setting differs
from natural images and ImageNet. In this paper we make an in-depth analysis of
contrastive learning for histopathology, pin-pointing how the contrastive
objective will behave differently due to the characteristics of histopathology
data. We bring forward a number of considerations, such as view generation for
the contrastive objective and hyper-parameter tuning. In a large battery of
experiments, we analyze how the downstream performance in tissue classification
will be affected by these considerations. The results point to how contrastive
learning can reduce the annotation effort within digital pathology, but that
the specific dataset characteristics need to be considered. To take full
advantage of the contrastive learning objective, different calibrations of view
generation and hyper-parameters are required. Our results pave the way for
realizing the full potential of self-supervised learning for histopathology
applications.

    

### [[2112.05761] Pre-training and Fine-tuning Transformers for fMRI Prediction Tasks](http://arxiv.org/abs/2112.05761)


  We present the TFF Transformer framework for the analysis of functional
Magnetic Resonance Imaging (fMRI) data. TFF employs a transformer-based
architecture and a two-phase training approach. First, self-supervised training
is applied to a collection of fMRI scans, where the model is trained for the
reconstruction of 3D volume data. Second, the pre-trained model is fine-tuned
on specific tasks, utilizing ground truth labels. Our results show
state-of-the-art performance on a variety of fMRI tasks, including age and
gender prediction, as well as schizophrenia recognition.

    

### [[2112.05779] Quantum Architecture Search via Continual Reinforcement Learning](http://arxiv.org/abs/2112.05779)


  Quantum computing has promised significant improvement in solving difficult
computational tasks over classical computers. Designing quantum circuits for
practical use, however, is not a trivial objective and requires expert-level
knowledge. To aid this endeavor, this paper proposes a machine learning-based
method to construct quantum circuit architectures. Previous works have
demonstrated that classical deep reinforcement learning (DRL) algorithms can
successfully construct quantum circuit architectures without encoded physics
knowledge. However, these DRL-based works are not generalizable to settings
with changing device noises, thus requiring considerable amounts of training
resources to keep the RL models up-to-date. With this in mind, we incorporated
continual learning to enhance the performance of our algorithm. In this paper,
we present the Probabilistic Policy Reuse with deep Q-learning (PPR-DQL)
framework to tackle this circuit design challenge. By conducting numerical
simulations over various noise patterns, we demonstrate that the RL agent with
PPR was able to find the quantum gate sequence to generate the two-qubit Bell
state faster than the agent that was trained from scratch. The proposed
framework is general and can be applied to other quantum gate synthesis or
control problems -- including the automatic calibration of quantum devices.

    

### [[2112.05785] TempoQR: Temporal Question Reasoning over Knowledge Graphs](http://arxiv.org/abs/2112.05785)


  Knowledge Graph Question Answering (KGQA) involves retrieving facts from a
Knowledge Graph (KG) using natural language queries. A KG is a curated set of
facts consisting of entities linked by relations. Certain facts include also
temporal information forming a Temporal KG (TKG). Although many natural
questions involve explicit or implicit time constraints, question answering
(QA) over TKGs has been a relatively unexplored area. Existing solutions are
mainly designed for simple temporal questions that can be answered directly by
a single TKG fact. This paper puts forth a comprehensive embedding-based
framework for answering complex questions over TKGs. Our method termed temporal
question reasoning (TempoQR) exploits TKG embeddings to ground the question to
the specific entities and time scope it refers to. It does so by augmenting the
question embeddings with context, entity and time-aware information by
employing three specialized modules. The first computes a textual
representation of a given question, the second combines it with the entity
embeddings for entities involved in the question, and the third generates
question-specific time embeddings. Finally, a transformer-based encoder learns
to fuse the generated temporal information with the question representation,
which is used for answer predictions. Extensive experiments show that TempoQR
improves accuracy by 25--45 percentage points on complex temporal questions
over state-of-the-art approaches and it generalizes better to unseen question
types.

    

### [[2112.05807] Computer-Assisted Creation of Boolean Search Rules for Text Classification in the Legal Domain](http://arxiv.org/abs/2112.05807)


  In this paper, we present a method of building strong, explainable
classifiers in the form of Boolean search rules. We developed an interactive
environment called CASE (Computer Assisted Semantic Exploration) which exploits
word co-occurrence to guide human annotators in selection of relevant search
terms. The system seamlessly facilitates iterative evaluation and improvement
of the classification rules. The process enables the human annotators to
leverage the benefits of statistical information while incorporating their
expert intuition into the creation of such rules. We evaluate classifiers
created with our CASE system on 4 datasets, and compare the results to machine
learning methods, including SKOPE rules, Random forest, Support Vector Machine,
and fastText classifiers. The results drive the discussion on trade-offs
between superior compactness, simplicity, and intuitiveness of the Boolean
search rules versus the better performance of state-of-the-art machine learning
models for text classification.

    

### [[2112.05812] Edge-Compatible Reinforcement Learning for Recommendations](http://arxiv.org/abs/2112.05812)


  Most reinforcement learning (RL) recommendation systems designed for edge
computing must either synchronize during recommendation selection or depend on
an unprincipled patchwork collection of algorithms. In this work, we build on
asynchronous coagent policy gradient algorithms \citep{kostas2020asynchronous}
to propose a principled solution to this problem. The class of algorithms that
we propose can be distributed over the internet and run asynchronously and in
real-time. When a given edge fails to respond to a request for data with
sufficient speed, this is not a problem; the algorithm is designed to function
and learn in the edge setting, and network issues are part of this setting. The
result is a principled, theoretically grounded RL algorithm designed to be
distributed in and learn in this asynchronous environment. In this work, we
describe this algorithm and a proposed class of architectures in detail, and
demonstrate that they work well in practice in the asynchronous setting, even
as the network quality degrades.

    

### [[2112.05816] Encoding priors in the brain: a reinforcement learning model for mouse decision making](http://arxiv.org/abs/2112.05816)


  In two-alternative forced choice tasks, prior knowledge can improve
performance, especially when operating near the psychophysical threshold. For
instance, if subjects know that one choice is much more likely than the other,
they can make that choice when evidence is weak. A common hypothesis for these
kinds of tasks is that the prior is stored in neural activity. Here we propose
a different hypothesis: the prior is stored in synaptic strengths. We study the
International Brain Laboratory task, in which a grating appears on either the
right or left side of a screen, and a mouse has to move a wheel to bring the
grating to the center. The grating is often low in contrast which makes the
task relatively difficult, and the prior probability that the grating appears
on the right is either 80% or 20%, in (unsignaled) blocks of about 50 trials.
We model this as a reinforcement learning task, using a feedforward neural
network to map states to actions, and adjust the weights of the network to
maximize reward, learning via policy gradient. Our model uses an internal state
that stores an estimate of the grating and confidence, and follows Bayesian
updates, and can switch between engaged and disengaged states to mimic animal
behavior. This model reproduces the main experimental finding - that the
psychometric curve with respect to contrast shifts after a block switch in
about 10 trials. Also, as seen in the experiments, in our model the difference
in neuronal activity in the right and left blocks is small - it is virtually
impossible to decode block structure from activity on single trials if noise is
about 2%. The hypothesis that priors are stored in weights is difficult to
test, but the technology to do so should be available in the not so distant
future.

    

### [[2112.05820] Building a great multi-lingual teacher with sparsely-gated mixture of experts for speech recognition](http://arxiv.org/abs/2112.05820)


  The sparsely-gated Mixture of Experts (MoE) can magnify a network capacity
with a little computational complexity. In this work, we investigate how
multi-lingual Automatic Speech Recognition (ASR) networks can be scaled up with
a simple routing algorithm in order to achieve better accuracy. More
specifically, we apply the sparsely-gated MoE technique to two types of
networks: Sequence-to-Sequence Transformer (S2S-T) and Transformer Transducer
(T-T). We demonstrate through a set of ASR experiments on multiple language
data that the MoE networks can reduce the relative word error rates by 16.5\%
and 4.7\% with the S2S-T and T-T, respectively. Moreover, we thoroughly
investigate the effect of the MoE on the T-T architecture in various
conditions: streaming mode, non-streaming mode, the use of language ID and the
label decoder with the MoE.

    

### [[2112.05826] Sequence-level self-learning with multiple hypotheses](http://arxiv.org/abs/2112.05826)


  In this work, we develop new self-learning techniques with an attention-based
sequence-to-sequence (seq2seq) model for automatic speech recognition (ASR).
For untranscribed speech data, the hypothesis from an ASR system must be used
as a label. However, the imperfect ASR result makes unsupervised learning
difficult to consistently improve recognition performance especially in the
case that multiple powerful teacher models are unavailable. In contrast to
conventional unsupervised learning approaches, we adopt the \emph{multi-task
learning} (MTL) framework where the $n$-th best ASR hypothesis is used as the
label of each task. The seq2seq network is updated through the MTL framework so
as to find the common representation that can cover multiple hypotheses. By
doing so, the effect of the \emph{hard-decision} errors can be alleviated.
We first demonstrate the effectiveness of our self-learning methods through
ASR experiments in an accent adaptation task between the US and British English
speech. Our experiment results show that our method can reduce the WER on the
British speech data from 14.55\% to 10.36\% compared to the baseline model
trained with the US English data only. Moreover, we investigate the effect of
our proposed methods in a federated learning scenario.

    

### [[2112.05827] Quality-Aware Multimodal Biometric Recognition](http://arxiv.org/abs/2112.05827)


  We present a quality-aware multimodal recognition framework that combines
representations from multiple biometric traits with varying quality and number
of samples to achieve increased recognition accuracy by extracting
complimentary identification information based on the quality of the samples.
We develop a quality-aware framework for fusing representations of input
modalities by weighting their importance using quality scores estimated in a
weakly-supervised fashion. This framework utilizes two fusion blocks, each
represented by a set of quality-aware and aggregation networks. In addition to
architecture modifications, we propose two task-specific loss functions:
multimodal separability loss and multimodal compactness loss. The first loss
assures that the representations of modalities for a class have comparable
magnitudes to provide a better quality estimation, while the multimodal
representations of different classes are distributed to achieve maximum
discrimination in the embedding space. The second loss, which is considered to
regularize the network weights, improves the generalization performance by
regularizing the framework. We evaluate the performance by considering three
multimodal datasets consisting of face, iris, and fingerprint modalities. The
efficacy of the framework is demonstrated through comparison with the
state-of-the-art algorithms. In particular, our framework outperforms the rank-
and score-level fusion of modalities of BIOMDATA by more than 30% for true
acceptance rate at false acceptance rate of $10^{-4}$.

    

### [[2112.05837] Learning distributed channel access policies for networked estimation: data-driven optimization in the mean-field regime](http://arxiv.org/abs/2112.05837)


  The problem of communicating sensor measurements over shared networks is
prevalent in many modern large-scale distributed systems such as cyber-physical
systems, wireless sensor networks, and the internet of things. Due to bandwidth
constraints, the system designer must jointly design decentralized medium
access transmission and estimation policies that accommodate a very large
number of devices in extremely contested environments such that the collection
of all observations is reproduced at the destination with the best possible
fidelity. We formulate a remote estimation problem in the mean-field regime
where a very large number of sensors communicate their observations to an
access point, or base station, under a strict constraint on the maximum
fraction of transmitting devices. We show that in the mean-field regime, this
problem exhibits a structure that enables tractable optimization algorithms.
More importantly, we obtain a data-driven learning scheme that admits a finite
sample-complexity guarantee on the performance of the resulting estimation
system under minimal assumptions on the data's probability density function.

    

### [[2112.05841] Logical Boltzmann Machines](http://arxiv.org/abs/2112.05841)


  The idea of representing symbolic knowledge in connectionist systems has been
a long-standing endeavour which has attracted much attention recently with the
objective of combining machine learning and scalable sound reasoning. Early
work has shown a correspondence between propositional logic and symmetrical
neural networks which nevertheless did not scale well with the number of
variables and whose training regime was inefficient. In this paper, we
introduce Logical Boltzmann Machines (LBM), a neurosymbolic system that can
represent any propositional logic formula in strict disjunctive normal form. We
prove equivalence between energy minimization in LBM and logical satisfiability
thus showing that LBM is capable of sound reasoning. We evaluate reasoning
empirically to show that LBM is capable of finding all satisfying assignments
of a class of logical formulae by searching fewer than 0.75% of the possible
(approximately 1 billion) assignments. We compare learning in LBM with a
symbolic inductive logic programming system, a state-of-the-art neurosymbolic
system and a purely neural network-based system, achieving better learning
performance in five out of seven data sets.

    

### [[2112.05842] Revisiting the Boundary between ASR and NLU in the Age of Conversational Dialog Systems](http://arxiv.org/abs/2112.05842)


  As more users across the world are interacting with dialog agents in their
daily life, there is a need for better speech understanding that calls for
renewed attention to the dynamics between research in automatic speech
recognition (ASR) and natural language understanding (NLU). We briefly review
these research areas and lay out the current relationship between them. In
light of the observations we make in this paper, we argue that (1) NLU should
be cognizant of the presence of ASR models being used upstream in a dialog
system's pipeline, (2) ASR should be able to learn from errors found in NLU,
(3) there is a need for end-to-end datasets that provide semantic annotations
on spoken input, (4) there should be stronger collaboration between ASR and NLU
research communities.

    

### [[2112.05847] A Novel Gaussian Process Based Ground Segmentation Algorithm with Local-Smoothness Estimation](http://arxiv.org/abs/2112.05847)


  Autonomous Land Vehicles (ALV) shall efficiently recognize the ground in
unknown environments. A novel $\mathcal{GP}$-based method is proposed for the
ground segmentation task in rough driving scenarios. A non-stationary
covariance function is utilized as the kernel for the $\mathcal{GP}$. The
ground surface behavior is assumed to only demonstrate local-smoothness. Thus,
point estimates of the kernel's length-scales are obtained. Thus, two Gaussian
processes are introduced to separately model the observation and local
characteristics of the data. While, the \textit{observation process} is used to
model the ground, the \textit{latent process} is put on length-scale values to
estimate point values of length-scales at each input location. Input locations
for this latent process are chosen in a physically-motivated procedure to
represent an intuition about ground condition. Furthermore, an intuitive guess
of length-scale value is represented by assuming the existence of hypothetical
surfaces in the environment that every bunch of data points may be assumed to
be resulted from measurements from this surfaces. Bayesian inference is
implemented using \textit{maximum a Posteriori} criterion. The log-marginal
likelihood function is assumed to be a multi-task objective function, to
represent a whole-frame unbiased view of the ground at each frame. Simulation
results shows the effectiveness of the proposed method even in an uneven, rough
scene which outperforms similar Gaussian process based ground segmentation
methods. While adjacent segments do not have similar ground structure in an
uneven scene, the proposed method gives an efficient ground estimation based on
a whole-frame viewpoint instead of just estimating segment-wise probable ground
surfaces.

    

### [[2112.05848] Deep Q-Network with Proximal Iteration](http://arxiv.org/abs/2112.05848)


  We employ Proximal Iteration for value-function optimization in reinforcement
learning. Proximal Iteration is a computationally efficient technique that
enables us to bias the optimization procedure towards more desirable solutions.
As a concrete application of Proximal Iteration in deep reinforcement learning,
we endow the objective function of the Deep Q-Network (DQN) agent with a
proximal term to ensure that the online-network component of DQN remains in the
vicinity of the target network. The resultant agent, which we call DQN with
Proximal Iteration, or DQNPro, exhibits significant improvements over the
original DQN on the Atari benchmark. Our results accentuate the power of
employing sound optimization techniques for deep reinforcement learning.

    

### [[2112.05863] Directed Speech Separation for Automatic Speech Recognition of Long Form Conversational Speech](http://arxiv.org/abs/2112.05863)


  Many of the recent advances in speech separation are primarily aimed at
synthetic mixtures of short audio utterances with high degrees of overlap.
These datasets significantly differ from the real conversational data and
hence, the models trained and evaluated on these datasets do not generalize to
real conversational scenarios. Another issue with using most of these models
for long form speech is the nondeterministic ordering of separated speech
segments due to either unsupervised clustering for time-frequency masks or
Permutation Invariant training (PIT) loss. This leads to difficulty in
accurately stitching homogenous speaker segments for downstream tasks like
Automatic Speech Recognition (ASR). In this paper, we propose a speaker
conditioned separator trained on speaker embeddings extracted directly from the
mixed signal. We train this model using a directed loss which regulates the
order of the separated segments. With this model, we achieve significant
improvements on Word error rate (WER) for real conversational data without the
need for an additional re-stitching step.

    

### [[2112.05872] SLOSH: Set LOcality Sensitive Hashing via Sliced-Wasserstein Embeddings](http://arxiv.org/abs/2112.05872)


  Learning from set-structured data is an essential problem with many
applications in machine learning and computer vision. This paper focuses on
non-parametric and data-independent learning from set-structured data using
approximate nearest neighbor (ANN) solutions, particularly locality-sensitive
hashing. We consider the problem of set retrieval from an input set query. Such
retrieval problem requires: 1) an efficient mechanism to calculate the
distances/dissimilarities between sets, and 2) an appropriate data structure
for fast nearest neighbor search. To that end, we propose Sliced-Wasserstein
set embedding as a computationally efficient "set-2-vector" mechanism that
enables downstream ANN, with theoretical guarantees. The set elements are
treated as samples from an unknown underlying distribution, and the
Sliced-Wasserstein distance is used to compare sets. We demonstrate the
effectiveness of our algorithm, denoted as Set-LOcality Sensitive Hashing
(SLOSH), on various set retrieval datasets and compare our proposed embedding
with standard set embedding approaches, including Generalized Mean (GeM)
embedding/pooling, Featurewise Sort Pooling (FSPool), and Covariance Pooling
and show consistent improvement in retrieval results. The code for replicating
our results is available here:
\href{this https URL}{this https URL}.

    

### [[2112.05876] The Past as a Stochastic Process](http://arxiv.org/abs/2112.05876)


  Historical processes manifest remarkable diversity. Nevertheless, scholars
have long attempted to identify patterns and categorize historical actors and
influences with some success. A stochastic process framework provides a
structured approach for the analysis of large historical datasets that allows
for detection of sometimes surprising patterns, identification of relevant
causal actors both endogenous and exogenous to the process, and comparison
between different historical cases. The combination of data, analytical tools
and the organizing theoretical framework of stochastic processes complements
traditional narrative approaches in history and archaeology.

    

### [[2112.05883] Self-supervised Spatiotemporal Representation Learning by Exploiting Video Continuity](http://arxiv.org/abs/2112.05883)


  Recent self-supervised video representation learning methods have found
significant success by exploring essential properties of videos, e.g. speed,
temporal order, etc. This work exploits an essential yet under-explored
property of videos, the \textit{video continuity}, to obtain supervision
signals for self-supervised representation learning. Specifically, we formulate
three novel continuity-related pretext tasks, i.e. continuity justification,
discontinuity localization, and missing section approximation, that jointly
supervise a shared backbone for video representation learning. This
self-supervision approach, termed as Continuity Perception Network (CPNet),
solves the three tasks altogether and encourages the backbone network to learn
local and long-ranged motion and context representations. It outperforms prior
arts on multiple downstream tasks, such as action recognition, video retrieval,
and action localization. Additionally, the video continuity can be
complementary to other coarse-grained video properties for representation
learning, and integrating the proposed pretext task to prior arts can yield
much performance gains.

    

### [[2112.05887] Distributed Graph Learning with Smooth Data Priors](http://arxiv.org/abs/2112.05887)


  Graph learning is often a necessary step in processing or representing
structured data, when the underlying graph is not given explicitly. Graph
learning is generally performed centrally with a full knowledge of the graph
signals, namely the data that lives on the graph nodes. However, there are
settings where data cannot be collected easily or only with a non-negligible
communication cost. In such cases, distributed processing appears as a natural
solution, where the data stays mostly local and all processing is performed
among neighbours nodes on the communication graph. We propose here a novel
distributed graph learning algorithm, which permits to infer a graph from
signal observations on the nodes under the assumption that the data is smooth
on the target graph. We solve a distributed optimization problem with local
projection constraints to infer a valid graph while limiting the communication
costs. Our results show that the distributed approach has a lower communication
cost than a centralised algorithm without compromising the accuracy in the
inferred graph. It also scales better in communication costs with the increase
of the network size, especially for sparse networks.

    

### [[2112.05888] A Sparse Expansion For Deep Gaussian Processes](http://arxiv.org/abs/2112.05888)


  Deep Gaussian Processes (DGP) enable a non-parametric approach to quantify
the uncertainty of complex deep machine learning models. Conventional
inferential methods for DGP models can suffer from high computational
complexity as they require large-scale operations with kernel matrices for
training and inference. In this work, we propose an efficient scheme for
accurate inference and prediction based on a range of Gaussian Processes,
called the Tensor Markov Gaussian Processes (TMGP). We construct an induced
approximation of TMGP referred to as the hierarchical expansion. Next, we
develop a deep TMGP (DTMGP) model as the composition of multiple hierarchical
expansion of TMGPs. The proposed DTMGP model has the following properties: (1)
the outputs of each activation function are deterministic while the weights are
chosen independently from standard Gaussian distribution; (2) in training or
prediction, only O(polylog(M)) (out of M) activation functions have non-zero
outputs, which significantly boosts the computational efficiency. Our numerical
experiments on real datasets show the superior computational efficiency of
DTMGP versus other DGP models.

    

### [[2112.05893] Hybrid Neural Networks for On-device Directional Hearing](http://arxiv.org/abs/2112.05893)


  On-device directional hearing requires audio source separation from a given
direction while achieving stringent human-imperceptible latency requirements.
While neural nets can achieve significantly better performance than traditional
beamformers, all existing models fall short of supporting low-latency causal
inference on computationally-constrained wearables. We present DeepBeam, a
hybrid model that combines traditional beamformers with a custom lightweight
neural net. The former reduces the computational burden of the latter and also
improves its generalizability, while the latter is designed to further reduce
the memory and computational overhead to enable real-time and low-latency
operations. Our evaluation shows comparable performance to state-of-the-art
causal inference models on synthetic data while achieving a 5x reduction of
model size, 4x reduction of computation per second, 5x reduction in processing
time and generalizing better to real hardware data. Further, our real-time
hybrid model runs in 8 ms on mobile CPUs designed for low-power wearable
devices and achieves an end-to-end latency of 17.5 ms.

    

### [[2112.05907] Smooth-Swap: A Simple Enhancement for Face-Swapping with Smoothness](http://arxiv.org/abs/2112.05907)


  In recent years, face-swapping models have progressed in generation quality
and drawn attention for their applications in privacy protection and
entertainment. However, their complex architectures and loss functions often
require careful tuning for successful training. In this paper, we propose a new
face-swapping model called `Smooth-Swap', which focuses on deriving the
smoothness of the identity embedding instead of employing complex handcrafted
designs. We postulate that the gist of the difficulty in face-swapping is
unstable gradients and it can be resolved by a smooth identity embedder.
Smooth-swap adopts an embedder trained using supervised contrastive learning,
where we find its improved smoothness allows faster and stable training even
with a simple U-Net-based generator and three basic loss functions. Extensive
experiments on face-swapping benchmarks (FFHQ, FaceForensics++) and face images
in the wild show that our model is also quantitatively and qualitatively
comparable or even superior to existing methods in terms of identity change.

    

### [[2112.05908] Federated Reinforcement Learning at the Edge](http://arxiv.org/abs/2112.05908)


  Modern cyber-physical architectures use data collected from systems at
different physical locations to learn appropriate behaviors and adapt to
uncertain environments. However, an important challenge arises as communication
exchanges at the edge of networked systems are costly due to limited resources.
This paper considers a setup where multiple agents need to communicate
efficiently in order to jointly solve a reinforcement learning problem over
time-series data collected in a distributed manner. This is posed as learning
an approximate value function over a communication network. An algorithm for
achieving communication efficiency is proposed, supported with theoretical
guarantees, practical implementations, and numerical evaluations. The approach
is based on the idea of communicating only when sufficiently informative data
is collected.

    

### [[2112.05909] Neural Attention Models in Deep Learning: Survey and Taxonomy](http://arxiv.org/abs/2112.05909)


  Attention is a state of arousal capable of dealing with limited processing
bottlenecks in human beings by focusing selectively on one piece of information
while ignoring other perceptible information. For decades, concepts and
functions of attention have been studied in philosophy, psychology,
neuroscience, and computing. Currently, this property has been widely explored
in deep neural networks. Many different neural attention models are now
available and have been a very active research area over the past six years.
From the theoretical standpoint of attention, this survey provides a critical
analysis of major neural attention models. Here we propose a taxonomy that
corroborates with theoretical aspects that predate Deep Learning. Our taxonomy
provides an organizational structure that asks new questions and structures the
understanding of existing attentional mechanisms. In particular, 17 criteria
derived from psychology and neuroscience classic studies are formulated for
qualitative comparison and critical analysis on the 51 main models found on a
set of more than 650 papers analyzed. Also, we highlight several theoretical
issues that have not yet been explored, including discussions about biological
plausibility, highlight current research trends, and provide insights for the
future.

    

### [[2112.05910] An Empirical Study on Relation Extraction in the Biomedical Domain](http://arxiv.org/abs/2112.05910)


  Relation extraction is a fundamental problem in natural language processing.
Most existing models are defined for relation extraction in the general domain.
However, their performance on specific domains (e.g., biomedicine) is yet
unclear. To fill this gap, this paper carries out an empirical study on
relation extraction in biomedical research articles. Specifically, we consider
both sentence-level and document-level relation extraction, and run a few
state-of-the-art methods on several benchmark datasets. Our results show that
(1) current document-level relation extraction methods have strong
generalization ability; (2) existing methods require a large amount of labeled
data for model fine-tuning in biomedicine. Our observations may inspire people
in this field to develop more effective models for biomedical relation
extraction.

    

### [[2112.05911] Learning Contraction Policies from Offline Data](http://arxiv.org/abs/2112.05911)


  This paper proposes a data-driven method for learning convergent control
policies from offline data using Contraction theory. Contraction theory enables
constructing a policy that makes the closed-loop system trajectories inherently
convergent towards a unique trajectory. At the technical level, identifying the
contraction metric, which is the distance metric with respect to which a
robot's trajectories exhibit contraction is often non-trivial. We propose to
jointly learn the control policy and its corresponding contraction metric while
enforcing contraction. To achieve this, we learn an implicit dynamics model of
the robotic system from an offline data set consisting of the robot's state and
input trajectories. Using this learned dynamics model, we propose a data
augmentation algorithm for learning contraction policies. We randomly generate
samples in the state-space and propagate them forward in time through the
learned dynamics model to generate auxiliary sample trajectories. We then learn
both the control policy and the contraction metric such that the distance
between the trajectories from the offline data set and our generated auxiliary
sample trajectories decreases over time. We evaluate the performance of our
proposed framework on simulated robotic goal-reaching tasks and demonstrate
that enforcing contraction results in faster convergence and greater robustness
of the learned policy.

    

### [[2112.05914] Leaping Through Time with Gradient-based Adaptation for Recommendation](http://arxiv.org/abs/2112.05914)


  Modern recommender systems are required to adapt to the change in user
preferences and item popularity. Such a problem is known as the temporal
dynamics problem, and it is one of the main challenges in recommender system
modeling. Different from the popular recurrent modeling approach, we propose a
new solution named LeapRec to the temporal dynamic problem by using
trajectory-based meta-learning to model time dependencies. LeapRec
characterizes temporal dynamics by two complement components named global time
leap (GTL) and ordered time leap (OTL). By design, GTL learns long-term
patterns by finding the shortest learning path across unordered temporal data.
Cooperatively, OTL learns short-term patterns by considering the sequential
nature of the temporal data. Our experimental results show that LeapRec
consistently outperforms the state-of-the-art methods on several datasets and
recommendation metrics. Furthermore, we provide an empirical study of the
interaction between GTL and OTL, showing the effects of long- and short-term
modeling.

    

### [[2112.05923] ElegantRL-Podracer: Scalable and Elastic Library for Cloud-Native Deep Reinforcement Learning](http://arxiv.org/abs/2112.05923)


  Deep reinforcement learning (DRL) has revolutionized learning and actuation
in applications such as game playing and robotic control. The cost of data
collection, i.e., generating transitions from agent-environment interactions,
remains a major challenge for wider DRL adoption in complex real-world
problems. Following a cloud-native paradigm to train DRL agents on a GPU cloud
platform is a promising solution. In this paper, we present a scalable and
elastic library ElegantRL-podracer for cloud-native deep reinforcement
learning, which efficiently supports millions of GPU cores to carry out
massively parallel training at multiple levels. At a high-level,
ElegantRL-podracer employs a tournament-based ensemble scheme to orchestrate
the training process on hundreds or even thousands of GPUs, scheduling the
interactions between a leaderboard and a training pool with hundreds of pods.
At a low-level, each pod simulates agent-environment interactions in parallel
by fully utilizing nearly 7,000 GPU CUDA cores in a single GPU. Our
ElegantRL-podracer library features high scalability, elasticity and
accessibility by following the development principles of containerization,
microservices and MLOps. Using an NVIDIA DGX SuperPOD cloud, we conduct
extensive experiments on various tasks in locomotion and stock trading and show
that ElegantRL-podracer substantially outperforms RLlib. Our codes are
available on GitHub.

    

### [[2112.05929] Server-Side Local Gradient Averaging and Learning Rate Acceleration for Scalable Split Learning](http://arxiv.org/abs/2112.05929)


  In recent years, there have been great advances in the field of decentralized
learning with private data. Federated learning (FL) and split learning (SL) are
two spearheads possessing their pros and cons, and are suited for many user
clients and large models, respectively. To enjoy both benefits, hybrid
approaches such as SplitFed have emerged of late, yet their fundamentals have
still been illusive. In this work, we first identify the fundamental
bottlenecks of SL, and thereby propose a scalable SL framework, coined SGLR.
The server under SGLR broadcasts a common gradient averaged at the split-layer,
emulating FL without any additional communication across clients as opposed to
SplitFed. Meanwhile, SGLR splits the learning rate into its server-side and
client-side rates, and separately adjusts them to support many clients in
parallel. Simulation results corroborate that SGLR achieves higher accuracy
than other baseline SL methods including SplitFed, which is even on par with FL
consuming higher energy and communication costs. As a secondary result, we
observe greater reduction in leakage of sensitive information via mutual
information using SLGR over the baselines.

    

### [[2112.05934] SPDCinv: Inverse Quantum-Optical Design of High-Dimensional Qudits](http://arxiv.org/abs/2112.05934)


  Spontaneous parametric down-conversion in quantum optics is an invaluable
resource for the realization of high-dimensional qudits with spatial modes of
light. One of the main open challenges is how to directly generate a desirable
qudit state in the SPDC process. This problem can be addressed through advanced
computational learning methods; however, due to difficulties in modeling the
SPDC process by a fully differentiable algorithm that takes into account all
interaction effects, progress has been limited. Here, we overcome these
limitations and introduce a physically-constrained and differentiable model,
validated against experimental results for shaped pump beams and structured
crystals, capable of learning every interaction parameter in the process. We
avoid any restrictions induced by the stochastic nature of our physical model
and integrate the dynamic equations governing the evolution under the SPDC
Hamiltonian. We solve the inverse problem of designing a nonlinear quantum
optical system that achieves the desired quantum state of down-converted photon
pairs. The desired states are defined using either the second-order
correlations between different spatial modes or by specifying the required
density matrix. By learning nonlinear volume holograms as well as different
pump shapes, we successfully show how to generate maximally entangled states.
Furthermore, we simulate all-optical coherent control over the generated
quantum state by actively changing the profile of the pump beam. Our work can
be useful for applications such as novel designs of high-dimensional quantum
key distribution and quantum information processing protocols. In addition, our
method can be readily applied for controlling other degrees of freedom of light
in the SPDC process, such as the spectral and temporal properties, and may even
be used in condensed-matter systems having a similar interaction Hamiltonian.

    

### [[2112.05958] You Only Need End-to-End Training for Long-Tailed Recognition](http://arxiv.org/abs/2112.05958)


  The generalization gap on the long-tailed data sets is largely owing to most
categories only occupying a few training samples. Decoupled training achieves
better performance by training backbone and classifier separately. What causes
the poorer performance of end-to-end model training (e.g., logits margin-based
methods)? In this work, we identify a key factor that affects the learning of
the classifier: the channel-correlated features with low entropy before
inputting into the classifier. From the perspective of information theory, we
analyze why cross-entropy loss tends to produce highly correlated features on
the imbalanced data. In addition, we theoretically analyze and prove its
impacts on the gradients of classifier weights, the condition number of
Hessian, and logits margin-based approach. Therefore, we firstly propose to use
Channel Whitening to decorrelate ("scatter") the classifier's inputs for
decoupling the weight update and reshaping the skewed decision boundary, which
achieves satisfactory results combined with logits margin-based method.
However, when the number of minor classes are large, batch imbalance and more
participation in training cause over-fitting of the major classes. We also
propose two novel modules, Block-based Relatively Balanced Batch Sampler (B3RS)
and Batch Embedded Training (BET) to solve the above problems, which makes the
end-to-end training achieve even better performance than decoupled training.
Experimental results on the long-tailed classification benchmarks, CIFAR-LT and
ImageNet-LT, demonstrate the effectiveness of our method.

    

### [[2112.05977] Test Set Sizing Via Random Matrix Theory](http://arxiv.org/abs/2112.05977)


  This paper uses techniques from Random Matrix Theory to find the ideal
training-testing data split for a simple linear regression with m data points,
each an independent n-dimensional multivariate Gaussian. It defines "ideal" as
satisfying the integrity metric, i.e. the empirical model error is the actual
measurement noise, and thus fairly reflects the value or lack of same of the
model. This paper is the first to solve for the training and test size for any
model in a way that is truly optimal. The number of data points in the training
set is the root of a quartic polynomial Theorem 1 derives which depends only on
m and n; the covariance matrix of the multivariate Gaussian, the true model
parameters, and the true measurement noise drop out of the calculations. The
critical mathematical difficulties were realizing that the problems herein were
discussed in the context of the Jacobi Ensemble, a probability distribution
describing the eigenvalues of a known random matrix model, and evaluating a new
integral in the style of Selberg and Aomoto. Mathematical results are supported
with thorough computational evidence. This paper is a step towards automatic
choices of training/test set sizes in machine learning.

    

### [[2112.06007] Determinantal point processes based on orthogonal polynomials for sampling minibatches in SGD](http://arxiv.org/abs/2112.06007)


  Stochastic gradient descent (SGD) is a cornerstone of machine learning. When
the number N of data items is large, SGD relies on constructing an unbiased
estimator of the gradient of the empirical risk using a small subset of the
original dataset, called a minibatch. Default minibatch construction involves
uniformly sampling a subset of the desired size, but alternatives have been
explored for variance reduction. In particular, experimental evidence suggests
drawing minibatches from determinantal point processes (DPPs), distributions
over minibatches that favour diversity among selected items. However, like in
recent work on DPPs for coresets, providing a systematic and principled
understanding of how and why DPPs help has been difficult. In this work, we
contribute an orthogonal polynomial-based DPP paradigm for minibatch sampling
in SGD. Our approach leverages the specific data distribution at hand, which
endows it with greater sensitivity and power over existing data-agnostic
methods. We substantiate our method via a detailed theoretical analysis of its
convergence properties, interweaving between the discrete data set and the
underlying continuous domain. In particular, we show how specific DPPs and a
string of controlled approximations can lead to gradient estimators with a
variance that decays faster with the batchsize than under uniform sampling.
Coupled with existing finite-time guarantees for SGD on convex objectives, this
entails that, DPP minibatches lead to a smaller bound on the mean square
approximation error than uniform minibatches. Moreover, our estimators are
amenable to a recent algorithm that directly samples linear statistics of DPPs
(i.e., the gradient estimator) without sampling the underlying DPP (i.e., the
minibatch), thereby reducing computational overhead. We provide detailed
synthetic as well as real data experiments to substantiate our theoretical
claims.

    

### [[2112.06008] Privacy Amplification via Shuffling for Linear Contextual Bandits](http://arxiv.org/abs/2112.06008)


  Contextual bandit algorithms are widely used in domains where it is desirable
to provide a personalized service by leveraging contextual information, that
may contain sensitive information that needs to be protected. Inspired by this
scenario, we study the contextual linear bandit problem with differential
privacy (DP) constraints. While the literature has focused on either
centralized (joint DP) or local (local DP) privacy, we consider the shuffle
model of privacy and we show that is possible to achieve a privacy/utility
trade-off between JDP and LDP. By leveraging shuffling from privacy and
batching from bandits, we present an algorithm with regret bound
$\widetilde{\mathcal{O}}(T^{2/3}/\varepsilon^{1/3})$, while guaranteeing both
central (joint) and local privacy. Our result shows that it is possible to
obtain a trade-off between JDP and LDP by leveraging the shuffle model while
preserving local privacy.

    

### [[2112.06011] Improving the Transferability of Adversarial Examples with Resized-Diverse-Inputs, Diversity-Ensemble and Region Fitting](http://arxiv.org/abs/2112.06011)


  We introduce a three stage pipeline: resized-diverse-inputs (RDIM),
diversity-ensemble (DEM) and region fitting, that work together to generate
transferable adversarial examples. We first explore the internal relationship
between existing attacks, and propose RDIM that is capable of exploiting this
relationship. Then we propose DEM, the multi-scale version of RDIM, to generate
multi-scale gradients. After the first two steps we transform value fitting
into region fitting across iterations. RDIM and region fitting do not require
extra running time and these three steps can be well integrated into other
attacks. Our best attack fools six black-box defenses with a 93% success rate
on average, which is higher than the state-of-the-art gradient-based attacks.
Besides, we rethink existing attacks rather than simply stacking new methods on
the old ones to get better performance. It is expected that our findings will
serve as the beginning of exploring the internal relationship between attack
methods. Codes are available at this https URL.

    

### [[2112.06018] Control-Tutored Reinforcement Learning: Towards the Integration of Data-Driven and Model-Based Control](http://arxiv.org/abs/2112.06018)


  We present an architecture where a feedback controller derived on an
approximate model of the environment assists the learning process to enhance
its data efficiency. This architecture, which we term as Control-Tutored
Q-learning (CTQL), is presented in two alternative flavours. The former is
based on defining the reward function so that a Boolean condition can be used
to determine when the control tutor policy is adopted, while the latter, termed
as probabilistic CTQL (pCTQL), is instead based on executing calls to the tutor
with a certain probability during learning. Both approaches are validated, and
thoroughly benchmarked against Q-Learning, by considering the stabilization of
an inverted pendulum as defined in OpenAI Gym as a representative problem.

    

### [[2112.06024] Optimization of Residual Convolutional Neural Network for Electrocardiogram Classification](http://arxiv.org/abs/2112.06024)


  The interpretation of the electrocardiogram (ECG) gives clinical information
and helps in the assessing of the heart function. There are distinct ECG
patterns associated with a specific class of arrythmia. The convolutional
neural network is actually one of the most applied deep learning algorithms in
ECG processing. However, with deep learning models there are many more
hyperparameters to tune. Selecting an optimum or best hyperparameter for the
convolutional neural network algorithm is challenging. Often, we end up tuning
the model manually with different possible range of values until a best fit
model is obtained. Automatic hyperparameters tuning using Bayesian optimization
(BO) and evolutionary algorithms brings a solution to the harbor manual
configuration. In this paper, we propose to optimize the Recurrent one
Dimensional Convolutional Neural Network model (R-1D-CNN) with two levels. At
the first level, a residual convolutional layer and one-dimensional
convolutional neural layers are trained to learn patient-specific ECG features
over which the multilayer perceptron layers can learn to produce the final
class vectors of each input. This level is manual and aims to lower the search
space. The second level is automatic and based on proposed algorithm based BO.
Our proposed optimized R-1D-CNN architecture is evaluated on two publicly
available ECG Datasets. The experimental results display that the proposed
algorithm based BO achieves an optimum rate of 99.95\%, while the baseline
model achieves 99.70\% for the MIT-BIH database. Moreover, experiments
demonstrate that the proposed architecture fine-tuned with BO achieves a higher
accuracy than the other proposed architectures. Our architecture achieves a
good result compared to previous works and based on different experiments.

    

### [[2112.06029] On Automatic Data Augmentation for 3D Point Cloud Classification](http://arxiv.org/abs/2112.06029)


  Data augmentation is an important technique to reduce overfitting and improve
learning performance, but existing works on data augmentation for 3D point
cloud data are based on heuristics. In this work, we instead propose to
automatically learn a data augmentation strategy using bilevel optimization. An
augmentor is designed in a similar fashion to a conditional generator and is
optimized by minimizing a base model's loss on a validation set when the
augmented input is used for training the model. This formulation provides a
more principled way to learn data augmentation on 3D point clouds. We evaluate
our approach on standard point cloud classification tasks and a more
challenging setting with pose misalignment between training and validation/test
sets. The proposed strategy achieves competitive performance on both tasks and
we provide further insight into the augmentor's ability to learn the validation
set distribution.

    

### [[2112.06033] Spatial Graph Convolutional Neural Network via Structured Subdomain Adaptation and Domain Adversarial Learning for Bearing Fault Diagnosis](http://arxiv.org/abs/2112.06033)


  Unsupervised domain adaptation (UDA) has shown remarkable results in bearing
fault diagnosis under changing working conditions in recent years. However,
most UDA methods do not consider the geometric structure of the data.
Furthermore, the global domain adaptation technique is commonly applied, which
ignores the relation between subdomains. This paper addresses mentioned
challenges by presenting the novel deep subdomain adaptation graph convolution
neural network (DSAGCN), which has two key characteristics: First, graph
convolution neural network (GCNN) is employed to model the structure of data.
Second, adversarial domain adaptation and local maximum mean discrepancy (LMMD)
methods are applied concurrently to align the subdomain's distribution and
reduce structure discrepancy between relevant subdomains and global domains.
CWRU and Paderborn bearing datasets are used to validate the DSAGCN method's
efficiency and superiority between comparison models. The experimental results
demonstrate the significance of aligning structured subdomains along with
domain adaptation methods to obtain an accurate data-driven model in
unsupervised fault diagnosis.

    

### [[2112.06044] Achieving Low Complexity Neural Decoders via Iterative Pruning](http://arxiv.org/abs/2112.06044)


  The advancement of deep learning has led to the development of neural
decoders for low latency communications. However, neural decoders can be very
complex which can lead to increased computation and latency. We consider
iterative pruning approaches (such as the lottery ticket hypothesis algorithm)
to prune weights in neural decoders. Decoders with fewer number of weights can
have lower latency and lower complexity while retaining the accuracy of the
original model. This will make neural decoders more suitable for mobile and
other edge devices with limited computational power. We also propose semi-soft
decision decoding for neural decoders which can be used to improve the bit
error rate performance of the pruned network.

    

### [[2112.06048] Behavior measures are predicted by how information is encoded in an individual's brain](http://arxiv.org/abs/2112.06048)


  Similar to how differences in the proficiency of the cardiovascular and
musculoskeletal system predict an individual's athletic ability, differences in
how the same brain region encodes information across individuals may explain
their behavior. However, when studying how the brain encodes information,
researchers choose different neuroimaging tasks (e.g., language or motor
tasks), which can rely on processing different types of information and can
modulate different brain regions. We hypothesize that individual differences in
how information is encoded in the brain are task-specific and predict different
behavior measures. We propose a framework using encoding-models to identify
individual differences in brain encoding and test if these differences can
predict behavior. We evaluate our framework using task functional magnetic
resonance imaging data. Our results indicate that individual differences
revealed by encoding-models are a powerful tool for predicting behavior, and
that researchers should optimize their choice of task and encoding-model for
their behavior of interest.

    

### [[2112.06049] Auto-Tag: Tagging-Data-By-Example in Data Lakes](http://arxiv.org/abs/2112.06049)


  As data lakes become increasingly popular in large enterprises today, there
is a growing need to tag or classify data assets (e.g., files and databases) in
data lakes with additional metadata (e.g., semantic column-types), as the
inferred metadata can enable a range of downstream applications like data
governance (e.g., GDPR compliance), and dataset search. Given the sheer size of
today's enterprise data lakes with petabytes of data and millions of data
assets, it is imperative that data assets can be ``auto-tagged'', using
lightweight inference algorithms and minimal user input. In this work, we
develop Auto-Tag, a corpus-driven approach that automates data-tagging of
\textit{custom} data types in enterprise data lakes. Using Auto-Tag, users only
need to provide \textit{one} example column to demonstrate the desired
data-type to tag. Leveraging an index structure built offline using a
lightweight scan of the data lake, which is analogous to pre-training in
machine learning, Auto-Tag can infer suitable data patterns to best
``describe'' the underlying ``domain'' of the given column at an interactive
speed, which can then be used to tag additional data of the same ``type'' in
data lakes. The Auto-Tag approach can adapt to custom data-types, and is shown
to be both accurate and efficient. Part of Auto-Tag ships as a
``custom-classification'' feature in a cloud-based data governance and catalog
solution \textit{Azure Purview}.

    

### [[2112.06053] FedSoft: Soft Clustered Federated Learning with Proximal Local Updating](http://arxiv.org/abs/2112.06053)


  Traditionally, clustered federated learning groups clients with the same data
distribution into a cluster, so that every client is uniquely associated with
one data distribution and helps train a model for this distribution. We relax
this hard association assumption to soft clustered federated learning, which
allows every local dataset to follow a mixture of multiple source
distributions. We propose FedSoft, which trains both locally personalized
models and high-quality cluster models in this setting. FedSoft limits client
workload by using proximal updates to require the completion of only one
optimization task from a subset of clients in every communication round. We
show, analytically and empirically, that FedSoft effectively exploits
similarities between the source distributions to learn personalized and cluster
models that perform well.

    

### [[2112.06054] Deterministic and Discriminative Imitation (D2-Imitation): Revisiting Adversarial Imitation for Sample Efficiency](http://arxiv.org/abs/2112.06054)


  Sample efficiency is crucial for imitation learning methods to be applicable
in real-world applications. Many studies improve sample efficiency by extending
adversarial imitation to be off-policy regardless of the fact that these
off-policy extensions could either change the original objective or involve
complicated optimization. We revisit the foundation of adversarial imitation
and propose an off-policy sample efficient approach that requires no
adversarial training or min-max optimization. Our formulation capitalizes on
two key insights: (1) the similarity between the Bellman equation and the
stationary state-action distribution equation allows us to derive a novel
temporal difference (TD) learning approach; and (2) the use of a deterministic
policy simplifies the TD learning. Combined, these insights yield a practical
algorithm, Deterministic and Discriminative Imitation (D2-Imitation), which
operates by first partitioning samples into two replay buffers and then
learning a deterministic policy via off-policy reinforcement learning. Our
empirical results show that D2-Imitation is effective in achieving good sample
efficiency, outperforming several off-policy extension approaches of
adversarial imitation on many control tasks.

    

### [[2112.06061] OstrichRL: A Musculoskeletal Ostrich Simulation to Study Bio-mechanical Locomotion](http://arxiv.org/abs/2112.06061)


  Muscle-actuated control is a research topic of interest spanning different
fields, in particular biomechanics, robotics and graphics. This type of control
is particularly challenging because models are often overactuated, and dynamics
are delayed and non-linear. It is however a very well tested and tuned
actuation model that has undergone millions of years of evolution and that
involves interesting properties exploiting passive forces of muscle-tendon
units and efficient energy storage and release. To facilitate research on
muscle-actuated simulation, we release a 3D musculoskeletal simulation of an
ostrich based on the MuJoCo simulator. Ostriches are one of the fastest bipeds
on earth and are therefore an excellent model for studying muscle-actuated
bipedal locomotion. The model is based on CT scans and dissections used to
gather actual muscle data such as insertion sites, lengths and pennation
angles. Along with this model, we also provide a set of reinforcement learning
tasks, including reference motion tracking and a reaching task with the neck.
The reference motion data are based on motion capture clips of various
behaviors which we pre-processed and adapted to our model. This paper describes
how the model was built and iteratively improved using the tasks. We evaluate
the accuracy of the muscle actuation patterns by comparing them to
experimentally collected electromyographic data from locomoting birds. We
believe that this work can be a useful bridge between the biomechanics,
reinforcement learning, graphics and robotics communities, by providing a fast
and easy to use simulation.

    

### [[2112.06063] MedAttacker: Exploring Black-Box Adversarial Attacks on Risk Prediction Models in Healthcare](http://arxiv.org/abs/2112.06063)


  Deep neural networks (DNNs) have been broadly adopted in health risk
prediction to provide healthcare diagnoses and treatments. To evaluate their
robustness, existing research conducts adversarial attacks in the
white/gray-box setting where model parameters are accessible. However, a more
realistic black-box adversarial attack is ignored even though most real-world
models are trained with private data and released as black-box services on the
cloud. To fill this gap, we propose the first black-box adversarial attack
method against health risk prediction models named MedAttacker to investigate
their vulnerability. MedAttacker addresses the challenges brought by EHR data
via two steps: hierarchical position selection which selects the attacked
positions in a reinforcement learning (RL) framework and substitute selection
which identifies substitute with a score-based principle. Particularly, by
considering the temporal context inside EHRs, it initializes its RL position
selection policy by using the contribution score of each visit and the saliency
score of each code, which can be well integrated with the deterministic
substitute selection process decided by the score changes. In experiments,
MedAttacker consistently achieves the highest average success rate and even
outperforms a recent white-box EHR adversarial attack technique in certain
cases when attacking three advanced health risk prediction models in the
black-box setting across multiple real-world datasets. In addition, based on
the experiment results we include a discussion on defending EHR adversarial
attacks.

    

### [[2112.06070] A Comparative Study on Robust Graph Neural Networks to Structural Noises](http://arxiv.org/abs/2112.06070)


  Graph neural networks (GNNs) learn node representations by passing and
aggregating messages between neighboring nodes. GNNs have been applied
successfully in several application domains and achieved promising performance.
However, GNNs could be vulnerable to structural noise because of the message
passing mechanism where noise may be propagated through the entire graph.
Although a series of robust GNNs have been proposed, they are evaluated with
different structural noises, and it lacks a systematic comparison with
consistent settings. In this work, we conduct a comprehensive and systematical
comparative study on different types of robust GNNs under consistent structural
noise settings. From the noise aspect, we design three different levels of
structural noises, i.e., local, community, and global noises. From the model
aspect, we select some representative models from sample-based, revision-based,
and construction-based robust GNNs. Based on the empirical results, we provide
some practical suggestions for robust GNNs selection.

    

### [[2112.06071] Multi-Attention Multiple Instance Learning](http://arxiv.org/abs/2112.06071)


  A new multi-attention based method for solving the MIL problem (MAMIL), which
takes into account the neighboring patches or instances of each analyzed patch
in a bag, is proposed. In the method, one of the attention modules takes into
account adjacent patches or instances, several attention modules are used to
get a diverse feature representation of patches, and one attention module is
used to unite different feature representations to provide an accurate
classification of each patch (instance) and the whole bag. Due to MAMIL, a
combined representation of patches and their neighbors in the form of
embeddings of a small dimensionality for simple classification is realized.
Moreover, different types of patches are efficiently processed, and a diverse
feature representation of patches in a bag by using several attention modules
is implemented. A simple approach for explaining the classification predictions
of patches is proposed. Numerical experiments with various datasets illustrate
the proposed method.

    

### [[2112.06087] Convergence of Generalized Belief Propagation Algorithm on Graphs with Motifs](http://arxiv.org/abs/2112.06087)


  Belief propagation is a fundamental message-passing algorithm for numerous
applications in machine learning. It is known that belief propagation algorithm
is exact on tree graphs. However, belief propagation is run on loopy graphs in
most applications. So, understanding the behavior of belief propagation on
loopy graphs has been a major topic for researchers in different areas. In this
paper, we study the convergence behavior of generalized belief propagation
algorithm on graphs with motifs (triangles, loops, etc.) We show under a
certain initialization, generalized belief propagation converges to the global
optimum of the Bethe free energy for ferromagnetic Ising models on graphs with
motifs.

    

### [[2112.06096] Selecting Parallel In-domain Sentences for Neural Machine Translation Using Monolingual Texts](http://arxiv.org/abs/2112.06096)


  Continuously-growing data volumes lead to larger generic models. Specific
use-cases are usually left out, since generic models tend to perform poorly in
domain-specific cases. Our work addresses this gap with a method for selecting
in-domain data from generic-domain (parallel text) corpora, for the task of
machine translation. The proposed method ranks sentences in parallel
general-domain data according to their cosine similarity with a monolingual
domain-specific data set. We then select the top K sentences with the highest
similarity score to train a new machine translation system tuned to the
specific in-domain data. Our experimental results show that models trained on
this in-domain data outperform models trained on generic or a mixture of
generic and domain data. That is, our method selects high-quality
domain-specific training instances at low computational cost and data size.

    

### [[2112.06098] CHAMP: Coherent Hardware-Aware Magnitude Pruning of Integrated Photonic Neural Networks](http://arxiv.org/abs/2112.06098)


  We propose a novel hardware-aware magnitude pruning technique for coherent
photonic neural networks. The proposed technique can prune 99.45% of network
parameters and reduce the static power consumption by 98.23% with a negligible
accuracy loss.

    

### [[2112.06101] Confidence intervals for the random forest generalization error](http://arxiv.org/abs/2112.06101)


  We show that underneath the training process of a random forest there lies
not only the well known and almost computationally free out-of-bag point
estimate of its generalization error, but also a path to compute a confidence
interval for the generalization error which does not demand a retraining of the
forest or any forms of data splitting. Besides the low computational cost
involved in its construction, this confidence interval is shown through
simulations to have good coverage and appropriate shrinking rate of its width
in terms of the training sample size.

    

### [[2112.06116] Stereoscopic Universal Perturbations across Different Architectures and Datasets](http://arxiv.org/abs/2112.06116)


  We study the effect of adversarial perturbations of images on deep stereo
matching networks for the disparity estimation task. We present a method to
craft a single set of perturbations that, when added to any stereo image pair
in a dataset, can fool a stereo network to significantly alter the perceived
scene geometry. Our perturbation images are "universal" in that they not only
corrupt estimates of the network on the dataset they are optimized for, but
also generalize to stereo networks with different architectures across
different datasets. We evaluate our approach on multiple public benchmark
datasets and show that our perturbations can increase D1-error (akin to fooling
rate) of state-of-the-art stereo networks from 1% to as much as 87%. We
investigate the effect of perturbations on the estimated scene geometry and
identify object classes that are most vulnerable. Our analysis on the
activations of registered points between left and right images led us to find
that certain architectural components, i.e. deformable convolution and explicit
matching, can increase robustness against adversaries. We demonstrate that by
simply designing networks with such components, one can reduce the effect of
adversaries by up to 60.5%, which rivals the robustness of networks fine-tuned
with costly adversarial data augmentation.

    

### [[2112.06121] Magnifying Networks for Images with Billions of Pixels](http://arxiv.org/abs/2112.06121)


  The shift towards end-to-end deep learning has brought unprecedented advances
in many areas of computer vision. However, there are cases where the input
images are excessively large, deeming end-to-end approaches impossible. In this
paper, we introduce a new network, the Magnifying Network (MagNet), which can
be trained end-to-end independently of the input image size. MagNets combine
convolutional neural networks with differentiable spatial transformers, in a
new way, to navigate and successfully learn from images with billions of
pixels. Drawing inspiration from the magnifying nature of an ordinary
brightfield microscope, a MagNet processes a downsampled version of an image,
and without supervision learns how to identify areas that may carry value to
the task at hand, upsamples them, and recursively repeats this process on each
of the extracted patches. Our results on the publicly available Camelyon16 and
Camelyon17 datasets first corroborate to the effectiveness of MagNets and the
proposed optimization framework and second, demonstrate the advantage of
Magnets' built-in transparency, an attribute of utmost importance for critical
processes such as medical diagnosis.

    

### [[2112.06125] Extending AdamW by Leveraging Its Second Moment and Magnitude](http://arxiv.org/abs/2112.06125)


  Recent work [4] analyses the local convergence of Adam in a neighbourhood of
an optimal solution for a twice-differentiable function. It is found that the
learning rate has to be sufficiently small to ensure local stability of the
optimal solution. The above convergence results also hold for AdamW. In this
work, we propose a new adaptive optimisation method by extending AdamW in two
aspects with the purpose to relax the requirement on small learning rate for
local stability, which we refer to as Aida. Firstly, we consider tracking the
2nd moment r_t of the pth power of the gradient-magnitudes. r_t reduces to v_t
of AdamW when p=2. Suppose {m_t} is the first moment of AdamW. It is known that
the update direction m_{t+1}/(v_{t+1}+epsilon)^0.5 (or
m_{t+1}/(v_{t+1}^0.5+epsilon) of AdamW (or Adam) can be decomposed as the sign
vector sign(m_{t+1}) multiplied elementwise by a vector of magnitudes
|m_{t+1}|/(v_{t+1}+epsilon)^0.5 (or |m_{t+1}|/(v_{t+1}^0.5+epsilon)). Aida is
designed to compute the qth power of the magnitude in the form of
|m_{t+1}|^q/(r_{t+1}+epsilon)^(q/p) (or |m_{t+1}|^q/((r_{t+1})^(q/p)+epsilon)),
which reduces to that of AdamW when (p,q)=(2,1).
Suppose the origin 0 is a local optimal solution of a twice-differentiable
function. It is found theoretically that when q>1 and p>1 in Aida, the origin 0
is locally stable only when the weight-decay is non-zero. Experiments are
conducted for solving ten toy optimisation problems and training Transformer
and Swin-Transformer for two deep learning (DL) tasks. The empirical study
demonstrates that in a number of scenarios (including the two DL tasks), Aida
with particular setups of (p,q) not equal to (2,1) outperforms the setup
(p,q)=(2,1) of AdamW.

    

### [[2112.06126] Neural Network Quantization for Efficient Inference: A Survey](http://arxiv.org/abs/2112.06126)


  As neural networks have become more powerful, there has been a rising desire
to deploy them in the real world; however, the power and accuracy of neural
networks is largely due to their depth and complexity, making them difficult to
deploy, especially in resource-constrained devices. Neural network quantization
has recently arisen to meet this demand of reducing the size and complexity of
neural networks by reducing the precision of a network. With smaller and
simpler networks, it becomes possible to run neural networks within the
constraints of their target hardware. This paper surveys the many neural
network quantization techniques that have been developed in the last decade.
Based on this survey and comparison of neural network quantization techniques,
we propose future directions of research in the area.

    

### [[2112.06127] Real-world challenges for reinforcement learning in building control](http://arxiv.org/abs/2112.06127)


  Building upon prior research that highlighted the need for standardizing
environments for building control research, and inspired by recently introduced
benchmarks for real life reinforcement learning control, here we propose a
non-exhaustive nine real world challenges for reinforcement learning building
controller. We argue that building control research should be expressed in this
framework in addition to providing a standardized environment for
repeatability. Advanced controllers such as model predictive control and
reinforcement learning control have both advantages and disadvantages that
prevent them from being implemented in real world buildings. Comparisons
between the two are seldom, and often biased. By focusing on the benchmark
problems and challenges, we can investigate the performance of the controllers
under a variety of situations and generate a fair comparison. Lastly, we call
for a more interdisciplinary effort of the research community to address the
real world challenges, and unlock the potentials of advanced building
controllers.

    

### [[2112.06129] Online Adaptation of Neural Network Models by Modified Extended Kalman Filter for Customizable and Transferable Driving Behavior Prediction](http://arxiv.org/abs/2112.06129)


  High fidelity behavior prediction of human drivers is crucial for efficient
and safe deployment of autonomous vehicles, which is challenging due to the
stochasticity, heterogeneity, and time-varying nature of human behaviors. On
one hand, the trained prediction model can only capture the motion pattern in
an average sense, while the nuances among individuals can hardly be reflected.
On the other hand, the prediction model trained on the training set may not
generalize to the testing set which may be in a different scenario or data
distribution, resulting in low transferability and generalizability. In this
paper, we applied a $\tau$-step modified Extended Kalman Filter parameter
adaptation algorithm (MEKF$_\lambda$) to the driving behavior prediction task,
which has not been studied before in literature. With the feedback of the
observed trajectory, the algorithm is applied to neural-network-based models to
improve the performance of driving behavior predictions across different human
subjects and scenarios. A new set of metrics is proposed for systematic
evaluation of online adaptation performance in reducing the prediction error
for different individuals and scenarios. Empirical studies on the best layer in
the model and steps of observation to adapt are also provided.

    

### [[2112.06130] On reducing the order of arm-passes bandit streaming algorithms under memory bottleneck](http://arxiv.org/abs/2112.06130)


  In this work we explore multi-arm bandit streaming model, especially in cases
where the model faces resource bottleneck. We build over existing algorithms
conditioned by limited arm memory at any instance of time. Specifically, we
improve the amount of streaming passes it takes for a bandit algorithm to incur
a $O(\sqrt{T\log(T)})$ regret by a logarithmic factor, and also provide 2-pass
algorithms with some initial conditions to incur a similar order of regret.

    

### [[2112.06132] PRNet: A Periodic Residual Learning Network for Crowd Flow Forecasting](http://arxiv.org/abs/2112.06132)


  Crowd flow forecasting, e.g., predicting the crowds entering or leaving
certain regions, is of great importance to real-world urban applications. One
of the key properties of crowd flow data is periodicity: a pattern that occurs
at regular time intervals, such as a weekly pattern. To capture such
periodicity, existing studies either explicitly model it based on the periodic
hidden states or implicitly learn it by feeding all periodic segments into
neural networks. In this paper, we devise a novel periodic residual learning
network (PRNet) for better modeling the periodicity in crowd flow data.
Differing from existing methods, PRNet frames the crowd flow forecasting as a
periodic residual learning problem by modeling the deviation between the input
(the previous time period) and the output (the future time period). As compared
to predicting highly dynamic crowd flows directly, learning such stationary
deviation is much easier, which thus facilitates the model training. Besides,
the learned deviation enables the network to produce the residual between
future conditions and its corresponding weekly observations at each time
interval, and therefore contributes to substantially better predictions. We
further propose a lightweight Spatial-Channel Enhanced Encoder to build more
powerful region representations, by jointly capturing global spatial
correlations and temporal dependencies. Experimental results on two real-world
datasets demonstrate that PRNet outperforms the state-of-the-art methods in
terms of both accuracy and robustness.

    

### [[2112.06134] Markov subsampling based Huber Criterion](http://arxiv.org/abs/2112.06134)


  Subsampling is an important technique to tackle the computational challenges
brought by big data. Many subsampling procedures fall within the framework of
importance sampling, which assigns high sampling probabilities to the samples
appearing to have big impacts. When the noise level is high, those sampling
procedures tend to pick many outliers and thus often do not perform
satisfactorily in practice. To tackle this issue, we design a new Markov
subsampling strategy based on Huber criterion (HMS) to construct an informative
subset from the noisy full data; the constructed subset then serves as a
refined working data for efficient processing. HMS is built upon a
Metropolis-Hasting procedure, where the inclusion probability of each sampling
unit is determined using the Huber criterion to prevent over scoring the
outliers. Under mild conditions, we show that the estimator based on the
subsamples selected by HMS is statistically consistent with a sub-Gaussian
deviation bound. The promising performance of HMS is demonstrated by extensive
studies on large scale simulations and real data examples.

    

### [[2112.06142] Semi-supervised teacher-student deep neural network for materials discovery](http://arxiv.org/abs/2112.06142)


  Data driven generative machine learning models have recently emerged as one
of the most promising approaches for new materials discovery. While the
generator models can generate millions of candidates, it is critical to train
fast and accurate machine learning models to filter out stable, synthesizable
materials with desired properties. However, such efforts to build supervised
regression or classification screening models have been severely hindered by
the lack of unstable or unsynthesizable samples, which usually are not
collected and deposited in materials databases such as ICSD and Materials
Project (MP). At the same time, there are a significant amount of unlabelled
data available in these databases. Here we propose a semi-supervised deep
neural network (TSDNN) model for high-performance formation energy and
synthesizability prediction, which is achieved via its unique teacher-student
dual network architecture and its effective exploitation of the large amount of
unlabeled data. For formation energy based stability screening, our
semi-supervised classifier achieves an absolute 10.3\% accuracy improvement
compared to the baseline CGCNN regression model. For synthesizability
prediction, our model significantly increases the baseline PU learning's true
positive rate from 87.9\% to 97.9\% using 1/49 model parameters.
To further prove the effectiveness of our models, we combined our
TSDNN-energy and TSDNN-synthesizability models with our CubicGAN generator to
discover novel stable cubic structures. Out of 1000 recommended candidate
samples by our models, 512 of them have negative formation energies as
validated by our DFT formation energy calculations. Our experimental results
show that our semi-supervised deep neural networks can significantly improve
the screening accuracy in large-scale generative materials design.

    

### [[2112.06148] Programming with Neural Surrogates of Programs](http://arxiv.org/abs/2112.06148)


  Surrogates, models that mimic the behavior of programs, form the basis of a
variety of development workflows. We study three surrogate-based design
patterns, evaluating each in case studies on a large-scale CPU simulator.
With surrogate compilation, programmers develop a surrogate that mimics the
behavior of a program to deploy to end-users in place of the original program.
Surrogate compilation accelerates the CPU simulator under study by $1.6\times$.
With surrogate adaptation, programmers develop a surrogate of a program then
retrain that surrogate on a different task. Surrogate adaptation decreases the
simulator's error by up to $50\%$. With surrogate optimization, programmers
develop a surrogate of a program, optimize input parameters of the surrogate,
then plug the optimized input parameters back into the original program.
Surrogate optimization finds simulation parameters that decrease the
simulator's error by $5\%$ compared to the error induced by expert-set
parameters.
In this paper we formalize this taxonomy of surrogate-based design patterns.
We further describe the programming methodology common to all three design
patterns. Our work builds a foundation for the emerging class of workflows
based on programming with surrogates of programs.

    

### [[2112.06160] Maintaining AUC and $H$-measure over time](http://arxiv.org/abs/2112.06160)


  Measuring the performance of a classifier is a vital task in machine
learning. The running time of an algorithm that computes the measure plays a
very small role in an offline setting, for example, when the classifier is
being developed by a researcher. However, the running time becomes more crucial
if our goal is to monitor the performance of a classifier over time.
In this paper we study three algorithms for maintaining two measures. The
first algorithm maintains area under the ROC curve (AUC) under addition and
deletion of data points in $O(\log n)$ time. This is done by maintaining the
data points sorted in a self-balanced search tree. In addition, we augment the
search tree that allows us to query the ROC coordinates of a data point in
$O(\log n)$ time. In doing so we are able to maintain AUC in $O(\log n)$ time.
Our next two algorithms involve in maintaining $H$-measure, an alternative
measure based on the ROC curve. Computing the measure is a two-step process:
first we need to compute a convex hull of the ROC curve, followed by a sum over
the convex hull. We demonstrate that we can maintain the convex hull using a
minor modification of the classic convex hull maintenance algorithm. We then
show that under certain conditions, we can compute the $H$-measure exactly in
$O(\log^2 n)$ time, and if the conditions are not met, then we can estimate the
$H$-measure in $O((\log n + \epsilon^{-1})\log n)$ time. We show empirically
that our methods are significantly faster than the baselines.

    

### [[2112.06161] Semi-supervised Domain Adaptive Structure Learning](http://arxiv.org/abs/2112.06161)


  Semi-supervised domain adaptation (SSDA) is quite a challenging problem
requiring methods to overcome both 1) overfitting towards poorly annotated data
and 2) distribution shift across domains. Unfortunately, a simple combination
of domain adaptation (DA) and semi-supervised learning (SSL) methods often fail
to address such two objects because of training data bias towards labeled
samples. In this paper, we introduce an adaptive structure learning method to
regularize the cooperation of SSL and DA. Inspired by the multi-views learning,
our proposed framework is composed of a shared feature encoder network and two
classifier networks, trained for contradictory purposes. Among them, one of the
classifiers is applied to group target features to improve intra-class density,
enlarging the gap of categorical clusters for robust representation learning.
Meanwhile, the other classifier, serviced as a regularizer, attempts to scatter
the source features to enhance the smoothness of the decision boundary. The
iterations of target clustering and source expansion make the target features
being well-enclosed inside the dilated boundary of the corresponding source
points. For the joint address of cross-domain features alignment and partially
labeled data learning, we apply the maximum mean discrepancy (MMD) distance
minimization and self-training (ST) to project the contradictory structures
into a shared view to make the reliable final decision. The experimental
results over the standard SSDA benchmarks, including DomainNet and Office-home,
demonstrate both the accuracy and robustness of our method over the
state-of-the-art approaches.

    

### [[2112.06189] MPLR: a novel model for multi-target learning of logical rules for knowledge graph reasoning](http://arxiv.org/abs/2112.06189)


  Large-scale knowledge graphs (KGs) provide structured representations of
human knowledge. However, as it is impossible to contain all knowledge, KGs are
usually incomplete. Reasoning based on existing facts paves a way to discover
missing facts. In this paper, we study the problem of learning logic rules for
reasoning on knowledge graphs for completing missing factual triplets. Learning
logic rules equips a model with strong interpretability as well as the ability
to generalize to similar tasks. We propose a model called MPLR that improves
the existing models to fully use training data and multi-target scenarios are
considered. In addition, considering the deficiency in evaluating the
performance of models and the quality of mined rules, we further propose two
novel indicators to help with the problem. Experimental results empirically
demonstrate that our MPLR model outperforms state-of-the-art methods on five
benchmark datasets. The results also prove the effectiveness of the indicators.

    

### [[2112.06200] Secure Routine: A Routine-Based Algorithm for Drivers Identification](http://arxiv.org/abs/2112.06200)


  The introduction of Information and Communication Technology (ICT) in
transportation systems leads to several advantages (efficiency of transport,
mobility, traffic management). However, it may bring some drawbacks in terms of
increasing security challenges, also related to human behaviour. As an example
, in the last decades attempts to characterize drivers' behaviour have been
mostly targeted. This paper presents Secure Routine, a paradigm that uses
driver's habits to driver identification and, in particular, to distinguish the
vehicle's owner from other drivers. We evaluate Secure Routine in combination
with other three existing research works based on machine learning techniques.
Results are measured using well-known metrics and show that Secure Routine
outperforms the compared works.

    

### [[2112.06206] Automatic differentiation approach for reconstructing spectral functions with neural networks](http://arxiv.org/abs/2112.06206)


  Reconstructing spectral functions from Euclidean Green's functions is an
important inverse problem in physics. The prior knowledge for specific physical
systems routinely offers essential regularization schemes for solving the
ill-posed problem approximately. Aiming at this point, we propose an automatic
differentiation framework as a generic tool for the reconstruction from
observable data. We represent the spectra by neural networks and set chi-square
as loss function to optimize the parameters with backward automatic
differentiation unsupervisedly. In the training process, there is no explicit
physical prior embedding into neural networks except the positive-definite
form. The reconstruction accuracy is assessed through Kullback-Leibler(KL)
divergence and mean square error(MSE) at multiple noise levels. It should be
noted that the automatic differential framework and the freedom of introducing
regularization are inherent advantages of the present approach and may lead to
improvements of solving inverse problem in the future.

    

### [[2112.06209] Measuring Complexity of Learning Schemes Using Hessian-Schatten Total-Variation](http://arxiv.org/abs/2112.06209)


  In this paper, we introduce the Hessian-Schatten total-variation (HTV) -- a
novel seminorm that quantifies the total "rugosity" of multivariate functions.
Our motivation for defining HTV is to assess the complexity of supervised
learning schemes. We start by specifying the adequate matrix-valued Banach
spaces that are equipped with suitable classes of mixed-norms. We then show
that HTV is invariant to rotations, scalings, and translations. Additionally,
its minimum value is achieved for linear mappings, supporting the common
intuition that linear regression is the least complex learning model. Next, we
present closed-form expressions for computing the HTV of two general classes of
functions. The first one is the class of Sobolev functions with a certain
degree of regularity, for which we show that HTV coincides with the
Hessian-Schatten seminorm that is sometimes used as a regularizer for image
reconstruction. The second one is the class of continuous and piecewise linear
(CPWL) functions. In this case, we show that the HTV reflects the total change
in slopes between linear regions that have a common facet. Hence, it can be
viewed as a convex relaxation (l1-type) of the number of linear regions
(l0-type) of CPWL mappings. Finally, we illustrate the use of our proposed
seminorm with some concrete examples.

    

### [[2112.06211] Quantum kernels for real-world predictions based on electronic health records](http://arxiv.org/abs/2112.06211)


  In recent years, research on near-term quantum machine learning has explored
how classical machine learning algorithms endowed with access to quantum
kernels (similarity measures) can outperform their purely classical
counterparts. Although theoretical work has shown provable advantage on
synthetic data sets, no work done to date has studied empirically whether
quantum advantage is attainable and with what kind of data set. In this paper,
we report the first systematic investigation of empirical quantum advantage
(EQA) in healthcare and life sciences and propose an end-to-end framework to
study EQA. We selected electronic health records (EHRs) data subsets and
created a configuration space of 5-20 features and 200-300 training samples.
For each configuration coordinate, we trained classical support vector machine
(SVM) models based on radial basis function (RBF) kernels and quantum models
with custom kernels using an IBM quantum computer. We empirically identified
regimes where quantum kernels could provide advantage on a particular data set
and introduced a terrain ruggedness index, a metric to help quantitatively
estimate how the accuracy of a given model will perform as a function of the
number of features and sample size. The generalizable framework introduced here
represents a key step towards a priori identification of data sets where
quantum advantage could exist.

    

### [[2112.06219] Visualising and Explaining Deep Learning Models for Speech Quality Prediction](http://arxiv.org/abs/2112.06219)


  Estimating quality of transmitted speech is known to be a non-trivial task.
While traditionally, test participants are asked to rate the quality of
samples; nowadays, automated methods are available. These methods can be
divided into: 1) intrusive models, which use both, the original and the
degraded signals, and 2) non-intrusive models, which only require the degraded
signal. Recently, non-intrusive models based on neural networks showed to
outperform signal processing based models. However, the advantages of deep
learning based models come with the cost of being more challenging to
interpret. To get more insight into the prediction models the non-intrusive
speech quality prediction model NISQA is analyzed in this paper. NISQA is
composed of a convolutional neural network (CNN) and a recurrent neural network
(RNN). The task of the CNN is to compute relevant features for the speech
quality prediction on a frame level, while the RNN models time-dependencies
between the individual speech frames. Different explanation algorithms are used
to understand the automatically learned features of the CNN. In this way,
several interpretable features could be identified, such as the sensitivity to
noise or strong interruptions. On the other hand, it was found that multiple
features carry redundant information.

    

### [[2112.06225] Approximation algorithms for confidence bands for time series](http://arxiv.org/abs/2112.06225)


  Confidence intervals are a standard technique for analyzing data. When
applied to time series, confidence intervals are computed for each time point
separately. Alternatively, we can compute confidence bands, where we are
required to find the smallest area enveloping $k$ time series, where $k$ is a
user parameter. Confidence bands can be then used to detect abnormal time
series, not just individual observations within the time series. We will show
that despite being an NP-hard problem it is possible to find optimal confidence
band for some $k$. We do this by considering a different problem: discovering
regularized bands, where we minimize the envelope area minus the number of
included time series weighted by a parameter $\alpha$. Unlike normal confidence
bands we can solve the problem exactly by using a minimum cut. By varying
$\alpha$ we can obtain solutions for various $k$. If we have a constraint $k$
for which we cannot find appropriate $\alpha$, we demonstrate a simple
algorithm that yields $O(\sqrt{n})$ approximation guarantee by connecting the
problem to a minimum $k$-union problem. This connection also implies that we
cannot approximate the problem better than $O(n^{1/4})$ under some (mild)
assumptions. Finally, we consider a variant where instead of minimizing the
area we minimize the maximum width. Here, we demonstrate a simple
2-approximation algorithm and show that we cannot achieve better approximation
guarantee.

    

### [[2112.06242] Image Reconstruction from Events. Why learn it?](http://arxiv.org/abs/2112.06242)


  Traditional cameras measure image intensity. Event cameras, by contrast,
measure per-pixel temporal intensity changes asynchronously. Recovering
intensity from events is a popular research topic since the reconstructed
images inherit the high dynamic range (HDR) and high-speed properties of
events; hence they can be used in many robotic vision applications and to
generate slow-motion HDR videos. However, state-of-the-art methods tackle this
problem by training an event-to-image recurrent neural network (RNN), which
lacks explainability and is difficult to tune. In this work we show, for the
first time, how tackling the joint problem of motion and intensity estimation
leads us to model event-based image reconstruction as a linear inverse problem
that can be solved without training an image reconstruction RNN. Instead,
classical and learning-based image priors can be used to solve the problem and
remove artifacts from the reconstructed images. The experiments show that the
proposed approach generates images with visual quality on par with
state-of-the-art methods despite only using data from a short time interval
(i.e., without recurrent connections). Our method can also be used to improve
the quality of images reconstructed by approaches that first estimate the image
Laplacian; here our method can be interpreted as Poisson reconstruction guided
by image priors.

    

### [[2112.06244] SHGNN: Structure-Aware Heterogeneous Graph Neural Network](http://arxiv.org/abs/2112.06244)


  Many real-world graphs (networks) are heterogeneous with different types of
nodes and edges. Heterogeneous graph embedding, aiming at learning the
low-dimensional node representations of a heterogeneous graph, is vital for
various downstream applications. Many meta-path based embedding methods have
been proposed to learn the semantic information of heterogeneous graphs in
recent years. However, most of the existing techniques overlook the graph
structure information when learning the heterogeneous graph embeddings. This
paper proposes a novel Structure-Aware Heterogeneous Graph Neural Network
(SHGNN) to address the above limitations. In detail, we first utilize a feature
propagation module to capture the local structure information of intermediate
nodes in the meta-path. Next, we use a tree-attention aggregator to incorporate
the graph structure information into the aggregation module on the meta-path.
Finally, we leverage a meta-path aggregator to fuse the information aggregated
from different meta-paths. We conducted experiments on node classification and
clustering tasks and achieved state-of-the-art results on the benchmark
datasets, which shows the effectiveness of our proposed method.

    

### [[2112.06247] DeepFIB: Self-Imputation for Time Series Anomaly Detection](http://arxiv.org/abs/2112.06247)


  Time series (TS) anomaly detection (AD) plays an essential role in various
applications, e.g., fraud detection in finance and healthcare monitoring. Due
to the inherently unpredictable and highly varied nature of anomalies and the
lack of anomaly labels in historical data, the AD problem is typically
formulated as an unsupervised learning problem. The performance of existing
solutions is often not satisfactory, especially in data-scarce scenarios. To
tackle this problem, we propose a novel self-supervised learning technique for
AD in time series, namely \emph{DeepFIB}. We model the problem as a \emph{Fill
In the Blank} game by masking some elements in the TS and imputing them with
the rest. Considering the two common anomaly shapes (point- or
sequence-outliers) in TS data, we implement two masking strategies with many
self-generated training samples. The corresponding self-imputation networks can
extract more robust temporal relations than existing AD solutions and
effectively facilitate identifying the two types of anomalies. For continuous
outliers, we also propose an anomaly localization algorithm that dramatically
reduces AD errors. Experiments on various real-world TS datasets demonstrate
that DeepFIB outperforms state-of-the-art methods by a large margin, achieving
up to $65.2\%$ relative improvement in F1-score.

    

### [[2112.06251] Learning with Subset Stacking](http://arxiv.org/abs/2112.06251)


  We propose a new algorithm that learns from a set of input-output pairs. Our
algorithm is designed for populations where the relation between the input
variables and the output variable exhibits a heterogeneous behavior across the
predictor space. The algorithm starts with generating subsets that are
concentrated around random points in the input space. This is followed by
training a local predictor for each subset. Those predictors are then combined
in a novel way to yield an overall predictor. We call this algorithm "LEarning
with Subset Stacking" or LESS, due to its resemblance to method of stacking
regressors. We compare the testing performance of LESS with the
state-of-the-art methods on several datasets. Our comparison shows that LESS is
a competitive supervised learning method. Moreover, we observe that LESS is
also efficient in terms of computation time and it allows a straightforward
parallel implementation.

    

### [[2112.06253] Up to 100x Faster Data-free Knowledge Distillation](http://arxiv.org/abs/2112.06253)


  Data-free knowledge distillation (DFKD) has recently been attracting
increasing attention from research communities, attributed to its capability to
compress a model only using synthetic data. Despite the encouraging results
achieved, state-of-the-art DFKD methods still suffer from the inefficiency of
data synthesis, making the data-free training process extremely time-consuming
and thus inapplicable for large-scale tasks. In this work, we introduce an
efficacious scheme, termed as FastDFKD, that allows us to accelerate DFKD by a
factor of orders of magnitude. At the heart of our approach is a novel strategy
to reuse the shared common features in training data so as to synthesize
different data instances. Unlike prior methods that optimize a set of data
independently, we propose to learn a meta-synthesizer that seeks common
features as the initialization for the fast data synthesis. As a result,
FastDFKD achieves data synthesis within only a few steps, significantly
enhancing the efficiency of data-free training. Experiments over CIFAR, NYUv2,
and ImageNet demonstrate that the proposed FastDFKD achieves 10$\times$ and
even 100$\times$ acceleration while preserving performances on par with state
of the art.

    

### [[2112.06261] Hidden Effects of COVID-19 on Healthcare Workers: A Machine Learning Analysis](http://arxiv.org/abs/2112.06261)


  In this paper, we analyze some effects of the COVID-19 pandemic on healthcare
workers. We specifically focus on alcohol consumption habit changes among
healthcare workers using a mental health survey data obtained from the
University of Michigan Inter-University Consortium for Political and Social
Research. We use supervised and unsupervised machine learning methods and
models such as Decision Trees, Logistic Regression, Naive Bayes classifier,
k-Nearest Neighbors, Support Vector Machines, Multilayer perceptron, Random
Forests, XGBoost, CatBoost, LightGBM, Synthetic Minority Oversampling,
Chi-Squared Test and mutual information method to find out relationships
between COVID-19 related negative effects and alcohol use changes in healthcare
workers. Our findings suggest that some effects of the COVID-19 pandemic such
as school closure, work schedule change and COVID-related news exposure may
lead to an increase in alcohol use.

    

### [[2112.06274] SparseFed: Mitigating Model Poisoning Attacks in Federated Learning with Sparsification](http://arxiv.org/abs/2112.06274)


  Federated learning is inherently vulnerable to model poisoning attacks
because its decentralized nature allows attackers to participate with
compromised devices. In model poisoning attacks, the attacker reduces the
model's performance on targeted sub-tasks (e.g. classifying planes as birds) by
uploading "poisoned" updates. In this report we introduce \algoname{}, a novel
defense that uses global top-k update sparsification and device-level gradient
clipping to mitigate model poisoning attacks. We propose a theoretical
framework for analyzing the robustness of defenses against poisoning attacks,
and provide robustness and convergence analysis of our algorithm. To validate
its empirical efficacy we conduct an open-source evaluation at scale across
multiple benchmark datasets for computer vision and federated learning.

    

### [[2112.06276] Quantifying and Understanding Adversarial Examples in Discrete Input Spaces](http://arxiv.org/abs/2112.06276)


  Modern classification algorithms are susceptible to adversarial
examples--perturbations to inputs that cause the algorithm to produce
undesirable behavior. In this work, we seek to understand and extend
adversarial examples across domains in which inputs are discrete, particularly
across new domains, such as computational biology. As a step towards this goal,
we formalize a notion of synonymous adversarial examples that applies in any
discrete setting and describe a simple domain-agnostic algorithm to construct
such examples. We apply this algorithm across multiple domains--including
sentiment analysis and DNA sequence classification--and find that it
consistently uncovers adversarial examples. We seek to understand their
prevalence theoretically and we attribute their existence to spurious token
correlations, a statistical phenomenon that is specific to discrete spaces. Our
work is a step towards a domain-agnostic treatment of discrete adversarial
examples analogous to that of continuous inputs.

    

### [[2112.06281] Spatial-Temporal-Fusion BNN: Variational Bayesian Feature Layer](http://arxiv.org/abs/2112.06281)


  Bayesian neural networks (BNNs) have become a principal approach to alleviate
overconfident predictions in deep learning, but they often suffer from scaling
issues due to a large number of distribution parameters. In this paper, we
discover that the first layer of a deep network possesses multiple disparate
optima when solely retrained. This indicates a large posterior variance when
the first layer is altered by a Bayesian layer, which motivates us to design a
spatial-temporal-fusion BNN (STF-BNN) for efficiently scaling BNNs to large
models: (1) first normally train a neural network from scratch to realize fast
training; and (2) the first layer is converted to Bayesian and inferred by
employing stochastic variational inference, while other layers are fixed.
Compared to vanilla BNNs, our approach can greatly reduce the training time and
the number of parameters, which contributes to scale BNNs efficiently. We
further provide theoretical guarantees on the generalizability and the
capability of mitigating overconfidence of STF-BNN. Comprehensive experiments
demonstrate that STF-BNN (1) achieves the state-of-the-art performance on
prediction and uncertainty quantification; (2) significantly improves
adversarial robustness and privacy preservation; and (3) considerably reduces
training time and memory costs.

    

### [[2112.06283] Bayesian Persuasion for Algorithmic Recourse](http://arxiv.org/abs/2112.06283)


  When subjected to automated decision-making, decision-subjects will
strategically modify their observable features in ways they believe will
maximize their chances of receiving a desirable outcome. In many situations,
the underlying predictive model is deliberately kept secret to avoid gaming and
maintain competitive advantage. This opacity forces the decision subjects to
rely on incomplete information when making strategic feature modifications. We
capture such settings as a game of Bayesian persuasion, in which the
decision-maker sends a signal, e.g., an action recommendation, to a decision
subject to incentivize them to take desirable actions. We formulate the
decision-maker's problem of finding the optimal Bayesian incentive-compatible
(BIC) action recommendation policy as an optimization problem and characterize
the solution via a linear program. Through this characterization, we observe
that while the problem of finding the optimal BIC recommendation policy can be
simplified dramatically, the computational complexity of solving this linear
program is closely tied to (1) the relative size of the decision-subjects'
action space, and (2) the number of features utilized by the underlying
predictive model. Finally, we provide bounds on the performance of the optimal
BIC recommendation policy and show that it can lead to arbitrarily better
outcomes compared to standard baselines.

    

### [[2112.06287] Identifying bias in cluster quality metrics](http://arxiv.org/abs/2112.06287)


  We study potential biases of popular cluster quality metrics, such as
conductance or modularity. We propose a method that uses both stochastic and
preferential attachment block models construction to generate networks with
preset community structures, to which quality metrics will be applied. These
models also allow us to generate multi-level structures of varying strength,
which will show if metrics favour partitions into a larger or smaller number of
clusters. Additionally, we propose another quality metric, the density ratio.
We observed that most of the studied metrics tend to favour partitions into a
smaller number of big clusters, even when their relative internal and external
connectivity are the same. The metrics found to be less biased are modularity
and density ratio.

    

### [[2112.06288] Fairness for Robust Learning to Rank](http://arxiv.org/abs/2112.06288)


  While conventional ranking systems focus solely on maximizing the utility of
the ranked items to users, fairness-aware ranking systems additionally try to
balance the exposure for different protected attributes such as gender or race.
To achieve this type of group fairness for ranking, we derive a new ranking
system based on the first principles of distributional robustness. We formulate
a minimax game between a player choosing a distribution over rankings to
maximize utility while satisfying fairness constraints against an adversary
seeking to minimize utility while matching statistics of the training data. We
show that our approach provides better utility for highly fair rankings than
existing baseline methods.

    

### [[2112.06292] Gamifying optimization: a Wasserstein distance-based analysis of human search](http://arxiv.org/abs/2112.06292)


  The main objective of this paper is to outline a theoretical framework to
characterise humans' decision-making strategies under uncertainty, in
particular active learning in a black-box optimization task and trading-off
between information gathering (exploration) and reward seeking (exploitation).
Humans' decisions making according to these two objectives can be modelled in
terms of Pareto rationality. If a decision set contains a Pareto efficient
strategy, a rational decision maker should always select the dominant strategy
over its dominated alternatives. A distance from the Pareto frontier determines
whether a choice is Pareto rational. To collect data about humans' strategies
we have used a gaming application that shows the game field, with previous
decisions and observations, as well as the score obtained. The key element in
this paper is the representation of behavioural patterns of human learners as a
discrete probability distribution. This maps the problem of the
characterization of humans' behaviour into a space whose elements are
probability distributions structured by a distance between histograms, namely
the Wasserstein distance (WST). The distributional analysis gives new insights
about human search strategies and their deviations from Pareto rationality.
Since the uncertainty is one of the two objectives defining the Pareto
frontier, the analysis has been performed for three different uncertainty
quantification measures to identify which better explains the Pareto compliant
behavioural patterns. Beside the analysis of individual patterns WST has also
enabled a global analysis computing the barycenters and WST k-means clustering.
A further analysis has been performed by a decision tree to relate non-Paretian
behaviour, characterized by exasperated exploitation, to the dynamics of the
evolution of the reward seeking process.

    

### [[2112.06305] Recalibrating probabilistic forecasts of epidemics](http://arxiv.org/abs/2112.06305)


  Distributional forecasts are important for a wide variety of applications,
including forecasting epidemics. Often, forecasts are miscalibrated, or
unreliable in assigning uncertainty to future events. We present a
recalibration method that can be applied to a black-box forecaster given
retrospective forecasts and observations, as well as an extension to make this
method more effective in recalibrating epidemic forecasts. This method is
guaranteed to improve calibration and log score performance when trained and
measured in-sample. We also prove that the increase in expected log score of a
recalibrated forecaster is equal to the entropy of the PIT distribution. We
apply this recalibration method to the 27 influenza forecasters in the FluSight
Network and show that recalibration reliably improves forecast accuracy and
calibration. This method is effective, robust, and easy to use as a
post-processing tool to improve epidemic forecasts.

    

### [[2112.06307] Image-to-Height Domain Translation for Synthetic Aperture Sonar](http://arxiv.org/abs/2112.06307)


  Observations of seabed texture with synthetic aperture sonar are dependent
upon several factors. In this work, we focus on collection geometry with
respect to isotropic and anisotropic textures. The low grazing angle of the
collection geometry, combined with orientation of the sonar path relative to
anisotropic texture, poses a significant challenge for image-alignment and
other multi-view scene understanding frameworks. We previously proposed using
features captured from estimated seabed relief to improve scene understanding.
While several methods have been developed to estimate seabed relief via
intensity, no large-scale study exists in the literature. Furthermore, a
dataset of coregistered seabed relief maps and sonar imagery is nonexistent to
learn this domain translation. We address these problems by producing a large
simulated dataset containing coregistered pairs of seabed relief and intensity
maps from two unique sonar data simulation techniques. We apply three types of
models, with varying complexity, to translate intensity imagery to seabed
relief: a Gaussian Markov Random Field approach (GMRF), a conditional
Generative Adversarial Network (cGAN), and UNet architectures. Methods are
compared in reference to the coregistered simulated datasets using L1 error.
Additionally, predictions on simulated and real SAS imagery are shown. Finally,
models are compared on two datasets of hand-aligned SAS imagery and evaluated
in terms of L1 error across multiple aspects in comparison to using intensity.
Our comprehensive experiments show that the proposed UNet architectures
outperform the GMRF and pix2pix cGAN models on seabed relief estimation for
simulated and real SAS imagery.

    

### [[2112.06336] Representing Knowledge as Predictions (and State as Knowledge)](http://arxiv.org/abs/2112.06336)


  This paper shows how a single mechanism allows knowledge to be constructed
layer by layer directly from an agent's raw sensorimotor stream. This
mechanism, the General Value Function (GVF) or "forecast," captures high-level,
abstract knowledge as a set of predictions about existing features and
knowledge, based exclusively on the agent's low-level senses and actions.
Thus, forecasts provide a representation for organizing raw sensorimotor data
into useful abstractions over an unlimited number of layers--a long-sought goal
of AI and cognitive science.
The heart of this paper is a detailed thought experiment providing a
concrete, step-by-step formal illustration of how an artificial agent can build
true, useful, abstract knowledge from its raw sensorimotor experience alone.
The knowledge is represented as a set of layered predictions (forecasts) about
the agent's observed consequences of its actions. This illustration shows
twelve separate layers: the lowest consisting of raw pixels, touch and force
sensors, and a small number of actions; the higher layers increasing in
abstraction, eventually resulting in rich knowledge about the agent's world,
corresponding roughly to doorways, walls, rooms, and floor plans. I then argue
that this general mechanism may allow the representation of a broad spectrum of
everyday human knowledge.

    

### [[2112.06345] A Survey on Societal Event Forecasting with Deep Learning](http://arxiv.org/abs/2112.06345)


  Population-level societal events, such as civil unrest and crime, often have
a significant impact on our daily life. Forecasting such events is of great
importance for decision-making and resource allocation. Event prediction has
traditionally been challenging due to the lack of knowledge regarding the true
causes and underlying mechanisms of event occurrence. In recent years, research
on event forecasting has made significant progress due to two main reasons: (1)
the development of machine learning and deep learning algorithms and (2) the
accessibility of public data such as social media, news sources, blogs,
economic indicators, and other meta-data sources. The explosive growth of data
and the remarkable advancement in software/hardware technologies have led to
applications of deep learning techniques in societal event studies. This paper
is dedicated to providing a systematic and comprehensive overview of deep
learning technologies for societal event predictions. We focus on two domains
of societal events: \textit{civil unrest} and \textit{crime}. We first
introduce how event forecasting problems are formulated as a machine learning
prediction task. Then, we summarize data resources, traditional methods, and
recent development of deep learning models for these problems. Finally, we
discuss the challenges in societal event forecasting and put forward some
promising directions for future research.

    

### [[2112.06351] Neural Point Process for Learning Spatiotemporal Event Dynamics](http://arxiv.org/abs/2112.06351)


  Learning the dynamics of spatiotemporal events is a fundamental problem.
Neural point processes enhance the expressivity of point process models with
deep neural networks. However, most existing methods only consider temporal
dynamics without spatial modeling. We propose Deep Spatiotemporal Point Process
(DeepSTPP), a deep dynamics model that integrates spatiotemporal point
processes. Our method is flexible, efficient, and can accurately forecast
irregularly sampled events over space and time. The key construction of our
approach is the nonparametric space-time intensity function, governed by a
latent process. The intensity function enjoys closed-form integration for the
density. The latent process captures the uncertainty of the event sequence. We
use amortized variational inference to infer the latent process with deep
networks. Using synthetic datasets, we validate our model can accurately learn
the true intensity function. On real-world benchmark datasets, our model
demonstrates superior performance over state-of-the-art baselines.

    

### [[2112.06362] Scheduling Servers with Stochastic Bilinear Rewards](http://arxiv.org/abs/2112.06362)


  In this paper we study a multi-class, multi-server queueing system with
stochastic rewards of job-server assignments following a bilinear model in
feature vectors representing jobs and servers. Our goal is regret minimization
against an oracle policy that has a complete information about system
parameters. We propose a scheduling algorithm that uses a linear bandit
algorithm along with dynamic allocation of jobs to servers. For the baseline
setting, in which mean job service times are identical for all jobs, we show
that our algorithm has a sub-linear regret, as well as a sub-linear bound on
the mean queue length, in the horizon time. We further show that similar bounds
hold under more general assumptions, allowing for non-identical mean job
service times for different job classes and a time-varying set of server
classes. We also show that better regret and mean queue length bounds can be
guaranteed by an algorithm having access to traffic intensities of job classes.
We present results of numerical experiments demonstrating how regret and mean
queue length of our algorithms depend on various system parameters and compare
their performance against a previously proposed algorithm using synthetic
randomly generated data and a real-world cluster computing data trace.

    

### [[2112.06363] Risk and optimal policies in bandit experiments](http://arxiv.org/abs/2112.06363)


  This paper provides a decision theoretic analysis of bandit experiments. The
bandit setting corresponds to a dynamic programming problem, but solving this
directly is typically infeasible. Working within the framework of diffusion
asymptotics, we define a suitable notion of asymptotic Bayes risk for bandit
settings. For normally distributed rewards, the minimal Bayes risk can be
characterized as the solution to a nonlinear second-order partial differential
equation (PDE). Using a limit of experiments approach, we show that this PDE
characterization also holds asymptotically under both parametric and
non-parametric distribution of the rewards. The approach further describes the
state variables it is asymptotically sufficient to restrict attention to, and
therefore suggests a practical strategy for dimension reduction. The upshot is
that we can approximate the dynamic programming problem defining the bandit
setting with a PDE which can be efficiently solved using sparse matrix
routines. We derive near-optimal policies from the numerical solutions to these
equations. The proposed policies substantially dominate existing methods such
Thompson sampling. The framework also allows for substantial generalizations to
the bandit problem such as time discounting and pure exploration motives.

    

### [[2112.06370] Dependency Learning for Legal Judgment Prediction with a Unified Text-to-Text Transformer](http://arxiv.org/abs/2112.06370)


  Given the fact of a case, Legal Judgment Prediction (LJP) involves a series
of sub-tasks such as predicting violated law articles, charges and term of
penalty. We propose leveraging a unified text-to-text Transformer for LJP,
where the dependencies among sub-tasks can be naturally established within the
auto-regressive decoder. Compared with previous works, it has three advantages:
(1) it fits in the pretraining pattern of masked language models, and thereby
can benefit from the semantic prompts of each sub-task rather than treating
them as atomic labels, (2) it utilizes a single unified architecture, enabling
full parameter sharing across all sub-tasks, and (3) it can incorporate both
classification and generative sub-tasks. We show that this unified transformer,
albeit pretrained on general-domain text, outperforms pretrained models
tailored specifically for the legal domain. Through an extensive set of
experiments, we find that the best order to capture dependencies is different
from human intuitions, and the most reasonable logical order for humans can be
sub-optimal for the model. We further include two more auxiliary tasks: court
view generation and article content prediction, showing they can not only
improve the prediction accuracy, but also provide interpretable explanations
for model outputs even when an error is made. With the best configuration, our
model outperforms both previous SOTA and a single-tasked version of the unified
transformer by a large margin.

    

### [[2112.06377] Surfer100: Generating Surveys From Web Resources on Wikipedia-style](http://arxiv.org/abs/2112.06377)


  Fast-developing fields such as Artificial Intelligence (AI) often outpace the
efforts of encyclopedic sources such as Wikipedia, which either do not
completely cover recently-introduced topics or lack such content entirely. As a
result, methods for automatically producing content are valuable tools to
address this information overload. We show that recent advances in pretrained
language modeling can be combined for a two-stage extractive and abstractive
approach for Wikipedia lead paragraph generation. We extend this approach to
generate longer Wikipedia-style summaries with sections and examine how such
methods struggle in this application through detailed studies with 100
reference human-collected surveys. This is the first study on utilizing web
resources for long Wikipedia-style summaries to the best of our knowledge.

    

### [[2112.06380] Robust Voting Rules from Algorithmic Robust Statistics](http://arxiv.org/abs/2112.06380)


  In this work we study the problem of robustly learning a Mallows model. We
give an algorithm that can accurately estimate the central ranking even when a
constant fraction of its samples are arbitrarily corrupted. Moreover our
robustness guarantees are dimension-independent in the sense that our overall
accuracy does not depend on the number of alternatives being ranked. Our work
can be thought of as a natural infusion of perspectives from algorithmic robust
statistics into one of the central inference problems in voting and
information-aggregation. Specifically, our voting rule is efficiently
computable and its outcome cannot be changed by much by a large group of
colluding voters.

    

### [[2112.06384] WOOD: Wasserstein-based Out-of-Distribution Detection](http://arxiv.org/abs/2112.06384)


  The training and test data for deep-neural-network-based classifiers are
usually assumed to be sampled from the same distribution. When part of the test
samples are drawn from a distribution that is sufficiently far away from that
of the training samples (a.k.a. out-of-distribution (OOD) samples), the trained
neural network has a tendency to make high confidence predictions for these OOD
samples. Detection of the OOD samples is critical when training a neural
network used for image classification, object detection, etc. It can enhance
the classifier's robustness to irrelevant inputs, and improve the system
resilience and security under different forms of attacks. Detection of OOD
samples has three main challenges: (i) the proposed OOD detection method should
be compatible with various architectures of classifiers (e.g., DenseNet,
ResNet), without significantly increasing the model complexity and requirements
on computational resources; (ii) the OOD samples may come from multiple
distributions, whose class labels are commonly unavailable; (iii) a score
function needs to be defined to effectively separate OOD samples from
in-distribution (InD) samples. To overcome these challenges, we propose a
Wasserstein-based out-of-distribution detection (WOOD) method. The basic idea
is to define a Wasserstein-distance-based score that evaluates the
dissimilarity between a test sample and the distribution of InD samples. An
optimization problem is then formulated and solved based on the proposed score
function. The statistical learning bound of the proposed method is investigated
to guarantee that the loss value achieved by the empirical optimizer
approximates the global optimum. The comparison study results demonstrate that
the proposed WOOD consistently outperforms other existing OOD detection
methods.

    

### [[2112.06397] N-Cloth: Predicting 3D Cloth Deformation with Mesh-Based Networks](http://arxiv.org/abs/2112.06397)


  We present a novel mesh-based learning approach (N-Cloth) for plausible 3D
cloth deformation prediction. Our approach is general and can handle cloth or
obstacles represented by triangle meshes with arbitrary topology. We use graph
convolution to transform the cloth and object meshes into a latent space to
reduce the non-linearity in the mesh space. Our network can predict the target
3D cloth mesh deformation based on the state of the initial cloth mesh template
and the target obstacle mesh. Our approach can handle complex cloth meshes with
up to $100$K triangles and scenes with various objects corresponding to SMPL
humans, Non-SMPL humans, or rigid bodies. In practice, our approach
demonstrates good temporal coherence between successive input frames and can be
used to generate plausible cloth simulation at $30-45$ fps on an NVIDIA GeForce
RTX 3090 GPU. We highlight its benefits over prior learning-based methods and
physically-based cloth simulators.

    

### [[2112.06405] CSI Feedback with Model-Driven Deep Learning of Massive MIMO Systems](http://arxiv.org/abs/2112.06405)


  In order to achieve reliable communication with a high data rate of massive
multiple-input multiple-output (MIMO) systems in frequency division duplex
(FDD) mode, the estimated channel state information (CSI) at the receiver needs
to be fed back to the transmitter. However, the feedback overhead becomes
exorbitant with the increasing number of antennas. In this paper, a two stages
low rank (TSLR) CSI feedback scheme for millimeter wave (mmWave) massive MIMO
systems is proposed to reduce the feedback overhead based on model-driven deep
learning. Besides, we design a deep iterative neural network, named FISTA-Net,
by unfolding the fast iterative shrinkage thresholding algorithm (FISTA) to
achieve more efficient CSI feedback. Moreover, a shrinkage thresholding network
(ST-Net) is designed in FISTA-Net based on the attention mechanism, which can
choose the threshold adaptively. Simulation results show that the proposed TSLR
CSI feedback scheme and FISTA-Net outperform the existing algorithms in various
scenarios.

    

### [[2112.06409] Data Collection and Quality Challenges in Deep Learning: A Data-Centric AI Perspective](http://arxiv.org/abs/2112.06409)


  Software 2.0 is a fundamental shift in software engineering where machine
learning becomes the new software, powered by big data and computing
infrastructure. As a result, software engineering needs to be re-thought where
data becomes a first-class citizen on par with code. One striking observation
is that 80-90% of the machine learning process is spent on data preparation.
Without good data, even the best machine learning algorithms cannot perform
well. As a result, data-centric AI practices are now becoming mainstream.
Unfortunately, many datasets in the real world are small, dirty, biased, and
even poisoned. In this survey, we study the research landscape for data
collection and data quality primarily for deep learning applications. Data
collection is important because there is lesser need for feature engineering
for recent deep learning approaches, but instead more need for large amounts of
data. For data quality, we study data validation and data cleaning techniques.
Even if the data cannot be fully cleaned, we can still cope with imperfect data
during model training where using robust model training techniques. In
addition, while bias and fairness have been less studied in traditional data
management research, these issues become essential topics in modern machine
learning applications. We thus study fairness measures and unfairness
mitigation techniques that can be applied before, during, or after model
training. We believe that the data management community is well poised to solve
problems in these directions.

    

### [[2112.06410] How Good are Low-Rank Approximations in Gaussian Process Regression?](http://arxiv.org/abs/2112.06410)


  We provide guarantees for approximate Gaussian Process (GP) regression
resulting from two common low-rank kernel approximations: based on random
Fourier features, and based on truncating the kernel's Mercer expansion. In
particular, we bound the Kullback-Leibler divergence between an exact GP and
one resulting from one of the afore-described low-rank approximations to its
kernel, as well as between their corresponding predictive densities, and we
also bound the error between predictive mean vectors and between predictive
covariance matrices computed using the exact versus using the approximate GP.
We provide experiments on both simulated data and standard benchmarks to
evaluate the effectiveness of our theoretical bounds.

    

### [[1906.05473] Selective prediction-set models with coverage guarantees](http://arxiv.org/abs/1906.05473)


  Though black-box predictors are state-of-the-art for many complex tasks, they
often fail to properly quantify predictive uncertainty and may provide
inappropriate predictions for unfamiliar data. Instead, we can learn more
reliable models by letting them either output a prediction set or abstain when
the uncertainty is high. We propose training these selective prediction-set
models using an uncertainty-aware loss minimization framework, which unifies
ideas from decision theory and robust maximum likelihood. Moreover, since
black-box methods are not guaranteed to output well-calibrated prediction sets,
we show how to calculate point estimates and confidence intervals for the true
coverage of any selective prediction-set model, as well as a uniform mixture of
K set models obtained from K-fold sample-splitting. When applied to predicting
in-hospital mortality and length-of-stay for ICU patients, our model
outperforms existing approaches on both in-sample and out-of-sample age groups,
and our recalibration method provides accurate inference for prediction set
coverage.

    

### [[1909.03889] Recovery of Future Data via Convolution Nuclear Norm Minimization](http://arxiv.org/abs/1909.03889)


  This paper studies the problem of time series forecasting (TSF) from the
perspective of compressed sensing. First of all, we convert TSF into a more
inclusive problem called tensor completion with arbitrary sampling (TCAS),
which is to restore a tensor from a subset of its entries sampled in an
arbitrary manner. While it is known that, in the framework of Tucker
low-rankness, it is theoretically impossible to identify the target tensor
based on some arbitrarily selected entries, in this work we shall show that
TCAS is indeed tackleable in the light of a new concept called convolutional
low-rankness, which is a generalization of the well-known Fourier sparsity.
Then we introduce a convex program termed Convolution Nuclear Norm Minimization
(CNNM), and we prove that CNNM succeeds in solving TCAS as long as a sampling
condition--which depends on the convolution rank of the target tensor--is
obeyed. This theory provides a meaningful answer to the fundamental question of
what is the minimum sampling size needed for making a given number of
forecasts. Experiments on univariate time series, images and videos show
encouraging results.

    

### [[1910.09394] Generalised learning of time-series: Ornstein-Uhlenbeck processes](http://arxiv.org/abs/1910.09394)


  In machine learning, statistics, econometrics and statistical physics,
cross-validation (CV) is used asa standard approach in quantifying the
generalisation performance of a statistical model. A directapplication of CV in
time-series leads to the loss of serial correlations, a requirement of
preserving anynon-stationarity and the prediction of the past data using the
future data. In this work, we proposea meta-algorithm called reconstructive
cross validation (rCV ) that avoids all these issues. At first,k folds are
formed with non-overlapping randomly selected subsets of the original
time-series. Then,we generate k new partial time-series by removing data points
from a given fold: every new partialtime-series have missing points at random
from a different entire fold. A suitable imputation or asmoothing technique is
used to reconstruct k time-series. We call these reconstructions
secondarymodels. Thereafter, we build the primary k time-series models using
new time-series coming fromthe secondary models. The performance of the primary
models are evaluated simultaneously bycomputing the deviations from the
originally removed data points and out-of-sample (OSS) data.Full
cross-validation in time-series models can be practiced with rCV along with
generating learning curves.

    

### [[1912.09484] Zeroth-order Stochastic Compositional Algorithms for Risk-Aware Learning](http://arxiv.org/abs/1912.09484)


  We present $\textit{Free-MESSAGE}^{p}$, the first zeroth-order algorithm for
(weakly-)convex mean-semideviation-based risk-aware learning, which is also the
first three-level zeroth-order compositional stochastic optimization algorithm
whatsoever. Using a non-trivial extension of Nesterov's classical results on
Gaussian smoothing, we develop the $\textit{Free-MESSAGE}^{p}$ algorithm from
first principles, and show that it essentially solves a smoothed surrogate to
the original problem, the former being a uniform approximation of the latter,
in a useful, convenient sense. We then present a complete analysis of the
$\textit{Free-MESSAGE}^{p}$ algorithm, which establishes convergence in a
user-tunable neighborhood of the optimal solutions of the original problem for
convex costs, as well as explicit convergence rates for convex, weakly convex,
and strongly convex costs, and in a unified way. Orderwise, and for fixed
problem parameters, our results demonstrate no sacrifice in convergence speed
as compared to existing first-order methods, while striking a certain balance
among the condition of the problem, its dimensionality, as well as the accuracy
of the obtained results, naturally extending previous results in zeroth-order
risk-neutral learning.

    

### [[2002.05233] Learning Multi-Agent Coordination through Connectivity-driven Communication](http://arxiv.org/abs/2002.05233)


  In artificial multi-agent systems, the ability to learn collaborative
policies is predicated upon the agents' communication skills: they must be able
to encode the information received from the environment and learn how to share
it with other agents as required by the task at hand. We present a deep
reinforcement learning approach, Connectivity Driven Communication (CDC), that
facilitates the emergence of multi-agent collaborative behaviour only through
experience. The agents are modelled as nodes of a weighted graph whose
state-dependent edges encode pair-wise messages that can be exchanged. We
introduce a graph-dependent attention mechanisms that controls how the agents'
incoming messages are weighted. This mechanism takes into full account the
current state of the system as represented by the graph, and builds upon a
diffusion process that captures how the information flows on the graph. The
graph topology is not assumed to be known a priori, but depends dynamically on
the agents' observations, and is learnt concurrently with the attention
mechanism and policy in an end-to-end fashion. Our empirical results show that
CDC is able to learn effective collaborative policies and can over-perform
competing learning algorithms on cooperative navigation tasks.

    

### [[2003.10667] Quantum circuit-like learning: A fast and scalable classical machine-learning algorithm with similar performance to quantum circuit learning](http://arxiv.org/abs/2003.10667)


  The application of near-term quantum devices to machine learning (ML) has
attracted much attention. In one such attempt, Mitarai et al. (2018) proposed a
framework to use a quantum circuit for supervised ML tasks, which is called
quantum circuit learning (QCL). Due to the use of a quantum circuit, QCL can
employ an exponentially high-dimensional Hilbert space as its feature space.
However, its efficiency compared to classical algorithms remains unexplored. In
this study, using a statistical technique called count sketch, we propose a
classical ML algorithm that uses the same Hilbert space. In numerical
simulations, our proposed algorithm demonstrates similar performance to QCL for
several ML tasks. This provides a new perspective with which to consider the
computational and memory efficiency of quantum ML algorithms.

    

### [[2004.01571] Tree-AMP: Compositional Inference with Tree Approximate Message Passing](http://arxiv.org/abs/2004.01571)


  We introduce Tree-AMP, standing for Tree Approximate Message Passing, a
python package for compositional inference in high-dimensional tree-structured
models. The package provides a unifying framework to study several approximate
message passing algorithms previously derived for a variety of machine learning
tasks such as generalized linear models, inference in multi-layer networks,
matrix factorization, and reconstruction using non-separable penalties. For
some models, the asymptotic performance of the algorithm can be theoretically
predicted by the state evolution, and the measurements entropy estimated by the
free entropy formalism. The implementation is modular by design: each module,
which implements a factor, can be composed at will with other modules to solve
complex inference tasks. The user only needs to declare the factor graph of the
model: the inference algorithm, state evolution and entropy estimation are
fully automated.

    

### [[2005.14612] Non-Local Graph Neural Networks](http://arxiv.org/abs/2005.14612)


  Modern graph neural networks (GNNs) learn node embeddings through multilayer
local aggregation and achieve great success in applications on assortative
graphs. However, tasks on disassortative graphs usually require non-local
aggregation. In addition, we find that local aggregation is even harmful for
some disassortative graphs. In this work, we propose a simple yet effective
non-local aggregation framework with an efficient attention-guided sorting for
GNNs. Based on it, we develop various non-local GNNs. We perform thorough
experiments to analyze disassortative graph datasets and evaluate our non-local
GNNs. Experimental results demonstrate that our non-local GNNs significantly
outperform previous state-of-the-art methods on seven benchmark datasets of
disassortative graphs, in terms of both model performance and efficiency.

    

### [[2006.11405] M2P2: Multimodal Persuasion Prediction using Adaptive Fusion](http://arxiv.org/abs/2006.11405)


  Identifying persuasive speakers in an adversarial environment is a critical
task. In a national election, politicians would like to have persuasive
speakers campaign on their behalf. When a company faces adverse publicity, they
would like to engage persuasive advocates for their position in the presence of
adversaries who are critical of them. Debates represent a common platform for
these forms of adversarial persuasion. This paper solves two problems: the
Debate Outcome Prediction (DOP) problem predicts who wins a debate while the
Intensity of Persuasion Prediction (IPP) problem predicts the change in the
number of votes before and after a speaker speaks. Though DOP has been
previously studied, we are the first to study IPP. Past studies on DOP fail to
leverage two important aspects of multimodal data: 1) multiple modalities are
often semantically aligned, and 2) different modalities may provide diverse
information for prediction. Our M2P2 (Multimodal Persuasion Prediction)
framework is the first to use multimodal (acoustic, visual, language) data to
solve the IPP problem. To leverage the alignment of different modalities while
maintaining the diversity of the cues they provide, M2P2 devises a novel
adaptive fusion learning framework which fuses embeddings obtained from two
modules -- an alignment module that extracts shared information between
modalities and a heterogeneity module that learns the weights of different
modalities with guidance from three separately trained unimodal reference
models. We test M2P2 on the popular IQ2US dataset designed for DOP. We also
introduce a new dataset called QPS (from Qipashuo, a popular Chinese debate TV
show ) for IPP. M2P2 significantly outperforms 4 recent baselines on both
datasets.

    

### [[2006.16431] VAE-KRnet and its applications to variational Bayes](http://arxiv.org/abs/2006.16431)


  In this work, we have proposed a generative model, called VAE-KRnet, for
density estimation or approximation, which combines the canonical variational
autoencoder (VAE) with our recently developed flow-based generative model,
called KRnet. VAE is used as a dimension reduction technique to capture the
latent space, and KRnet is used to model the distribution of the latent
variable. Using a linear model between the data and the latent variable, we
show that VAE-KRnet can be more effective and robust than the canonical VAE.
VAE-KRnet can be used as a density model to approximate either data
distribution or an arbitrary probability density function (PDF) known up to a
constant. VAE-KRnet is flexible in terms of dimensionality. When the number of
dimensions is relatively small, KRnet can effectively approximate the
distribution in terms of the original random variable. For high-dimensional
cases, we may use VAE-KRnet to incorporate dimension reduction. One important
application of VAE-KRnet is the variational Bayes for the approximation of the
posterior distribution. The variational Bayes approaches are usually based on
the minimization of the Kullback-Leibler (KL) divergence between the model and
the posterior. For high-dimensional distributions, it is very challenging to
construct an accurate density model due to the curse of dimensionality, where
extra assumptions are often introduced for efficiency. For instance, the
classical mean-field approach assumes mutual independence between dimensions,
which often yields an underestimated variance due to oversimplification. To
alleviate this issue, we include into the loss the maximization of the mutual
information between the latent random variable and the original random
variable, which helps keep more information from the region of low density such
that the estimation of variance is improved.

    

### [[2007.03069] Outcome-Driven Dynamic Refugee Assignment with Allocation Balancing](http://arxiv.org/abs/2007.03069)


  This study proposes two new dynamic assignment algorithms to match refugees
and asylum seekers to geographic localities within a host country. The first,
currently implemented in a multi-year pilot in Switzerland, seeks to maximize
the average expected employment level (or any measured outcome of interest) of
refugees through a minimum-discord online assignment algorithm. Although the
proposed algorithm achieves near-optimal expected employment compared to the
hindsight-optimal solution, it can result in a periodically imbalanced
allocation to the localities over time. This leads to undesirable workload
inefficiencies for resettlement resources and agents, who cannot move between
localities. To address this problem, the second algorithm balances the goal of
improving refugee outcomes with the desire for an even allocation to each
locality over time. The performance of the proposed methods is illustrated
using real refugee resettlement data from one of the largest resettlement
agencies in the United States. On this dataset, we find that the allocation
balancing algorithm can achieve near-perfect balance over time with virtually
no loss in expected employment compared to the pure employment-maximizing
algorithm. In addition, the allocation balancing algorithm offers a number of
ancillary benefits, including robustness to unknown arrival flows and increased
resilience through greater exploration.

    

### [[2007.03838] Making Adversarial Examples More Transferable and Indistinguishable](http://arxiv.org/abs/2007.03838)


  Fast gradient sign attack series are popular methods that are used to
generate adversarial examples. However, most of the approaches based on fast
gradient sign attack series cannot balance the indistinguishability and
transferability due to the limitations of the basic sign structure. To address
this problem, we propose a method, called Adam Iterative Fast Gradient Tanh
Method (AI-FGTM), to generate indistinguishable adversarial examples with high
transferability. Besides, smaller kernels and dynamic step size are also
applied to generate adversarial examples for further increasing the attack
success rates. Extensive experiments on an ImageNet-compatible dataset show
that our method generates more indistinguishable adversarial examples and
achieves higher attack success rates without extra running time and resource.
Our best transfer-based attack NI-TI-DI-AITM can fool six classic defense
models with an average success rate of 89.3% and three advanced defense models
with an average success rate of 82.7%, which are higher than the
state-of-the-art gradient-based attacks. Additionally, our method can also
reduce nearly 20% mean perturbation. We expect that our method will serve as a
new baseline for generating adversarial examples with better transferability
and indistinguishability.

    

### [[2007.11120] On Linear Convergence of Policy Gradient Methods for Finite MDPs](http://arxiv.org/abs/2007.11120)


  We revisit the finite time analysis of policy gradient methods in the one of
the simplest settings: finite state and action MDPs with a policy class
consisting of all stochastic policies and with exact gradient evaluations.
There has been some recent work viewing this setting as an instance of smooth
non-linear optimization problems and showing sub-linear convergence rates with
small step-sizes. Here, we take a different perspective based on connections
with policy iteration and show that many variants of policy gradient methods
succeed with large step-sizes and attain a linear rate of convergence.

    

### [[2008.12249] PIGNet: A physics-informed deep learning model toward generalized drug-target interaction predictions](http://arxiv.org/abs/2008.12249)


  Recently, deep neural network (DNN)-based drug-target interaction (DTI)
models were highlighted for their high accuracy with affordable computational
costs. Yet, the models' insufficient generalization remains a challenging
problem in the practice of in-silico drug discovery. We propose two key
strategies to enhance generalization in the DTI model. The first is to predict
the atom-atom pairwise interactions via physics-informed equations
parameterized with neural networks and provides the total binding affinity of a
protein-ligand complex as their sum. We further improved the model
generalization by augmenting a broader range of binding poses and ligands to
training data. We validated our model, PIGNet, in the comparative assessment of
scoring functions (CASF) 2016, demonstrating the outperforming docking and
screening powers than previous methods. Our physics-informing strategy also
enables the interpretation of predicted affinities by visualizing the
contribution of ligand substructures, providing insights for further ligand
optimization.

    

### [[2010.01037] Encoded Prior Sliced Wasserstein AutoEncoder for learning latent manifold representations](http://arxiv.org/abs/2010.01037)


  While variational autoencoders have been successful in several tasks, the use
of conventional priors are limited in their ability to encode the underlying
structure of input data. We introduce an Encoded Prior Sliced Wasserstein
AutoEncoder wherein an additional prior-encoder network learns an embedding of
the data manifold which preserves topological and geometric properties of the
data, thus improving the structure of latent space. The autoencoder and
prior-encoder networks are iteratively trained using the Sliced Wasserstein
distance. The effectiveness of the learned manifold encoding is explored by
traversing latent space through interpolations along geodesics which generate
samples that lie on the data manifold and hence are more realistic compared to
Euclidean interpolation. To this end, we introduce a graph-based algorithm for
exploring the data manifold and interpolating along network-geodesics in latent
space by maximizing the density of samples along the path while minimizing
total energy. We use the 3D-spiral data to show that the prior encodes the
geometry underlying the data unlike conventional autoencoders, and to
demonstrate the exploration of the embedded data manifold through the network
algorithm. We apply our framework to benchmarked image datasets to demonstrate
the advantages of learning data representations in outlier generation, latent
structure, and geodesic interpolation.

    

### [[2010.04223] Fictitious play in zero-sum stochastic games](http://arxiv.org/abs/2010.04223)


  We present a novel variant of fictitious play dynamics combining classical
fictitious play with Q-learning for stochastic games and analyze its
convergence properties in two-player zero-sum stochastic games. Our dynamics
involves players forming beliefs on the opponent strategy and their own
continuation payoff (Q-function), and playing a greedy best response by using
the estimated continuation payoffs. Players update their beliefs from
observations of opponent actions. A key property of the learning dynamics is
that update of the beliefs on Q-functions occurs at a slower timescale than
update of the beliefs on strategies. We show both in the model-based and
model-free cases (without knowledge of player payoff functions and state
transition probabilities), the beliefs on strategies converge to a stationary
mixed Nash equilibrium of the zero-sum stochastic game.

    

### [[2010.09235] Ensemble Chinese End-to-End Spoken Language Understanding for Abnormal Event Detection from audio stream](http://arxiv.org/abs/2010.09235)


  Conventional spoken language understanding (SLU) consist of two stages, the
first stage maps speech to text by automatic speech recognition (ASR), and the
second stage maps text to intent by natural language understanding (NLU).
End-to-end SLU maps speech directly to intent through a single deep learning
model. Previous end-to-end SLU models are primarily used for English
environment due to lacking large scale SLU dataset in Chines, and use only one
ASR model to extract features from speech. With the help of Kuaishou
technology, a large scale SLU dataset in Chinese is collected to detect
abnormal event in their live audio stream. Based on this dataset, this paper
proposed a ensemble end-to-end SLU model used for Chinese environment. This
ensemble SLU models extracted hierarchies features using multiple pre-trained
ASR models, leading to better representation of phoneme level and word level
information. This proposed approached achieve 9.7% increase of accuracy
compared to previous end-to-end SLU model.

    

### [[2010.14641] Learning to Plan Optimistically: Uncertainty-Guided Deep Exploration via Latent Model Ensembles](http://arxiv.org/abs/2010.14641)


  Learning complex robot behaviors through interaction requires structured
exploration. Planning should target interactions with the potential to optimize
long-term performance, while only reducing uncertainty where conducive to this
objective. This paper presents Latent Optimistic Value Exploration (LOVE), a
strategy that enables deep exploration through optimism in the face of
uncertain long-term rewards. We combine latent world models with value function
estimation to predict infinite-horizon returns and recover associated
uncertainty via ensembling. The policy is then trained on an upper confidence
bound (UCB) objective to identify and select the interactions most promising to
improve long-term performance. We apply LOVE to visual robot control tasks in
continuous action spaces and demonstrate on average more than 20% improved
sample efficiency in comparison to state-of-the-art and other exploration
objectives. In sparse and hard to explore environments we achieve an average
improvement of over 30%.

    

### [[2010.14712] Socially-Compatible Behavior Design of Autonomous Vehicles with Verification on Real Human Data](http://arxiv.org/abs/2010.14712)


  As more and more autonomous vehicles (AVs) are being deployed on public
roads, designing socially compatible behaviors for them is becoming
increasingly important. In order to generate safe and efficient actions, AVs
need to not only predict the future behaviors of other traffic participants,
but also be aware of the uncertainties associated with such behavior
prediction. In this paper, we propose an uncertain-aware integrated prediction
and planning (UAPP) framework. It allows the AVs to infer the characteristics
of other road users online and generate behaviors optimizing not only their own
rewards, but also their courtesy to others, and their confidence regarding the
prediction uncertainties. We first propose the definitions for courtesy and
confidence. Based on that, their influences on the behaviors of AVs in
interactive driving scenarios are explored. Moreover, we evaluate the proposed
algorithm on naturalistic human driving data by comparing the generated
behavior against ground truth. Results show that the online inference can
significantly improve the human-likeness of the generated behaviors.
Furthermore, we find that human drivers show great courtesy to others, even for
those without right-of-way. We also find that such driving preferences vary
significantly in different cultures.

    

### [[2012.01775] DialogBERT: Discourse-Aware Response Generation via Learning to Recover and Rank Utterances](http://arxiv.org/abs/2012.01775)


  Recent advances in pre-trained language models have significantly improved
neural response generation. However, existing methods usually view the dialogue
context as a linear sequence of tokens and learn to generate the next word
through token-level self-attention. Such token-level encoding hinders the
exploration of discourse-level coherence among utterances. This paper presents
DialogBERT, a novel conversational response generation model that enhances
previous PLM-based dialogue models. DialogBERT employs a hierarchical
Transformer architecture. To efficiently capture the discourse-level coherence
among utterances, we propose two training objectives, including masked
utterance regression and distributed utterance order ranking in analogy to the
original BERT training. Experiments on three multi-turn conversation datasets
show that our approach remarkably outperforms the baselines, such as BART and
DialoGPT, in terms of quantitative evaluation. The human evaluation suggests
that DialogBERT generates more coherent, informative, and human-like responses
than the baselines with significant margins.

    

### [[2101.08937] Prior Preference Learning from Experts:Designing a Reward with Active Inference](http://arxiv.org/abs/2101.08937)


  Active inference may be defined as Bayesian modeling of a brain with a
biologically plausible model of the agent. Its primary idea relies on the free
energy principle and the prior preference of the agent. An agent will choose an
action that leads to its prior preference for a future observation. In this
paper, we claim that active inference can be interpreted using reinforcement
learning (RL) algorithms and find a theoretical connection between them. We
extend the concept of expected free energy (EFE), which is a core quantity in
active inference, and claim that EFE can be treated as a negative value
function. Motivated by the concept of prior preference and a theoretical
connection, we propose a simple but novel method for learning a prior
preference from experts. This illustrates that the problem with inverse RL can
be approached with a new perspective of active inference. Experimental results
of prior preference learning show the possibility of active inference with
EFE-based rewards and its application to an inverse RL problem.

    

### [[2101.09500] Disentangled Sequence Clustering for Human Intention Inference](http://arxiv.org/abs/2101.09500)


  Equipping robots with the ability to infer human intent is a vital
precondition for effective collaboration. Most computational approaches towards
this objective employ probabilistic reasoning to recover a distribution of
"intent" conditioned on the robot's perceived sensory state. However, these
approaches typically assume task-specific notions of human intent (e.g.
labelled goals) are known a priori. To overcome this constraint, we propose the
Disentangled Sequence Clustering Variational Autoencoder (DiSCVAE), a
clustering framework that can be used to learn such a distribution of intent in
an unsupervised manner. The DiSCVAE leverages recent advances in unsupervised
learning to derive a disentangled latent representation of sequential data,
separating time-varying local features from time-invariant global aspects.
Though unlike previous frameworks for disentanglement, the proposed variant
also infers a discrete variable to form a latent mixture model and enable
clustering of global sequence concepts, e.g. intentions from observed human
behaviour. To evaluate the DiSCVAE, we first validate its capacity to discover
classes from unlabelled sequences using video datasets of bouncing digits and
2D animations. We then report results from a real-world human-robot interaction
experiment conducted on a robotic wheelchair. Our findings glean insights into
how the inferred discrete variable coincides with human intent and thus serves
to improve assistance in collaborative settings, such as shared control.

    

### [[2102.07211] Efficient Designs of SLOPE Penalty Sequences in Finite Dimension](http://arxiv.org/abs/2102.07211)


  In linear regression, SLOPE is a new convex analysis method that generalizes
the Lasso via the sorted L1 penalty: larger fitted coefficients are penalized
more heavily. This magnitude-dependent regularization requires an input of
penalty sequence $\lambda$, instead of a scalar penalty as in the Lasso case,
thus making the design extremely expensive in computation. In this paper, we
propose two efficient algorithms to design the possibly high-dimensional SLOPE
penalty, in order to minimize the mean squared error. For Gaussian data
matrices, we propose a first order Projected Gradient Descent (PGD) under the
Approximate Message Passing regime. For general data matrices, we present a
zero-th order Coordinate Descent (CD) to design a sub-class of SLOPE, referred
to as the k-level SLOPE. Our CD allows a useful trade-off between the accuracy
and the computation speed. We demonstrate the performance of SLOPE with our
designs via extensive experiments on synthetic data and real-world datasets.

    

### [[2102.07987] The Elliptical Potential Lemma for General Distributions with an Application to Linear Thompson Sampling](http://arxiv.org/abs/2102.07987)


  In this note, we introduce a general version of the well-known elliptical
potential lemma that is a widely used technique in the analysis of algorithms
in sequential learning and decision-making problems. We consider a stochastic
linear bandit setting where a decision-maker sequentially chooses among a set
of given actions, observes their noisy rewards, and aims to maximize her
cumulative expected reward over a decision-making horizon. The elliptical
potential lemma is a key tool for quantifying uncertainty in estimating
parameters of the reward function, but it requires the noise and the prior
distributions to be Gaussian. Our general elliptical potential lemma relaxes
this Gaussian requirement which is a highly non-trivial extension for a number
of reasons; unlike the Gaussian case, there is no closed-form solution for the
covariance matrix of the posterior distribution, the covariance matrix is not a
deterministic function of the actions, and the covariance matrix is not
decreasing with respect to the semidefinite inequality. While this result is of
broad interest, we showcase an application of it to prove an improved Bayesian
regret bound for the well-known Thompson sampling algorithm in stochastic
linear bandits with changing action sets where prior and noise distributions
are general. This bound is minimax optimal up to constants.

    

### [[2103.00360] Exploration and Incentives in Reinforcement Learning](http://arxiv.org/abs/2103.00360)


  How do you incentivize self-interested agents to $\textit{explore}$ when they
prefer to $\textit{exploit}$? We consider complex exploration problems, where
each agent faces the same (but unknown) MDP. In contrast with traditional
formulations of reinforcement learning, agents control the choice of policies,
whereas an algorithm can only issue recommendations. However, the algorithm
controls the flow of information, and can incentivize the agents to explore via
information asymmetry. We design an algorithm which explores all reachable
states in the MDP. We achieve provable guarantees similar to those for
incentivizing exploration in static, stateless exploration problems studied
previously. To the best of our knowledge, this is the first work to consider
mechanism design in a stateful, reinforcement learning setting.

    

### [[2103.14653] Quantum Self-Supervised Learning](http://arxiv.org/abs/2103.14653)


  The resurgence of self-supervised learning, whereby a deep learning model
generates its own supervisory signal from the data, promises a scalable way to
tackle the dramatically increasing size of real-world data sets without human
annotation. However, the staggering computational complexity of these methods
is such that for state-of-the-art performance, classical hardware requirements
represent a significant bottleneck to further progress. Here we take the first
steps to understanding whether quantum neural networks could meet the demand
for more powerful architectures and test its effectiveness in
proof-of-principle hybrid experiments. Interestingly, we observe a numerical
advantage for the learning of visual representations using small-scale quantum
neural networks over equivalently structured classical networks, even when the
quantum circuits are sampled with only 100 shots. Furthermore, we apply our
best quantum model to classify unseen images on the ibmq\_paris quantum
computer and find that current noisy devices can already achieve equal accuracy
to the equivalent classical model on downstream tasks.

    

### [[2104.05043] Learn Goal-Conditioned Policy with Intrinsic Motivation for Deep Reinforcement Learning](http://arxiv.org/abs/2104.05043)


  It is of significance for an agent to learn a widely applicable and
general-purpose policy that can achieve diverse goals including images and text
descriptions. Considering such perceptually-specific goals, the frontier of
deep reinforcement learning research is to learn a goal-conditioned policy
without hand-crafted rewards. To learn this kind of policy, recent works
usually take as the reward the non-parametric distance to a given goal in an
explicit embedding space. From a different viewpoint, we propose a novel
unsupervised learning approach named goal-conditioned policy with intrinsic
motivation (GPIM), which jointly learns both an abstract-level policy and a
goal-conditioned policy. The abstract-level policy is conditioned on a latent
variable to optimize a discriminator and discovers diverse states that are
further rendered into perceptually-specific goals for the goal-conditioned
policy. The learned discriminator serves as an intrinsic reward function for
the goal-conditioned policy to imitate the trajectory induced by the
abstract-level policy. Experiments on various robotic tasks demonstrate the
effectiveness and efficiency of our proposed GPIM method which substantially
outperforms prior techniques.

    

### [[2104.06826] Towards Automatic Model Specialization for Edge Video Analytics](http://arxiv.org/abs/2104.06826)


  Judging by popular and generic computer vision challenges, such as the
ImageNet or PASCAL VOC, neural networks have proven to be exceptionally
accurate in recognition tasks. However, state-of-the-art accuracy often comes
at a high computational price, requiring hardware acceleration to achieve
real-time performance, while use cases, such as smart cities, require images
from fixed cameras to be analyzed in real-time. Due to the amount of network
bandwidth these streams would generate, we cannot rely on offloading compute to
a centralized cloud. Thus, a distributed edge cloud is expected to process
images locally. However, the edge is, by nature, resource-constrained, which
puts a limit on the computational complexity that can execute. Yet, there is a
need for a meeting point between the edge and accurate real-time video
analytics. Specializing lightweight models on a per-camera basis may help but
it quickly becomes unfeasible as the number of cameras grows unless the process
is automated. In this paper, we present and evaluate COVA (Contextually
Optimized Video Analytics), a framework to assist in the automatic
specialization of models for video analytics in edge cameras. COVA
automatically improves the accuracy of lightweight models through their
specialization. Moreover, we discuss and review each step involved in the
process to understand the different trade-offs that each one entails.
Additionally, we show how the sole assumption of static cameras allows us to
make a series of considerations that greatly simplify the scope of the problem.
Finally, experiments show that state-of-the-art models, i.e., able to
generalize to unseen environments, can be effectively used as teachers to
tailor smaller networks to a specific context, boosting accuracy at a constant
computational cost. Results show that our COVA can automatically improve
accuracy of pre-trained models by an average of 21%.

    

### [[2104.10340] CVLight: Decentralized Learning for Adaptive Traffic Signal Control with Connected Vehicles](http://arxiv.org/abs/2104.10340)


  This paper develops a decentralized reinforcement learning (RL) scheme for
multi-intersection adaptive traffic signal control (TSC), called "CVLight",
that leverages data collected from connected vehicles (CVs). The state and
reward design facilitates coordination among agents and considers travel delays
collected by CVs. A novel algorithm, Asymmetric Advantage Actor-critic
(Asym-A2C), is proposed where both CV and non-CV information is used to train
the critic network, while only CV information is used to execute optimal signal
timing. Comprehensive experiments show the superiority of CVLight over
state-of-the-art algorithms under a 2-by-2 synthetic road network with various
traffic demand patterns and penetration rates. The learned policy is then
visualized to further demonstrate the advantage of Asym-A2C. A pre-train
technique is applied to improve the scalability of CVLight, which significantly
shortens the training time and shows the advantage in performance under a
5-by-5 road network. A case study is performed on a 2-by-2 road network located
in State College, Pennsylvania, USA, to further demonstrate the effectiveness
of the proposed algorithm under real-world scenarios. Compared to other
baseline models, the trained CVLight agent can efficiently control multiple
intersections solely based on CV data and achieve the best performance,
especially under low CV penetration rates.

    

### [[2104.14787] A User-Guided Bayesian Framework for Ensemble Feature Selection in Life Science Applications (UBayFS)](http://arxiv.org/abs/2104.14787)


  Feature selection represents a measure to reduce the complexity of
high-dimensional datasets and gain insights into the systematic variation in
the data. This aspect is of specific importance in domains that rely on model
interpretability, such as life sciences. We propose UBayFS, an ensemble feature
selection technique embedded in a Bayesian statistical framework. Our approach
considers two sources of information: data and domain knowledge. We build a
meta-model from an ensemble of elementary feature selectors and aggregate this
information in a multinomial likelihood. The user guides UBayFS by weighting
features and penalizing specific feature blocks or combinations, implemented
via a Dirichlet-type prior distribution and a regularization term. In a
quantitative evaluation, we demonstrate that our framework (a) allows for a
balanced trade-off between user knowledge and data observations, and (b)
achieves competitive performance with state-of-the-art methods.

    

### [[2105.00579] BACKDOORL: Backdoor Attack against Competitive Reinforcement Learning](http://arxiv.org/abs/2105.00579)


  Recent research has confirmed the feasibility of backdoor attacks in deep
reinforcement learning (RL) systems. However, the existing attacks require the
ability to arbitrarily modify an agent's observation, constraining the
application scope to simple RL systems such as Atari games. In this paper, we
migrate backdoor attacks to more complex RL systems involving multiple agents
and explore the possibility of triggering the backdoor without directly
manipulating the agent's observation. As a proof of concept, we demonstrate
that an adversary agent can trigger the backdoor of the victim agent with its
own action in two-player competitive RL systems. We prototype and evaluate
BACKDOORL in four competitive environments. The results show that when the
backdoor is activated, the winning rate of the victim drops by 17% to 37%
compared to when not activated.

    

### [[2105.02446] DiffSinger: Singing Voice Synthesis via Shallow Diffusion Mechanism](http://arxiv.org/abs/2105.02446)


  Singing voice synthesis (SVS) systems are built to synthesize high-quality
and expressive singing voice, in which the acoustic model generates the
acoustic features (eg, mel-spectrogram) given a music score. Previous singing
acoustic models adopt a simple loss (eg, L1 and L2) or generative adversarial
network (GAN) to reconstruct the acoustic features, while they suffer from
over-smoothing and unstable training issues respectively, which hinder the
naturalness of synthesized singing. In this work, we propose DiffSinger, an
acoustic model for SVS based on the diffusion probabilistic model. DiffSinger
is a parameterized Markov chain that iteratively converts the noise into
mel-spectrogram conditioned on the music score. By implicitly optimizing
variational bound, DiffSinger can be stably trained and generate realistic
outputs. To further improve the voice quality and speed up inference, we
introduce a shallow diffusion mechanism to make better use of the prior
knowledge learned by the simple loss. Specifically, DiffSinger starts
generation at a shallow step smaller than the total number of diffusion steps,
according to the intersection of the diffusion trajectories of the ground-truth
mel-spectrogram and the one predicted by a simple mel-spectrogram decoder.
Besides, we propose boundary prediction methods to locate the intersection and
determine the shallow step adaptively. The evaluations conducted on a Chinese
singing dataset demonstrate that DiffSinger outperforms state-of-the-art SVS
work. Extensional experiments also prove the generalization of our methods on
text-to-speech task (DiffSpeech). Audio samples are available via
\url{this https URL}.

    

### [[2105.07024] A hyperparameter-tuning approach to automated inverse planning](http://arxiv.org/abs/2105.07024)


  Radiotherapy inverse planning often requires planners to modify parameters in
the treatment planning system's objective function to produce clinically
acceptable plans. Due to the manual steps in this process, plan quality can
vary depending on the planning time available and the planner's skills. This
study investigates two hyperparameter-tuning methods for automated inverse
planning. Because this framework does not train a model on previously-optimized
plans, it can be readily adapted to practice pattern changes, and plan quality
is not limited by that of a training cohort. We selected 10 patients who
received lung SBRT using manually-generated clinical plans. We used random
sampling (RS) and Bayesian optimization (BO) to tune parameters using
linear-quadratic utility functions based on 11 clinical goals. Normalizing all
plans to have PTV D95 equal to 48 Gy, we compared plan quality for the
automatically-generated and manually-generated plans. We also investigated the
impact of iteration count on the automatically-generated plans, comparing
planning time and plan utility for RS and BO plans with and without stopping
criteria. Without stopping criteria, the median planning time was 1.9 and 2.3
hours for RS and BO plans. The OAR doses in the RS and BO plans had a median
percent difference (MPD) of 48.7% and 60.4% below clinical dose limits and an
MPD of 2.8% and 3.3% below clinical plan doses. With stopping criteria, the
utility decreased by an MPD of 5.3% and 3.9% for RS and BO plans, but the
median planning time was reduced to 0.5 and 0.7 hours, and the OAR doses still
had an MPD of 42.9% and 49.7% below clinical dose limits and an MPD of 0.3% and
1.8% below clinical plan doses. This study demonstrates that
hyperparameter-tuning approaches to automated inverse planning can reduce
active planning time with plan quality that is similar to or better than
manually-generated plans.

    

### [[2105.07190] A Comprehensive Taxonomy for Explainable Artificial Intelligence: A Systematic Survey of Surveys on Methods and Concepts](http://arxiv.org/abs/2105.07190)


  In the meantime, a wide variety of terminologies, motivations, approaches and
evaluation criteria have been developed within the research field of
explainable artificial intelligence (XAI). With the amount of XAI methods
vastly growing, a taxonomy of methods is needed by researchers as well as
practitioners: To grasp the breadth of the topic, compare methods, and to
select the right XAI method based on traits required by a specific use-case
context. In the literature many taxonomies for XAI methods of varying level of
detail and depth can be found. While they often have a different focus, they
also exhibit many points of overlap. This paper unifies these efforts, and
provides a taxonomy of XAI methods that is complete with respect to notions
present in the current state-of-research. In a structured literature analysis
and meta-study we identified and reviewed more than 50 of the most cited and
current surveys on XAI methods, metrics, and method traits. After summarizing
them in a survey of surveys, we merge terminologies and concepts of the
articles into a unified structured taxonomy. Single concepts therein are
illustrated by in total more than 50 diverse selected example methods, which we
categorize accordingly. The taxonomy may serve both beginners, researchers, and
practitioners as a reference and wide-ranging overview on XAI method traits and
aspects. Hence, it provides foundations for targeted, use-case-oriented, and
context-sensitive future research.

    

### [[2105.10585] Properties of the After Kernel](http://arxiv.org/abs/2105.10585)


  The Neural Tangent Kernel (NTK) is the wide-network limit of a kernel defined
using neural networks at initialization, whose embedding is the gradient of the
output of the network with respect to its parameters. We study the "after
kernel", which is defined using the same embedding, except after training, for
neural networks with standard architectures, on binary classification problems
extracted from MNIST and CIFAR-10, trained using SGD in a standard way. For
some dataset-architecture pairs, after a few epochs of neural network training,
a hard-margin SVM using the network's after kernel is much more accurate than
when the network's initial kernel is used. For networks with an architecture
similar to VGG, the after kernel is more "global", in the sense that it is less
invariant to transformations of input images that disrupt the global structure
of the image while leaving the local statistics largely intact. For fully
connected networks, the after kernel is less global in this sense. The after
kernel tends to be more invariant to small shifts, rotations and zooms; data
augmentation does not improve these invariances. The (finite approximation to
the) conjugate kernel, obtained using the last layer of hidden nodes,
sometimes, but not always, provides a good approximation to the NTK and the
after kernel.
Training a network with a larger learning rate (while holding the training
error constant) produces a better kernel, as measured by the test error of a
hard-margin SVM. The after kernels of networks trained with larger learning
rates tend to be more global, and more invariant to small shifts, rotations and
zooms.

    

### [[2105.12204] Safe Value Functions](http://arxiv.org/abs/2105.12204)


  Safety constraints and optimality are important, but sometimes conflicting
criteria for controllers. Although these criteria are often solved separately
with different tools to maintain formal guarantees, it is also common practice
in reinforcement learning to simply modify reward functions by penalizing
failures, with the penalty treated as a mere heuristic. We rigorously examine
the relationship of both safety and optimality to penalties, and formalize
sufficient conditions for safe value functions: value functions that are both
optimal for a given task, and enforce safety constraints. We reveal the
structure of this relationship through a proof of strong duality, showing that
there always exists a finite penalty that induces a safe value function. This
penalty is not unique, but upper-unbounded: larger penalties do not harm
optimality. Although it is often not possible to compute the minimum required
penalty, we reveal clear structure of how the penalty, rewards, discount
factor, and dynamics interact. This insight suggests practical, theory-guided
heuristics to design reward functions for control problems where safety is
important.

    

### [[2105.13553] A Machine Learning and Computer Vision Approach to Rapidly Optimize Multiscale Droplet Generation](http://arxiv.org/abs/2105.13553)


  Generating droplets from a continuous stream of fluid requires precise tuning
of a device to find optimized control parameter conditions. It is analytically
intractable to compute the necessary control parameter values of a
droplet-generating device that produces optimized droplets. Furthermore, as the
length scale of the fluid flow changes, the formation physics and optimized
conditions that induce flow decomposition into droplets also change. Hence, a
single proportional integral derivative controller is too inflexible to
optimize devices of different length scales or different control parameters,
while classification machine learning techniques take days to train and require
millions of droplet images. Therefore, the question is posed, can a single
method be created that universally optimizes multiple length-scale droplets
using only a few data points and is faster than previous approaches? In this
paper, a Bayesian optimization and computer vision feedback loop is designed to
quickly and reliably discover the control parameter values that generate
optimized droplets within different length-scale devices. This method is
demonstrated to converge on optimum parameter values using 60 images in only
2.3 hours, 30x faster than previous approaches. Model implementation is
demonstrated for two different length-scale devices: a milliscale inkjet device
and a micofluidics device.

    

### [[2106.00467] The Zoo of Fairness metrics in Machine Learning](http://arxiv.org/abs/2106.00467)


  In recent years, the problem of addressing fairness in Machine Learning (ML)
and automatic decision-making has attracted a lot of attention in the
scientific communities dealing with Artificial Intelligence. A plethora of
different definitions of fairness in ML have been proposed, that consider
different notions of what is a "fair decision" in situations impacting
individuals in the population. The precise differences, implications and
"orthogonality" between these notions have not yet been fully analyzed in the
literature. In this work, we try to make some order out of this zoo of
definitions.

    

### [[2106.02585] A Procedural World Generation Framework for Systematic Evaluation of Continual Learning](http://arxiv.org/abs/2106.02585)


  Several families of continual learning techniques have been proposed to
alleviate catastrophic interference in deep neural network training on
non-stationary data. However, a comprehensive comparison and analysis of
limitations remains largely open due to the inaccessibility to suitable
datasets. Empirical examination not only varies immensely between individual
works, it further currently relies on contrived composition of benchmarks
through subdivision and concatenation of various prevalent static vision
datasets. In this work, our goal is to bridge this gap by introducing a
computer graphics simulation framework that repeatedly renders only upcoming
urban scene fragments in an endless real-time procedural world generation
process. At its core lies a modular parametric generative model with adaptable
generative factors. The latter can be used to flexibly compose data streams,
which significantly facilitates a detailed analysis and allows for effortless
investigation of various continual learning schemes.

    

### [[2106.02748] Decentralized Q-Learning in Zero-sum Markov Games](http://arxiv.org/abs/2106.02748)


  We study multi-agent reinforcement learning (MARL) in infinite-horizon
discounted zero-sum Markov games. We focus on the practical but challenging
setting of decentralized MARL, where agents make decisions without coordination
by a centralized controller, but only based on their own payoffs and local
actions executed. The agents need not observe the opponent's actions or
payoffs, possibly being even oblivious to the presence of the opponent, nor be
aware of the zero-sum structure of the underlying game, a setting also referred
to as radically uncoupled in the literature of learning in games. In this
paper, we develop a radically uncoupled Q-learning dynamics that is both
rational and convergent: the learning dynamics converges to the best response
to the opponent's strategy when the opponent follows an asymptotically
stationary strategy; when both agents adopt the learning dynamics, they
converge to the Nash equilibrium of the game. The key challenge in this
decentralized setting is the non-stationarity of the environment from an
agent's perspective, since both her own payoffs and the system evolution depend
on the actions of other agents, and each agent adapts her policies
simultaneously and independently. To address this issue, we develop a
two-timescale learning dynamics where each agent updates her local Q-function
and value function estimates concurrently, with the latter happening at a
slower timescale.

    

### [[2106.07898] Divergence Frontiers for Generative Models: Sample Complexity, Quantization Effects, and Frontier Integrals](http://arxiv.org/abs/2106.07898)


  The spectacular success of deep generative models calls for quantitative
tools to measure their statistical performance. Divergence frontiers have
recently been proposed as an evaluation framework for generative models, due to
their ability to measure the quality-diversity trade-off inherent to deep
generative modeling. We establish non-asymptotic bounds on the sample
complexity of divergence frontiers. We also introduce frontier integrals which
provide summary statistics of divergence frontiers. We show how smoothed
estimators such as Good-Turing or Krichevsky-Trofimov can overcome the missing
mass problem and lead to faster rates of convergence. We illustrate the
theoretical results with numerical examples from natural language processing
and computer vision.

    

### [[2106.09857] Effective Model Sparsification by Scheduled Grow-and-Prune Methods](http://arxiv.org/abs/2106.09857)


  Deep neural networks (DNNs) are effective in solving many real-world
problems. Larger DNN models usually exhibit better quality (e.g., accuracy) but
their excessive computation results in long inference time. Model
sparsification can reduce the computation and memory cost while maintaining
model quality. Most existing sparsification algorithms unidirectionally remove
weights, while others randomly or greedily explore a small subset of weights in
each layer for pruning. The limitations of these algorithms reduce the level of
achievable sparsity. In addition, many algorithms still require pre-trained
dense models and thus suffer from large memory footprint. In this paper, we
propose a novel scheduled grow-and-prune (GaP) methodology without having to
pre-train a dense model. It addresses the shortcomings of the previous works by
repeatedly growing a subset of layers to dense and then pruning them back to
sparse after some training. Experiments show that the models pruned using the
proposed methods match or beat the quality of the highly optimized dense models
at 80% sparsity on a variety of tasks, such as image classification, objective
detection, 3D object part segmentation, and translation. They also outperform
other state-of-the-art (SOTA) methods for model sparsification. As an example,
a 90% non-uniform sparse ResNet-50 model obtained via GaP achieves 77.9% top-1
accuracy on ImageNet, improving the previous SOTA results by 1.5%. All code
will be publicly released.

    

### [[2106.09898] Bad Characters: Imperceptible NLP Attacks](http://arxiv.org/abs/2106.09898)


  Several years of research have shown that machine-learning systems are
vulnerable to adversarial examples, both in theory and in practice. Until now,
such attacks have primarily targeted visual models, exploiting the gap between
human and machine perception. Although text-based models have also been
attacked with adversarial examples, such attacks struggled to preserve semantic
meaning and indistinguishability. In this paper, we explore a large class of
adversarial examples that can be used to attack text-based models in a
black-box setting without making any human-perceptible visual modification to
inputs. We use encoding-specific perturbations that are imperceptible to the
human eye to manipulate the outputs of a wide range of Natural Language
Processing (NLP) systems from neural machine-translation pipelines to web
search engines. We find that with a single imperceptible encoding injection --
representing one invisible character, homoglyph, reordering, or deletion -- an
attacker can significantly reduce the performance of vulnerable models, and
with three injections most models can be functionally broken. Our attacks work
against currently-deployed commercial systems, including those produced by
Microsoft and Google, in addition to open source models published by Facebook,
IBM, and HuggingFace. This novel series of attacks presents a significant
threat to many language processing systems: an attacker can affect systems in a
targeted manner without any assumptions about the underlying model. We conclude
that text-based NLP systems require careful input sanitization, just like
conventional applications, and that given such systems are now being deployed
rapidly at scale, the urgent attention of architects and operators is required.

    

### [[2106.10316] Proper Value Equivalence](http://arxiv.org/abs/2106.10316)


  One of the main challenges in model-based reinforcement learning (RL) is to
decide which aspects of the environment should be modeled. The
value-equivalence (VE) principle proposes a simple answer to this question: a
model should capture the aspects of the environment that are relevant for
value-based planning. Technically, VE distinguishes models based on a set of
policies and a set of functions: a model is said to be VE to the environment if
the Bellman operators it induces for the policies yield the correct result when
applied to the functions. As the number of policies and functions increase, the
set of VE models shrinks, eventually collapsing to a single point corresponding
to a perfect model. A fundamental question underlying the VE principle is thus
how to select the smallest sets of policies and functions that are sufficient
for planning. In this paper we take an important step towards answering this
question. We start by generalizing the concept of VE to order-$k$ counterparts
defined with respect to $k$ applications of the Bellman operator. This leads to
a family of VE classes that increase in size as $k \rightarrow \infty$. In the
limit, all functions become value functions, and we have a special
instantiation of VE which we call proper VE or simply PVE. Unlike VE, the PVE
class may contain multiple models even in the limit when all value functions
are used. Crucially, all these models are sufficient for planning, meaning that
they will yield an optimal policy despite the fact that they may ignore many
aspects of the environment. We construct a loss function for learning PVE
models and argue that popular algorithms such as MuZero can be understood as
minimizing an upper bound for this loss. We leverage this connection to propose
a modification to MuZero and show that it can lead to improved performance in
practice.

    

### [[2106.12417] False perfection in machine prediction: Detecting and assessing circularity problems in machine learning](http://arxiv.org/abs/2106.12417)


  This paper is an excerpt of an early version of Chapter 2 of the book
"Validity, Reliability, and Significance. Empirical Methods for NLP and Data
Science", by Stefan Riezler and Michael Hagmann, published in December 2021 by
Morgan & Claypool. Please see the book's homepage at
this https URL
for a more recent and comprehensive discussion.

    

### [[2106.13876] Rationale-Inspired Natural Language Explanations with Commonsense](http://arxiv.org/abs/2106.13876)


  Extractive rationales (i.e., subsets of input features) and natural language
explanations (NLEs) are two predominant types of explanations for machine
learning models. While NLEs can be more comprehensive than extractive
rationales, machine-generated NLEs have been shown to fall short in terms of
commonsense knowledge. In this paper, we show that commonsense knowledge can
act as a bridge between extractive rationales and NLEs, rendering both types of
explanations better. We introduce a self-rationalizing framework, called RExC,
that (1) extracts rationales as most responsible features for the predictions,
(2) expands the extractive rationales using commonsense resources, and (3)
selects the best-suited commonsense knowledge to generate NLEs and give the
final prediction. Our framework surpasses by a large margin the previous
state-of-the-art in generating NLEs across five tasks in both natural language
and vision-language understanding. Self-rationalization with commonsense also
strongly improves the quality of the extractive rationale and task performances
over the previous best performing models that also produce explanations.

    

### [[2107.00363] Valid prediction intervals for regression problems](http://arxiv.org/abs/2107.00363)


  Over the last few decades, various methods have been proposed for estimating
prediction intervals in regression settings, including Bayesian methods,
ensemble methods, direct interval estimation methods and conformal prediction
methods. An important issue is the calibration of these methods: the generated
prediction intervals should have a predefined coverage level, without being
overly conservative. In this work, we review the above four classes of methods
from a conceptual and experimental point of view. Results on benchmark data
sets from various domains highlight large fluctuations in performance from one
data set to another. These observations can be attributed to the violation of
certain assumptions that are inherent to some classes of methods. We illustrate
how conformal prediction can be used as a general calibration procedure for
methods that deliver poor results without a calibration step.

    

### [[2108.10241] Back to the Drawing Board: A Critical Evaluation of Poisoning Attacks on Production Federated Learning](http://arxiv.org/abs/2108.10241)


  While recent works have indicated that federated learning (FL) may be
vulnerable to poisoning attacks by compromised clients, their real impact on
production FL systems is not fully understood. In this work, we aim to develop
a comprehensive systemization for poisoning attacks on FL by enumerating all
possible threat models, variations of poisoning, and adversary capabilities. We
specifically put our focus on untargeted poisoning attacks, as we argue that
they are significantly relevant to production FL deployments.
We present a critical analysis of untargeted poisoning attacks under
practical, production FL environments by carefully characterizing the set of
realistic threat models and adversarial capabilities. Our findings are rather
surprising: contrary to the established belief, we show that FL is highly
robust in practice even when using simple, low-cost defenses. We go even
further and propose novel, state-of-the-art data and model poisoning attacks,
and show via an extensive set of experiments across three benchmark datasets
how (in)effective poisoning attacks are in the presence of simple defense
mechanisms. We aim to correct previous misconceptions and offer concrete
guidelines to conduct more accurate (and more realistic) research on this
topic.

    

### [[2111.04665] Evaluating Predictive Uncertainty and Robustness to Distributional Shift Using Real World Data](http://arxiv.org/abs/2111.04665)


  Most machine learning models operate under the assumption that the training,
testing and deployment data is independent and identically distributed
(i.i.d.). This assumption doesn't generally hold true in a natural setting.
Usually, the deployment data is subject to various types of distributional
shifts. The magnitude of a model's performance is proportional to this shift in
the distribution of the dataset. Thus it becomes necessary to evaluate a
model's uncertainty and robustness to distributional shifts to get a realistic
estimate of its expected performance on real-world data. Present methods to
evaluate uncertainty and model's robustness are lacking and often fail to paint
the full picture. Moreover, most analysis so far has primarily focused on
classification tasks. In this paper, we propose more insightful metrics for
general regression tasks using the Shifts Weather Prediction Dataset. We also
present an evaluation of the baseline methods using these metrics.

    

### [[2111.07695] Joint Synthesis of Safety Certificate and Safe Control Policy using Constrained Reinforcement Learning](http://arxiv.org/abs/2111.07695)


  Safety is the major consideration in controlling complex dynamical systems
using reinforcement learning (RL), where the safety certificate can provide
provable safety guarantee. A valid safety certificate is an energy function
indicating that safe states are with low energy, and there exists a
corresponding safe control policy that allows the energy function to always
dissipate. The safety certificate and the safe control policy are closely
related to each other and both challenging to synthesize. Therefore, existing
learning-based studies treat either of them as prior knowledge to learn the
other, which limits their applicability with general unknown dynamics. This
paper proposes a novel approach that simultaneously synthesizes the
energy-function-based safety certificate and learns the safe control policy
with CRL. We do not rely on prior knowledge about either an available
model-based controller or a perfect safety certificate. In particular, we
formulate a loss function to optimize the safety certificate parameters by
minimizing the occurrence of energy increases. By adding this optimization
procedure as an outer loop to the Lagrangian-based constrained reinforcement
learning (CRL), we jointly update the policy and safety certificate parameters
and prove that they will converge to their respective local optima, the optimal
safe policy and a valid safety certificate. We evaluate our algorithms on
multiple safety-critical benchmark environments. The results show that the
proposed algorithm learns provably safe policies with no constraint violation.
The validity or feasibility of synthesized safety certificate is also verified
numerically.

    

### [[2112.02807] MDPFuzzer: Finding Crash-Triggering State Sequences in Models Solving the Markov Decision Process](http://arxiv.org/abs/2112.02807)


  The Markov decision process (MDP) provides a mathematical framework for
modeling sequential decision-making problems, many of which are crucial to
security and safety, such as autonomous driving and robot control. The rapid
development of artificial intelligence research has created efficient methods
for solving MDPs, such as deep neural networks (DNNs), reinforcement learning
(RL), and imitation learning (IL). However, these popular models for solving
MDPs are neither thoroughly tested nor rigorously reliable.
We present MDPFuzzer, the first blackbox fuzz testing framework for models
solving MDPs. MDPFuzzer forms testing oracles by checking whether the target
model enters abnormal and dangerous states. During fuzzing, MDPFuzzer decides
which mutated state to retain by measuring if it can reduce cumulative rewards
or form a new state sequence. We design efficient techniques to quantify the
"freshness" of a state sequence using Gaussian mixture models (GMMs) and
dynamic expectation-maximization (DynEM). We also prioritize states with high
potential of revealing crashes by estimating the local sensitivity of target
models over states.
MDPFuzzer is evaluated on five state-of-the-art models for solving MDPs,
including supervised DNN, RL, IL, and multi-agent RL. Our evaluation includes
scenarios of autonomous driving, aircraft collision avoidance, and two games
that are often used to benchmark RL. During a 12-hour run, we find over 80
crash-triggering state sequences on each model. We show inspiring findings that
crash-triggering states, though look normal, induce distinct neuron activation
patterns compared with normal states. We further develop an abnormal behavior
detector to harden all the evaluated models and repair them with the findings
of MDPFuzzer to significantly enhance their robustness without sacrificing
accuracy.

    

### [[2112.02834] A Generalized Zero-Shot Quantization of Deep Convolutional Neural Networks via Learned Weights Statistics](http://arxiv.org/abs/2112.02834)


  Quantizing the floating-point weights and activations of deep convolutional
neural networks to fixed-point representation yields reduced memory footprints
and inference time. Recently, efforts have been afoot towards zero-shot
quantization that does not require original unlabelled training samples of a
given task. These best-published works heavily rely on the learned batch
normalization (BN) parameters to infer the range of the activations for
quantization. In particular, these methods are built upon either empirical
estimation framework or the data distillation approach, for computing the range
of the activations. However, the performance of such schemes severely degrades
when presented with a network that does not accommodate BN layers. In this line
of thought, we propose a generalized zero-shot quantization (GZSQ) framework
that neither requires original data nor relies on BN layer statistics. We have
utilized the data distillation approach and leveraged only the pre-trained
weights of the model to estimate enriched data for range calibration of the
activations. To the best of our knowledge, this is the first work that utilizes
the distribution of the pretrained weights to assist the process of zero-shot
quantization. The proposed scheme has significantly outperformed the existing
zero-shot works, e.g., an improvement of ~ 33% in classification accuracy for
MobileNetV2 and several other models that are w & w/o BN layers, for a variety
of tasks. We have also demonstrated the efficacy of the proposed work across
multiple open-source quantization frameworks. Importantly, our work is the
first attempt towards the post-training zero-shot quantization of futuristic
unnormalized deep neural networks.

    

### [[2112.06396] Does Fully Homomorphic Encryption Need Compute Acceleration?](http://arxiv.org/abs/2112.06396)


  Fully Homomorphic Encryption (FHE) allows arbitrarily complex computations on
encrypted data without ever needing to decrypt it, thus enabling us to maintain
data privacy on third-party systems. Unfortunately, sustaining deep
computations with FHE requires a periodic noise reduction step known as
bootstrapping. The cost of the bootstrapping operation is one of the primary
barriers to the wide-spread adoption of FHE. In this paper, we present an
in-depth architectural analysis of the bootstrapping step in FHE. First, we
observe that secure implementations of bootstrapping exhibit a low arithmetic
intensity (<1 Op/byte), require large caches (>100 MB) and as such, are heavily
bound by the main memory bandwidth. Consequently, we demonstrate that existing
workloads observe marginal performance gains from the design of bespoke
high-throughput arithmetic units tailored to FHE. Secondly, we propose several
cache-friendly algorithmic optimizations that improve the throughput in FHE
bootstrapping by enabling up to 3.2x higher arithmetic intensity and 4.6x lower
memory bandwidth. Our optimizations apply to a wide range of structurally
similar computations such as private evaluation and training of machine
learning models. Finally, we incorporate these optimizations into an
architectural tool which, given a cache size, memory subsystem, the number of
functional units and a desired security level, selects optimal cryptosystem
parameters to maximize the bootstrapping throughput. Our optimized
bootstrapping implementation represents a best-case scenario for compute
acceleration of FHE. We show that despite these optimizations, bootstrapping
continues to remain bottlenecked by main memory bandwidth. We thus conclude
that secure FHE implementations need to look beyond accelerated compute for
further performance improvements and propose new research directions to address
the underlying memory bottleneck.

    

### [[2112.05830] Collecting Coupons is Faster with Friends](http://arxiv.org/abs/2112.05830)


  In this note, we introduce a distributed twist on the classic coupon
collector problem: a set of $m$ collectors wish to each obtain a set of $n$
coupons; for this, they can each sample coupons uniformly at random, but can
also meet in pairwise interactions, during which they can exchange coupons. By
doing so, they hope to reduce the number of coupons that must be sampled by
each collector in order to obtain a full set. This extension is natural when
considering real-world manifestations of the coupon collector phenomenon, and
has been remarked upon and studied empirically [Hayes and Hannigan 2006, Ahmad
et al. 2014, Delmarcelle 2019].
We provide the first theoretical analysis for such a scenario. We find that
"coupon collecting with friends" can indeed significantly reduce the number of
coupons each collector must sample, and raises interesting connections to the
more traditional variants of the problem. While our analysis is in most cases
asymptotically tight, there are several open questions raised, regarding
finer-grained analysis of both "coupon collecting with friends", and of a
long-studied variant of the original problem in which a collector requires
multiple full sets of coupons.

    

### [[2112.05831] Improved Deterministic $(Δ+1)$-Coloring in Low-Space MPC](http://arxiv.org/abs/2112.05831)


  We present a deterministic $O(\log \log \log n)$-round low-space Massively
Parallel Computation (MPC) algorithm for the classical problem of
$(\Delta+1)$-coloring on $n$-vertex graphs. In this model, every machine has a
sublinear local memory of size $n^{\phi}$ for any arbitrary constant $\phi \in
(0,1)$. Our algorithm works under the relaxed setting where each machine is
allowed to perform exponential (in $n^{\phi}$) local computation, while
respecting the $n^{\phi}$ space and bandwidth limitations.
Our key technical contribution is a novel derandomization of the ingenious
$(\Delta+1)$-coloring LOCAL algorithm by Chang-Li-Pettie (STOC 2018, SIAM J.
Comput. 2020). The Chang-Li-Pettie algorithm runs in $T_{local}=poly(\log\log
n)$ rounds, which sets the state-of-the-art randomized round complexity for the
problem in the local model. Our derandomization employs a combination of tools,
most notably pseudorandom generators (PRG) and bounded-independence hash
functions.
The achieved round complexity of $O(\log\log\log n)$ rounds matches the bound
of $\log(T_{local})$, which currently serves an upper bound barrier for all
known randomized algorithms for locally-checkable problems in this model.
Furthermore, no deterministic sublogarithmic low-space MPC algorithms for the
$(\Delta+1)$-coloring problem were previously known.

    

### [[2112.05858] MANA-2.0: A Future-Proof Design for Transparent Checkpointing of MPI at Scale](http://arxiv.org/abs/2112.05858)


  MANA-2.0 is a scalable, future-proof design for transparent checkpointing of
MPI-based computations. Its network transparency ("network-agnostic") feature
ensures that MANA-2.0 will provide a viable, efficient mechanism for
transparently checkpointing MPI applications on current and future
supercomputers. MANA-2.0 is an enhancement of previous work, the original MANA,
which interposes MPI calls, and is a work in progress intended for production
deployment. MANA-2.0 implements a series of new algorithms and features that
improve MANA's scalability and reliability, enabling transparent
checkpoint-restart over thousands of MPI processes. MANA-2.0 is being tested on
today's Cori supercomputer at NERSC using Cray MPICH library over the Cray GNI
network, but it is designed to work over any standard MPI running over an
arbitrary network. Two widely-used HPC applications were selected to
demonstrate the enhanced features of MANA-2.0: GROMACS, a molecular dynamics
simulation code with frequent point-to-point communication, and VASP, a
materials science code with frequent MPI collective communication. Perhaps the
most important lesson to be learned from MANA-2.0 is a series of algorithms and
data structures for library-based transformations that enable MPI-based
computations over MANA-2.0 to reliably survive the checkpoint-restart
transition.

    

### [[2112.05928] Efficient Device Scheduling with Multi-Job Federated Learning](http://arxiv.org/abs/2112.05928)


  Recent years have witnessed a large amount of decentralized data in multiple
(edge) devices of end-users, while the aggregation of the decentralized data
remains difficult for machine learning jobs due to laws or regulations.
Federated Learning (FL) emerges as an effective approach to handling
decentralized data without sharing the sensitive raw data, while
collaboratively training global machine learning models. The servers in FL need
to select (and schedule) devices during the training process. However, the
scheduling of devices for multiple jobs with FL remains a critical and open
problem. In this paper, we propose a novel multi-job FL framework to enable the
parallel training process of multiple jobs. The framework consists of a system
model and two scheduling methods. In the system model, we propose a parallel
training process of multiple jobs, and construct a cost model based on the
training time and the data fairness of various devices during the training
process of diverse jobs. We propose a reinforcement learning-based method and a
Bayesian optimization-based method to schedule devices for multiple jobs while
minimizing the cost. We conduct extensive experimentation with multiple jobs
and datasets. The experimental results show that our proposed approaches
significantly outperform baseline approaches in terms of training time (up to
8.67 times faster) and accuracy (up to 44.6% higher).

    

### [[2112.06006] Towards the Internet of Behaviors in airports with a fog-to-cloud approach](http://arxiv.org/abs/2112.06006)


  Recent advances in Internet of Things (IoT) and the rising of the Internet of
Behavior (IoB) have made it possible to develop real-time improved traveler
assistance tools for mobile phones, assisted by cloud-based machine learning,
and using fog computing in between IoT and the Cloud. Within the
Horizon2020-funded mF2C project an Android app has been developed exploiting
the proximity marketing concept and covers the essential path through the
airport onto the flight, from the least busy security queue through to the time
to walk to gate, gate changes, and other obstacles. It gives chance to
travelers to discover the facilities of the airport, aided by a recommender
system using machine learning, that can make recommendations and offer voucher
according with the traveler's preferences or on similarities to other
travelers. The system provides obvious benefits to the airport planners, not
only people tracking in the shops area, but also aggregated and anonymized
view, like heat maps that can highlight bottlenecks in the infrastructure, or
suggest situations that require intervention, such as emergencies. With the
emerging of the COVID pandemic the tool could be adapted to help in the social
distancing to guarantee safety. The use of the fog-to-cloud platform and the
fulfilling of all centricity and privacy requirements of the IoB give evidence
of the impact of the solution in a smart city environment.

    

### [[2112.06128] R2: A Distributed Remote Function Execution Mechanism With Built-in Metadata](http://arxiv.org/abs/2112.06128)


  Named data networking (NDN) constructs a network by names, providing a
flexible and decentralized way to manage resources within the edge computing
continuum. This paper aims to solve the question, "Given a function with its
parameters and metadata, how to select the executor in a distributed manner and
obtain the result in NDN?" To answer it, we design R2 that involves the
following stages. First, we design a name structure including data, function
names, and other function parameters. Second, we develop a 2-phase mechanism,
where in the first phase, the function request from a client-first reaches the
data source and retrieves the metadata, then the best node is selected while
the metadata is responding to the client. In the second phase, the chosen node
directly retrieves the data, executes the function, and provides the result to
the client. Furthermore, we propose a stop condition to intelligently reduce
the processing time of the first phase and provide a simple proof and range
analysis. Simulations confirm that R2 outperforms the current solutions in
terms of resource allocation, especially when the data volume and the function
complexity are high. In the experiments, when the data size is 100 KiB and the
function complexity is $\mathcal{O}(n^2)$, the speedup ratio is 4.61. To
further evaluate R2, we also implement a general intermediate data processing
logic named ``Bolt'' implemented on an app-level in ndnSIM. We believe that R2
shall help the researchers and developers to verify their ideas smoothly.

    

### [[2112.06254] Sinan: Data Driven Resource Management for Cloud Microservices](http://arxiv.org/abs/2112.06254)


  Cloud applications are increasingly shifting to interactive and
loosely-coupled microservices. Despite their advantages, microservices
complicate resource management, due to inter-tier dependencies.
We present Sinan, a cluster manager for interactive microservices that
leverages easily-obtainable tracing data instead of empirical decisions, to
infer the impact of a resource allocation on on end-to-end performance, and
allocate appropriate resources to each tier. In a preliminary evaluation of
Sinan with an end-to-end social network built with microservices, we show that
Sinan's data-driven approach, allows the service to always meet its QoS without
sacrificing resource efficiency.

    

### [[2112.06263] Sage: Leveraging ML to Diagnose Unpredictable Performance in Cloud Microservices](http://arxiv.org/abs/2112.06263)


  Cloud applications are increasingly shifting from large monolithic services,
to complex graphs of loosely-coupled microservices. Despite their advantages,
microservices also introduce cascading QoS violations in cloud applications,
which are difficult to diagnose and correct.
We present Sage, a ML-driven root cause analysis system for interactive cloud
microservices. Sage leverages unsupervised learning models to circumvent the
overhead of trace labeling, determines the root cause of unpredictable
performance online, and applies corrective actions to restore performance. On
experiments on both dedicated local clusters and large GCE clusters we show
that Sage achieves high root cause detection accuracy and predictable
performance.

    

### [[2112.06275] A Restless Bandit Model for Energy-Efficient Job Assignments in Server Farms](http://arxiv.org/abs/2112.06275)


  We aim to maximize the energy efficiency, gauged as average energy cost per
job, in a large-scale server farm with various storage or/and computing
components, which are modeled as parallel abstracted servers. Each server works
in multiple power modes characterized by potentially different service and
energy consumption rates. The heterogeneity of servers and multiple power modes
significantly complicate the maximization problem, where optimal solutions are
generally intractable. Relying on the Whittle relaxation technique, we resort
to a near-optimal and scalable job-assignment policy. Under certain conditions
including the assumption of exponentially distributed job sizes, we prove that
our proposed policy approaches optimality as the size of the entire system
tends to infinity; that is, it is asymptotically optimal. Nevertheless, we
demonstrate by simulations that the effectiveness of our policies is not
significantly limited by the conditions used for mathematical rigor and that
our model still has wide practical applicability. In particular, the asymptotic
optimality is very much relevant for many real-world large-scale systems with
tens or hundreds of thousands of components, where conventional optimization
techniques can hardly apply. Furthermore, for non-asymptotic scenarios, we show
the effectiveness of the proposed policy through extensive numerical
simulations, where the policy substantially outperforms all the tested
baselines, and we especially demonstrate numerically its robustness against
heavy-tailed job-size distributions.

    

### [[2112.06280] In-Memory Indexed Caching for Distributed Data Processing](http://arxiv.org/abs/2112.06280)


  Powerful abstractions such as dataframes are only as efficient as their
underlying runtime system. The de-facto distributed data processing framework,
Apache Spark, is poorly suited for the modern cloud-based data-science
workloads due to its outdated assumptions: static datasets analyzed using
coarse-grained transformations. In this paper, we introduce the Indexed
DataFrame, an in-memory cache that supports a dataframe abstraction which
incorporates indexing capabilities to support fast lookup and join operations.
Moreover, it supports appends with multi-version concurrency control. We
implement the Indexed DataFrame as a lightweight, standalone library which can
be integrated with minimum effort in existing Spark programs. We analyze the
performance of the Indexed DataFrame in cluster and cloud deployments with
real-world datasets and benchmarks using both Apache Spark and Databricks
Runtime. In our evaluation, we show that the Indexed DataFrame significantly
speeds-up query execution when compared to a non-indexed dataframe, incurring
modest memory overhead.

    

### [[2002.07659] Distributed graph problems through an automata-theoretic lens](http://arxiv.org/abs/2002.07659)


  The locality of a graph problem is the smallest distance $T$ such that each
node can choose its own part of the solution based on its radius-$T$
neighborhood. In many settings, a graph problem can be solved efficiently with
a distributed or parallel algorithm if and only if it has a small locality.
In this work we seek to automate the study of solvability and locality: given
the description of a graph problem $\Pi$, we would like to determine if $\Pi$
is solvable and what is the asymptotic locality of $\Pi$ as a function of the
size of the graph. Put otherwise, we seek to automatically synthesize efficient
distributed and parallel algorithms for solving $\Pi$.
We focus on locally checkable graph problems; these are problems in which a
solution is globally feasible if it looks feasible in all constant-radius
neighborhoods. Prior work on such problems has brought primarily bad news:
questions related to locality are undecidable in general, and even if we focus
on the case of labeled paths and cycles, determining locality is
$\mathsf{PSPACE}$-hard (Balliu et al., PODC 2019).
We complement prior negative results with efficient algorithms for the cases
of unlabeled paths and cycles and, as an extension, for rooted trees. We
introduce a new automata-theoretic perspective for studying locally checkable
graph problems. We represent a locally checkable problem $\Pi$ as a
nondeterministic finite automaton $\mathcal{M}$ over a unary alphabet. We
identify polynomial-time-computable properties of the automaton $\mathcal{M}$
that near-completely capture the solvability and locality of $\Pi$ in cycles
and paths, with the exception of one specific case that is
$\mbox{co-$\mathsf{NP}$}$-complete.

    

### [[2004.01562] Search via Parallel L{é}vy Walks on ${\mathbb Z}^2$](http://arxiv.org/abs/2004.01562)


  Motivated by the \emph{L{é}vy foraging hypothesis} -- the premise that
various animal species have adapted to follow \emph{L{é}vy walks} to optimize
their search efficiency -- we study the parallel hitting time of L{é}vy walks
on the infinite two-dimensional grid.We consider $k$ independent discrete-time
L{é}vy walks, with the same exponent $\alpha \in(1,\infty)$, that start from
the same node, and analyze the number of steps until the first walk visits a
given target at distance $\ell$.We show that for any choice of $k$ and $\ell$
from a large range, there is a unique optimal exponent $\alpha_{k,\ell} \in
(2,3)$, for which the hitting time is $\tilde O(\ell^2/k)$ w.h.p., while
modifying the exponent by an $\epsilon$ term increases the hitting time by a
polynomial factor, or the walks fail to hit the target almost surely.Based on
that, we propose a surprisingly simple and effective parallel search strategy,
for the setting where $k$ and $\ell$ are unknown: the exponent of each L{é}vy
walk is just chosen independently and uniformly at random from the interval
$(2,3)$.This strategy achieves optimal search time (modulo polylogarithmic
factors) among all possible algorithms (even centralized ones that know
$k$).Our results should be contrasted with a line of previous work showing that
the exponent $\alpha = 2$ is optimal for various search this http URL our setting
of $k$ parallel walks, we show that the optimal exponent depends on $k$ and
$\ell$, and that randomizing the choice of the exponents works simultaneously
for all $k$ and $\ell$.

    

### [[2011.11325] A Game-Theoretic Analysis of Cross-Chain Atomic Swaps with HTLCs](http://arxiv.org/abs/2011.11325)


  To achieve interoperability between unconnected ledgers, hash time lock
contracts (HTLCs) are commonly used for cross-chain asset exchange. The
solution tolerates transaction failure, and can "make the best out of worst''
by allowing transacting agents to at least keep their original assets in case
of an abort. Nonetheless, as an undesired outcome, reoccurring transaction
failures prompt a critical and analytical examination of the protocol. In this
study, we propose a game-theoretic framework to study the strategic behaviors
of agents taking part in cross-chain atomic swaps implemented with HTLCs. We
study the success rate of the transaction as a function of the exchange rate of
the swap, the token price and its volatility, among other variables. We
demonstrate that in an attempt to maximize one's own utility as asset price
changes, either agent might withdraw from the swap. An extension of our model
confirms that collateral deposits can improve the transaction success rate,
motivating further research towards collateralization without a trusted third
party. A second model variation suggests that a swap is more likely to succeed
when agents dynamically adjust the exchange rate in response to price
fluctuations.

    

### [[2112.05780] A Scoping Review of Publicly Available Language Tasks in Clinical Natural Language Processing](http://arxiv.org/abs/2112.05780)


  Objective: to provide a scoping review of papers on clinical natural language
processing (NLP) tasks that use publicly available electronic health record
data from a cohort of patients. Materials and Methods: We searched six
databases, including biomedical research and computer science literature
database. A round of title/abstract screening and full-text screening were
conducted by two reviewers. Our method followed the Preferred Reporting Items
for Systematic Reviews and Meta-Analysis (PRISMA) guidelines. Results: A total
of 35 papers with 47 clinical NLP tasks met inclusion criteria between 2007 and
2021. We categorized the tasks by the type of NLP problems, including name
entity recognition, summarization, and other NLP tasks. Some tasks were
introduced with a topic of clinical decision support applications, such as
substance abuse, phenotyping, cohort selection for clinical trial. We
summarized the tasks by publication and dataset information. Discussion: The
breadth of clinical NLP tasks keeps growing as the field of NLP evolves with
advancements in language systems. However, gaps exist in divergent interests
between general domain NLP community and clinical informatics community, and in
generalizability of the data sources. We also identified issues in data
selection and preparation including the lack of time-sensitive data, and
invalidity of problem size and evaluation. Conclusions: The existing clinical
NLP tasks cover a wide range of topics and the field will continue to grow and
attract more attention from both general domain NLP and clinical informatics
community. We encourage future work to incorporate multi-disciplinary
collaboration, reporting transparency, and standardization in data preparation.

    

### [[2112.05786] Guided Generative Models using Weak Supervision for Detecting Object Spatial Arrangement in Overhead Images](http://arxiv.org/abs/2112.05786)


  The increasing availability and accessibility of numerous overhead images
allows us to estimate and assess the spatial arrangement of groups of
geospatial target objects, which can benefit many applications, such as traffic
monitoring and agricultural monitoring. Spatial arrangement estimation is the
process of identifying the areas which contain the desired objects in overhead
images. Traditional supervised object detection approaches can estimate
accurate spatial arrangement but require large amounts of bounding box
annotations. Recent semi-supervised clustering approaches can reduce manual
labeling but still require annotations for all object categories in the image.
This paper presents the target-guided generative model (TGGM), under the
Variational Auto-encoder (VAE) framework, which uses Gaussian Mixture Models
(GMM) to estimate the distributions of both hidden and decoder variables in
VAE. Modeling both hidden and decoder variables by GMM reduces the required
manual annotations significantly for spatial arrangement estimation. Unlike
existing approaches that the training process can only update the GMM as a
whole in the optimization iterations (e.g., a "minibatch"), TGGM allows the
update of individual GMM components separately in the same optimization
iteration. Optimizing GMM components separately allows TGGM to exploit the
semantic relationships in spatial data and requires only a few labels to
initiate and guide the generative process. Our experiments shows that TGGM
achieves results comparable to the state-of-the-art semi-supervised methods and
outperforms unsupervised methods by 10% based on the $F_{1}$ scores, while
requiring significantly fewer labeled data.

    

### [[2112.05808] Benchmarking human visual search computational models in natural scenes: models comparison and reference datasets](http://arxiv.org/abs/2112.05808)


  Visual search is an essential part of almost any everyday human goal-directed
interaction with the environment. Nowadays, several algorithms are able to
predict gaze positions during simple observation, but few models attempt to
simulate human behavior during visual search in natural scenes. Furthermore,
these models vary widely in their design and exhibit differences in the
datasets and metrics with which they were evaluated. Thus, there is a need for
a reference point, on which each model can be tested and from where potential
improvements can be derived. In the present work, we select publicly available
state-of-the-art visual search models in natural scenes and evaluate them on
different datasets, employing the same metrics to estimate their efficiency and
similarity with human subjects. In particular, we propose an improvement to the
Ideal Bayesian Searcher through a combination with a neural network-based
visual search model, enabling it to generalize to other datasets. The present
work sheds light on the limitations of current models and how potential
improvements can be accomplished by combining approaches. Moreover, it moves
forward on providing a solution for the urgent need for benchmarking data and
metrics to support the development of more general human visual search
computational models.

    

### [[2112.05982] Overview of The MediaEval 2021 Predicting Media Memorability Task](http://arxiv.org/abs/2112.05982)


  This paper describes the MediaEval 2021 Predicting Media Memorability}task,
which is in its 4th edition this year, as the prediction of short-term and
long-term video memorability remains a challenging task. In 2021, two datasets
of videos are used: first, a subset of the TRECVid 2019 Video-to-Text dataset;
second, the Memento10K dataset in order to provide opportunities to explore
cross-dataset generalisation. In addition, an Electroencephalography
(EEG)-based prediction pilot subtask is introduced. In this paper, we outline
the main aspects of the task and describe the datasets, evaluation metrics, and
requirements for participants' submissions.

    

### [[2112.05996] Formalising the Foundations of Discrete Reinforcement Learning in Isabelle/HOL](http://arxiv.org/abs/2112.05996)


  We present a formalisation of finite Markov decision processes with rewards
in the Isabelle theorem prover. We focus on the foundations required for
dynamic programming and the use of reinforcement learning agents over such
processes. In particular, we derive the Bellman equation from first principles
(in both scalar and vector form), derive a vector calculation that produces the
expected value of any policy p, and go on to prove the existence of a
universally optimal policy where there is a discounting factor less than one.
Lastly, we prove that the value iteration and the policy iteration algorithms
work in finite time, producing an epsilon-optimal and a fully optimal policy
respectively.

    

### [[2112.05999] Curvature-guided dynamic scale networks for Multi-view Stereo](http://arxiv.org/abs/2112.05999)


  Multi-view stereo (MVS) is a crucial task for precise 3D reconstruction. Most
recent studies tried to improve the performance of matching cost volume in MVS
by designing aggregated 3D cost volumes and their regularization. This paper
focuses on learning a robust feature extraction network to enhance the
performance of matching costs without heavy computation in the other steps. In
particular, we present a dynamic scale feature extraction network, namely,
CDSFNet. It is composed of multiple novel convolution layers, each of which can
select a proper patch scale for each pixel guided by the normal curvature of
the image surface. As a result, CDFSNet can estimate the optimal patch scales
to learn discriminative features for accurate matching computation between
reference and source images. By combining the robust extracted features with an
appropriate cost formulation strategy, our resulting MVS architecture can
estimate depth maps more precisely. Extensive experiments showed that the
proposed method outperforms other state-of-the-art methods on complex outdoor
scenes. It significantly improves the completeness of reconstructed models. As
a result, the method can process higher resolution inputs within faster
run-time and lower memory than other MVS methods. Our source code is available
at url{this https URL}.

    

### [[2112.06028] Retrosynthetic Planning with Experience-Guided Monte Carlo Tree Search](http://arxiv.org/abs/2112.06028)


  Retrosynthetic planning problem is to analyze a complex molecule and give a
synthetic route using simple building blocks. The huge number of chemical
reactions leads to a combinatorial explosion of possibilities, and even the
experienced chemists could not select the most promising transformations. The
current approaches rely on human-defined or machine-trained score functions
which have limited chemical knowledge or use expensive estimation methods such
as rollout to guide the search. In this paper, we propose {\tt MCTS}, a novel
MCTS-based retrosynthetic planning approach, to deal with retrosynthetic
planning problem. Instead of exploiting rollout, we build an Experience
Guidance Network to learn knowledge from synthetic experiences during the
search. Experiments on benchmark USPTO datasets show that, our {\tt MCTS} gains
significant improvement over state-of-the-art approaches both in efficiency and
effectiveness.

    

### [[2112.06055] Towards Autonomous Satellite Communications: An AI-based Framework to Address System-level Challenges](http://arxiv.org/abs/2112.06055)


  The next generation of satellite constellations is designed to better address
the future needs of our connected society: highly-variable data demand, mobile
connectivity, and reaching more under-served regions. Artificial Intelligence
(AI) and learning-based methods are expected to become key players in the
industry, given the poor scalability and slow reaction time of current resource
allocation mechanisms. While AI frameworks have been validated for isolated
communication tasks or subproblems, there is still not a clear path to achieve
fully-autonomous satellite systems. Part of this issue results from the focus
on subproblems when designing models, instead of the necessary system-level
perspective. In this paper we try to bridge this gap by characterizing the
system-level needs that must be met to increase satellite autonomy, and
introduce three AI-based components (Demand Estimator, Offline Planner, and
Real Time Engine) that jointly address them. We first do a broad literature
review on the different subproblems and identify the missing links to the
system-level goals. In response to these gaps, we outline the three necessary
components and highlight their interactions. We also discuss how current models
can be incorporated into the framework and possible directions of future work.

    

### [[2112.06080] UPV at TREC Health Misinformation Track 2021 Ranking with SBERT and Quality Estimators](http://arxiv.org/abs/2112.06080)


  Health misinformation on search engines is a significant problem that could
negatively affect individuals or public health. To mitigate the problem, TREC
organizes a health misinformation track. This paper presents our submissions to
this track. We use a BM25 and a domain-specific semantic search engine for
retrieving initial documents. Later, we examine a health news schema for
quality assessment and apply it to re-rank documents. We merge the scores from
the different components by using reciprocal rank fusion. Finally, we discuss
the results and conclude with future works.

    

### [[2112.06104] Synthetic Map Generation to Provide Unlimited Training Data for Historical Map Text Detection](http://arxiv.org/abs/2112.06104)


  Many historical map sheets are publicly available for studies that require
long-term historical geographic data. The cartographic design of these maps
includes a combination of map symbols and text labels. Automatically reading
text labels from map images could greatly speed up the map interpretation and
helps generate rich metadata describing the map content. Many text detection
algorithms have been proposed to locate text regions in map images
automatically, but most of the algorithms are trained on out-ofdomain datasets
(e.g., scenic images). Training data determines the quality of machine learning
models, and manually annotating text regions in map images is labor-extensive
and time-consuming. On the other hand, existing geographic data sources, such
as Open- StreetMap (OSM), contain machine-readable map layers, which allow us
to separate out the text layer and obtain text label annotations easily.
However, the cartographic styles between OSM map tiles and historical maps are
significantly different. This paper proposes a method to automatically generate
an unlimited amount of annotated historical map images for training text
detection models. We use a style transfer model to convert contemporary map
images into historical style and place text labels upon them. We show that the
state-of-the-art text detection models (e.g., PSENet) can benefit from the
synthetic historical maps and achieve significant improvement for historical
map text detection.

    

### [[2112.06106] Controlled-rearing studies of newborn chicks and deep neural networks](http://arxiv.org/abs/2112.06106)


  Convolutional neural networks (CNNs) can now achieve human-level performance
on challenging object recognition tasks. CNNs are also the leading quantitative
models in terms of predicting neural and behavioral responses in visual
recognition tasks. However, there is a widely accepted critique of CNN models:
unlike newborn animals, which learn rapidly and efficiently, CNNs are thought
to be "data hungry," requiring massive amounts of training data to develop
accurate models for object recognition. This critique challenges the promise of
using CNNs as models of visual development. Here, we directly examined whether
CNNs are more data hungry than newborn animals by performing parallel
controlled-rearing experiments on newborn chicks and CNNs. We raised newborn
chicks in strictly controlled visual environments, then simulated the training
data available in that environment by constructing a virtual animal chamber in
a video game engine. We recorded the visual images acquired by an agent moving
through the virtual chamber and used those images to train CNNs. When CNNs
received similar visual training data as chicks, the CNNs successfully solved
the same challenging view-invariant object recognition tasks as the chicks.
Thus, the CNNs were not more data hungry than animals: both CNNs and chicks
successfully developed robust object models from training data of a single
object.

    

### [[2112.06185] Multi-Agent Vulnerability Discovery for Autonomous Driving with Hazard Arbitration Reward](http://arxiv.org/abs/2112.06185)


  Discovering hazardous scenarios is crucial in testing and further improving
driving policies. However, conducting efficient driving policy testing faces
two key challenges. On the one hand, the probability of naturally encountering
hazardous scenarios is low when testing a well-trained autonomous driving
strategy. Thus, discovering these scenarios by purely real-world road testing
is extremely costly. On the other hand, a proper determination of accident
responsibility is necessary for this task. Collecting scenarios with
wrong-attributed responsibilities will lead to an overly conservative
autonomous driving strategy. To be more specific, we aim to discover hazardous
scenarios that are autonomous-vehicle responsible (AV-responsible), i.e., the
vulnerabilities of the under-test driving policy.
To this end, this work proposes a Safety Test framework by finding
Av-Responsible Scenarios (STARS) based on multi-agent reinforcement learning.
STARS guides other traffic participants to produce Av-Responsible Scenarios and
make the under-test driving policy misbehave via introducing Hazard Arbitration
Reward (HAR). HAR enables our framework to discover diverse, complex, and
AV-responsible hazardous scenarios. Experimental results against four different
driving policies in three environments demonstrate that STARS can effectively
discover AV-responsible hazardous scenarios. These scenarios indeed correspond
to the vulnerabilities of the under-test driving policies, thus are meaningful
for their further improvements.

    

### [[2112.06194] Improving Performance of Federated Learning based Medical Image Analysis in Non-IID Settings using Image Augmentation](http://arxiv.org/abs/2112.06194)


  Federated Learning (FL) is a suitable solution for making use of sensitive
data belonging to patients, people, companies, or industries that are
obligatory to work under rigid privacy constraints. FL mainly or partially
supports data privacy and security issues and provides an alternative to model
problems facilitating multiple edge devices or organizations to contribute a
training of a global model using a number of local data without having them.
Non-IID data of FL caused from its distributed nature presents a significant
performance degradation and stabilization skews. This paper introduces a novel
method dynamically balancing the data distributions of clients by augmenting
images to address the non-IID data problem of FL. The introduced method
remarkably stabilizes the model training and improves the model's test accuracy
from 83.22% to 89.43% for multi-chest diseases detection of chest X-ray images
in highly non-IID FL setting. The results of IID, non-IID and non-IID with
proposed method federated trainings demonstrated that the proposed method might
help to encourage organizations or researchers in developing better systems to
get values from data with respect to data privacy not only for healthcare but
also other fields.

    

### [[2112.06196] Predicting Above-Sentence Discourse Structure using Distant Supervision from Topic Segmentation](http://arxiv.org/abs/2112.06196)


  RST-style discourse parsing plays a vital role in many NLP tasks, revealing
the underlying semantic/pragmatic structure of potentially complex and diverse
documents. Despite its importance, one of the most prevailing limitations in
modern day discourse parsing is the lack of large-scale datasets. To overcome
the data sparsity issue, distantly supervised approaches from tasks like
sentiment analysis and summarization have been recently proposed. Here, we
extend this line of research by exploiting distant supervision from topic
segmentation, which can arguably provide a strong and oftentimes complementary
signal for high-level discourse structures. Experiments on two human-annotated
discourse treebanks confirm that our proposal generates accurate tree
structures on sentence and paragraph level, consistently outperforming previous
distantly supervised models on the sentence-to-document task and occasionally
reaching even higher scores on the sentence-to-paragraph level.

    

### [[2112.06197] Video as Conditional Graph Hierarchy for Multi-Granular Question Answering](http://arxiv.org/abs/2112.06197)


  Video question answering requires models to understand and reason about both
complex video and language data to correctly derive answers. Existing efforts
focus on designing sophisticated cross-modal interactions to fuse the
information from two modalities, while encoding the video and question
holistically as frame and word sequences. Despite their success, these methods
are essentially revolving around the sequential nature of video- and
question-contents, providing little insight to the problem of
question-answering and lacking interpretability as well. In this work, we argue
that while video is presented in frame sequence, the visual elements (eg,
objects, actions, activities and events) are not sequential but rather
hierarchical in semantic space. To align with the multi-granular essence of
linguistic concepts in language queries, we propose to model video as a
conditional graph hierarchy which weaves together visual facts of different
granularity in a level-wise manner, with the guidance of corresponding textual
cues. Despite the simplicity, our extensive experiments demonstrate the
superiority of such conditional hierarchical graph architecture, with clear
performance improvements over prior methods and also better generalization
across different type of questions. Further analyses also consolidate the
model's reliability as it shows meaningful visual-textual evidences for the
predicted answers.

    

### [[2112.06311] Weakly Supervised Mapping of Natural Language to SQL through Question Decomposition](http://arxiv.org/abs/2112.06311)


  Natural Language Interfaces to Databases (NLIDBs), where users pose queries
in Natural Language (NL), are crucial for enabling non-experts to gain insights
from data. Developing such interfaces, by contrast, is dependent on experts who
often code heuristics for mapping NL to SQL. Alternatively, NLIDBs based on
machine learning models rely on supervised examples of NL to SQL mappings
(NL-SQL pairs) used as training data. Such examples are again procured using
experts, which typically involves more than a one-off interaction. Namely, each
data domain in which the NLIDB is deployed may have different characteristics
and therefore require either dedicated heuristics or domain-specific training
examples. To this end, we propose an alternative approach for training machine
learning-based NLIDBs, using weak supervision. We use the recently proposed
question decomposition representation called QDMR, an intermediate between NL
and formal query languages. Recent work has shown that non-experts are
generally successful in translating NL to QDMR. We consequently use NL-QDMR
pairs, along with the question answers, as supervision for automatically
synthesizing SQL queries. The NL questions and synthesized SQL are then used to
train NL-to-SQL models, which we test on five benchmark datasets. Extensive
experiments show that our solution, requiring zero expert annotations, performs
competitively with models trained on expert annotated data.

    

### [[2112.06346] ValueNet: A New Dataset for Human Value Driven Dialogue System](http://arxiv.org/abs/2112.06346)


  Building a socially intelligent agent involves many challenges, one of which
is to teach the agent to speak guided by its value like a human. However,
value-driven chatbots are still understudied in the area of dialogue systems.
Most existing datasets focus on commonsense reasoning or social norm modeling.
In this work, we present a new large-scale human value dataset called ValueNet,
which contains human attitudes on 21,374 text scenarios. The dataset is
organized in ten dimensions that conform to the basic human value theory in
intercultural research. We further develop a Transformer-based value regression
model on ValueNet to learn the utility distribution. Comprehensive empirical
results show that the learned value model could benefit a wide range of
dialogue tasks. For example, by teaching a generative agent with reinforcement
learning and the rewards from the value model, our method attains
state-of-the-art performance on the personalized dialog generation dataset:
Persona-Chat. With values as additional features, existing emotion recognition
models enable capturing rich human emotions in the context, which further
improves the empathetic response generation performance in the
EmpatheticDialogues dataset. To the best of our knowledge, ValueNet is the
first large-scale text dataset for human value modeling, and we are the first
one trying to incorporate a value model into emotionally intelligent dialogue
systems. The dataset is available at this https URL.

    

### [[2112.06389] Local and Global Point Cloud Reconstruction for 3D Hand Pose Estimation](http://arxiv.org/abs/2112.06389)


  This paper addresses the 3D point cloud reconstruction and 3D pose estimation
of the human hand from a single RGB image. To that end, we present a novel
pipeline for local and global point cloud reconstruction using a 3D hand
template while learning a latent representation for pose estimation. To
demonstrate our method, we introduce a new multi-view hand posture dataset to
obtain complete 3D point clouds of the hand in the real world. Experiments on
our newly proposed dataset and four public benchmarks demonstrate the model's
strengths. Our method outperforms competitors in 3D pose estimation while
reconstructing realistic-looking complete 3D hand point clouds.

    

### [[2112.06412] A Survey of Toxic Comment Classification Methods](http://arxiv.org/abs/2112.06412)


  While in real life everyone behaves themselves at least to some extent, it is
much more difficult to expect people to behave themselves on the internet,
because there are few checks or consequences for posting something toxic to
others. Yet, for people on the other side, toxic texts often lead to serious
psychological consequences. Detecting such toxic texts is challenging. In this
paper, we attempt to build a toxicity detector using machine learning methods
including CNN, Naive Bayes model, as well as LSTM. While there has been
numerous groundwork laid by others, we aim to build models that provide higher
accuracy than the predecessors. We produced very high accuracy models using
LSTM and CNN, and compared them to the go-to solutions in language processing,
the Naive Bayes model. A word embedding approach is also applied to empower the
accuracy of our models.

    

### [[2112.06419] Stacked Generative Machine Learning Models for Fast Approximations of Steady-State Navier-Stokes Equations](http://arxiv.org/abs/2112.06419)


  Computational fluid dynamics (CFD) simulations are broadly applied in
engineering and physics. A standard description of fluid dynamics requires
solving the Navier-Stokes (N-S) equations in different flow regimes. However,
applications of CFD simulations are computationally-limited by the
availability, speed, and parallelism of high-performance computing. To improve
computational efficiency, machine learning techniques have been used to create
accelerated data-driven approximations for CFD. A majority of such approaches
rely on large labeled CFD datasets that are expensive to obtain at the scale
necessary to build robust data-driven models. We develop a weakly-supervised
approach to solve the steady-state N-S equations under various boundary
conditions, using a multi-channel input with boundary and geometric conditions.
We achieve state-of-the-art results without any labeled simulation data, but
using a custom data-driven and physics-informed loss function by using and
small-scale solutions to prime the model to solve the N-S equations. To improve
the resolution and predictability, we train stacked models of increasing
complexity generating the numerical solutions for N-S equations. Without
expensive computations, our model achieves high predictability with a variety
of obstacles and boundary conditions. Given its high flexibility, the model can
generate a solution on a 64 x 64 domain within 5 ms on a regular desktop
computer which is 1000 times faster than a regular CFD solver. Translation of
interactive CFD simulation on local consumer computing hardware enables new
applications in real-time predictions on the internet of things devices where
data transfer is prohibitive and can increase the scale, speed, and
computational cost of boundary-value fluid problems.

    

### [[2001.11507] Towards an Ontology for Scenario Definition for the Assessment of Automated Vehicles: An Object-Oriented Framework](http://arxiv.org/abs/2001.11507)


  The development of new assessment methods for the performance of automated
vehicles is essential to enable the deployment of automated driving
technologies, due to the complex operational domain of automated vehicles. One
contributing method is scenario-based assessment in which test cases are
derived from real-world road traffic scenarios obtained from driving data.
Given the complexity of the reality that is being modeled in these scenarios,
it is a challenge to define a structure for capturing these scenarios. An
intensional definition that provides a set of characteristics that are deemed
to be both necessary and sufficient to qualify as a scenario assures that the
scenarios constructed are both complete and intercomparable.
In this article, we develop a comprehensive and operable definition of the
notion of scenario while considering existing definitions in the literature.
This is achieved by proposing an object-oriented framework in which scenarios
and their building blocks are defined as classes of objects having attributes,
methods, and relationships with other objects. The object-oriented approach
promotes clarity, modularity, reusability, and encapsulation of the objects. We
provide definitions and justifications of each of the terms. Furthermore, the
framework is used to translate the terms in a coding language that is publicly
available.

    

### [[2009.05912] DualDE: Dually Distilling Knowledge Graph Embedding for Faster and Cheaper Reasoning](http://arxiv.org/abs/2009.05912)


  Knowledge Graph Embedding (KGE) is a popular method for KG reasoning and
training KGEs with higher dimension are usually preferred since they have
better reasoning capability. However, high-dimensional KGEs pose huge
challenges to storage and computing resources and are not suitable for
resource-limited or time-constrained applications, for which faster and cheaper
reasoning is necessary. To address this problem, we propose DualDE, a knowledge
distillation method to build low-dimensional student KGE from pre-trained
high-dimensional teacher KGE. DualDE considers the dual-influence between the
teacher and the student. In DualDE, we propose a soft label evaluation
mechanism to adaptively assign different soft label and hard label weights to
different triples, and a two-stage distillation approach to improve the
student's acceptance of the teacher. Our DualDE is general enough to be applied
to various KGEs. Experimental results show that our method can successfully
reduce the embedding parameters of a high-dimensional KGE by 7 times - 15 times
and increase the inference speed by 2 times - 6 times while retaining a high
performance. We also experimentally prove the effectiveness of our soft label
evaluation mechanism and two-stage distillation approach via ablation study.

    

### [[2102.11529] Differentiable Logic Machines](http://arxiv.org/abs/2102.11529)


  The integration of reasoning, learning, and decision-making is key to build
more general AI systems. As a step in this direction, we propose a novel
neural-logic architecture that can solve both inductive logic programming (ILP)
and deep reinforcement learning (RL) problems. Our architecture defines a
restricted but expressive continuous space of first-order logic programs by
assigning weights to predicates instead of rules. Therefore, it is fully
differentiable and can be efficiently trained with gradient descent. Besides,
in the deep RL setting with actor-critic algorithms, we propose a novel
efficient critic architecture. Compared to state-of-the-art methods on both ILP
and RL problems, our proposition achieves excellent performance, while being
able to provide a fully interpretable solution and scaling much better,
especially during the testing phase.

    

### [[2103.09488] Revisiting the Loss Weight Adjustment in Object Detection](http://arxiv.org/abs/2103.09488)


  Object detection is a typical multi-task learning application, which
optimizes classification and regression simultaneously. However, classification
loss always dominates the multi-task loss in anchor-based methods, hampering
the consistent and balanced optimization of the tasks. In this paper, we find
that shifting the bounding boxes can change the division of positive and
negative samples in classification, meaning classification depends on
regression. Moreover, we summarize three important conclusions about
fine-tuning loss weights, considering different datasets, optimizers and
regression loss functions. Based on the above conclusions, we propose Adaptive
Loss Weight Adjustment(ALWA) to solve the imbalance in optimizing anchor-based
methods according to statistical characteristics of losses. By incorporating
ALWA into previous state-of-the-art detectors, we achieve a significant
performance gain on PASCAL VOC and MS COCO, even with L1, SmoothL1 and CIoU
loss. The code is available at this https URL.

    

### [[2104.01791] A Heuristic-driven Uncertainty based Ensemble Framework for Fake News Detection in Tweets and News Articles](http://arxiv.org/abs/2104.01791)


  The significance of social media has increased manifold in the past few
decades as it helps people from even the most remote corners of the world to
stay connected. With the advent of technology, digital media has become more
relevant and widely used than ever before and along with this, there has been a
resurgence in the circulation of fake news and tweets that demand immediate
attention. In this paper, we describe a novel Fake News Detection system that
automatically identifies whether a news item is "real" or "fake", as an
extension of our work in the CONSTRAINT COVID-19 Fake News Detection in English
challenge. We have used an ensemble model consisting of pre-trained models
followed by a statistical feature fusion network , along with a novel heuristic
algorithm by incorporating various attributes present in news items or tweets
like source, username handles, URL domains and authors as statistical feature.
Our proposed framework have also quantified reliable predictive uncertainty
along with proper class output confidence level for the classification task. We
have evaluated our results on the COVID-19 Fake News dataset and FakeNewsNet
dataset to show the effectiveness of the proposed algorithm on detecting fake
news in short news content as well as in news articles. We obtained a best
F1-score of 0.9892 on the COVID-19 dataset, and an F1-score of 0.9073 on the
FakeNewsNet dataset.

    

### [[2105.06597] RetGen: A Joint framework for Retrieval and Grounded Text Generation Modeling](http://arxiv.org/abs/2105.06597)


  Recent advances in large-scale pre-training such as GPT-3 allow seemingly
high quality text to be generated from a given prompt. However, such generation
systems often suffer from problems of hallucinated facts, and are not
inherently designed to incorporate useful external information. Grounded
generation models appear to offer remedies, but their training typically relies
on rarely-available parallel data where information-relevant documents are
provided for context. We propose a framework that alleviates this data
constraint by jointly training a grounded generator and document retriever on
the language model signal. The model learns to reward retrieval of the
documents with the highest utility in generation, and attentively combines them
using a Mixture-of-Experts (MoE) ensemble to generate follow-on text. We
demonstrate that both generator and retriever can take advantage of this joint
training and work synergistically to produce more informative and relevant text
in both prose and dialogue generation.

    

### [[2106.12154] Neural Fashion Image Captioning : Accounting for Data Diversity](http://arxiv.org/abs/2106.12154)


  Image captioning has increasingly large domains of application, and fashion
is not an exception. Having automatic item descriptions is of great interest
for fashion web platforms, sometimes hosting hundreds of thousands of images.
This paper is one of the first to tackle image captioning for fashion images.
To address dataset diversity issues, we introduced the InFashAIv1 dataset
containing almost 16.000 African fashion item images with their titles, prices,
and general descriptions. We also used the well-known DeepFashion dataset in
addition to InFashAIv1. Captions are generated using the Show and Tell model
made of CNN encoder and RNN Decoder. We showed that jointly training the model
on both datasets improves captions quality for African style fashion images,
suggesting a transfer learning from Western style data. The InFashAIv1 dataset
is released on Github to encourage works with more diversity inclusion.

    

### [[2106.15398] Automated Repair of Process Models with Non-Local Constraints Using State-Based Region Theory](http://arxiv.org/abs/2106.15398)


  State-of-the-art process discovery methods construct free-choice process
models from event logs. Consequently, the constructed models do not take into
account indirect dependencies between events. Whenever the input behaviour is
not free-choice, these methods fail to provide a precise model. In this paper,
we propose a novel approach for enhancing free-choice process models by adding
non-free-choice constructs discovered a-posteriori via region-based techniques.
This allows us to benefit from the performance of existing process discovery
methods and the accuracy of the employed fundamental synthesis techniques. We
prove that the proposed approach preserves fitness with respect to the event
log while improving the precision when indirect dependencies exist. The
approach has been implemented and tested on both synthetic and real-life
datasets. The results show its effectiveness in repairing models discovered
from event logs.

    

### [[2112.06342] Faster-Than-Native Alternatives for x86 VP2INTERSECT Instructions](http://arxiv.org/abs/2112.06342)


  We present faster-than-native alternatives for the full AVX512-VP2INTERSECT
instruction subset using basic AVX512F instructions. These alternatives compute
only one of the output masks, which is sufficient for the typical case of
computing the intersection of two sorted lists of integers, or computing the
size of such an intersection. While the naïve implementation (compare the
first input vector against all rotations of the second) is slower than the
native instructions, we show that by rotating both the first and second
operands at the same time there is a significant saving in the total number of
vector rotations, resulting in the emulations being faster than the native
instructions, for all instructions in the VP2INTERSECT subset. Additionally,
the emulations can be easily extended to other types of inputs (e.g. packed
vectors of 16-bit integers) for which native instructions are not available.

    

### [[2112.05964] Overcoming Restraint: Composing Verification of Foreign Functions with Cogent](http://arxiv.org/abs/2112.05964)


  Cogent is a restricted functional language designed to reduce the cost of
developing verified systems code. Because of its sometimes-onerous
restrictions, such as the lack of support for recursion and its strict
uniqueness type system, Cogent provides an escape hatch in the form of a
foreign function interface (FFI) to C code. This poses a problem when verifying
Cogent programs, as imported C components do not enjoy the same level of static
guarantees that Cogent does. Previous verification of file systems implemented
in Cogent merely assumed that their C components were correct and that they
preserved the invariants of Cogent's type system. In this paper, we instead
prove such obligations. We demonstrate how they smoothly compose with existing
Cogent theorems, and result in a correctness theorem of the overall Cogent-C
system. The Cogent FFI constraints ensure that key invariants of Cogent's type
system are maintained even when calling C code. We verify reusable higher-order
and polymorphic functions including a generic loop combinator and array
iterators and demonstrate their application to several examples including
binary search and the BilbyFs file system. We demonstrate the feasibility of
verification of mixed Cogent-C systems, and provide some insight into
verification of software comprised of code in multiple languages with differing
levels of static guarantees.

    

### [[2112.06039] CertiStr: A Certified String Solver (technical report)](http://arxiv.org/abs/2112.06039)


  Theories over strings are among the most heavily researched logical theories
in the SMT community in the past decade, owing to the error-prone nature of
string manipulations, which often leads to security vulnerabilities (e.g.
cross-site scripting and code injection). The majority of the existing decision
procedures and solvers for these theories are themselves intricate; they are
complicated algorithmically, and also have to deal with a very rich vocabulary
of operations. This has led to a plethora of bugs in implementation, which have
for instance been discovered through fuzzing. In this paper, we present
CertiStr, a certified implementation of a string constraint solver for the
theory of strings with concatenation and regular constraints. CertiStr aims to
solve string constraints using a forward-propagation algorithm based on
symbolic representations of regular constraints as symbolic automata, which
returns three results: sat, unsat, and unknown, and is guaranteed to terminate
for the string constraints whose concatenation dependencies are acyclic. The
implementation has been developed and proven correct in Isabelle/HOL, through
which an effective solver in OCaml was generated. We demonstrate the
effectiveness and efficiency of CertiStr against the standard Kaluza benchmark,
in which 80.4% tests are in the string constraint fragment of CertiStr. Of
these 80.4% tests, CertiStr can solve 83.5% (i.e. CertiStr returns sat or
unsat) within 60s.

    

### [[2112.06233] A simple proof of three properties on Simpson's 4-slot Algorithm](http://arxiv.org/abs/2112.06233)


  In this paper we present an invariance proof of three properties on Simpson's
4-slot algorithm, i.e. data-race freedom, data coherence and data freshness,
which together implies linearisability of the algorithm. It is an extension of
previous works whose proof focuses mostly on data-race freedom. In addition,
our proof uses simply inductive invariants and transition invariants, whereas
previous work uses more sophisticated machinery like separation logics,
rely-guarantee or ownership transfer.

    

### [[2104.08638] SAILFISH: Vetting Smart Contract State-Inconsistency Bugs in Seconds](http://arxiv.org/abs/2104.08638)


  This paper presents SAILFISH, a scalable system for automatically finding
state-inconsistency bugs in smart contracts. To make the analysis tractable, we
introduce a hybrid approach that includes (i) a light-weight exploration phase
that dramatically reduces the number of instructions to analyze, and (ii) a
precise refinement phase based on symbolic evaluation guided by our novel
value-summary analysis, which generates extra constraints to over-approximate
the side effects of whole-program execution, thereby ensuring the precision of
the symbolic evaluation. We developed a prototype of SAILFISH and evaluated its
ability to detect two state-inconsistency flaws, viz., reentrancy and
transaction order dependence (TOD) in Ethereum smart contracts. Further, we
present detection rules for other kinds of smart contract flaws that SAILFISH
can be extended to detect.
Our experiments demonstrate the efficiency of our hybrid approach as well as
the benefit of the value summary analysis. In particular, we show that S
SAILFISH outperforms five state-of-the-art smart contract analyzers (SECURITY,
MYTHRIL, OYENTE, SEREUM and VANDAL ) in terms of performance, and precision. In
total, SAILFISH discovered 47 previously unknown vulnerable smart contracts out
of 89,853 smart contracts from ETHERSCAN .

    

### [[2105.05398] Sound, Precise, and Fast Abstract Interpretation with Tristate Numbers](http://arxiv.org/abs/2105.05398)


  Extended Berkeley Packet Filter (BPF) is a language and run-time system that
allows non-superusers to extend the Linux and Windows operating systems by
downloading user code into the kernel. To ensure that user code is safe to run
in kernel context, BPF relies on a static analyzer that proves properties about
the code, such as bounded memory access and the absence of operations that
crash. The BPF static analyzer checks safety using abstract interpretation with
several abstract domains. Among these, the domain of tnums (tristate numbers)
is a key domain used to reason about the bitwise uncertainty in program values.
This paper formally specifies the tnum abstract domain and its arithmetic
operators. We provide the first proofs of soundness and optimality of the
abstract arithmetic operators for tnum addition and subtraction used in the BPF
analyzer. Further, we describe a novel sound algorithm for multiplication of
tnums that is more precise and efficient (runs 33% faster on average) than the
Linux kernel's algorithm. Our tnum multiplication is now merged in the Linux
kernel.

    

### [<title>Parameters: { scale_post_weight } might not be used - XGBoost</title>](https://discuss.xgboost.ai/t/parameters-scale-post-weight-might-not-be-used/2597/3)