
## 2021-10-4

### [[2110.00035] Energy-Efficient and Delay-Guaranteed Joint Resource Allocation and DU Selection in O-RAN](http://arxiv.org/abs/2110.00035)


  The radio access network (RAN) part of the next-generation wireless networks
will require efficient solutions for satisfying low latency and high-throughput
services. The open RAN (O-RAN) is one of the candidates to achieve this goal,
in addition to increasing vendor diversity and promoting openness. In the O-RAN
architecture, network functions are executed in central units (CU), distributed
units (DU), and radio units (RU). These entities are virtualized on
general-purpose CPUs and form a processing pool. These processing pools can be
located in different geographical places and have limited capacity, affecting
the energy consumption and the performance of networks. Additionally, since
user demand is not deterministic, special attention should be paid to
allocating resource blocks to users by ensuring their expected quality of
service for latency-sensitive traffic flows. In this paper, we propose a joint
optimization solution to enhance energy efficiency and provide delay guarantees
to the users in the O-RAN architecture. We formulate this novel problem and
linearize it to provide a solution with a mixed-integer linear problem (MILP)
solver. We compare this with a baseline that addresses this optimization
problem using a disjoint approach. The results show that our approach
outperforms the baseline method in terms of energy efficiency.

    

### [[2110.00060] Automating Internet of Things Network Traffic Collection with Robotic Arm Interactions](http://arxiv.org/abs/2110.00060)


  Consumer Internet of things research often involves collecting network
traffic sent or received by IoT devices. These data are typically collected via
crowdsourcing or while researchers manually interact with IoT devices in a
laboratory setting. However, manual interactions and crowdsourcing are often
tedious, expensive, inaccurate, or do not provide comprehensive coverage of
possible IoT device behaviors. We present a new method for generating IoT
network traffic using a robotic arm to automate user interactions with devices.
This eliminates manual button pressing and enables permutation-based
interaction sequences that rigorously explore the range of possible device
behaviors. We test this approach with an Arduino-controlled robotic arm, a
smart speaker and a smart thermostat, using machine learning to demonstrate
that collected network traffic contains information about device interactions
that could be useful for network, security, or privacy analyses. We also
provide source code and documentation allowing researchers to easily automate
IoT device interactions and network traffic collection in future studies.

    

### [[2110.00101] Shaping mmWave Wireless Channel via Multi-Beam Design using Reconfigurable Intelligent Surfaces](http://arxiv.org/abs/2110.00101)


  Millimeter-wave (mmWave) communications is considered as a key enabler
towards the realization of next-generation wireless networks, due to the
abundance of available spectrum at mmWave frequencies. However, mmWave suffers
from high free-space path-loss and poor scattering resulting in mostly
line-of-sight (LoS) channels which result in a lack of coverage. Reconfigurable
intelligent surfaces (RIS), as a new paradigm, have the potential to fill the
coverage holes by shaping the wireless channel. In this paper, we propose a
novel approach for designing RIS with elements arranged in a uniform planar
array (UPA) structure. In what we refer to as multi-beamforming, We propose and
design RIS such that the reflected beam comprises multiple disjoint lobes.
Moreover, the beams have optimized gain within the desired angular coverage
with fairly sharp edges avoiding power leakage to other regions. We provide a
closed-form low-complexity solution for the multi-beamforming design. We
confirm our theoretical results by numerical analysis.

    

### [[2110.00118] Unbiased Experiments in Congested Networks](http://arxiv.org/abs/2110.00118)


  When developing a new networking algorithm, it is established practice to run
a randomized experiment, or A/B test, to evaluate its performance. In an A/B
test, traffic is randomly allocated between a treatment group, which uses the
new algorithm, and a control group, which uses the existing algorithm. However,
because networks are congested, both treatment and control traffic compete
against each other for resources in a way that biases the outcome of these
tests. This bias can have a surprisingly large effect; for example, in lab A/B
tests with two widely used congestion control algorithms, the treatment
appeared to deliver 150% higher throughput when used by a few flows, and 75%
lower throughput when used by most flows-despite the fact that the two
algorithms have identical throughput when used by all traffic.
Beyond the lab, we show that A/B tests can also be biased at scale. In an
experiment run in cooperation with Netflix, estimates from A/B tests mistake
the direction of change of some metrics, miss changes in other metrics, and
overestimate the size of effects. We propose alternative experiment designs,
previously used in online platforms, to more accurately evaluate new algorithms
and allow experimenters to better understand the impact of congestion on their
tests.

    

### [[2110.00133] A Novel Simplified Swarm Optimization for Generalized Reliability Redundancy Allocation Problem](http://arxiv.org/abs/2110.00133)


  Network systems are commonly used in various fields, such as power grid,
Internet of Things (IoT), and gas networks. Reliability redundancy allocation
problem (RRAP) is a well-known reliability design tool, which needs to be
developed when the system is extended from the series-parallel structure to a
more general network structure. Therefore, this study proposes a novel RRAP
called General RRAP (GRRAP) to be applied to network systems. The Binary
Addition Tree Algorithm (BAT) is used to solve the network reliability. Since
GRRAP is an NP-hard problem, a new algorithm called Binary-addition simplified
swarm optimization (BSSO) is also proposed in this study. BSSO combines the
accuracy of the BAT with the efficiency of SSO, which can effectively reduce
the solution space and speed up the time to find high-quality solutions. The
experimental results show that BSSO outperforms three well-known algorithms,
Genetic Algorithm (GA), Particle Swarm Optimization (PSO), and Swarm
Optimization (SSO), on six network benchmarks.

    

### [[2110.00236] Simulation-based Evaluation of a Synchronous Transaction Model for Time-Sensitive Software-Defined Networks](http://arxiv.org/abs/2110.00236)


  Real-time networks based on Ethernet require robust quality-of-service for
time-critical traffic. The Time-Sensitive Networking (TSN) collection of
standards enables this in real-time environments like vehicle on-board
networks. Runtime reconfigurations in TSN must respect the deadlines of
real-time traffic. Software-Defined Networking (SDN) moves the control plane of
network devices to the SDN controller, making these networks programmable. This
allows reconfigurations from a central point in the network. In this work, we
present a transactional model for network reconfigurations that are
synchronously executed in all network devices. We evaluate its performance in a
case study against nontransactional reconfigurations and show that synchronous
transactions enable consistency for reconfigurations in TSN without increased
latencies for real-time frames.

    

### [[2110.00250] A Step Towards On-Path Security Function Outsourcing](http://arxiv.org/abs/2110.00250)


  Security function outsourcing has witnessed both research and deployment in
the recent years. While most existing services take a straight-forward approach
of cloud hosting, on-path transit networks (such as ISPs) are increasingly more
interested in offering outsourced security services to end users. Recent
proposals (such as SafeBricks and mbTLS) have made it possible to outsource
sensitive security applications to untrusted, arbitrary networks, rendering
on-path security function outsourcing more promising than ever. However, to
provide on-path security function outsourcing, there is one crucial component
that is still missing -- a practical end-to-end network protocol. Thus, the
discovery and orchestration of multiple capable and willing transit networks
for user-requested security functions have only been assumed in many studies
without any practical solutions. In this work, we propose Opsec, an end-to-end
security-outsourcing protocol that fills this gap and brings us closer to the
vision of on-path security function outsourcing. Opsec automatically discovers
one or more transit ISPs between a client and a server, and requests
user-specified security functions efficiently. When designing Opsec, we
prioritize the practicality and applicability of this new end-to-end protocol
in the current Internet. Our proof-of-concept implementation of Opsec for web
sessions shows that an end user can easily start a new web session with a few
clicks of a browser plug-in, to specify a series of security functions of her
choice. We show that it is possible to implement such a new end-to-end service
model in the current Internet for the majority of the web services without any
major changes to the standard protocols (e.g., TCP, TLS, HTTP) and the existing
network infrastructure (e.g., ISP's routing primitives).

    

### [[2110.00397] Cellular traffic offloading via Opportunistic Networking with Reinforcement Learning](http://arxiv.org/abs/2110.00397)


  The widespread diffusion of mobile phones is triggering an exponential growth
of mobile data traffic that is likely to cause, in the near future,
considerable traffic overload issues even in last-generation cellular networks.
Offloading part of the traffic to other networks is considered a very promising
approach and, in particular, in this paper, we consider offloading through
opportunistic networks of users' devices. However, the performance of this
solution strongly depends on the pattern of encounters between mobile nodes,
which should therefore be taken into account when designing offloading control
algorithms. In this paper, we propose an adaptive offloading solution based on
the Reinforcement Learning framework and we evaluate and compare the
performance of two well-known learning algorithms: Actor-Critic and Q-Learning.
More precisely, in our solution the controller of the dissemination process,
once trained, is able to select a proper number of content replicas to be
injected into the opportunistic network to guarantee the timely delivery of
contents to all interested users. We show that our system based on
Reinforcement Learning is able to automatically learn a very efficient strategy
to reduce the traffic on the cellular network, without relying on any
additional context information about the opportunistic network. Our solution
achieves a higher level of offloading with respect to other state-of-the-art
approaches, in a range of different mobility settings. Moreover, we show that a
more refined learning solution, based on the Actor-Critic algorithm, is
significantly more efficient than a simpler solution based on Q-learning.

    

### [[2110.00424] Correlation-Based Device Energy-Efficient Dynamic Multi-Task Offloading for Mobile Edge Computing](http://arxiv.org/abs/2110.00424)


  Task offloading to mobile edge computing (MEC) has emerged as a key
technology to alleviate the computation workloads of mobile devices and
decrease service latency for the computation-intensive applications. Device
battery consumption is one of the limiting factors needs to be considered
during task offloading. In this paper, multi-task offloading strategies have
been investigated to improve device energy efficiency. Correlations among tasks
in time domain as well as task domain are proposed to be employed to reduce the
number of tasks to be transmitted to MEC. Furthermore, a binary decision tree
based algorithm is investigated to jointly optimize the mobile device clock
frequency, transmission power, structure and number of tasks to be transmitted.
MATLAB based simulation is employed to demonstrate the performance of our
proposed algorithm. It is observed that the proposed dynamic multi-task
offloading strategies can reduce the total energy consumption at device along
various transmit power versus noise power point compared with the conventional
one.

    

### [[2110.00492] Dynamic CU-DU Selection for Resource Allocation in O-RAN Using Actor-Critic Learning](http://arxiv.org/abs/2110.00492)


  Recently, there has been tremendous efforts by network operators and
equipment vendors to adopt intelligence and openness in the next generation
radio access network (RAN). The goal is to reach a RAN that can self-optimize
in a highly complex setting with multiple platforms, technologies and vendors
in a converged compute and connect architecture. In this paper, we propose two
nested actor-critic learning based techniques to optimize the placement of
resource allocation function, and as well, the decisions for resource
allocation. By this, we investigate the impact of observability on the
performance of the reinforcement learning based resource allocation. We show
that when a network function (NF) is dynamically relocated based on service
requirements, using reinforcement learning techniques, latency and throughput
gains are obtained.

    

### [[2110.00041] Learning Multi-Site Harmonization of Magnetic Resonance Images Without Traveling Human Phantoms](http://arxiv.org/abs/2110.00041)


  Harmonization improves data consistency and is central to effective
integration of diverse imaging data acquired across multiple sites. Recent deep
learning techniques for harmonization are predominantly supervised in nature
and hence require imaging data of the same human subjects to be acquired at
multiple sites. Data collection as such requires the human subjects to travel
across sites and is hence challenging, costly, and impractical, more so when
sufficient sample size is needed for reliable network training. Here we show
how harmonization can be achieved with a deep neural network that does not rely
on traveling human phantom data. Our method disentangles site-specific
appearance information and site-invariant anatomical information from images
acquired at multiple sites and then employs the disentangled information to
generate the image of each subject for any target site. We demonstrate with
more than 6,000 multi-site T1- and T2-weighted images that our method is
remarkably effective in generating images with realistic site-specific
appearances without altering anatomical details. Our method allows
retrospective harmonization of data in a wide range of existing modern
large-scale imaging studies, conducted via different scanners and protocols,
without additional data collection.

    

### [[2110.00046] SpliceOut: A Simple and Efficient Audio Augmentation Method](http://arxiv.org/abs/2110.00046)


  Time masking has become a de facto augmentation technique for speech and
audio tasks, including automatic speech recognition (ASR) and audio
classification, most notably as a part of SpecAugment. In this work, we propose
SpliceOut, a simple modification to time masking which makes it computationally
more efficient. SpliceOut performs comparably to (and sometimes outperforms)
SpecAugment on a wide variety of speech and audio tasks, including ASR for
seven different languages using varying amounts of training data, as well as on
speech translation, sound and music classification, thus establishing itself as
a broadly applicable audio augmentation method. SpliceOut also provides
additional gains when used in conjunction with other augmentation techniques.
Apart from the fully-supervised setting, we also demonstrate that SpliceOut can
complement unsupervised representation learning with performance gains in the
semi-supervised and self-supervised settings.

    

### [[2110.00053] Sparse Quadratic Optimisation over the Stiefel Manifold with Application to Permutation Synchronisation](http://arxiv.org/abs/2110.00053)


  We address the non-convex optimisation problem of finding a sparse matrix on
the Stiefel manifold (matrices with mutually orthogonal columns of unit length)
that maximises (or minimises) a quadratic objective function. Optimisation
problems on the Stiefel manifold occur for example in spectral relaxations of
various combinatorial problems, such as graph matching, clustering, or
permutation synchronisation. Although sparsity is a desirable property in such
settings, it is mostly neglected in spectral formulations since existing
solvers, e.g. based on eigenvalue decomposition, are unable to account for
sparsity while at the same time maintaining global optimality guarantees. We
fill this gap and propose a simple yet effective sparsity-promoting
modification of the Orthogonal Iteration algorithm for finding the dominant
eigenspace of a matrix. By doing so, we can guarantee that our method finds a
Stiefel matrix that is globally optimal with respect to the quadratic objective
function, while in addition being sparse. As a motivating application we
consider the task of permutation synchronisation, which can be understood as a
constrained clustering problem that has particular relevance for matching
multiple images or 3D shapes in computer vision, computer graphics, and beyond.
We demonstrate that the proposed approach outperforms previous methods in this
domain.

    

### [[2110.00061] Scientific evidence extraction](http://arxiv.org/abs/2110.00061)


  Recently, interest has grown in applying machine learning to the problem of
table structure inference and extraction from unstructured documents. However,
progress in this area has been challenging both to make and to measure, due to
several issues that arise in training and evaluating models from labeled data.
This includes challenges as fundamental as the lack of a single definitive
ground truth output for each input sample and the lack of an ideal metric for
measuring partial correctness for this task. To address these we propose a new
dataset, PubMed Tables One Million (PubTables-1M), and a new class of metric,
grid table similarity (GriTS). PubTables-1M is nearly twice as large as the
previous largest comparable dataset, can be used for models across multiple
architectures and modalities, and addresses issues such as ambiguity and lack
of consistency in the annotations. We apply DETR to table extraction for the
first time and show that object detection models trained on PubTables-1M
produce excellent results out-of-the-box for all three tasks of detection,
structure recognition, and functional analysis. We describe the dataset in
detail to enable others to build on our work and combine this data with other
datasets for these and related tasks. It is our hope that PubTables-1M and the
proposed metrics can further progress in this area by creating a benchmark
suitable for training and evaluating a wide variety of models for table
extraction. Data and code will be released at
this https URL.

    

### [[2110.00071] PIETS: Parallelised Irregularity Encoders for Forecasting with Heterogeneous Time-Series](http://arxiv.org/abs/2110.00071)


  Heterogeneity and irregularity of multi-source data sets present a
significant challenge to time-series analysis. In the literature, the fusion of
multi-source time-series has been achieved either by using ensemble learning
models which ignore temporal patterns and correlation within features or by
defining a fixed-size window to select specific parts of the data sets. On the
other hand, many studies have shown major improvement to handle the
irregularity of time-series, yet none of these studies has been applied to
multi-source data. In this work, we design a novel architecture, PIETS, to
model heterogeneous time-series. PIETS has the following characteristics: (1)
irregularity encoders for multi-source samples that can leverage all available
information and accelerate the convergence of the model; (2) parallelised
neural networks to enable flexibility and avoid information overwhelming; and
(3) attention mechanism that highlights different information and gives high
importance to the most related data. Through extensive experiments on
real-world data sets related to COVID-19, we show that the proposed
architecture is able to effectively model heterogeneous temporal data and
outperforms other state-of-the-art approaches in the prediction task.

    

### [[2110.00078] Determining Standard Occupational Classification Codes from Job Descriptions in Immigration Petitions](http://arxiv.org/abs/2110.00078)


  Accurate specification of standard occupational classification (SOC) code is
critical to the success of many U.S. work visa applications. Determination of
correct SOC code relies on careful study of job requirements and comparison to
definitions given by the U.S. Bureau of Labor Statistics, which is often a
tedious activity. In this paper, we apply methods from natural language
processing (NLP) to computationally determine SOC code based on job
description. We implement and empirically evaluate a broad variety of
predictive models with respect to quality of prediction and training time, and
identify models best suited for this task.

    

### [[2110.00082] Two ways towards combining Sequential Neural Network and Statistical Methods to Improve the Prediction of Time Series](http://arxiv.org/abs/2110.00082)


  Statistic modeling and data-driven learning are the two vital fields that
attract many attentions. Statistic models intend to capture and interpret the
relationships among variables, while data-based learning attempt to extract
information directly from the data without pre-processing through complex
models. Given the extensive studies in both fields, a subtle issue is how to
properly integrate data based methods with existing knowledge or models. In
this paper, based on the time series data, we propose two different directions
to integrate the two, a decomposition-based method and a method exploiting the
statistic extraction of data features. The first one decomposes the data into
linear stable, nonlinear stable and unstable parts, where suitable statistical
models are used for the linear stable and nonlinear stable parts while the
appropriate machine learning tools are used for the unstable parts. The second
one applies statistic models to extract statistics features of data and feed
them as additional inputs into the machine learning platform for training. The
most critical and challenging thing is how to determine and extract the
valuable information from mathematical or statistical models to boost the
performance of machine learning algorithms. We evaluate the proposal using time
series data with varying degrees of stability. Performance results show that
both methods can outperform existing schemes that use models and learning
separately, and the improvements can be over 60%. Both our proposed methods are
promising in bridging the gap between model-based and data-driven schemes and
integrating the two to provide an overall higher learning performance.

    

### [[2110.00086] On the Trustworthiness of Tree Ensemble Explainability Methods](http://arxiv.org/abs/2110.00086)


  The recent increase in the deployment of machine learning models in critical
domains such as healthcare, criminal justice, and finance has highlighted the
need for trustworthy methods that can explain these models to stakeholders.
Feature importance methods (e.g. gain and SHAP) are among the most popular
explainability methods used to address this need. For any explainability
technique to be trustworthy and meaningful, it has to provide an explanation
that is accurate and stable. Although the stability of local feature importance
methods (explaining individual predictions) has been studied before, there is
yet a knowledge gap about the stability of global features importance methods
(explanations for the whole model). Additionally, there is no study that
evaluates and compares the accuracy of global feature importance methods with
respect to feature ordering. In this paper, we evaluate the accuracy and
stability of global feature importance methods through comprehensive
experiments done on simulations as well as four real-world datasets. We focus
on tree-based ensemble methods as they are used widely in industry and measure
the accuracy and stability of explanations under two scenarios: 1) when inputs
are perturbed 2) when models are perturbed. Our findings provide a comparison
of these methods under a variety of settings and shed light on the limitations
of global feature importance methods by indicating their lack of accuracy with
and without noisy inputs, as well as their lack of stability with respect to:
1) increase in input dimension or noise in the data; 2) perturbations in models
initialized by different random seeds or hyperparameter settings.

    

### [[2110.00087] Seeing Glass: Joint Point Cloud and Depth Completion for Transparent Objects](http://arxiv.org/abs/2110.00087)


  The basis of many object manipulation algorithms is RGB-D input. Yet,
commodity RGB-D sensors can only provide distorted depth maps for a wide range
of transparent objects due light refraction and absorption. To tackle the
perception challenges posed by transparent objects, we propose TranspareNet, a
joint point cloud and depth completion method, with the ability to complete the
depth of transparent objects in cluttered and complex scenes, even with
partially filled fluid contents within the vessels. To address the shortcomings
of existing transparent object data collection schemes in literature, we also
propose an automated dataset creation workflow that consists of
robot-controlled image collection and vision-based automatic annotation.
Through this automated workflow, we created Toronto Transparent Objects Depth
Dataset (TODD), which consists of nearly 15000 RGB-D images. Our experimental
evaluation demonstrates that TranspareNet outperforms existing state-of-the-art
depth completion methods on multiple datasets, including ClearGrasp, and that
it also handles cluttered scenes when trained on TODD. Code and dataset will be
released at this https URL


### [[2110.00091] Strengthening Probabilistic Graphical Models: The Purge-and-merge Algorithm](http://arxiv.org/abs/2110.00091)


  Probabilistic graphical models (PGMs) are powerful tools for solving systems
of complex relationships over a variety of probability distributions.
Tree-structured PGMs always result in efficient and exact solutions, while
inference on graph (or loopy) structured PGMs is not guaranteed to discover the
optimal solutions. It is in principle possible to convert loopy PGMs to an
equivalent tree structure, but for most interesting problems this is
impractical due to exponential blow-up. To address this, we developed the
purge-and-merge algorithm. The idea behind this algorithm is to iteratively
nudge a malleable graph structure towards a tree structure by selectively
merging factors. The merging process is designed to avoid exponential blow-up
by making use of sparse structures from which redundancy is purged as the
algorithm progresses. This approach is evaluated on a number of
constraint-satisfaction puzzles such as Sudoku, Fill-a-pix, and Kakuro. On
these tasks, our system outperformed other PGM-based approaches reported in the
literature. Although these tasks were limited to the binary logic of CSP, we
believe it holds promise for extension to general PGM inference.

    

### [[2110.00109] DeepMCAT: Large-Scale Deep Clustering for Medical Image Categorization](http://arxiv.org/abs/2110.00109)


  In recent years, the research landscape of machine learning in medical
imaging has changed drastically from supervised to semi-, weakly- or
unsupervised methods. This is mainly due to the fact that ground-truth labels
are time-consuming and expensive to obtain manually. Generating labels from
patient metadata might be feasible but it suffers from user-originated errors
which introduce biases. In this work, we propose an unsupervised approach for
automatically clustering and categorizing large-scale medical image datasets,
with a focus on cardiac MR images, and without using any labels. We
investigated the end-to-end training using both class-balanced and imbalanced
large-scale datasets. Our method was able to create clusters with high purity
and achieved over 0.99 cluster purity on these datasets. The results
demonstrate the potential of the proposed method for categorizing unstructured
large medical databases, such as organizing clinical PACS systems in hospitals.

    

### [[2110.00115] Comparing Sequential Forecasters](http://arxiv.org/abs/2110.00115)


  We consider two or more forecasters each making a sequence of predictions
over time and tackle the problem of how to compare them -- either online or
post-hoc. In fields ranging from meteorology to sports, forecasters make
predictions on different events or quantities over time, and this work
describes how to compare them in a statistically rigorous manner. Specifically,
we design a nonasymptotic sequential inference procedure for estimating the
time-varying difference in forecast quality when using a relatively large class
of scoring rules (bounded scores with a linear equivalent). The resulting
confidence intervals can be continuously monitored and yield statistically
valid comparisons at arbitrary data-dependent stopping times ("anytime-valid");
this is enabled by adapting recent variance-adaptive confidence sequences (CS)
to our setting. In the spirit of Shafer and Vovk's game-theoretic probability,
the coverage guarantees for our CSs are also distribution-free, in the sense
that they make no distributional assumptions whatsoever on the forecasts or
outcomes. Additionally, in contrast to a recent preprint by Henzi and Ziegel,
we show how to sequentially test a weak null hypothesis about whether one
forecaster outperforms another on average over time, by designing different
e-processes that quantify the evidence at any stopping time. We examine the
validity of our methods over their fixed-time and asymptotic counterparts in
synthetic experiments and demonstrate their effectiveness in real-data
settings, including comparing probability forecasts on Major League Baseball
(MLB) games and comparing statistical postprocessing methods for ensemble
weather forecasts.

    

### [[2110.00116] #ContextMatters: Advantages and Limitations of Using Machine Learning to Support Women in Politics](http://arxiv.org/abs/2110.00116)


  The United Nations identified gender equality as a Sustainable Development
Goal in 2015, recognizing the underrepresentation of women in politics as a
specific barrier to achieving gender equality. Political systems around the
world experience gender inequality across all levels of elected government as
fewer women run for office than men. This is due in part to online abuse,
particularly on social media platforms like Twitter, where women seeking or in
power tend to be targeted with more toxic maltreatment than their male
counterparts. In this paper, we present reflections on ParityBOT - the first
natural language processing-based intervention designed to affect online
discourse for women in politics for the better, at scale. Deployed across
elections in Canada, the United States and New Zealand, ParityBOT was used to
analyse and classify more than 12 million tweets directed at women candidates
and counter toxic tweets with supportive ones. From these elections we present
three case studies highlighting the current limitations of, and future research
and application opportunities for, using a natural language processing-based
system to detect online toxicity, specifically with regards to contextually
important microaggressions. We examine the rate of false negatives, where
ParityBOT failed to pick up on insults directed at specific high profile women,
which would be obvious to human users. We examine the unaddressed harms of
microaggressions and the potential of yet unseen damage they cause for women in
these communities, and for progress towards gender equality overall, in light
of these technological blindspots. This work concludes with a discussion on the
benefits of partnerships between nonprofit social groups and technology experts
to develop responsible, socially impactful approaches to addressing online
hate.

    

### [[2110.00124] Tree-Constrained Graph Neural Networks For Argument Mining](http://arxiv.org/abs/2110.00124)


  We propose a novel architecture for Graph Neural Networks that is inspired by
the idea behind Tree Kernels of measuring similarity between trees by taking
into account their common substructures, named fragments. By imposing a series
of regularization constraints to the learning problem, we exploit a pooling
mechanism that incorporates such notion of fragments within the node soft
assignment function that produces the embeddings. We present an extensive
experimental evaluation on a collection of sentence classification tasks
conducted on several argument mining corpora, showing that the proposed
approach performs well with respect to state-of-the-art techniques.

    

### [[2110.00135] UserIdentifier: Implicit User Representations for Simple and Effective Personalized Sentiment Analysis](http://arxiv.org/abs/2110.00135)


  Global models are trained to be as generalizable as possible, with user
invariance considered desirable since the models are shared across multitudes
of users. As such, these models are often unable to produce personalized
responses for individual users, based on their data. Contrary to widely-used
personalization techniques based on few-shot learning, we propose
UserIdentifier, a novel scheme for training a single shared model for all
users. Our approach produces personalized responses by adding fixed,
non-trainable user identifiers to the input data. We empirically demonstrate
that this proposed method outperforms the prefix-tuning based state-of-the-art
approach by up to 13%, on a suite of sentiment analysis datasets. We also show
that, unlike prior work, this method needs neither any additional model
parameters nor any extra rounds of few-shot fine-tuning.

    

### [[2110.00137] Iterative Teacher-Aware Learning](http://arxiv.org/abs/2110.00137)


  In human pedagogy, teachers and students can interact adaptively to maximize
communication efficiency. The teacher adjusts her teaching method for different
students, and the student, after getting familiar with the teacher's
instruction mechanism, can infer the teacher's intention to learn faster.
Recently, the benefits of integrating this cooperative pedagogy into machine
concept learning in discrete spaces have been proved by multiple works.
However, how cooperative pedagogy can facilitate machine parameter learning
hasn't been thoroughly studied. In this paper, we propose a gradient
optimization based teacher-aware learner who can incorporate teacher's
cooperative intention into the likelihood function and learn provably faster
compared with the naive learning algorithms used in previous machine teaching
works. We give theoretical proof that the iterative teacher-aware learning
(ITAL) process leads to local and global improvements. We then validate our
algorithms with extensive experiments on various tasks including regression,
classification, and inverse reinforcement learning using synthetic and real
data. We also show the advantage of modeling teacher-awareness when agents are
learning from human teachers.

    

### [[2110.00151] Lagrangian Inference for Ranking Problems](http://arxiv.org/abs/2110.00151)


  We propose a novel combinatorial inference framework to conduct general
uncertainty quantification in ranking problems. We consider the widely adopted
Bradley-Terry-Luce (BTL) model, where each item is assigned a positive
preference score that determines the Bernoulli distributions of pairwise
comparisons' outcomes. Our proposed method aims to infer general ranking
properties of the BTL model. The general ranking properties include the "local"
properties such as if an item is preferred over another and the "global"
properties such as if an item is among the top $K$-ranked items. We further
generalize our inferential framework to multiple testing problems where we
control the false discovery rate (FDR), and apply the method to infer the
top-$K$ ranked items. We also derive the information-theoretic lower bound to
justify the minimax optimality of the proposed method. We conduct extensive
numerical studies using both synthetic and real datasets to back up our theory.

    

### [[2110.00155] Incremental Layer-wise Self-Supervised Learning for Efficient Speech Domain Adaptation On Device](http://arxiv.org/abs/2110.00155)


  Streaming end-to-end speech recognition models have been widely applied to
mobile devices and show significant improvement in efficiency. These models are
typically trained on the server using transcribed speech data. However, the
server data distribution can be very different from the data distribution on
user devices, which could affect the model performance. There are two main
challenges for on device training, limited reliable labels and limited training
memory. While self-supervised learning algorithms can mitigate the mismatch
between domains using unlabeled data, they are not applicable on mobile devices
directly because of the memory constraint. In this paper, we propose an
incremental layer-wise self-supervised learning algorithm for efficient speech
domain adaptation on mobile devices, in which only one layer is updated at a
time. Extensive experimental results demonstrate that the proposed algorithm
obtains a Word Error Rate (WER) on the target domain $24.2\%$ better than
supervised baseline and costs $89.7\%$ less training memory than the end-to-end
self-supervised learning algorithm.

    

### [[2110.00157] Under the Microscope: Interpreting Readability Assessment Models for Filipino](http://arxiv.org/abs/2110.00157)


  Readability assessment is the process of identifying the level of ease or
difficulty of a certain piece of text for its intended audience. Approaches
have evolved from the use of arithmetic formulas to more complex
pattern-recognizing models trained using machine learning algorithms. While
using these approaches provide competitive results, limited work is done on
analyzing how linguistic variables affect model inference quantitatively. In
this work, we dissect machine learning-based readability assessment models in
Filipino by performing global and local model interpretation to understand the
contributions of varying linguistic features and discuss its implications in
the context of the Filipino language. Results show that using a model trained
with top features from global interpretation obtained higher performance than
the ones using features selected by Spearman correlation. Likewise, we also
empirically observed local feature weight boundaries for discriminating reading
difficulty at an extremely fine-grained level and their corresponding effects
if values are perturbed.

    

### [[2110.00158] Asymptotic Performance of Thompson Sampling in the Batched Multi-Armed Bandits](http://arxiv.org/abs/2110.00158)


  We study the asymptotic performance of the Thompson sampling algorithm in the
batched multi-armed bandit setting where the time horizon $T$ is divided into
batches, and the agent is not able to observe the rewards of her actions until
the end of each batch. We show that in this batched setting, Thompson sampling
achieves the same asymptotic performance as in the case where instantaneous
feedback is available after each action, provided that the batch sizes increase
subexponentially. This result implies that Thompson sampling can maintain its
performance even if it receives delayed feedback in $\omega(\log T)$ batches.
We further propose an adaptive batching scheme that reduces the number of
batches to $\Theta(\log T)$ while maintaining the same performance. Although
the batched multi-armed bandit setting has been considered in several recent
works, previous results rely on tailored algorithms for the batched setting,
which optimize the batch structure and prioritize exploration in the beginning
of the experiment to eliminate suboptimal actions. We show that Thompson
sampling, on the other hand, is able to achieve a similar asymptotic
performance in the batched setting without any modifications.

    

### [[2110.00165] Large-scale ASR Domain Adaptation by Self- and Semi-supervised Learning](http://arxiv.org/abs/2110.00165)


  Self- and Semi-supervised learning methods have been actively investigated to
reduce labeled training data or enhance the model performance. However, the
approach mostly focus on in-domain performance for public datasets. In this
study, we utilize the combination of self- and semi-supervised learning methods
to solve unseen domain adaptation problem in a large-scale production setting
for online ASR model. This approach demonstrates that using the source domain
data with a small fraction of the target domain data (3%) can recover the
performance gap compared to a full data baseline: relative 13.5% WER
improvement for target domain data.

    

### [[2110.00167] Machine learning models for prediction of droplet collision outcomes](http://arxiv.org/abs/2110.00167)


  Predicting the outcome of liquid droplet collisions is an extensively studied
phenomenon but the current physics based models for predicting the outcomes are
poor (accuracy $\approx 43\%$). The key weakness of these models is their
limited complexity. They only account for 3 features while there are many more
relevant features that go unaccounted for. This limitation of traditional
models can be easily overcome through machine learning modeling of the problem.
In an ML setting this problem directly translates to a classification problem
with 4 classes. Here we compile a large labelled dataset and tune different ML
classifiers over this dataset. We evaluate the accuracy and robustness of the
classifiers. ML classifiers, with accuracies over 90\%, significantly
outperform the physics based models. Another key question we try to answer in
this paper is whether existing knowledge of the physics based models can be
exploited to boost the accuracy of the ML classifiers. We find that while this
knowledge improves the accuracy marginally for small datasets, it does not
improve accuracy with if larger datasets are used for training the models.

    

### [[2110.00174] Empirical Quantitative Analysis of COVID-19 Forecasting Models](http://arxiv.org/abs/2110.00174)


  COVID-19 has been a public health emergency of international concern since
early 2020. Reliable forecasting is critical to diminish the impact of this
disease. To date, a large number of different forecasting models have been
proposed, mainly including statistical models, compartmental models, and deep
learning models. However, due to various uncertain factors across different
regions such as economics and government policy, no forecasting model appears
to be the best for all scenarios. In this paper, we perform quantitative
analysis of COVID-19 forecasting of confirmed cases and deaths across different
regions in the United States with different forecasting horizons, and evaluate
the relative impacts of the following three dimensions on the predictive
performance (improvement and variation) through different evaluation metrics:
model selection, hyperparameter tuning, and the length of time series required
for training. We find that if a dimension brings about higher performance
gains, if not well-tuned, it may also lead to harsher performance penalties.
Furthermore, model selection is the dominant factor in determining the
predictive performance. It is responsible for both the largest improvement and
the largest variation in performance in all prediction tasks across different
regions. While practitioners may perform more complicated time series analysis
in practice, they should be able to achieve reasonable results if they have
adequate insight into key decisions like model selection.

    

### [[2110.00175] DualNet: Continual Learning, Fast and Slow](http://arxiv.org/abs/2110.00175)


  According to Complementary Learning Systems (CLS)
theory~\citep{mcclelland1995there} in neuroscience, humans do effective
\emph{continual learning} through two complementary systems: a fast learning
system centered on the hippocampus for rapid learning of the specifics and
individual experiences, and a slow learning system located in the neocortex for
the gradual acquisition of structured knowledge about the environment.
Motivated by this theory, we propose a novel continual learning framework named
"DualNet", which comprises a fast learning system for supervised learning of
pattern-separated representation from specific tasks and a slow learning system
for unsupervised representation learning of task-agnostic general
representation via a Self-Supervised Learning (SSL) technique. The two fast and
slow learning systems are complementary and work seamlessly in a holistic
continual learning framework. Our extensive experiments on two challenging
continual learning benchmarks of CORE50 and miniImageNet show that DualNet
outperforms state-of-the-art continual learning methods by a large margin. We
further conduct ablation studies of different SSL objectives to validate
DualNet's efficacy, robustness, and scalability. Code will be made available
upon acceptance.

    

### [[2110.00181] Improving Load Forecast in Energy Markets During COVID-19](http://arxiv.org/abs/2110.00181)


  The abrupt outbreak of the COVID-19 pandemic was the most significant event
in 2020, which had profound and lasting impacts across the world. Studies on
energy markets observed a decline in energy demand and changes in energy
consumption behaviors during COVID-19. However, as an essential part of system
operation, how the load forecasting performs amid COVID-19 is not well
understood. This paper aims to bridge the research gap by systematically
evaluating models and features that can be used to improve the load forecasting
performance amid COVID-19. Using real-world data from the New York Independent
System Operator, our analysis employs three deep learning models and adopts
both novel COVID-related features as well as classical weather-related
features. We also propose simulating the stay-at-home situation with
pre-stay-at-home weekend data and demonstrate its effectiveness in improving
load forecasting accuracy during COVID-19.

    

### [[2110.00183] Predicting COVID-19 Patient Shielding: A Comprehensive Study](http://arxiv.org/abs/2110.00183)


  There are many ways machine learning and big data analytics are used in the
fight against the COVID-19 pandemic, including predictions, risk management,
diagnostics, and prevention. This study focuses on predicting COVID-19 patient
shielding -- identifying and protecting patients who are clinically extremely
vulnerable from coronavirus. This study focuses on techniques used for the
multi-label classification of medical text. Using the information published by
the United Kingdom NHS and the World Health Organisation, we present a novel
approach to predicting COVID-19 patient shielding as a multi-label
classification problem. We use publicly available, de-identified ICU medical
text data for our experiments. The labels are derived from the published
COVID-19 patient shielding data. We present an extensive comparison across 12
multi-label classifiers from the simple binary relevance to neural networks and
the most recent transformers. To the best of our knowledge this is the first
comprehensive study, where such a range of multi-label classifiers for medical
text are considered. We highlight the benefits of various approaches, and argue
that, for the task at hand, both predictive accuracy and processing time are
essential.

    

### [[2110.00188] Offline Reinforcement Learning with Reverse Model-based Imagination](http://arxiv.org/abs/2110.00188)


  In offline reinforcement learning (offline RL), one of the main challenges is
to deal with the distributional shift between the learning policy and the given
dataset. To address this problem, recent offline RL methods attempt to
introduce conservatism bias to encourage learning on high-confidence areas.
Model-free approaches directly encode such bias into policy or value function
learning using conservative regularizations or special network structures, but
their constrained policy search limits the generalization beyond the offline
dataset. Model-based approaches learn forward dynamics models with conservatism
quantifications and then generate imaginary trajectories to extend the offline
datasets. However, due to limited samples in offline dataset, conservatism
quantifications often suffer from overgeneralization in out-of-support regions.
The unreliable conservative measures will mislead forward model-based
imaginations to undesired areas, leading to overaggressive behaviors. To
encourage more conservatism, we propose a novel model-based offline RL
framework, called Reverse Offline Model-based Imagination (ROMI). We learn a
reverse dynamics model in conjunction with a novel reverse policy, which can
generate rollouts leading to the target goal states within the offline dataset.
These reverse imaginations provide informed data augmentation for the
model-free policy learning and enable conservative generalization beyond the
offline dataset. ROMI can effectively combine with off-the-shelf model-free
algorithms to enable model-based generalization with proper conservatism.
Empirical results show that our method can generate more conservative behaviors
and achieve state-of-the-art performance on offline RL benchmark tasks.

    

### [[2110.00199] Update in Unit Gradient](http://arxiv.org/abs/2110.00199)


  In Machine Learning, optimization mostly has been done by using a gradient
descent method to find the minimum value of the loss. However, especially in
deep learning, finding a global minimum from a nonconvex loss function across a
high dimensional space is an extraordinarily difficult task. Recently, a
generalization learning algorithm, Sharpness-Aware Minimization (SAM), has made
a great success in image classification task. Despite the great performance in
creating convex space, proper direction leading by SAM is still remained
unclear. We, thereby, propose a creating a Unit Vector space in SAM, which not
only consisted of the mathematical instinct in linear algebra but also kept the
advantages of adaptive gradient algorithm. Moreover, applying SAM in unit
gradient brings models competitive performances in image classification
datasets, such as CIFAR - {10, 100}. The experiment showed that it performed
even better and more robust than SAM.

    

### [[2110.00201] Error-free approximation of explicit linear MPC through lattice piecewise affine expression](http://arxiv.org/abs/2110.00201)


  In this paper, the disjunctive and conjunctive lattice piecewise affine (PWA)
approximations of explicit linear model predictive control (MPC) are proposed.
The training data is generated uniformly in the domain of interest, consisting
of the state samples and corresponding affine control laws, based on which the
lattice PWA approximations are constructed. Resampling of data is also proposed
to guarantee that the lattice PWA approximations are identical to the explicit
MPC control law in unique order (UO) regions containing the sample points as
interior points. Besides, under mild assumptions, the equivalence of the 2
lattice PWA approximations guarantees the approximations are error-free in the
domain of interest. The algorithms for deriving statistical error-free
approximation to the explicit linear MPC is proposed and the complexity of the
whole procedure is analyzed, which is polynomial with respect to the number of
samples. The performance of the proposed approximation strategy is tested
through 2 simulation examples, and the result shows that with a moderate number
of sample points, we can construct lattice PWA approximations that are
equivalent to optimal control law of the explicit linear MPC.

    

### [[2110.00202] Batched Thompson Sampling](http://arxiv.org/abs/2110.00202)


  We introduce a novel anytime Batched Thompson sampling policy for multi-armed
bandits where the agent observes the rewards of her actions and adjusts her
policy only at the end of a small number of batches. We show that this policy
simultaneously achieves a problem dependent regret of order $O(\log(T))$ and a
minimax regret of order $O(\sqrt{T\log(T)})$ while the number of batches can be
bounded by $O(\log(T))$ independent of the problem instance over a time horizon
$T$. We also show that in expectation the number of batches used by our policy
can be bounded by an instance dependent bound of order $O(\log\log(T))$. These
results indicate that Thompson sampling maintains the same performance in this
batched setting as in the case when instantaneous feedback is available after
each action, while requiring minimal feedback. These results also indicate that
Thompson sampling performs competitively with recently proposed algorithms
tailored for the batched setting. These algorithms optimize the batch structure
for a given time horizon $T$ and prioritize exploration in the beginning of the
experiment to eliminate suboptimal actions. We show that Thompson sampling
combined with an adaptive batching strategy can achieve a similar performance
without knowing the time horizon $T$ of the problem and without having to
carefully optimize the batch structure to achieve a target regret bound (i.e.
problem dependent vs minimax regret) for a given $T$.

    

### [[2110.00203] Q-Net: A Quantitative Susceptibility Mapping-based Deep Neural Network for Differential Diagnosis of Brain Iron Deposition in Hemochromatosis](http://arxiv.org/abs/2110.00203)


  Brain iron deposition, in particular deep gray matter nuclei, increases with
advancing age. Hereditary Hemochromatosis (HH) is the most common inherited
disorder of systemic iron excess in Europeans and recent studies claimed high
brain iron accumulation in patient with Hemochromatosis. In this study, we
focus on Artificial Intelligence (AI)-based differential diagnosis of brain
iron deposition in HH via Quantitative Susceptibility Mapping (QSM), which is
an established Magnetic Resonance Imaging (MRI) technique to study the
distribution of iron in the brain. Our main objective is investigating
potentials of AI-driven frameworks to accurately and efficiently differentiate
individuals with Hemochromatosis from those of the healthy control group. More
specifically, we developed the Q-Net framework, which is a data-driven model
that processes information on iron deposition in the brain obtained from
multi-echo gradient echo imaging data and anatomical information on T1-Weighted
images of the brain. We illustrate that the Q-Net framework can assist in
differentiating between someone with HH and Healthy control (HC) of the same
age, something that is not possible by just visualizing images. The study is
performed based on a unique dataset that was collected from 52 subjects with HH
and 47 HC. The Q-Net provides a differential diagnosis accuracy of 83.16% and
80.37% in the scan-level and image-level classification, respectively.

    

### [[2110.00210] Unsupervised Belief Representation Learning in Polarized Networks with Information-Theoretic Variational Graph Auto-Encoders](http://arxiv.org/abs/2110.00210)


  This paper develops a novel unsupervised algorithm for belief representation
learning in polarized networks that (i) uncovers the latent dimensions of the
underlying belief space and (ii) jointly embeds users and content items (that
they interact with) into that space in a manner that facilitates a number of
downstream tasks, such as stance detection, stance prediction, and ideology
mapping. Inspired by total correlation in information theory, we propose a
novel Information-Theoretic Variational Graph Auto-Encoder (InfoVGAE) that
learns to project both users and content items (e.g., posts that represent user
views) into an appropriate disentangled latent space. In order to better
disentangle orthogonal latent variables in that space, we develop total
correlation regularization, PI control module, and adopt rectified Gaussian
Distribution for the latent space. The latent representation of users and
content can then be used to quantify their ideological leaning and
detect/predict their stances on issues. We evaluate the performance of the
proposed InfoVGAE on three real-world datasets, of which two are collected from
Twitter and one from U.S. Congress voting records. The evaluation results show
that our model outperforms state-of-the-art unsupervised models and produce
comparable result with supervised models. We also discuss stance prediction and
user ranking within ideological groups.

    

### [[2110.00211] DNN-Opt: An RL Inspired Optimization for Analog Circuit Sizing using Deep Neural Networks](http://arxiv.org/abs/2110.00211)


  Analog circuit sizing takes a significant amount of manual effort in a
typical design cycle. With rapidly developing technology and tight schedules,
bringing automated solutions for sizing has attracted great attention. This
paper presents DNN-Opt, a Reinforcement Learning (RL) inspired Deep Neural
Network (DNN) based black-box optimization framework for analog circuit sizing.
The key contributions of this paper are a novel sample-efficient two-stage deep
learning optimization framework leveraging RL actor-critic algorithms, and a
recipe to extend it on large industrial circuits using critical device
identification. Our method shows 5--30x sample efficiency compared to other
black-box optimization methods both on small building blocks and on large
industrial circuits with better performance metrics. To the best of our
knowledge, this is the first application of DNN-based circuit sizing on
industrial scale circuits.

    

### [[2110.00212] Inverse airfoil design method for generating varieties of smooth airfoils using conditional WGAN-gp](http://arxiv.org/abs/2110.00212)


  Machine learning models are recently utilized for airfoil shape generation
methods. It is desired to obtain airfoil shapes that satisfies required lift
coefficient. Generative adversarial networks (GAN) output reasonable airfoil
shapes. However, shapes obtained from ordinal GAN models are not smooth, and
they need smoothing before flow analysis. Therefore, the models need to be
coupled with Bezier curves or other smoothing methods to obtain smooth shapes.
Generating shapes without any smoothing methods is challenging. In this study,
we employed conditional Wasserstein GAN with gradient penalty (CWGAN-GP) to
generate airfoil shapes, and the obtained shapes are as smooth as those
obtained using smoothing methods. With the proposed method, no additional
smoothing method is needed to generate airfoils. Moreover, the proposed model
outputs shapes that satisfy the lift coefficient requirements.

    

### [[2110.00214] Spiking Hyperdimensional Network: Neuromorphic Models Integrated with Memory-Inspired Framework](http://arxiv.org/abs/2110.00214)


  Recently, brain-inspired computing models have shown great potential to
outperform today's deep learning solutions in terms of robustness and energy
efficiency. Particularly, Spiking Neural Networks (SNNs) and HyperDimensional
Computing (HDC) have shown promising results in enabling efficient and robust
cognitive learning. Despite the success, these two brain-inspired models have
different strengths. While SNN mimics the physical properties of the human
brain, HDC models the brain on a more abstract and functional level. Their
design philosophies demonstrate complementary patterns that motivate their
combination. With the help of the classical psychological model on memory, we
propose SpikeHD, the first framework that fundamentally combines Spiking neural
network and hyperdimensional computing. SpikeHD generates a scalable and strong
cognitive learning system that better mimics brain functionality. SpikeHD
exploits spiking neural networks to extract low-level features by preserving
the spatial and temporal correlation of raw event-based spike data. Then, it
utilizes HDC to operate over SNN output by mapping the signal into
high-dimensional space, learning the abstract information, and classifying the
data. Our extensive evaluation on a set of benchmark classification problems
shows that SpikeHD provides the following benefit compared to SNN architecture:
(1) significantly enhance learning capability by exploiting two-stage
information processing, (2) enables substantial robustness to noise and
failure, and (3) reduces the network size and required parameters to learn
complex information.

    

### [[2110.00218] On the Importance of Gradients for Detecting Distributional Shifts in the Wild](http://arxiv.org/abs/2110.00218)


  Detecting out-of-distribution (OOD) data has become a critical component in
ensuring the safe deployment of machine learning models in the real world.
Existing OOD detection approaches primarily rely on the output or feature space
for deriving OOD scores, while largely overlooking information from the
gradient space. In this paper, we present GradNorm, a simple and effective
approach for detecting OOD inputs by utilizing information extracted from the
gradient space. GradNorm directly employs the vector norm of gradients,
backpropagated from the KL divergence between the softmax output and a uniform
probability distribution. Our key idea is that the magnitude of gradients is
higher for in-distribution (ID) data than that for OOD data, making it
informative for OOD detection. GradNorm demonstrates superior performance,
reducing the average FPR95 by up to 10.89% compared to the previous best
method.

    

### [[2110.00252] Open-set Classification of Common Waveforms Using A Deep Feed-forward Network and Binary Isolation Forest Models](http://arxiv.org/abs/2110.00252)


  In this paper, we examine the use of a deep multi-layer perceptron
architecture to classify received signals as one of seven common waveforms,
single carrier (SC), single-carrier frequency division multiple access
(SC-FDMA), orthogonal frequency division multiplexing (OFDM), linear frequency
modulation (LFM), amplitude modulation (AM), frequency modulation (FM), and
phase-coded pulse modulation used in communication and radar networks.
Synchronization of the signals is not needed as we assume there is an unknown
and uncompensated time and frequency offset. The classifier is open-set meaning
it assumes unknown waveforms may appear. Isolation forest (IF) models acting as
binary classifiers are used for each known signal class to perform detection of
possible unknown signals. This is accomplished using the 32-length feature
vector from a dense layer as input to the IF models. The classifier and IF
models work together to monitor the spectrum and identify waveforms along with
detecting unknown waveforms. Results showed the classifier had 100%
classification rate above 0 dB with an accuracy of 83.2% and 94.7% at -10 dB
and -5 dB, respectively, with signal impairments present. Results for the IF
models showed an overall accuracy of 98% when detecting known and unknown
signals with signal impairments present. IF models were able to reject all
unknown signals while signals similar to known signals were able to pass
through 2% of the time due to the contamination rate used during training.
Overall, the entire system can classify correctly in an open-set mode with 98%
accuracy at SNR greater than 0 dB.

    

### [[2110.00254] The Complexity of Learning Approval-Based Multiwinner Voting Rules](http://arxiv.org/abs/2110.00254)


  We study the PAC learnability of multiwinner voting, focusing on the class of
approval-based committee scoring (ABCS) rules. These are voting rules applied
on profiles with approval ballots, where each voter approves some of the
candidates. ABCS rules adapt positional scoring rules in single-winner voting
by assuming that each committee of $k$ candidates collects from each voter a
score, that depends on the size of the voter's ballot and on the size of its
intersection with the committee. Then, committees of maximum score are the
winning ones. Our goal is to learn a target rule (i.e., to learn the
corresponding scoring function) using information about the winning committees
of a small number of sampled profiles. Despite the existence of exponentially
many outcomes compared to single-winner elections, we show that the sample
complexity is still low: a polynomial number of samples carries enough
information for learning the target committee with high confidence and
accuracy. Unfortunately, even simple tasks that need to be solved for learning
from these samples are intractable. We prove that deciding whether there exists
some ABCS rule that makes a given committee winning in a given profile is a
computationally hard problem. Our results extend to the class of sequential
Thiele rules, which have received attention due to their simplicity.

    

### [[2110.00260] Rapid Assessments of Light-Duty Gasoline Vehicle Emissions Using On-Road Remote Sensing and Machine Learning](http://arxiv.org/abs/2110.00260)


  In-time and accurate assessments of on-road vehicle emissions play a central
role in urban air quality and health policymaking. However, official insight is
hampered by the Inspection/Maintenance (I/M) procedure conducted in the
laboratory annually. It not only has a large gap to real-world situations
(e.g., meteorological conditions) but also is incapable of regular supervision.
Here we build a unique dataset including 103831 light-duty gasoline vehicles,
in which on-road remote sensing (ORRS) measurements are linked to the I/M
records based on the vehicle identification numbers and license plates. On this
basis, we develop an ensemble model framework that integrates three machining
learning algorithms, including neural network (NN), extreme gradient boosting
(XGBoost), and random forest (RF). We demonstrate that this ensemble model
could rapidly assess the vehicle-specific emissions (i.e., CO, HC, and NO). In
particular, the model performs quite well for the passing vehicles under normal
conditions (i.e., lower VSP (< 18 kw/t), temperature (6 ~ 32 °C), relative
humidity (< 80%), and wind speed (< 5m/s)). Together with the current emission
standard, we identify a large number of the dirty (2.33%) or clean (74.92%)
vehicles in the real world. Our results show that the ORRS measurements,
assisted by the machine-learning-based ensemble model developed here, can
realize day-to-day supervision of on-road vehicle-specific emissions. This
approach framework provides a valuable opportunity to reform the I/M procedures
globally and mitigate urban air pollution deeply.

    

### [[2110.00272] Learn to Communicate with Neural Calibration: Scalability and Generalization](http://arxiv.org/abs/2110.00272)


  The conventional design of wireless communication systems typically relies on
established mathematical models that capture the characteristics of different
communication modules. Unfortunately, such design cannot be easily and directly
applied to future wireless networks, which will be characterized by large-scale
ultra-dense networks whose design complexity scales exponentially with the
network size. Furthermore, such networks will vary dynamically in a significant
way, which makes it intractable to develop comprehensive analytical models.
Recently, deep learning-based approaches have emerged as potential alternatives
for designing complex and dynamic wireless systems. However, existing
learning-based methods have limited capabilities to scale with the problem size
and to generalize with varying network settings. In this paper, we propose a
scalable and generalizable neural calibration framework for future wireless
system design, where a neural network is adopted to calibrate the input of
conventional model-based algorithms. Specifically, the backbone of a
traditional time-efficient algorithm is integrated with deep neural networks to
achieve a high computational efficiency, while enjoying enhanced performance.
The permutation equivariance property, carried out by the topological structure
of wireless systems, is furthermore utilized to develop a generalizable neural
network architecture. The proposed neural calibration framework is applied to
solve challenging resource management problems in massive multiple-input
multiple-output (MIMO) systems. Simulation results will show that the proposed
neural calibration approach enjoys significantly improved scalability and
generalization compared with the existing learning-based methods.

    

### [[2110.00276] TyXe: Pyro-based Bayesian neural nets for Pytorch](http://arxiv.org/abs/2110.00276)


  We introduce TyXe, a Bayesian neural network library built on top of Pytorch
and Pyro. Our leading design principle is to cleanly separate architecture,
prior, inference and likelihood specification, allowing for a flexible workflow
where users can quickly iterate over combinations of these components. In
contrast to existing packages TyXe does not implement any layer classes, and
instead relies on architectures defined in generic Pytorch code. TyXe then
provides modular choices for canonical priors, variational guides, inference
techniques, and layer selections for a Bayesian treatment of the specified
architecture. Sampling tricks for variance reduction, such as local
reparameterization or flipout, are implemented as effect handlers, which can be
applied independently of other specifications. We showcase the ease of use of
TyXe to explore Bayesian versions of popular models from various libraries: toy
regression with a pure Pytorch neural network; large-scale image classification
with torchvision ResNets; graph neural networks based on DGL; and Neural
Radiance Fields built on top of Pytorch3D. Finally, we provide convenient
abstractions for variational continual learning. In all cases the change from a
deterministic to a Bayesian neural network comes with minimal modifications to
existing code, offering a broad range of researchers and practitioners alike
practical access to uncertainty estimation techniques. The library is available
at this https URL.

    

### [[2110.00284] Learning Reward Functions from Scale Feedback](http://arxiv.org/abs/2110.00284)


  Today's robots are increasingly interacting with people and need to
efficiently learn inexperienced user's preferences. A common framework is to
iteratively query the user about which of two presented robot trajectories they
prefer. While this minimizes the users effort, a strict choice does not yield
any information on how much one trajectory is preferred. We propose scale
feedback, where the user utilizes a slider to give more nuanced information. We
introduce a probabilistic model on how users would provide feedback and derive
a learning framework for the robot. We demonstrate the performance benefit of
slider feedback in simulations, and validate our approach in two user studies
suggesting that scale feedback enables more effective learning in practice.

    

### [[2110.00296] Powerpropagation: A sparsity inducing weight reparameterisation](http://arxiv.org/abs/2110.00296)


  The training of sparse neural networks is becoming an increasingly important
tool for reducing the computational footprint of models at training and
evaluation, as well enabling the effective scaling up of models. Whereas much
work over the years has been dedicated to specialised pruning techniques,
little attention has been paid to the inherent effect of gradient based
training on model sparsity. In this work, we introduce Powerpropagation, a new
weight-parameterisation for neural networks that leads to inherently sparse
models. Exploiting the behaviour of gradient descent, our method gives rise to
weight updates exhibiting a "rich get richer" dynamic, leaving low-magnitude
parameters largely unaffected by learning. Models trained in this manner
exhibit similar performance, but have a distribution with markedly higher
density at zero, allowing more parameters to be pruned safely. Powerpropagation
is general, intuitive, cheap and straight-forward to implement and can readily
be combined with various other techniques. To highlight its versatility, we
explore it in two very different settings: Firstly, following a recent line of
work, we investigate its effect on sparse training for resource-constrained
settings. Here, we combine Powerpropagation with a traditional weight-pruning
technique as well as recent state-of-the-art sparse-to-sparse algorithms,
showing superior performance on the ImageNet benchmark. Secondly, we advocate
the use of sparsity in overcoming catastrophic forgetting, where compressed
representations allow accommodating a large number of tasks at fixed model
capacity. In all cases our reparameterisation considerably increases the
efficacy of the off-the-shelf methods.

    

### [[2110.00304] Divergence-Regularized Multi-Agent Actor-Critic](http://arxiv.org/abs/2110.00304)


  Entropy regularization is a popular method in reinforcement learning (RL).
Although it has many advantages, it alters the RL objective and makes the
converged policy deviate from the optimal policy of the original Markov
Decision Process. Though divergence regularization has been proposed to settle
this problem, it cannot be trivially applied to cooperative multi-agent
reinforcement learning (MARL). In this paper, we investigate divergence
regularization in cooperative MARL and propose a novel off-policy cooperative
MARL framework, divergence-regularized multi-agent actor-critic (DMAC).
Mathematically, we derive the update rule of DMAC which is naturally
off-policy, guarantees a monotonic policy improvement and is not biased by the
regularization. DMAC is a flexible framework and can be combined with many
existing MARL algorithms. We evaluate DMAC in a didactic stochastic game and
StarCraft Multi-Agent Challenge and empirically show that DMAC substantially
improves the performance of existing MARL algorithms.

    

### [[2110.00306] Leveraging power grid topology in machine learning assisted optimal power flow](http://arxiv.org/abs/2110.00306)


  Machine learning assisted optimal power flow (OPF) aims to reduce the
computational complexity of these non-linear and non-convex constrained
optimisation problems by consigning expensive (online) optimisation to offline
training. The majority of work in this area typically employs fully-connected
neural networks (FCNN). However, recently convolutional (CNN) and graph (GNN)
neural networks have been also investigated, in effort to exploit topological
information within the power grid. Although promising results have been
obtained, there lacks a systematic comparison between these architectures
throughout literature. Accordingly, we assess the performance of a variety of
FCNN, CNN and GNN models for two fundamental approaches to machine learning
assisted OPF: regression (predicting optimal generator set-points) and
classification (predicting the active set of constraints). For several
synthetic grids with interconnected utilities, we show that locality properties
between feature and target variables are scarce, hence find limited merit of
harnessing topological information in NN models for this set of problems.

    

### [[2110.00329] Student Helping Teacher: Teacher Evolution via Self-Knowledge Distillation](http://arxiv.org/abs/2110.00329)


  Knowledge distillation usually transfers the knowledge from a pre-trained
cumbersome teacher network to a compact student network, which follows the
classical teacher-teaching-student paradigm. Based on this paradigm, previous
methods mostly focus on how to efficiently train a better student network for
deployment. Different from the existing practices, in this paper, we propose a
novel student-helping-teacher formula, Teacher Evolution via Self-Knowledge
Distillation (TESKD), where the target teacher (for deployment) is learned with
the help of multiple hierarchical students by sharing the structural backbone.
The diverse feedback from multiple students allows the teacher to improve
itself through the shared feature representations. The effectiveness of our
proposed framework is demonstrated by extensive experiments with various
network settings on two standard benchmarks including CIFAR-100 and ImageNet.
Notably, when trained together with our proposed method, ResNet-18 achieves
79.15% and 71.14% accuracy on CIFAR-100 and ImageNet, outperforming the
baseline results by 4.74% and 1.43%, respectively. The code is available at:
this https URL.

    

### [[2110.00330] Discovering Boundary Values of Feature-based Machine Learning Classifiers through Exploratory Datamorphic Testing](http://arxiv.org/abs/2110.00330)


  Testing has been widely recognised as difficult for AI applications. This
paper proposes a set of testing strategies for testing machine learning
applications in the framework of the datamorphism testing methodology. In these
strategies, testing aims at exploring the data space of a classification or
clustering application to discover the boundaries between classes that the
machine learning application defines. This enables the tester to understand
precisely the behaviour and function of the software under test. In the paper,
three variants of exploratory strategies are presented with the algorithms
implemented in the automated datamorphic testing tool Morphy. The correctness
of these algorithms are formally proved. Their capability and cost of
discovering borders between classes are evaluated via a set of controlled
experiments with manually designed subjects and a set of case studies with real
machine learning models.

    

### [[2110.00337] PhiNets: a scalable backbone for low-power AI at the edge](http://arxiv.org/abs/2110.00337)


  In the Internet of Things era, where we see many interconnected and
heterogeneous mobile and fixed smart devices, distributing the intelligence
from the cloud to the edge has become a necessity. Due to limited computational
and communication capabilities, low memory and limited energy budget, bringing
artificial intelligence algorithms to peripheral devices, such as the end-nodes
of a sensor network, is a challenging task and requires the design of
innovative methods. In this work, we present PhiNets, a new scalable backbone
optimized for deep-learning-based image processing on resource-constrained
platforms. PhiNets are based on inverted residual blocks specifically designed
to decouple the computational cost, working memory, and parameter memory, thus
exploiting all the available resources. With a YoloV2 detection head and Simple
Online and Realtime Tracking, the proposed architecture has achieved the
state-of-the-art results in (i) detection on the COCO and VOC2012 benchmarks,
and (ii) tracking on the MOT15 benchmark. PhiNets reduce the parameter count of
87% to 93% with respect to previous state-of-the-art models (EfficientNetv1,
MobileNetv2) and achieve better performance with lower computational cost.
Moreover, we demonstrate our approach on a prototype node based on a STM32H743
microcontroller (MCU) with 2MB of internal Flash and 1MB of RAM and achieve
power requirements in the order of 10 mW. The code for the PhiNets is publicly
available on GitHub.

    

### [[2110.00351] Smooth Normalizing Flows](http://arxiv.org/abs/2110.00351)


  Normalizing flows are a promising tool for modeling probability distributions
in physical systems. While state-of-the-art flows accurately approximate
distributions and energies, applications in physics additionally require smooth
energies to compute forces and higher-order derivatives. Furthermore, such
densities are often defined on non-trivial topologies. A recent example are
Boltzmann Generators for generating 3D-structures of peptides and small
proteins. These generative models leverage the space of internal coordinates
(dihedrals, angles, and bonds), which is a product of hypertori and compact
intervals. In this work, we introduce a class of smooth mixture transformations
working on both compact intervals and hypertori. Mixture transformations employ
root-finding methods to invert them in practice, which has so far prevented
bi-directional flow training. To this end, we show that parameter gradients and
forces of such inverses can be computed from forward evaluations via the
inverse function theorem. We demonstrate two advantages of such smooth flows:
they allow training by force matching to simulation data and can be used as
potentials in molecular dynamics simulations.

    

### [[2110.00367] Multi Expression Programming -- an in-depth description](http://arxiv.org/abs/2110.00367)


  Multi Expression Programming (MEP) is a Genetic Programming variant that uses
a linear representation of chromosomes. MEP individuals are strings of genes
encoding complex computer programs. When MEP individuals encode expressions,
their representation is similar to the way in which compilers translate $C$ or
$Pascal$ expressions into machine code. A unique MEP feature is the ability to
store multiple solutions of a problem in a single chromosome. Usually, the best
solution is chosen for fitness assignment. When solving symbolic regression or
classification problems (or any other problems for which the training set is
known before the problem is solved) MEP has the same complexity as other
techniques storing a single solution in a chromosome (such as GP, CGP, GEP or
GE). Evaluation of the expressions encoded into an MEP individual can be
performed by a single parsing of the chromosome. Offspring obtained by
crossover and mutation is always syntactically correct MEP individuals
(computer programs). Thus, no extra processing for repairing newly obtained
individuals is needed.

    

### [[2110.00375] Fully Spiking Variational Autoencoder](http://arxiv.org/abs/2110.00375)


  Spiking neural networks (SNNs) can be run on neuromorphic devices with
ultra-high speed and ultra-low energy consumption because of their binary and
event-driven nature. Therefore, SNNs are expected to have various applications,
including as generative models being running on edge devices to create
high-quality images. In this study, we build a variational autoencoder (VAE)
with SNN to enable image generation. VAE is known for its stability among
generative models; recently, its quality advanced. In vanilla VAE, the latent
space is represented as a normal distribution, and floating-point calculations
are required in sampling. However, this is not possible in SNNs because all
features must be binary time series data. Therefore, we constructed the latent
space with an autoregressive SNN model, and randomly selected samples from its
output to sample the latent variables. This allows the latent variables to
follow the Bernoulli process and allows variational learning. Thus, we build
the Fully Spiking Variational Autoencoder where all modules are constructed
with SNN. To the best of our knowledge, we are the first to build a VAE only
with SNN layers. We experimented with several datasets, and confirmed that it
can generate images with the same or better quality compared to conventional
ANNs. The code will be available soon.

    

### [[2110.00385] Neural Dependency Coding inspired Multimodal Fusion](http://arxiv.org/abs/2110.00385)


  Information integration from different modalities is an active area of
research. Human beings and, in general, biological neural systems are quite
adept at using a multitude of signals from different sensory perceptive fields
to interact with the environment and each other. Recent work in deep fusion
models via neural networks has led to substantial improvements over unimodal
approaches in areas like speech recognition, emotion recognition and analysis,
captioning and image description. However, such research has mostly focused on
architectural changes allowing for fusion of different modalities while keeping
the model complexity manageable. Inspired by recent neuroscience ideas about
multisensory integration and processing, we investigate the effect of synergy
maximizing loss functions. Experiments on multimodal sentiment analysis tasks:
CMU-MOSI and CMU-MOSEI with different models show that our approach provides a
consistent performance boost.

    

### [[2110.00391] Online Primal-Dual Algorithms with Predictions for Packing Problems](http://arxiv.org/abs/2110.00391)


  The domain of online algorithms with predictions has been extensively studied
for different applications such as scheduling, caching (paging), clustering,
ski rental, etc. Recently, Bamas et al., aiming for an unified method, have
provided a primal-dual framework for linear covering problems. They extended
the online primal-dual method by incorporating predictions in order to achieve
a performance beyond the worst-case case analysis. In this paper, we consider
this research line and present a framework to design algorithms with
predictions for non-linear packing problems. We illustrate the applicability of
our framework in submodular maximization and in particular ad-auction
maximization in which the optimal bound is given and supporting experiments are
provided.

    

### [[2110.00392] Tree in Tree: from Decision Trees to Decision Graphs](http://arxiv.org/abs/2110.00392)


  Decision trees have been widely used as classifiers in many machine learning
applications thanks to their lightweight and interpretable decision process.
This paper introduces Tree in Tree decision graph (TnT), a framework that
extends the conventional decision tree to a more generic and powerful directed
acyclic graph. TnT constructs decision graphs by recursively growing decision
trees inside the internal or leaf nodes instead of greedy training. The time
complexity of TnT is linear to the number of nodes in the graph, and it can
construct decision graphs on large datasets. Compared to decision trees, we
show that TnT achieves better classification performance with reduced model
size, both as a stand-alone classifier and as a base estimator in
bagging/AdaBoost ensembles. Our proposed model is a novel, more efficient, and
accurate alternative to the widely-used decision trees.

    

### [[2110.00394] Personalized Retrogress-Resilient Framework for Real-World Medical Federated Learning](http://arxiv.org/abs/2110.00394)


  Nowadays, deep learning methods with large-scale datasets can produce
clinically useful models for computer-aided diagnosis. However, the privacy and
ethical concerns are increasingly critical, which make it difficult to collect
large quantities of data from multiple institutions. Federated Learning (FL)
provides a promising decentralized solution to train model collaboratively by
exchanging client models instead of private data. However, the server
aggregation of existing FL methods is observed to degrade the model performance
in real-world medical FL setting, which is termed as retrogress. To address
this problem, we propose a personalized retrogress-resilient framework to
produce a superior personalized model for each client. Specifically, we devise
a Progressive Fourier Aggregation (PFA) at the server to achieve more stable
and effective global knowledge gathering by integrating client models from
low-frequency to high-frequency gradually. Moreover, with an introduced deputy
model to receive the aggregated server model, we design a Deputy-Enhanced
Transfer (DET) strategy at the client and conduct three steps of
Recover-Exchange-Sublimate to ameliorate the personalized local model by
transferring the global knowledge smoothly. Extensive experiments on real-world
dermoscopic FL dataset prove that our personalized retrogress-resilient
framework outperforms state-of-the-art FL methods, as well as the
generalization on an out-of-distribution cohort. The code and dataset are
available at this https URL.

    

### [[2110.00413] Detecting Harmful Memes and Their Targets](http://arxiv.org/abs/2110.00413)


  Among the various modes of communication in social media, the use of Internet
memes has emerged as a powerful means to convey political, psychological, and
socio-cultural opinions. Although memes are typically humorous in nature,
recent days have witnessed a proliferation of harmful memes targeted to abuse
various social entities. As most harmful memes are highly satirical and
abstruse without appropriate contexts, off-the-shelf multimodal models may not
be adequate to understand their underlying semantics. In this work, we propose
two novel problem formulations: detecting harmful memes and the social entities
that these harmful memes target. To this end, we present HarMeme, the first
benchmark dataset, containing 3,544 memes related to COVID-19. Each meme went
through a rigorous two-stage annotation process. In the first stage, we labeled
a meme as very harmful, partially harmful, or harmless; in the second stage, we
further annotated the type of target(s) that each harmful meme points to:
individual, organization, community, or society/general public/other. The
evaluation results using ten unimodal and multimodal models highlight the
importance of using multimodal signals for both tasks. We further discuss the
limitations of these models and we argue that more research is needed to
address these problems.

    

### [[2110.00414] Predicting Flat-Fading Channels via Meta-Learned Closed-Form Linear Filters and Equilibrium Propagation](http://arxiv.org/abs/2110.00414)


  Predicting fading channels is a classical problem with a vast array of
applications, including as an enabler of artificial intelligence (AI)-based
proactive resource allocation for cellular networks. Under the assumption that
the fading channel follows a stationary complex Gaussian process, as for
Rayleigh and Rician fading models, the optimal predictor is linear, and it can
be directly computed from the Doppler spectrum via standard linear minimum mean
squared error (LMMSE) estimation. However, in practice, the Doppler spectrum is
unknown, and the predictor has only access to a limited time series of
estimated channels. This paper proposes to leverage meta-learning in order to
mitigate the requirements in terms of training data for channel fading
prediction. Specifically, it first develops an offline low-complexity solution
based on linear filtering via a meta-trained quadratic regularization. Then, an
online method is proposed based on gradient descent and equilibrium propagation
(EP). Numerical results demonstrate the advantages of the proposed approach,
showing its capacity to approach the genie-aided LMMSE solution with a small
number of training data points.

    

### [[2110.00415] Optimization Networks for Integrated Machine Learning](http://arxiv.org/abs/2110.00415)


  Optimization networks are a new methodology for holistically solving
interrelated problems that have been developed with combinatorial optimization
problems in mind. In this contribution we revisit the core principles of
optimization networks and demonstrate their suitability for solving machine
learning problems. We use feature selection in combination with linear model
creation as a benchmark application and compare the results of optimization
networks to ordinary least squares with optional elastic net regularization.
Based on this example we justify the advantages of optimization networks by
adapting the network to solve other machine learning problems. Finally,
optimization analysis is presented, where optimal input values of a system have
to be found to achieve desired output values. Optimization analysis can be
divided into three subproblems: model creation to describe the system, model
selection to choose the most appropriate one and parameter optimization to
obtain the input values. Therefore, optimization networks are an obvious choice
for handling optimization analysis tasks.

    

### [[2110.00416] FiLMing Multimodal Sarcasm Detection with Attention](http://arxiv.org/abs/2110.00416)


  Sarcasm detection identifies natural language expressions whose intended
meaning is different from what is implied by its surface meaning. It finds
applications in many NLP tasks such as opinion mining, sentiment analysis, etc.
Today, social media has given rise to an abundant amount of multimodal data
where users express their opinions through text and images. Our paper aims to
leverage multimodal data to improve the performance of the existing systems for
sarcasm detection. So far, various approaches have been proposed that uses text
and image modality and a fusion of both. We propose a novel architecture that
uses the RoBERTa model with a co-attention layer on top to incorporate context
incongruity between input text and image attributes. Further, we integrate
feature-wise affine transformation by conditioning the input image through
FiLMed ResNet blocks with the textual features using the GRU network to capture
the multimodal information. The output from both the models and the CLS token
from RoBERTa is concatenated and used for the final prediction. Our results
demonstrate that our proposed model outperforms the existing state-of-the-art
method by 6.14% F1 score on the public Twitter multimodal sarcasm detection
dataset.

    

### [[2110.00418] Evaluation of Non-Negative Matrix Factorization and n-stage Latent Dirichlet Allocation for Emotion Analysis in Turkish Tweets](http://arxiv.org/abs/2110.00418)


  With the development of technology, the use of social media has become quite
common. Analyzing comments on social media in areas such as media and
advertising plays an important role today. For this reason, new and traditional
natural language processing methods are used to detect the emotion of these
shares. In this paper, the Latent Dirichlet Allocation, namely LDA, and
Non-Negative Matrix Factorization methods in topic modeling were used to
determine which emotion the Turkish tweets posted via Twitter. In addition, the
accuracy of a proposed n-level method based on LDA was analyzed. Dataset
consists of 5 emotions, namely angry, fear, happy, sad and confused. NMF was
the most successful method among all topic modeling methods in this study.
Then, the F1-measure of Random Forest, Naive Bayes and Support Vector Machine
methods was analyzed by obtaining a file suitable for Weka by using the word
weights and class labels of the topics. Among the Weka results, the most
successful method was n-stage LDA, and the most successful algorithm was Random
Forest.

    

### [[2110.00429] Topologically-Informed Atlas Learning](http://arxiv.org/abs/2110.00429)


  We present a new technique that enables manifold learning to accurately embed
data manifolds that contain holes, without discarding any topological
information. Manifold learning aims to embed high dimensional data into a lower
dimensional Euclidean space by learning a coordinate chart, but it requires
that the entire manifold can be embedded in a single chart. This is impossible
for manifolds with holes. In such cases, it is necessary to learn an atlas: a
collection of charts that collectively cover the entire manifold. We begin with
many small charts, and combine them in a bottom-up approach, where charts are
only combined if doing so will not introduce problematic topological features.
When it is no longer possible to combine any charts, each chart is individually
embedded with standard manifold learning techniques, completing the
construction of the atlas. We show the efficacy of our method by constructing
atlases for challenging synthetic manifolds; learning human motion embeddings
from motion capture data; and learning kinematic models of articulated objects.

    

### [[2110.00438] Guiding Evolutionary Strategies by Differentiable Robot Simulators](http://arxiv.org/abs/2110.00438)


  In recent years, Evolutionary Strategies were actively explored in robotic
tasks for policy search as they provide a simpler alternative to reinforcement
learning algorithms. However, this class of algorithms is often claimed to be
extremely sample-inefficient. On the other hand, there is a growing interest in
Differentiable Robot Simulators (DRS) as they potentially can find successful
policies with only a handful of trajectories. But the resulting gradient is not
always useful for the first-order optimization. In this work, we demonstrate
how DRS gradient can be used in conjunction with Evolutionary Strategies.
Preliminary results suggest that this combination can reduce sample complexity
of Evolutionary Strategies by 3x-5x times in both simulation and the real
world.

    

### [[2110.00445] Sim and Real: Better Together](http://arxiv.org/abs/2110.00445)


  Simulation is used extensively in autonomous systems, particularly in robotic
manipulation. By far, the most common approach is to train a controller in
simulation, and then use it as an initial starting point for the real system.
We demonstrate how to learn simultaneously from both simulation and interaction
with the real environment. We propose an algorithm for balancing the large
number of samples from the high throughput but less accurate simulation and the
low-throughput, high-fidelity and costly samples from the real environment. We
achieve that by maintaining a replay buffer for each environment the agent
interacts with. We analyze such multi-environment interaction theoretically,
and provide convergence properties, through a novel theoretical replay buffer
analysis. We demonstrate the efficacy of our method on a sim-to-real
environment.

    

### [[2110.00449] Arbitrary Marginal Neural Ratio Estimation for Simulation-based Inference](http://arxiv.org/abs/2110.00449)


  In many areas of science, complex phenomena are modeled by stochastic
parametric simulators, often featuring high-dimensional parameter spaces and
intractable likelihoods. In this context, performing Bayesian inference can be
challenging. In this work, we present a novel method that enables amortized
inference over arbitrary subsets of the parameters, without resorting to
numerical integration, which makes interpretation of the posterior more
convenient. Our method is efficient and can be implemented with arbitrary
neural network architectures. We demonstrate the applicability of the method on
parameter inference of binary black hole systems from gravitational waves
observations.

    

### [[2110.00452] SAM: A Self-adaptive Attention Module for Context-Aware Recommendation System](http://arxiv.org/abs/2110.00452)


  Recently, textual information has been proved to play a positive role in
recommendation systems. However, most of the existing methods only focus on
representation learning of textual information in ratings, while potential
selection bias induced by the textual information is ignored. In this work, we
propose a novel and general self-adaptive module, the Self-adaptive Attention
Module (SAM), which adjusts the selection bias by capturing contextual
information based on its representation. This module can be embedded into
recommendation systems that contain learning components of contextual
information. Experimental results on three real-world datasets demonstrate the
effectiveness of our proposal, and the state-of-the-art models with SAM
significantly outperform the original ones.

    

### [[2110.00455] Towards Gradient-based Bilevel Optimization with Non-convex Followers and Beyond](http://arxiv.org/abs/2110.00455)


  In recent years, Bi-Level Optimization (BLO) techniques have received
extensive attentions from both learning and vision communities. A variety of
BLO models in complex and practical tasks are of non-convex follower structure
in nature (a.k.a., without Lower-Level Convexity, LLC for short). However, this
challenging class of BLOs is lack of developments on both efficient solution
strategies and solid theoretical guarantees. In this work, we propose a new
algorithmic framework, named Initialization Auxiliary and Pessimistic
Trajectory Truncated Gradient Method (IAPTT-GM), to partially address the above
issues. In particular, by introducing an auxiliary as initialization to guide
the optimization dynamics and designing a pessimistic trajectory truncation
operation, we construct a reliable approximate version of the original BLO in
the absence of LLC hypothesis. Our theoretical investigations establish the
convergence of solutions returned by IAPTT-GM towards those of the original BLO
without LLC. As an additional bonus, we also theoretically justify the quality
of our IAPTT-GM embedded with Nesterov's accelerated dynamics under LLC. The
experimental results confirm both the convergence of our algorithm without LLC,
and the theoretical findings under LLC.

    

### [[2110.00459] Characterizing Concurrency Mechanisms for NVIDIA GPUs under Deep Learning Workloads](http://arxiv.org/abs/2110.00459)


  We investigate the performance of the concurrency mechanisms available on
NVIDIA's new Ampere GPU microarchitecture under deep learning training and
inference workloads. In contrast to previous studies that treat the GPU as a
black box, we examine scheduling at the microarchitectural level. We find that
the lack of fine-grained preemption mechanisms, robust task prioritization
options, and contention-aware thread block placement policies limits the
effectiveness of NVIDIA's concurrency mechanisms. In summary, the sequential
nature of deep learning workloads and their fluctuating resource requirements
and kernel runtimes make executing such workloads while maintaining
consistently high utilization and low, predictable turnaround times difficult
on current NVIDIA hardware.

    

### [[2110.00468] New Evolutionary Computation Models and their Applications to Machine Learning](http://arxiv.org/abs/2110.00468)


  Automatic Programming is one of the most important areas of computer science
research today. Hardware speed and capability have increased exponentially, but
the software is years behind. The demand for software has also increased
significantly, but it is still written in old fashion: by using humans.
There are multiple problems when the work is done by humans: cost, time,
quality. It is costly to pay humans, it is hard to keep them satisfied for a
long time, it takes a lot of time to teach and train them and the quality of
their output is in most cases low (in software, mostly due to bugs).
The real advances in human civilization appeared during the industrial
revolutions. Before the first revolution, most people worked in agriculture.
Today, very few percent of people work in this field.
A similar revolution must appear in the computer programming field.
Otherwise, we will have so many people working in this field as we had in the
past working in agriculture.
How do people know how to write computer programs? Very simple: by learning.
Can we do the same for software? Can we put the software to learn how to write
software?
It seems that is possible (to some degree) and the term is called Machine
Learning. It was first coined in 1959 by the first person who made a computer
perform a serious learning task, namely, Arthur Samuel.
However, things are not so easy as in humans (well, truth to be said - for
some humans it is impossible to learn how to write software). So far we do not
have software that can learn perfectly to write software. We have some
particular cases where some programs do better than humans, but the examples
are sporadic at best. Learning from experience is difficult for computer
programs. Instead of trying to simulate how humans teach humans how to write
computer programs, we can simulate nature.

    

### [[2110.00473] Score-Based Generative Classifiers](http://arxiv.org/abs/2110.00473)


  The tremendous success of generative models in recent years raises the
question whether they can also be used to perform classification. Generative
models have been used as adversarially robust classifiers on simple datasets
such as MNIST, but this robustness has not been observed on more complex
datasets like CIFAR-10. Additionally, on natural image datasets, previous
results have suggested a trade-off between the likelihood of the data and
classification accuracy. In this work, we investigate score-based generative
models as classifiers for natural images. We show that these models not only
obtain competitive likelihood values but simultaneously achieve
state-of-the-art classification accuracy for generative classifiers on
CIFAR-10. Nevertheless, we find that these models are only slightly, if at all,
more robust than discriminative baseline models on out-of-distribution tasks
based on common image corruptions. Similarly and contrary to prior results, we
find that score-based are prone to worst-case distribution shifts in the form
of adversarial perturbations. Our work highlights that score-based generative
models are closing the gap in classification accuracy compared to standard
discriminative models. While they do not yet deliver on the promise of
adversarial and out-of-domain robustness, they provide a different approach to
classification that warrants further research.

    

### [[2110.00478] SECDA: Efficient Hardware/Software Co-Design of FPGA-based DNN Accelerators for Edge Inference](http://arxiv.org/abs/2110.00478)


  Edge computing devices inherently face tight resource constraints, which is
especially apparent when deploying Deep Neural Networks (DNN) with high memory
and compute demands. FPGAs are commonly available in edge devices. Since these
reconfigurable circuits can achieve higher throughput and lower power
consumption than general purpose processors, they are especially well-suited
for DNN acceleration. However, existing solutions for designing FPGA-based DNN
accelerators for edge devices come with high development overheads, given the
cost of repeated FPGA synthesis passes, reimplementation in a Hardware
Description Language (HDL) of the simulated design, and accelerator system
integration.
In this paper we propose SECDA, a new hardware/software co-design methodology
to reduce design time of optimized DNN inference accelerators on edge devices
with FPGAs. SECDA combines cost-effective SystemC simulation with hardware
execution, streamlining design space exploration and the development process
via reduced design evaluation time. As a case study, we use SECDA to
efficiently develop two different DNN accelerator designs on a PYNQ-Z1 board, a
platform that includes an edge FPGA. We quickly and iteratively explore the
system's hardware/software stack, while identifying and mitigating performance
bottlenecks. We evaluate the two accelerator designs with four common DNN
models, achieving an average performance speedup across models of up to
3.5$\times$ with a 2.9$\times$ reduction in energy consumption over CPU-only
inference. Our code is available at this https URL


### [[2110.00481] Personalized Rehabilitation Robotics based on Online Learning Control](http://arxiv.org/abs/2110.00481)


  The use of rehabilitation robotics in clinical applications gains increasing
importance, due to therapeutic benefits and the ability to alleviate
labor-intensive works. However, their practical utility is dependent on the
deployment of appropriate control algorithms, which adapt the level of
task-assistance according to each individual patient's need. Generally, the
required personalization is achieved through manual tuning by clinicians, which
is cumbersome and error-prone. In this work we propose a novel online learning
control architecture, which is able to personalize the control force at run
time to each individual user. To this end, we deploy Gaussian process-based
online learning with previously unseen prediction and update rates. Finally, we
evaluate our method in an experimental user study, where the learning
controller is shown to provide personalized control, while also obtaining safe
interaction forces.

    

### [[2110.00494] Probabilistic Robust Autoencoders for Anomaly Detection](http://arxiv.org/abs/2110.00494)


  Empirical observations often consist of anomalies (or outliers) that
contaminate the data. Accurate identification of anomalous samples is crucial
for the success of downstream data analysis tasks. To automatically identify
anomalies, we propose a new type of autoencoder (AE) which we term
Probabilistic Robust autoencoder (PRAE). PRAE is designed to simultaneously
remove outliers and identify a low-dimensional representation for the inlier
samples. We first describe Robust AE (RAE) as a model that aims to split the
data to inlier samples from which a low dimensional representation is learned
via an AE, and anomalous (outlier) samples that are excluded as they do not fit
the low dimensional representation. Robust AE minimizes the reconstruction of
the AE while attempting to incorporate as many observations as possible. This
could be realized by subtracting from the reconstruction term an $\ell_0$ norm
counting the number of selected observations. Since the $\ell_0$ norm is not
differentiable, we propose two probabilistic relaxations for the RAE approach
and demonstrate that they can effectively identify anomalies. We prove that the
solution to PRAE is equivalent to the solution of RAE and demonstrate using
extensive simulations that PRAE is at par with state-of-the-art methods for
anomaly detection.

    

### [[2110.00502] Predicting Consumer Purchasing Decision in The Online Food Delivery Industry](http://arxiv.org/abs/2110.00502)


  This transformation of food delivery businesses to online platforms has
gained high attention in recent years. This due to the availability of
customizing ordering experiences, easy payment methods, fast delivery, and
others. The competition between online food delivery providers has intensified
to attain a wider range of customers. Hence, they should have a better
understanding of their customers' needs and predict their purchasing decisions.
Machine learning has a significant impact on companies' bottom line. They are
used to construct models and strategies in industries that rely on big data and
need a system to evaluate it fast and effectively. Predictive modeling is a
type of machine learning that uses various regression algorithms, analytics,
and statistics to estimate the probability of an occurrence. The incorporation
of predictive models helps online food delivery providers to understand their
customers. In this study, a dataset collected from 388 consumers in Bangalore,
India was provided to predict their purchasing decisions. Four prediction
models are considered: CART and C4.5 decision trees, random forest, and
rule-based classifiers, and their accuracies in providing the correct class
label are evaluated. The findings show that all models perform similarly, but
the C4.5 outperforms them all with an accuracy of 91.67%.

    

### [[2110.00508] An Ensemble-based Multi-Criteria Decision Making Method for COVID-19 Cough Classification](http://arxiv.org/abs/2110.00508)


  The objectives of this research are analysing the performance of the
state-of-the-art machine learning techniques for classifying COVID-19 from
cough sound and identifying the model(s) that consistently perform well across
different cough datasets. Different performance evaluation metrics (such as
precision, sensitivity, specificity, AUC, accuracy, etc.) make it difficult to
select the best performance model. To address this issue, in this paper, we
propose an ensemble-based multi-criteria decision making (MCDM) method for
selecting top performance machine learning technique(s) for COVID-19 cough
classification. We use four cough datasets, namely Cambridge, Coswara, Virufy,
and NoCoCoDa to verify the proposed method. At first, our proposed method uses
the audio features of cough samples and then applies machine learning (ML)
techniques to classify them as COVID-19 or non-COVID-19. Then, we consider a
multi-criteria decision-making (MCDM) method that combines ensemble
technologies (i.e., soft and hard) to select the best model. In MCDM, we use
the technique for order preference by similarity to ideal solution (TOPSIS) for
ranking purposes, while entropy is applied to calculate evaluation criteria
weights. In addition, we apply the feature reduction process through recursive
feature elimination with cross-validation under different estimators. The
results of our empirical evaluations show that the proposed method outperforms
the state-of-the-art models.

    

### [[2110.00512] Optic Disc Segmentation using Disk-Centered Patch Augmentation](http://arxiv.org/abs/2110.00512)


  The optic disc is a crucial diagnostic feature in the eye since changes to
its physiognomy is correlated with the severity of various ocular and
cardiovascular diseases. While identifying the bulk of the optic disc in a
color fundus image is straightforward, accurately segmenting its boundary at
the pixel level is very challenging. In this work, we propose disc-centered
patch augmentation (DCPA) -- a simple, yet novel training scheme for deep
neural networks -- to address this problem. DCPA achieves state-of-the-art
results on full-size images even when using small neural networks, specifically
a U-Net with only 7 million parameters as opposed to the original 31 million.
In DCPA, we restrict the training data to patches that fully contain the optic
nerve. In addition, we also train the network using dynamic cost functions to
increase its robustness. We tested DCPA-trained networks on five retinal
datasets: DRISTI, DRIONS-DB, DRIVE, AV-WIDE, and CHASE-DB. The first two had
available optic disc ground truth, and we manually estimated the ground truth
for the latter three. Our approach achieved state-of-the-art F1 and IOU results
on four datasets (95 % F1, 91 % IOU on DRISTI; 92 % F1, 84 % IOU on DRIVE; 83 %
F1, 71 % IOU on AV-WIDE; 83 % F1, 71 % IOU on CHASEDB) and competitive results
on the fifth (95 % F1, 91 % IOU on DRIONS-DB), confirming its generality. Our
open-source code and ground-truth annotations are available at:
this https URL


### [[2110.00513] Belief propagation for permutations, rankings, and partial orders](http://arxiv.org/abs/2110.00513)


  Many datasets give partial information about an ordering or ranking by
indicating which team won a game, which item a user prefers, or who infected
whom. We define a continuous spin system whose Gibbs distribution is the
posterior distribution on permutations, given a probabilistic model of these
interactions. Using the cavity method we derive a belief propagation algorithm
that computes the marginal distribution of each node's position. In addition,
the Bethe free energy lets us approximate the number of linear extensions of a
partial order and perform model selection.

    

### [[2110.00516] LEMON: Explainable Entity Matching](http://arxiv.org/abs/2110.00516)


  State-of-the-art entity matching (EM) methods are hard to interpret, and
there is significant value in bringing explainable AI to EM. Unfortunately,
most popular explainability methods do not work well out of the box for EM and
need adaptation. In this paper, we identify three challenges of applying local
post hoc feature attribution methods to entity matching: cross-record
interaction effects, non-match explanations, and variation in sensitivity. We
propose our novel model-agnostic and schema-flexible method LEMON that
addresses all three challenges by (i) producing dual explanations to avoid
cross-record interaction effects, (ii) introducing the novel concept of
attribution potential to explain how two records could have matched, and (iii)
automatically choosing explanation granularity to match the sensitivity of the
matcher and record pair in question. Experiments on public datasets demonstrate
that the proposed method is more faithful to the matcher and does a better job
of helping users understand the decision boundary of the matcher than previous
work. Furthermore, user studies show that the rate at which human subjects can
construct counterfactual examples after seeing an explanation from our proposed
method increases from 54% to 64% for matches and from 15% to 49% for
non-matches compared to explanations from a standard adaptation of LIME.

    

### [[2110.00528] Do Self-Supervised and Supervised Methods Learn Similar Visual Representations?](http://arxiv.org/abs/2110.00528)


  Despite the success of a number of recent techniques for visual
self-supervised deep learning, there remains limited investigation into the
representations that are ultimately learned. By using recent advances in
comparing neural representations, we explore in this direction by comparing a
constrastive self-supervised algorithm (SimCLR) to supervision for simple image
data in a common architecture. We find that the methods learn similar
intermediate representations through dissimilar means, and that the
representations diverge rapidly in the final few layers. We investigate this
divergence, finding that it is caused by these layers strongly fitting to the
distinct learning objectives. We also find that SimCLR's objective implicitly
fits the supervised objective in intermediate layers, but that the reverse is
not true. Our work particularly highlights the importance of the learned
intermediate representations, and raises important questions for auxiliary task
design.

    

### [[2110.00529] Unsupervised Motion Representation Learning with Capsule Autoencoders](http://arxiv.org/abs/2110.00529)


  We propose the Motion Capsule Autoencoder (MCAE), which addresses a key
challenge in the unsupervised learning of motion representations:
transformation invariance. MCAE models motion in a two-level hierarchy. In the
lower level, a spatio-temporal motion signal is divided into short, local, and
semantic-agnostic snippets. In the higher level, the snippets are aggregated to
form full-length semantic-aware segments. For both levels, we represent motion
with a set of learned transformation invariant templates and the corresponding
geometric transformations by using capsule autoencoders of a novel design. This
leads to a robust and efficient encoding of viewpoint changes. MCAE is
evaluated on a novel Trajectory20 motion dataset and various real-world
skeleton-based human action datasets. Notably, it achieves better results than
baselines on Trajectory20 with considerably fewer parameters and
state-of-the-art performance on the unsupervised skeleton-based action
recognition task.

    

### [[2110.00530] A survey on datasets for fairness-aware machine learning](http://arxiv.org/abs/2110.00530)


  As decision-making increasingly relies on machine learning and (big) data,
the issue of fairness in data-driven AI systems is receiving increasing
attention from both research and industry. A large variety of fairness-aware
machine learning solutions have been proposed which propose fairness-related
interventions in the data, learning algorithms and/or model outputs. However, a
vital part of proposing new approaches is evaluating them empirically on
benchmark datasets that represent realistic and diverse settings. Therefore, in
this paper, we overview real-world datasets used for fairness-aware machine
learning. We focus on tabular data as the most common data representation for
fairness-aware machine learning. We start our analysis by identifying
relationships among the different attributes, particularly w.r.t. protected
attributes and class attributes, using a Bayesian network. For a deeper
understanding of bias and fairness in the datasets, we investigate the
interesting relationships using exploratory analysis.

    

### [[2110.00531] A survey on active noise control techniques -- Part I: Linear systems](http://arxiv.org/abs/2110.00531)


  Active noise control (ANC) is an effective way for reducing the noise level
in electroacoustic or electromechanical systems. Since its first introduction
in 1936, this approach has been greatly developed. This paper focuses on
discussing the development of ANC techniques over the past decade. Linear ANC
algorithms, including the celebrated filtered-x least-mean-square (FxLMS)-based
algorithms and distributed ANC algorithms, are investigated and evaluated.
Nonlinear ANC (NLANC) techniques, such as functional link artificial neural
network (FLANN)-based algorithms, are pursued in Part II. Furthermore, some
novel methods and applications of ANC emerging in the past decade are
summarized. Finally, future research challenges regarding the ANC technique are
discussed.

    

### [[2110.00532] Fed-LAMB: Layerwise and Dimensionwise Locally Adaptive Optimization Algorithm](http://arxiv.org/abs/2110.00532)


  In the emerging paradigm of federated learning (FL), large amount of clients,
such as mobile devices, are used to train possibly high-dimensional models on
their respective data. Due to the low bandwidth of mobile devices,
decentralized optimization methods need to shift the computation burden from
those clients to the computation server while preserving privacy and reasonable
communication cost. In this paper, we focus on the training of deep, as in
multilayered, neural networks, under the FL settings. We present Fed-LAMB, a
novel federated learning method based on a layerwise and dimensionwise updates
of the local models, alleviating the nonconvexity and the multilayered nature
of the optimization task at hand. We provide a thorough finite-time convergence
analysis for Fed-LAMB characterizing how fast its gradient decreases. We
provide experimental results under iid and non-iid settings to corroborate not
only our theory, but also exhibit the faster convergence of our method,
compared to the state-of-the-art.

    

### [[2110.00535] A Cramér Distance perspective on Non-crossing Quantile Regression in Distributional Reinforcement Learning](http://arxiv.org/abs/2110.00535)


  Distributional reinforcement learning (DRL) extends the value-based approach
by using a deep convolutional network to approximate the full distribution over
future returns instead of the mean only, providing a richer signal that leads
to improved performances. Quantile-based methods like QR-DQN project arbitrary
distributions onto a parametric subset of staircase distributions by minimizing
the 1-Wasserstein distance, however, due to biases in the gradients, the
quantile regression loss is used instead for training, guaranteeing the same
minimizer and enjoying unbiased gradients. Recently, monotonicity constraints
on the quantiles have been shown to improve the performance of QR-DQN for
uncertainty-based exploration strategies. The contribution of this work is in
the setting of fixed quantile levels and is twofold. First, we prove that the
Cramér distance yields a projection that coincides with the 1-Wasserstein one
and that, under monotonicity constraints, the squared Cramér and the quantile
regression losses yield collinear gradients, shedding light on the connection
between these important elements of DRL. Second, we propose a novel
non-crossing neural architecture that allows a good training performance using
a novel algorithm to compute the Cramér distance, yielding significant
improvements over QR-DQN in a number of games of the standard Atari 2600
benchmark.

    

### [[2110.00538] Evaluating the fairness of fine-tuning strategies in self-supervised learning](http://arxiv.org/abs/2110.00538)


  In this work we examine how fine-tuning impacts the fairness of contrastive
Self-Supervised Learning (SSL) models. Our findings indicate that Batch
Normalization (BN) statistics play a crucial role, and that updating only the
BN statistics of a pre-trained SSL backbone improves its downstream fairness
(36% worst subgroup, 25% mean subgroup gap). This procedure is competitive with
supervised learning, while taking 4.4x less time to train and requiring only
0.35% as many parameters to be updated. Finally, inspired by recent work in
supervised learning, we find that updating BN statistics and training residual
skip connections (12.3% of the parameters) achieves parity with a fully
fine-tuned model, while taking 1.33x less time to train.

    

### [[2110.00539] Applying Differential Privacy to Tensor Completion](http://arxiv.org/abs/2110.00539)


  Tensor completion aims at filling the missing or unobserved entries based on
partially observed tensors. However, utilization of the observed tensors often
raises serious privacy concerns in many practical scenarios. To address this
issue, we propose a solid and unified framework that contains several
approaches for applying differential privacy to the two most widely used tensor
decomposition methods: i) CANDECOMP/PARAFAC~(CP) and ii) Tucker decompositions.
For each approach, we establish a rigorous privacy guarantee and meanwhile
evaluate the privacy-accuracy trade-off. Experiments on synthetic and
real-world datasets demonstrate that our proposal achieves high accuracy for
tensor completion while ensuring strong privacy protections.

    

### [[2110.00552] Stochastic Contrastive Learning](http://arxiv.org/abs/2110.00552)


  While state-of-the-art contrastive Self-Supervised Learning (SSL) models
produce results competitive with their supervised counterparts, they lack the
ability to infer latent variables. In contrast, prescribed latent variable (LV)
models enable attributing uncertainty, inducing task specific compression, and
in general allow for more interpretable representations. In this work, we
introduce LV approximations to large scale contrastive SSL models. We
demonstrate that this addition improves downstream performance (resulting in
96.42% and 77.49% test top-1 fine-tuned performance on CIFAR10 and ImageNet
respectively with a ResNet50) as well as producing highly compressed
representations (588x reduction) that are useful for interpretability,
classification and regression downstream tasks.

    

### [[2110.00567] Weight Vector Tuning and Asymptotic Analysis of Binary Linear Classifiers](http://arxiv.org/abs/2110.00567)


  Unlike its intercept, a linear classifier's weight vector cannot be tuned by
a simple grid search. Hence, this paper proposes weight vector tuning of a
generic binary linear classifier through the parameterization of a
decomposition of the discriminant by a scalar which controls the trade-off
between conflicting informative and noisy terms. By varying this parameter, the
original weight vector is modified in a meaningful way. Applying this method to
a number of linear classifiers under a variety of data dimensionality and
sample size settings reveals that the classification performance loss due to
non-optimal native hyperparameters can be compensated for by weight vector
tuning. This yields computational savings as the proposed tuning method reduces
to tuning a scalar compared to tuning the native hyperparameter, which may
involve repeated weight vector generation along with its burden of
optimization, dimensionality reduction, etc., depending on the classifier. It
is also found that weight vector tuning significantly improves the performance
of Linear Discriminant Analysis (LDA) under high estimation noise. Proceeding
from this second finding, an asymptotic study of the misclassification
probability of the parameterized LDA classifier in the growth regime where the
data dimensionality and sample size are comparable is conducted. Using random
matrix theory, the misclassification probability is shown to converge to a
quantity that is a function of the true statistics of the data. Additionally,
an estimator of the misclassification probability is derived. Finally,
computationally efficient tuning of the parameter using this estimator is
demonstrated on real data.

    

### [[2110.00568] Conditional Deep Gaussian Processes: empirical Bayes hyperdata learning](http://arxiv.org/abs/2110.00568)


  It is desirable to combine the expressive power of deep learning with
Gaussian Process (GP) in one expressive Bayesian learning model. Deep kernel
learning proposed in [1] showed success in adopting a deep network for feature
extraction followed by a GP used as function model. Recently, [2] suggested
that the deterministic nature of feature extractor may lead to overfitting
while the replacement with a Bayesian network seemed to cure it. Here, we
propose the conditional Deep Gaussian Process (DGP) in which the intermediate
GPs in hierarchical composition are supported by the hyperdata and the exposed
GP remains zero mean. Motivated by the inducing points in sparse GP, the
hyperdata also play the role of function supports, but are hyperparameters
rather than random variables. We use the moment matching method [3] to
approximate the marginal prior for conditional DGP with a GP carrying an
effective kernel. Thus, as in empirical Bayes, the hyperdata are learned by
optimizing the approximate marginal likelihood which implicitly depends on the
hyperdata via the kernel. We shall show the equivalence with the deep kernel
learning in the limit of dense hyperdata in latent space. However, the
conditional DGP and the corresponding approximate inference enjoy the benefit
of being more Bayesian than deep kernel learning. Preliminary extrapolation
results demonstrate expressive power of the proposed model compared with GP
kernel composition, DGP variational inference, and deep kernel learning. We
also address the non-Gaussian aspect of our model as well as way of upgrading
to a full Bayes inference.

    

### [[2110.00577] Reconstruction for Powerful Graph Representations](http://arxiv.org/abs/2110.00577)


  Graph neural networks (GNNs) have limited expressive power, failing to
represent many graph classes correctly. While more expressive graph
representation learning (GRL) alternatives can distinguish some of these
classes, they are significantly harder to implement, may not scale well, and
have not been shown to outperform well-tuned GNNs in real-world tasks. Thus,
devising simple, scalable, and expressive GRL architectures that also achieve
real-world improvements remains an open challenge. In this work, we show the
extent to which graph reconstruction -- reconstructing a graph from its
subgraphs -- can mitigate the theoretical and practical problems currently
faced by GRL architectures. First, we leverage graph reconstruction to build
two new classes of expressive graph representations. Secondly, we show how
graph reconstruction boosts the expressive power of any GNN architecture while
being a (provably) powerful inductive bias for invariances to vertex removals.
Empirically, we show how reconstruction can boost GNN's expressive power --
while maintaining its invariance to permutations of the vertices -- by solving
seven graph property tasks not solvable by the original GNN. Further, we
demonstrate how it boosts state-of-the-art GNN's performance across nine
real-world benchmark datasets.

    

### [[2110.00578] SMATE: Semi-Supervised Spatio-Temporal Representation Learning on Multivariate Time Series](http://arxiv.org/abs/2110.00578)


  Learning from Multivariate Time Series (MTS) has attracted widespread
attention in recent years. In particular, label shortage is a real challenge
for the classification task on MTS, considering its complex dimensional and
sequential data structure. Unlike self-training and positive unlabeled learning
that rely on distance-based classifiers, in this paper, we propose SMATE, a
novel semi-supervised model for learning the interpretable Spatio-Temporal
representation from weakly labeled MTS. We validate empirically the learned
representation on 22 public datasets from the UEA MTS archive. We compare it
with 13 state-of-the-art baseline methods for fully supervised tasks and four
baselines for semi-supervised tasks. The results show the reliability and
efficiency of our proposed method.

    

### [[1408.6032] PMCE: efficient inference of expressive models of cancer evolution with high prognostic power](http://arxiv.org/abs/1408.6032)


  Motivation: Driver (epi)genomic alterations underlie the positive selection
of cancer subpopulations, which promotes drug resistance and relapse. Even
though substantial heterogeneity is witnessed in most cancer types, mutation
accumulation patterns can be regularly found and can be exploited to
reconstruct predictive models of cancer evolution. Yet, available methods
cannot infer logical formulas connecting events to represent alternative
evolutionary routes or convergent evolution. Results: We introduce PMCE, an
expressive framework that leverages mutational profiles from cross-sectional
sequencing data to infer probabilistic graphical models of cancer evolution
including arbitrary logical formulas, and which outperforms the
state-of-the-art in terms of accuracy and robustness to noise, on simulations.
The application of PMCE to 7866 samples from the TCGA database allows us to
identify a highly significant correlation between the predicted evolutionary
paths and the overall survival in 7 tumor types, proving that our approach can
effectively stratify cancer patients in reliable risk groups. Availability:
PMCE is freely available at this https URL, in addition to
the code to replicate all the analyses presented in the manuscript. Contacts:
daniele.ramazzotti@unimib.it, alex.graudenzi@ibfm.this http URL.

    

### [[2001.07752] Emergence of Pragmatics from Referential Game between Theory of Mind Agents](http://arxiv.org/abs/2001.07752)


  Pragmatics studies how context can contribute to language meanings. In human
communication, language is never interpreted out of context, and sentences can
usually convey more information than their literal meanings. However, this
mechanism is missing in most multi-agent systems, restricting the communication
efficiency and the capability of human-agent interaction. In this paper, we
propose an algorithm, using which agents can spontaneously learn the ability to
"read between lines" without any explicit hand-designed rules. We integrate the
theory of mind (ToM) in a cooperative multi-agent pedagogical situation and
propose an adaptive reinforcement learning (RL) algorithm to develop a
communication protocol. ToM is a profound cognitive science concept, claiming
that people regularly reason about other's mental states, including beliefs,
goals, and intentions, to obtain performance advantage in competition,
cooperation or coalition. With this ability, agents consider language as not
only messages but also rational acts reflecting others' hidden states. Our
experiments demonstrate the advantage of pragmatic protocols over non-pragmatic
protocols. We also show the teaching complexity following the pragmatic
protocol empirically approximates to recursive teaching dimension (RTD).

    

### [[2001.08537] Best Principal Submatrix Selection for the Maximum Entropy Sampling Problem: Scalable Algorithms and Performance Guarantees](http://arxiv.org/abs/2001.08537)


  This paper studies a classic maximum entropy sampling problem (MESP), which
aims to select the most informative principal submatrix of a prespecified size
from a covariance matrix. MESP has been widely applied to many areas, including
healthcare, power system, manufacturing and data science. By investigating its
Lagrangian dual and primal characterization, we derive a novel convex integer
program for MESP and show that its continuous relaxation yields a near-optimal
solution. The results motivate us to study an efficient sampling algorithm and
develop its approximation bound for MESP, which improves the best-known bound
in literature. We then provide an efficient deterministic implementation of the
sampling algorithm with the same approximation bound. By developing new
mathematical tools for the singular matrices and analyzing the Lagrangian dual
of the proposed convex integer program, we investigate the widely-used local
search algorithm and prove its first-known approximation bound for MESP. The
proof techniques further inspire us with an efficient implementation of the
local search algorithm. Our numerical experiments demonstrate that these
approximation algorithms can efficiently solve medium-sized and large-scale
instances to near-optimality. Our proposed algorithms are coded and released as
open-source software. Finally, we extend the analyses to the A-Optimal MESP
(A-MESP), where the objective is to minimize the trace of the inverse of the
selected principal submatrix.

    

### [[2002.02620] Gaussian Variational State Estimation for Nonlinear State-Space Models](http://arxiv.org/abs/2002.02620)


  In this paper, the problem of state estimation, in the context of both
filtering and smoothing, for nonlinear state-space models is considered. Due to
the nonlinear nature of the models, the state estimation problem is generally
intractable as it involves integrals of general nonlinear functions and the
filtered and smoothed state distributions lack closed-form solutions. As such,
it is common to approximate the state estimation problem. In this paper, we
develop an assumed Gaussian solution based on variational inference, which
offers the key advantage of a flexible, but principled, mechanism for
approximating the required distributions. Our main contribution lies in a new
formulation of the state estimation problem as an optimisation problem, which
can then be solved using standard optimisation routines that employ exact
first- and second-order derivatives. The resulting state estimation approach
involves a minimal number of assumptions and applies directly to nonlinear
systems with both Gaussian and non-Gaussian probabilistic models. The
performance of our approach is demonstrated on several examples; a challenging
scalar system, a model of a simple robotic system, and a target tracking
problem using a von Mises-Fisher distribution and outperforms alternative
assumed Gaussian approaches to state estimation.

    

### [[2003.06959] PFPN: Continuous Control of Physically Simulated Characters using Particle Filtering Policy Network](http://arxiv.org/abs/2003.06959)


  Data-driven methods for physics-based character control using reinforcement
learning have been successfully applied to generate high-quality motions.
However, existing approaches typically rely on Gaussian distributions to
represent the action policy, which can prematurely commit to suboptimal actions
when solving high-dimensional continuous control problems for
highly-articulated characters. In this paper, to improve the learning
performance of physics-based character controllers, we propose a framework that
considers a particle-based action policy as a substitute for Gaussian policies.
We exploit particle filtering to dynamically explore and discretize the action
space, and track the posterior policy represented as a mixture distribution.
The resulting policy can replace the unimodal Gaussian policy which has been
the staple for character control problems, without changing the underlying
model architecture of the reinforcement learning algorithm used to perform
policy optimization. We demonstrate the applicability of our approach on
various motion capture imitation tasks. Baselines using our particle-based
policies achieve better imitation performance and speed of convergence as
compared to corresponding implementations using Gaussians, and are more robust
to external perturbations during character control. Related code is available
at: this https URL.

    

### [[2003.07939] Neural Networks for Encoding Dynamic Security-Constrained Optimal Power Flow](http://arxiv.org/abs/2003.07939)


  This paper introduces a framework to capture previously intractable
optimization constraints and transform them to a mixed-integer linear program,
through the use of neural networks. We encode the feasible space of
optimization problems characterized by both tractable and intractable
constraints, e.g. differential equations, to a neural network. Leveraging an
exact mixed-integer reformulation of neural networks, we solve mixed-integer
linear programs that accurately approximate solutions to the originally
intractable non-linear optimization problem. We apply our methods to the AC
optimal power flow problem (AC-OPF), where directly including dynamic security
constraints renders the AC-OPF intractable. Our proposed approach has the
potential to be significantly more scalable than traditional approaches. We
demonstrate our approach for power system operation considering N-1 security
and small-signal stability, showing how it can efficiently obtain cost-optimal
solutions which at the same time satisfy both static and dynamic security
constraints.

    

### [[2004.07740] Really Useful Synthetic Data -- A Framework to Evaluate the Quality of Differentially Private Synthetic Data](http://arxiv.org/abs/2004.07740)


  Recent advances in generating synthetic data that allow to add principled
ways of protecting privacy -- such as Differential Privacy -- are a crucial
step in sharing statistical information in a privacy preserving way. But while
the focus has been on privacy guarantees, the resulting private synthetic data
is only useful if it still carries statistical information from the original
data. To further optimise the inherent trade-off between data privacy and data
quality, it is necessary to think closely about the latter. What is it that
data analysts want? Acknowledging that data quality is a subjective concept, we
develop a framework to evaluate the quality of differentially private synthetic
data from an applied researcher's perspective. Data quality can be measured
along two dimensions. First, quality of synthetic data can be evaluated against
training data or against an underlying population. Second, the quality of
synthetic data depends on general similarity of distributions or specific tasks
such as inference or prediction. It is clear that accommodating all goals at
once is a formidable challenge. We invite the academic community to jointly
advance the privacy-quality frontier.

    

### [[2006.04101] Hybrid Model for Anomaly Detection on Call Detail Records by Time Series Forecasting](http://arxiv.org/abs/2006.04101)


  Mobile network operators store an enormous amount of information like log
files that describe various events and users' activities. Analysis of these
logs might be used in many critical applications such as detecting
cyber-attacks, finding behavioral patterns of users, security incident
response, network forensics, etc. In a cellular network Call Detail Records
(CDR) is one type of such logs containing metadata of calls and usually
includes valuable information about contact such as the phone numbers of
originating and receiving subscribers, call duration, the area of activity,
type of call (SMS or voice call) and a timestamp. With anomaly detection, it is
possible to determine abnormal reduction or increment of network traffic in an
area or for a particular person. This paper's primary goal is to study
subscribers' behavior in a cellular network, mainly predicting the number of
calls in a region and detecting anomalies in the network traffic. In this
paper, a new hybrid method is proposed based on various anomaly detection
methods such as GARCH, K-means, and Neural Network to determine the anomalous
data. Moreover, we have discussed the possible causes of such anomalies.

    

### [[2007.05756] Generative Compositional Augmentations for Scene Graph Prediction](http://arxiv.org/abs/2007.05756)


  Inferring objects and their relationships from an image in the form of a
scene graph is useful in many applications at the intersection of vision and
language. We consider a challenging problem of compositional generalization
that emerges in this task due to a long tail data distribution. Current scene
graph generation models are trained on a tiny fraction of the distribution
corresponding to the most frequent compositions, e.g. <cup, on, table>.
However, test images might contain zero- and few-shot compositions of objects
and relationships, e.g. <cup, on, surfboard>. Despite each of the object
categories and the predicate (e.g. 'on') being frequent in the training data,
the models often fail to properly understand such unseen or rare compositions.
To improve generalization, it is natural to attempt increasing the diversity of
the training distribution. However, in the graph domain this is non-trivial. To
that end, we propose a method to synthesize rare yet plausible scene graphs by
perturbing real ones. We then propose and empirically study a model based on
conditional generative adversarial networks (GANs) that allows us to generate
visual features of perturbed scene graphs and learn from them in a joint
fashion. When evaluated on the Visual Genome dataset, our approach yields
marginal, but consistent improvements in zero- and few-shot metrics. We analyze
the limitations of our approach indicating promising directions for future
research.

    

### [[2007.06731] Regularized linear autoencoders recover the principal components, eventually](http://arxiv.org/abs/2007.06731)


  Our understanding of learning input-output relationships with neural nets has
improved rapidly in recent years, but little is known about the convergence of
the underlying representations, even in the simple case of linear autoencoders
(LAEs). We show that when trained with proper regularization, LAEs can directly
learn the optimal representation -- ordered, axis-aligned principal components.
We analyze two such regularization schemes: non-uniform $\ell_2$ regularization
and a deterministic variant of nested dropout [Rippel et al, ICML' 2014].
Though both regularization schemes converge to the optimal representation, we
show that this convergence is slow due to ill-conditioning that worsens with
increasing latent dimension. We show that the inefficiency of learning the
optimal representation is not inevitable -- we present a simple modification to
the gradient descent update that greatly speeds up convergence empirically.

    

### [[2010.05109] AEGD: Adaptive Gradient Descent with Energy](http://arxiv.org/abs/2010.05109)


  We propose AEGD, a new algorithm for first-order gradient-based optimization
of non-convex objective functions, based on a dynamically updated energy
variable. The method is shown to be unconditionally energy stable, irrespective
of the step size. We prove energy-dependent convergence rates of AEGD for both
non-convex and convex objectives, which for a suitably small step size recovers
desired convergence rates for the batch gradient descent. We also provide an
energy-dependent bound on the stationary convergence of AEGD in the stochastic
non-convex setting. The method is straightforward to implement and requires
little tuning of hyper-parameters. Experimental results demonstrate that AEGD
works well for a large variety of optimization problems: it is robust with
respect to initial data, capable of making rapid initial progress. The
stochastic AEGD shows comparable and often better generalization performance
than SGD with momentum for deep neural networks.

    

### [[2011.03853] A fast randomized incremental gradient method for decentralized non-convex optimization](http://arxiv.org/abs/2011.03853)


  We study decentralized non-convex finite-sum minimization problems described
over a network of nodes, where each node possesses a local batch of data
samples. In this context, we analyze a single-timescale randomized incremental
gradient method, called GT-SAGA. GT-SAGA is computationally efficient as it
evaluates one component gradient per node per iteration and achieves provably
fast and robust performance by leveraging node-level variance reduction and
network-level gradient tracking. For general smooth non-convex problems, we
show the almost sure and mean-squared convergence of GT-SAGA to a first-order
stationary point and further describe regimes of practical significance where
it outperforms the existing approaches and achieves a network
topology-independent iteration complexity respectively. When the global
function satisfies the Polyak-Lojaciewisz condition, we show that GT-SAGA
exhibits linear convergence to an optimal solution in expectation and describe
regimes of practical interest where the performance is network
topology-independent and improves upon the existing methods. Numerical
experiments are included to highlight the main convergence aspects of GT-SAGA
in non-convex settings.

    

### [[2011.12245] Effect of barren plateaus on gradient-free optimization](http://arxiv.org/abs/2011.12245)


  Barren plateau landscapes correspond to gradients that vanish exponentially
in the number of qubits. Such landscapes have been demonstrated for variational
quantum algorithms and quantum neural networks with either deep circuits or
global cost functions. For obvious reasons, it is expected that gradient-based
optimizers will be significantly affected by barren plateaus. However, whether
or not gradient-free optimizers are impacted is a topic of debate, with some
arguing that gradient-free approaches are unaffected by barren plateaus. Here
we show that, indeed, gradient-free optimizers do not solve the barren plateau
problem. Our main result proves that cost function differences, which are the
basis for making decisions in a gradient-free optimization, are exponentially
suppressed in a barren plateau. Hence, without exponential precision,
gradient-free optimizers will not make progress in the optimization. We
numerically confirm this by training in a barren plateau with several
gradient-free optimizers (Nelder-Mead, Powell, and COBYLA algorithms), and show
that the numbers of shots required in the optimization grows exponentially with
the number of qubits.

    

### [[2102.01645] Generating images from caption and vice versa via CLIP-Guided Generative Latent Space Search](http://arxiv.org/abs/2102.01645)


  In this research work we present CLIP-GLaSS, a novel zero-shot framework to
generate an image (or a caption) corresponding to a given caption (or image).
CLIP-GLaSS is based on the CLIP neural network, which, given an image and a
descriptive caption, provides similar embeddings. Differently, CLIP-GLaSS takes
a caption (or an image) as an input, and generates the image (or the caption)
whose CLIP embedding is the most similar to the input one. This optimal image
(or caption) is produced via a generative network, after an exploration by a
genetic algorithm. Promising results are shown, based on the experimentation of
the image Generators BigGAN and StyleGAN2, and of the text Generator GPT2

    

### [[2103.01009] Snowflake: Scaling GNNs to High-Dimensional Continuous Control via Parameter Freezing](http://arxiv.org/abs/2103.01009)


  Recent research has shown that graph neural networks (GNNs) can learn
policies for locomotion control that are as effective as a typical multi-layer
perceptron (MLP), with superior transfer and multi-task performance (Wang et
al., 2018; Huang et al., 2020). Results have so far been limited to training on
small agents, with the performance of GNNs deteriorating rapidly as the number
of sensors and actuators grows. A key motivation for the use of GNNs in the
supervised learning setting is their applicability to large graphs, but this
benefit has not yet been realised for locomotion control. We identify the
weakness with a common GNN architecture that causes this poor scaling:
overfitting in the MLPs within the network that encode, decode, and propagate
messages. To combat this, we introduce Snowflake, a GNN training method for
high-dimensional continuous control that freezes parameters in parts of the
network that suffer from overfitting. Snowflake significantly boosts the
performance of GNNs for locomotion control on large agents, now matching the
performance of MLPs, and with superior transfer properties.

    

### [[2103.05745] Content-Preserving Unpaired Translation from Simulated to Realistic Ultrasound Images](http://arxiv.org/abs/2103.05745)


  Interactive simulation of ultrasound imaging greatly facilitates sonography
training. Although ray-tracing based methods have shown promising results,
obtaining realistic images requires substantial modeling effort and manual
parameter tuning. In addition, current techniques still result in a significant
appearance gap between simulated images and real clinical scans. Herein we
introduce a novel content-preserving image translation framework (ConPres) to
bridge this appearance gap, while maintaining the simulated anatomical layout.
We achieve this goal by leveraging both simulated images with semantic
segmentations and unpaired in-vivo ultrasound scans. Our framework is based on
recent contrastive unpaired translation techniques and we propose a
regularization approach by learning an auxiliary segmentation-to-real image
translation task, which encourages the disentanglement of content and style. In
addition, we extend the generator to be class-conditional, which enables the
incorporation of additional losses, in particular a cyclic consistency loss, to
further improve the translation quality. Qualitative and quantitative
comparisons against state-of-the-art unpaired translation methods demonstrate
the superiority of our proposed framework.

    

### [[2103.09404] Collapsible Linear Blocks for Super-Efficient Super Resolution](http://arxiv.org/abs/2103.09404)


  With the advent of smart devices that support 4K and 8K resolution, Single
Image Super Resolution (SISR) has become an important computer vision problem.
However, most super resolution deep networks are computationally very
expensive. In this paper, we propose SESR, a new class of Super-Efficient Super
Resolution networks that significantly improve image quality and reduce
computational complexity. Detailed experiments across six benchmark datasets
demonstrate that SESR achieves similar or better image quality than
state-of-the-art models while requiring 2x to 330x fewer Multiply-Accumulate
(MAC) operations. As a result, SESR can be used on constrained hardware to
perform x2 (1080p to 4K) and x4 SISR (1080p to 8K). Towards this, we simulate
hardware performance numbers for a commercial mobile Neural Processing Unit
(NPU) for 1080p to 4K (x2) and 1080p to 8K (x4) SISR. Our results highlight the
challenges faced by super resolution on AI accelerators and demonstrate that
SESR is significantly faster than existing models. Overall, SESR establishes a
new Pareto frontier on the quality (PSNR)-computation relationship for the
super resolution task. The code for this work is available at
this https URL.

    

### [[2103.14187] Beyond Low-Pass Filters: Adaptive Feature Propagation on Graphs](http://arxiv.org/abs/2103.14187)


  Graph neural networks (GNNs) have been extensively studied for prediction
tasks on graphs. As pointed out by recent studies, most GNNs assume local
homophily, i.e., strong similarities in local neighborhoods. This assumption
however limits the generalizability power of GNNs. To address this limitation,
we propose a flexible GNN model, which is capable of handling any graphs
without being restricted by their underlying homophily. At its core, this model
adopts a node attention mechanism based on multiple learnable spectral filters;
therefore, the aggregation scheme is learned adaptively for each graph in the
spectral domain. We evaluated the proposed model on node classification tasks
over eight benchmark datasets. The proposed model is shown to generalize well
to both homophilic and heterophilic graphs. Further, it outperforms all
state-of-the-art baselines on heterophilic graphs and performs comparably with
them on homophilic graphs.

    

### [[2103.14797] Unsupervised Self-Training for Sentiment Analysis of Code-Switched Data](http://arxiv.org/abs/2103.14797)


  Sentiment analysis is an important task in understanding social media content
like customer reviews, Twitter and Facebook feeds etc. In multilingual
communities around the world, a large amount of social media text is
characterized by the presence of Code-Switching. Thus, it has become important
to build models that can handle code-switched data. However, annotated
code-switched data is scarce and there is a need for unsupervised models and
algorithms. We propose a general framework called Unsupervised Self-Training
and show its applications for the specific use case of sentiment analysis of
code-switched data. We use the power of pre-trained BERT models for
initialization and fine-tune them in an unsupervised manner, only using pseudo
labels produced by zero-shot transfer. We test our algorithm on multiple
code-switched languages and provide a detailed analysis of the learning
dynamics of the algorithm with the aim of answering the question - `Does our
unsupervised model understand the Code-Switched languages or does it just learn
its representations?'. Our unsupervised models compete well with their
supervised counterparts, with their performance reaching within 1-7\% (weighted
F1 scores) when compared to supervised models trained for a two class problem.

    

### [[2104.04179] X2CT-FLOW: Maximum a posteriori reconstruction using a progressive flow-based deep generative model for ultra sparse-view computed tomography in ultra low-dose protocols](http://arxiv.org/abs/2104.04179)


  Ultra sparse-view computed tomography (CT) algorithms can reduce radiation
exposure of patients, but those algorithms lack an explicit cycle consistency
loss minimization and an explicit log-likelihood maximization in testing. Here,
we propose X2CT-FLOW for the maximum a posteriori (MAP) reconstruction of a
three-dimensional (3D) chest CT image from a single or a few two-dimensional
(2D) projection images using a progressive flow-based deep generative model,
especially for ultra low-dose protocols. The MAP reconstruction can
simultaneously optimize the cycle consistency loss and the log-likelihood. The
proposed algorithm is built upon a newly developed progressive flow-based deep
generative model, which is featured with exact log-likelihood estimation,
efficient sampling, and progressive learning. We applied X2CT-FLOW to
reconstruction of 3D chest CT images from biplanar projection images without
noise contamination (assuming a standard-dose protocol) and with strong noise
contamination (assuming an ultra low-dose protocol). With the standard-dose
protocol, our images reconstructed from 2D projected images and 3D ground-truth
CT images showed good agreement in terms of structural similarity (SSIM, 0.7675
on average), peak signal-to-noise ratio (PSNR, 25.89 dB on average), mean
absolute error (MAE, 0.02364 on average), and normalized root mean square error
(NRMSE, 0.05731 on average). Moreover, with the ultra low-dose protocol, our
images reconstructed from 2D projected images and the 3D ground-truth CT images
also showed good agreement in terms of SSIM (0.7008 on average), PSNR (23.58 dB
on average), MAE (0.02991 on average), and NRMSE (0.07349 on average).

    

### [[2104.04483] Inverse Reinforcement Learning: A Control Lyapunov Approach](http://arxiv.org/abs/2104.04483)


  Inferring the intent of an intelligent agent from demonstrations and
subsequently predicting its behavior, is a critical task in many collaborative
settings. A common approach to solve this problem is the framework of inverse
reinforcement learning (IRL), where the observed agent, e.g., a human
demonstrator, is assumed to behave according to an intrinsic cost function that
reflects its intent and informs its control actions. In this work, we
reformulate the IRL inference problem to learning control Lyapunov functions
(CLF) from demonstrations by exploiting the inverse optimality property, which
states that every CLF is also a meaningful value function. Moreover, the
derived CLF formulation directly guarantees stability of inferred control
policies. We show the flexibility of our proposed method by learning from
goal-directed movement demonstrations in a continuous environment.

    

### [[2104.08835] CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in NLP](http://arxiv.org/abs/2104.08835)


  Humans can learn a new language task efficiently with only few examples, by
leveraging their knowledge obtained when learning prior tasks. In this paper,
we explore whether and how such cross-task generalization ability can be
acquired, and further applied to build better few-shot learners across diverse
NLP tasks. We introduce CrossFit, a problem setup for studying cross-task
generalization ability, which standardizes seen/unseen task partitions, data
access during different learning stages, and the evaluation protocols. To
instantiate different seen/unseen task partitions in CrossFit and facilitate
in-depth analysis, we present the NLP Few-shot Gym, a repository of 160 diverse
few-shot NLP tasks created from open-access NLP datasets and converted to a
unified text-to-text format. Our analysis reveals that the few-shot learning
ability on unseen tasks can be improved via an upstream learning stage using a
set of seen tasks. We also observe that the selection of upstream learning
tasks can significantly influence few-shot performance on unseen tasks, asking
further analysis on task similarity and transferability.

    

### [[2104.08840] On the Influence of Masking Policies in Intermediate Pre-training](http://arxiv.org/abs/2104.08840)


  Current NLP models are predominantly trained through a two-stage "pre-train
then fine-tune" pipeline. Prior work has shown that inserting an intermediate
pre-training stage, using heuristic masking policies for masked language
modeling (MLM), can significantly improve final performance. However, it is
still unclear (1) in what cases such intermediate pre-training is helpful, (2)
whether hand-crafted heuristic objectives are optimal for a given task, and (3)
whether a masking policy designed for one task is generalizable beyond that
task. In this paper, we perform a large-scale empirical study to investigate
the effect of various masking policies in intermediate pre-training with nine
selected tasks across three categories. Crucially, we introduce methods to
automate the discovery of optimal masking policies via direct supervision or
meta-learning. We conclude that the success of intermediate pre-training is
dependent on appropriate pre-train corpus, selection of output format (i.e.,
masked spans or full sentence), and clear understanding of the role that MLM
plays for the downstream task. In addition, we find our learned masking
policies outperform the heuristic of masking named entities on TriviaQA, and
policies learned from one task can positively transfer to other tasks in
certain cases, inviting future research in this direction.

    

### [[2104.08942] Attention-based Clinical Note Summarization](http://arxiv.org/abs/2104.08942)


  The trend of deploying digital systems in numerous industries has induced a
hike in recording digital information. The health sector has observed an
extensive adoption of digital devices and systems that generate large volumes
of personal medical records. Electronic health records contain valuable
information for retrospective and prospective analysis that is often not
entirely exploited because of the dense information storage. The crude purpose
of condensing health records is to select the information that holds most
characteristics of the original documents based on reported disease. These
summaries may boost diagnosis and extend a doctor's time with the patient
during a high workload situation like the COVID-19 pandemic. In this paper, we
propose applying a multi-head attention-based mechanism to perform extractive
summarization of meaningful phrases in clinical notes. This method finds major
sentences for a summary by correlating tokens, segments, and positional
embeddings. The model outputs attention scores that are statistically
transformed to extract key phrases and can be used to projection on the
heat-mapping tool for visual and human use.

    

### [[2105.03842] FastCorrect: Fast Error Correction with Edit Alignment for Automatic Speech Recognition](http://arxiv.org/abs/2105.03842)


  Error correction techniques have been used to refine the output sentences
from automatic speech recognition (ASR) models and achieve a lower word error
rate (WER) than original ASR outputs. Previous works usually use a
sequence-to-sequence model to correct an ASR output sentence autoregressively,
which causes large latency and cannot be deployed in online ASR services. A
straightforward solution to reduce latency, inspired by non-autoregressive
(NAR) neural machine translation, is to use an NAR sequence generation model
for ASR error correction, which, however, comes at the cost of significantly
increased ASR error rate. In this paper, observing distinctive error patterns
and correction operations (i.e., insertion, deletion, and substitution) in ASR,
we propose FastCorrect, a novel NAR error correction model based on edit
alignment. In training, FastCorrect aligns each source token from an ASR output
sentence to the target tokens from the corresponding ground-truth sentence
based on the edit distance between the source and target sentences, and
extracts the number of target tokens corresponding to each source token during
edition/correction, which is then used to train a length predictor and to
adjust the source tokens to match the length of the target sentence for
parallel generation. In inference, the token number predicted by the length
predictor is used to adjust the source tokens for target sequence generation.
Experiments on the public AISHELL-1 dataset and an internal industrial-scale
ASR dataset show the effectiveness of FastCorrect for ASR error correction: 1)
it speeds up the inference by 6-9 times and maintains the accuracy (8-14% WER
reduction) compared with the autoregressive correction model; and 2) it
outperforms the popular NAR models adopted in neural machine translation and
text edition by a large margin.

    

### [[2105.08232] Sharp Restricted Isometry Property Bounds for Low-rank Matrix Recovery Problems with Corrupted Measurements](http://arxiv.org/abs/2105.08232)


  In this paper, we study a general low-rank matrix recovery problem with
linear measurements corrupted by some noise. The objective is to understand
under what conditions on the restricted isometry property (RIP) of the problem
local search methods can find the ground truth with a small error. By analyzing
the landscape of the non-convex problem, we first propose a global guarantee on
the maximum distance between an arbitrary local minimizer and the ground truth
under the assumption that the RIP constant is smaller than $1/2$. We show that
this distance shrinks to zero as the intensity of the noise reduces. Our new
guarantee is sharp in terms of the RIP constant and is much stronger than the
existing results. We then present a local guarantee for problems with an
arbitrary RIP constant, which states that any local minimizer is either
considerably close to the ground truth or far away from it. Next, we prove the
strict saddle property, which guarantees the global convergence of the
perturbed gradient descent method in polynomial time. The developed results
demonstrate how the noise intensity and the RIP constant of the problem affect
the landscape of the problem.

    

### [[2105.13052] A generalization of the randomized singular value decomposition](http://arxiv.org/abs/2105.13052)


  The randomized singular value decomposition (SVD) is a popular and effective
algorithm for computing a near-best rank $k$ approximation of a matrix $A$
using matrix-vector products with standard Gaussian vectors. Here, we
generalize the theory of randomized SVD to multivariate Gaussian vectors,
allowing one to incorporate prior knowledge of $A$ into the algorithm. This
enables us to explore the continuous analogue of the randomized SVD for
Hilbert--Schmidt (HS) operators using operator-function products with functions
drawn from a Gaussian process (GP). We then construct a new covariance kernel
for GPs, based on weighted Jacobi polynomials, which allows us to rapidly
sample the GP and control the smoothness of the randomly generated functions.
Numerical examples on matrices and HS operators demonstrate the applicability
of the algorithm.

    

### [[2106.00092] Generalized AdaGrad (G-AdaGrad) and Adam: A State-Space Perspective](http://arxiv.org/abs/2106.00092)


  Accelerated gradient-based methods are being extensively used for solving
non-convex machine learning problems, especially when the data points are
abundant or the available data is distributed across several agents. Two of the
prominent accelerated gradient algorithms are AdaGrad and Adam. AdaGrad is the
simplest accelerated gradient method, which is particularly effective for
sparse data. Adam has been shown to perform favorably in deep learning problems
compared to other methods. In this paper, we propose a new fast optimizer,
Generalized AdaGrad (G-AdaGrad), for accelerating the solution of potentially
non-convex machine learning problems. Specifically, we adopt a state-space
perspective for analyzing the convergence of gradient acceleration algorithms,
namely G-AdaGrad and Adam, in machine learning. Our proposed state-space models
are governed by ordinary differential equations. We present simple convergence
proofs of these two algorithms in the deterministic settings with minimal
assumptions. Our analysis also provides intuition behind improving upon
AdaGrad's convergence rate. We provide empirical results on MNIST dataset to
reinforce our claims on the convergence and performance of G-AdaGrad and Adam.

    

### [[2106.00393] Learning Representations for Sub-Symbolic Reasoning](http://arxiv.org/abs/2106.00393)


  Neuro-symbolic methods integrate neural architectures, knowledge
representation and reasoning. However, they have been struggling at both
dealing with the intrinsic uncertainty of the observations and scaling to real
world applications. This paper presents Relational Reasoning Networks (R2N), a
novel end-to-end model that performs relational reasoning in the latent space
of a deep learner architecture, where the representations of constants, ground
atoms and their manipulations are learned in an integrated fashion. Unlike flat
architectures like Knowledge Graph Embedders, which can only represent
relations between entities, R2Ns define an additional computational structure,
accounting for higher-level relations among the ground atoms. The considered
relations can be explicitly known, like the ones defined by logic formulas, or
defined as unconstrained correlations among groups of ground atoms. R2Ns can be
applied to purely symbolic tasks or as a neuro-symbolic platform to integrate
learning and reasoning in heterogeneous problems with both symbolic and
feature-based represented entities. The proposed model bridges the gap between
previous neuro-symbolic methods that have been either limited in terms of
scalability or expressivity. The proposed methodology is shown to achieve
state-of-the-art results in different experimental settings.

    

### [[2106.03904] When in Doubt: Neural Non-Parametric Uncertainty Quantification for Epidemic Forecasting](http://arxiv.org/abs/2106.03904)


  Accurate and trustworthy epidemic forecasting is an important problem that
has impact on public health planning and disease mitigation. Most existing
epidemic forecasting models disregard uncertainty quantification, resulting in
mis-calibrated predictions. Recent works in deep neural models for
uncertainty-aware time-series forecasting also have several limitations; e.g.
it is difficult to specify meaningful priors in Bayesian NNs, while methods
like deep ensembling are computationally expensive in practice. In this paper,
we fill this important gap. We model the forecasting task as a probabilistic
generative process and propose a functional neural process model called EPIFNP,
which directly models the probability density of the forecast value. EPIFNP
leverages a dynamic stochastic correlation graph to model the correlations
between sequences in a non-parametric way, and designs different stochastic
latent variables to capture functional uncertainty from different perspectives.
Our extensive experiments in a real-time flu forecasting setting show that
EPIFNP significantly outperforms previous state-of-the-art models in both
accuracy and calibration metrics, up to 2.5x in accuracy and 2.4x in
calibration. Additionally, due to properties of its generative process,EPIFNP
learns the relations between the current season and similar patterns of
historical seasons,enabling interpretable forecasts. Beyond epidemic
forecasting, the EPIFNP can be of independent interest for advancing principled
uncertainty quantification in deep sequential models for predictive analytics

    

### [[2106.09385] On Deep Neural Network Calibration by Regularization and its Impact on Refinement](http://arxiv.org/abs/2106.09385)


  Deep neural networks have been shown to be highly miscalibrated. often they
tend to be overconfident in their predictions. It poses a significant challenge
for safety-critical systems to utilise deep neural networks (DNNs), reliably.
Many recently proposed approaches to mitigate this have demonstrated
substantial progress in improving DNN calibration. However, they hardly touch
upon refinement, which historically has been an essential aspect of
calibration. Refinement indicates separability of a network's correct and
incorrect predictions. This paper presents a theoretically and empirically
supported exposition reviewing refinement of a calibrated model. Firstly, we
show the breakdown of expected calibration error (ECE), into predicted
confidence and refinement under the assumption of over-confident predictions.
Secondly, linking with this result, we highlight that regularization based
calibration only focuses on naively reducing a model's confidence. This
logically has a severe downside to a model's refinement as correct and
incorrect predictions become tightly coupled. Lastly, connecting refinement
with ECE also provides support to existing refinement based approaches which
improve calibration but do not explain the reasoning behind it. We support our
claims through rigorous empirical evaluations of many state of the art
calibration approaches on widely used datasets and neural networks. We find
that many calibration approaches with the likes of label smoothing, mixup etc.
lower the usefulness of a DNN by degrading its refinement. Even under natural
data shift, this calibration-refinement trade-off holds for the majority of
calibration methods.

    

### [[2107.00055] Approximate Regions of Attraction in Learning with Decision-Dependent Distributions](http://arxiv.org/abs/2107.00055)


  As data-driven methods are deployed in real-world settings, the processes
that generate the observed data will often react to the decisions of the
learner. For example, a data source may have some incentive for the algorithm
to provide a particular label (e.g. approve a bank loan), and manipulate their
features accordingly. Work in strategic classification and decision-dependent
distributions seeks to characterize the closed-loop behavior of deploying
learning algorithms by explicitly considering the effect of the classifier on
the underlying data distribution. More recently, works in performative
prediction seek to classify the closed-loop behavior by considering general
properties of the mapping from classifier to data distribution, rather than an
explicit form. Building on this notion, we analyze repeated risk minimization
as the perturbed trajectories of the gradient flows of performative risk
minimization. We consider the case where there may be multiple local minimizers
of performative risk, motivated by situations where the initial conditions may
have significant impact on the long-term behavior of the system. We provide
sufficient conditions to characterize the region of attraction for the various
equilibria in this settings. Additionally, we introduce the notion of
performative alignment, which provides a geometric condition on the convergence
of repeated risk minimization to performative risk minimizers.

    

### [[2109.13226] BigSSL: Exploring the Frontier of Large-Scale Semi-Supervised Learning for Automatic Speech Recognition](http://arxiv.org/abs/2109.13226)


  We summarize the results of a host of efforts using giant automatic speech
recognition (ASR) models pre-trained using large, diverse unlabeled datasets
containing approximately a million hours of audio. We find that the combination
of pre-training, self-training and scaling up model size greatly increases data
efficiency, even for extremely large tasks with tens of thousands of hours of
labeled data. In particular, on an ASR task with 34k hours of labeled data, by
fine-tuning an 8 billion parameter pre-trained Conformer model we can match
state-of-the-art (SoTA) performance with only 3% of the training data and
significantly improve SoTA with the full training set. We also report on the
universal benefits gained from using big pre-trained and self-trained models
for a large set of downstream tasks that cover a wide range of speech domains
and span multiple orders of magnitudes of dataset sizes, including obtaining
SoTA performance on many public benchmarks. In addition, we utilize the learned
representation of pre-trained networks to achieve SoTA results on non-ASR
tasks.

    

### [[2103.00948] Cross Modal Focal Loss for RGBD Face Anti-Spoofing](http://arxiv.org/abs/2103.00948)


  Automatic methods for detecting presentation attacks are essential to ensure
the reliable use of facial recognition technology. Most of the methods
available in the literature for presentation attack detection (PAD) fails in
generalizing to unseen attacks. In recent years, multi-channel methods have
been proposed to improve the robustness of PAD systems. Often, only a limited
amount of data is available for additional channels, which limits the
effectiveness of these methods. In this work, we present a new framework for
PAD that uses RGB and depth channels together with a novel loss function. The
new architecture uses complementary information from the two modalities while
reducing the impact of overfitting. Essentially, a cross-modal focal loss
function is proposed to modulate the loss contribution of each channel as a
function of the confidence of individual channels. Extensive evaluations in two
publicly available datasets demonstrate the effectiveness of the proposed
approach.

    

### [[2109.14860] Physics and Equality Constrained Artificial Neural Networks: Application to Partial Differential Equations](http://arxiv.org/abs/2109.14860)


  Physics-informed neural networks (PINNs) have been proposed to learn the
solution of partial differential equations (PDE). In PINNs, the residual form
of the PDE of interest and its boundary conditions are lumped into a composite
objective function as an unconstrained optimization problem, which is then used
to train a deep feed-forward neural network. Here, we show that this specific
way of formulating the objective function is the source of severe limitations
in the PINN approach when applied to different kinds of PDEs. To address these
limitations, we propose a versatile framework that can tackle both inverse and
forward problems. The framework is adept at multi-fidelity data fusion and can
seamlessly constrain the governing physics equations with proper initial and
boundary conditions. The backbone of the proposed framework is a nonlinear,
equality-constrained optimization problem formulation aimed at minimizing a
loss functional, where an augmented Lagrangian method (ALM) is used to formally
convert a constrained-optimization problem into an unconstrained-optimization
problem. We implement the ALM within a stochastic, gradient-descent type
training algorithm in a way that scrupulously focuses on meeting the
constraints without sacrificing other loss terms. Additionally, as a
modification of the original residual layers, we propose lean residual layers
in our neural network architecture to address the so-called vanishing-gradient
problem. We demonstrate the efficacy and versatility of our physics- and
equality-constrained deep-learning framework by applying it to learn the
solutions of various multi-dimensional PDEs, including a nonlinear inverse
problem from the hydrology field with multi-fidelity data fusion. The results
produced with our proposed model match exact solutions very closely for all the
cases considered.

    

### [[2110.00232] Enhanced Multigradient Dilution Preparation](http://arxiv.org/abs/2110.00232)


  Abstract: In our paper the new algorithm enhanced multi gradient Dilution
Preparation (EMDP) is discussed. This new algorithm is reported with a lab on
chip or digital Microfluidic biochip to operate multiple operation on a tiny
chip. We can use Digital Microfluidic biochip to operate multiple operation on
a tiny chip. Samples are very costly which are used in any Biochemical
laboratory Protocols. For the case of fast and high throughput application, It
is essential to minimize the cost of operations and the time of operations and
that is why one of the most challenging and important phase is sample
preparation. In our proposed algorithm, we have hide to reduce sample droplets
and waste droplets and for this purpose waste recycling is used, when different
series of multi gradient targets concentration factors (CFS) are generated. We
have compared our proposed algorithm with recent dilution techniques such as
MTC, REMIA, and WARA. For the storage of intermediate droplets which, and
generated during this process, on chip storage space 0(n) is needed. Key words:
Digital microfluidic Biochip, Drug discovery, sample preparation, Electro
wetting.

    

### [[2110.00145] Towards Generalised Half-Duplex Systems](http://arxiv.org/abs/2110.00145)


  FIFO automata are finite state machines communicating through FIFO queues.
They can be used for instance to model distributed protocols. Due to the
unboundedness of the FIFO queues, several verification problems are undecidable
for these systems. In order to model-check such systems, one may look for
decidable subclasses of FIFO systems. Binary half-duplex systems are systems of
two FIFO automata exchanging over a half-duplex channel. They were studied by
Cece and Finkel who established the decidability in polynomial time of several
properties. These authors also identified some problems in generalising
half-duplex systems to multi-party communications. We introduce greedy systems,
as a candidate to generalise binary half-duplex systems. We show that greedy
systems retain the same good properties as binary half-duplex systems, and
that, in the setting of mailbox communications, greedy systems are quite
closely related to a multiparty generalisation of half-duplex systems.

    

### [[2110.00511] ASH: A Modern Framework for Parallel Spatial Hashing in 3D Perception](http://arxiv.org/abs/2110.00511)


  We present ASH, a modern and high-performance framework for parallel spatial
hashing on GPU. Compared to existing GPU hash map implementations, ASH achieves
higher performance, supports richer functionality, and requires fewer lines of
code (LoC) when used for implementing spatially varying operations from
volumetric geometry reconstruction to differentiable appearance reconstruction.
Unlike existing GPU hash maps, the ASH framework provides a versatile tensor
interface, hiding low-level details from the users. In addition, by decoupling
the internal hashing data structures and key-value data in buffers, we offer
direct access to spatially varying data via indices, enabling seamless
integration to modern libraries such as PyTorch. To achieve this, we 1) detach
stored key-value data from the low-level hash map implementation; 2) bridge the
pointer-first low level data structures to index-first high-level tensor
interfaces via an index heap; 3) adapt both generic and non-generic
integer-only hash map implementations as backends to operate on
multi-dimensional keys. We first profile our hash map against state-of-the-art
hash maps on synthetic data to show the performance gain from this
architecture. We then show that ASH can consistently achieve higher performance
on various large-scale 3D perception tasks with fewer LoC by showcasing several
applications, including 1) point cloud voxelization, 2) dense volumetric SLAM,
3) non-rigid point cloud registration and volumetric deformation, and 4)
spatially varying geometry and appearance refinement. ASH and its example
applications are open sourced in Open3D (this http URL).

    

### [[1909.00083] A Distributed Algorithm for High-Dimension Convex Quadratically Constrained Quadratic Programs](http://arxiv.org/abs/1909.00083)


  We propose a Jacobi-style distributed algorithm to solve convex,
quadratically constrained quadratic programs (QCQPs), which arise from a broad
range of applications. While small to medium-sized convex QCQPs can be solved
efficiently by interior-point algorithms, large-scale problems pose significant
challenges to traditional algorithms that are mainly designed to be implemented
on a single computing unit. The exploding volume of data (and hence, the
problem size), however, may overwhelm any such units. In this paper, we propose
a distributed algorithm for general, non-separable, large-scale convex QCQPs,
using a novel idea of predictor-corrector primal-dual update with an adaptive
step size. The algorithm enables distributed storage of data as well as
parallel distributed computing. We establish the conditions for the proposed
algorithm to converge to a global optimum, and implement our algorithm on a
computer cluster with multiple nodes using Message Passing Interface (MPI). The
numerical experiments are conducted on data sets of various scales from
different applications, and the results show that our algorithm exhibits
favorable scalability for solving large-scale problems.

    

### [[2110.00096] Decentralized Graph-Based Multi-Agent Reinforcement Learning Using Reward Machines](http://arxiv.org/abs/2110.00096)


  In multi-agent reinforcement learning (MARL), it is challenging for a
collection of agents to learn complex temporally extended tasks. The
difficulties lie in computational complexity and how to learn the high-level
ideas behind reward functions. We study the graph-based Markov Decision Process
(MDP) where the dynamics of neighboring agents are coupled. We use a reward
machine (RM) to encode each agent's task and expose reward function internal
structures. RM has the capacity to describe high-level knowledge and encode
non-Markovian reward functions. We propose a decentralized learning algorithm
to tackle computational complexity, called decentralized graph-based
reinforcement learning using reward machines (DGRM), that equips each agent
with a localized policy, allowing agents to make decisions independently, based
on the information available to the agents. DGRM uses the actor-critic
structure, and we introduce the tabular Q-function for discrete state problems.
We show that the dependency of Q-function on other agents decreases
exponentially as the distance between them increases. Furthermore, the
complexity of DGRM is related to the local information size of the largest
$\kappa$-hop neighborhood, and DGRM can find an
$O(\rho^{\kappa+1})$-approximation of a stationary point of the objective
function. To further improve efficiency, we also propose the deep DGRM
algorithm, using deep neural networks to approximate the Q-function and policy
function to solve large-scale or continuous state problems. The effectiveness
of the proposed DGRM algorithm is evaluated by two case studies, UAV package
delivery and COVID-19 pandemic mitigation. Experimental results show that local
information is sufficient for DGRM and agents can accomplish complex tasks with
the help of RM. DGRM improves the global accumulated reward by 119% compared to
the baseline in the case of COVID-19 pandemic mitigation.

    

### [[2110.00121] Emergence of Theory of Mind Collaboration in Multiagent Systems](http://arxiv.org/abs/2110.00121)


  Currently, in the study of multiagent systems, the intentions of agents are
usually ignored. Nonetheless, as pointed out by Theory of Mind (ToM), people
regularly reason about other's mental states, including beliefs, goals, and
intentions, to obtain performance advantage in competition, cooperation or
coalition. However, due to its intrinsic recursion and intractable modeling of
distribution over belief, integrating ToM in multiagent planning and decision
making is still a challenge. In this paper, we incorporate ToM in multiagent
partially observable Markov decision process (POMDP) and propose an adaptive
training algorithm to develop effective collaboration between agents with ToM.
We evaluate our algorithms with two games, where our algorithm surpasses all
previous decentralized execution algorithms without modeling ToM.

    

### [[2110.00125] MemBERT: Injecting Unstructured Knowledge into BERT](http://arxiv.org/abs/2110.00125)


  Transformers changed modern NLP in many ways. However, they can hardly
exploit domain knowledge, and like other blackbox models, they lack
interpretability. Unfortunately, structured knowledge injection, in the long
run, risks to suffer from a knowledge acquisition bottleneck. We thus propose a
memory enhancement of transformer models that makes use of unstructured domain
knowledge expressed in plain natural language. An experimental evaluation
conducted on two challenging NLP tasks demonstrates that our approach yields
better performance and model interpretability than baseline transformer-based
architectures.

    

### [[2110.00244] Lightweight Transformer in Federated Setting for Human Activity Recognition](http://arxiv.org/abs/2110.00244)


  Human Activity Recognition (HAR) has been a challenging problem yet it needs
to be solved. It will mainly be used for eldercare and healthcare as an
assistive technology when ensemble with other technologies like Internet of
Things(IoT). HAR can be achieved with the help of sensors, smartphones or
images. Deep neural network techniques like artificial neural networks,
convolutional neural networks and recurrent neural networks have been used in
HAR, both in centralized and federated setting. However, these techniques have
certain limitations. RNNs have limitation of parallelization, CNNS have the
limitation of sequence length and they are computationally expensive. In this
paper, to address the state of art challenges, we present a inertial
sensors-based novel one patch transformer which gives the best of both RNNs and
CNNs for Human activity recognition. We also design a testbed to collect
real-time human activity data. The data collected is further used to train and
test the proposed transformer. With the help of experiments, we show that the
proposed transformer outperforms the state of art CNN and RNN based
classifiers, both in federated and centralized setting. Moreover, the proposed
transformer is computationally inexpensive as it uses very few parameter
compared to the existing state of art CNN and RNN based classifier. Thus its
more suitable for federated learning as it provides less communication and
computational cost.

    

### [[2110.00247] Learner to learner fuzzy profiles similarity using a hybrid interaction analysis grid](http://arxiv.org/abs/2110.00247)


  The analysis of remote discussions is not yet at the same level as the
face-to-face ones. The present paper aspires twofold. On the one hand, it
attempts to establish a suitable environment of interaction and collaboration
among learners by using the speech acts via a semi structured synchronous
communication tool. On the other, it aims to define behavioral profiles and
interpersonal skills hybrid grid by matching the BALES' IPA and PLETY's
analysis system. By applying the fuzzy logic, we formalize human reasoning and,
thus, giving very appreciable flexibility to the reasoning that use it, which
makes it possible to take into account imprecisions and uncertainties. In
addition, the educational data mining techniques are used to optimize the
mapping of behaviors to learner's profile, with similarity-based clustering,
using Eros and PCA measures. In order to show the validity of our system, we
performed an experiment on real-world data. The results show, among others: (1)
the usefulness of fuzzy logic to properly translate the profile text
descriptions into a mathematical format, (2) an irregularity in the behavior of
the learners, (3) the correlation between the profiles, (4) the superiority of
Eros method to the PCA factor in precision.

    

### [[2110.00267] Inductive Representation Learning in Temporal Networks via Mining Neighborhood and Community Influences](http://arxiv.org/abs/2110.00267)


  Network representation learning aims to generate an embedding for each node
in a network, which facilitates downstream machine learning tasks such as node
classification and link prediction. Current work mainly focuses on transductive
network representation learning, i.e. generating fixed node embeddings, which
is not suitable for real-world applications. Therefore, we propose a new
inductive network representation learning method called MNCI by mining
neighborhood and community influences in temporal networks. We propose an
aggregator function that integrates neighborhood influence with community
influence to generate node embeddings at any time. We conduct extensive
experiments on several real-world datasets and compare MNCI with several
state-of-the-art baseline methods on various tasks, including node
classification and network visualization. The experimental results show that
MNCI achieves better performance than baselines.

    

### [[2110.00269] A Survey of Knowledge Enhanced Pre-trained Models](http://arxiv.org/abs/2110.00269)


  Pre-trained models learn contextualized word representations on large-scale
text corpus through a self-supervised learning method, which has achieved
promising performance after fine-tuning. These models, however, suffer from
poor robustness and lack of interpretability. Pre-trained models with knowledge
injection, which we call knowledge enhanced pre-trained models (KEPTMs),
possess deep understanding and logical reasoning and introduce interpretability
to some extent. In this survey, we provide a comprehensive overview of KEPTMs
for natural language processing. We first introduce the progress of pre-trained
models and knowledge representation learning. Then we systematically categorize
existing KEPTMs from three different perspectives. Finally, we outline some
potential directions of KEPTMs for future research.

    

### [[2110.00273] From SLAM to Situational Awareness: Challenges and Survey](http://arxiv.org/abs/2110.00273)


  The knowledge that an intelligent and autonomous mobile robot has and is able
to acquire of itself and the environment, namely the situation, limits its
reasoning, decision-making, and execution skills to efficiently and safely
perform complex missions. Situational awareness is a basic capability of humans
that has been deeply studied in fields like Psychology, Military, Aerospace,
Education, etc., but it has barely been considered in robotics, which has
focused on ideas such as sensing, perception, sensor fusion, state estimation,
localization and mapping, spatial AI, etc. In our research, we connected the
broad multidisciplinary existing knowledge on situational awareness with its
counterpart in mobile robotics. In this paper, we survey the state-of-the-art
robotics algorithms, we analyze the situational awareness aspects that have
been covered by them, and we discuss their missing points. We found out that
the existing robotics algorithms are still missing manifold important aspects
of situational awareness. As a consequence, we conclude that these missing
features are limiting the performance of robotic situational awareness, and
further research is needed to overcome this challenge. We see this as an
opportunity, and provide our vision for future research on robotic situational
awareness.

    

### [[2110.00423] A Web Scale Entity Extraction System](http://arxiv.org/abs/2110.00423)


  Understanding the semantic meaning of content on the web through the lens of
entities and concepts has many practical advantages. However, when building
large-scale entity extraction systems, practitioners are facing unique
challenges involving finding the best ways to leverage the scale and variety of
data available on internet platforms. We present learnings from our efforts in
building an entity extraction system for multiple document types at large scale
using multi-modal Transformers. We empirically demonstrate the effectiveness of
multi-lingual, multi-task and cross-document type learning. We also discuss the
label collection schemes that help to minimize the amount of noise in the
collected data.

    

### [[2110.00428] Zero-shot Natural Language Video Localization](http://arxiv.org/abs/2110.00428)


  Understanding videos to localize moments with natural language often requires
large expensive annotated video regions paired with language queries. To
eliminate the annotation costs, we make a first attempt to train a natural
language video localization model in zero-shot manner. Inspired by unsupervised
image captioning setup, we merely require random text corpora, unlabeled video
collections, and an off-the-shelf object detector to train a model. With the
unpaired data, we propose to generate pseudo-supervision of candidate temporal
regions and corresponding query sentences, and develop a simple NLVL model to
train with the pseudo-supervision. Our empirical validations show that the
proposed pseudo-supervised method outperforms several baseline approaches and a
number of methods using stronger supervision on Charades-STA and
ActivityNet-Captions.

    

### [[2110.00433] External knowledge transfer deployment inside a simple double agent Viterbi algorithm](http://arxiv.org/abs/2110.00433)


  We consider in this paper deploying external knowledge transfer inside a
simple double agent Viterbi algorithm which is an algorithm firstly introduced
by the author in his preprint "Hidden Markov Based Mathematical Model dedicated
to Extract Ingredients from Recipe Text". The key challenge of this work lies
in discovering the reason why our old model does have bad performances when it
is confronted with estimating ingredient state for unknown words and see if
deploying external knowledge transfer directly on calculating state matrix
could be the solution instead of deploying it only on back propagating step.

    

### [[2110.00435] Attention based Sequence to Sequence Learning for Machine Translation of Low Resourced Indic Languages -- A case of Sanskrit to Hindi](http://arxiv.org/abs/2110.00435)


  Deep Learning techniques are powerful in mimicking humans in a particular set
of problems. They have achieved a remarkable performance in complex learning
tasks. Deep learning inspired Neural Machine Translation (NMT) is a proficient
technique that outperforms traditional machine translation. Performing
machine-aided translation on Indic languages has always been a challenging task
considering their rich and diverse grammar. The neural machine translation has
shown quality results compared to the traditional machine translation
approaches. The fully automatic machine translation becomes problematic when it
comes to low-resourced languages, especially with Sanskrit. This paper presents
attention mechanism based neural machine translation by selectively focusing on
a particular part of language sentences during translation. The work shows the
construction of Sanskrit to Hindi bilingual parallel corpus with nearly 10K
samples and having 178,000 tokens. The neural translation model equipped with
an attention mechanism has been trained on Sanskrit to Hindi parallel corpus.
The approach has shown the significance of attention mechanisms to overcome
long-term dependencies, primarily associated with low resources Indic
languages. The paper shows the attention plots on testing data to demonstrate
the alignment between source and translated words. For the evaluation of the
translated sentences, manual score based human evaluation and automatic
evaluation metric based techniques have been adopted. The attention mechanism
based neural translation has achieved 88% accuracy in human evaluation and a
BLEU score of 0.92 on Sanskrit to Hindi translation.

    

### [[2110.00453] Phonology Recognition in American Sign Language](http://arxiv.org/abs/2110.00453)


  Inspired by recent developments in natural language processing, we propose a
novel approach to sign language processing based on phonological properties
validated by American Sign Language users. By taking advantage of datasets
composed of phonological data and people speaking sign language, we use a
pretrained deep model based on mesh reconstruction to extract the 3D
coordinates of the signers keypoints. Then, we train standard statistical and
deep machine learning models in order to assign phonological classes to each
temporal sequence of coordinates.
Our paper introduces the idea of exploiting the phonological properties
manually assigned by sign language users to classify videos of people
performing signs by regressing a 3D mesh. We establish a new baseline for this
problem based on the statistical distribution of 725 different signs. Our
best-performing models achieve a micro-averaged F1-score of 58% for the major
location class and 70% for the sign type using statistical and deep learning
algorithms, compared to their corresponding baselines of 35% and 39%.

    

### [[2110.00464] MonoCInIS: Camera Independent Monocular 3D Object Detection using Instance Segmentation](http://arxiv.org/abs/2110.00464)


  Monocular 3D object detection has recently shown promising results, however
there remain challenging problems. One of those is the lack of invariance to
different camera intrinsic parameters, which can be observed across different
3D object datasets. Little effort has been made to exploit the combination of
heterogeneous 3D object datasets. In contrast to general intuition, we show
that more data does not automatically guarantee a better performance, but
rather, methods need to have a degree of 'camera independence' in order to
benefit from large and heterogeneous training data. In this paper we propose a
category-level pose estimation method based on instance segmentation, using
camera independent geometric reasoning to cope with the varying camera
viewpoints and intrinsics of different datasets. Every pixel of an instance
predicts the object dimensions, the 3D object reference points projected in 2D
image space and, optionally, the local viewing angle. Camera intrinsics are
only used outside of the learned network to lift the predicted 2D reference
points to 3D. We surpass camera independent methods on the challenging KITTI3D
benchmark and show the key benefits compared to camera dependent methods.

    

### [[2110.00470] Instance Segmentation Challenge Track Technical Report, VIPriors Workshop at ICCV 2021: Task-Specific Copy-Paste Data Augmentation Method for Instance Segmentation](http://arxiv.org/abs/2110.00470)


  Copy-Paste has proven to be a very effective data augmentation for instance
segmentation which can improve the generalization of the model. We used a
task-specific Copy-Paste data augmentation method to achieve good performance
on the instance segmentation track of the 2nd VIPriors workshop challenge. We
also applied additional data augmentation techniques including RandAugment and
GridMask. Our segmentation model is the HTC detector on the CBSwin-B with CBFPN
with some tweaks. This model was trained at the multi-scale mode by a random
sampler on the 6x schedule and tested at the single-scale mode. By combining
these techniques, we achieved 0.398 AP@0.50:0.95 with the validation set and
0.433 AP@0.50:0.95 with the test set. Finally, we reached 0.477 AP@0.50:0.95
with the test set by adding the validation set to the training data. Source
code is available at this https URL.

    

### [[2110.00521] Unpacking the Interdependent Systems of Discrimination: Ableist Bias in NLP Systems through an Intersectional Lens](http://arxiv.org/abs/2110.00521)


  Much of the world's population experiences some form of disability during
their lifetime. Caution must be exercised while designing natural language
processing (NLP) systems to prevent systems from inadvertently perpetuating
ableist bias against people with disabilities, i.e., prejudice that favors
those with typical abilities. We report on various analyses based on word
predictions of a large-scale BERT language model. Statistically significant
results demonstrate that people with disabilities can be disadvantaged.
Findings also explore overlapping forms of discrimination related to
interconnected gender and race identities.

    

### [[2110.00534] TEACh: Task-driven Embodied Agents that Chat](http://arxiv.org/abs/2110.00534)


  Robots operating in human spaces must be able to engage in natural language
interaction with people, both understanding and executing instructions, and
using conversation to resolve ambiguity and recover from mistakes. To study
this, we introduce TEACh, a dataset of over 3,000 human--human, interactive
dialogues to complete household tasks in simulation. A Commander with access to
oracle information about a task communicates in natural language with a
Follower. The Follower navigates through and interacts with the environment to
complete tasks varying in complexity from "Make Coffee" to "Prepare Breakfast",
asking questions and getting additional information from the Commander. We
propose three benchmarks using TEACh to study embodied intelligence challenges,
and we evaluate initial models' abilities in dialogue understanding, language
grounding, and task execution.

    

### [[2110.00558] Natural language understanding for logical games](http://arxiv.org/abs/2110.00558)


  We developed a system able to automatically solve logical puzzles in natural
language. Our solution is composed by a parser and an inference module. The
parser translates the text into first order logic (FOL), while the MACE4 model
finder is used to compute the models of the given FOL theory. We also empower
our software agent with the capability to provide Yes/No answers to natural
language questions related to each puzzle. Moreover, in line with Explainalbe
Artificial Intelligence (XAI), the agent can back its answer, providing a
graphical representation of the proof. The advantage of using reasoning for
Natural Language Understanding (NLU) instead of Machine learning is that the
user can obtain an explanation of the reasoning chain. We illustrate how the
system performs on various types of natural language puzzles, including 382
knights and knaves puzzles. These features together with the overall
performance rate of 80.89\% makes the proposed solution an improvement upon
similar solvers for natural language understanding in the puzzles domain.

    

### [[2104.04830] FRAKE: Fusional Real-time Automatic Keyword Extraction](http://arxiv.org/abs/2104.04830)


  Keyword extraction is the process of identifying the words or phrases that
express the main concepts of text to the best of one's ability. Electronic
infrastructure creates a considerable amount of text every day and at all
times. This massive volume of documents makes it practically impossible for
human resources to study and manage them. Nevertheless, the need for these
documents to be accessed efficiently and effectively is evident in numerous
purposes. A blog, news article, or technical note is considered a relatively
long text since the reader aims to learn the subject based on keywords or
topics. Our approach consists of a combination of two models: graph centrality
features and textural features. The proposed method has been used to extract
the best keyword among the candidate keywords with an optimal combination of
graph centralities, such as degree, betweenness, eigenvector, closeness
centrality and etc, and textural, such as Casing, Term position, Term frequency
normalization, Term different sentence, Part Of Speech tagging. There have also
been attempts to distinguish keywords from candidate phrases and consider them
on separate keywords. For evaluating the proposed method, seven datasets were
used: Semeval2010, SemEval2017, Inspec, fao30, Thesis100, pak2018, and
Wikinews, with results reported as Precision, Recall, and F- measure. Our
proposed method performed much better in terms of evaluation metrics in all
reviewed datasets compared with available methods in literature. An approximate
16.9% increase was witnessed in F-score metric and this was much more for the
Inspec in English datasets and WikiNews in forgone languages.

    

### [[2104.05848] Family of Origin and Family of Choice: Massively Parallel Lexiconized Iterative Pretraining for Severely Low Resource Machine Translation](http://arxiv.org/abs/2104.05848)


  We translate a closed text that is known in advance into a severely low
resource language by leveraging massive source parallelism. In other words,
given a text in 124 source languages, we translate it into a severely low
resource language using only ~1,000 lines of low resource data without any
external help. Firstly, we propose a systematic method to rank and choose
source languages that are close to the low resource language. We call the
linguistic definition of language family Family of Origin (FAMO), and we call
the empirical definition of higher-ranked languages using our metrics Family of
Choice (FAMC). Secondly, we build an Iteratively Pretrained Multilingual
Order-preserving Lexiconized Transformer (IPML) to train on ~1,000 lines
(~3.5%) of low resource data. To translate named entities correctly, we build a
massive lexicon table for 2,939 Bible named entities in 124 source languages,
and include many that occur once and covers more than 66 severely low resource
languages. Moreover, we also build a novel method of combining translations
from different source languages into one. Using English as a hypothetical low
resource language, we get a +23.9 BLEU increase over a multilingual baseline,
and a +10.3 BLEU increase over our asymmetric baseline in the Bible dataset. We
get a 42.8 BLEU score for Portuguese-English translation on the medical EMEA
dataset. We also have good results for a real severely low resource Mayan
language, Eastern Pokomchi.

    

### [[2110.00186] An Attempt to Generate Code for Symmetric Tensor Computations](http://arxiv.org/abs/2110.00186)


  This document describes an attempt to develop a compiler-based approach for
computations with symmetric tensors. Given a computation and the symmetries of
its input tensors, we derive formulas for random access under a storage scheme
that eliminates redundancies; construct intermediate representations to
describe the loop structure; and translate this information, using the taco
tensor algebra compiler, into code. While we achieve a framework for reasoning
about a fairly general class of symmetric computations, the resulting code is
not performant when the symmetries are misaligned.

    

### [[2110.00446] CHAD for Expressive Total Languages](http://arxiv.org/abs/2110.00446)


  We show how to apply forward and reverse mode Combinatory Homomorphic
Automatic Differentiation (CHAD) to total functional programming languages with
expressive type systems featuring the combination of - tuple types; - sum
types; - inductive types; - coinductive types; - function types. We achieve
this by analysing the categorical semantics of such types in $\Sigma$-types
(Grothendieck constructions) of suitable categories. Using a novel categorical
logical relations technique for such expressive type systems, we give a
correctness proof of CHAD in this setting by showing that it computes the usual
mathematical derivative of the function that the original program implements.
The result is a principled, purely functional and provably correct method for
performing forward and reverse mode automatic differentiation (AD) on total
functional programming languages with expressive type systems.

    

### [[2009.01489] HACCLE: Metaprogramming for Secure Multi-Party Computation -- Extended Version](http://arxiv.org/abs/2009.01489)


  Cryptographic techniques have the potential to enable distrusting parties to
collaborate in fundamentally new ways, but their practical implementation poses
numerous challenges. An important class of such cryptographic techniques is
known as Secure Multi-Party Computation (MPC). Developing Secure MPC
applications in realistic scenarios requires extensive knowledge spanning
multiple areas of cryptography and systems. And while the steps to arrive at a
solution for a particular application are often straightforward, it remains
difficult to make the implementation efficient, and tedious to apply those same
steps to a slightly different application from scratch. Hence, it is an
important problem to design platforms for implementing Secure MPC applications
with minimum effort and using techniques accessible to non-experts in
cryptography. In this paper, we present the HACCLE (High Assurance
Compositional Cryptography: Languages and Environments) toolchain, specifically
targeted to MPC applications. HACCLE contains an embedded domain-specific
language Harpoon, for software developers without cryptographic expertise to
write MPC-based programs, and uses Lightweight Modular Staging (LMS) for code
generation. Harpoon programs are compiled into acyclic circuits represented in
HACCLE's Intermediate Representation (HIR) that serves as an abstraction over
different cryptographic protocols such as secret sharing, homomorphic
encryption, or garbled circuits. Implementations of different cryptographic
protocols serve as different backends of our toolchain. The extensible design
of HIR allows cryptographic experts to plug in new primitives and protocols to
realize computation. And the use of standard metaprogramming techniques lowers
the development effort significantly.

    