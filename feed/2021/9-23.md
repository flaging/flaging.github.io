
## 2021-9-23

### [[2109.10489] Enabling Large-Scale Federated Learning over Wireless Edge Networks](http://arxiv.org/abs/2109.10489)


  Major bottlenecks of large-scale Federated Learning(FL) networks are the high
costs for communication and computation. This is due to the fact that most of
current FL frameworks only consider a star network topology where all local
trained models are aggregated at a single server (e.g., a cloud server). This
causes significant overhead at the server when the number of users are huge and
local models' sizes are large. This paper proposes a novel edge network
architecture which decentralizes the model aggregation process at the server,
thereby significantly reducing the aggregation latency of the whole network. In
this architecture, we propose a highly-effective in-network computation
protocol consisting of two components. First, an in-network aggregation process
is designed so that the majority of aggregation computations can be offloaded
from cloud server to edge nodes. Second, a joint routing and resource
allocation optimization problem is formulated to minimize the aggregation
latency for the whole system at every learning round. The problem turns out to
be NP-hard, and thus we propose a polynomial time routing algorithm which can
achieve near optimal performance with a theoretical bound. Numerical results
show that our proposed framework can dramatically reduce the network latency,
up to 4.6 times. Furthermore, this framework can significantly decrease cloud's
traffic and computing overhead by a factor of K/M, where K is the number of
users and M is the number of edge nodes, in comparison with conventional
baselines.

    

### [[2109.10868] A Context-aware Radio Resource Management in Heterogeneous Virtual RANs](http://arxiv.org/abs/2109.10868)


  New-generation wireless networks are designed to support a wide range of
services with diverse key performance indicators (KPIs) requirements. A
fundamental component of such networks, and a pivotal factor to the fulfillment
of the target KPIs, is the virtual radio access network (vRAN), which allows
high flexibility on the control of the radio link. However, to fully exploit
the potentiality of vRANs, an efficient mapping of the rapidly varying context
to radio control decisions is not only essential, but also challenging owing to
the interdependence of user traffic demand, channel conditions, and resource
allocation. Here, we propose CAREM, a reinforcement learning framework for
dynamic radio resource allocation in heterogeneous vRANs, which selects the
best available link and transmission parameters for packet transfer, so as to
meet the KPI requirements. To show its effectiveness, we develop a testbed for
proof-of-concept. Experimental results demonstrate that CAREM enables an
efficient radio resource allocation under different settings and traffic
demand. Also, compared to the closest existing scheme based on neural network
and the standard LTE, CAREM exhibits an improvement of one order of magnitude
in packet loss and latency, while it provides a 65% latency improvement
relatively to the contextual bandit approach.

    

### [[2109.10883] ENERO: Efficient Real-Time Routing Optimization](http://arxiv.org/abs/2109.10883)


  Wide Area Networks (WAN) are a key infrastructure in today's society. During
the last years, WANs have seen a considerable increase in network's traffic as
well as in the number of network applications. To enable the deployment of
emergent network applications (e.g., Vehicular networks, Internet of Things),
existing Traffic Engineering (TE) solutions must be able to achieve high
performance real-time network operation. In addition, TE solutions must be able
to adapt to dynamic scenarios (e.g., changes in the traffic matrix or topology
link failures). However, current TE technologies rely on hand-crafted
heuristics or computationally expensive solvers, which are not suitable for
highly dynamic TE scenarios.
In this paper we propose Enero, an efficient real-time TE engine. Enero is
based on a two-stage optimization process. In the first one, it leverages Deep
Reinforcement Learning (DRL) to optimize the routing configuration by
generating a long-term TE strategy. We integrated a Graph Neural Network (GNN)
into the DRL agent to enable efficient TE on dynamic networks. In the second
stage, Enero uses a Local Search algorithm to improve DRL's solution without
adding computational overhead to the optimization process. Enero offers a lower
bound in performance, enabling the network operator to know the worst-case
performance of the DRL agent. We believe that the lower bound in performance
will lighten the path of deploying DRL-based solutions in real-world network
scenarios. The experimental results indicate that Enero is able to operate in
real-world dynamic network topologies in 4.5 seconds on average for topologies
up to 100 edges.

    

### [[2002.09055] Encryption without Centralization: Distributing DNS Queries Across Recursive Resolvers](http://arxiv.org/abs/2002.09055)


  Emerging protocols such as DNS-over-HTTPS (DoH) and DNS-over-TLS (DoT)
improve the privacy of DNS queries and responses. While this trend towards
encryption is positive, deployment of these protocols has in some cases
resulted in further centralization of the DNS, which introduces new challenges.
In particular, centralization has consequences for performance, privacy, and
availability; a potentially greater concern is that it has become more
difficult to control the choice of DNS recursive resolver, particularly for IoT
devices. Ultimately, the best strategy for selecting among one or more
recursive resolvers may ultimately depend on circumstance, user, and even
device. Accordingly, the DNS architecture must permit flexibility in allowing
users, devices, and applications to specify these strategies. Towards this goal
of increased de-centralization and improved flexibility, this paper presents
the design and implementation of a refactored DNS resolver architecture that
allows for de-centralized name resolution, preserving the benefits of encrypted
DNS while satisfying other desirable properties, including performance and
privacy.

    

### [[2003.11999] Software-Defined Elastic Provisioning of IoT Edge Computing Virtual Resources](http://arxiv.org/abs/2003.11999)


  The fast growth of Internet-connected embedded devices demands for new
capabilities at the network edge. These new capabilities are local processing,
efficient communications, and resource virtualization. The current work aims to
address these capabilities by designing and deploying a new management
proposal, which offers on-demand activation of offline Internet of Things (IoT)
fog computing assets via a Software Defined Networking (SDN) based solution
combined with containerization and sensor virtualization. We propose a testbed
as a proof of concept for the main functionalities of this novel solution. The
obtained results evidence that the current SDN-based solution can deploy with
success activation policies on computational edge containers, which are located
within the same network domain of the SDN controller. In addition, different
application-level scenarios are also investigated.

    

### [[2102.07252] On Topology Optimization and Routing in Integrated Access and Backhaul Networks: A Genetic Algorithm-based Approach](http://arxiv.org/abs/2102.07252)


  In this paper, we study the problem of topology optimization and routing in
integrated access and backhaul (IAB) networks, as one of the promising
techniques for evolving 5G networks. We study the problem from different
perspectives. We develop efficient genetic algorithm-based schemes for both IAB
node placement and non-IAB backhaul link distribution, and evaluate the effect
of routing on bypassing temporal blockages. Here, concentrating on millimeter
wave-based communications, we study the service coverage probability, defined
as the probability of the event that the user equipments' (UEs) minimum rate
requirements are satisfied. Moreover, we study the effect of different
parameters such as the antenna gain, blockage and tree foliage on the system
performance. Finally, we summarize the recent Rel-16 as well as the upcoming
Rel-17 3GPP discussions on routing in IAB networks, and discuss the main
challenges for enabling mesh-based IAB networks. As we show, with a proper
network topology, IAB is an attractive approach to enable the network
densification required by 5G and beyond.

    

### [[2109.10360] Robust marginalization of baryonic effects for cosmological inference at the field level](http://arxiv.org/abs/2109.10360)


  We train neural networks to perform likelihood-free inference from
$(25\,h^{-1}{\rm Mpc})^2$ 2D maps containing the total mass surface density
from thousands of hydrodynamic simulations of the CAMELS project. We show that
the networks can extract information beyond one-point functions and power
spectra from all resolved scales ($\gtrsim 100\,h^{-1}{\rm kpc}$) while
performing a robust marginalization over baryonic physics at the field level:
the model can infer the value of $\Omega_{\rm m} (\pm 4\%)$ and $\sigma_8 (\pm
2.5\%)$ from simulations completely different to the ones used to train it.

    

### [[2109.10376] Learning through structure: towards deep neuromorphic knowledge graph embeddings](http://arxiv.org/abs/2109.10376)


  Computing latent representations for graph-structured data is an ubiquitous
learning task in many industrial and academic applications ranging from
molecule synthetization to social network analysis and recommender systems.
Knowledge graphs are among the most popular and widely used data
representations related to the Semantic Web. Next to structuring factual
knowledge in a machine-readable format, knowledge graphs serve as the backbone
of many artificial intelligence applications and allow the ingestion of context
information into various learning algorithms. Graph neural networks attempt to
encode graph structures in low-dimensional vector spaces via a message passing
heuristic between neighboring nodes. Over the recent years, a multitude of
different graph neural network architectures demonstrated ground-breaking
performances in many learning tasks. In this work, we propose a strategy to map
deep graph learning architectures for knowledge graph reasoning to neuromorphic
architectures. Based on the insight that randomly initialized and untrained
(i.e., frozen) graph neural networks are able to preserve local graph
structures, we compose a frozen neural network with shallow knowledge graph
embedding models. We experimentally show that already on conventional computing
hardware, this leads to a significant speedup and memory reduction while
maintaining a competitive performance level. Moreover, we extend the frozen
architecture to spiking neural networks, introducing a novel, event-based and
highly sparse knowledge graph embedding algorithm that is suitable for
implementation in neuromorphic hardware.

    

### [[2109.10380] Deep Policies for Online Bipartite Matching: A Reinforcement Learning Approach](http://arxiv.org/abs/2109.10380)


  From assigning computing tasks to servers and advertisements to users,
sequential online matching problems arise in a wide variety of domains. The
challenge in online matching lies in making irrevocable assignments while there
is uncertainty about future inputs. In the theoretical computer science
literature, most policies are myopic or greedy in nature. In real-world
applications where the matching process is repeated on a regular basis, the
underlying data distribution can be leveraged for better decision-making. We
present an end-to-end Reinforcement Learning framework for deriving better
matching policies based on trial-and-error on historical data. We devise a set
of neural network architectures, design feature representations, and
empirically evaluate them across two online matching problems: Edge-Weighted
Online Bipartite Matching and Online Submodular Bipartite Matching. We show
that most of the learning approaches perform significantly better than
classical greedy algorithms on four synthetic and real-world datasets. Our code
is publicly available at this https URL.

    

### [[2109.10393] Towards a Real-Time Facial Analysis System](http://arxiv.org/abs/2109.10393)


  Facial analysis is an active research area in computer vision, with many
practical applications. Most of the existing studies focus on addressing one
specific task and maximizing its performance. For a complete facial analysis
system, one needs to solve these tasks efficiently to ensure a smooth
experience. In this work, we present a system-level design of a real-time
facial analysis system. With a collection of deep neural networks for object
detection, classification, and regression, the system recognizes age, gender,
facial expression, and facial similarity for each person that appears in the
camera view. We investigate the parallelization and interplay of individual
tasks. Results on common off-the-shelf architecture show that the system's
accuracy is comparable to the state-of-the-art methods, and the recognition
speed satisfies real-time requirements. Moreover, we propose a multitask
network for jointly predicting the first three attributes, i.e., age, gender,
and facial expression. Source code and trained models are available at
this https URL.

    

### [[2109.10399] Learned Benchmarks for Subseasonal Forecasting](http://arxiv.org/abs/2109.10399)


  We develop a subseasonal forecasting toolkit of simple learned benchmark
models that outperform both operational practice and state-of-the-art machine
learning and deep learning methods. Our new models include (a) Climatology++,
an adaptive alternative to climatology that, for precipitation, is 9% more
accurate and 250% more skillful than the United States operational Climate
Forecasting System (CFSv2); (b) CFSv2++, a learned CFSv2 correction that
improves temperature and precipitation accuracy by 7-8% and skill by 50-275%;
and (c) Persistence++, an augmented persistence model that combines CFSv2
forecasts with lagged measurements to improve temperature and precipitation
accuracy by 6-9% and skill by 40-130%. Across the contiguous U.S., our
Climatology++, CFSv2++, and Persistence++ toolkit consistently outperforms
standard meteorological baselines, state-of-the-art machine and deep learning
methods, and the European Centre for Medium-Range Weather Forecasts ensemble.
Overall, we find that augmenting traditional forecasting approaches with
learned enhancements yields an effective and computationally inexpensive
strategy for building the next generation of subseasonal forecasting
benchmarks.

    

### [[2109.10404] Digital Signal Processing Using Deep Neural Networks](http://arxiv.org/abs/2109.10404)


  Currently there is great interest in the utility of deep neural networks
(DNNs) for the physical layer of radio frequency (RF) communications. In this
manuscript, we describe a custom DNN specially designed to solve problems in
the RF domain. Our model leverages the mechanisms of feature extraction and
attention through the combination of an autoencoder convolutional network with
a transformer network, to accomplish several important communications network
and digital signals processing (DSP) tasks. We also present a new open dataset
and physical data augmentation model that enables training of DNNs that can
perform automatic modulation classification, infer and correct transmission
channel effects, and directly demodulate baseband RF signals.

    

### [[2109.10410] RETRONLU: Retrieval Augmented Task-Oriented Semantic Parsing](http://arxiv.org/abs/2109.10410)


  While large pre-trained language models accumulate a lot of knowledge in
their parameters, it has been demonstrated that augmenting it with
non-parametric retrieval-based memory has a number of benefits from accuracy
improvements to data efficiency for knowledge-focused tasks, such as question
answering. In this paper, we are applying retrieval-based modeling ideas to the
problem of multi-domain task-oriented semantic parsing for conversational
assistants. Our approach, RetroNLU, extends a sequence-to-sequence model
architecture with a retrieval component, used to fetch existing similar
examples and provide them as an additional input to the model. In particular,
we analyze two settings, where we augment an input with (a) retrieved nearest
neighbor utterances (utterance-nn), and (b) ground-truth semantic parses of
nearest neighbor utterances (semparse-nn). Our technique outperforms the
baseline method by 1.5% absolute macro-F1, especially at the low resource
setting, matching the baseline model accuracy with only 40% of the data.
Furthermore, we analyze the nearest neighbor retrieval component's quality,
model sensitivity and break down the performance for semantic parses of
different utterance complexity.

    

### [[2109.10431] Fairness without Imputation: A Decision Tree Approach for Fair Prediction with Missing Values](http://arxiv.org/abs/2109.10431)


  We investigate the fairness concerns of training a machine learning model
using data with missing values. Even though there are a number of fairness
intervention methods in the literature, most of them require a complete
training set as input. In practice, data can have missing values, and data
missing patterns can depend on group attributes (e.g. gender or race). Simply
applying off-the-shelf fair learning algorithms to an imputed dataset may lead
to an unfair model. In this paper, we first theoretically analyze different
sources of discrimination risks when training with an imputed dataset. Then, we
propose an integrated approach based on decision trees that does not require a
separate process of imputation and learning. Instead, we train a tree with
missing incorporated as attribute (MIA), which does not require explicit
imputation, and we optimize a fairness-regularized objective function. We
demonstrate that our approach outperforms existing fairness intervention
methods applied to an imputed dataset, through several experiments on
real-world datasets.

    

### [[2109.10432] Beyond Discriminant Patterns: On the Robustness of Decision Rule Ensembles](http://arxiv.org/abs/2109.10432)


  Local decision rules are commonly understood to be more explainable, due to
the local nature of the patterns involved. With numerical optimization methods
such as gradient boosting, ensembles of local decision rules can gain good
predictive performance on data involving global structure. Meanwhile, machine
learning models are being increasingly used to solve problems in high-stake
domains including healthcare and finance. Here, there is an emerging consensus
regarding the need for practitioners to understand whether and how those models
could perform robustly in the deployment environments, in the presence of
distributional shifts. Past research on local decision rules has focused mainly
on maximizing discriminant patterns, without due consideration of robustness
against distributional shifts. In order to fill this gap, we propose a new
method to learn and ensemble local decision rules, that are robust both in the
training and deployment environments. Specifically, we propose to leverage
causal knowledge by regarding the distributional shifts in subpopulations and
deployment environments as the results of interventions on the underlying
system. We propose two regularization terms based on causal knowledge to search
for optimal and stable rules. Experiments on both synthetic and benchmark
datasets show that our method is effective and robust against distributional
shifts in multiple environments.

    

### [[2109.10436] Classification with Nearest Disjoint Centroids](http://arxiv.org/abs/2109.10436)


  In this paper, we develop a new classification method based on nearest
centroid, and it is called the nearest disjoint centroid classifier. Our method
differs from the nearest centroid classifier in the following two aspects: (1)
the centroids are defined based on disjoint subsets of features instead of all
the features, and (2) the distance is induced by the dimensionality-normalized
norm instead of the Euclidean norm. We provide a few theoretical results
regarding our method. In addition, we propose a simple algorithm based on
adapted k-means clustering that can find the disjoint subsets of features used
in our method, and extend the algorithm to perform feature selection. We
evaluate and compare the performance of our method to other closely related
classifiers on both simulated data and real-world gene expression datasets. The
results demonstrate that our method is able to outperform other competing
classifiers by having smaller misclassification rates and/or using fewer
features in various settings and situations.

    

### [[2109.10442] Selecting Datasets for Evaluating an Enhanced Deep Learning Framework](http://arxiv.org/abs/2109.10442)


  A framework was developed to address limitations associated with existing
techniques for analysing sequences. This work deals with the steps followed to
select suitable datasets characterised by discrete irregular sequential
patterns. To identify, select, explore and evaluate which datasets from various
sources extracted from more than 400 research articles, an interquartile range
method for outlier calculation and a qualitative Billauer's algorithm was
adapted to provide periodical peak detection in such datasets.
The developed framework was then tested using the most appropriate datasets.
The research concluded that the financial market-daily currency exchange
domain is the most suitable kind of data set for the evaluation of the designed
deep learning framework, as it provides high levels of discrete irregular
patterns.

    

### [[2109.10452] Personalized Online Machine Learning](http://arxiv.org/abs/2109.10452)


  In this work, we introduce the Personalized Online Super Learner (POSL) -- an
online ensembling algorithm for streaming data whose optimization procedure
accommodates varying degrees of personalization. Namely, POSL optimizes
predictions with respect to baseline covariates, so personalization can vary
from completely individualized (i.e., optimization with respect to baseline
covariate subject ID) to many individuals (i.e., optimization with respect to
common baseline covariates). As an online algorithm, POSL learns in real-time.
POSL can leverage a diversity of candidate algorithms, including online
algorithms with different training and update times, fixed algorithms that are
never updated during the procedure, pooled algorithms that learn from many
individuals' time-series, and individualized algorithms that learn from within
a single time-series. POSL's ensembling of this hybrid of base learning
strategies depends on the amount of data collected, the stationarity of the
time-series, and the mutual characteristics of a group of time-series. In
essence, POSL decides whether to learn across samples, through time, or both,
based on the underlying (unknown) structure in the data. For a wide range of
simulations that reflect realistic forecasting scenarios, and in a medical data
application, we examine the performance of POSL relative to other current
ensembling and online learning methods. We show that POSL is able to provide
reliable predictions for time-series data and adjust to changing
data-generating environments. We further cultivate POSL's practicality by
extending it to settings where time-series enter/exit dynamically over
chronological time.

    

### [[2109.10458] Achieving Counterfactual Fairness for Causal Bandit](http://arxiv.org/abs/2109.10458)


  In online recommendation, customers arrive in a sequential and stochastic
manner from an underlying distribution and the online decision model recommends
a chosen item for each arriving individual based on some strategy. We study how
to recommend an item at each step to maximize the expected reward while
achieving user-side fairness for customers, i.e., customers who share similar
profiles will receive a similar reward regardless of their sensitive attributes
and items being recommended. By incorporating causal inference into bandits and
adopting soft intervention to model the arm selection strategy, we first
propose the d-separation based UCB algorithm (D-UCB) to explore the utilization
of the d-separation set in reducing the amount of exploration needed to achieve
low cumulative regret. Based on that, we then propose the fair causal bandit
(F-UCB) for achieving the counterfactual individual fairness. Both theoretical
analysis and empirical evaluation demonstrate effectiveness of our algorithms.

    

### [[2109.10462] A Hierarchical Network-Oriented Analysis of User Participation in Misinformation Spread on WhatsApp](http://arxiv.org/abs/2109.10462)


  WhatsApp emerged as a major communication platform in many countries in the
recent years. Despite offering only one-to-one and small group conversations,
WhatsApp has been shown to enable the formation of a rich underlying network,
crossing the boundaries of existing groups, and with structural properties that
favor information dissemination at large. Indeed, WhatsApp has reportedly been
used as a forum of misinformation campaigns with significant social, political
and economic consequences in several countries. In this article, we aim at
complementing recent studies on misinformation spread on WhatsApp, mostly
focused on content properties and propagation dynamics, by looking into the
network that connects users sharing the same piece of content. Specifically, we
present a hierarchical network-oriented characterization of the users engaged
in misinformation spread by focusing on three perspectives: individuals,
WhatsApp groups and user communities, i.e., groupings of users who,
intentionally or not, share the same content disproportionately often. By
analyzing sharing and network topological properties, our study offers valuable
insights into how WhatsApp users leverage the underlying network connecting
different groups to gain large reach in the spread of misinformation on the
platform.

    

### [[2109.10465] Scalable and Efficient MoE Training for Multitask Multilingual Models](http://arxiv.org/abs/2109.10465)


  The Mixture of Experts (MoE) models are an emerging class of sparsely
activated deep learning models that have sublinear compute costs with respect
to their parameters. In contrast with dense models, the sparse architecture of
MoE offers opportunities for drastically growing model size with significant
accuracy gain while consuming much lower compute budget. However, supporting
large scale MoE training also has its own set of system and modeling
challenges. To overcome the challenges and embrace the opportunities of MoE, we
first develop a system capable of scaling MoE models efficiently to trillions
of parameters. It combines multi-dimensional parallelism and heterogeneous
memory technologies harmoniously with MoE to empower 8x larger models on the
same hardware compared with existing work. Besides boosting system efficiency,
we also present new training methods to improve MoE sample efficiency and
leverage expert pruning strategy to improve inference time efficiency. By
combining the efficient system and training methods, we are able to
significantly scale up large multitask multilingual models for language
generation which results in a great improvement in model accuracy. A model
trained with 10 billion parameters on 50 languages can achieve state-of-the-art
performance in Machine Translation (MT) and multilingual natural language
generation tasks. The system support of efficient MoE training has been
implemented and open-sourced with the DeepSpeed library.

    

### [[2109.10469] Differentiable Scaffolding Tree for Molecular Optimization](http://arxiv.org/abs/2109.10469)


  The structural design of functional molecules, also called molecular
optimization, is an essential chemical science and engineering task with
important applications, such as drug discovery. Deep generative models and
combinatorial optimization methods achieve initial success but still struggle
with directly modeling discrete chemical structures and often heavily rely on
brute-force enumeration. The challenge comes from the discrete and
non-differentiable nature of molecule structures. To address this, we propose
differentiable scaffolding tree (DST) that utilizes a learned knowledge network
to convert discrete chemical structures to locally differentiable ones. DST
enables a gradient-based optimization on a chemical graph structure by
back-propagating the derivatives from the target properties through a graph
neural network (GNN). Our empirical studies show the gradient-based molecular
optimizations are both effective and sample efficient. Furthermore, the learned
graph parameters can also provide an explanation that helps domain experts
understand the model output.

    

### [[2109.10471] The First Vision For Vitals (V4V) Challenge for Non-Contact Video-Based Physiological Estimation](http://arxiv.org/abs/2109.10471)


  Telehealth has the potential to offset the high demand for help during public
health emergencies, such as the COVID-19 pandemic. Remote Photoplethysmography
(rPPG) - the problem of non-invasively estimating blood volume variations in
the microvascular tissue from video - would be well suited for these
situations. Over the past few years a number of research groups have made rapid
advances in remote PPG methods for estimating heart rate from digital video and
obtained impressive results. How these various methods compare in naturalistic
conditions, where spontaneous behavior, facial expressions, and illumination
changes are present, is relatively unknown. To enable comparisons among
alternative methods, the 1st Vision for Vitals Challenge (V4V) presented a
novel dataset containing high-resolution videos time-locked with varied
physiological signals from a diverse population. In this paper, we outline the
evaluation protocol, the data used, and the results. V4V is to be held in
conjunction with the 2021 International Conference on Computer Vision.

    

### [[2109.10476] Self-Supervised Learning to Prove Equivalence Between Programs via Semantics-Preserving Rewrite Rules](http://arxiv.org/abs/2109.10476)


  We target the problem of synthesizing proofs of semantic equivalence between
two programs made of sequences of statements with complex symbolic expressions.
We propose a neural network architecture based on the transformer to generate
axiomatic proofs of equivalence between program pairs. We generate expressions
which include scalars and vectors and support multi-typed rewrite rules to
prove equivalence. For training the system, we develop an original training
technique, which we call self-supervised sample selection. This incremental
training improves the quality, generalizability and extensibility of the
learned model. We study the effectiveness of the system to generate proofs of
increasing length, and we demonstrate how transformer models learn to represent
complex and verifiable symbolic reasoning. Our system, S4Eq, achieves 97% proof
success on 10,000 pairs of programs while ensuring zero false positives by
design.

    

### [[2109.10477] Generating Compositional Color Representations from Text](http://arxiv.org/abs/2109.10477)


  We consider the cross-modal task of producing color representations for text
phrases. Motivated by the fact that a significant fraction of user queries on
an image search engine follow an (attribute, object) structure, we propose a
generative adversarial network that generates color profiles for such bigrams.
We design our pipeline to learn composition - the ability to combine seen
attributes and objects to unseen pairs. We propose a novel dataset curation
pipeline from existing public sources. We describe how a set of phrases of
interest can be compiled using a graph propagation technique, and then mapped
to images. While this dataset is specialized for our investigations on color,
the method can be extended to other visual dimensions where composition is of
interest. We provide detailed ablation studies that test the behavior of our
GAN architecture with loss functions from the contrastive learning literature.
We show that the generative model achieves lower Frechet Inception Distance
than discriminative ones, and therefore predicts color profiles that better
match those from real images. Finally, we demonstrate improved performance in
image retrieval and classification, indicating the crucial role that color
plays in these downstream tasks.

    

### [[2109.10478] AI in Osteoporosis](http://arxiv.org/abs/2109.10478)


  In this chapter we explore and evaluate methods for trabecular bone
characterization and osteoporosis diagnosis with increased interest in sparse
approximations. We first describe texture representation and classification
techniques, patch-based methods such as Bag of Keypoints, and more recent deep
neural networks. Then we introduce the concept of sparse representations for
pattern recognition and we detail integrative sparse analysis methods and
classifier decision fusion methods. We report cross-validation results on
osteoporosis datasets of bone radiographs and compare the results produced by
the different categories of methods. We conclude that advances in the AI and
machine learning fields have enabled the development of methods that can be
used as diagnostic tools in clinical settings.

    

### [[2109.10490] Benchmarking Lane-changing Decision-making for Deep Reinforcement Learning](http://arxiv.org/abs/2109.10490)


  The development of autonomous driving has attracted extensive attention in
recent years, and it is essential to evaluate the performance of autonomous
driving. However, testing on the road is expensive and inefficient. Virtual
testing is the primary way to validate and verify self-driving cars, and the
basis of virtual testing is to build simulation scenarios. In this paper, we
propose a training, testing, and evaluation pipeline for the lane-changing task
from the perspective of deep reinforcement learning. First, we design lane
change scenarios for training and testing, where the test scenarios include
stochastic and deterministic parts. Then, we deploy a set of benchmarks
consisting of learning and non-learning approaches. We train several
state-of-the-art deep reinforcement learning methods in the designed training
scenarios and provide the benchmark metrics evaluation results of the trained
models in the test scenarios. The designed lane-changing scenarios and
benchmarks are both opened to provide a consistent experimental environment for
the lane-changing task.

    

### [[2109.10502] A Spectral Approach to Off-Policy Evaluation for POMDPs](http://arxiv.org/abs/2109.10502)


  We consider off-policy evaluation (OPE) in Partially Observable Markov
Decision Processes, where the evaluation policy depends only on observable
variables but the behavior policy depends on latent states (Tennenholtz et al.
(2020a)). Prior work on this problem uses a causal identification strategy
based on one-step observable proxies of the hidden state, which relies on the
invertibility of certain one-step moment matrices. In this work, we relax this
requirement by using spectral methods and extending one-step proxies both into
the past and future. We empirically compare our OPE methods to existing ones
and demonstrate their improved prediction accuracy and greater generality.
Lastly, we derive a separate Importance Sampling (IS) algorithm which relies on
rank, distinctness, and positivity conditions, and not on the strict
sufficiency conditions of observable trajectories with respect to the reward
and hidden-state structure required by Tennenholtz et al. (2020a).

    

### [[2109.10503] Identifying Potential Exomoon Signals with Convolutional Neural Networks](http://arxiv.org/abs/2109.10503)


  Targeted observations of possible exomoon host systems will remain difficult
to obtain and time-consuming to analyze in the foreseeable future. As such,
time-domain surveys such as Kepler, K2 and TESS will continue to play a
critical role as the first step in identifying candidate exomoon systems, which
may then be followed-up with premier ground- or space-based telescopes. In this
work, we train an ensemble of convolutional neural networks (CNNs) to identify
candidate exomoon signals in single-transit events observed by Kepler. Our
training set consists of ${\sim}$27,000 examples of synthetic, planet-only and
planet+moon single transits, injected into Kepler light curves. We achieve up
to 88\% classification accuracy with individual CNN architectures and 97\%
precision in identifying the moons in the validation set when the CNN ensemble
is in total agreement. We then apply the CNN ensemble to light curves from 1880
Kepler Objects of Interest with periods $>10$ days ($\sim$57,000 individual
transits), and further test the accuracy of the CNN classifier by injecting
planet transits into each light curve, thus quantifying the extent to which
residual stellar activity may result in false positive classifications. We find
a small fraction of these transits contain moon-like signals, though we caution
against strong inferences of the exomoon occurrence rate from this result. We
conclude by discussing some ongoing challenges to utilizing neural networks for
the exomoon search.

    

### [[2109.10506] Tecnologica cosa: Modeling Storyteller Personalities in Boccaccio's Decameron](http://arxiv.org/abs/2109.10506)


  We explore Boccaccio's Decameron to see how digital humanities tools can be
used for tasks that have limited data in a language no longer in contemporary
use: medieval Italian. We focus our analysis on the question: Do the different
storytellers in the text exhibit distinct personalities? To answer this
question, we curate and release a dataset based on the authoritative edition of
the text. We use supervised classification methods to predict storytellers
based on the stories they tell, confirming the difficulty of the task, and
demonstrate that topic modeling can extract thematic storyteller "profiles."

    

### [[2109.10509] Unsupervised Contextualized Document Representation](http://arxiv.org/abs/2109.10509)


  Several NLP tasks need the effective representation of text documents. Arora
et. al., 2017 demonstrate that simple weighted averaging of word vectors
frequently outperforms neural models. SCDV (Mekala et. al., 2017) further
extends this from sentences to documents by employing soft and sparse
clustering over pre-computed word vectors. However, both techniques ignore the
polysemy and contextual character of words. In this paper, we address this
issue by proposing SCDV+BERT(ctxd), a simple and effective unsupervised
representation that combines contextualized BERT (Devlin et al., 2019) based
word embedding for word sense disambiguation with SCDV soft clustering
approach. We show that our embeddings outperform original SCDV, pre-train BERT,
and several other baselines on many classification datasets. We also
demonstrate our embeddings effectiveness on other tasks, such as concept
matching and sentence similarity. In addition, we show that SCDV+BERT(ctxd)
outperforms fine-tune BERT and different embedding approaches in scenarios with
limited data and only few shots examples.

    

### [[2109.10512] Backdoor Attacks on Federated Learning with Lottery Ticket Hypothesis](http://arxiv.org/abs/2109.10512)


  Edge devices in federated learning usually have much more limited computation
and communication resources compared to servers in a data center. Recently,
advanced model compression methods, like the Lottery Ticket Hypothesis, have
already been implemented on federated learning to reduce the model size and
communication cost. However, Backdoor Attack can compromise its implementation
in the federated learning scenario. The malicious edge device trains the client
model with poisoned private data and uploads parameters to the center,
embedding a backdoor to the global shared model after unwitting aggregative
optimization. During the inference phase, the model with backdoors classifies
samples with a certain trigger as one target category, while shows a slight
decrease in inference accuracy to clean samples. In this work, we empirically
demonstrate that Lottery Ticket models are equally vulnerable to backdoor
attacks as the original dense models, and backdoor attacks can influence the
structure of extracted tickets. Based on tickets' similarities between each
other, we provide a feasible defense for federated learning against backdoor
attacks on various datasets.

    

### [[2109.10514] Towards The Automatic Coding of Medical Transcripts to Improve Patient-Centered Communication](http://arxiv.org/abs/2109.10514)


  This paper aims to provide an approach for automatic coding of
physician-patient communication transcripts to improve patient-centered
communication (PCC). PCC is a central part of high-quality health care. To
improve PCC, dialogues between physicians and patients have been recorded and
tagged with predefined codes. Trained human coders have manually coded the
transcripts. Since it entails huge labor costs and poses possible human errors,
automatic coding methods should be considered for efficiency and effectiveness.
We adopted three machine learning algorithms (Naïve Bayes, Random Forest, and
Support Vector Machine) to categorize lines in transcripts into corresponding
codes. The result showed that there is evidence to distinguish the codes, and
this is considered to be sufficient for training of human annotators.

    

### [[2109.10523] Investigating and Modeling the Dynamics of Long Ties](http://arxiv.org/abs/2109.10523)


  Long ties, the social ties that bridge different communities, are widely
believed to play crucial roles in spreading novel information in social
networks. However, some existing network theories and prediction models
indicate that long ties might dissolve quickly or eventually become redundant,
thus putting into question the long-term value of long ties. Our empirical
analysis of real-world dynamic networks shows that contrary to such reasoning,
long ties are more likely to persist than other social ties, and that many of
them constantly function as social bridges without being embedded in local
networks. Using a novel cost-benefit analysis model combined with machine
learning, we show that long ties are highly beneficial, which instinctively
motivates people to expend extra effort to maintain them. This partly explains
why long ties are more persistent than what has been suggested by many existing
theories and models. Overall, our study suggests the need for social
interventions that can promote the formation of long ties, such as mixing
people with diverse backgrounds.

    

### [[2109.10528] A unified interpretation of the Gaussian mechanism for differential privacy through the sensitivity index](http://arxiv.org/abs/2109.10528)


  The Gaussian mechanism (GM) represents a universally employed tool for
achieving differential privacy (DP), and a large body of work has been devoted
to its analysis. We argue that the three prevailing interpretations of the GM,
namely $(\varepsilon, \delta)$-DP, f-DP and Rényi DP can be expressed by
using a single parameter $\psi$, which we term the sensitivity index. $\psi$
uniquely characterises the GM and its properties by encapsulating its two
fundamental quantities: the sensitivity of the query and the magnitude of the
noise perturbation. With strong links to the ROC curve and the
hypothesis-testing interpretation of DP, $\psi$ offers the practitioner a
powerful method for interpreting, comparing and communicating the privacy
guarantees of Gaussian mechanisms.

    

### [[2109.10535] Cramér-Rao bound-informed training of neural networks for quantitative MRI](http://arxiv.org/abs/2109.10535)


  Neural networks are increasingly used to estimate parameters in quantitative
MRI, in particular in magnetic resonance fingerprinting. Their advantages over
the gold standard non-linear least square fitting are their superior speed and
their immunity to the non-convexity of many fitting problems. We find, however,
that in heterogeneous parameter spaces, i.e. in spaces in which the variance of
the estimated parameters varies considerably, good performance is hard to
achieve and requires arduous tweaking of the loss function, hyper parameters,
and the distribution of the training data in parameter space. Here, we address
these issues with a theoretically well-founded loss function: the Cramér-Rao
bound (CRB) provides a theoretical lower bound for the variance of an unbiased
estimator and we propose to normalize the squared error with respective CRB.
With this normalization, we balance the contributions of hard-to-estimate and
not-so-hard-to-estimate parameters and areas in parameter space, and avoid a
dominance of the former in the overall training loss. Further, the CRB-based
loss function equals one for a maximally-efficient unbiased estimator, which we
consider the ideal estimator. Hence, the proposed CRB-based loss function
provides an absolute evaluation metric. We compare a network trained with the
CRB-based loss with a network trained with the commonly used means squared
error loss and demonstrate the advantages of the former in numerical, phantom,
and in vivo experiments.

    

### [[2109.10538] Index $t$-SNE: Tracking Dynamics of High-Dimensional Datasets with Coherent Embeddings](http://arxiv.org/abs/2109.10538)


  $t$-SNE is an embedding method that the data science community has widely Two
interesting characteristics of t-SNE are the structure preservation property
and the answer to the crowding problem, where all neighbors in high dimensional
space cannot be represented correctly in low dimensional space. $t$-SNE
preserves the local neighborhood, and similar items are nicely spaced by
adjusting to the local density. These two characteristics produce a meaningful
representation, where the cluster area is proportional to its size in number,
and relationships between clusters are materialized by closeness on the
embedding.
This algorithm is non-parametric, therefore two initializations of the
algorithm would lead to two different embedding. In a forensic approach,
analysts would like to compare two or more datasets using their embedding. An
approach would be to learn a parametric model over an embedding built with a
subset of data. While this approach is highly scalable, points could be mapped
at the same exact position, making them indistinguishable. This type of model
would be unable to adapt to new outliers nor concept drift.
This paper presents a methodology to reuse an embedding to create a new one,
where cluster positions are preserved. The optimization process minimizes two
costs, one relative to the embedding shape and the second relative to the
support embedding' match. The proposed algorithm has the same complexity than
the original $t$-SNE to embed new items, and a lower one when considering the
embedding of a dataset sliced into sub-pieces. The method showed promising
results on a real-world dataset, allowing to observe the birth, evolution and
death of clusters. The proposed approach facilitates identifying significant
trends and changes, which empowers the monitoring high dimensional datasets'
dynamics.

    

### [[2109.10552] MEPG: A Minimalist Ensemble Policy Gradient Framework for Deep Reinforcement Learning](http://arxiv.org/abs/2109.10552)


  Ensemble reinforcement learning (RL) aims to mitigate instability in
Q-learning and to learn a robust policy, which introduces multiple value and
policy functions. In this paper, we consider finding a novel but simple
ensemble Deep RL algorithm to solve the resource consumption issue.
Specifically, we consider integrating multiple models into a single model. To
this end, we propose the \underline{M}inimalist \underline{E}nsemble
\underline{P}olicy \underline{G}radient framework (MEPG), which introduces
minimalist ensemble consistent Bellman update. And we find one value network is
sufficient in our framework. Moreover, we theoretically show that the policy
evaluation phase in the MEPG is mathematically equivalent to a deep Gaussian
Process. To verify the effectiveness of the MEPG framework, we conduct
experiments on the gym simulator, which show that the MEPG framework matches or
outperforms the state-of-the-art ensemble methods and model-free methods
without additional computational resource costs.

    

### [[2109.10569] The Curse Revisited: a Newly Quantified Concept of Meaningful Distances for Learning from High-Dimensional Noisy Data](http://arxiv.org/abs/2109.10569)


  Distances between data points are widely used in point cloud representation
learning. Yet, it is no secret that under the effect of noise, these
distances-and thus the models based upon them-may lose their usefulness in high
dimensions. Indeed, the small marginal effects of the noise may then accumulate
quickly, shifting empirical closest and furthest neighbors away from the ground
truth. In this paper, we characterize such effects in high-dimensional data
using an asymptotic probabilistic expression. Furthermore, while it has been
previously argued that neighborhood queries become meaningless and unstable
when there is a poor relative discrimination between the furthest and closest
point, we conclude that this is not necessarily the case when explicitly
separating the ground truth data from the noise. More specifically, we derive
that under particular conditions, empirical neighborhood relations affected by
noise are still likely to be true even when we observe this discrimination to
be poor. We include thorough empirical verification of our results, as well as
experiments that interestingly show our derived phase shift where neighbors
become random or not is identical to the phase shift where common
dimensionality reduction methods perform poorly or well for finding
low-dimensional representations of high-dimensional data with dense noise.

    

### [[2109.10573] An automatic differentiation system for the age of differential privacy](http://arxiv.org/abs/2109.10573)


  We introduce Tritium, an automatic differentiation-based sensitivity analysis
framework for differentially private (DP) machine learning (ML). Optimal noise
calibration in this setting requires efficient Jacobian matrix computations and
tight bounds on the L2-sensitivity. Our framework achieves these objectives by
relying on a functional analysis-based method for sensitivity tracking, which
we briefly outline. This approach interoperates naturally and seamlessly with
static graph-based automatic differentiation, which enables order-of-magnitude
improvements in compilation times compared to previous work. Moreover, we
demonstrate that optimising the sensitivity of the entire computational graph
at once yields substantially tighter estimates of the true sensitivity compared
to interval bound propagation techniques. Our work naturally befits recent
developments in DP such as individual privacy accounting, aiming to offer
improved privacy-utility trade-offs, and represents a step towards the
integration of accessible machine learning tooling with advanced privacy
accounting systems.

    

### [[2109.10581] Deep Augmented MUSIC Algorithm for Data-Driven DoA Estimation](http://arxiv.org/abs/2109.10581)


  Direction of arrival (DoA) estimation is a crucial task in sensor array
signal processing, giving rise to various successful model-based (MB)
algorithms as well as recently developed data-driven (DD) methods. This paper
introduces a new hybrid MB/DD DoA estimation architecture, based on the
classical multiple signal classification (MUSIC) algorithm. Our approach
augments crucial aspects of the original MUSIC structure with specifically
designed neural architectures, allowing it to overcome certain limitations of
the purely MB method, such as its inability to successfully localize coherent
sources. The deep augmented MUSIC algorithm is shown to outperform its
unaltered version with a superior resolution.

    

### [[2109.10591] High-dimensional Bayesian Optimization for CNN Auto Pruning with Clustering and Rollback](http://arxiv.org/abs/2109.10591)


  Pruning has been widely used to slim convolutional neural network (CNN)
models to achieve a good trade-off between accuracy and model size so that the
pruned models become feasible for power-constrained devices such as mobile
phones. This process can be automated to avoid the expensive hand-crafted
efforts and to explore a large pruning space automatically so that the
high-performance pruning policy can be achieved efficiently. Nowadays,
reinforcement learning (RL) and Bayesian optimization (BO)-based auto pruners
are widely used due to their solid theoretical foundation, universality, and
high compressing quality. However, the RL agent suffers from long training
times and high variance of results, while the BO agent is time-consuming for
high-dimensional design spaces. In this work, we propose an enhanced BO agent
to obtain significant acceleration for auto pruning in high-dimensional design
spaces. To achieve this, a novel clustering algorithm is proposed to reduce the
dimension of the design space to speedup the searching process. Then, a
roll-back algorithm is proposed to recover the high-dimensional design space so
that higher pruning accuracy can be obtained. We validate our proposed method
on ResNet, MobileNet, and VGG models, and our experiments show that the
proposed method significantly improves the accuracy of BO when pruning very
deep CNN models. Moreover, our method achieves lower variance and shorter time
than the RL-based counterpart.

    

### [[2109.10593] Emulating Aerosol Microphysics with a Machine Learning](http://arxiv.org/abs/2109.10593)


  Aerosol particles play an important role in the climate system by absorbing
and scattering radiation and influencing cloud properties. They are also one of
the biggest sources of uncertainty for climate modeling. Many climate models do
not include aerosols in sufficient detail. In order to achieve higher accuracy,
aerosol microphysical properties and processes have to be accounted for. This
is done in the ECHAM-HAM global climate aerosol model using the M7 microphysics
model, but increased computational costs make it very expensive to run at
higher resolutions or for a longer time. We aim to use machine learning to
approximate the microphysics model at sufficient accuracy and reduce the
computational cost by being fast at inference time. The original M7 model is
used to generate data of input-output pairs to train a neural network on it. By
using a special logarithmic transform we are able to learn the variables
tendencies achieving an average $R^2$ score of $89\%$. On a GPU we achieve a
speed-up of 120 compared to the original model.

    

### [[2109.10596] Fully probabilistic design for knowledge fusion between Bayesian filters under uniform disturbances](http://arxiv.org/abs/2109.10596)


  This paper considers the problem of Bayesian transfer learning-based
knowledge fusion between linear state-space processes driven by uniform state
and observation noise processes. The target task conditions on probabilistic
state predictor(s) supplied by the source filtering task(s) to improve its own
state estimate. A joint model of the target and source(s) is not required and
is not elicited. The resulting decision-making problem for choosing the optimal
conditional target filtering distribution under incomplete modelling is solved
via fully probabilistic design (FPD), i.e. via appropriate minimization of
Kullback-Leibler divergence (KLD). The resulting FPD-optimal target learner is
robust, in the sense that it can reject poor-quality source knowledge. In
addition, the fact that this Bayesian transfer learning (BTL) scheme does not
depend on a model of interaction between the source and target tasks ensures
robustness to the misspecification of such a model. The latter is a problem
that affects conventional transfer learning methods. The properties of the
proposed BTL scheme are demonstrated via extensive simulations, and in
comparison with two contemporary alternatives.

    

### [[2109.10598] Diarisation using Location tracking with agglomerative clustering](http://arxiv.org/abs/2109.10598)


  Previous works have shown that spatial location information can be
complementary to speaker embeddings for a speaker diarisation task. However,
the models used often assume that speakers are fairly stationary throughout a
meeting. This paper proposes to relax this assumption, by explicitly modelling
the movements of speakers within an Agglomerative Hierarchical Clustering (AHC)
diarisation framework. Kalman filters, which track the locations of speakers,
are used to compute log-likelihood ratios that contribute to the cluster
affinity computations for the AHC merging and stopping decisions. Experiments
show that the proposed approach is able to yield improvements on a Microsoft
rich meeting transcription task, compared to methods that do not use location
information or that make stationarity assumptions.

    

### [[2109.10617] Solving Large Steiner Tree Problems in Graphs for Cost-Efficient Fiber-To-The-Home Network Expansion](http://arxiv.org/abs/2109.10617)


  The expansion of Fiber-To-The-Home (FTTH) networks creates high costs due to
expensive excavation procedures. Optimizing the planning process and minimizing
the cost of the earth excavation work therefore lead to large savings.
Mathematically, the FTTH network problem can be described as a minimum Steiner
Tree problem. Even though the Steiner Tree problem has already been
investigated intensively in the last decades, it might be further optimized
with the help of new computing paradigms and emerging approaches. This work
studies upcoming technologies, such as Quantum Annealing, Simulated Annealing
and nature-inspired methods like Evolutionary Algorithms or slime-mold-based
optimization. Additionally, we investigate partitioning and simplifying
methods. Evaluated on several real-life problem instances, we could outperform
a traditional, widely-used baseline (NetworkX Approximate Solver) on most of
the domains. Prior partitioning of the initial graph and the presented
slime-mold-based approach were especially valuable for a cost-efficient
approximation. Quantum Annealing seems promising, but was limited by the number
of available qubits.

    

### [[2109.10623] Sharp Analysis of Random Fourier Features in Classification](http://arxiv.org/abs/2109.10623)


  We study the theoretical properties of random Fourier features classification
with Lipschitz continuous loss functions such as support vector machine and
logistic regression. Utilizing the regularity condition, we show for the first
time that random Fourier features classification can achieve $O(1/\sqrt{n})$
learning rate with only $\Omega(\sqrt{n} \log n)$ features, as opposed to
$\Omega(n)$ features suggested by previous results. Our study covers the
standard feature sampling method for which we reduce the number of features
required, as well as a problem-dependent sampling method which further reduces
the number of features while still keeping the optimal generalization property.
Moreover, we prove that the random Fourier features classification can obtain a
fast $O(1/n)$ learning rate for both sampling schemes under Massart's low noise
assumption. Our results demonstrate the potential effectiveness of random
Fourier features approximation in reducing the computational complexity
(roughly from $O(n^3)$ in time and $O(n^2)$ in space to $O(n^2)$ and
$O(n\sqrt{n})$ respectively) without having to trade-off the statistical
prediction accuracy. In addition, the achieved trade-off in our analysis is at
least the same as the optimal results in the literature under the worst case
scenario and significantly improves the optimal results under benign regularity
conditions.

    

### [[2109.10632] Locality Matters: A Scalable Value Decomposition Approach for Cooperative Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2109.10632)


  Cooperative multi-agent reinforcement learning (MARL) faces significant
scalability issues due to state and action spaces that are exponentially large
in the number of agents. As environments grow in size, effective credit
assignment becomes increasingly harder and often results in infeasible learning
times. Still, in many real-world settings, there exist simplified underlying
dynamics that can be leveraged for more scalable solutions. In this work, we
exploit such locality structures effectively whilst maintaining global
cooperation. We propose a novel, value-based multi-agent algorithm called
LOMAQ, which incorporates local rewards in the Centralized Training
Decentralized Execution paradigm. Additionally, we provide a direct reward
decomposition method for finding these local rewards when only a global signal
is provided. We test our method empirically, showing it scales well compared to
other methods, significantly improving performance and convergence speed.

    

### [[2109.10640] LDC-VAE: A Latent Distribution Consistency Approach to Variational AutoEncoders](http://arxiv.org/abs/2109.10640)


  Variational autoencoders (VAEs), as an important aspect of generative models,
have received a lot of research interests and reached many successful
applications. However, it is always a challenge to achieve the consistency
between the learned latent distribution and the prior latent distribution when
optimizing the evidence lower bound (ELBO), and finally leads to an
unsatisfactory performance in data generation. In this paper, we propose a
latent distribution consistency approach to avoid such substantial
inconsistency between the posterior and prior latent distributions in ELBO
optimizing. We name our method as latent distribution consistency VAE
(LDC-VAE). We achieve this purpose by assuming the real posterior distribution
in latent space as a Gibbs form, and approximating it by using our encoder.
However, there is no analytical solution for such Gibbs posterior in
approximation, and traditional approximation ways are time consuming, such as
using the iterative sampling-based MCMC. To address this problem, we use the
Stein Variational Gradient Descent (SVGD) to approximate the Gibbs posterior.
Meanwhile, we use the SVGD to train a sampler net which can obtain efficient
samples from the Gibbs posterior. Comparative studies on the popular image
generation datasets show that our method has achieved comparable or even better
performance than several powerful improvements of VAEs.

    

### [[2109.10641] Uncertainty-Aware Training for Cardiac Resynchronisation Therapy Response Prediction](http://arxiv.org/abs/2109.10641)


  Evaluation of predictive deep learning (DL) models beyond conventional
performance metrics has become increasingly important for applications in
sensitive environments like healthcare. Such models might have the capability
to encode and analyse large sets of data but they often lack comprehensive
interpretability methods, preventing clinical trust in predictive outcomes.
Quantifying uncertainty of a prediction is one way to provide such
interpretability and promote trust. However, relatively little attention has
been paid to how to include such requirements into the training of the model.
In this paper we: (i) quantify the data (aleatoric) and model (epistemic)
uncertainty of a DL model for Cardiac Resynchronisation Therapy response
prediction from cardiac magnetic resonance images, and (ii) propose and perform
a preliminary investigation of an uncertainty-aware loss function that can be
used to retrain an existing DL image-based classification model to encourage
confidence in correct predictions and reduce confidence in incorrect
predictions. Our initial results are promising, showing a significant increase
in the (epistemic) confidence of true positive predictions, with some evidence
of a reduction in false negative confidence.

    

### [[2109.10642] Decentralized Learning of Tree-Structured Gaussian Graphical Models from Noisy Data](http://arxiv.org/abs/2109.10642)


  This paper studies the decentralized learning of tree-structured Gaussian
graphical models (GGMs) from noisy data. In decentralized learning, data set is
distributed across different machines (sensors), and GGMs are widely used to
model complex networks such as gene regulatory networks and social networks.
The proposed decentralized learning uses the Chow-Liu algorithm for estimating
the tree-structured GGM.
In previous works, upper bounds on the probability of incorrect tree
structure recovery were given mostly without any practical noise for
simplification. While this paper investigates the effects of three common types
of noisy channels: Gaussian, Erasure, and binary symmetric channel. For
Gaussian channel case, to satisfy the failure probability upper bound $\delta >
0$ in recovering a $d$-node tree structure, our proposed theorem requires only
$\mathcal{O}(\log(\frac{d}{\delta}))$ samples for the smallest sample size
($n$) comparing to the previous literature \cite{Nikolakakis} with
$\mathcal{O}(\log^4(\frac{d}{\delta}))$ samples by using the positive
correlation coefficient assumption that is used in some important works in the
literature. Moreover, the approximately bounded Gaussian random variable
assumption does not appear in \cite{Nikolakakis}. Given some knowledge about
the tree structure, the proposed Algorithmic Bound will achieve obviously
better performance with small sample size (e.g., $< 2000$) comparing with
formulaic bounds. Finally, we validate our theoretical results by performing
simulations on synthetic data sets.

    

### [[2109.10656] Vehicle Behavior Prediction and Generalization Using Imbalanced Learning Techniques](http://arxiv.org/abs/2109.10656)


  The use of learning-based methods for vehicle behavior prediction is a
promising research topic. However, many publicly available data sets suffer
from class distribution skews which limits learning performance if not
addressed. This paper proposes an interaction-aware prediction model consisting
of an LSTM autoencoder and SVM classifier. Additionally, an imbalanced learning
technique, the multiclass balancing ensemble is proposed. Evaluations show that
the method enhances model performance, resulting in improved classification
accuracy. Good generalization properties of learned models are important and
therefore a generalization study is done where models are evaluated on unseen
traffic data with dissimilar traffic behavior stemming from different road
configurations. This is realized by using two distinct highway traffic
recordings, the publicly available NGSIM US-101 and I80 data sets. Moreover,
methods for encoding structural and static features into the learning process
for improved generalization are evaluated. The resulting methods show
substantial improvements in classification as well as generalization
performance.

    

### [[2109.10679] Application of Video-to-Video Translation Networks to Computational Fluid Dynamics](http://arxiv.org/abs/2109.10679)


  In recent years, the evolution of artificial intelligence, especially deep
learning, has been remarkable, and its application to various fields has been
growing rapidly. In this paper, I report the results of the application of
generative adversarial networks (GANs), specifically video-to-video translation
networks, to computational fluid dynamics (CFD) simulations. The purpose of
this research is to reduce the computational cost of CFD simulations with GANs.
The architecture of GANs in this research is a combination of the
image-to-image translation networks (the so-called "pix2pix") and Long
Short-Term Memory (LSTM). It is shown that the results of high-cost and
high-accuracy simulations (with high-resolution computational grids) can be
estimated from those of low-cost and low-accuracy simulations (with
low-resolution grids). In particular, the time evolution of density
distributions in the cases of a high-resolution grid is reproduced from that in
the cases of a low-resolution grid through GANs, and the density inhomogeneity
estimated from the image generated by GANs recovers the ground truth with good
accuracy. Qualitative and quantitative comparisons of the results of the
proposed method with those of several super-resolution algorithms are also
presented.

    

### [[2109.10681] A Latent Restoring Force Approach to Nonlinear System Identification](http://arxiv.org/abs/2109.10681)


  Identification of nonlinear dynamic systems remains a significant challenge
across engineering. This work suggests an approach based on Bayesian filtering
to extract and identify the contribution of an unknown nonlinear term in the
system which can be seen as an alternative viewpoint on restoring force surface
type approaches. To achieve this identification, the contribution which is the
nonlinear restoring force is modelled, initially, as a Gaussian process in
time. That Gaussian process is converted into a state-space model and combined
with the linear dynamic component of the system. Then, by inference of the
filtering and smoothing distributions, the internal states of the system and
the nonlinear restoring force can be extracted. In possession of these states a
nonlinear model can be constructed. The approach is demonstrated to be
effective in both a simulated case study and on an experimental benchmark
dataset.

    

### [[2109.10683] Adaptive Neural Message Passing for Inductive Learning on Hypergraphs](http://arxiv.org/abs/2109.10683)


  Graphs are the most ubiquitous data structures for representing relational
datasets and performing inferences in them. They model, however, only pairwise
relations between nodes and are not designed for encoding the higher-order
relations. This drawback is mitigated by hypergraphs, in which an edge can
connect an arbitrary number of nodes. Most hypergraph learning approaches
convert the hypergraph structure to that of a graph and then deploy existing
geometric deep learning methods. This transformation leads to information loss,
and sub-optimal exploitation of the hypergraph's expressive power. We present
HyperMSG, a novel hypergraph learning framework that uses a modular two-level
neural message passing strategy to accurately and efficiently propagate
information within each hyperedge and across the hyperedges. HyperMSG adapts to
the data and task by learning an attention weight associated with each node's
degree centrality. Such a mechanism quantifies both local and global importance
of a node, capturing the structural properties of a hypergraph. HyperMSG is
inductive, allowing inference on previously unseen nodes. Further, it is robust
and outperforms state-of-the-art hypergraph learning methods on a wide range of
tasks and datasets. Finally, we demonstrate the effectiveness of HyperMSG in
learning multimodal relations through detailed experimentation on a challenging
multimedia dataset.

    

### [[2109.10686] Scale Efficiently: Insights from Pre-training and Fine-tuning Transformers](http://arxiv.org/abs/2109.10686)


  There remain many open questions pertaining to the scaling behaviour of
Transformer architectures. These scaling decisions and findings can be
critical, as training runs often come with an associated computational cost
which have both financial and/or environmental impact. The goal of this paper
is to present scaling insights from pretraining and finetuning Transformers.
While Kaplan et al. presents a comprehensive study of the scaling behaviour of
Transformer language models, the scope is only on the upstream (pretraining)
loss. Therefore, it is still unclear if these set of findings transfer to
downstream task within the context of the pretrain-finetune paradigm. The key
findings of this paper are as follows: (1) we show that aside from only the
model size, model shape matters for downstream fine-tuning, (2) scaling
protocols operate differently at different compute regions, (3) widely adopted
T5-base and T5-large sizes are Pareto-inefficient. To this end, we present
improved scaling protocols whereby our redesigned models achieve similar
downstream fine-tuning quality while having 50\% fewer parameters and training
40\% faster compared to the widely adopted T5-base model. We publicly release
over 100 pretrained checkpoints of different T5 configurations to facilitate
future research and analysis.

    

### [[2109.10696] CC-Cert: A Probabilistic Approach to Certify General Robustness of Neural Networks](http://arxiv.org/abs/2109.10696)


  In safety-critical machine learning applications, it is crucial to defend
models against adversarial attacks -- small modifications of the input that
change the predictions. Besides rigorously studied $\ell_p$-bounded additive
perturbations, recently proposed semantic perturbations (e.g. rotation,
translation) raise a serious concern on deploying ML systems in real-world.
Therefore, it is important to provide provable guarantees for deep learning
models against semantically meaningful input transformations. In this paper, we
propose a new universal probabilistic certification approach based on
Chernoff-Cramer bounds that can be used in general attack settings. We estimate
the probability of a model to fail if the attack is sampled from a certain
distribution. Our theoretical findings are supported by experimental results on
different datasets.

    

### [[2109.10697] Towards Automatic Bias Detection in Knowledge Graphs](http://arxiv.org/abs/2109.10697)


  With the recent surge in social applications relying on knowledge graphs, the
need for techniques to ensure fairness in KG based methods is becoming
increasingly evident. Previous works have demonstrated that KGs are prone to
various social biases, and have proposed multiple methods for debiasing them.
However, in such studies, the focus has been on debiasing techniques, while the
relations to be debiased are specified manually by the user. As manual
specification is itself susceptible to human cognitive bias, there is a need
for a system capable of quantifying and exposing biases, that can support more
informed decisions on what to debias. To address this gap in the literature, we
describe a framework for identifying biases present in knowledge graph
embeddings, based on numerical bias metrics. We illustrate the framework with
three different bias measures on the task of profession prediction, and it can
be flexibly extended to further bias definitions and applications. The
relations flagged as biased can then be handed to decision makers for judgement
upon subsequent debiasing.

    

### [[2109.10736] Estimation Error Correction in Deep Reinforcement Learning for Deterministic Actor-Critic Methods](http://arxiv.org/abs/2109.10736)


  In value-based deep reinforcement learning methods, approximation of value
functions induces overestimation bias and leads to suboptimal policies. We show
that in deep actor-critic methods that aim to overcome the overestimation bias,
if the reinforcement signals received by the agent have a high variance, a
significant underestimation bias arises. To minimize the underestimation, we
introduce a parameter-free, novel deep Q-learning variant. Our Q-value update
rule combines the notions behind Clipped Double Q-learning and Maxmin
Q-learning by computing the critic objective through the nested combination of
maximum and minimum operators to bound the approximate value estimates. We
evaluate our modification on the suite of several OpenAI Gym continuous control
tasks, improving the state-of-the-art in every environment tested.

    

### [[2109.10742] Early Lane Change Prediction for Automated Driving Systems Using Multi-Task Attention-based Convolutional Neural Networks](http://arxiv.org/abs/2109.10742)


  Lane change (LC) is one of the safety-critical manoeuvres in highway driving
according to various road accident records. Thus, reliably predicting such
manoeuvre in advance is critical for the safe and comfortable operation of
automated driving systems. The majority of previous studies rely on detecting a
manoeuvre that has been already started, rather than predicting the manoeuvre
in advance. Furthermore, most of the previous works do not estimate the key
timings of the manoeuvre (e.g., crossing time), which can actually yield more
useful information for the decision making in the ego vehicle. To address these
shortcomings, this paper proposes a novel multi-task model to simultaneously
estimate the likelihood of LC manoeuvres and the time-to-lane-change (TTLC). In
both tasks, an attention-based convolutional neural network (CNN) is used as a
shared feature extractor from a bird's eye view representation of the driving
environment. The spatial attention used in the CNN model improves the feature
extraction process by focusing on the most relevant areas of the surrounding
environment. In addition, two novel curriculum learning schemes are employed to
train the proposed approach. The extensive evaluation and comparative analysis
of the proposed method in existing benchmark datasets show that the proposed
method outperforms state-of-the-art LC prediction models, particularly
considering long-term prediction performance.

    

### [[2109.10743] Natural Typing Recognition vis Surface Electromyography](http://arxiv.org/abs/2109.10743)


  By using a computer keyboard as a finger recording device, we construct the
largest existing dataset for gesture recognition via surface electromyography
(sEMG), and use deep learning to achieve over 90% character-level accuracy on
reconstructing typed text entirely from measured muscle potentials. We
prioritize the temporal structure of the EMG signal instead of the spatial
structure of the electrode layout, using network architectures inspired by
those used for real-time spoken language transcription. Our architecture
recognizes the rapid movements of natural computer typing, which occur at
irregular intervals and often overlap in time. The extensive size of our
dataset also allows us to study gesture recognition after synthetically
downgrading the spatial or temporal resolution, showing the system capabilities
necessary for real-time gesture recognition.

    

### [[2109.10757] Unsupervised Movement Detection in Indoor Positioning Systems](http://arxiv.org/abs/2109.10757)


  In recent years, the usage of indoor positioning systems for manufacturing
processes became increasingly popular. Typically, the production hall is
equipped with satellites which receive position data of sensors that can be
pinned on components, load carriers or industrial trucks. This enables a
company e.g. to reduce search efforts and to optimize individual system
processes. In our research context, a sensor only sends position information
when it is moved. However, various circumstances frequently affect that data is
undesirably sent, e.g. due to disrupting factors nearby. This has a negative
impact on the data quality, the energy consumption, and the reliability of the
whole system. Motivated by this, we aim to distinguish between actual movements
and signals that were undesirably sent which is in particular challenging due
to the susceptibility of indoor systems in terms of noise and measuring errors.
Therefore, we propose two novel unsupervised classification algorithms suitable
for this task. Depending on the question of interest, they rely either on a
distance-based or on a time-based criterion, which allows to make use of all
essential information. Furthermore, we propose an approach to combine both
classifications and to aggregate them on spatial production areas. This enables
us to generate a comprehensive map of the underlying production hall with the
sole usage of the position data. Aside from the analysis and detection of the
underlying movement structure, the user benefits from a better understanding of
own system processes and from the detection of problematic system areas which
leads to a more efficient usage of positioning systems. Since all our
approaches are constructed with unsupervised techniques, they are handily
applicable in practice and do not require more information than the output data
of the positioning system.

    

### [[2109.10765] An artificial neural network approach to bifurcating phenomena in computational fluid dynamics](http://arxiv.org/abs/2109.10765)


  This work deals with the investigation of bifurcating fluid phenomena using a
reduced order modelling setting aided by artificial neural networks. We discuss
the POD-NN approach dealing with non-smooth solutions set of nonlinear
parametrized PDEs. Thus, we study the Navier-Stokes equations describing: (i)
the Coanda effect in a channel, and (ii) the lid driven triangular cavity flow,
in a physical/geometrical multi-parametrized setting, considering the effects
of the domain's configuration on the position of the bifurcation points.
Finally, we propose a reduced manifold-based bifurcation diagram for a
non-intrusive recovery of the critical points evolution. Exploiting such
detection tool, we are able to efficiently obtain information about the pattern
flow behaviour, from symmetry breaking profiles to attaching/spreading
vortices, even at high Reynolds numbers.

    

### [[2109.10770] Exploring Adversarial Examples for Efficient Active Learning in Machine Learning Classifiers](http://arxiv.org/abs/2109.10770)


  Machine learning researchers have long noticed the phenomenon that the model
training process will be more effective and efficient when the training samples
are densely sampled around the underlying decision boundary. While this
observation has already been widely applied in a range of machine learning
security techniques, it lacks theoretical analyses of the correctness of the
observation. To address this challenge, we first add particular perturbation to
original training examples using adversarial attack methods so that the
generated examples could lie approximately on the decision boundary of the ML
classifiers. We then investigate the connections between active learning and
these particular training examples. Through analyzing various representative
classifiers such as k-NN classifiers, kernel methods as well as deep neural
networks, we establish a theoretical foundation for the observation. As a
result, our theoretical proofs provide support to more efficient active
learning methods with the help of adversarial examples, contrary to previous
works where adversarial examples are often used as destructive solutions.
Experimental results show that the established theoretical foundation will
guide better active learning strategies based on adversarial examples.

    

### [[2109.10777] Deep Variational Clustering Framework for Self-labeling of Large-scale Medical Images](http://arxiv.org/abs/2109.10777)


  We propose a Deep Variational Clustering (DVC) framework for unsupervised
representation learning and clustering of large-scale medical images. DVC
simultaneously learns the multivariate Gaussian posterior through the
probabilistic convolutional encoder and the likelihood distribution with the
probabilistic convolutional decoder; and optimizes cluster labels assignment.
Here, the learned multivariate Gaussian posterior captures the latent
distribution of a large set of unlabeled images. Then, we perform unsupervised
clustering on top of the variational latent space using a clustering loss. In
this approach, the probabilistic decoder helps to prevent the distortion of
data points in the latent space and to preserve the local structure of data
generating distribution. The training process can be considered as a
self-training process to refine the latent space and simultaneously optimizing
cluster assignments iteratively. We evaluated our proposed framework on three
public datasets that represented different medical imaging modalities. Our
experimental results show that our proposed framework generalizes better across
different datasets. It achieves compelling results on several medical imaging
benchmarks. Thus, our approach offers potential advantages over conventional
deep unsupervised learning in real-world applications. The source code of the
method and all the experiments are available publicly at:
this https URL


### [[2109.10778] Label Cleaning Multiple Instance Learning: Refining Coarse Annotations on Single Whole-Slide Images](http://arxiv.org/abs/2109.10778)


  Annotating cancerous regions in whole-slide images (WSIs) of pathology
samples plays a critical role in clinical diagnosis, biomedical research, and
machine learning algorithms development. However, generating exhaustive and
accurate annotations is labor-intensive, challenging, and costly. Drawing only
coarse and approximate annotations is a much easier task, less costly, and it
alleviates pathologists' workload. In this paper, we study the problem of
refining these approximate annotations in digital pathology to obtain more
accurate ones. Some previous works have explored obtaining machine learning
models from these inaccurate annotations, but few of them tackle the refinement
problem where the mislabeled regions should be explicitly identified and
corrected, and all of them require a - often very large - number of training
samples. We present a method, named Label Cleaning Multiple Instance Learning
(LC-MIL), to refine coarse annotations on a single WSI without the need of
external training data. Patches cropped from a WSI with inaccurate labels are
processed jointly with a MIL framework, and a deep-attention mechanism is
leveraged to discriminate mislabeled instances, mitigating their impact on the
predictive model and refining the segmentation. Our experiments on a
heterogeneous WSI set with breast cancer lymph node metastasis, liver cancer,
and colorectal cancer samples show that LC-MIL significantly refines the coarse
annotations, outperforming the state-of-the-art alternatives, even while
learning from a single slide. These results demonstrate the LC-MIL is a
promising, lightweight tool to provide fine-grained annotations from coarsely
annotated pathology sets.

    

### [[2109.10781] Introducing Symmetries to Black Box Meta Reinforcement Learning](http://arxiv.org/abs/2109.10781)


  Meta reinforcement learning (RL) attempts to discover new RL algorithms
automatically from environment interaction. In so-called black-box approaches,
the policy and the learning algorithm are jointly represented by a single
neural network. These methods are very flexible, but they tend to underperform
in terms of generalisation to new, unseen environments. In this paper, we
explore the role of symmetries in meta-generalisation. We show that a recent
successful meta RL approach that meta-learns an objective for
backpropagation-based learning exhibits certain symmetries (specifically the
reuse of the learning rule, and invariance to input and output permutations)
that are not present in typical black-box meta RL systems. We hypothesise that
these symmetries can play an important role in meta-generalisation. Building
off recent work in black-box supervised meta learning, we develop a black-box
meta RL system that exhibits these same symmetries. We show through careful
experimentation that incorporating these symmetries can lead to algorithms with
a greater ability to generalise to unseen action & observation spaces, tasks,
and environments.

    

### [[2109.10793] Physics-informed Neural Networks-based Model Predictive Control for Multi-link Manipulators](http://arxiv.org/abs/2109.10793)


  We discuss nonlinear model predictive control (NMPC) for multi-body dynamics
via physics-informed machine learning methods. Physics-informed neural networks
(PINNs) are a promising tool to approximate (partial) differential equations.
PINNs are not suited for control tasks in their original form since they are
not designed to handle variable control actions or variable initial values. We
thus present the idea of enhancing PINNs by adding control actions and initial
conditions as additional network inputs. The high-dimensional input space is
subsequently reduced via a sampling strategy and a zero-hold assumption. This
strategy enables the controller design based on a PINN as an approximation of
the underlying system dynamics. The additional benefit is that the
sensitivities are easily computed via automatic differentiation, thus leading
to efficient gradient-based algorithms. Finally, we present our results using
our PINN-based MPC to solve a tracking problem for a complex mechanical system,
a multi-link manipulator.

    

### [[2109.10794] Entropic Issues in Likelihood-Based OOD Detection](http://arxiv.org/abs/2109.10794)


  Deep generative models trained by maximum likelihood remain very popular
methods for reasoning about data probabilistically. However, it has been
observed that they can assign higher likelihoods to out-of-distribution (OOD)
data than in-distribution data, thus calling into question the meaning of these
likelihood values. In this work we provide a novel perspective on this
phenomenon, decomposing the average likelihood into a KL divergence term and an
entropy term. We argue that the latter can explain the curious OOD behaviour
mentioned above, suppressing likelihood values on datasets with higher entropy.
Although our idea is simple, we have not seen it explored yet in the
literature. This analysis provides further explanation for the success of OOD
detection methods based on likelihood ratios, as the problematic entropy term
cancels out in expectation. Finally, we discuss how this observation relates to
recent success in OOD detection with manifold-supported models, for which the
above decomposition does not hold.

    

### [[2109.10795] Neural network relief: a pruning algorithm based on neural activity](http://arxiv.org/abs/2109.10795)


  Current deep neural networks (DNNs) are overparameterized and use most of
their neuronal connections during inference for each task. The human brain,
however, developed specialized regions for different tasks and performs
inference with a small fraction of its neuronal connections. We propose an
iterative pruning strategy introducing a simple importance-score metric that
deactivates unimportant connections, tackling overparameterization in DNNs and
modulating the firing patterns. The aim is to find the smallest number of
connections that is still capable of solving a given task with comparable
accuracy, i.e. a simpler subnetwork. We achieve comparable performance for
LeNet architectures on MNIST, and significantly higher parameter compression
than state-of-the-art algorithms for VGG and ResNet architectures on
CIFAR-10/100 and Tiny-ImageNet. Our approach also performs well for the two
different optimizers considered -- Adam and SGD. The algorithm is not designed
to minimize FLOPs when considering current hardware and software
implementations, although it performs reasonably when compared to the state of
the art.

    

### [[2109.10797] Improved Multi-label Classification with Frequent Label-set Mining and Association](http://arxiv.org/abs/2109.10797)


  Multi-label (ML) data deals with multiple classes associated with individual
samples at the same time. This leads to the co-occurrence of several classes
repeatedly, which indicates some existing correlation among them. In this
article, the correlation among classes has been explored to improve the
classification performance of existing ML classifiers. A novel approach of
frequent label-set mining has been proposed to extract these correlated classes
from the label-sets of the data. Both co-presence (CP) and co-absence (CA) of
classes have been taken into consideration. The rules mined from the ML data
has been further used to incorporate class correlation information into
existing ML classifiers. The soft scores generated by an ML classifier are
modified through a novel approach using the CP-CA rules. A concept of certain
and uncertain scores has been defined here, where the proposed method aims to
improve the uncertain scores with the help of the certain scores and their
corresponding CP-CA rules. This has been experimentally analysed on ten ML
datasets for three ML existing classifiers which shows substantial improvement
in their overall performance.

    

### [[2109.10803] Multi-Slice Clustering for 3-order Tensor Data](http://arxiv.org/abs/2109.10803)


  Several methods of triclustering of three dimensional data require the
specification of the cluster size in each dimension. This introduces a certain
degree of arbitrariness. To address this issue, we propose a new method, namely
the multi-slice clustering (MSC) for a 3-order tensor data set. We analyse, in
each dimension or tensor mode, the spectral decomposition of each tensor slice,
i.e. a matrix. Thus, we define a similarity measure between matrix slices up to
a threshold (precision) parameter, and from that, identify a cluster. The
intersection of all partial clusters provides the desired triclustering. The
effectiveness of our algorithm is shown on both synthetic and real-world data
sets.

    

### [[2109.10813] A Workflow for Offline Model-Free Robotic Reinforcement Learning](http://arxiv.org/abs/2109.10813)


  Offline reinforcement learning (RL) enables learning control policies by
utilizing only prior experience, without any online interaction. This can allow
robots to acquire generalizable skills from large and diverse datasets, without
any costly or unsafe online data collection. Despite recent algorithmic
advances in offline RL, applying these methods to real-world problems has
proven challenging. Although offline RL methods can learn from prior data,
there is no clear and well-understood process for making various design
choices, from model architecture to algorithm hyperparameters, without actually
evaluating the learned policies online. In this paper, our aim is to develop a
practical workflow for using offline RL analogous to the relatively
well-understood workflows for supervised learning problems. To this end, we
devise a set of metrics and conditions that can be tracked over the course of
offline training, and can inform the practitioner about how the algorithm and
model architecture should be adjusted to improve final performance. Our
workflow is derived from a conceptual understanding of the behavior of
conservative offline RL algorithms and cross-validation in supervised learning.
We demonstrate the efficacy of this workflow in producing effective policies
without any online tuning, both in several simulated robotic learning scenarios
and for three tasks on two distinct real robots, focusing on learning
manipulation skills with raw image observations with sparse binary rewards.
Explanatory video and additional results can be found at
this http URL


### [[2109.10817] Causal Inference in Non-linear Time-series usingDeep Networks and Knockoff Counterfactuals](http://arxiv.org/abs/2109.10817)


  Estimating causal relations is vital in understanding the complex
interactions in multivariate time series. Non-linear coupling of variables is
one of the major challenges inaccurate estimation of cause-effect relations. In
this paper, we propose to use deep autoregressive networks (DeepAR) in tandem
with counterfactual analysis to infer nonlinear causal relations in
multivariate time series. We extend the concept of Granger causality using
probabilistic forecasting with DeepAR. Since deep networks can neither handle
missing input nor out-of-distribution intervention, we propose to use the
Knockoffs framework (Barberand Cand`es, 2015) for generating intervention
variables and consequently counterfactual probabilistic forecasting. Knockoff
samples are independent of their output given the observed variables and
exchangeable with their counterpart variables without changing the underlying
distribution of the data. We test our method on synthetic as well as real-world
time series datasets. Overall our method outperforms the widely used vector
autoregressive Granger causality and PCMCI in detecting nonlinear causal
dependency in multivariate time series.

    

### [[2109.10824] Learning by Examples Based on Multi-level Optimization](http://arxiv.org/abs/2109.10824)


  Learning by examples, which learns to solve a new problem by looking into how
similar problems are solved, is an effective learning method in human learning.
When a student learns a new topic, he/she finds out exemplar topics that are
similar to this new topic and studies the exemplar topics to deepen the
understanding of the new topic. We aim to investigate whether this powerful
learning skill can be borrowed from humans to improve machine learning as well.
In this work, we propose a novel learning approach called Learning By Examples
(LBE). Our approach automatically retrieves a set of training examples that are
similar to query examples and predicts labels for query examples by using class
labels of the retrieved examples. We propose a three-level optimization
framework to formulate LBE which involves three stages of learning: learning a
Siamese network to retrieve similar examples; learning a matching network to
make predictions on query examples by leveraging class labels of retrieved
similar examples; learning the ``ground-truth'' similarities between training
examples by minimizing the validation loss. We develop an efficient algorithm
to solve the LBE problem and conduct extensive experiments on various
benchmarks where the results demonstrate the effectiveness of our method on
both supervised and few-shot learning.

    

### [[2109.10834] SCSS-Net: Solar Corona Structures Segmentation by Deep Learning](http://arxiv.org/abs/2109.10834)


  Structures in the solar corona are the main drivers of space weather
processes that might directly or indirectly affect the Earth. Thanks to the
most recent space-based solar observatories, with capabilities to acquire
high-resolution images continuously, the structures in the solar corona can be
monitored over the years with a time resolution of minutes. For this purpose,
we have developed a method for automatic segmentation of solar corona
structures observed in EUV spectrum that is based on a deep learning approach
utilizing Convolutional Neural Networks. The available input datasets have been
examined together with our own dataset based on the manual annotation of the
target structures. Indeed, the input dataset is the main limitation of the
developed model's performance. Our \textit{SCSS-Net} model provides results for
coronal holes and active regions that could be compared with other generally
used methods for automatic segmentation. Even more, it provides a universal
procedure to identify structures in the solar corona with the help of the
transfer learning technique. The outputs of the model can be then used for
further statistical studies of connections between solar activity and the
influence of space weather on Earth.

    

### [[2109.10847] Small-Bench NLP: Benchmark for small single GPU trained models in Natural Language Processing](http://arxiv.org/abs/2109.10847)


  Recent progress in the Natural Language Processing domain has given us
several State-of-the-Art (SOTA) pretrained models which can be finetuned for
specific tasks. These large models with billions of parameters trained on
numerous GPUs/TPUs over weeks are leading in the benchmark leaderboards. In
this paper, we discuss the need for a benchmark for cost and time effective
smaller models trained on a single GPU. This will enable researchers with
resource constraints experiment with novel and innovative ideas on
tokenization, pretraining tasks, architecture, fine tuning methods etc. We set
up Small-Bench NLP, a benchmark for small efficient neural language models
trained on a single GPU. Small-Bench NLP benchmark comprises of eight NLP tasks
on the publicly available GLUE datasets and a leaderboard to track the progress
of the community. Our ELECTRA-DeBERTa (15M parameters) small model architecture
achieves an average score of 81.53 which is comparable to that of BERT-Base's
82.20 (110M parameters). Our models, code and leaderboard are available at
this https URL


### [[2109.10852] Pix2seq: A Language Modeling Framework for Object Detection](http://arxiv.org/abs/2109.10852)


  This paper presents Pix2Seq, a simple and generic framework for object
detection. Unlike existing approaches that explicitly integrate prior knowledge
about the task, we simply cast object detection as a language modeling task
conditioned on the observed pixel inputs. Object descriptions (e.g., bounding
boxes and class labels) are expressed as sequences of discrete tokens, and we
train a neural net to perceive the image and generate the desired sequence. Our
approach is based mainly on the intuition that if a neural net knows about
where and what the objects are, we just need to teach it how to read them out.
Beyond the use of task-specific data augmentations, our approach makes minimal
assumptions about the task, yet it achieves competitive results on the
challenging COCO dataset, compared to highly specialized and well optimized
detection algorithms.

    

### [[2109.10854] Imitation Learning of Stabilizing Policies for Nonlinear Systems](http://arxiv.org/abs/2109.10854)


  There has been a recent interest in imitation learning methods that are
guaranteed to produce a stabilizing control law with respect to a known system.
Work in this area has generally considered linear systems and controllers, for
which stabilizing imitation learning takes the form of a biconvex optimization
problem. In this paper it is demonstrated that the same methods developed for
linear systems and controllers can be readily extended to polynomial systems
and controllers using sum of squares techniques. A projected gradient descent
algorithm and an alternating direction method of multipliers algorithm are
proposed as heuristics for solving the stabilizing imitation learning problem,
and their performance is illustrated through numerical experiments.

    

### [[2109.10855] BFClass: A Backdoor-free Text Classification Framework](http://arxiv.org/abs/2109.10855)


  Backdoor attack introduces artificial vulnerabilities into the model by
poisoning a subset of the training data via injecting triggers and modifying
labels. Various trigger design strategies have been explored to attack text
classifiers, however, defending such attacks remains an open problem. In this
work, we propose BFClass, a novel efficient backdoor-free training framework
for text classification. The backbone of BFClass is a pre-trained discriminator
that predicts whether each token in the corrupted input was replaced by a
masked language model. To identify triggers, we utilize this discriminator to
locate the most suspicious token from each training sample and then distill a
concise set by considering their association strengths with particular labels.
To recognize the poisoned subset, we examine the training samples with these
identified triggers as the most suspicious token, and check if removing the
trigger will change the poisoned model's prediction. Extensive experiments
demonstrate that BFClass can identify all the triggers, remove 95% poisoned
training samples with very limited false alarms, and achieve almost the same
performance as the models trained on the benign training data.

    

### [[2109.10856] Coarse2Fine: Fine-grained Text Classification on Coarsely-grained Annotated Data](http://arxiv.org/abs/2109.10856)


  Existing text classification methods mainly focus on a fixed label set,
whereas many real-world applications require extending to new fine-grained
classes as the number of samples per label increases. To accommodate such
requirements, we introduce a new problem called coarse-to-fine grained
classification, which aims to perform fine-grained classification on coarsely
annotated data. Instead of asking for new fine-grained human annotations, we
opt to leverage label surface names as the only human guidance and weave in
rich pre-trained generative language models into the iterative weak supervision
strategy. Specifically, we first propose a label-conditioned finetuning
formulation to attune these generators for our task. Furthermore, we devise a
regularization objective based on the coarse-fine label constraints derived
from our problem setting, giving us even further improvements over the prior
formulation. Our framework uses the fine-tuned generative models to sample
pseudo-training data for training the classifier, and bootstraps on real
unlabeled data for model refinement. Extensive experiments and case studies on
two real-world datasets demonstrate superior performance over SOTA zero-shot
classification baselines.

    

### [[2109.10862] Recursively Summarizing Books with Human Feedback](http://arxiv.org/abs/2109.10862)


  A major challenge for scaling machine learning is training models to perform
tasks that are very difficult or time-consuming for humans to evaluate. We
present progress on this problem on the task of abstractive summarization of
entire fiction novels. Our method combines learning from human feedback with
recursive task decomposition: we use models trained on smaller parts of the
task to assist humans in giving feedback on the broader task. We collect a
large volume of demonstrations and comparisons from human labelers, and
fine-tune GPT-3 using behavioral cloning and reward modeling to do
summarization recursively. At inference time, the model first summarizes small
sections of the book and then recursively summarizes these summaries to produce
a summary of the entire book. Our human labelers are able to supervise and
evaluate the models quickly, despite not having read the entire books
themselves. Our resulting model generates sensible summaries of entire books,
even matching the quality of human-written summaries in a few cases ($\sim5\%$
of books). We achieve state-of-the-art results on the recent BookSum dataset
for book-length summarization. A zero-shot question-answering model using these
summaries achieves state-of-the-art results on the challenging NarrativeQA
benchmark for answering questions about books and movie scripts. We release
datasets of samples from our model.

    

### [[2109.10870] SoK: Machine Learning Governance](http://arxiv.org/abs/2109.10870)


  The application of machine learning (ML) in computer systems introduces not
only many benefits but also risks to society. In this paper, we develop the
concept of ML governance to balance such benefits and risks, with the aim of
achieving responsible applications of ML. Our approach first systematizes
research towards ascertaining ownership of data and models, thus fostering a
notion of identity specific to ML systems. Building on this foundation, we use
identities to hold principals accountable for failures of ML systems through
both attribution and auditing. To increase trust in ML systems, we then survey
techniques for developing assurance, i.e., confidence that the system meets its
security requirements and does not exhibit certain known failures. This leads
us to highlight the need for techniques that allow a model owner to manage the
life cycle of their system, e.g., to patch or retire their ML system. Put
altogether, our systematization of knowledge standardizes the interactions
between principals involved in the deployment of ML throughout its life cycle.
We highlight opportunities for future work, e.g., to formalize the resulting
game between ML principals.

    

### [[2109.10888] Quantifying Model Predictive Uncertainty with Perturbation Theory](http://arxiv.org/abs/2109.10888)


  We propose a framework for predictive uncertainty quantification of a neural
network that replaces the conventional Bayesian notion of weight probability
density function (PDF) with a physics based potential field representation of
the model weights in a Gaussian reproducing kernel Hilbert space (RKHS)
embedding. This allows us to use perturbation theory from quantum physics to
formulate a moment decomposition problem over the model weight-output
relationship. The extracted moments reveal successive degrees of regularization
of the weight potential field around the local neighborhood of the model
output. Such localized moments represent well the PDF tails and provide
significantly greater accuracy of the model's predictive uncertainty than the
central moments characterized by Bayesian and ensemble methods or their
variants. We show that this consequently leads to a better ability to detect
false model predictions of test data that has undergone a covariate shift away
from the training PDF learned by the model. We evaluate our approach against
baseline uncertainty quantification methods on several benchmark datasets that
are corrupted using common distortion techniques. Our approach provides fast
model predictive uncertainty estimates with much greater precision and
calibration.

    

### [[2109.10895] Geo-Context Aware Study of Vision-Based Autonomous Driving Models and Spatial Video Data](http://arxiv.org/abs/2109.10895)


  Vision-based deep learning (DL) methods have made great progress in learning
autonomous driving models from large-scale crowd-sourced video datasets. They
are trained to predict instantaneous driving behaviors from video data captured
by on-vehicle cameras. In this paper, we develop a geo-context aware
visualization system for the study of Autonomous Driving Model (ADM)
predictions together with large-scale ADM video data. The visual study is
seamlessly integrated with the geographical environment by combining DL model
performance with geospatial visualization techniques. Model performance
measures can be studied together with a set of geospatial attributes over map
views. Users can also discover and compare prediction behaviors of multiple DL
models in both city-wide and street-level analysis, together with road images
and video contents. Therefore, the system provides a new visual exploration
platform for DL model designers in autonomous driving. Use cases and domain
expert evaluation show the utility and effectiveness of the visualization
system.

    

### [[2109.10896] Updating Embeddings for Dynamic Knowledge Graphs](http://arxiv.org/abs/2109.10896)


  Data in Knowledge Graphs often represents part of the current state of the
real world. Thus, to stay up-to-date the graph data needs to be updated
frequently. To utilize information from Knowledge Graphs, many state-of-the-art
machine learning approaches use embedding techniques. These techniques
typically compute an embedding, i.e., vector representations of the nodes as
input for the main machine learning algorithm. If a graph update occurs later
on -- specifically when nodes are added or removed -- the training has to be
done all over again. This is undesirable, because of the time it takes and also
because downstream models which were trained with these embeddings have to be
retrained if they change significantly. In this paper, we investigate embedding
updates that do not require full retraining and evaluate them in combination
with various embedding models on real dynamic Knowledge Graphs covering
multiple use cases. We study approaches that place newly appearing nodes
optimally according to local information, but notice that this does not work
well. However, we find that if we continue the training of the old embedding,
interleaved with epochs during which we only optimize for the added and removed
parts, we obtain good results in terms of typical metrics used in link
prediction. This performance is obtained much faster than with a complete
retraining and hence makes it possible to maintain embeddings for dynamic
Knowledge Graphs.

    

### [[2109.10898] A Robust Asymmetric Kernel Function for Bayesian Optimization, with Application to Image Defect Detection in Manufacturing Systems](http://arxiv.org/abs/2109.10898)


  Some response surface functions in complex engineering systems are usually
highly nonlinear, unformed, and expensive-to-evaluate. To tackle this
challenge, Bayesian optimization, which conducts sequential design via a
posterior distribution over the objective function, is a critical method used
to find the global optimum of black-box functions. Kernel functions play an
important role in shaping the posterior distribution of the estimated function.
The widely used kernel function, e.g., radial basis function (RBF), is very
vulnerable and susceptible to outliers; the existence of outliers is causing
its Gaussian process surrogate model to be sporadic. In this paper, we propose
a robust kernel function, Asymmetric Elastic Net Radial Basis Function
(AEN-RBF). Its validity as a kernel function and computational complexity are
evaluated. When compared to the baseline RBF kernel, we prove theoretically
that AEN-RBF can realize smaller mean squared prediction error under mild
conditions. The proposed AEN-RBF kernel function can also realize faster
convergence to the global optimum. We also show that the AEN-RBF kernel
function is less sensitive to outliers, and hence improves the robustness of
the corresponding Bayesian optimization with Gaussian processes. Through
extensive evaluations carried out on synthetic and real-world optimization
problems, we show that AEN-RBF outperforms existing benchmark kernel functions.

    

### [[1906.06427] Real-Time Privacy-Preserving Data Release for Smart Meters](http://arxiv.org/abs/1906.06427)


  Smart Meters (SMs) are a fundamental component of smart grids, but they carry
sensitive information about users such as occupancy status of houses and
therefore, they have raised serious concerns about leakage of consumers'
private information. In particular, we focus on real-time privacy threats,
i.e., potential attackers that try to infer sensitive data from SMs reported
data in an online fashion. We adopt an information-theoretic privacy measure
and show that it effectively limits the performance of any real-time attacker.
Using this privacy measure, we propose a general formulation to design a
privatization mechanism that can provide a target level of privacy by adding a
minimal amount of distortion to the SMs measurements. On the other hand, to
cope with different applications, a flexible distortion measure is considered.
This formulation leads to a general loss function, which is optimized using a
deep learning adversarial framework, where two neural networks $-$ referred to
as the releaser and the adversary $-$ are trained with opposite goals. An
exhaustive empirical study is then performed to validate the performances of
the proposed approach for the occupancy detection privacy problem, assuming the
attacker disposes of either limited or full access to the training dataset.

    

### [[1906.08823] Cross-Subject Statistical Shift Estimation for Generalized Electroencephalography-based Mental Workload Assessment](http://arxiv.org/abs/1906.08823)


  Assessment of mental workload in real-world conditions is key to ensure the
performance of workers executing tasks that demand sustained attention.
Previous literature has employed electroencephalography (EEG) to this end
despite having observed that EEG correlates of mental workload vary across
subjects and physical strain, thus making it difficult to devise models capable
of simultaneously presenting reliable performance across users. Domain
adaptation consists of a set of strategies that aim at allowing for improving
machine learning systems performance on unseen data at training time. Such
methods, however, might rely on assumptions over the considered data
distributions, which typically do not hold for applications of EEG data.
Motivated by this observation, in this work we propose a strategy to estimate
two types of discrepancies between multiple data distributions, namely marginal
and conditional shifts, observed on data collected from different subjects.
Besides shedding light on the assumptions that hold for a particular dataset,
the estimates of statistical shifts obtained with the proposed approach can be
used for investigating other aspects of a machine learning pipeline, such as
quantitatively assessing the effectiveness of domain adaptation strategies. In
particular, we consider EEG data collected from individuals performing mental
tasks while running on a treadmill and pedaling on a stationary bike and
explore the effects of different normalization strategies commonly used to
mitigate cross-subject variability. We show the effects that different
normalization schemes have on statistical shifts and their relationship with
the accuracy of mental workload prediction as assessed on unseen participants
at training time.

    

### [[1908.05715] Automated classification of plasma regions using 3D particle energy distributions](http://arxiv.org/abs/1908.05715)


  We investigate the properties of the ion sky maps produced by the Dual Ion
Spectrometers (DIS) from the Fast Plasma Investigation (FPI). We have trained a
convolutional neural network classifier to predict four regions crossed by the
MMS on the dayside magnetosphere: solar wind, ion foreshock, magnetosheath, and
magnetopause using solely DIS spectrograms. The accuracy of the classifier is
>98%. We use the classifier to detect mixed plasma regions, in particular to
find the bow shock regions. A similar approach can be used to identify the
magnetopause crossings and reveal regions prone to magnetic reconnection. Data
processing through the trained classifier is fast and efficient and thus can be
used for classification for the whole MMS database.

    

### [[1910.08880] Improved error rates for sparse (group) learning with Lipschitz loss functions](http://arxiv.org/abs/1910.08880)


  We study a family of sparse estimators defined as minimizers of some
empirical Lipschitz loss function -- which include the hinge loss, the logistic
loss and the quantile regression loss -- with a convex, sparse or group-sparse
regularization. In particular, we consider the L1 norm on the coefficients, its
sorted Slope version, and the Group L1-L2 extension. We propose a new
theoretical framework that uses common assumptions in the literature to
simultaneously derive new high-dimensional L2 estimation upper bounds for all
three regularization schemes. %, and to improve over existing results. For L1
and Slope regularizations, our bounds scale as $(k^*/n) \log(p/k^*)$ --
$n\times p$ is the size of the design matrix and $k^*$ the dimension of the
theoretical loss minimizer $\B{\beta}^*$ -- and match the optimal minimax rate
achieved for the least-squares case. For Group L1-L2 regularization, our bounds
scale as $(s^*/n) \log\left( G / s^* \right) + m^* / n$ -- $G$ is the total
number of groups and $m^*$ the number of coefficients in the $s^*$ groups which
contain $\B{\beta}^*$ -- and improve over the least-squares case. We show that,
when the signal is strongly group-sparse, Group L1-L2 is superior to L1 and
Slope. In addition, we adapt our approach to the sub-Gaussian linear regression
framework and reach the optimal minimax rate for Lasso, and an improved rate
for Group-Lasso. Finally, we release an accelerated proximal algorithm that
computes the nine main convex estimators of interest when the number of
variables is of the order of $100,000s$.

    

### [[1912.09526] Inference for Hit Enrichment Curves, with Applications to Drug Discovery](http://arxiv.org/abs/1912.09526)


  In virtual screening for drug discovery, hit enrichment curves are widely
used to assess the performance of ranking algorithms with regard to their
ability to identify early enrichment. Unfortunately, researchers almost never
consider the uncertainty associated with estimating such curves before
declaring differences between performance of competing algorithms. Appropriate
inference is complicated by two sources of correlation that are often
overlooked: correlation across different testing fractions within a single
algorithm, and correlation between competing algorithms. Additionally,
researchers are often interested in making comparisons along the entire curve,
not only at a few testing fractions. We develop inferential procedures to
address both the needs of those interested in a few testing fractions, as well
as those interested in the entire curve. For the former, four hypothesis
testing and (pointwise) confidence intervals are investigated, and a newly
developed EmProc approach is found to be most effective. For inference along
entire curves, EmProc-based confidence bands are recommended for simultaneous
coverage and minimal width. Our inferential procedures trivially extend to
enrichment factors, as well.

    

### [[2002.10631] Batch norm with entropic regularization turns deterministic autoencoders into generative models](http://arxiv.org/abs/2002.10631)


  The variational autoencoder is a well defined deep generative model that
utilizes an encoder-decoder framework where an encoding neural network outputs
a non-deterministic code for reconstructing an input. The encoder achieves this
by sampling from a distribution for every input, instead of outputting a
deterministic code per input. The great advantage of this process is that it
allows the use of the network as a generative model for sampling from the data
distribution beyond provided samples for training. We show in this work that
utilizing batch normalization as a source for non-determinism suffices to turn
deterministic autoencoders into generative models on par with variational ones,
so long as we add a suitable entropic regularization to the training objective.

    

### [[2004.06383] Extending Adversarial Attacks to Produce Adversarial Class Probability Distributions](http://arxiv.org/abs/2004.06383)


  Despite the remarkable performance and generalization levels of deep learning
models in a wide range of artificial intelligence tasks, it has been
demonstrated that these models can be easily fooled by the addition of
imperceptible yet malicious perturbations to natural inputs. These altered
inputs are known in the literature as adversarial examples. In this paper, we
propose a novel probabilistic framework to generalize and extend adversarial
attacks in order to produce a desired probability distribution for the classes
when we apply the attack method to a large number of inputs. This novel attack
strategy provides the attacker with greater control over the target model, and
increases the complexity of detecting that the model is being systematically
attacked. We introduce four different strategies to efficiently generate such
attacks, and illustrate our approach by extending multiple adversarial attack
algorithms. We also experimentally validate our approach for the spoken command
classification task, an exemplary machine learning problem in the audio domain.
Our results demonstrate that we can closely approximate any probability
distribution for the classes while maintaining a high fooling rate and by
injecting imperceptible perturbations to the inputs.

    

### [[2006.01738] Jointly Learning Environments and Control Policies with Projected Stochastic Gradient Ascent](http://arxiv.org/abs/2006.01738)


  We consider the joint design and control of discrete-time stochastic
dynamical systems over a finite time horizon. We formulate the problem as a
multi-step optimization problem under uncertainty seeking to identify a system
design and a control policy that jointly maximize the expected sum of rewards
collected over the time horizon considered. The transition function, the reward
function and the policy are all parametrized, assumed known and differentiable
with respect to their parameters. We then introduce a deep reinforcement
learning algorithm combining policy gradient methods with model-based
optimization techniques to solve this problem. In essence, our algorithm
iteratively approximates the gradient of the expected return via Monte-Carlo
sampling and automatic differentiation and takes projected gradient ascent
steps in the space of environment and policy parameters. This algorithm is
referred to as Direct Environment and Policy Search (DEPS). We assess the
performance of our algorithm in three environments concerned with the design
and control of a mass-spring-damper system, a small-scale off-grid power system
and a drone, respectively. In addition, our algorithm is benchmarked against a
state-of-the-art deep reinforcement learning algorithm used to tackle joint
design and control problems. We show that DEPS performs at least as well or
better in all three environments, consistently yielding solutions with higher
returns in fewer iterations. Finally, solutions produced by our algorithm are
also compared with solutions produced by an algorithm that does not jointly
optimize environment and policy parameters, highlighting the fact that higher
returns can be achieved when joint optimization is performed.

    

### [[2006.05826] Transient Non-Stationarity and Generalisation in Deep Reinforcement Learning](http://arxiv.org/abs/2006.05826)


  Non-stationarity can arise in Reinforcement Learning (RL) even in stationary
environments. For example, most RL algorithms collect new data throughout
training, using a non-stationary behaviour policy. Due to the transience of
this non-stationarity, it is often not explicitly addressed in deep RL and a
single neural network is continually updated. However, we find evidence that
neural networks exhibit a memory effect where these transient
non-stationarities can permanently impact the latent representation and
adversely affect generalisation performance. Consequently, to improve
generalisation of deep RL agents, we propose Iterated Relearning (ITER). ITER
augments standard RL training by repeated knowledge transfer of the current
policy into a freshly initialised network, which thereby experiences less
non-stationarity during training. Experimentally, we show that ITER improves
performance on the challenging generalisation benchmarks ProcGen and Multiroom.

    

### [[2007.08970] Compositional Generalization in Semantic Parsing: Pre-training vs. Specialized Architectures](http://arxiv.org/abs/2007.08970)


  While mainstream machine learning methods are known to have limited ability
to compositionally generalize, new architectures and techniques continue to be
proposed to address this limitation. We investigate state-of-the-art techniques
and architectures in order to assess their effectiveness in improving
compositional generalization in semantic parsing tasks based on the SCAN and
CFQ datasets. We show that masked language model (MLM) pre-training rivals
SCAN-inspired architectures on primitive holdout splits. On a more complex
compositional task, we show that pre-training leads to significant improvements
in performance vs. comparable non-pre-trained models, whereas architectures
proposed to encourage compositional generalization on SCAN or in the area of
algorithm learning fail to lead to significant improvements. We establish a new
state of the art on the CFQ compositional generalization benchmark using MLM
pre-training together with an intermediate representation.

    

### [[2007.12911] Tighter risk certificates for neural networks](http://arxiv.org/abs/2007.12911)


  This paper presents an empirical study regarding training probabilistic
neural networks using training objectives derived from PAC-Bayes bounds. In the
context of probabilistic neural networks, the output of training is a
probability distribution over network weights. We present two training
objectives, used here for the first time in connection with training neural
networks. These two training objectives are derived from tight PAC-Bayes
bounds. We also re-implement a previously used training objective based on a
classical PAC-Bayes bound, to compare the properties of the predictors learned
using the different training objectives. We compute risk certificates for the
learnt predictors, based on part of the data used to learn the predictors. We
further experiment with different types of priors on the weights (both
data-free and data-dependent priors) and neural network architectures. Our
experiments on MNIST and CIFAR-10 show that our training methods produce
competitive test set errors and non-vacuous risk bounds with much tighter
values than previous results in the literature, showing promise not only to
guide the learning algorithm through bounding the risk but also for model
selection. These observations suggest that the methods studied here might be
good candidates for self-certified learning, in the sense of using the whole
data set for learning a predictor and certifying its risk on any unseen data
(from the same distribution as the training data) potentially without the need
for holding out test data.

    

### [[2008.10087] Blindness of score-based methods to isolated components and mixing proportions](http://arxiv.org/abs/2008.10087)


  Statistical tasks such as density estimation and approximate Bayesian
inference often involve densities with unknown normalising constants.
Score-based methods, including score matching, are popular techniques as they
are free of normalising constants. Although these methods enjoy theoretical
guarantees, a little-known fact is that they suffer from practical failure
modes when the unnormalised distribution of interest has isolated components --
they cannot discover isolated components or identify the correct mixing
proportions between components. We demonstrate these findings using simple
distributions and present heuristic attempts to address these issues. We hope
to bring the attention of theoreticians and practitioners to these issues when
developing new algorithms and applications.

    

### [[2010.11773] On Resource-Efficient Bayesian Network Classifiers and Deep Neural Networks](http://arxiv.org/abs/2010.11773)


  We present two methods to reduce the complexity of Bayesian network (BN)
classifiers. First, we introduce quantization-aware training using the
straight-through gradient estimator to quantize the parameters of BNs to few
bits. Second, we extend a recently proposed differentiable tree-augmented naive
Bayes (TAN) structure learning approach by also considering the model size.
Both methods are motivated by recent developments in the deep learning
community, and they provide effective means to trade off between model size and
prediction accuracy, which is demonstrated in extensive experiments.
Furthermore, we contrast quantized BN classifiers with quantized deep neural
networks (DNNs) for small-scale scenarios which have hardly been investigated
in the literature. We show Pareto optimal models with respect to model size,
number of operations, and test error and find that both model classes are
viable options.

    

### [[2011.03186] Revisiting Model-Agnostic Private Learning: Faster Rates and Active Learning](http://arxiv.org/abs/2011.03186)


  The Private Aggregation of Teacher Ensembles (PATE) framework is one of the
most promising recent approaches in differentially private learning. Existing
theoretical analysis shows that PATE consistently learns any VC-classes in the
realizable setting, but falls short in explaining its success in more general
cases where the error rate of the optimal classifier is bounded away from zero.
We fill in this gap by introducing the Tsybakov Noise Condition (TNC) and
establish stronger and more interpretable learning bounds. These bounds provide
new insights into when PATE works and improve over existing results even in the
narrower realizable setting. We also investigate the compelling idea of using
active learning for saving privacy budget, and empirical studies show the
effectiveness of this new idea. The novel components in the proofs include a
more refined analysis of the majority voting classifier -- which could be of
independent interest -- and an observation that the synthetic "student"
learning problem is nearly realizable by construction under the Tsybakov noise
condition.

    

### [[2011.08558] On the Transferability of Adversarial Attacksagainst Neural Text Classifier](http://arxiv.org/abs/2011.08558)


  Deep neural networks are vulnerable to adversarial attacks, where a small
perturbation to an input alters the model prediction. In many cases, malicious
inputs intentionally crafted for one model can fool another model. In this
paper, we present the first study to systematically investigate the
transferability of adversarial examples for text classification models and
explore how various factors, including network architecture, tokenization
scheme, word embedding, and model capacity, affect the transferability of
adversarial examples. Based on these studies, we propose a genetic algorithm to
find an ensemble of models that can be used to induce adversarial examples to
fool almost all existing models. Such adversarial examples reflect the defects
of the learning process and the data bias in the training set. Finally, we
derive word replacement rules that can be used for model diagnostics from these
adversarial examples.

    

### [[2012.07483] On the Treatment of Optimization Problems with L1 Penalty Terms via Multiobjective Continuation](http://arxiv.org/abs/2012.07483)


  We present a novel algorithm that allows us to gain detailed insight into the
effects of sparsity in linear and nonlinear optimization, which is of great
importance in many scientific areas such as image and signal processing,
medical imaging, compressed sensing, and machine learning (e.g., for the
training of neural networks). Sparsity is an important feature to ensure
robustness against noisy data, but also to find models that are interpretable
and easy to analyze due to the small number of relevant terms. It is common
practice to enforce sparsity by adding the $\ell_1$-norm as a weighted penalty
term. In order to gain a better understanding and to allow for an informed
model selection, we directly solve the corresponding multiobjective
optimization problem (MOP) that arises when we minimize the main objective and
the $\ell_1$-norm simultaneously. As this MOP is in general non-convex for
nonlinear objectives, the weighting method will fail to provide all optimal
compromises. To avoid this issue, we present a continuation method which is
specifically tailored to MOPs with two objective functions one of which is the
$\ell_1$-norm. Our method can be seen as a generalization of well-known
homotopy methods for linear regression problems to the nonlinear case. Several
numerical examples - including neural network training - demonstrate our
theoretical findings and the additional insight that can be gained by this
multiobjective approach.

    

### [[2012.08496] Spectral Methods for Data Science: A Statistical Perspective](http://arxiv.org/abs/2012.08496)


  Spectral methods have emerged as a simple yet surprisingly effective approach
for extracting information from massive, noisy and incomplete data. In a
nutshell, spectral methods refer to a collection of algorithms built upon the
eigenvalues (resp. singular values) and eigenvectors (resp. singular vectors)
of some properly designed matrices constructed from data. A diverse array of
applications have been found in machine learning, data science, and signal
processing. Due to their simplicity and effectiveness, spectral methods are not
only used as a stand-alone estimator, but also frequently employed to
initialize other more sophisticated algorithms to improve performance.
While the studies of spectral methods can be traced back to classical matrix
perturbation theory and methods of moments, the past decade has witnessed
tremendous theoretical advances in demystifying their efficacy through the lens
of statistical modeling, with the aid of non-asymptotic random matrix theory.
This monograph aims to present a systematic, comprehensive, yet accessible
introduction to spectral methods from a modern statistical perspective,
highlighting their algorithmic implications in diverse large-scale
applications. In particular, our exposition gravitates around several central
questions that span various applications: how to characterize the sample
efficiency of spectral methods in reaching a target level of statistical
accuracy, and how to assess their stability in the face of random noise,
missing data, and adversarial corruptions? In addition to conventional $\ell_2$
perturbation analysis, we present a systematic $\ell_{\infty}$ and
$\ell_{2,\infty}$ perturbation theory for eigenspace and singular subspaces,
which has only recently become available owing to a powerful "leave-one-out"
analysis framework.

    

### [[2012.15739] Uncertainty Bounds for Multivariate Machine Learning Predictions on High-Strain Brittle Fracture](http://arxiv.org/abs/2012.15739)


  Simulation of the crack network evolution on high strain rate impact
experiments performed in brittle materials is very compute-intensive. The cost
increases even more if multiple simulations are needed to account for the
randomness in crack length, location, and orientation, which is inherently
found in real-world materials. Constructing a machine learning emulator can
make the process faster by orders of magnitude. There has been little work,
however, on assessing the error associated with their predictions. Estimating
these errors is imperative for meaningful overall uncertainty quantification.
In this work, we extend the heteroscedastic uncertainty estimates to bound a
multiple output machine learning emulator. We find that the response prediction
is accurate within its predicted errors, but with a somewhat conservative
estimate of uncertainty.

    

### [[2101.07415] ES-ENAS: Blackbox Optimization over Hybrid Spaces via Combinatorial and Continuous Evolution](http://arxiv.org/abs/2101.07415)


  We consider the problem of efficient blackbox optimization over a large
hybrid search space, consisting of a mixture of a high dimensional continuous
space and a complex combinatorial space. Such examples arise commonly in
evolutionary computation, but also more recently, neuroevolution and
architecture search for Reinforcement Learning (RL) policies. In this paper, we
introduce ES-ENAS, a simple joint optimization procedure by combining
Evolutionary Strategies (ES) and combinatorial optimization techniques in a
highly scalable and intuitive way, inspired by the \textit{one-shot} or
\textit{supernet} paradigm introduced in Efficient Neural Architecture Search
(ENAS). Our main insight is noticing that ES is already a highly distributed
algorithm involving hundreds of blackbox evaluations which can not only be used
for training neural network weights, but also for feedback to a combinatorial
optimizer. Through this relatively simple marriage between two different lines
of research, we are able to gain the best of both worlds, and empirically
demonstrate our approach by optimizing BBOB functions over hybrid spaces as
well as combinatorial neural network architectures via edge pruning and
quantization on popular RL benchmarks. Due to the modularity of the algorithm,
we also are able incorporate a wide variety of popular techniques ranging from
use of different continuous and combinatorial optimizers, as well as
constrained optimization.

    

### [[2101.12190] Practical distributed quantum information processing with LOCCNet](http://arxiv.org/abs/2101.12190)


  Distributed quantum information processing is essential for building quantum
networks and enabling more extensive quantum computations. In this regime,
several spatially separated parties share a multipartite quantum system, and
the most natural set of operations is Local Operations and Classical
Communication (LOCC). As a pivotal part in quantum information theory and
practice, LOCC has led to many vital protocols such as quantum teleportation.
However, designing practical LOCC protocols is challenging due to LOCC's
intractable structure and limitations set by near-term quantum devices. Here we
introduce LOCCNet, a machine learning framework facilitating protocol design
and optimization for distributed quantum information processing tasks. As
applications, we explore various quantum information tasks such as entanglement
distillation, quantum state discrimination, and quantum channel simulation. We
discover protocols with evident improvements, in particular, for entanglement
distillation with quantum states of interest in quantum information. Our
approach opens up new opportunities for exploring entanglement and its
applications with machine learning, which will potentially sharpen our
understanding of the power and limitations of LOCC. An implementation of
LOCCNet is available in Paddle Quantum, a quantum machine learning Python
package based on PaddlePaddle deep learning platform.

    

### [[2102.10242] Towards Automatic Evaluation of Dialog Systems: A Model-Free Off-Policy Evaluation Approach](http://arxiv.org/abs/2102.10242)


  Reliable automatic evaluation of dialogue systems under an interactive
environment has long been overdue. An ideal environment for evaluating dialog
systems, also known as the Turing test, needs to involve human interaction,
which is usually not affordable for large-scale experiments. Though researchers
have attempted to use metrics (e.g., perplexity, BLEU) in language generation
tasks or some model-based reinforcement learning methods (e.g., self-play
evaluation) for automatic evaluation, these methods only show a very weak
correlation with the actual human evaluation in practice. To bridge such a gap,
we propose a new framework named ENIGMA for estimating human evaluation scores
based on recent advances of off-policy evaluation in reinforcement learning.
ENIGMA only requires a handful of pre-collected experience data, and therefore
does not involve human interaction with the target policy during the
evaluation, making automatic evaluations feasible. More importantly, ENIGMA is
model-free and agnostic to the behavior policies for collecting the experience
data (see details in Section 2), which significantly alleviates the technical
difficulties of modeling complex dialogue environments and human behaviors. Our
experiments show that ENIGMA significantly outperforms existing methods in
terms of correlation with human evaluation scores.

    

### [[2102.10556] Inductive logic programming at 30](http://arxiv.org/abs/2102.10556)


  Inductive logic programming (ILP) is a form of logic-based machine learning.
The goal is to induce a hypothesis (a logic program) that generalises given
training examples. As ILP turns 30, we review the last decade of research. We
focus on (i) new meta-level search methods, (ii) techniques for learning
recursive programs, (iii) new approaches for predicate invention, and (iv) the
use of different technologies. We conclude by discussing current limitations of
ILP and directions for future research.

    

### [[2103.05134] Constrained Learning with Non-Convex Losses](http://arxiv.org/abs/2103.05134)


  Though learning has become a core technology of modern information
processing, there is now ample evidence that it can lead to biased, unsafe, and
prejudiced solutions. The need to impose requirements on learning is therefore
paramount, especially as it reaches critical applications in social,
industrial, and medical domains. However, the non-convexity of most modern
learning problems is only exacerbated by the introduction of constraints.
Whereas good unconstrained solutions can often be learned using empirical risk
minimization (ERM), even obtaining a model that satisfies statistical
constraints can be challenging, all the more so a good one. In this paper, we
overcome this issue by learning in the empirical dual domain, where constrained
statistical learning problems become unconstrained, finite dimensional, and
deterministic. We analyze the generalization properties of this approach by
bounding the empirical duality gap, i.e., the difference between our
approximate, tractable solution and the solution of the original
(non-convex)~statistical problem, and provide a practical constrained learning
algorithm. These results establish a constrained counterpart of classical
learning theory and enable the explicit use of constraints in learning. We
illustrate this algorithm and theory in rate-constrained learning applications.

    

### [[2103.09448] Adversarial Attacks on Camera-LiDAR Models for 3D Car Detection](http://arxiv.org/abs/2103.09448)


  Most autonomous vehicles (AVs) rely on LiDAR and RGB camera sensors for
perception. Using these point cloud and image data, perception models based on
deep neural nets (DNNs) have achieved state-of-the-art performance in 3D
detection. The vulnerability of DNNs to adversarial attacks has been heavily
investigated in the RGB image domain and more recently in the point cloud
domain, but rarely in both domains simultaneously. Multi-modal perception
systems used in AVs can be divided into two broad types: cascaded models which
use each modality independently, and fusion models which learn from different
modalities simultaneously. We propose a universal and physically realizable
adversarial attack for each type, and study and contrast their respective
vulnerabilities to attacks. We place a single adversarial object with specific
shape and texture on top of a car with the objective of making this car evade
detection. Evaluating on the popular KITTI benchmark, our adversarial object
made the host vehicle escape detection by each model type more than 50% of the
time. The dense RGB input contributed more to the success of the adversarial
attacks on both cascaded and fusion models.

    

### [[2105.14103] An Attention Free Transformer](http://arxiv.org/abs/2105.14103)


  We introduce Attention Free Transformer (AFT), an efficient variant of
Transformers that eliminates the need for dot product self attention. In an AFT
layer, the key and value are first combined with a set of learned position
biases, the result of which is multiplied with the query in an element-wise
fashion. This new operation has a memory complexity linear w.r.t. both the
context size and the dimension of features, making it compatible to both large
input and model sizes. We also introduce AFT-local and AFT-conv, two model
variants that take advantage of the idea of locality and spatial weight sharing
while maintaining global connectivity. We conduct extensive experiments on two
autoregressive modeling tasks (CIFAR10 and Enwik8) as well as an image
recognition task (ImageNet-1K classification). We show that AFT demonstrates
competitive performance on all the benchmarks, while providing excellent
efficiency at the same time.

    

### [[2106.00730] Enabling Efficiency-Precision Trade-offs for Label Trees in Extreme Classification](http://arxiv.org/abs/2106.00730)


  Extreme multi-label classification (XMC) aims to learn a model that can tag
data points with a subset of relevant labels from an extremely large label set.
Real world e-commerce applications like personalized recommendations and
product advertising can be formulated as XMC problems, where the objective is
to predict for a user a small subset of items from a catalog of several million
products. For such applications, a common approach is to organize these labels
into a tree, enabling training and inference times that are logarithmic in the
number of labels. While training a model once a label tree is available is well
studied, designing the structure of the tree is a difficult task that is not
yet well understood, and can dramatically impact both model latency and
statistical performance. Existing approaches to tree construction fall at an
extreme point, either optimizing exclusively for statistical performance, or
for latency. We propose an efficient information theory inspired algorithm to
construct intermediary operating points that trade off between the benefits of
both. Our algorithm enables interpolation between these objectives, which was
not previously possible. We corroborate our theoretical analysis with numerical
results, showing that on the Wiki-500K benchmark dataset our method can reduce
a proxy for expected latency by up to 28% while maintaining the same accuracy
as Parabel. On several datasets derived from e-commerce customer logs, our
modified label tree is able to improve this expected latency metric by up to
20% while maintaining the same accuracy. Finally, we discuss challenges in
realizing these latency improvements in deployed models.

    

### [[2106.02036] Anticipative Video Transformer](http://arxiv.org/abs/2106.02036)


  We propose Anticipative Video Transformer (AVT), an end-to-end
attention-based video modeling architecture that attends to the previously
observed video in order to anticipate future actions. We train the model
jointly to predict the next action in a video sequence, while also learning
frame feature encoders that are predictive of successive future frames'
features. Compared to existing temporal aggregation strategies, AVT has the
advantage of both maintaining the sequential progression of observed actions
while still capturing long-range dependencies--both critical for the
anticipation task. Through extensive experiments, we show that AVT obtains the
best reported performance on four popular action anticipation benchmarks:
EpicKitchens-55, EpicKitchens-100, EGTEA Gaze+, and 50-Salads; and it wins
first place in the EpicKitchens-100 CVPR'21 challenge.

    

### [[2106.10934] GRAND: Graph Neural Diffusion](http://arxiv.org/abs/2106.10934)


  We present Graph Neural Diffusion (GRAND) that approaches deep learning on
graphs as a continuous diffusion process and treats Graph Neural Networks
(GNNs) as discretisations of an underlying PDE. In our model, the layer
structure and topology correspond to the discretisation choices of temporal and
spatial operators. Our approach allows a principled development of a broad new
class of GNNs that are able to address the common plights of graph learning
models such as depth, oversmoothing, and bottlenecks. Key to the success of our
models are stability with respect to perturbations in the data and this is
addressed for both implicit and explicit discretisation schemes. We develop
linear and nonlinear versions of GRAND, which achieve competitive results on
many standard graph benchmarks.

    

### [[2107.02168] DPPIN: A Biological Repository of Dynamic Protein-Protein Interaction Network Data](http://arxiv.org/abs/2107.02168)


  Nowadays, many network representation learning algorithms and downstream
network mining tasks have already paid attention to dynamic networks or
temporal networks, which are more suitable for real-world complex scenarios by
modeling evolving patterns and temporal dependencies between node interactions.
Moreover, representing and mining temporal networks have a wide range of
applications, such as fraud detection, social network analysis, and drug
discovery. To contribute to the network representation learning and network
mining research community, in this paper, we generate a new biological
repository of dynamic protein-protein interaction network data (i.e., DPPIN),
which consists of twelve dynamic network datasets describing protein-level
interactions of yeast cells at different scales. We first introduce the
generation process of DPPIN. To demonstrate the value of our published
repository DPPIN, we then list the potential applications that would be
benefited. Furthermore, we design dynamic local clustering, dynamic spectral
clustering, dynamic subgraph matching, dynamic node classification, and dynamic
graph classification experiments, where network datasets of DPPIN could
indicate future research opportunities for some tasks by presenting challenges
on state-of-the-art baseline algorithms. Finally, we identify future directions
for improving the utility of this repository and welcome constructive inputs
from the community. All resources of this work are deployed and publicly
available at this https URL.

    

### [[2109.06148] DAFNe: A One-Stage Anchor-Free Deep Model for Oriented Object Detection](http://arxiv.org/abs/2109.06148)


  Object detection is a fundamental task in computer vision. While approaches
for axis-aligned bounding box detection have made substantial progress in
recent years, they perform poorly on oriented objects which are common in
several real-world scenarios such as aerial view imagery and security camera
footage. In these cases, a large part of a predicted bounding box will,
undesirably, cover non-object related areas. Therefore, oriented object
detection has emerged with the aim of generalizing object detection to
arbitrary orientations. This enables a tighter fit to oriented objects, leading
to a better separation of bounding boxes especially in case of dense object
distributions. The vast majority of the work in this area has focused on
complex two-stage anchor-based approaches. Anchors act as priors on the
bounding box shape and require attentive hyper-parameter fine-tuning on a
per-dataset basis, increased model size, and come with computational overhead.
In this work, we present DAFNe: A Dense one-stage Anchor-Free deep Network for
oriented object detection. As a one-stage model, DAFNe performs predictions on
a dense grid over the input image, being architecturally simpler and faster, as
well as easier to optimize than its two-stage counterparts. Furthermore, as an
anchor-free model, DAFNe reduces the prediction complexity by refraining from
employing bounding box anchors. Moreover, we introduce an orientation-aware
generalization of the center-ness function for arbitrarily oriented bounding
boxes to down-weight low-quality predictions and a center-to-corner bounding
box prediction strategy that improves object localization performance. DAFNe
improves the prediction accuracy over the previous best one-stage anchor-free
model results on DOTA 1.0 by 4.65% mAP, setting the new state-of-the-art
results by achieving 76.95% mAP.

    

### [[2109.09307] Assisted Learning for Organizations with Limited Data](http://arxiv.org/abs/2109.09307)


  We develop an assisted learning framework for assisting organization-level
learners to improve their learning performance with limited and imbalanced
data. In particular, learners at the organization level usually have sufficient
computation resource, but are subject to stringent collaboration policy and
information privacy. Their limited imbalanced data often cause biased inference
and sub-optimal decision-making. In our assisted learning framework, an
organizational learner purchases assistance service from a service provider and
aims to enhance its model performance within a few assistance rounds. We
develop effective stochastic training algorithms for assisted deep learning and
assisted reinforcement learning. Different from existing distributed algorithms
that need to frequently transmit gradients or models, our framework allows the
learner to only occasionally share information with the service provider, and
still achieve a near-oracle model as if all the data were centralized.

    

### [[2109.10774] "It's a Trap!"-How Speculation Invariance Can Be Abused with Forward Speculative Interference](http://arxiv.org/abs/2109.10774)


  Speculative side-channel attacks access sensitive data and use transmitters
to leak the data during wrong-path execution. Various defenses have been
proposed to prevent such information leakage. However, not all speculatively
executed instructions are unsafe: Recent work demonstrates that speculation
invariant instructions are independent of speculative control-flow paths and
are guaranteed to eventually commit, regardless of the speculation outcome.
Compile-time information coupled with run-time mechanisms can then selectively
lift defenses for speculation invariant instructions, reclaiming some of the
lost performance.
Unfortunately, speculation invariant instructions can easily be manipulated
by a form of speculative interference to leak information via a new
side-channel that we introduce in this paper. We show that forward speculative
interference whereolder speculative instructions interfere with younger
speculation invariant instructions effectively turns them into transmitters for
secret data accessed during speculation. We demonstrate forward speculative
interference on actual hardware, by selectively filling the reorder buffer
(ROB) with instructions, pushing speculative invariant instructions in-or-out
of the ROB on demand, based on a speculatively accessed secret. This reveals
the speculatively accessed secret, as the occupancy of the ROB itself becomes a
new speculative side-channel.

    

### [[2109.10430] GAP2WSS: A Genetic Algorithm based on the Pareto Principle for Web Service Selection](http://arxiv.org/abs/2109.10430)


  Despite all the progress in Web service selection, the need for an approach
with a better optimality and performance still remains. This paper presents a
genetic algorithm by adopting the Pareto principle that is called GAP2WSS for
selecting a Web service for each task of a composite Web service from a pool of
candidate Web services. In contrast to the existing approaches, all global QoS
constraints, interservice constraints, and transactional constraints are
considered simultaneously. At first, all candidate Web services are scored and
ranked per each task using the proposed mechanism. Then, the top 20 percent of
the candidate Web services of each task are considered as the candidate Web
services of the corresponding task to reduce the problem search space. Finally,
the Web service selection problem is solved by focusing only on these 20
percent candidate Web services of each task using a genetic algorithm.
Empirical studies demonstrate this approach leads to a higher efficiency and
efficacy as compared with the case that all the candidate Web services are
considered in solving the problem.

    

### [[2109.10433] Transcoding Billions of Unicode Characters per Second with SIMD Instructions](http://arxiv.org/abs/2109.10433)


  In software, text is often represented using Unicode formats (UTF-8 and
UTF-16). We frequently have to convert text from one format to the other, a
process called transcoding. Popular transcoding functions are slower than
state-of-the-art disks and networks. These transcoding functions make little
use of the single-instruction-multiple-data (SIMD) instructions available on
commodity processors. By designing transcoding algorithms for SIMD
instructions, we multiply the speed of transcoding on current systems (x64 and
ARM). To ensure reproducibility, we make our software freely available as an
open source library.

    

### [[2109.10554] On Conflict-Free Replicated Data Types and Equivocation in Byzantine Setups](http://arxiv.org/abs/2109.10554)


  We explore the property of equivocation tolerance for Conflict-Free
Replicated Data Types (CRDTs). We show that a subclass of CRDTs is
equivocation-tolerant and can thereby cope with any number of Byzantine faults:
Without equivocation detection, prevention or remediation, they still fulfill
strong eventual consistency (SEC). We also conjecture that there is only one
operation-based CRDT design supporting non-commutative operations that fulfills
SEC in Byzantine environments with any number of faults.

    

### [[2109.10727] Frisbee: automated testing of Cloud-native applications in Kubernetes](http://arxiv.org/abs/2109.10727)


  As more and more companies are migrating (or planning to migrate) from
on-premise to Cloud, their focus is to find anomalies and deficits as early as
possible in the development life cycle. We propose Frisbee, a declarative
language and associated runtime components for testing cloud-native
applications on top of Kubernetes. Given a template describing the system under
test and a workflow describing the experiment, Frisbee automatically interfaces
with Kubernetes to deploy the necessary software in containers, launch needed
sidecars, execute the workflow steps, and perform automated checks for
deviation from expected behavior. We evaluate Frisbee through a series of
tests, to demonstrate its role in designing, and evaluating cloud-native
applications; Frisbee helps in testing uncertainties at the level of
application (e.g., dynamically changing request patterns), infrastructure
(e.g., crashes, network partitions), and deployment (e.g., saturation points).
Our findings have strong implications for the design, deployment, and
evaluation of cloud applications. The most prominent is that: erroneous
benchmark outputs can cause an apparent performance improvement, automated
failover mechanisms may require interoperability with clients, and that a
proper placement policy should also account for the clock frequency, not only
the number of cores.

    

### [[2109.10787] DHT-based Communications Survey: Architectures and Use Cases](http://arxiv.org/abs/2109.10787)


  Several distributed system paradigms utilize Distributed Hash Tables (DHTs)
to realize structured peer-to-peer (P2P) overlays. DHT structures arise as the
most commonly used organizations for peers that can efficiently perform crucial
services such as data storage, replication, query resolution, and load
balancing. With the advances in various distributed system technologies, novel
and efficient solutions based on DHTs emerge and play critical roles in system
design. DHT-based methods and communications have been proposed to address
challenges such as scalability, availability, reliability and performance, by
considering unique characteristics of these technologies. In this article, we
propose a classification of the state-of-the-art DHT-based methods focusing on
their system architecture, communication, routing and technological aspects
across various system domains. To the best of our knowledge, there is no
comprehensive survey on DHT-based applications from system architecture and
communication perspectives that spans various domains of recent distributed
system technologies. We investigate the recently emerged DHT-based solutions in
the seven key domains of edge and fog computing, cloud computing, blockchain,
the Internet of Things (IoT), Online Social Networks (OSNs), Mobile Ad Hoc
Networks (MANETs), and Vehicular Ad Hoc Networks (VANETs). In contrast to the
existing surveys, our study goes beyond the commonly known DHT methods such as
storage, routing, and lookup, and identifies diverse DHT-based solutions
including but not limited to aggregation, task scheduling, resource management
and discovery, clustering and group management, federation, data dependency
management, and data transmission. Furthermore, we identify open problems and
discuss future research guidelines for each domain.

    

### [[2109.10876] Code modernization strategies for short-range non-bonded molecular dynamics simulations](http://arxiv.org/abs/2109.10876)


  As modern HPC systems increasingly rely on greater core counts and wider
vector registers, applications need to be adapted to fully utilize these
hardware capabilities. One class of applications that can benefit from this
increase in parallelism are molecular dynamics simulations. In this paper, we
describe our efforts at modernizing the ESPResSo++ molecular dynamics
simulation package by restructuring its particle data layout for efficient
memory accesses and applying vectorization techniques to benefit the
calculation of short-range non-bonded forces, which results in an overall 3
times speedup and serves as a baseline for further optimizations. We also
implement finer-grain parallelism for multi-core CPUs through HPX, a C++
runtime system which uses lightweight threads and an asynchronous many-task
approach to maximize parallelism. Our goal is to evaluate the performance of an
HPX-based approach compared to the bulk-synchronous MPI-based implementation.
This requires the introduction of an additional layer to the domain
decomposition scheme that defines the task granularity. On spatially
inhomogeneous systems, which impose a corresponding load-imbalance in
traditional MPI-based approaches, we demonstrate that by choosing an optimal
task size, the efficient work-stealing mechanisms of HPX can overcome the
overhead of communication resulting in an overall 1.3 times speedup compared to
the baseline MPI version.

    

### [[2005.12873] Benchmarking Graph Data Management and Processing Systems: A Survey](http://arxiv.org/abs/2005.12873)


  The development of scalable, representative, and widely adopted benchmarks
for graph data systems have been a question for which answers has been sought
for decades. We conduct an in-depth study of the existing literature on
benchmarks for graph data management and processing, covering 20 different
benchmarks developed during the last 15 years. We categorize the benchmarks
into three areas focusing on benchmarks for graph processing systems, graph
database benchmarks, and bigdata benchmarks with graph processing workloads.
This systematic approach allows us to identify multiple issues existing in this
area, including i) few benchmarks exist which can produce high workload
scenarios, ii) no significant work done on benchmarking graph stream processing
as well as graph based machine learning, iii) benchmarks tend to use
conventional metrics despite new meaningful metrics have been around for years,
iv) increasing number of big data benchmarks appear with graph processing
workloads. Following these observations, we conclude the survey by describing
key challenges for future research on graph data systems benchmarking.

    

### [[2010.15559] Quantum Computing: A Taxonomy, Systematic Review and Future Directions](http://arxiv.org/abs/2010.15559)


  Quantum computing is an emerging paradigm with the potential to offer
significant computational advantage over conventional classical computing by
exploiting quantum-mechanical principles such as entanglement and
superposition. It is anticipated that this computational advantage of quantum
computing will help to solve many complex and computationally intractable
problems in several areas such as drug design, data science, clean energy,
finance, industrial chemical development, secure communications, and quantum
chemistry. In recent years, tremendous progress in both quantum hardware
development and quantum software/algorithm have brought quantum computing much
closer to reality. Indeed, the demonstration of quantum supremacy marks a
significant milestone in the Noisy Intermediate Scale Quantum (NISQ) era - the
next logical step being the quantum advantage whereby quantum computers solve a
real-world problem much more efficiently than classical computing. As the
quantum devices are expected to steadily scale up in the next few years,
quantum decoherence and qubit interconnectivity are two of the major challenges
to achieve quantum advantage in the NISQ era. Quantum computing is a highly
topical and fast-moving field of research with significant ongoing progress in
all facets. This article presents a comprehensive review of quantum computing
literature, and taxonomy of quantum computing. Further, the proposed taxonomy
is used to map various related studies to identify the research gaps. A
detailed overview of quantum software tools and technologies, post-quantum
cryptography and quantum computer hardware development to document the current
state-of-the-art in the respective areas. We finish the article by highlighting
various open challenges and promising future directions for research.

    

### [[2109.10415] What Would it Take to get Biomedical QA Systems into Practice?](http://arxiv.org/abs/2109.10415)


  Medical question answering (QA) systems have the potential to answer
clinicians uncertainties about treatment and diagnosis on demand, informed by
the latest evidence. However, despite the significant progress in general QA
made by the NLP community, medical QA systems are still not widely used in
clinical environments. One likely reason for this is that clinicians may not
readily trust QA system outputs, in part because transparency, trustworthiness,
and provenance have not been key considerations in the design of such models.
In this paper we discuss a set of criteria that, if met, we argue would likely
increase the utility of biomedical QA systems, which may in turn lead to
adoption of such systems in practice. We assess existing models, tasks, and
datasets with respect to these criteria, highlighting shortcomings of
previously proposed approaches and pointing toward what might be more usable QA
systems.

    

### [[2109.10475] Salience-Aware Event Chain Modeling for Narrative Understanding](http://arxiv.org/abs/2109.10475)


  Storytelling, whether via fables, news reports, documentaries, or memoirs,
can be thought of as the communication of interesting and related events that,
taken together, form a concrete process. It is desirable to extract the event
chains that represent such processes. However, this extraction remains a
challenging problem. We posit that this is due to the nature of the texts from
which chains are discovered. Natural language text interleaves a narrative of
concrete, salient events with background information, contextualization,
opinion, and other elements that are important for a variety of necessary
discourse and pragmatics acts but are not part of the principal chain of events
being communicated. We introduce methods for extracting this principal chain
from natural language text, by filtering away non-salient events and supportive
sentences. We demonstrate the effectiveness of our methods at isolating
critical event chains by comparing their effect on downstream tasks. We show
that by pre-training large language models on our extracted chains, we obtain
improvements in two tasks that benefit from a clear understanding of event
chains: narrative prediction and event-based temporal question answering. The
demonstrated improvements and ablative studies confirm that our extraction
method isolates critical event chains.

    

### [[2109.10480] DialogueBERT: A Self-Supervised Learning based Dialogue Pre-training Encoder](http://arxiv.org/abs/2109.10480)


  With the rapid development of artificial intelligence, conversational bots
have became prevalent in mainstream E-commerce platforms, which can provide
convenient customer service timely. To satisfy the user, the conversational
bots need to understand the user's intention, detect the user's emotion, and
extract the key entities from the conversational utterances. However,
understanding dialogues is regarded as a very challenging task. Different from
common language understanding, utterances in dialogues appear alternately from
different roles and are usually organized as hierarchical structures. To
facilitate the understanding of dialogues, in this paper, we propose a novel
contextual dialogue encoder (i.e. DialogueBERT) based on the popular
pre-trained language model BERT. Five self-supervised learning pre-training
tasks are devised for learning the particularity of dialouge utterances. Four
different input embeddings are integrated to catch the relationship between
utterances, including turn embedding, role embedding, token embedding and
position embedding. DialogueBERT was pre-trained with 70 million dialogues in
real scenario, and then fine-tuned in three different downstream dialogue
understanding tasks. Experimental results show that DialogueBERT achieves
exciting results with 88.63% accuracy for intent recognition, 94.25% accuracy
for emotion recognition and 97.04% F1 score for named entity recognition, which
outperforms several strong baselines by a large margin.

    

### [[2109.10493] Learning Robust Agents for Visual Navigation in Dynamic Environments: The Winning Entry of iGibson Challenge 2021](http://arxiv.org/abs/2109.10493)


  This paper presents an approach for improving navigation in dynamic and
interactive environments, which won the 1st place in the iGibson Interactive
Navigation Challenge 2021. While the last few years have produced impressive
progress on PointGoal Navigation in static environments, relatively little
effort has been made on more realistic dynamic environments. The iGibson
Challenge proposed two new navigation tasks, Interactive Navigation and Social
Navigation, which add displaceable obstacles and moving pedestrians into the
simulator environment. Our approach to study these problems uses two key ideas.
First, we employ large-scale reinforcement learning by leveraging the Habitat
simulator, which supports high performance parallel computing for both
simulation and synchronized learning. Second, we employ a new data augmentation
technique that adds more dynamic objects into the environment, which can also
be combined with traditional image-based augmentation techniques to boost the
performance further. Lastly, we achieve sim-to-sim transfer from Habitat to the
iGibson simulator, and demonstrate that our proposed methods allow us to train
robust agents in dynamic environments with interactive objects or moving
humans. Video link: this https URL


### [[2109.10500] HyperExpan: Taxonomy Expansion with Hyperbolic Representation Learning](http://arxiv.org/abs/2109.10500)


  Taxonomies are valuable resources for many applications, but the limited
coverage due to the expensive manual curation process hinders their general
applicability. Prior works attempt to automatically expand existing taxonomies
to improve their coverage by learning concept embeddings in Euclidean space,
while taxonomies, inherently hierarchical, more naturally align with the
geometric properties of a hyperbolic space. In this paper, we present
HyperExpan, a taxonomy expansion algorithm that seeks to preserve the structure
of a taxonomy in a more expressive hyperbolic embedding space and learn to
represent concepts and their relations with a Hyperbolic Graph Neural Network
(HGNN). Specifically, HyperExpan leverages position embeddings to exploit the
structure of the existing taxonomies, and characterizes the concept profile
information to support the inference on unseen concepts during training.
Experiments show that our proposed HyperExpan outperforms baseline models with
representation learning in a Euclidean feature space and achieves
state-of-the-art performance on the taxonomy expansion benchmarks.

    

### [[2109.10540] Awakening Latent Grounding from Pretrained Language Models for Semantic Parsing](http://arxiv.org/abs/2109.10540)


  Recent years pretrained language models (PLMs) hit a success on several
downstream tasks, showing their power on modeling language. To better
understand and leverage what PLMs have learned, several techniques have emerged
to explore syntactic structures entailed by PLMs. However, few efforts have
been made to explore grounding capabilities of PLMs, which are also essential.
In this paper, we highlight the ability of PLMs to discover which token should
be grounded to which concept, if combined with our proposed
erasing-then-awakening approach. Empirical studies on four datasets demonstrate
that our approach can awaken latent grounding which is understandable to human
experts, even if it is not exposed to such labels during training. More
importantly, our approach shows great potential to benefit downstream semantic
parsing models. Taking text-to-SQL as a case study, we successfully couple our
approach with two off-the-shelf parsers, obtaining an absolute improvement of
up to 9.8%.

    

### [[2109.10547] K-AID: Enhancing Pre-trained Language Models with Domain Knowledge for Question Answering](http://arxiv.org/abs/2109.10547)


  Knowledge enhanced pre-trained language models (K-PLMs) are shown to be
effective for many public tasks in the literature but few of them have been
successfully applied in practice. To address this problem, we propose K-AID, a
systematic approach that includes a low-cost knowledge acquisition process for
acquiring domain knowledge, an effective knowledge infusion module for
improving model performance, and a knowledge distillation component for
reducing the model size and deploying K-PLMs on resource-restricted devices
(e.g., CPU) for real-world application. Importantly, instead of capturing
entity knowledge like the majority of existing K-PLMs, our approach captures
relational knowledge, which contributes to better-improving sentence-level text
classification and text matching tasks that play a key role in question
answering (QA). We conducted a set of experiments on five text classification
tasks and three text matching tasks from three domains, namely E-commerce,
Government, and Film&TV, and performed online A/B tests in E-commerce.
Experimental results show that our approach is able to achieve substantial
improvement on sentence-level question answering tasks and bring beneficial
business value in industrial settings.

    

### [[2109.10557] A Reinforcement Learning Benchmark for Autonomous Driving in Intersection Scenarios](http://arxiv.org/abs/2109.10557)


  In recent years, control under urban intersection scenarios becomes an
emerging research topic. In such scenarios, the autonomous vehicle confronts
complicated situations since it must deal with the interaction with social
vehicles timely while obeying the traffic rules. Generally, the autonomous
vehicle is supposed to avoid collisions while pursuing better efficiency. The
existing work fails to provide a framework that emphasizes the integrity of the
scenarios while being able to deploy and test reinforcement learning(RL)
methods. Specifically, we propose a benchmark for training and testing RL-based
autonomous driving agents in complex intersection scenarios, which is called
RL-CIS. Then, a set of baselines are deployed consists of various algorithms.
The test benchmark and baselines are to provide a fair and comprehensive
training and testing platform for the study of RL for autonomous driving in the
intersection scenario, advancing the progress of RL-based methods for
intersection autonomous driving control. The code of our proposed framework can
be found at this https URL.

    

### [[2109.10559] Hierarchical Multimodal Transformer to Summarize Videos](http://arxiv.org/abs/2109.10559)


  Although video summarization has achieved tremendous success benefiting from
Recurrent Neural Networks (RNN), RNN-based methods neglect the global
dependencies and multi-hop relationships among video frames, which limits the
performance. Transformer is an effective model to deal with this problem, and
surpasses RNN-based methods in several sequence modeling tasks, such as machine
translation, video captioning, \emph{etc}. Motivated by the great success of
transformer and the natural structure of video (frame-shot-video), a
hierarchical transformer is developed for video summarization, which can
capture the dependencies among frame and shots, and summarize the video by
exploiting the scene information formed by shots. Furthermore, we argue that
both the audio and visual information are essential for the video summarization
task. To integrate the two kinds of information, they are encoded in a
two-stream scheme, and a multimodal fusion mechanism is developed based on the
hierarchical transformer. In this paper, the proposed method is denoted as
Hierarchical Multimodal Transformer (HMT). Practically, extensive experiments
show that HMT surpasses most of the traditional, RNN-based and attention-based
video summarization methods.

    

### [[2109.10582] Partial sensitivity analysis in differential privacy](http://arxiv.org/abs/2109.10582)


  Differential privacy (DP) allows the quantification of privacy loss when the
data of individuals is subjected to algorithmic processing such as machine
learning, as well as the provision of objective privacy guarantees. However,
while techniques such as individual Rényi DP (RDP) allow for granular,
per-person privacy accounting, few works have investigated the impact of each
input feature on the individual's privacy loss. Here we extend the view of
individual RDP by introducing a new concept we call partial sensitivity, which
leverages symbolic automatic differentiation to determine the influence of each
input feature on the gradient norm of a function. We experimentally evaluate
our approach on queries over private databases, where we obtain a feature-level
contribution of private attributes to the DP guarantee of individuals.
Furthermore, we explore our findings in the context of neural network training
on synthetic data by investigating the partial sensitivity of input pixels on
an image classification task.

    

### [[2109.10602] Context-aware Tree-based Deep Model for Recommender Systems](http://arxiv.org/abs/2109.10602)


  How to predict precise user preference and how to make efficient retrieval
from a big corpus are two major challenges of large-scale industrial
recommender systems. In tree-based methods, a tree structure T is adopted as
index and each item in corpus is attached to a leaf node on T . Then the
recommendation problem is converted into a hierarchical retrieval problem
solved by a beam search process efficiently. In this paper, we argue that the
tree index used to support efficient retrieval in tree-based methods also has
rich hierarchical information about the corpus. Furthermore, we propose a novel
context-aware tree-based deep model (ConTDM) for recommender systems. In
ConTDM, a context-aware user preference prediction model M is designed to
utilize both horizontal and vertical contexts on T . Horizontally, a graph
convolutional layer is used to enrich the representation of both users and
nodes on T with their neighbors. Vertically, a parent fusion layer is designed
in M to transmit the user preference representation in higher levels of T to
the current level, grasping the essence that tree-based methods are generating
the candidate set from coarse to detail during the beam search retrieval.
Besides, we argue that the proposed user preference model in ConTDM can be
conveniently extended to other tree-based methods for recommender systems. Both
experiments on large scale real-world datasets and online A/B test in large
scale industrial applications show the significant improvements brought by
ConTDM.

    

### [[2109.10633] Reactive Answer Set Programming](http://arxiv.org/abs/2109.10633)


  Logic Production System (LPS) is a logic-based framework for modelling
reactive behaviour. Based on abductive logic programming, it combines reactive
rules with logic programs, a database and a causal theory that specifies
transitions between the states of the database. This paper proposes a
systematic mapping of the Kernel of this framework (called KELPS) into an
answer set program (ASP). For this purpose a new variant of KELPS with finite
models, called $n$-distance KELPS, is introduced. A formal definition of the
mapping from this $n$-distance KELPS to ASP is given and proven sound and
complete. The Answer Set Programming paradigm allows to capture additional
behaviours to the basic reactivity of KELPS, in particular proactive,
preemptive and prospective behaviours. These are all discussed and illustrated
with examples. Then a hybrid framework is proposed that integrates KELPS and
ASP, allowing to combine the strengths of both paradigms. Under consideration
in Theory and Practice of Logic Programming (TPLP).

    

### [[2109.10637] Facilitating human-wildlife cohabitation through conflict prediction](http://arxiv.org/abs/2109.10637)


  With increasing world population and expanded use of forests as cohabited
regions, interactions and conflicts with wildlife are increasing, leading to
large-scale loss of lives (animal and human) and livelihoods (economic). While
community knowledge is valuable, forest officials and conservation
organisations can greatly benefit from predictive analysis of human-wildlife
conflict, leading to targeted interventions that can potentially help save
lives and livelihoods. However, the problem of prediction is a complex
socio-technical problem in the context of limited data in low-resource regions.
Identifying the "right" features to make accurate predictions of conflicts at
the required spatial granularity using a sparse conflict training dataset} is
the key challenge that we address in this paper. Specifically, we do an
illustrative case study on human-wildlife conflicts in the Bramhapuri Forest
Division in Chandrapur, Maharashtra, India. Most existing work has considered
human-wildlife conflicts in protected areas and to the best of our knowledge,
this is the first effort at prediction of human-wildlife conflicts in
unprotected areas and using those predictions for deploying interventions on
the ground.

    

### [[2109.10645] Contrastive Learning for Fair Representations](http://arxiv.org/abs/2109.10645)


  Trained classification models can unintentionally lead to biased
representations and predictions, which can reinforce societal preconceptions
and stereotypes. Existing debiasing methods for classification models, such as
adversarial training, are often expensive to train and difficult to optimise.
In this paper, we propose a method for mitigating bias in classifier training
by incorporating contrastive learning, in which instances sharing the same
class label are encouraged to have similar representations, while instances
sharing a protected attribute are forced further apart. In such a way our
method learns representations which capture the task label in focused regions,
while ensuring the protected attribute has diverse spread, and thus has limited
impact on prediction and thereby results in fairer models. Extensive
experimental results across four tasks in NLP and computer vision show (a) that
our proposed method can achieve fairer representations and realises bias
reductions compared with competitive baselines; and (b) that it can do so
without sacrificing main task performance; (c) that it sets a new
state-of-the-art performance in one task despite reducing the bias. Finally,
our method is conceptually simple and agnostic to network architectures, and
incurs minimal additional compute cost.

    

### [[2109.10649] Caption Enriched Samples for Improving Hateful Memes Detection](http://arxiv.org/abs/2109.10649)


  The recently introduced hateful meme challenge demonstrates the difficulty of
determining whether a meme is hateful or not. Specifically, both unimodal
language models and multimodal vision-language models cannot reach the human
level of performance. Motivated by the need to model the contrast between the
image content and the overlayed text, we suggest applying an off-the-shelf
image captioning tool in order to capture the first. We demonstrate that the
incorporation of such automatic captions during fine-tuning improves the
results for various unimodal and multimodal models. Moreover, in the unimodal
case, continuing the pre-training of language models on augmented and original
caption pairs, is highly beneficial to the classification accuracy.

    

### [[2109.10691] Query Evaluation in DatalogMTL -- Taming Infinite Query Results](http://arxiv.org/abs/2109.10691)


  In this paper, we investigate finite representations of DatalogMTL. First, we
introduce programs that have finite models and propose a toolkit for
structuring the execution of DatalogMTL rules into sequential phases. Then, we
study infinite models that eventually become constant and introduce sufficient
criteria for programs that allow for such representation. We proceed by
considering infinite models that are eventually periodic and show that such a
representation encompasses all DatalogMTLFP programs, a widely discussed
fragment. Finally, we provide a novel algorithm for reasoning over finite
representable DatalogMTL programs that incorporates all of the previously
discussed representations.

    

### [[2109.10698] Complementing the Linear-Programming Learning Experience with the Design and Use of Computerized Games: The Formula 1 Championship Game](http://arxiv.org/abs/2109.10698)


  This document focuses on modeling a complex situations to achieve an
advantage within a competitive context. Our goal is to devise the
characteristics of games to teach and exercise non-easily quantifiable tasks
crucial to the math-modeling process. A computerized game to exercise the
math-modeling process and optimization problem formulation is introduced. The
game is named The Formula 1 Championship, and models of the game were developed
in the computerized simulation platform MoNet. It resembles some situations in
which team managers must make crucial decisions to enhance their racing cars up
to the feasible, most advantageous conditions. This paper describes the game's
rules, limitations, and five Formula 1 circuit simulators used for the
championship development. We present several formulations of this situation in
the form of optimization problems. Administering the budget to reach the best
car adjustment to a set of circuits to win the respective races can be an
approach. Focusing on the best distribution of each Grand Prix's budget and
then deciding how to use the assigned money to improve the car is also the
right approach. In general, there may be a degree of conflict among these
approaches because they are different aspects of the same multi-scale
optimization problem. Therefore, we evaluate the impact of assigning the
highest priority to an element, or another, when formulating the optimization
problem. Studying the effectiveness of solving such optimization problems turns
out to be an exciting way of evaluating the advantages of focusing on one scale
or another. Another thread of this research directs to the meaning of the game
in the teaching-learning process. We believe applying the Formula 1 Game is an
effective way to discover opportunities in a complex-system situation and
formulate them to finally extract and concrete the related benefit to the
context described.

    

### [[2109.10716] A formalisation of BPMN in Description Logics](http://arxiv.org/abs/2109.10716)


  In this paper we present a textual description, in terms of Description
Logics, of the BPMN Ontology, which provides a clear semantic formalisation of
the structural components of the Business Process Modelling Notation (BPMN),
based on the latest stable BPMN specifications from OMG [BPMN Version 1.1 --
January 2008]. The development of the ontology was guided by the description of
the complete set of BPMN Element Attributes and Types contained in Annex B of
the BPMN specifications.

    

### [[2109.10767] HybridSDF: Combining Free Form Shapes and Geometric Primitives for effective Shape Manipulation](http://arxiv.org/abs/2109.10767)


  CAD modeling typically involves the use of simple geometric primitives
whereas recent advances in deep-learning based 3D surface modeling have opened
new shape design avenues. Unfortunately, these advances have not yet been
accepted by the CAD community because they cannot be integrated into
engineering workflows. To remedy this, we propose a novel approach to
effectively combining geometric primitives and free-form surfaces represented
by implicit surfaces for accurate modeling that preserves interpretability,
enforces consistency, and enables easy manipulation.

    

### [[2109.10836] AI-HRI 2021 Proceedings](http://arxiv.org/abs/2109.10836)


  The Artificial Intelligence (AI) for Human-Robot Interaction (HRI) Symposium
has been a successful venue of discussion and collaboration since 2014. During
that time, these symposia provided a fertile ground for numerous collaborations
and pioneered many discussions revolving trust in HRI, XAI for HRI, service
robots, interactive learning, and more.
This year, we aim to review the achievements of the AI-HRI community in the
last decade, identify the challenges facing ahead, and welcome new researchers
who wish to take part in this growing community. Taking this wide perspective,
this year there will be no single theme to lead the symposium and we encourage
AI-HRI submissions from across disciplines and research interests. Moreover,
with the rising interest in AR and VR as part of an interaction and following
the difficulties in running physical experiments during the pandemic, this year
we specifically encourage researchers to submit works that do not include a
physical robot in their evaluation, but promote HRI research in general. In
addition, acknowledging that ethics is an inherent part of the human-robot
interaction, we encourage submissions of works on ethics for HRI. Over the
course of the two-day meeting, we will host a collaborative forum for
discussion of current efforts in AI-HRI, with additional talks focused on the
topics of ethics in HRI and ubiquitous HRI.

    

### [[2109.10859] Pushing the Right Buttons: Adversarial Evaluation of Quality Estimation](http://arxiv.org/abs/2109.10859)


  Current Machine Translation (MT) systems achieve very good results on a
growing variety of language pairs and datasets. However, they are known to
produce fluent translation outputs that can contain important meaning errors,
thus undermining their reliability in practice. Quality Estimation (QE) is the
task of automatically assessing the performance of MT systems at test time.
Thus, in order to be useful, QE systems should be able to detect such errors.
However, this ability is yet to be tested in the current evaluation practices,
where QE systems are assessed only in terms of their correlation with human
judgements. In this work, we bridge this gap by proposing a general methodology
for adversarial testing of QE for MT. First, we show that despite a high
correlation with human judgements achieved by the recent SOTA, certain types of
meaning errors are still problematic for QE to detect. Second, we show that on
average, the ability of a given model to discriminate between
meaning-preserving and meaning-altering perturbations is predictive of its
overall performance, thus potentially allowing for comparing QE systems without
relying on manual quality annotation.

    

### [[2109.10900] Towards Multi-Agent Reinforcement Learning using Quantum Boltzmann Machines](http://arxiv.org/abs/2109.10900)


  Reinforcement learning has driven impressive advances in machine learning.
Simultaneously, quantum-enhanced machine learning algorithms using quantum
annealing underlie heavy developments. Recently, a multi-agent reinforcement
learning (MARL) architecture combining both paradigms has been proposed. This
novel algorithm, which utilizes Quantum Boltzmann Machines (QBMs) for Q-value
approximation has outperformed regular deep reinforcement learning in terms of
time-steps needed to converge. However, this algorithm was restricted to
single-agent and small 2x2 multi-agent grid domains. In this work, we propose
an extension to the original concept in order to solve more challenging
problems. Similar to classic DQNs, we add an experience replay buffer and use
different networks for approximating the target and policy values. The
experimental results show that learning becomes more stable and enables agents
to find optimal policies in grid-domains with higher complexity. Additionally,
we assess how parameter sharing influences the agents behavior in multi-agent
domains. Quantum sampling proves to be a promising method for reinforcement
learning tasks, but is currently limited by the QPU size and therefore by the
size of the input and Boltzmann machine.

    

### [[1910.05126] Prediction-based Resource Allocation using Bayesian Neural Networks and Minimum Cost and Maximum Flow Algorithm](http://arxiv.org/abs/1910.05126)


  Predictive business process monitoring aims at providing predictions about
running instances by analyzing logs of completed cases in a business process.
Recently, a lot of research focuses on increasing productivity and efficiency
in a business process by forecasting potential problems during its executions.
However, most of the studies lack suggesting concrete actions to improve the
process. They leave it up to the subjective judgment of a user. In this paper,
we propose a novel method to connect the results from predictive business
process monitoring to actual business process improvements. More in detail, we
optimize the resource allocation in a non-clairvoyant online environment, where
we have limited information required for scheduling, by exploiting the
predictions. The proposed method integrates the offline prediction model
construction that predicts the processing time and the next activity of an
ongoing instance using Bayesian Neural Networks (BNNs) with the online resource
allocation that is extended from the minimum cost and maximum flow algorithm.
To validate the proposed method, we performed experiments using an artificial
event log and a real-life event log from a global financial organization.

    

### [[2109.08934] Fairness Maximization among Offline Agents in Online-Matching Markets](http://arxiv.org/abs/2109.08934)


  Matching markets involve heterogeneous agents (typically from two parties)
who are paired for mutual benefit. During the last decade, matching markets
have emerged and grown rapidly through the medium of the Internet. They have
evolved into a new format, called Online Matching Markets (OMMs), with examples
ranging from crowdsourcing to online recommendations to ridesharing. There are
two features distinguishing OMMs from traditional matching markets. One is the
dynamic arrival of one side of the market: we refer to these as online agents
while the rest are offline agents. Examples of online and offline agents
include keywords (online) and sponsors (offline) in Google Advertising; workers
(online) and tasks (offline) in Amazon Mechanical Turk (AMT); riders (online)
and drivers (offline when restricted to a short time window) in ridesharing.
The second distinguishing feature of OMMs is the real-time decision-making
element. However, studies have shown that the algorithms making decisions in
these OMMs leave disparities in the match rates of offline agents. For example,
tasks in neighborhoods of low socioeconomic status rarely get matched to gig
workers, and drivers of certain races/genders get discriminated against in
matchmaking. In this paper, we propose online matching algorithms which
optimize for either individual or group-level fairness among offline agents in
OMMs. We present two linear-programming (LP) based sampling algorithms, which
achieve online competitive ratios at least 0.725 for individual fairness
maximization (IFM) and 0.719 for group fairness maximization (GFM),
respectively. We conduct extensive numerical experiments and results show that
our boosted version of sampling algorithms are not only conceptually easy to
implement but also highly effective in practical instances of
fairness-maximization-related models.

    

### [[2109.10241] Life, the universe and the hidden meaning of everything](http://arxiv.org/abs/2109.10241)


  It is hard to look at the universe and not wonder about the meaning, of,
well, everything. A natural question is whether what we see is a sign of
intelligent design. The antithesis of design would be a random universe or,
assuming laws of physics, one whose fundamental physical parameters were
randomly selected, but conditioned on life (ourselves) being here to observe
it. In unpublished work, the British physicist Dennis Sciama argued that such a
randomly selected universe would display a statistical signature. He concluded
that a random universe would almost certainly have parameters only just
allowing for the possibility of life. Here we consider whether this signature
is definitive. We find that with plausible additional assumptions Sciama's
signature would appear to reverse: Were our universe random, it could give the
false impression of being intelligently designed, with the fundamental
constants appearing to be fine-tuned to a strong probability for life to emerge
and be maintained.

    