
## 2021-10-27

### [[2110.13300] Adaptive Probabilistic Model for Energy-Efficient Distance-based Clustering in WSNs (Adapt-P): A LEACH-based Analytical Study](http://arxiv.org/abs/2110.13300)


  Network lifetime and energy consumption of data transmission have been
primary Quality of Service (QoS) obligations in Wireless Sensor Networks
(WSNs). The environment of a WSN is often organized into clusters to mitigate
the management complexity of such obligations. However, the distance between
Sensor Nodes (SNs) and the number of clusters per round are vital factors that
affect QoS performance of a WSN. A designer's conundrum resolves around the
desire to sustain a balance between the limited residual energy of SNs and the
demand for prolonged network lifetime. Any imbalance in controlling such
objectives results in either QoS penalties due to draining SN energies, or an
over-cost environment that is significantly difficult to distribute and
operate. Low-Energy Adaptive Clustering Hierarchy (LEACH) is a distributed
algorithm proposed to tackle such difficulties. Proposed LEACH-based algorithms
focus on residual energies of SNs to compute a probability function that
selects cluster-heads and an optimal energy-efficient path toward a destination
SN. Nevertheless, these algorithms do not consider variations in network's
state at run-time. Such a state changes in an adaptive manner according to
existing network structures and conditions. Thus, cluster-heads per round are
not elected adaptively depending on the state and distances between SNs. This
paper proposes an energy-efficient adaptive distance-based clustering called
Adapt-P, in which an adaptive probability function is developed to formulate
clusters. A near-optimal distance between each cluster-head and its
cluster-members is formulated so that energy consumption of the network is
mitigated and network lifetime is maximized. The cluster-head selection
probability is adapted at the end of each round based on the maximum number of
cluster-heads permitted per round found a priori and the number of alive SNs in
the network.

    

### [[2110.13390] Novel Binary Addition Tree Algorithm (BAT) for Calculating the Direct Lower-Bound of the Highly Reliable Binary-State Network Reliability](http://arxiv.org/abs/2110.13390)


  Real-world applications such as the internet of things, wireless sensor
networks, smart grids, transportation networks, communication networks, social
networks, and computer grid systems are typically modeled as network
structures. Network reliability represents the success probability of a network
and it is an effective and popular metric for evaluating the performance of all
types of networks. Binary-state networks composed of binary-state (e.g.,
working or failed) components (arcs and/or nodes) are some of the most popular
network structures. The scale of networks has grown dramatically in recent
years. For example, social networks have more than a billion users.
Additionally, the reliability of components has increased as a result of both
mature and emergent technology. For highly reliable networks, it is more
practical to calculate approximated reliability, rather than exact reliability,
which is an NP-hard problem. Therefore, we propose a novel direct reliability
lower bound based on the binary addition tree algorithm to calculate
approximate reliability. The efficiency and effectiveness of the proposed
reliability bound are analyzed based on time complexity and validated through
numerical experiments.

    

### [[2110.13392] Graph-based Heuristic Solution for Placing Distributed Video Processing Applications on Moving Vehicle Clusters](http://arxiv.org/abs/2110.13392)


  Vehicular fog computing (VFC) is envisioned as an extension of cloud and
mobile edge computing to utilize the rich sensing and processing resources
available in vehicles. We focus on slow-moving cars that spend a significant
time in urban traffic congestion as a potential pool of on-board sensors, video
cameras, and processing capacity. For leveraging the dynamic network and
processing resources, we utilize a stochastic mobility model to select nodes
with similar mobility patterns. We then design two distributed applications
that are scaled in real-time and placed as multiple instances on selected
vehicular fog nodes. We handle the unstable vehicular environment by a), Using
real vehicle density data to build a realistic mobility model that helps in
selecting nodes for service deployment b), Using community-detection algorithms
for selecting a robust vehicular cluster using the predicted mobility behavior
of vehicles. The stability of the chosen cluster is validated using a graph
centrality measure, and c), Graph-based placement heuristics are developed to
find the optimal placement of service graphs based on a multi-objective
constrained optimization problem with the objective of efficient resource
utilization. The heuristic solves an important problem of processing data
generated from distributed devices by balancing the trade-off between
increasing the number of service instances to have enough redundancy of
processing instances to increase resilience in the service in case of node or
link failure, versus reducing their number to minimise resource usage. We
compare our heuristic to an integer linear program solution and a first-fit
heuristic. Our approach performs better than these comparable schemes in terms
of resource utilization and/or has a lesser service latency, which is a crucial
requirement for safety-related applications.

    

### [[2110.13621] Model-based Reinforcement Learning for Service Mesh Fault Resiliency in a Web Application-level](http://arxiv.org/abs/2110.13621)


  Microservice-based architectures enable different aspects of web applications
to be created and updated independently, even after deployment. Associated
technologies such as service mesh provide application-level fault resilience
through attribute configurations that govern the behavior of request-response
service -- and the interactions among them -- in the presence of failures.
While this provides tremendous flexibility, the configured values of these
attributes -- and the relationships among them -- can significantly affect the
performance and fault resilience of the overall application. Furthermore, it is
impossible to determine the best and worst combinations of attribute values
with respect to fault resiliency via testing, due to the complexities of the
underlying distributed system and the many possible attribute value
combinations. In this paper, we present a model-based reinforcement learning
workflow towards service mesh fault resiliency. Our approach enables the
prediction of the most significant fault resilience behaviors at a web
application-level, scratching from single service to aggregated multi-service
management with efficient agent collaborations.

    

### [[2110.13871] LayerZero: Trustless Omnichain Interoperability Protocol](http://arxiv.org/abs/2110.13871)


  The proliferation of blockchains has given developers a variety of platforms
on which to run their smart contracts based on application features and
requirements for throughput, security, and cost. However, a consequence of this
freedom is severe fragmentation; Each chain is isolated, forcing users to silo
their liquidity and limiting options to move liquidity and state between walled
ecosystems. This paper presents LayerZero, the first trustless omnichain
interoperability protocol, which provides a powerful, low level communication
primitive upon which a diverse set of cross-chain applications can be built.
Using this new primitive, developers can implement seamless inter-chain
applications like a cross-chain DEX or multi-chain yield aggregator without
having to rely on a trusted custodian or intermediate transactions. Simply put,
LayerZero is the first system to trustlessly enable direct transactions across
all chains. Allowing transactions to flow freely between chains provides
opportunities for users to consolidate fragmented pockets of liquidity while
also making full use of applications on separate chains. With LayerZero, we
provide the network fabric underlying the fully-connected omnichain ecosystem
of the future.

    

### [[2106.15905] Faithful Edge Federated Learning: Scalability and Privacy](http://arxiv.org/abs/2106.15905)


  Federated learning enables machine learning algorithms to be trained over a
network of multiple decentralized edge devices without requiring the exchange
of local datasets. Successfully deploying federated learning requires ensuring
that agents (e.g., mobile devices) faithfully execute the intended algorithm,
which has been largely overlooked in the literature. In this study, we first
use risk bounds to analyze how the key feature of federated learning,
unbalanced and non-i.i.d. data, affects agents' incentives to voluntarily
participate and obediently follow traditional federated learning algorithms. To
be more specific, our analysis reveals that agents with less typical data
distributions and relatively more samples are more likely to opt out of or
tamper with federated learning algorithms. To this end, we formulate the first
faithful implementation problem of federated learning and design two faithful
federated learning mechanisms which satisfy economic properties, scalability,
and privacy. Further, the time complexity of computing all agents' payments in
the number of agents is $\mathcal{O}(1)$. First, we design a Faithful Federated
Learning (FFL) mechanism which approximates the Vickrey-Clarke-Groves (VCG)
payments via an incremental computation. We show that it achieves (probably
approximate) optimality, faithful implementation, voluntary participation, and
some other economic properties (such as budget balance). Second, by
partitioning agents into several subsets, we present a scalable VCG mechanism
approximation. We further design a scalable and Differentially Private FFL
(DP-FFL) mechanism, the first differentially private faithful mechanism, that
maintains the economic properties. Our mechanism enables one to make three-way
performance tradeoffs among privacy, the iterations needed, and payment
accuracy loss.

    

### [[2110.04997] IoT Equipped Intelligent Distributed Framework for Smart Healthcare Systems](http://arxiv.org/abs/2110.04997)


  The fundamental aim of the healthcare sector is to incorporate different
technologies to observe and keep a track of the various clinical parameters of
the patients in day to day life. Distant patient observation applications are
becoming popular as economical healthcare services are facilitated by these
apps. The process of data management gathered through these applications also
require due attention. Although cloud facilitated healthcare applications cater
a variety of solutions to store patients record and deliver the required data
as per need of all the stakeholders but are affected by security issues, more
response time and affecting the continues availability of the system. To
overcome these challenges, an intelligent IoT based distributed framework to
deploy remote healthcare services is proposed in this chapter. In the proposed
model, various entities of the system are interconnected using IoT and
Distributed Database Management Systems is used to cater secure and fast data
availability to the patients and health care workers. The concept of Blockchain
is used to ensure the security of the patient medical records. The proposed
model will comprise of intelligent analysis of the clinical records fetched
from Distributed Database Management Systems secured with Blockchain. Proposed
model is tested with true clinical data and results are discussed in detail.

    

### [[2110.13142] Light-Field Microscopy for optical imaging of neuronal activity: when model-based methods meet data-driven approaches](http://arxiv.org/abs/2110.13142)


  Understanding how networks of neurons process information is one of the key
challenges in modern neuroscience. A necessary step to achieve this goal is to
be able to observe the dynamics of large populations of neurons over a large
area of the brain. Light-field microscopy (LFM), a type of scanless microscope,
is a particularly attractive candidate for high-speed three-dimensional (3D)
imaging. It captures volumetric information in a single snapshot, allowing
volumetric imaging at video frame-rates. Specific features of imaging neuronal
activity using LFM call for the development of novel machine learning
approaches that fully exploit priors embedded in physics and optics models.
Signal processing theory and wave-optics theory could play a key role in
filling this gap, and contribute to novel computational methods with enhanced
interpretability and generalization by integrating model-driven and data-driven
approaches. This paper is devoted to a comprehensive survey to state-of-the-art
of computational methods for LFM, with a focus on model-based and data-driven
approaches.

    

### [[2110.13144] Faster Perturbed Stochastic Gradient Methods for Finding Local Minima](http://arxiv.org/abs/2110.13144)


  Escaping from saddle points and finding local minima is a central problem in
nonconvex optimization. Perturbed gradient methods are perhaps the simplest
approach for this problem. However, to find $(\epsilon,
\sqrt{\epsilon})$-approximate local minima, the existing best stochastic
gradient complexity for this type of algorithms is $\tilde O(\epsilon^{-3.5})$,
which is not optimal. In this paper, we propose \texttt{Pullback}, a faster
perturbed stochastic gradient framework for finding local minima. We show that
Pullback with stochastic gradient estimators such as SARAH/SPIDER and STORM can
find $(\epsilon, \epsilon_{H})$-approximate local minima within $\tilde
O(\epsilon^{-3} + \epsilon_{H}^{-6})$ stochastic gradient evaluations (or
$\tilde O(\epsilon^{-3})$ when $\epsilon_H = \sqrt{\epsilon}$). The core idea
of our framework is a step-size ``pullback'' scheme to control the average
movement of the iterates, which leads to faster convergence to the local
minima. Experiments on matrix factorization problems corroborate our theory.

    

### [[2110.13162] Quantum machine learning beyond kernel methods](http://arxiv.org/abs/2110.13162)


  With noisy intermediate-scale quantum computers showing great promise for
near-term applications, a number of machine learning algorithms based on
parametrized quantum circuits have been suggested as possible means to achieve
learning advantages. Yet, our understanding of how these quantum machine
learning models compare, both to existing classical models and to each other,
remains limited. A big step in this direction has been made by relating them to
so-called kernel methods from classical machine learning. By building on this
connection, previous works have shown that a systematic reformulation of many
quantum machine learning models as kernel models was guaranteed to improve
their training performance. In this work, we first extend the applicability of
this result to a more general family of parametrized quantum circuit models
called data re-uploading circuits. Secondly, we show, through simple
constructions and numerical simulations, that models defined and trained
variationally can exhibit a critically better generalization performance than
their kernel formulations, which is the true figure of merit of machine
learning tasks. Our results constitute another step towards a more
comprehensive theory of quantum machine learning models next to kernel
formulations.

    

### [[2110.13179] Probabilistic Hierarchical Forecasting with Deep Poisson Mixtures](http://arxiv.org/abs/2110.13179)


  Hierarchical forecasting problems arise when time series compose a group
structure that naturally defines aggregation and disaggregation coherence
constraints for the predictions. In this work, we explore a new forecast
representation, the Poisson Mixture Mesh (PMM), that can produce probabilistic,
coherent predictions; it is compatible with the neural forecasting innovations,
and defines simple aggregation and disaggregation rules capable of
accommodating hierarchical structures, unknown during its optimization. We
performed an empirical evaluation to compare the PMM \ to other hierarchical
forecasting methods on Australian domestic tourism data, where we obtain a 20
percent relative improvement.

    

### [[2110.13188] Multi-Task Meta-Learning Modification with Stochastic Approximation](http://arxiv.org/abs/2110.13188)


  Meta-learning methods aim to build learning algorithms capable of quickly
adapting to new tasks in low-data regime. One of the main benchmarks of such an
algorithms is a few-shot learning problem. In this paper we investigate the
modification of standard meta-learning pipeline that takes a multi-task
approach during training. The proposed method simultaneously utilizes
information from several meta-training tasks in a common loss function. The
impact of each of these tasks in the loss function is controlled by the
corresponding weight. Proper optimization of these weights can have a big
influence on training of the entire model and might improve the quality on test
time tasks. In this work we propose and investigate the use of methods from the
family of simultaneous perturbation stochastic approximation (SPSA) approaches
for meta-train tasks weights optimization. We have also compared the proposed
algorithms with gradient-based methods and found that stochastic approximation
demonstrates the largest quality boost in test time. Proposed multi-task
modification can be applied to almost all methods that use meta-learning
pipeline. In this paper we study applications of this modification on
Prototypical Networks and Model-Agnostic Meta-Learning algorithms on CIFAR-FS,
FC100, tieredImageNet and miniImageNet few-shot learning benchmarks. During
these experiments, multi-task modification has demonstrated improvement over
original methods. The proposed SPSA-Tracking algorithm shows the largest
accuracy boost. Our code is available online.

    

### [[2110.13189] Spectral unmixing of Raman microscopic images of single human cells using Independent Component Analysis](http://arxiv.org/abs/2110.13189)


  Application of independent component analysis (ICA) as an unmixing and image
clustering technique for high spatial resolution Raman maps is reported. A
hyperspectral map of a fixed human cell was collected by a Raman micro
spectrometer in a raster pattern on a 0.5um grid. Unlike previously used
unsupervised machine learning techniques such as principal component analysis,
ICA is based on non-Gaussianity and statistical independence of data which is
the case for mixture Raman spectra. Hence, ICA is a great candidate for
assembling pseudo-colour maps from the spectral hypercube of Raman spectra. Our
experimental results revealed that ICA is capable of reconstructing false
colour maps of Raman hyperspectral data of human cells, showing the nuclear
region constituents as well as subcellular organelle in the cytoplasm and
distribution of mitochondria in the perinuclear region. Minimum preprocessing
requirements and label-free nature of the ICA method make it a great unmixed
method for extraction of endmembers in Raman hyperspectral maps of living
cells.

    

### [[2110.13194] Covariance-Generalized Matching Component Analysis for Data Fusion and Transfer Learning](http://arxiv.org/abs/2110.13194)


  In order to allow for the encoding of additional statistical information in
data fusion and transfer learning applications, we introduce a generalized
covariance constraint for the matching component analysis (MCA) transfer
learning technique. After proving a semi-orthogonally constrained trace
maximization lemma, we develop a closed-form solution to the resulting
covariance-generalized optimization problem and provide an algorithm for its
computation. We call this technique -- applicable to both data fusion and
transfer learning -- covariance-generalized MCA (CGMCA).

    

### [[2110.13197] Nested Graph Neural Networks](http://arxiv.org/abs/2110.13197)


  Graph neural network (GNN)'s success in graph classification is closely
related to the Weisfeiler-Lehman (1-WL) algorithm. By iteratively aggregating
neighboring node features to a center node, both 1-WL and GNN obtain a node
representation that encodes a rooted subtree around the center node. These
rooted subtree representations are then pooled into a single representation to
represent the whole graph. However, rooted subtrees are of limited
expressiveness to represent a non-tree graph. To address it, we propose Nested
Graph Neural Networks (NGNNs). NGNN represents a graph with rooted subgraphs
instead of rooted subtrees, so that two graphs sharing many identical subgraphs
(rather than subtrees) tend to have similar representations. The key is to make
each node representation encode a subgraph around it more than a subtree. To
achieve this, NGNN extracts a local subgraph around each node and applies a
base GNN to each subgraph to learn a subgraph representation. The whole-graph
representation is then obtained by pooling these subgraph representations. We
provide a rigorous theoretical analysis showing that NGNN is strictly more
powerful than 1-WL. In particular, we proved that NGNN can discriminate almost
all r-regular graphs, where 1-WL always fails. Moreover, unlike other more
powerful GNNs, NGNN only introduces a constant-factor higher time complexity
than standard GNNs. NGNN is a plug-and-play framework that can be combined with
various base GNNs. We test NGNN with different base GNNs on several benchmark
datasets. NGNN uniformly improves their performance and shows highly
competitive performance on all datasets.

    

### [[2110.13202] Transportation Scenario Planning with Graph Neural Networks](http://arxiv.org/abs/2110.13202)


  Providing efficient human mobility services and infrastructure is one of the
major concerns of most mid-sized to large cities around the world. A proper
understanding of the dynamics of commuting flows is, therefore, a requisite to
better plan urban areas. In this context, an important task is to study
hypothetical scenarios in which possible future changes are evaluated. For
instance, how the increase in residential units or transportation modes in a
neighborhood will change the commuting flows to or from that region? In this
paper, we propose to leverage GMEL, a recently introduced graph neural network
model, to evaluate changes in commuting flows taking into account different
land use and infrastructure scenarios. We validate the usefulness of our
methodology through real-world case studies set in two large cities in Brazil.

    

### [[2110.13205] A Probabilistic Framework for Knowledge Graph Data Augmentation](http://arxiv.org/abs/2110.13205)


  We present NNMFAug, a probabilistic framework to perform data augmentation
for the task of knowledge graph completion to counter the problem of data
scarcity, which can enhance the learning process of neural link predictors. Our
method can generate potentially diverse triples with the advantage of being
efficient and scalable as well as agnostic to the choice of the link prediction
model and dataset used. Experiments and analysis done on popular models and
benchmarks show that NNMFAug can bring notable improvements over the baselines.

    

### [[2110.13214] IconQA: A New Benchmark for Abstract Diagram Understanding and Visual Language Reasoning](http://arxiv.org/abs/2110.13214)


  Current visual question answering (VQA) tasks mainly consider answering
human-annotated questions for natural images. However, aside from natural
images, abstract diagrams with semantic richness are still understudied in
visual understanding and reasoning research. In this work, we introduce a new
challenge of Icon Question Answering (IconQA) with the goal of answering a
question in an icon image context. We release IconQA, a large-scale dataset
that consists of 107,439 questions and three sub-tasks: multi-image-choice,
multi-text-choice, and filling-in-the-blank. The IconQA dataset is inspired by
real-world diagram word problems that highlight the importance of abstract
diagram understanding and comprehensive cognitive reasoning. Thus, IconQA
requires not only perception skills like object recognition and text
understanding, but also diverse cognitive reasoning skills, such as geometric
reasoning, commonsense reasoning, and arithmetic reasoning. To facilitate
potential IconQA models to learn semantic representations for icon images, we
further release an icon dataset Icon645 which contains 645,687 colored icons on
377 classes. We conduct extensive user studies and blind experiments and
reproduce a wide range of advanced VQA methods to benchmark the IconQA task.
Also, we develop a strong IconQA baseline Patch-TRM that applies a pyramid
cross-modal Transformer with input diagram embeddings pre-trained on the icon
dataset. IconQA and Icon645 are available at this https URL.

    

### [[2110.13217] RBSRICNN: Raw Burst Super-Resolution through Iterative Convolutional Neural Network](http://arxiv.org/abs/2110.13217)


  Modern digital cameras and smartphones mostly rely on image signal processing
(ISP) pipelines to produce realistic colored RGB images. However, compared to
DSLR cameras, low-quality images are usually obtained in many portable mobile
devices with compact camera sensors due to their physical limitations. The
low-quality images have multiple degradations i.e., sub-pixel shift due to
camera motion, mosaick patterns due to camera color filter array,
low-resolution due to smaller camera sensors, and the rest information are
corrupted by the noise. Such degradations limit the performance of current
Single Image Super-resolution (SISR) methods in recovering high-resolution (HR)
image details from a single low-resolution (LR) image. In this work, we propose
a Raw Burst Super-Resolution Iterative Convolutional Neural Network (RBSRICNN)
that follows the burst photography pipeline as a whole by a forward (physical)
model. The proposed Burst SR scheme solves the problem with classical image
regularization, convex optimization, and deep learning techniques, compared to
existing black-box data-driven methods. The proposed network produces the final
output by an iterative refinement of the intermediate SR estimates. We
demonstrate the effectiveness of our proposed approach in quantitative and
qualitative experiments that generalize robustly to real LR burst inputs with
onl synthetic burst data available for training.

    

### [[2110.13220] Demystifying and Generalizing BinaryConnect](http://arxiv.org/abs/2110.13220)


  BinaryConnect (BC) and its many variations have become the de facto standard
for neural network quantization. However, our understanding of the inner
workings of BC is still quite limited. We attempt to close this gap in four
different aspects: (a) we show that existing quantization algorithms, including
post-training quantization, are surprisingly similar to each other; (b) we
argue for proximal maps as a natural family of quantizers that is both easy to
design and analyze; (c) we refine the observation that BC is a special case of
dual averaging, which itself is a special case of the generalized conditional
gradient algorithm; (d) consequently, we propose ProxConnect (PC) as a
generalization of BC and we prove its convergence properties by exploiting the
established connections. We conduct experiments on CIFAR-10 and ImageNet, and
verify that PC achieves competitive performance.

    

### [[2110.13221] Prediction-focused Mixture Models](http://arxiv.org/abs/2110.13221)


  In several applications, besides getting a generative model of the data, we
also want the model to be useful for specific downstream tasks. Mixture models
are useful for identifying discrete components in the data, but may not
identify components useful for downstream tasks if misspecified; further,
current inference techniques often fail to overcome misspecification even when
a supervisory signal is provided. We introduce the prediction-focused mixture
model, which selects and models input features relevant to predicting the
targets. We demonstrate that our approach identifies relevant signal from
inputs even when the model is highly misspecified.

    

### [[2110.13223] Identifying and Benchmarking Natural Out-of-Context Prediction Problems](http://arxiv.org/abs/2110.13223)


  Deep learning systems frequently fail at out-of-context (OOC) prediction, the
problem of making reliable predictions on uncommon or unusual inputs or
subgroups of the training distribution. To this end, a number of benchmarks for
measuring OOC performance have recently been introduced. In this work, we
introduce a framework unifying the literature on OOC performance measurement,
and demonstrate how rich auxiliary information can be leveraged to identify
candidate sets of OOC examples in existing datasets. We present NOOCh: a suite
of naturally-occurring "challenge sets", and show how varying notions of
context can be used to probe specific OOC failure modes. Experimentally, we
explore the tradeoffs between various learning approaches on these challenge
sets and demonstrate how the choices made in designing OOC benchmarks can yield
varying conclusions.

    

### [[2110.13228] Variational framework for partially-measured physical system control: examples of vision neuroscience and optical random media](http://arxiv.org/abs/2110.13228)


  To characterize a physical system to behave as desired, either its underlying
governing rules must be known a priori or the system itself be accurately
measured. The complexity of full measurements of the system scales with its
size. When exposed to real-world conditions, such as perturbations or
time-varying settings, the system calibrated for a fixed working condition
might require non-trivial re-calibration, a process that could be prohibitively
expensive, inefficient and impractical for real-world use cases. In this work,
we propose a learning procedure to obtain a desired target output from a
physical system. We use Variational Auto-Encoders (VAE) to provide a generative
model of the system function and use this model to obtain the required input of
the system that produces the target output. We showcase the applicability of
our method for two datasets in optical physics and neuroscience.

    

### [[2110.13229] Distributionally Robust Recurrent Decoders with Random Network Distillation](http://arxiv.org/abs/2110.13229)


  Neural machine learning models can successfully model language that is
similar to their training distribution, but they are highly susceptible to
degradation under distribution shift, which occurs in many practical
applications when processing out-of-domain (OOD) text. This has been attributed
to "shortcut learning": relying on weak correlations over arbitrary large
contexts.
We propose a method based on OOD detection with Random Network Distillation
to allow an autoregressive language model to automatically disregard OOD
context during inference, smoothly transitioning towards a less expressive but
more robust model as the data becomes more OOD while retaining its full context
capability when operating in-distribution. We apply our method to a GRU
architecture, demonstrating improvements on multiple language modeling (LM)
datasets.

    

### [[2110.13233] Decomposed Inductive Procedure Learning](http://arxiv.org/abs/2110.13233)


  Recent advances in machine learning have made it possible to train
artificially intelligent agents that perform with super-human accuracy on a
great diversity of complex tasks. However, the process of training these
capabilities often necessitates millions of annotated examples -- far more than
humans typically need in order to achieve a passing level of mastery on similar
tasks. Thus, while contemporary methods in machine learning can produce agents
that exhibit super-human performance, their rate of learning per opportunity in
many domains is decidedly lower than human-learning. In this work we formalize
a theory of Decomposed Inductive Procedure Learning (DIPL) that outlines how
different forms of inductive symbolic learning can be used in combination to
build agents that learn educationally relevant tasks such as mathematical, and
scientific procedures, at a rate similar to human learners. We motivate the
construction of this theory along Marr's concepts of the computational,
algorithmic, and implementation levels of cognitive modeling, and outline at
the computational-level six learning capacities that must be achieved to
accurately model human learning. We demonstrate that agents built along the
DIPL theory are amenable to satisfying these capacities, and demonstrate, both
empirically and theoretically, that DIPL enables the creation of agents that
exhibit human-like learning performance.

    

### [[2110.13240] Integrative Clustering of Multi-View Data by Nonnegative Matrix Factorization](http://arxiv.org/abs/2110.13240)


  Learning multi-view data is an emerging problem in machine learning research,
and nonnegative matrix factorization (NMF) is a popular
dimensionality-reduction method for integrating information from multiple
views. These views often provide not only consensus but also diverse
information. However, most multi-view NMF algorithms assign equal weight to
each view or tune the weight via line search empirically, which can be
computationally expensive or infeasible without any prior knowledge of the
views. In this paper, we propose a weighted multi-view NMF (WM-NMF) algorithm.
In particular, we aim to address the critical technical gap, which is to learn
both view-specific and observation-specific weights to quantify each view's
information content. The introduced weighting scheme can alleviate unnecessary
views' adverse effects and enlarge the positive effects of the important views
by assigning smaller and larger weights, respectively. In addition, we provide
theoretical investigations about the convergence, perturbation analysis, and
generalization error of the WM-NMF algorithm. Experimental results confirm the
effectiveness and advantages of the proposed algorithm in terms of achieving
better clustering performance and dealing with the corrupted data compared to
the existing algorithms.

    

### [[2110.13241] Multitask Adaptation by Retrospective Exploration with Learned World Models](http://arxiv.org/abs/2110.13241)


  Model-based reinforcement learning (MBRL) allows solving complex tasks in a
sample-efficient manner. However, no information is reused between the tasks.
In this work, we propose a meta-learned addressing model called RAMa that
provides training samples for the MBRL agent taken from continuously growing
task-agnostic storage. The model is trained to maximize the expected agent's
performance by selecting promising trajectories solving prior tasks from the
storage. We show that such retrospective exploration can accelerate the
learning process of the MBRL agent by better informing learned dynamics and
prompting agent with exploratory trajectories. We test the performance of our
approach on several domains from the DeepMind control suite, from Metaworld
multitask benchmark, and from our bespoke environment implemented with a
robotic NVIDIA Isaac simulator to test the ability of the model to act in a
photorealistic, ray-traced environment.

    

### [[2110.13244] DeepHelp: Deep Learning for Shout Crisis Text Conversations](http://arxiv.org/abs/2110.13244)


  The Shout Crisis Text Line provides individuals undergoing mental health
crises an opportunity to have an anonymous text message conversation with a
trained Crisis Volunteer (CV). This project partners with Shout and its parent
organisation, Mental Health Innovations, to explore the applications of Machine
Learning in understanding Shout's conversations and improving its service. The
overarching aim of this project is to develop a proof-of-concept model to
demonstrate the potential of applying deep learning to crisis text messages.
Specifically, this project aims to use deep learning to (1) predict an
individual's risk of suicide or self-harm, (2) assess conversation success and
CV skill using robust metrics, and (3) extrapolate demographic information from
a texter survey to conversations where the texter did not complete the survey.
To these ends, contributions to deep learning include a modified
Transformer-over-BERT model; a framework for multitask learning to improve
generalisation in the presence of sparse labels; and a mathematical model for
using imperfect machine learning models to estimate population parameters from
a biased training set.
Key results include a deep learning model with likely better performance at
predicting suicide risk than trained CVs and the ability to predict whether a
texter is 21 or under with 88.4% accuracy. We produce three metrics for
conversation success and evaluate the validity and usefulness for each.
Finally, reversal of participation bias provides evidence that women, who make
up 80.3% of conversations with an associated texter survey, make up closer to
73.5%- 74.8% of all conversations; and that if, after every conversation, the
texter had shared whether they found their conversation helpful, affirmative
answers would fall from 85.1% to 45.45% - 46.51%.

    

### [[2110.13252] CNNC: A Visual Analytics System for Comparative Studies of Deep Convolutional Neural Networks](http://arxiv.org/abs/2110.13252)


  The rapid development of Convolutional Neural Networks (CNNs) in recent years
has triggered significant breakthroughs in many machine learning (ML)
applications. The ability to understand and compare various CNN models
available is thus essential. The conventional approach with visualizing each
model's quantitative features, such as classification accuracy and
computational complexity, is not sufficient for a deeper understanding and
comparison of the behaviors of different models. Moreover, most of the existing
tools for assessing CNN behaviors only support comparison between two models
and lack the flexibility of customizing the analysis tasks according to user
needs. This paper presents a visual analytics system, CNN Comparator (CNNC),
that supports the in-depth inspection of a single CNN model as well as
comparative studies of two or more models. The ability to compare a larger
number of (e.g., tens of) models especially distinguishes our system from
previous ones. With a carefully designed model visualization and explaining
support, CNNC facilitates a highly interactive workflow that promptly presents
both quantitative and qualitative information at each analysis stage. We
demonstrate CNNC's effectiveness for assisting ML practitioners in evaluating
and comparing multiple CNN models through two use cases and one preliminary
evaluation study using the image classification tasks on the ImageNet dataset.

    

### [[2110.13254] Pediatric Otoscopy Video Screening with Shift Contrastive Anomaly Detection](http://arxiv.org/abs/2110.13254)


  Ear related concerns and symptoms represents the leading indication for
seeking pediatric healthcare attention. Despite the high incidence of such
encounters, the diagnostic process of commonly encountered disease of the
middle and external presents significant challenge. Much of this challenge
stems from the lack of cost effective diagnostic testing, which necessitating
the presence or absence of ear pathology to be determined clinically. Research
has however demonstrated considerable variation among clinicians in their
ability to accurately diagnose and consequently manage ear pathology. With
recent advances in computer vision and machine learning, there is an increasing
interest in helping clinicians to accurately diagnose middle and external ear
pathology with computer-aided systems. It has been shown that AI has the
capacity to analyse a single clinical image captured during examination of the
ear canal and eardrum from which it can determine the likelihood of a
pathognomonic pattern for a specific diagnosis being present. The capture of
such an image can however be challenging especially to inexperienced
clinicians. To help mitigate this technical challenge we have developed and
tested a method using video sequences. We present a two stage method that
first, identifies valid frames by detecting and extracting ear drum patches
from the video sequence, and second, performs the proposed shift contrastive
anomaly detection to flag the otoscopy video sequences as normal or abnormal.
Our method achieves an AUROC of 88.0% on the patient-level and also outperforms
the average of a group of 25 clinicians in a comparative study, which is the
largest of such published to date. We conclude that the presented method
achieves a promising first step towards automated analysis of otoscopy video.

    

### [[2110.13265] On the Second-order Convergence Properties of Random Search Methods](http://arxiv.org/abs/2110.13265)


  We study the theoretical convergence properties of random-search methods when
optimizing non-convex objective functions without having access to derivatives.
We prove that standard random-search methods that do not rely on second-order
information converge to a second-order stationary point. However, they suffer
from an exponential complexity in terms of the input dimension of the problem.
In order to address this issue, we propose a novel variant of random search
that exploits negative curvature by only relying on function evaluations. We
prove that this approach converges to a second-order stationary point at a much
faster rate than vanilla methods: namely, the complexity in terms of the number
of function evaluations is only linear in the problem dimension. We test our
algorithm empirically and find good agreements with our theoretical results.

    

### [[2110.13282] The Pareto Frontier of model selection for general Contextual Bandits](http://arxiv.org/abs/2110.13282)


  Recent progress in model selection raises the question of the fundamental
limits of these techniques. Under specific scrutiny has been model selection
for general contextual bandits with nested policy classes, resulting in a
COLT2020 open problem. It asks whether it is possible to obtain simultaneously
the optimal single algorithm guarantees over all policies in a nested sequence
of policy classes, or if otherwise this is possible for a trade-off
$\alpha\in[\frac{1}{2},1)$ between complexity term and time:
$\ln(|\Pi_m|)^{1-\alpha}T^\alpha$. We give a disappointing answer to this
question. Even in the purely stochastic regime, the desired results are
unobtainable. We present a Pareto frontier of up to logarithmic factors
matching upper and lower bounds, thereby proving that an increase in the
complexity term $\ln(|\Pi_m|)$ independent of $T$ is unavoidable for general
policy classes. As a side result, we also resolve a COLT2016 open problem
concerning second-order bounds in full-information games.

    

### [[2110.13285] Generative Flows as a General Purpose Solution for Inverse Problems](http://arxiv.org/abs/2110.13285)


  Due to the success of generative flows to model data distributions, they have
been explored in inverse problems. Given a pre-trained generative flow,
previous work proposed to minimize the 2-norm of the latent variables as a
regularization term in the main objective. The intuition behind it was to
ensure high likelihood latent variables, however this does not ensure the
generation of realistic samples as we show in our experiments. We therefore
propose a regularization term to directly produce high likelihood
reconstructions. Our hypothesis is that our method could make generative flows
a general-purpose solver for inverse problems. We evaluate our method in image
denoising, image deblurring, image inpainting, and image colorization. We
observe a compelling improvement of our method over prior works in the PSNR and
SSIM metrics.

    

### [[2110.13287] Towards Realistic Market Simulations: a Generative Adversarial Networks Approach](http://arxiv.org/abs/2110.13287)


  Simulated environments are increasingly used by trading firms and investment
banks to evaluate trading strategies before approaching real markets.
Backtesting, a widely used approach, consists of simulating experimental
strategies while replaying historical market scenarios. Unfortunately, this
approach does not capture the market response to the experimental agents'
actions. In contrast, multi-agent simulation presents a natural bottom-up
approach to emulating agent interaction in financial markets. It allows to set
up pools of traders with diverse strategies to mimic the financial market
trader population, and test the performance of new experimental strategies.
Since individual agent-level historical data is typically proprietary and not
available for public use, it is difficult to calibrate multiple market agents
to obtain the realism required for testing trading strategies. To addresses
this challenge we propose a synthetic market generator based on Conditional
Generative Adversarial Networks (CGANs) trained on real aggregate-level
historical data. A CGAN-based "world" agent can generate meaningful orders in
response to an experimental agent. We integrate our synthetic market generator
into ABIDES, an open source simulator of financial markets. By means of
extensive simulations we show that our proposal outperforms previous work in
terms of stylized facts reflecting market responsiveness and realism.

    

### [[2110.13290] Exploring System Performance of Continual Learning for Mobile and Embedded Sensing Applications](http://arxiv.org/abs/2110.13290)


  Continual learning approaches help deep neural network models adapt and learn
incrementally by trying to solve catastrophic forgetting. However, whether
these existing approaches, applied traditionally to image-based tasks, work
with the same efficacy to the sequential time series data generated by mobile
or embedded sensing systems remains an unanswered question.
To address this void, we conduct the first comprehensive empirical study that
quantifies the performance of three predominant continual learning schemes
(i.e., regularization, replay, and replay with examples) on six datasets from
three mobile and embedded sensing applications in a range of scenarios having
different learning complexities. More specifically, we implement an end-to-end
continual learning framework on edge devices. Then we investigate the
generalizability, trade-offs between performance, storage, computational costs,
and memory footprint of different continual learning methods.
Our findings suggest that replay with exemplars-based schemes such as iCaRL
has the best performance trade-offs, even in complex scenarios, at the expense
of some storage space (few MBs) for training examples (1% to 5%). We also
demonstrate for the first time that it is feasible and practical to run
continual learning on-device with a limited memory budget. In particular, the
latency on two types of mobile and embedded devices suggests that both
incremental learning time (few seconds - 4 minutes) and training time (1 - 75
minutes) across datasets are acceptable, as training could happen on the device
when the embedded device is charging thereby ensuring complete data privacy.
Finally, we present some guidelines for practitioners who want to apply a
continual learning paradigm for mobile sensing tasks.

    

### [[2110.13293] Emulation of physical processes with Emukit](http://arxiv.org/abs/2110.13293)


  Decision making in uncertain scenarios is an ubiquitous challenge in real
world systems. Tools to deal with this challenge include simulations to gather
information and statistical emulation to quantify uncertainty. The machine
learning community has developed a number of methods to facilitate decision
making, but so far they are scattered in multiple different toolkits, and
generally rely on a fixed backend. In this paper, we present Emukit, a highly
adaptable Python toolkit for enriching decision making under uncertainty.
Emukit allows users to: (i) use state of the art methods including Bayesian
optimization, multi-fidelity emulation, experimental design, Bayesian
quadrature and sensitivity analysis; (ii) easily prototype new decision making
methods for new problems. Emukit is agnostic to the underlying modeling
framework and enables users to use their own custom models. We show how Emukit
can be used on three exemplary case studies.

    

### [[2110.13297] Fast PDE-constrained optimization via self-supervised operator learning](http://arxiv.org/abs/2110.13297)


  Design and optimal control problems are among the fundamental, ubiquitous
tasks we face in science and engineering. In both cases, we aim to represent
and optimize an unknown (black-box) function that associates a
performance/outcome to a set of controllable variables through an experiment.
In cases where the experimental dynamics can be described by partial
differential equations (PDEs), such problems can be mathematically translated
into PDE-constrained optimization tasks, which quickly become intractable as
the number of control variables and the cost of experiments increases. In this
work we leverage physics-informed deep operator networks (DeepONets) -- a
self-supervised framework for learning the solution operator of parametric PDEs
-- to build fast and differentiable surrogates for rapidly solving
PDE-constrained optimization problems, even in the absence of any paired
input-output training data. The effectiveness of the proposed framework will be
demonstrated across different applications involving continuous functions as
control or design variables, including time-dependent optimal control of heat
transfer, and drag minimization of obstacles in Stokes flow. In all cases, we
observe that DeepONets can minimize high-dimensional cost functionals in a
matter of seconds, yielding a significant speed up compared to traditional
adjoint PDE solvers that are typically costly and limited to relatively
low-dimensional control/design parametrizations.

    

### [[2110.13303] Negotiating Networks in Oligopoly Markets for Price-Sensitive Products](http://arxiv.org/abs/2110.13303)


  We present a novel framework to learn functions that estimate decisions of
sellers and buyers simultaneously in an oligopoly market for a price-sensitive
product. In this setting, the aim of the seller network is to come up with a
price for a given context such that the expected revenue is maximized by
considering the buyer's satisfaction as well. On the other hand, the aim of the
buyer network is to assign probability of purchase to the offered price to
mimic the real world buyers' responses while also showing price sensitivity
through its action. In other words, rejecting the unnecessarily high priced
products. Similar to generative adversarial networks, this framework
corresponds to a minimax two-player game. In our experiments with simulated and
real-world transaction data, we compared our framework with the baseline model
and demonstrated its potential through proposed evaluation metrics.

    

### [[2110.13306] Reconciling Risk Allocation and Prevalence Estimation in Public Health Using Batched Bandits](http://arxiv.org/abs/2110.13306)


  In many public health settings, there is a perceived tension between
allocating resources to known vulnerable areas and learning about the overall
prevalence of the problem. Inspired by a door-to-door Covid-19 testing program
we helped design, we combine multi-armed bandit strategies and insights from
sampling theory to demonstrate how to recover accurate prevalence estimates
while continuing to allocate resources to at-risk areas. We use the outbreak of
an infectious disease as our running example. The public health setting has
several characteristics distinguishing it from typical bandit settings, such as
distribution shift (the true disease prevalence is changing with time) and
batched sampling (multiple decisions must be made simultaneously).
Nevertheless, we demonstrate that several bandit algorithms are capable
out-performing greedy resource allocation strategies, which often perform worse
than random allocation as they fail to notice outbreaks in new areas.

    

### [[2110.13311] Physics Informed Machine Learning of SPH: Machine Learning Lagrangian Turbulence](http://arxiv.org/abs/2110.13311)


  Smoothed particle hydrodynamics (SPH) is a mesh-free Lagrangian method for
obtaining approximate numerical solutions of the equations of fluid dynamics;
which has been widely applied to weakly- and strongly compressible turbulence
in astrophysics and engineering applications. We present a learn-able hierarchy
of parameterized and "physics-explainable" SPH informed fluid simulators using
both physics based parameters and Neural Networks (NNs) as universal function
approximators. Our learning algorithm develops a mixed mode approach, mixing
forward and reverse mode automatic differentiation with forward and adjoint
based sensitivity analyses to efficiently perform gradient based optimization.
We show that our physics informed learning method is capable of: (a) solving
inverse problems over the physically interpretable parameter space, as well as
over the space of NN parameters; (b) learning Lagrangian statistics of
turbulence (interpolation); (c) combining Lagrangian trajectory based,
probabilistic, and Eulerian field based loss functions; and (d) extrapolating
beyond training sets into more complex regimes of interest. Furthermore, this
hierarchy of models gradually introduces more physical structure, which we show
improves interpretability, generalizability (over larger ranges of time scales
and Reynolds numbers), preservation of physical symmetries, and requires less
training data.

    

### [[2110.13315] EarthGAN: Can we visualize the Earth's mantle convection using a surrogate model?](http://arxiv.org/abs/2110.13315)


  Scientific simulations are often used to gain insight into foundational
questions. However, many potentially useful simulation results are difficult to
visualize without powerful computers. In this research, we seek to build a
surrogate model, using a generative adversarial network, to allow for the
visualization of the Earth's Mantle Convection data set on readily accessible
hardware. We present our preliminary method and results, and all code is made
publicly available. The preliminary results show that a surrogate model of the
Earth's Mantle Convection data set can generate useful results. A comparison to
the "ground-truth" is provided.

    

### [[2110.13323] Deep Learning Tools for Audacity: Helping Researchers Expand the Artist's Toolkit](http://arxiv.org/abs/2110.13323)


  We present a software framework that integrates neural networks into the
popular open-source audio editing software, Audacity, with a minimal amount of
developer effort. In this paper, we showcase some example use cases for both
end-users and neural network developers. We hope that this work fosters a new
level of interactivity between deep learning practitioners and end-users.

    

### [[2110.13330] Robust Learning of Physics Informed Neural Networks](http://arxiv.org/abs/2110.13330)


  Physics-informed Neural Networks (PINNs) have been shown to be effective in
solving partial differential equations by capturing the physics induced
constraints as a part of the training loss function. This paper shows that a
PINN can be sensitive to errors in training data and overfit itself in
dynamically propagating these errors over the domain of the solution of the
PDE. It also shows how physical regularizations based on continuity criteria
and conservation laws fail to address this issue and rather introduce problems
of their own causing the deep network to converge to a physics-obeying local
minimum instead of the global minimum. We introduce Gaussian Process (GP) based
smoothing that recovers the performance of a PINN and promises a robust
architecture against noise/errors in measurements. Additionally, we illustrate
an inexpensive method of quantifying the evolution of uncertainty based on the
variance estimation of GPs on boundary data. Robust PINN performance is also
shown to be achievable by choice of sparse sets of inducing points based on
sparsely induced GPs. We demonstrate the performance of our proposed methods
and compare the results from existing benchmark models in literature for
time-dependent Schrödinger and Burgers' equations.

    

### [[2110.13340] Privacy-Preserving Multi-Target Multi-Domain Recommender Systems with Assisted AutoEncoders](http://arxiv.org/abs/2110.13340)


  A long-standing challenge in Recommender Systems (RCs) is the data sparsity
problem that often arises when users rate very few items. Multi-Target
Multi-Domain Recommender Systems (MTMDR) aim to improve the recommendation
performance in multiple domains simultaneously. The existing works assume that
the data of different domains can be fully shared, and the computation can be
performed in a centralized manner. However, in many realistic scenarios,
separate recommender systems are operated by different organizations, which do
not allow the sharing of private data, models, and recommendation tasks. This
work proposes an MTMDR based on Assisted AutoEncoders (AAE) and Multi-Target
Assisted Learning (MTAL) to help organizational learners improve their
recommendation performance simultaneously without sharing sensitive assets.
Moreover, AAE has a broad application scope since it allows explicit or
implicit feedback, user- or item-based alignment, and with or without side
information. Extensive experiments demonstrate that our method significantly
outperforms the case where each domain is locally trained, and it performs
competitively with the centralized training where all data are shared. As a
result, AAE can effectively integrate organizations from different domains to
form a community of shared interest.

    

### [[2110.13344] Sinusoidal Flow: A Fast Invertible Autoregressive Flow](http://arxiv.org/abs/2110.13344)


  Normalising flows offer a flexible way of modelling continuous probability
distributions. We consider expressiveness, fast inversion and exact Jacobian
determinant as three desirable properties a normalising flow should possess.
However, few flow models have been able to strike a good balance among all
these properties. Realising that the integral of a convex sum of sinusoidal
functions squared leads to a bijective residual transformation, we propose
Sinusoidal Flow, a new type of normalising flows that inherits the expressive
power and triangular Jacobian from fully autoregressive flows while guaranteed
by Banach fixed-point theorem to remain fast invertible and thereby obviate the
need for sequential inversion typically required in fully autoregressive flows.
Experiments show that our Sinusoidal Flow is not only able to model complex
distributions, but can also be reliably inverted to generate realistic-looking
samples even with many layers of transformations stacked.

    

### [[2110.13361] Physics-Informed Neural Networks (PINNs) for Parameterized PDEs: A Metalearning Approach](http://arxiv.org/abs/2110.13361)


  Physics-informed neural networks (PINNs) as a means of discretizing partial
differential equations (PDEs) are garnering much attention in the Computational
Science and Engineering (CS&E) world. At least two challenges exist for PINNs
at present: an understanding of accuracy and convergence characteristics with
respect to tunable parameters and identification of optimization strategies
that make PINNs as efficient as other computational science tools. The cost of
PINNs training remains a major challenge of Physics-informed Machine Learning
(PiML) -- and, in fact, machine learning (ML) in general. This paper is meant
to move towards addressing the latter through the study of PINNs for
parameterized PDEs. Following the ML world, we introduce metalearning of PINNs
for parameterized PDEs. By introducing metalearning and transfer learning
concepts, we can greatly accelerate the PINNs optimization process. We present
a survey of model-agnostic metalearning, and then discuss our model-aware
metalearning applied to PINNs. We provide theoretically motivated and
empirically backed assumptions that make our metalearning approach possible. We
then test our approach on various canonical forward parameterized PDEs that
have been presented in the emerging PINNs literature.

    

### [[2110.13363] Exponential Graph is Provably Efficient for Decentralized Deep Training](http://arxiv.org/abs/2110.13363)


  Decentralized SGD is an emerging training method for deep learning known for
its much less (thus faster) communication per iteration, which relaxes the
averaging step in parallel SGD to inexact averaging. The less exact the
averaging is, however, the more the total iterations the training needs to
take. Therefore, the key to making decentralized SGD efficient is to realize
nearly-exact averaging using little communication. This requires a skillful
choice of communication topology, which is an under-studied topic in
decentralized optimization.
In this paper, we study so-called exponential graphs where every node is
connected to $O(\log(n))$ neighbors and $n$ is the total number of nodes. This
work proves such graphs can lead to both fast communication and effective
averaging simultaneously. We also discover that a sequence of $\log(n)$
one-peer exponential graphs, in which each node communicates to one single
neighbor per iteration, can together achieve exact averaging. This favorable
property enables one-peer exponential graph to average as effective as its
static counterpart but communicates more efficiently. We apply these
exponential graphs in decentralized (momentum) SGD to obtain the
state-of-the-art balance between per-iteration communication and iteration
complexity among all commonly-used topologies. Experimental results on a
variety of tasks and models demonstrate that decentralized (momentum) SGD over
exponential graphs promises both fast and high-quality training. Our code is
implemented through BlueFog and available at
this https URL.

    

### [[2110.13365] Multi-Faceted Hierarchical Multi-Task Learning for a Large Number of Tasks with Multi-dimensional Relations](http://arxiv.org/abs/2110.13365)


  There has been many studies on improving the efficiency of shared learning in
Multi-Task Learning(MTL). Previous work focused on the "micro" sharing
perspective for a small number of tasks, while in Recommender Systems(RS) and
other AI applications, there are often demands to model a large number of tasks
with multi-dimensional task relations. For example, when using MTL to model
various user behaviors in RS, if we differentiate new users and new items from
old ones, there will be a cartesian product style increase of tasks with
multi-dimensional relations. This work studies the "macro" perspective of
shared learning network design and proposes a Multi-Faceted Hierarchical MTL
model(MFH). MFH exploits the multi-dimension task relations with a nested
hierarchical tree structure which maximizes the shared learning. We evaluate
MFH and SOTA models in a large industry video platform of 10 billion samples
and results show that MFH outperforms SOTA MTL models significantly in both
offline and online evaluations across all user groups, especially remarkable
for new users with an online increase of 9.1\% in app time per user and 1.85\%
in next-day retention rate. MFH now has been deployed in a large scale online
video recommender system. MFH is especially beneficial to the cold-start
problems in RS where new users and new items often suffer from a "local
overfitting" phenomenon. However, the idea is actually generic and widely
applicable to other MTL scenarios.

    

### [[2110.13369] Partial order: Finding Consensus among Uncertain Feature Attributions](http://arxiv.org/abs/2110.13369)


  Post-hoc feature importance is progressively being employed to explain
decisions of complex machine learning models. Yet in practice, reruns of the
training algorithm and/or the explainer can result in contradicting statements
of feature importance, henceforth reducing trust in those techniques. A
possible avenue to address this issue is to develop strategies to aggregate
diverse explanations about feature importance. While the arithmetic mean, which
yields a total order, has been advanced, we introduce an alternative: the
consensus among multiple models, which results in partial orders. The two
aggregation strategies are compared using Integrated Gradients and Shapley
values on two regression datasets, and we show that a large portion of the
information provided by the mean aggregation is not supported by the consensus
of each individual model, raising suspicion on the trustworthiness of this
practice.

    

### [[2110.13373] EnTRPO: Trust Region Policy Optimization Method with Entropy Regularization](http://arxiv.org/abs/2110.13373)


  Trust Region Policy Optimization (TRPO) is a popular and empirically
successful policy search algorithm in reinforcement learning (RL). It
iteratively solved the surrogate problem which restricts consecutive policies
to be close to each other. TRPO is an on-policy algorithm. On-policy methods
bring many benefits, like the ability to gauge each resulting policy. However,
they typically discard all the knowledge about the policies which existed
before. In this work, we use a replay buffer to borrow from the off-policy
learning setting to TRPO. Entropy regularization is usually used to improve
policy optimization in reinforcement learning. It is thought to aid exploration
and generalization by encouraging more random policy choices. We add an Entropy
regularization term to advantage over {\pi}, accumulated over time steps, in
TRPO. We call this update EnTRPO. Our experiments demonstrate EnTRPO achieves
better performance for controlling a Cart-Pole system compared with the
original TRPO

    

### [[2110.13388] Semi-Supervised Federated Learning with non-IID Data: Algorithm and System Design](http://arxiv.org/abs/2110.13388)


  Federated Learning (FL) allows edge devices (or clients) to keep data locally
while simultaneously training a shared high-quality global model. However,
current research is generally based on an assumption that the training data of
local clients have ground-truth. Furthermore, FL faces the challenge of
statistical heterogeneity, i.e., the distribution of the client's local
training data is non-independent identically distributed (non-IID). In this
paper, we present a robust semi-supervised FL system design, where the system
aims to solve the problem of data availability and non-IID in FL. In
particular, this paper focuses on studying the labels-at-server scenario where
there is only a limited amount of labeled data on the server and only unlabeled
data on the clients. In our system design, we propose a novel method to tackle
the problems, which we refer to as Federated Mixing (FedMix). FedMix improves
the naive combination of FL and semi-supervised learning methods and designs
parameter decomposition strategies for disjointed learning of labeled,
unlabeled data, and global models. To alleviate the non-IID problem, we propose
a novel aggregation rule based on the frequency of the client's participation
in training, namely the FedFreq aggregation algorithm, which can adjust the
weight of the corresponding local model according to this frequency. Extensive
evaluations conducted on CIFAR-10 dataset show that the performance of our
proposed method is significantly better than those of the current baseline. It
is worth noting that our system is robust to different non-IID levels of client
data.

    

### [[2110.13400] Scale-Free Adversarial Multi-Armed Bandit with Arbitrary Feedback Delays](http://arxiv.org/abs/2110.13400)


  We consider the Scale-Free Adversarial Multi Armed Bandit (MAB) problem with
unrestricted feedback delays. In contrast to the standard assumption that all
losses are $[0,1]$-bounded, in our setting, losses can fall in a general
bounded interval $[-L, L]$, unknown to the agent before-hand. Furthermore, the
feedback of each arm pull can experience arbitrary delays. We propose an
algorithm named \texttt{SFBanker} for this novel setting, which combines a
recent banker online mirror descent technique and elaborately designed doubling
tricks. We show that \texttt{SFBanker} achieves $\mathcal
O(\sqrt{K(D+T)}L)\cdot {\rm polylog}(T, L)$ total regret, where $T$ is the
total number of steps and $D$ is the total feedback delay. \texttt{SFBanker}
also outperforms existing algorithm for non-delayed (i.e., $D=0$) scale-free
adversarial MAB problem instances. We also present a variant of
\texttt{SFBanker} for problem instances with non-negative losses (i.e., they
range in $[0, L]$ for some unknown $L$), achieving an $\tilde{\mathcal
O}(\sqrt{K(D+T)}L)$ total regret, which is near-optimal compared to the
$\Omega(\sqrt{KT}+\sqrt{D\log K}L)$ lower-bound ([Cesa-Bianchi et al., 2016]).

    

### [[2110.13402] Revisiting randomized choices in isolation forests](http://arxiv.org/abs/2110.13402)


  Isolation forest or "iForest" is an intuitive and widely used algorithm for
anomaly detection that follows a simple yet effective idea: in a given data
distribution, if a threshold (split point) is selected uniformly at random
within the range of some variable and data points are divided according to
whether they are greater or smaller than this threshold, outlier points are
more likely to end up alone or in the smaller partition. The original procedure
suggested the choice of variable to split and split point within a variable to
be done uniformly at random at each step, but this paper shows that "clustered"
diverse outliers - oftentimes a more interesting class of outliers than others
- can be more easily identified by applying a non-uniformly-random choice of
variables and/or thresholds. Different split guiding criteria are compared and
some are found to result in significantly better outlier discrimination for
certain classes of outliers.

    

### [[2110.13413] Convergent Boosted Smoothing for Modeling Graph Data with Tabular Node Features](http://arxiv.org/abs/2110.13413)


  For supervised learning with tabular data, decision tree ensembles produced
via boosting techniques generally dominate real-world applications involving
iid training/test sets. However for graph data where the iid assumption is
violated due to structured relations between samples, it remains unclear how to
best incorporate this structure within existing boosting pipelines. To this
end, we propose a generalized framework for iterating boosting with graph
propagation steps that share node/sample information across edges connecting
related samples. Unlike previous efforts to integrate graph-based models with
boosting, our approach is anchored in a principled meta loss function such that
provable convergence can be guaranteed under relatively mild assumptions.
Across a variety of non-iid graph datasets with tabular node features, our
method achieves comparable or superior performance than both tabular and graph
neural network models, as well as existing hybrid strategies that combine the
two. Beyond producing better predictive performance than recently proposed
graph models, our proposed techniques are easy to implement, computationally
more efficient, and enjoy stronger theoretical guarantees (which make our
results more reproducible).

    

### [[2110.13422] Relay Variational Inference: A Method for Accelerated Encoderless VI](http://arxiv.org/abs/2110.13422)


  Variational Inference (VI) offers a method for approximating intractable
likelihoods. In neural VI, inference of approximate posteriors is commonly done
using an encoder. Alternatively, encoderless VI offers a framework for learning
generative models from data without encountering suboptimalities caused by
amortization via an encoder (e.g. in presence of missing or uncertain data).
However, in absence of an encoder, such methods often suffer in convergence due
to the slow nature of gradient steps required to learn the approximate
posterior parameters. In this paper, we introduce Relay VI (RVI), a framework
that dramatically improves both the convergence and performance of encoderless
VI. In our experiments over multiple datasets, we study the effectiveness of
RVI in terms of convergence speed, loss, representation power and missing data
imputation. We find RVI to be a unique tool, often superior in both performance
and convergence speed to previously proposed encoderless as well as amortized
VI models (e.g. VAE).

    

### [[2110.13423] Towards More Generalizable One-shot Visual Imitation Learning](http://arxiv.org/abs/2110.13423)


  A general-purpose robot should be able to master a wide range of tasks and
quickly learn a novel one by leveraging past experiences. One-shot imitation
learning (OSIL) approaches this goal by training an agent with (pairs of)
expert demonstrations, such that at test time, it can directly execute a new
task from just one demonstration. However, so far this framework has been
limited to training on many variations of one task, and testing on other unseen
but similar variations of the same task. In this work, we push for a higher
level of generalization ability by investigating a more ambitious multi-task
setup. We introduce a diverse suite of vision-based robot manipulation tasks,
consisting of 7 tasks, a total of 61 variations, and a continuum of instances
within each variation. For consistency and comparison purposes, we first train
and evaluate single-task agents (as done in prior few-shot imitation work). We
then study the multi-task setting, where multi-task training is followed by (i)
one-shot imitation on variations within the training tasks, (ii) one-shot
imitation on new tasks, and (iii) fine-tuning on new tasks. Prior
state-of-the-art, while performing well within some single tasks, struggles in
these harder multi-task settings. To address these limitations, we propose
MOSAIC (Multi-task One-Shot Imitation with self-Attention and Contrastive
learning), which integrates a self-attention model architecture and a temporal
contrastive module to enable better task disambiguation and more robust
representation learning. Our experiments show that MOSAIC outperforms prior
state of the art in learning efficiency, final performance, and learns a
multi-task policy with promising generalization ability via fine-tuning on
novel tasks.

    

### [[2110.13435] Understanding the Role of Self-Supervised Learning in Out-of-Distribution Detection Task](http://arxiv.org/abs/2110.13435)


  Self-supervised learning (SSL) has achieved great success in a variety of
computer vision tasks. However, the mechanism of how SSL works in these tasks
remains a mystery. In this paper, we study how SSL can enhance the performance
of the out-of-distribution (OOD) detection task. We first point out two general
properties that a good OOD detector should have: 1) the overall feature space
should be large and 2) the inlier feature space should be small. Then we
demonstrate that SSL can indeed increase the intrinsic dimension of the overall
feature space. In the meantime, SSL even has the potential to shrink the inlier
feature space. As a result, there will be more space spared for the outliers,
making OOD detection much easier. The conditions when SSL can shrink the inlier
feature space is also discussed and validated. By understanding the role of SSL
in the OOD detection task, our study can provide a guideline for designing
better OOD detection algorithms. Moreover, this work can also shed light to
other tasks where SSL can improve the performance.

    

### [[2110.13440] A deep learning driven pseudospectral PCE based FFT homogenization algorithm for complex microstructures](http://arxiv.org/abs/2110.13440)


  This work is directed to uncertainty quantification of homogenized effective
properties for composite materials with complex, three dimensional
microstructure. The uncertainties arise in the material parameters of the
single constituents as well as in the fiber volume fraction. They are taken
into account by multivariate random variables. Uncertainty quantification is
achieved by an efficient surrogate model based on pseudospectral polynomial
chaos expansion and artificial neural networks. An artificial neural network is
trained on synthetic binary voxelized unit cells of composite materials with
uncertain three dimensional microstructures, uncertain linear elastic material
parameters and different loading directions. The prediction goals of the
artificial neural network are the corresponding effective components of the
elasticity tensor, where the labels for training are generated via a fast
Fourier transform based numerical homogenization method. The trained artificial
neural network is then used as a deterministic solver for a pseudospectral
polynomial chaos expansion based surrogate model to achieve the corresponding
statistics of the effective properties. Three numerical examples deal with the
comparison of the presented method to the literature as well as the application
to different microstructures. It is shown, that the proposed method is able to
predict central moments of interest while being magnitudes faster to evaluate
than traditional approaches.

    

### [[2110.13444] A time-weighted metric for sets of trajectories to assess multi-object tracking algorithms](http://arxiv.org/abs/2110.13444)


  This paper proposes a metric for sets of trajectories to evaluate
multi-object tracking algorithms that includes time-weighted costs for
localisation errors of properly detected targets, for false targets, missed
targets and track switches. The proposed metric extends the metric in [1] by
including weights to the costs associated to different time steps. The
time-weighted costs increase the flexibility of the metric [1] to fit more
applications and user preferences. We first introduce a metric based on
multi-dimensional assignments, and then its linear programming relaxation,
which is computable in polynomial time and is also a metric. The metrics can
also be extended to metrics on random finite sets of trajectories to evaluate
and rank algorithms across different scenarios, each with a ground truth set of
trajectories.

    

### [[2110.13450] Distributed Multi-Agent Deep Reinforcement Learning Framework for Whole-building HVAC Control](http://arxiv.org/abs/2110.13450)


  It is estimated that about 40%-50% of total electricity consumption in
commercial buildings can be attributed to Heating, Ventilation, and Air
Conditioning (HVAC) systems. Minimizing the energy cost while considering the
thermal comfort of the occupants is very challenging due to unknown and complex
relationships between various HVAC controls and thermal dynamics inside a
building. To this end, we present a multi-agent, distributed deep reinforcement
learning (DRL) framework based on Energy Plus simulation environment for
optimizing HVAC in commercial buildings. This framework learns the complex
thermal dynamics in the building and takes advantage of the differential effect
of cooling and heating systems in the building to reduce energy costs, while
maintaining the thermal comfort of the occupants. With adaptive penalty, the RL
algorithm can be prioritized for energy savings or maintaining thermal comfort.
Using DRL, we achieve more than 75\% savings in energy consumption. The
distributed DRL framework can be scaled to multiple GPUs and CPUs of
heterogeneous types.

    

### [[2110.13452] On the Optimization Landscape of Maximum Mean Discrepancy](http://arxiv.org/abs/2110.13452)


  Generative models have been successfully used for generating realistic
signals. Because the likelihood function is typically intractable in most of
these models, the common practice is to use "implicit" models that avoid
likelihood calculation. However, it is hard to obtain theoretical guarantees
for such models. In particular, it is not understood when they can globally
optimize their non-convex objectives. Here we provide such an analysis for the
case of Maximum Mean Discrepancy (MMD) learning of generative models. We prove
several optimality results, including for a Gaussian distribution with low rank
covariance (where likelihood is inapplicable) and a mixture of Gaussians. Our
analysis shows that that the MMD optimization landscape is benign in these
cases, and therefore gradient based methods will globally minimize the MMD
objective.

    

### [[2110.13464] MarS-FL: A Market Share-based Decision Support Framework for Participation in Federated Learning](http://arxiv.org/abs/2110.13464)


  Federated learning (FL) enables multiple participants (PTs) to build an
aggregate and more powerful learning model without sharing data, thus
maintaining data privacy and security. Among the key application scenarios is a
competitive market where market shares represent PTs' competitiveness. An
understanding of the role of FL in evolving market shares plays a key role in
advancing the adoption of FL by PTs.
In terms of modeling, we adapt a general economic model to the FL context and
introduce two notions of $\delta$-stable market and friendliness to measure the
viability of FL and the market acceptability to FL. Further, we address related
decision-making issues with FL designer and PTs. First, we characterize the
process by which each PT participates in FL as a non-cooperative game and prove
its dominant strategy. Second, as an FL designer, the final model performance
improvement of each PT should be bounded, which relates to the market
conditions of a particular FL application scenario; we give a sufficient and
necessary condition $Q$ to maintain the market $\delta$-stability and quantify
the friendliness $\kappa$. The condition $Q$ gives a specific requirement while
an FL designer allocates performance improvements among PTs. In a typical case
of oligopoly, closed-form expressions of $Q$ and $\kappa$ are given. Finally,
numerical results are given to show the viability of FL in a wide range of
market conditions. Our results help identify optimal PT strategies, the viable
operational space of an FL designer, and the market conditions under which FL
is especially beneficial.

    

### [[2110.13465] CS-Rep: Making Speaker Verification Networks Embracing Re-parameterization](http://arxiv.org/abs/2110.13465)


  Automatic speaker verification (ASV) systems, which determine whether two
speeches are from the same speaker, mainly focus on verification accuracy while
ignoring inference speed. However, in real applications, both inference speed
and verification accuracy are essential. This study proposes cross-sequential
re-parameterization (CS-Rep), a novel topology re-parameterization strategy for
multi-type networks, to increase the inference speed and verification accuracy
of models. CS-Rep solves the problem that existing re-parameterization methods
are unsuitable for typical ASV backbones. When a model applies CS-Rep, the
training-period network utilizes a multi-branch topology to capture speaker
information, whereas the inference-period model converts to a time-delay neural
network (TDNN)-like plain backbone with stacked TDNN layers to achieve the fast
inference speed. Based on CS-Rep, an improved TDNN with friendly test and
deployment called Rep-TDNN is proposed. Compared with the state-of-the-art
model ECAPA-TDNN, which is highly recognized in the industry, Rep-TDNN
increases the actual inference speed by about 50% and reduces the EER by 10%.
The code will be released.

    

### [[2110.13475] Vector-valued Distance and Gyrocalculus on the Space of Symmetric Positive Definite Matrices](http://arxiv.org/abs/2110.13475)


  We propose the use of the vector-valued distance to compute distances and
extract geometric information from the manifold of symmetric positive definite
matrices (SPD), and develop gyrovector calculus, constructing analogs of vector
space operations in this curved space. We implement these operations and
showcase their versatility in the tasks of knowledge graph completion, item
recommendation, and question answering. In experiments, the SPD models
outperform their equivalents in Euclidean and hyperbolic space. The
vector-valued distance allows us to visualize embeddings, showing that the
models learn to disentangle representations of positive samples from negative
ones.

    

### [[2110.13484] Applications of Multi-Agent Reinforcement Learning in Future Internet: A Comprehensive Survey](http://arxiv.org/abs/2110.13484)


  Future Internet involves several emerging technologies such as 5G and beyond
5G networks, vehicular networks, unmanned aerial vehicle (UAV) networks, and
Internet of Things (IoTs). Moreover, future Internet becomes heterogeneous and
decentralized with a large number of involved network entities. Each entity may
need to make its local decision to improve the network performance under
dynamic and uncertain network environments. Standard learning algorithms such
as single-agent Reinforcement Learning (RL) or Deep Reinforcement Learning
(DRL) have been recently used to enable each network entity as an agent to
learn an optimal decision-making policy adaptively through interacting with the
unknown environments. However, such an algorithm fails to model the
cooperations or competitions among network entities, and simply treats other
entities as a part of the environment that may result in the non-stationarity
issue. Multi-agent Reinforcement Learning (MARL) allows each network entity to
learn its optimal policy by observing not only the environments, but also other
entities' policies. As a result, MARL can significantly improve the learning
efficiency of the network entities, and it has been recently used to solve
various issues in the emerging networks. In this paper, we thus review the
applications of MARL in the emerging networks. In particular, we provide a
tutorial of MARL and a comprehensive survey of applications of MARL in next
generation Internet. In particular, we first introduce single-agent RL and
MARL. Then, we review a number of applications of MARL to solve emerging issues
in future Internet. The issues consist of network access, transmit power
control, computation offloading, content caching, packet routing, trajectory
design for UAV-aided networks, and network security issues.

    

### [[2110.13492] TUNet: A Block-online Bandwidth Extension Model based on Transformers and Self-supervised Pretraining](http://arxiv.org/abs/2110.13492)


  We introduce a block-online variant of the temporal feature-wise linear
modulation (TFiLM) model to achieve bandwidth extension. The proposed
architecture simplifies the UNet backbone of the TFiLM to reduce inference time
and employs an efficient transformer at the bottleneck to alleviate performance
degradation. We also utilize self-supervised pretraining and data augmentation
to enhance the quality of bandwidth extended signals and reduce the sensitivity
with respect to downsampling methods. Experiment results on the VCTK dataset
show that the proposed method outperforms several recent baselines in terms of
spectral distance and source-to-distortion ratio. Pretraining and filter
augmentation also help stabilize and enhance the overall performance.

    

### [[2110.13501] Tensor Network Kalman Filtering for Large-Scale LS-SVMs](http://arxiv.org/abs/2110.13501)


  Least squares support vector machines are a commonly used supervised learning
method for nonlinear regression and classification. They can be implemented in
either their primal or dual form. The latter requires solving a linear system,
which can be advantageous as an explicit mapping of the data to a possibly
infinite-dimensional feature space is avoided. However, for large-scale
applications, current low-rank approximation methods can perform inadequately.
For example, current methods are probabilistic due to their sampling
procedures, and/or suffer from a poor trade-off between the ranks and
approximation power. In this paper, a recursive Bayesian filtering framework
based on tensor networks and the Kalman filter is presented to alleviate the
demanding memory and computational complexities associated with solving
large-scale dual problems. The proposed method is iterative, does not require
explicit storage of the kernel matrix, and allows the formulation of early
stopping conditions. Additionally, the framework yields confidence estimates of
obtained models, unlike alternative methods. The performance is tested on two
regression and three classification experiments, and compared to the Nyström
and fixed size LS-SVM methods. Results show that our method can achieve high
performance and is particularly useful when alternative methods are
computationally infeasible due to a slowly decaying kernel matrix spectrum.

    

### [[2110.13502] Shared Independent Component Analysis for Multi-Subject Neuroimaging](http://arxiv.org/abs/2110.13502)


  We consider shared response modeling, a multi-view learning problem where one
wants to identify common components from multiple datasets or views. We
introduce Shared Independent Component Analysis (ShICA) that models each view
as a linear transform of shared independent components contaminated by additive
Gaussian noise. We show that this model is identifiable if the components are
either non-Gaussian or have enough diversity in noise variances. We then show
that in some cases multi-set canonical correlation analysis can recover the
correct unmixing matrices, but that even a small amount of sampling noise makes
Multiset CCA fail. To solve this problem, we propose to use joint
diagonalization after Multiset CCA, leading to a new approach called ShICA-J.
We show via simulations that ShICA-J leads to improved results while being very
fast to fit. While ShICA-J is based on second-order statistics, we further
propose to leverage non-Gaussianity of the components using a
maximum-likelihood method, ShICA-ML, that is both more accurate and more
costly. Further, ShICA comes with a principled method for shared components
estimation. Finally, we provide empirical evidence on fMRI and MEG datasets
that ShICA yields more accurate estimation of the components than alternatives.

    

### [[2110.13506] A DPDK-Based Acceleration Method for Experience Sampling of Distributed Reinforcement Learning](http://arxiv.org/abs/2110.13506)


  A computing cluster that interconnects multiple compute nodes is used to
accelerate distributed reinforcement learning based on DQN (Deep Q-Network). In
distributed reinforcement learning, Actor nodes acquire experiences by
interacting with a given environment and a Learner node optimizes their DQN
model. Since data transfer between Actor and Learner nodes increases depending
on the number of Actor nodes and their experience size, communication overhead
between them is one of major performance bottlenecks. In this paper, their
communication is accelerated by DPDK-based network optimizations, and
DPDK-based low-latency experience replay memory server is deployed between
Actor and Learner nodes interconnected with a 40GbE (40Gbit Ethernet) network.
Evaluation results show that, as a network optimization technique, kernel
bypassing by DPDK reduces network access latencies to a shared memory server by
32.7% to 58.9%. As another network optimization technique, an in-network
experience replay memory server between Actor and Learner nodes reduces access
latencies to the experience replay memory by 11.7% to 28.1% and communication
latencies for prioritized experience sampling by 21.9% to 29.1%.

    

### [[2110.13511] AutoDEUQ: Automated Deep Ensemble with Uncertainty Quantification](http://arxiv.org/abs/2110.13511)


  Deep neural networks are powerful predictors for a variety of tasks. However,
they do not capture uncertainty directly. Using neural network ensembles to
quantify uncertainty is competitive with approaches based on Bayesian neural
networks while benefiting from better computational scalability. However,
building ensembles of neural networks is a challenging task because, in
addition to choosing the right neural architecture or hyperparameters for each
member of the ensemble, there is an added cost of training each model. We
propose AutoDEUQ, an automated approach for generating an ensemble of deep
neural networks. Our approach leverages joint neural architecture and
hyperparameter search to generate ensembles. We use the law of total variance
to decompose the predictive variance of deep ensembles into aleatoric (data)
and epistemic (model) uncertainties. We show that AutoDEUQ outperforms
probabilistic backpropagation, Monte Carlo dropout, deep ensemble,
distribution-free ensembles, and hyper ensemble methods on a number of
regression benchmarks.

    

### [[2110.13515] Modular Gaussian Processes for Transfer Learning](http://arxiv.org/abs/2110.13515)


  We present a framework for transfer learning based on modular variational
Gaussian processes (GP). We develop a module-based method that having a
dictionary of well fitted GPs, one could build ensemble GP models without
revisiting any data. Each model is characterised by its hyperparameters,
pseudo-inputs and their corresponding posterior densities. Our method avoids
undesired data centralisation, reduces rising computational costs and allows
the transfer of learned uncertainty metrics after training. We exploit the
augmentation of high-dimensional integral operators based on the
Kullback-Leibler divergence between stochastic processes to introduce an
efficient lower bound under all the sparse variational GPs, with different
complexity and even likelihood distribution. The method is also valid for
multi-output GPs, learning correlations a posteriori between independent
modules. Extensive results illustrate the usability of our framework in
large-scale and multi-task experiments, also compared with the exact inference
methods in the literature.

    

### [[2110.13521] Machine learning spectral functions in lattice QCD](http://arxiv.org/abs/2110.13521)


  We study the inverse problem of reconstructing spectral functions from
Euclidean correlation functions via machine learning. We propose a novel
neutral network, sVAE, which is based on the variational autoencoder (VAE) and
can be naturally applied to the inverse problem. The prominent feature of the
sVAE is that a Shannon-Jaynes entropy term having the ground truth values of
spectral functions as prior information is included in the loss function to be
minimized. We train the network with general spectral functions produced from a
Gaussian mixture model. As a test, we use correlators generated from four
different types of physically motivated spectral functions made of one
resonance peak, a continuum term and perturbative spectral function obtained
using non-relativistic QCD. From the mock data test we find that the sVAE in
most cases is comparable to the maximum entropy method (MEM) in the quality of
reconstructing spectral functions and even outperforms the MEM in the case
where the spectral function has sharp peaks with insufficient number of data
points in the correlator. By applying to temporal correlation functions of
charmonium in the pseudoscalar channel obtained in the quenched lattice QCD at
0.75 $T_c$ on $128^3\times96$ lattices and $1.5$ $T_c$ on $128^3\times48$
lattices, we find that the resonance peak of $\eta_c$ extracted from both the
sVAE and MEM has a substantial dependence on the number of points in the
temporal direction ($N_\tau$) adopted in the lattice simulation and $N_\tau$
larger than 48 is needed to resolve the fate of $\eta_c$ at 1.5 $T_c$.

    

### [[2110.13522] Probabilistic Entity Representation Model for Chain Reasoning over Knowledge Graphs](http://arxiv.org/abs/2110.13522)


  Logical reasoning over Knowledge Graphs (KGs) is a fundamental technique that
can provide efficient querying mechanism over large and incomplete databases.
Current approaches employ spatial geometries such as boxes to learn query
representations that encompass the answer entities and model the logical
operations of projection and intersection. However, their geometry is
restrictive and leads to non-smooth strict boundaries, which further results in
ambiguous answer entities. Furthermore, previous works propose transformation
tricks to handle unions which results in non-closure and, thus, cannot be
chained in a stream. In this paper, we propose a Probabilistic Entity
Representation Model (PERM) to encode entities as a Multivariate Gaussian
density with mean and covariance parameters to capture its semantic position
and smooth decision boundary, respectively. Additionally, we also define the
closed logical operations of projection, intersection, and union that can be
aggregated using an end-to-end objective function. On the logical query
reasoning problem, we demonstrate that the proposed PERM significantly
outperforms the state-of-the-art methods on various public benchmark KG
datasets on standard evaluation metrics. We also evaluate PERM's competence on
a COVID-19 drug-repurposing case study and show that our proposed work is able
to recommend drugs with substantially better F1 than current methods. Finally,
we demonstrate the working of our PERM's query answering process through a
low-dimensional visualization of the Gaussian representations.

    

### [[2110.13523] Automating Control of Overestimation Bias for Continuous Reinforcement Learning](http://arxiv.org/abs/2110.13523)


  Bias correction techniques are used by most of the high-performing methods
for off-policy reinforcement learning. However, these techniques rely on a
pre-defined bias correction policy that is either not flexible enough or
requires environment-specific tuning of hyperparameters. In this work, we
present a simple data-driven approach for guiding bias correction. We
demonstrate its effectiveness on the Truncated Quantile Critics -- a
state-of-the-art continuous control algorithm. The proposed technique can
adjust the bias correction across environments automatically. As a result, it
eliminates the need for an extensive hyperparameter search, significantly
reducing the actual number of interactions and computation.

    

### [[2110.13530] An extended physics informed neural network for preliminary analysis of parametric optimal control problems](http://arxiv.org/abs/2110.13530)


  In this work we propose an extension of physics informed supervised learning
strategies to parametric partial differential equations. Indeed, even if the
latter are indisputably useful in many applications, they can be
computationally expensive most of all in a real-time and many-query setting.
Thus, our main goal is to provide a physics informed learning paradigm to
simulate parametrized phenomena in a small amount of time. The physics
information will be exploited in many ways, in the loss function (standard
physics informed neural networks), as an augmented input (extra feature
employment) and as a guideline to build an effective structure for the neural
network (physics informed architecture). These three aspects, combined
together, will lead to a faster training phase and to a more accurate
parametric prediction. The methodology has been tested for several equations
and also in an optimal control framework.

    

### [[2110.13541] Qu-ANTI-zation: Exploiting Quantization Artifacts for Achieving Adversarial Outcomes](http://arxiv.org/abs/2110.13541)


  Quantization is a popular technique that $transforms$ the parameter
representation of a neural network from floating-point numbers into
lower-precision ones ($e.g.$, 8-bit integers). It reduces the memory footprint
and the computational cost at inference, facilitating the deployment of
resource-hungry models. However, the parameter perturbations caused by this
transformation result in $behavioral$ $disparities$ between the model before
and after quantization. For example, a quantized model can misclassify some
test-time samples that are otherwise classified correctly. It is not known
whether such differences lead to a new security vulnerability. We hypothesize
that an adversary may control this disparity to introduce specific behaviors
that activate upon quantization. To study this hypothesis, we weaponize
quantization-aware training and propose a new training framework to implement
adversarial quantization outcomes. Following this framework, we present three
attacks we carry out with quantization: (i) an indiscriminate attack for
significant accuracy loss; (ii) a targeted attack against specific samples; and
(iii) a backdoor attack for controlling the model with an input trigger. We
further show that a single compromised model defeats multiple quantization
schemes, including robust quantization techniques. Moreover, in a federated
learning scenario, we demonstrate that a set of malicious participants who
conspire can inject our quantization-activated backdoor. Lastly, we discuss
potential counter-measures and show that only re-training consistently removes
the attack artifacts. Our code is available at
this https URL


### [[2110.13549] Online Variational Filtering and Parameter Learning](http://arxiv.org/abs/2110.13549)


  We present a variational method for online state estimation and parameter
learning in state-space models (SSMs), a ubiquitous class of latent variable
models for sequential data. As per standard batch variational techniques, we
use stochastic gradients to simultaneously optimize a lower bound on the log
evidence with respect to both model parameters and a variational approximation
of the states' posterior distribution. However, unlike existing approaches, our
method is able to operate in an entirely online manner, such that historic
observations do not require revisitation after being incorporated and the cost
of updates at each time step remains constant, despite the growing
dimensionality of the joint posterior distribution of the states. This is
achieved by utilizing backward decompositions of this joint posterior
distribution and of its variational approximation, combined with Bellman-type
recursions for the evidence lower bound and its gradients. We demonstrate the
performance of this methodology across several examples, including
high-dimensional SSMs and sequential Variational Auto-Encoders.

    

### [[2110.13550] Coherent False Seizure Prediction in Epilepsy, Coincidence or Providence?](http://arxiv.org/abs/2110.13550)


  Seizure forecasting using machine learning is possible, but the performance
is far from ideal, as indicated by many false predictions and low specificity.
Here, we examine false and missing alarms of two algorithms on long-term
datasets to show that the limitations are less related to classifiers or
features, but rather to intrinsic changes in the data. We evaluated two
algorithms on three datasets by computing the correlation of false predictions
and estimating the information transfer between both classification methods.
For 9 out of 12 individuals both methods showed a performance better than
chance. For all individuals we observed a positive correlation in predictions.
For individuals with strong correlation in false predictions we were able to
boost the performance of one method by excluding test samples based on the
results of the second method. Substantially different algorithms exhibit a
highly consistent performance and a strong coherency in false and missing
alarms. Hence, changing the underlying hypothesis of a preictal state of fixed
time length prior to each seizure to a proictal state is more helpful than
further optimizing classifiers. The outcome is significant for the evaluation
of seizure prediction algorithms on continuous data.

    

### [[2110.13561] Non-Gaussian Gaussian Processes for Few-Shot Regression](http://arxiv.org/abs/2110.13561)


  Gaussian Processes (GPs) have been widely used in machine learning to model
distributions over functions, with applications including multi-modal
regression, time-series prediction, and few-shot learning. GPs are particularly
useful in the last application since they rely on Normal distributions and
enable closed-form computation of the posterior probability function.
Unfortunately, because the resulting posterior is not flexible enough to
capture complex distributions, GPs assume high similarity between subsequent
tasks - a requirement rarely met in real-world conditions. In this work, we
address this limitation by leveraging the flexibility of Normalizing Flows to
modulate the posterior predictive distribution of the GP. This makes the GP
posterior locally non-Gaussian, therefore we name our method Non-Gaussian
Gaussian Processes (NGGPs). More precisely, we propose an invertible ODE-based
mapping that operates on each component of the random variable vectors and
shares the parameters across all of them. We empirically tested the flexibility
of NGGPs on various few-shot learning regression datasets, showing that the
mapping can incorporate context embedding information to model different noise
levels for periodic functions. As a result, our method shares the structure of
the problem between subsequent tasks, but the contextualization allows for
adaptation to dissimilarities. NGGPs outperform the competing state-of-the-art
approaches on a diversified set of benchmarks and applications.

    

### [[2110.13567] Pairwise Half-graph Discrimination: A Simple Graph-level Self-supervised Strategy for Pre-training Graph Neural Networks](http://arxiv.org/abs/2110.13567)


  Self-supervised learning has gradually emerged as a powerful technique for
graph representation learning. However, transferable, generalizable, and robust
representation learning on graph data still remains a challenge for
pre-training graph neural networks. In this paper, we propose a simple and
effective self-supervised pre-training strategy, named Pairwise Half-graph
Discrimination (PHD), that explicitly pre-trains a graph neural network at
graph-level. PHD is designed as a simple binary classification task to
discriminate whether two half-graphs come from the same source. Experiments
demonstrate that the PHD is an effective pre-training strategy that offers
comparable or superior performance on 13 graph classification tasks compared
with state-of-the-art strategies, and achieves notable improvements when
combined with node-level strategies. Moreover, the visualization of learned
representation revealed that PHD strategy indeed empowers the model to learn
graph-level knowledge like the molecular scaffold. These results have
established PHD as a powerful and effective self-supervised learning strategy
in graph-level representation learning.

    

### [[2110.13572] Periodic Activation Functions Induce Stationarity](http://arxiv.org/abs/2110.13572)


  Neural network models are known to reinforce hidden data biases, making them
unreliable and difficult to interpret. We seek to build models that `know what
they do not know' by introducing inductive biases in the function space. We
show that periodic activation functions in Bayesian neural networks establish a
connection between the prior on the network weights and translation-invariant,
stationary Gaussian process priors. Furthermore, we show that this link goes
beyond sinusoidal (Fourier) activations by also covering triangular wave and
periodic ReLU activation functions. In a series of experiments, we show that
periodic activation functions obtain comparable performance for in-domain data
and capture sensitivity to perturbed inputs in deep neural networks for
out-of-domain detection.

    

### [[2110.13576] Learning Robust Controllers Via Probabilistic Model-Based Policy Search](http://arxiv.org/abs/2110.13576)


  Model-based Reinforcement Learning estimates the true environment through a
world model in order to approximate the optimal policy. This family of
algorithms usually benefits from better sample efficiency than their model-free
counterparts. We investigate whether controllers learned in such a way are
robust and able to generalize under small perturbations of the environment. Our
work is inspired by the PILCO algorithm, a method for probabilistic policy
search. We show that enforcing a lower bound to the likelihood noise in the
Gaussian Process dynamics model regularizes the policy updates and yields more
robust controllers. We demonstrate the empirical benefits of our method in a
simulation benchmark.

    

### [[2110.13578] Distributional Reinforcement Learning for Multi-Dimensional Reward Functions](http://arxiv.org/abs/2110.13578)


  A growing trend for value-based reinforcement learning (RL) algorithms is to
capture more information than scalar value functions in the value network. One
of the most well-known methods in this branch is distributional RL, which
models return distribution instead of scalar value. In another line of work,
hybrid reward architectures (HRA) in RL have studied to model source-specific
value functions for each source of reward, which is also shown to be beneficial
in performance. To fully inherit the benefits of distributional RL and hybrid
reward architectures, we introduce Multi-Dimensional Distributional DQN
(MD3QN), which extends distributional RL to model the joint return distribution
from multiple reward sources. As a by-product of joint distribution modeling,
MD3QN can capture not only the randomness in returns for each source of reward,
but also the rich reward correlation between the randomness of different
sources. We prove the convergence for the joint distributional Bellman operator
and build our empirical algorithm by minimizing the Maximum Mean Discrepancy
between joint return distribution and its Bellman target. In experiments, our
method accurately models the joint return distribution in environments with
richly correlated reward functions, and outperforms previous RL methods
utilizing multi-dimensional reward functions in the control setting.

    

### [[2110.13581] Gradient representations in ReLU networks as similarity functions](http://arxiv.org/abs/2110.13581)


  Feed-forward networks can be interpreted as mappings with linear decision
surfaces at the level of the last layer. We investigate how the tangent space
of the network can be exploited to refine the decision in case of ReLU
(Rectified Linear Unit) activations. We show that a simple Riemannian metric
parametrized on the parameters of the network forms a similarity function at
least as good as the original network and we suggest a sparse metric to
increase the similarity gap.

    

### [[2110.13583] Real-time Human Response Prediction Using a Non-intrusive Data-driven Model Reduction Scheme](http://arxiv.org/abs/2110.13583)


  Recent research in non-intrusive data-driven model order reduction (MOR)
enabled accurate and efficient approximation of parameterized ordinary
differential equations (ODEs). However, previous studies have focused on
constant parameters, whereas time-dependent parameters have been neglected. The
purpose of this paper is to introduce a novel two-step MOR scheme to tackle
this issue. In a first step, classic MOR approaches are applied to calculate a
low-dimensional representation of high-dimensional ODE solutions, i.e. to
extract the most important features of simulation data. Based on this
representation, a long short-term memory (LSTM) is trained to predict the
reduced dynamics iteratively in a second step. This enables the parameters to
be taken into account during the respective time step. The potential of this
approach is demonstrated on an occupant model within a car driving scenario.
The reduced model's response to time-varying accelerations matches the
reference data with high accuracy for a limited amount of time. Furthermore,
real-time capability is achieved. Accordingly, it is concluded that the
presented method is well suited to approximate parameterized ODEs and can
handle time-dependent parameters in contrast to common methods.

    

### [[2110.13585] Concepts for Automated Machine Learning in Smart Grid Applications](http://arxiv.org/abs/2110.13585)


  Undoubtedly, the increase of available data and competitive machine learning
algorithms has boosted the popularity of data-driven modeling in energy
systems. Applications are forecasts for renewable energy generation and energy
consumption. Forecasts are elementary for sector coupling, where
energy-consuming sectors are interconnected with the power-generating sector to
address electricity storage challenges by adding flexibility to the power
system. However, the large-scale application of machine learning methods in
energy systems is impaired by the need for expert knowledge, which covers
machine learning expertise and a profound understanding of the application's
process. The process knowledge is required for the problem formalization, as
well as the model validation and application. The machine learning skills
include the processing steps of i) data pre-processing, ii) feature
engineering, extraction, and selection, iii) algorithm selection, iv)
hyperparameter optimization, and possibly v) post-processing of the model's
output. Tailoring a model for a particular application requires selecting the
data, designing various candidate models and organizing the data flow between
the processing steps, selecting the most suitable model, and monitoring the
model during operation - an iterative and time-consuming procedure. Automated
design and operation of machine learning aim to reduce the human effort to
address the increasing demand for data-driven models. We define five levels of
automation for forecasting in alignment with the SAE standard for autonomous
vehicles, where manual design and application reflect Automation level 0.

    

### [[2110.13587] Arbitrary Distribution Modeling with Censorship in Real-Time Bidding Advertising](http://arxiv.org/abs/2110.13587)


  The purpose of Inventory Pricing is to bid the right prices to online ad
opportunities, which is crucial for a Demand-Side Platform (DSP) to win
advertising auctions in Real-Time Bidding (RTB). In the planning stage,
advertisers need the forecast of probabilistic models to make bidding
decisions. However, most of the previous works made strong assumptions on the
distribution form of the winning price, which reduced their accuracy and
weakened their ability to make generalizations. Though some works recently
tried to fit the distribution directly, their complex structure lacked
efficiency on online inference. In this paper, we devise a novel loss function,
Neighborhood Likelihood Loss (NLL), collaborating with a proposed framework,
Arbitrary Distribution Modeling (ADM), to predict the winning price
distribution under censorship with no pre-assumption required. We conducted
experiments on two real-world experimental datasets and one large-scale,
non-simulated production dataset in our system. Experiments showed that ADM
outperformed the baselines both on algorithm and business metrics. By replaying
historical data of the production environment, this method was shown to lead to
good yield in our system. Without any pre-assumed specific distribution form,
ADM showed significant advantages in effectiveness and efficiency,
demonstrating its great capability in modeling sophisticated price landscapes.

    

### [[2110.13596] TME-BNA: Temporal Motif-Preserving Network Embedding with Bicomponent Neighbor Aggregation](http://arxiv.org/abs/2110.13596)


  Evolving temporal networks serve as the abstractions of many real-life
dynamic systems, e.g., social network and e-commerce. The purpose of temporal
network embedding is to map each node to a time-evolving low-dimension vector
for downstream tasks, e.g., link prediction and node classification. The
difficulty of temporal network embedding lies in how to utilize the topology
and time information jointly to capture the evolution of a temporal network. In
response to this challenge, we propose a temporal motif-preserving network
embedding method with bicomponent neighbor aggregation, named TME-BNA.
Considering that temporal motifs are essential to the understanding of topology
laws and functional properties of a temporal network, TME-BNA constructs
additional edge features based on temporal motifs to explicitly utilize complex
topology with time information. In order to capture the topology dynamics of
nodes, TME-BNA utilizes Graph Neural Networks (GNNs) to aggregate the
historical and current neighbors respectively according to the timestamps of
connected edges. Experiments are conducted on three public temporal network
datasets, and the results show the effectiveness of TME-BNA.

    

### [[2110.13601] DAG Card is the new Model Card](http://arxiv.org/abs/2110.13601)


  With the progressive commoditization of modeling capabilities, data-centric
AI recognizes that what happens before and after training becomes crucial for
real-world deployments. Following the intuition behind Model Cards, we propose
DAG Cards as a form of documentation encompassing the tenets of a data-centric
point of view. We argue that Machine Learning pipelines (rather than models)
are the most appropriate level of documentation for many practical use cases,
and we share with the community an open implementation to generate cards from
code.

    

### [[2110.13610] Robust physics discovery via supervised and unsupervised pattern recognition using the Euler characteristic](http://arxiv.org/abs/2110.13610)


  Machine learning approaches have been widely used for discovering the
underlying physics of dynamical systems from measured data. Existing
approaches, however, still lack robustness, especially when the measured data
contain a large level of noise. The lack of robustness is mainly attributed to
the insufficient representativeness of used features. As a result, the
intrinsic mechanism governing the observed system cannot be accurately
identified. In this study, we use an efficient topological descriptor for
complex data, i.e., the Euler characteristics (ECs), as features to
characterize the spatiotemporal data collected from dynamical systems and
discover the underlying physics. Unsupervised manifold learning and supervised
classification results show that EC can be used to efficiently distinguish
systems with different while similar governing models. We also demonstrate that
the machine learning approaches using EC can improve the confidence level of
sparse regression methods of physics discovery.

    

### [[2110.13611] Dendritic Self-Organizing Maps for Continual Learning](http://arxiv.org/abs/2110.13611)


  Current deep learning architectures show remarkable performance when trained
in large-scale, controlled datasets. However, the predictive ability of these
architectures significantly decreases when learning new classes incrementally.
This is due to their inclination to forget the knowledge acquired from
previously seen data, a phenomenon termed catastrophic-forgetting. On the other
hand, Self-Organizing Maps (SOMs) can model the input space utilizing
constrained k-means and thus maintain past knowledge. Here, we propose a novel
algorithm inspired by biological neurons, termed Dendritic-Self-Organizing Map
(DendSOM). DendSOM consists of a single layer of SOMs, which extract patterns
from specific regions of the input space accompanied by a set of hit matrices,
one per SOM, which estimate the association between units and labels. The
best-matching unit of an input pattern is selected using the maximum cosine
similarity rule, while the point-wise mutual information is employed for class
inference. DendSOM performs unsupervised feature extraction as it does not use
labels for targeted updating of the weights. It outperforms classical SOMs and
several state-of-the-art continual learning algorithms on benchmark datasets,
such as the Split-MNIST and Split-CIFAR-10. We propose that the incorporation
of neuronal properties in SOMs may help remedy catastrophic forgetting.

    

### [[2110.13619] Vaccine skepticism detection by network embedding](http://arxiv.org/abs/2110.13619)


  We demonstrate the applicability of network embedding to vaccine skepticism,
a controversial topic of long-past history. With the Covid-19 pandemic outbreak
at the end of 2019, the topic is more important than ever. Only a year after
the first international cases were registered, multiple vaccines were developed
and passed clinical testing. Besides the challenges of development, testing,
and logistics, another factor that might play a significant role in the fight
against the pandemic are people who are hesitant to get vaccinated, or even
state that they will refuse any vaccine offered to them. Two groups of people
commonly referred to as a) pro-vaxxer, those who support vaccinating people b)
vax-skeptic, those who question vaccine efficacy or the need for general
vaccination against Covid-19. It is very difficult to tell exactly how many
people share each of these views. It is even more difficult to understand all
the reasoning why vax-skeptic opinions are getting more popular. In this work,
our intention was to develop techniques that are able to efficiently
differentiate between pro-vaxxer and vax-skeptic content. After multiple data
preprocessing steps, we analyzed the tweet text as well as the structure of
user interactions on Twitter. We deployed several node embedding and community
detection models that scale well for graphs with millions of edges.

    

### [[2110.13623] Contrastive Neural Processes for Self-Supervised Learning](http://arxiv.org/abs/2110.13623)


  Recent contrastive methods show significant improvement in self-supervised
learning in several domains. In particular, contrastive methods are most
effective where data augmentation can be easily constructed e.g. in computer
vision. However, they are less successful in domains without established data
transformations such as time series data. In this paper, we propose a novel
self-supervised learning framework that combines contrastive learning with
neural processes. It relies on recent advances in neural processes to perform
time series forecasting. This allows to generate augmented versions of data by
employing a set of various sampling functions and, hence, avoid manually
designed augmentations. We extend conventional neural processes and propose a
new contrastive loss to learn times series representations in a self-supervised
setup. Therefore, unlike previous self-supervised methods, our augmentation
pipeline is task-agnostic, enabling our method to perform well across various
applications. In particular, a ResNet with a linear classifier trained using
our approach is able to outperform state-of-the-art techniques across
industrial, medical and audio datasets improving accuracy over 10% in ECG
periodic data. We further demonstrate that our self-supervised representations
are more efficient in the latent space, improving multiple clustering indexes
and that fine-tuning our method on 10% of labels achieves results competitive
to fully-supervised learning.

    

### [[2110.13624] Deep Learning-based Technology Fitness Landscape: A Biological Analogy](http://arxiv.org/abs/2110.13624)


  This research note presents a deep learning-based technology fitness
landscape premised on a technology embedding space and the estimated
improvement rates of all domains in it. The technology embedding space is
trained via neural embedding techniques on both intrinsic (semantic) features
and connective (citation) information to derive high-dimensional embedding
vectors for the 1,757 technology domains curated by Singh et al. (2021),
covering 97.2% of the patent database. The estimated improvement rates of these
1,757 domains were also drawn from Singh et al. (2021). The technology fitness
landscape exhibits a high hill related to information, electronics, and
electrical technologies and a vast low plain of the remaining domains. The
construction of the technology fitness landscape based on neural embedding
training presents a global picture and bird's eye view of the co-evolution of
heterogeneous technology domains in the unified technology space.

    

### [[2110.13625] Landmark-Guided Subgoal Generation in Hierarchical Reinforcement Learning](http://arxiv.org/abs/2110.13625)


  Goal-conditioned hierarchical reinforcement learning (HRL) has shown
promising results for solving complex and long-horizon RL tasks. However, the
action space of high-level policy in the goal-conditioned HRL is often large,
so it results in poor exploration, leading to inefficiency in training. In this
paper, we present HIerarchical reinforcement learning Guided by Landmarks
(HIGL), a novel framework for training a high-level policy with a reduced
action space guided by landmarks, i.e., promising states to explore. The key
component of HIGL is twofold: (a) sampling landmarks that are informative for
exploration and (b) encouraging the high-level policy to generate a subgoal
towards a selected landmark. For (a), we consider two criteria: coverage of the
entire visited state space (i.e., dispersion of states) and novelty of states
(i.e., prediction error of a state). For (b), we select a landmark as the very
first landmark in the shortest path in a graph whose nodes are landmarks. Our
experiments demonstrate that our framework outperforms prior-arts across a
variety of control tasks, thanks to efficient exploration guided by landmarks.

    

### [[2110.13627] Degree-Based Random Walk Approach for Graph Embedding](http://arxiv.org/abs/2110.13627)


  Graph embedding, representing local and global neighborhood information by
numerical vectors, is a crucial part of the mathematical modeling of a wide
range of real-world systems. Among the embedding algorithms, random walk-based
algorithms have proven to be very successful. These algorithms collect
information by creating numerous random walks with a redefined number of steps.
Creating random walks is the most demanding part of the embedding process. The
computation demand increases with the size of the network. Moreover, for
real-world networks, considering all nodes on the same footing, the abundance
of low-degree nodes creates an imbalanced data problem. In this work, a
computationally less intensive and node connectivity aware uniform sampling
method is proposed. In the proposed method, the number of random walks is
created proportionally with the degree of the node. The advantages of the
proposed algorithm become more enhanced when the algorithm is applied to large
graphs. A comparative study by using two networks namely CORA and CiteSeer is
presented. Comparing with the fixed number of walks case, the proposed method
requires 50% less computational effort to reach the same accuracy for node
classification and link prediction calculations.

    

### [[2110.13629] Bayesian Optimization and Deep Learning forsteering wheel angle prediction](http://arxiv.org/abs/2110.13629)


  Automated driving systems (ADS) have undergone a significant improvement in
the last years. ADS and more precisely self-driving cars technologies will
change the way we perceive and know the world of transportation systems in
terms of user experience, mode choices and business models. The emerging field
of Deep Learning (DL) has been successfully applied for the development of
innovative ADS solutions. However, the attempt to single out the best deep
neural network architecture and tuning its hyperparameters are all expensive
processes, both in terms of time and computational resources. In this work,
Bayesian Optimization (BO) is used to optimize the hyperparameters of a
Spatiotemporal-Long Short Term Memory (ST-LSTM) network with the aim to obtain
an accurate model for the prediction of the steering angle in a ADS. BO was
able to identify, within a limited number of trials, a model -- namely
BOST-LSTM -- which resulted, on a public dataset, the most accurate when
compared to classical end-to-end driving models.

    

### [[2110.13632] Generative Networks for Precision Enthusiasts](http://arxiv.org/abs/2110.13632)


  Generative networks are opening new avenues in fast event generation for the
LHC. We show how generative flow networks can reach percent-level precision for
kinematic distributions, how they can be trained jointly with a discriminator,
and how this discriminator improves the generation. Our joint training relies
on a novel coupling of the two networks which does not require a Nash
equilibrium. We then estimate the generation uncertainties through a Bayesian
network setup and through conditional data augmentation, while the
discriminator ensures that there are no systematic inconsistencies compared to
the training data.

    

### [[2110.13633] Optimal non-pharmaceutical intervention policy for Covid-19 epidemic via neuroevolution algorithm](http://arxiv.org/abs/2110.13633)


  National responses to the Covid-19 pandemic varied markedly across countries,
from business-as-usual to complete shutdowns. Policies aimed at disrupting the
viral transmission cycle and preventing the healthcare system from being
overwhelmed, simultaneously exact an economic toll. We developed a intervention
policy model that comprised the relative human, economic and healthcare costs
of non-pharmaceutical epidemic intervention and arrived at the optimal strategy
using the neuroevolution algorithm. The proposed model finds the minimum
required reduction in contact rates to maintain the burden on the healthcare
system below the maximum capacity. We find that such a policy renders a sharp
increase in the control strength at the early stages of the epidemic, followed
by a steady increase in the subsequent ten weeks as the epidemic approaches its
peak, and finally control strength is gradually decreased as the population
moves towards herd immunity. We have also shown how such a model can provide an
efficient adaptive intervention policy at different stages of the epidemic
without having access to the entire history of its progression in the
population. This work emphasizes the importance of imposing intervention
measures early and provides insights into adaptive intervention policies to
minimize the economic impacts of the epidemic without putting an extra burden
on the healthcare system.

    

### [[2110.13638] EDLaaS; Fully Homomorphic Encryption Over Neural Network Graphs](http://arxiv.org/abs/2110.13638)


  We present automatically parameterised Fully Homomorphic Encryption (FHE),
for encrypted neural network inference. We present and exemplify our inference
over FHE compatible neural networks with our own open-source framework and
reproducible step-by-step examples. We use the 4th generation Cheon, Kim, Kim
and Song (CKKS) FHE scheme over fixed points provided by the Microsoft Simple
Encrypted Arithmetic Library (MS-SEAL). We significantly enhance the usability
and applicability of FHE in deep learning contexts, with a focus on the
constituent graphs, traversal, and optimisation. We find that FHE is not a
panacea for all privacy preserving machine learning (PPML) problems, and that
certain limitations still remain, such as model training. However we also find
that in certain contexts FHE is well suited for computing completely private
predictions with neural networks. We focus on convolutional neural networks
(CNNs), fashion-MNIST, and levelled FHE operations. The ability to privately
compute sensitive problems more easily, while lowering the barriers to entry,
can allow otherwise too-sensitive fields to begin advantaging themselves of
performant third-party neural networks. Lastly we show encrypted deep learning,
applied to a sensitive real world problem in agri-food, and how this can have a
large positive impact on food-waste and encourage much-needed data sharing.

    

### [[2110.13649] An algorithm for the computation of joint Hawkes moments with exponential kernel](http://arxiv.org/abs/2110.13649)


  The purpose of this paper is to present a recursive algorithm and its
implementation in Maple and Mathematica for the computation of joint moments
and cumulants of Hawkes processes with exponential kernels. Numerical results
and computation times are also discussed. Obtaining closed form expressions can
be computationally intensive, as joint fifth cumulant and moment formulas can
be respectively expanded into up to 3,288 and 27,116 summands.

    

### [[2110.13652] A Precision Diagnostic Framework of Renal Cell Carcinoma on Whole-Slide Images using Deep Learning](http://arxiv.org/abs/2110.13652)


  Diagnostic pathology, which is the basis and gold standard of cancer
diagnosis, provides essential information on the prognosis of the disease and
vital evidence for clinical treatment. Tumor region detection, subtype and
grade classification are the fundamental diagnostic indicators for renal cell
carcinoma (RCC) in whole-slide images (WSIs). However, pathological diagnosis
is subjective, differences in observation and diagnosis between pathologists is
common in hospitals with inadequate diagnostic capacity. The main challenge for
developing deep learning based RCC diagnostic system is the lack of large-scale
datasets with precise annotations. In this work, we proposed a deep
learning-based framework for analyzing histopathological images of patients
with renal cell carcinoma, which has the potential to achieve pathologist-level
accuracy in diagnosis. A deep convolutional neural network (InceptionV3) was
trained on the high-quality annotated dataset of The Cancer Genome Atlas (TCGA)
whole-slide histopathological image for accurate tumor area detection,
classification of RCC subtypes, and ISUP grades classification of clear cell
carcinoma subtypes. These results suggest that our framework can help
pathologists in the detection of cancer region and classification of subtypes
and grades, which could be applied to any cancer type, providing auxiliary
diagnosis and promoting clinical consensus.

    

### [[2110.13653] Learning Speaker Representation with Semi-supervised Learning approach for Speaker Profiling](http://arxiv.org/abs/2110.13653)


  Speaker profiling, which aims to estimate speaker characteristics such as age
and height, has a wide range of applications inforensics, recommendation
systems, etc. In this work, we propose a semisupervised learning approach to
mitigate the issue of low training data for speaker profiling. This is done by
utilizing external corpus with speaker information to train a better
representation which can help to improve the speaker profiling systems.
Specifically, besides the standard supervised learning path, the proposed
framework has two more paths: (1) an unsupervised speaker representation
learning path that helps to capture the speaker information; (2) a consistency
training path that helps to improve the robustness of the system by enforcing
it to produce similar predictions for utterances of the same speaker.The
proposed approach is evaluated on the TIMIT and NISP datasets for age, height,
and gender estimation, while the Librispeech is used as the unsupervised
external corpus. Trained both on single-task and multi-task settings, our
approach was able to achieve state-of-the-art results on age estimation on the
TIMIT Test dataset with Root Mean Square Error(RMSE) of6.8 and 7.4 years and
Mean Absolute Error(MAE) of 4.8 and5.0 years for male and female speakers
respectively.

    

### [[2110.13655] Bridging the gap to real-world for network intrusion detection systems with data-centric approach](http://arxiv.org/abs/2110.13655)


  Most research using machine learning (ML) for network intrusion detection
systems (NIDS) uses well-established datasets such as KDD-CUP99, NSL-KDD,
UNSW-NB15, and CICIDS-2017. In this context, the possibilities of machine
learning techniques are explored, aiming for metrics improvements compared to
the published baselines (model-centric approach). However, those datasets
present some limitations as aging that make it unfeasible to transpose those
ML-based solutions to real-world applications. This paper presents a systematic
data-centric approach to address the current limitations of NIDS research,
specifically the datasets. This approach generates NIDS datasets composed of
the most recent network traffic and attacks, with the labeling process
integrated by design.

    

### [[2110.13656] CLLD: Contrastive Learning with Label Distance for Text Classificatioin](http://arxiv.org/abs/2110.13656)


  Existed pre-trained models have achieved state-of-the-art performance on
various text classification tasks. These models have proven to be useful in
learning universal language representations. However, the semantic discrepancy
between similar texts cannot be effectively distinguished by advanced
pre-trained models, which have a great influence on the performance of
hard-to-distinguish classes. To address this problem, we propose a novel
Contrastive Learning with Label Distance (CLLD) in this work. Inspired by
recent advances in contrastive learning, we specifically design a
classification method with label distance for learning contrastive classes.
CLLD ensures the flexibility within the subtle differences that lead to
different label assignments, and generates the distinct representations for
each class having similarity simultaneously. Extensive experiments on public
benchmarks and internal datasets demonstrate that our method improves the
performance of pre-trained models on classification tasks. Importantly, our
experiments suggest that the learned label distance relieve the adversarial
nature of interclasses.

    

### [[2110.13658] Can Character-based Language Models Improve Downstream Task Performance in Low-Resource and Noisy Language Scenarios?](http://arxiv.org/abs/2110.13658)


  Recent impressive improvements in NLP, largely based on the success of
contextual neural language models, have been mostly demonstrated on at most a
couple dozen high-resource languages. Building language models and, more
generally, NLP systems for non-standardized and low-resource languages remains
a challenging task. In this work, we focus on North-African colloquial
dialectal Arabic written using an extension of the Latin script, called
NArabizi, found mostly on social media and messaging communication. In this
low-resource scenario with data displaying a high level of variability, we
compare the downstream performance of a character-based language model on
part-of-speech tagging and dependency parsing to that of monolingual and
multilingual models. We show that a character-based model trained on only 99k
sentences of NArabizi and fined-tuned on a small treebank of this language
leads to performance close to those obtained with the same architecture
pre-trained on large multilingual and monolingual models. Confirming these
results a on much larger data set of noisy French user-generated content, we
argue that such character-based language models can be an asset for NLP in
low-resource and high language variability set-tings.

    

### [[2110.13661] Hybrid physics-based and data-driven modeling with calibrated uncertainty for lithium-ion battery degradation diagnosis and prognosis](http://arxiv.org/abs/2110.13661)


  Advancing lithium-ion batteries (LIBs) in both design and usage is key to
promoting electrification in the coming decades to mitigate human-caused
climate change. Inadequate understanding of LIB degradation is an important
bottleneck that limits battery durability and safety. Here, we propose hybrid
physics-based and data-driven modeling for online diagnosis and prognosis of
battery degradation. Compared to existing battery modeling efforts, we aim to
build a model with physics as its backbone and statistical learning techniques
as enhancements. Such a hybrid model has better generalizability and
interpretability together with a well-calibrated uncertainty associated with
its prediction, rendering it more valuable and relevant to safety-critical
applications under realistic usage scenarios.

    

### [[2110.13664] Iterative Rule Extension for Logic Analysis of Data: an MILP-based heuristic to derive interpretable binary classification from large datasets](http://arxiv.org/abs/2110.13664)


  Data-driven decision making is rapidly gaining popularity, fueled by the
ever-increasing amounts of available data and encouraged by the development of
models that can identify beyond linear input-output relationships.
Simultaneously the need for interpretable prediction- and classification
methods is increasing, as this improves both our trust in these models and the
amount of information we can abstract from data. An important aspect of this
interpretability is to obtain insight in the sensitivity-specificity trade-off
constituted by multiple plausible input-output relationships. These are often
shown in a receiver operating characteristic (ROC) curve. These developments
combined lead to the need for a method that can abstract complex yet
interpretable input-output relationships from large data, i.e. data containing
large numbers of samples and sample features. Boolean phrases in disjunctive
normal form (DNF) are highly suitable for explaining non-linear input-output
relationships in a comprehensible way. Mixed integer linear programming (MILP)
can be used to abstract these Boolean phrases from binary data, though its
computational complexity prohibits the analysis of large datasets. This work
presents IRELAND, an algorithm that allows for abstracting Boolean phrases in
DNF from data with up to 10,000 samples and sample characteristics. The results
show that for large datasets IRELAND outperforms the current state-of-the-art
and can find solutions for datasets where current models run out of memory or
need excessive runtimes. Additionally, by construction IRELAND allows for an
efficient computation of the sensitivity-specificity trade-off curve, allowing
for further understanding of the underlying input-output relationship.

    

### [[2110.13674] C$^2$SP-Net: Joint Compression and Classification Network for Epilepsy Seizure Prediction](http://arxiv.org/abs/2110.13674)


  Recent development in brain-machine interface technology has made seizure
prediction possible. However, the communication of large volume of
electrophysiological signals between sensors and processing apparatus and
related computation become two major bottlenecks for seizure prediction systems
due to the constrained bandwidth and limited computation resource, especially
for wearable and implantable medical devices. Although compressive sensing (CS)
can be adopted to compress the signals to reduce communication bandwidth
requirement, it needs a complex reconstruction procedure before the signal can
be used for seizure prediction. In this paper, we propose C$^2$SP-Net, to
jointly solve compression, prediction, and reconstruction with a single neural
network. A plug-and-play in-sensor compression matrix is constructed to reduce
transmission bandwidth requirement. The compressed signal can be used for
seizure prediction without additional reconstruction steps. Reconstruction of
the original signal can also be carried out in high fidelity. Prediction
accuracy, sensitivity, false prediction rate, and reconstruction quality of the
proposed framework are evaluated under various compression ratios. The
experimental results illustrate that our model outperforms the competitive
state-of-the-art baselines by a large margin in prediction accuracy. In
particular, our proposed method produces an average loss of 0.35 % in
prediction accuracy with a compression ratio ranging from 1/2 to 1/16.

    

### [[2110.13680] Uncertainty quantification in a mechanical submodel driven by a Wasserstein-GAN](http://arxiv.org/abs/2110.13680)


  The analysis of parametric and non-parametric uncertainties of very large
dynamical systems requires the construction of a stochastic model of said
system. Linear approaches relying on random matrix theory and principal
componant analysis can be used when systems undergo low-frequency vibrations.
In the case of fast dynamics and wave propagation, we investigate a random
generator of boundary conditions for fast submodels by using machine learning.
We show that the use of non-linear techniques in machine learning and
data-driven methods is highly relevant.
Physics-informed neural networks is a possible choice for a data-driven
method to replace linear modal analysis. An architecture that support a random
component is necessary for the construction of the stochastic model of the
physical system for non-parametric uncertainties, since the goal is to learn
the underlying probabilistic distribution of uncertainty in the data.
Generative Adversarial Networks (GANs) are suited for such applications, where
the Wasserstein-GAN with gradient penalty variant offers improved convergence
results for our problem.
The objective of our approach is to train a GAN on data from a finite element
method code (Fenics) so as to extract stochastic boundary conditions for faster
finite element predictions on a submodel. The submodel and the training data
have both the same geometrical support. It is a zone of interest for
uncertainty quantification and relevant to engineering purposes. In the
exploitation phase, the framework can be viewed as a randomized and
parametrized simulation generator on the submodel, which can be used as a Monte
Carlo estimator.

    

### [[2110.13688] A Closer Look at Reference Learning for Fourier Phase Retrieval](http://arxiv.org/abs/2110.13688)


  Reconstructing images from their Fourier magnitude measurements is a problem
that often arises in different research areas. This process is also referred to
as phase retrieval. In this work, we consider a modified version of the phase
retrieval problem, which allows for a reference image to be added onto the
image before the Fourier magnitudes are measured. We analyze an unrolled
Gerchberg-Saxton (GS) algorithm that can be used to learn a good reference
image from a dataset. Furthermore, we take a closer look at the learned
reference images and propose a simple and efficient heuristic to construct
reference images that, in some cases, yields reconstructions of comparable
quality as approaches that learn references. Our code is available at
this https URL.

    

### [[2110.13705] Causal Effect Estimation using Variational Information Bottleneck](http://arxiv.org/abs/2110.13705)


  Causal inference is to estimate the causal effect in a causal relationship
when intervention is applied. Precisely, in a causal model with binary
interventions, i.e., control and treatment, the causal effect is simply the
difference between the factual and counterfactual. The difficulty is that the
counterfactual may never been obtained which has to be estimated and so the
causal effect could only be an estimate. The key challenge for estimating the
counterfactual is to identify confounders which effect both outcomes and
treatments. A typical approach is to formulate causal inference as a supervised
learning problem and so counterfactual could be predicted. Including linear
regression and deep learning models, recent machine learning methods have been
adapted to causal inference. In this paper, we propose a method to estimate
Causal Effect by using Variational Information Bottleneck (CEVIB). The
promising point is that VIB is able to naturally distill confounding variables
from the data, which enables estimating causal effect by using observational
data. We have compared CEVIB to other methods by applying them to three data
sets showing that our approach achieved the best performance. We also
experimentally showed the robustness of our method.

    

### [[2110.13711] Hierarchical Transformers Are More Efficient Language Models](http://arxiv.org/abs/2110.13711)


  Transformer models yield impressive results on many NLP and sequence modeling
tasks. Remarkably, Transformers can handle long sequences which allows them to
produce long coherent outputs: full paragraphs produced by GPT-3 or
well-structured images produced by DALL-E. These large language models are
impressive but also very inefficient and costly, which limits their
applications and accessibility. We postulate that having an explicit
hierarchical architecture is the key to Transformers that efficiently handle
long sequences. To verify this claim, we first study different ways to
downsample and upsample activations in Transformers so as to make them
hierarchical. We use the best performing upsampling and downsampling layers to
create Hourglass - a hierarchical Transformer language model. Hourglass
improves upon the Transformer baseline given the same amount of computation and
can yield the same results as Transformers more efficiently. In particular,
Hourglass sets new state-of-the-art for Transformer models on the ImageNet32
generation task and improves language modeling efficiency on the widely studied
enwik8 benchmark.

    

### [[2110.13716] HIST: A Graph-based Framework for Stock Trend Forecasting via Mining Concept-Oriented Shared Information](http://arxiv.org/abs/2110.13716)


  Stock trend forecasting, which forecasts stock prices' future trends, plays
an essential role in investment. The stocks in a market can share information
so that their stock prices are highly correlated. Several methods were recently
proposed to mine the shared information through stock concepts (e.g.,
technology, Internet Retail) extracted from the Web to improve the forecasting
results. However, previous work assumes the connections between stocks and
concepts are stationary, and neglects the dynamic relevance between stocks and
concepts, limiting the forecasting results. Moreover, existing methods overlook
the invaluable shared information carried by hidden concepts, which measure
stocks' commonness beyond the manually defined stock concepts. To overcome the
shortcomings of previous work, we proposed a novel stock trend forecasting
framework that can adequately mine the concept-oriented shared information from
predefined concepts and hidden concepts. The proposed framework simultaneously
utilize the stock's shared information and individual information to improve
the stock trend forecasting performance. Experimental results on the real-world
tasks demonstrate the efficiency of our framework on stock trend forecasting.
The investment simulation shows that our framework can achieve a higher
investment return than the baselines.

    

### [[2110.13721] Geometric Transformer for End-to-End Molecule Properties Prediction](http://arxiv.org/abs/2110.13721)


  Transformers have become methods of choice in many applications thanks to
their ability to represent complex interaction between elements. However,
extending the Transformer architecture to non-sequential data such as molecules
and enabling its training on small datasets remain a challenge. In this work,
we introduce a Transformer-based architecture for molecule property prediction,
which is able to capture the geometry of the molecule. We modify the classical
positional encoder by an initial encoding of the molecule geometry, as well as
a learned gated self-attention mechanism. We further suggest an augmentation
scheme for molecular data capable of avoiding the overfitting induced by the
overparameterized architecture. The proposed framework outperforms the
state-of-the-art methods while being based on pure machine learning solely,
i.e. the method does not incorporate domain knowledge from quantum chemistry
and does not use extended geometric inputs beside the pairwise atomic
distances.

    

### [[2110.13732] Improving the efficacy of Deep Learning models for Heart Beat detection on heterogeneous datasets](http://arxiv.org/abs/2110.13732)


  Deep Learning (DL) have greatly contributed to bioelectric signals
processing, in particular to extract physiological markers. However, the
efficacy and applicability of the results proposed in the literature is often
constrained to the population represented by the data used to train the models.
In this study, we investigate the issues related to applying a DL model on
heterogeneous datasets. In particular, by focusing on heart beat detection from
Electrocardiogram signals (ECG), we show that the performance of a model
trained on data from healthy subjects decreases when applied to patients with
cardiac conditions and to signals collected with different devices. We then
evaluate the use of Transfer Learning (TL) to adapt the model to the different
datasets. In particular, we show that the classification performance is
improved, even with datasets with a small sample size. These results suggest
that a greater effort should be made towards generalizability of DL models
applied on bioelectric signals, in particular by retrieving more representative
datasets.

    

### [[2110.13741] Disrupting Deep Uncertainty Estimation Without Harming Accuracy](http://arxiv.org/abs/2110.13741)


  Deep neural networks (DNNs) have proven to be powerful predictors and are
widely used for various tasks. Credible uncertainty estimation of their
predictions, however, is crucial for their deployment in many risk-sensitive
applications. In this paper we present a novel and simple attack, which unlike
adversarial attacks, does not cause incorrect predictions but instead cripples
the network's capacity for uncertainty estimation. The result is that after the
attack, the DNN is more confident of its incorrect predictions than about its
correct ones without having its accuracy reduced. We present two versions of
the attack. The first scenario focuses on a black-box regime (where the
attacker has no knowledge of the target network) and the second scenario
attacks a white-box setting. The proposed attack is only required to be of
minuscule magnitude for its perturbations to cause severe uncertainty
estimation damage, with larger magnitudes resulting in completely unusable
uncertainty estimations. We demonstrate successful attacks on three of the most
popular uncertainty estimation methods: the vanilla softmax score, Deep
Ensembles and MC-Dropout. Additionally, we show an attack on SelectiveNet, the
selective classification architecture. We test the proposed attack on several
contemporary architectures such as MobileNetV2 and EfficientNetB0, all trained
to classify ImageNet.

    

### [[2110.13745] PARIS: Personalized Activity Recommendation for Improving Sleep Quality](http://arxiv.org/abs/2110.13745)


  The quality of sleep has a deep impact on people's physical and mental
health. People with insufficient sleep are more likely to report physical and
mental distress, activity limitation, anxiety, and pain. Moreover, in the past
few years, there has been an explosion of applications and devices for activity
monitoring and health tracking. Signals collected from these wearable devices
can be used to study and improve sleep quality. In this paper, we utilize the
relationship between physical activity and sleep quality to find ways of
assisting people improve their sleep using machine learning techniques. People
usually have several behavior modes that their bio-functions can be divided
into. Performing time series clustering on activity data, we find cluster
centers that would correlate to the most evident behavior modes for a specific
subject. Activity recipes are then generated for good sleep quality for each
behavior mode within each cluster. These activity recipes are supplied to an
activity recommendation engine for suggesting a mix of relaxed to intense
activities to subjects during their daily routines. The recommendations are
further personalized based on the subjects' lifestyle constraints, i.e. their
age, gender, body mass index (BMI), resting heart rate, etc, with the objective
of the recommendation being the improvement of that night's quality of sleep.
This would in turn serve a longer-term health objective, like lowering heart
rate, improving the overall quality of sleep, etc.

    

### [[2110.13748] Learning to Pre-process Laser Induced Breakdown Spectroscopy Signals Without Clean Data](http://arxiv.org/abs/2110.13748)


  This work tests whether deep neural networks can clean laser induced
breakdown spectroscopy (LIBS) signals by using only uncleaned raw measurements.
Our view of this problem considers a disentanglement of the effects of the
target of interest from those of the nuisance factors (with non-zero mean) by
leveraging the vast amounts of redundancies in LIBS data and our proposed
learning formulation. This later aims at promoting consistency between repeated
measurement views of a target while simultaneously removing consistencies with
all other LIBS measurements taken throughout the history of the instrument.
Evaluations on real data from the ChemCam instrument onboard the Martian
Curiosity rover show a superior performance in cleaning LIBS signals compared
to the standard approaches being used by the ChemCam team.

    

### [[2110.13749] Topologically penalized regression on manifolds](http://arxiv.org/abs/2110.13749)


  We study a regression problem on a compact manifold M. In order to take
advantage of the underlying geometry and topology of the data, the regression
task is performed on the basis of the first several eigenfunctions of the
Laplace-Beltrami operator of the manifold, that are regularized with
topological penalties. The proposed penalties are based on the topology of the
sub-level sets of either the eigenfunctions or the estimated function. The
overall approach is shown to yield promising and competitive performance on
various applications to both synthetic and real data sets. We also provide
theoretical guarantees on the regression function estimates, on both its
prediction error and its smoothness (in a topological sense). Taken together,
these results support the relevance of our approach in the case where the
targeted function is "topologically smooth".

    

### [[2110.13750] Optimizing Information-theoretical Generalization Bounds via Anisotropic Noise in SGLD](http://arxiv.org/abs/2110.13750)


  Recently, the information-theoretical framework has been proven to be able to
obtain non-vacuous generalization bounds for large models trained by Stochastic
Gradient Langevin Dynamics (SGLD) with isotropic noise. In this paper, we
optimize the information-theoretical generalization bound by manipulating the
noise structure in SGLD. We prove that with constraint to guarantee low
empirical risk, the optimal noise covariance is the square root of the expected
gradient covariance if both the prior and the posterior are jointly optimized.
This validates that the optimal noise is quite close to the empirical gradient
covariance. Technically, we develop a new information-theoretical bound that
enables such an optimization analysis. We then apply matrix analysis to derive
the form of optimal noise covariance. Presented constraint and results are
validated by the empirical observations.

    

### [[2110.13760] DPCOVID: Privacy-Preserving Federated Covid-19 Detection](http://arxiv.org/abs/2110.13760)


  Coronavirus (COVID-19) has shown an unprecedented global crisis by the
detrimental effect on the global economy and health. The number of COVID-19
cases has been rapidly increasing, and there is no sign of stopping. It leads
to a severe shortage of test kits and accurate detection models. A recent study
demonstrated that the chest X-ray radiography outperformed laboratory testing
in COVID-19 detection. Therefore, using chest X-ray radiography analysis can
help to screen suspected COVID-19 cases at an early stage. Moreover, the
patient data is sensitive, and it must be protected to avoid revealing through
model updates and reconstruction from the malicious attacker. In this paper, we
present a privacy-preserving Federated Learning system for COVID-19 detection
based on chest X-ray images. First, a Federated Learning system is constructed
from chest X-ray images. The main idea is to build a decentralized model across
multiple hospitals without sharing data among hospitals. Second, we first show
that the accuracy of Federated Learning for COVID-19 identification reduces
significantly for Non-IID data. We then propose a strategy to improve model's
accuracy on Non-IID COVID-19 data by increasing the total number of clients,
parallelism (client fraction), and computation per client. Finally, we apply a
Differential Privacy Stochastic Gradient Descent (DP-SGD) to enhance the
preserving of patient data privacy for our Federated Learning model. A strategy
is also proposed to keep the robustness of Federated Learning to ensure the
security and accuracy of the model.

    

### [[2110.13769] Min-similarity association rules for identifying past comorbidities of recurrent ED and inpatient patients](http://arxiv.org/abs/2110.13769)


  In the hospital setting, a small percentage of recurrent frequent patients
contribute to a disproportional amount of healthcare resource usage. Moreover,
in many of these cases, patient outcomes can be greatly improved by reducing
reoccurring visits, especially when they are associated with substance abuse,
mental health, and medical factors that could be improved by social-behavioral
interventions, outpatient or preventative care. To address this, we developed a
computationally efficient and interpretable framework that both identifies
recurrent patients with high utilization and determines which comorbidities
contribute most to their recurrent visits. Specifically, we present a novel
algorithm, called the minimum similarity association rules (MSAR), balancing
confidence-support trade-off, to determine the conditions most associated with
reoccurring Emergency department (ED) and inpatient visits. We validate MSAR on
a large Electric Health Record (EHR) dataset. Part of the solution is deployed
in Philips product Patient Flow Capacity Suite (PFCS).

    

### [[2110.13771] AugMax: Adversarial Composition of Random Augmentations for Robust Training](http://arxiv.org/abs/2110.13771)


  Data augmentation is a simple yet effective way to improve the robustness of
deep neural networks (DNNs). Diversity and hardness are two complementary
dimensions of data augmentation to achieve robustness. For example, AugMix
explores random compositions of a diverse set of augmentations to enhance
broader coverage, while adversarial training generates adversarially hard
samples to spot the weakness. Motivated by this, we propose a data augmentation
framework, termed AugMax, to unify the two aspects of diversity and hardness.
AugMax first randomly samples multiple augmentation operators and then learns
an adversarial mixture of the selected operators. Being a stronger form of data
augmentation, AugMax leads to a significantly augmented input distribution
which makes model training more challenging. To solve this problem, we further
design a disentangled normalization module, termed DuBIN
(Dual-Batch-and-Instance Normalization), that disentangles the instance-wise
feature heterogeneity arising from AugMax. Experiments show that AugMax-DuBIN
leads to significantly improved out-of-distribution robustness, outperforming
prior arts by 3.03%, 3.49%, 1.82% and 0.71% on CIFAR10-C, CIFAR100-C, Tiny
ImageNet-C and ImageNet-C. Codes and pretrained models are available:
this https URL.

    

### [[2110.13772] Data-Driven Time Series Reconstruction for Modern Power Systems Research](http://arxiv.org/abs/2110.13772)


  A critical aspect of power systems research is the availability of suitable
data, access to which is limited by privacy concerns and the sensitive nature
of energy infrastructure. This lack of data, in turn, hinders the development
of modern research avenues such as machine learning approaches or stochastic
formulations. To overcome this challenge, this paper proposes a systematic,
data-driven framework for reconstructing high-fidelity time series, using
publicly-available grid snapshots and historical data published by transmission
system operators. The proposed approach, from geo-spatial data and generation
capacity reconstruction, to time series disaggregation, is applied to the
French transmission grid. Thereby, synthetic but highly realistic time series
data, spanning multiple years with a 5-minute granularity, is generated at the
individual component level.

    

### [[2110.13786] Diversity and Generalization in Neural Network Ensembles](http://arxiv.org/abs/2110.13786)


  Ensembles are widely used in machine learning and, usually, provide
state-of-the-art performance in many prediction tasks. From the very beginning,
the diversity of an ensemble has been identified as a key factor for the
superior performance of these models. But the exact role that diversity plays
in ensemble models is poorly understood, specially in the context of neural
networks. In this work, we combine and expand previously published results in a
theoretically sound framework that describes the relationship between diversity
and ensemble performance for a wide range of ensemble methods. More precisely,
we provide sound answers to the following questions: how to measure diversity,
how diversity relates to the generalization error of an ensemble, and how
diversity is promoted by neural network ensemble algorithms. This analysis
covers three widely used loss functions, namely, the squared loss, the
cross-entropy loss, and the 0-1 loss; and two widely used model combination
strategies, namely, model averaging and weighted majority vote. We empirically
validate this theoretical analysis with neural network ensembles.

    

### [[2110.13796] Post-processing for Individual Fairness](http://arxiv.org/abs/2110.13796)


  Post-processing in algorithmic fairness is a versatile approach for
correcting bias in ML systems that are already used in production. The main
appeal of post-processing is that it avoids expensive retraining. In this work,
we propose general post-processing algorithms for individual fairness (IF). We
consider a setting where the learner only has access to the predictions of the
original model and a similarity graph between individuals, guiding the desired
fairness constraints. We cast the IF post-processing problem as a graph
smoothing problem corresponding to graph Laplacian regularization that
preserves the desired "treat similar individuals similarly" interpretation. Our
theoretical results demonstrate the connection of the new objective function to
a local relaxation of the original individual fairness. Empirically, our
post-processing algorithms correct individual biases in large-scale NLP models
such as BERT, while preserving accuracy.

    

### [[2110.13798] Tackling Oversmoothing of GNNs with Contrastive Learning](http://arxiv.org/abs/2110.13798)


  Graph neural networks (GNNs) integrate the comprehensive relation of graph
data and the representation learning capability of neural networks, which is
one of the most popular deep learning methods and achieves state-of-the-art
performance in many applications, such as natural language processing and
computer vision. In real-world scenarios, increasing the depth (i.e., the
number of layers) of GNNs is sometimes necessary to capture more latent
knowledge of the input data to mitigate the uncertainty caused by missing
values. However, involving more complex structures and more parameters will
decrease the performance of GNN models. One reason called oversmoothing is
recently introduced but the relevant research remains nascent. In general,
oversmoothing makes the final representations of nodes indiscriminative, thus
deteriorating the node classification and link prediction performance. In this
paper, we first survey the current de-oversmoothing methods and propose three
major metrics to evaluate a de-oversmoothing method, i.e., constant divergence
indicator, easy-to-determine divergence indicator, and model-agnostic strategy.
Then, we propose the Topology-guided Graph Contrastive Layer, named TGCL, which
is the first de-oversmoothing method maintaining all three mentioned metrics.
With the contrastive learning manner, we provide the theoretical analysis of
the effectiveness of the proposed TGCL. Last but not least, we design extensive
experiments to illustrate the empirical performance of TGCL comparing with
state-of-the-art baselines.

    

### [[2110.13799] Hinge Policy Optimization: Rethinking Policy Improvement and Reinterpreting PPO](http://arxiv.org/abs/2110.13799)


  Policy optimization is a fundamental principle for designing reinforcement
learning algorithms, and one example is the proximal policy optimization
algorithm with a clipped surrogate objective (PPO-clip), which has been
popularly used in deep reinforcement learning due to its simplicity and
effectiveness. Despite its superior empirical performance, PPO-clip has not
been justified via theoretical proof up to date. This paper proposes to rethink
policy optimization and reinterpret the theory of PPO-clip based on hinge
policy optimization (HPO), called to improve policy by hinge loss in this
paper. Specifically, we first identify sufficient conditions of state-wise
policy improvement and then rethink policy update as solving a large-margin
classification problem with hinge loss. By leveraging various types of
classifiers, the proposed design opens up a whole new family of policy-based
algorithms, including the PPO-clip as a special case. Based on this construct,
we prove that these algorithms asymptotically attain a globally optimal policy.
To our knowledge, this is the first ever that can prove global convergence to
an optimal policy for a variant of PPO-clip. We corroborate the performance of
a variety of HPO algorithms through experiments and an ablation study.

    

### [[2110.13805] Driving Style Recognition Using Interval Type-2 Fuzzy Inference System and Multiple Experts Decision Making](http://arxiv.org/abs/2110.13805)


  Driving styles summarize different driving behaviors that reflect in the
movements of the vehicles. These behaviors may indicate a tendency to perform
riskier maneuvers, consume more fuel or energy, break traffic rules, or drive
carefully. Therefore, this paper presents a driving style recognition using
Interval Type-2 Fuzzy Inference System with Multiple Experts Decision-Making
for classifying drivers into calm, moderate and aggressive. This system
receives as input features longitudinal and lateral kinematic parameters of the
vehicle motion. The type-2 fuzzy sets are more robust than type-1 fuzzy sets
when handling noisy data, because their membership function are also fuzzy
sets. In addition, a multiple experts approach can reduce the bias and
imprecision while building the fuzzy rulebase, which stores the knowledge of
the fuzzy system. The proposed approach was evaluated using descriptive
statistics analysis, and compared with clustering algorithms and a type-1 fuzzy
inference system. The results show the tendency to associate lower kinematic
profiles for the driving styles classified with the type-2 fuzzy inference
system when compared to other algorithms, which is in line with the more
conservative approach adopted in the aggregation of the experts' opinions.

    

### [[2110.13809] A deep learning based surrogate model for stochastic simulators](http://arxiv.org/abs/2110.13809)


  We propose a deep learning-based surrogate model for stochastic simulators.
The basic idea is to use generative neural network to approximate the
stochastic response. The challenge with such a framework resides in designing
the network architecture and selecting loss-function suitable for stochastic
response. While we utilize a simple feed-forward neural network, we propose to
use conditional maximum mean discrepancy (CMMD) as the loss-function. CMMD
exploits the property of reproducing kernel Hilbert space and allows capturing
discrepancy between the between the target and the neural network predicted
distributions. The proposed approach is mathematically rigorous, in the sense
that it makes no assumptions about the probability density function of the
response. Performance of the proposed approach is illustrated using four
benchmark problems selected from the literature. Results obtained indicate the
excellent performance of the proposed approach.

    

### [[2110.13813] Semantic Segmentation for Urban-Scene Images](http://arxiv.org/abs/2110.13813)


  Urban-scene Image segmentation is an important and trending topic in computer
vision with wide use cases like autonomous driving [1]. Starting with the
breakthrough work of Long et al. [2] that introduces Fully Convolutional
Networks (FCNs), the development of novel architectures and practical uses of
neural networks in semantic segmentation has been expedited in the recent 5
years. Aside from seeking solutions in general model design for information
shrinkage due to pooling, urban-scene image itself has intrinsic features like
positional patterns [3]. Our project seeks an advanced and integrated solution
that specifically targets urban-scene image semantic segmentation among the
most novel approaches in the current field. We re-implement the cutting edge
model DeepLabv3+ [4] with ResNet-101 [5] backbone as our strong baseline model.
Based upon DeepLabv3+, we incorporate HANet [3] to account for the vertical
spatial priors in urban-scene image tasks. To boost up model efficiency and
performance, we further explore the Atrous Spatial Pooling (ASP) layer in
DeepLabv3+ and infuse a computational efficient variation called "Waterfall"
Atrous Spatial Pooling (WASP) [6] architecture in our model. We find that our
two-step integrated model improves the mean Intersection-Over-Union (mIoU)
score gradually from the baseline model. In particular, HANet successfully
identifies height-driven patterns and improves per-class IoU of common class
labels in urban scenario like fence and bus. We also demonstrate the
improvement of model efficiency with help of WASP in terms of computational
times during training and parameter reduction from the original ASPP module.

    

### [[2110.13819] CloudFindr: A Deep Learning Cloud Artifact Masker for Satellite DEM Data](http://arxiv.org/abs/2110.13819)


  Artifact removal is an integral component of cinematic scientific
visualization, and is especially challenging with big datasets in which
artifacts are difficult to define. In this paper, we describe a method for
creating cloud artifact masks which can be used to remove artifacts from
satellite imagery using a combination of traditional image processing together
with deep learning based on U-Net. Compared to previous methods, our approach
does not require multi-channel spectral imagery but performs successfully on
single-channel Digital Elevation Models (DEMs). DEMs are a representation of
the topography of the Earth and have a variety applications including planetary
science, geology, flood modeling, and city planning.

    

### [[2110.13827] Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization](http://arxiv.org/abs/2110.13827)


  Self-Driven Particles (SDP) describe a category of multi-agent systems common
in everyday life, such as flocking birds and traffic flows. In a SDP system,
each agent pursues its own goal and constantly changes its cooperative or
competitive behaviors with its nearby agents. Manually designing the
controllers for such SDP system is time-consuming, while the resulting emergent
behaviors are often not realistic nor generalizable. Thus the realistic
simulation of SDP systems remains challenging. Reinforcement learning provides
an appealing alternative for automating the development of the controller for
SDP. However, previous multi-agent reinforcement learning (MARL) methods define
the agents to be teammates or enemies before hand, which fail to capture the
essence of SDP where the role of each agent varies to be cooperative or
competitive even within one episode. To simulate SDP with MARL, a key challenge
is to coordinate agents' behaviors while still maximizing individual
objectives. Taking traffic simulation as the testing bed, in this work we
develop a novel MARL method called Coordinated Policy Optimization (CoPO),
which incorporates social psychology principle to learn neural controller for
SDP. Experiments show that the proposed method can achieve superior performance
compared to MARL baselines in various metrics. Noticeably the trained vehicles
exhibit complex and diverse social behaviors that improve performance and
safety of the population as a whole. Demo video and source code are available
at: this https URL


### [[2110.13854] Learning Optimal Decision Trees Using MaxSAT](http://arxiv.org/abs/2110.13854)


  We present a Combinatorial Optimization approach based on Maximum
Satisfiability technology to compute Minimum Pure Decision Trees (MPDTs) for
the sake of interpretability. We show that our approach outperforms clearly in
terms of runtime previous approaches to compute MPDTs. We additionally show
that these MPDTs can outperform on average the DT classifiers generated with
sklearn in terms of accuracy. Therefore, our approach tackles favourably the
challenge of balancing interpretability and accuracy.

    

### [[2110.13855] Average-Reward Learning and Planning with Options](http://arxiv.org/abs/2110.13855)


  We extend the options framework for temporal abstraction in reinforcement
learning from discounted Markov decision processes (MDPs) to average-reward
MDPs. Our contributions include general convergent off-policy inter-option
learning algorithms, intra-option algorithms for learning values and models, as
well as sample-based planning variants of our learning algorithms. Our
algorithms and convergence proofs extend those recently developed by Wan, Naik,
and Sutton. We also extend the notion of option-interrupting behavior from the
discounted to the average-reward formulation. We show the efficacy of the
proposed algorithms with experiments on a continuing version of the Four-Room
domain.

    

### [[2110.13859] Defensive Tensorization](http://arxiv.org/abs/2110.13859)


  We propose defensive tensorization, an adversarial defence technique that
leverages a latent high-order factorization of the network. The layers of a
network are first expressed as factorized tensor layers. Tensor dropout is then
applied in the latent subspace, therefore resulting in dense reconstructed
weights, without the sparsity or perturbations typically induced by the
randomization.Our approach can be readily integrated with any arbitrary neural
architecture and combined with techniques like adversarial training. We
empirically demonstrate the effectiveness of our approach on standard image
classification benchmarks. We validate the versatility of our approach across
domains and low-precision architectures by considering an audio classification
task and binary networks. In all cases, we demonstrate improved performance
compared to prior works.

    

### [[2110.13864] FL-WBC: Enhancing Robustness against Model Poisoning Attacks in Federated Learning from a Client Perspective](http://arxiv.org/abs/2110.13864)


  Federated learning (FL) is a popular distributed learning framework that
trains a global model through iterative communications between a central server
and edge devices. Recent works have demonstrated that FL is vulnerable to model
poisoning attacks. Several server-based defense approaches (e.g. robust
aggregation), have been proposed to mitigate such attacks. However, we
empirically show that under extremely strong attacks, these defensive methods
fail to guarantee the robustness of FL. More importantly, we observe that as
long as the global model is polluted, the impact of attacks on the global model
will remain in subsequent rounds even if there are no subsequent attacks. In
this work, we propose a client-based defense, named White Blood Cell for
Federated Learning (FL-WBC), which can mitigate model poisoning attacks that
have already polluted the global model. The key idea of FL-WBC is to identify
the parameter space where long-lasting attack effect on parameters resides and
perturb that space during local training. Furthermore, we derive a certified
robustness guarantee against model poisoning attacks and a convergence
guarantee to FedAvg after applying our FL-WBC. We conduct experiments on
FasionMNIST and CIFAR10 to evaluate the defense against state-of-the-art model
poisoning attacks. The results demonstrate that our method can effectively
mitigate model poisoning attack impact on the global model within 5
communication rounds with nearly no accuracy drop under both IID and Non-IID
settings. Our defense is also complementary to existing server-based robust
aggregation approaches and can further improve the robustness of FL under
extremely strong attacks.

    

### [[2110.13876] Breaking the Moments Condition Barrier: No-Regret Algorithm for Bandits with Super Heavy-Tailed Payoffs](http://arxiv.org/abs/2110.13876)


  Despite a large amount of effort in dealing with heavy-tailed error in
machine learning, little is known when moments of the error can become
non-existential: the random noise $\eta$ satisfies Pr$\left[|\eta| > |y|\right]
\le 1/|y|^{\alpha}$ for some $\alpha > 0$. We make the first attempt to
actively handle such super heavy-tailed noise in bandit learning problems: We
propose a novel robust statistical estimator, mean of medians, which estimates
a random variable by computing the empirical mean of a sequence of empirical
medians. We then present a generic reductionist algorithmic framework for
solving bandit learning problems (including multi-armed and linear bandit
problem): the mean of medians estimator can be applied to nearly any bandit
learning algorithm as a black-box filtering for its reward signals and obtain
similar regret bound as if the reward is sub-Gaussian. We show that the regret
bound is near-optimal even with very heavy-tailed noise. We also empirically
demonstrate the effectiveness of the proposed algorithm, which further
corroborates our theoretical results.

    

### [[2110.13878] Deep Explicit Duration Switching Models for Time Series](http://arxiv.org/abs/2110.13878)


  Many complex time series can be effectively subdivided into distinct regimes
that exhibit persistent dynamics. Discovering the switching behavior and the
statistical patterns in these regimes is important for understanding the
underlying dynamical system. We propose the Recurrent Explicit Duration
Switching Dynamical System (RED-SDS), a flexible model that is capable of
identifying both state- and time-dependent switching dynamics. State-dependent
switching is enabled by a recurrent state-to-switch connection and an explicit
duration count variable is used to improve the time-dependent switching
behavior. We demonstrate how to perform efficient inference using a hybrid
algorithm that approximates the posterior of the continuous states via an
inference network and performs exact inference for the discrete switches and
counts. The model is trained by maximizing a Monte Carlo lower bound of the
marginal log-likelihood that can be computed efficiently as a byproduct of the
inference routine. Empirical results on multiple datasets demonstrate that
RED-SDS achieves considerable improvement in time series segmentation and
competitive forecasting performance against the state of the art.

    

### [[2110.13880] Understanding Interlocking Dynamics of Cooperative Rationalization](http://arxiv.org/abs/2110.13880)


  Selective rationalization explains the prediction of complex neural networks
by finding a small subset of the input that is sufficient to predict the neural
model output. The selection mechanism is commonly integrated into the model
itself by specifying a two-component cascaded system consisting of a rationale
generator, which makes a binary selection of the input features (which is the
rationale), and a predictor, which predicts the output based only on the
selected features. The components are trained jointly to optimize prediction
performance. In this paper, we reveal a major problem with such cooperative
rationalization paradigm -- model interlocking. Interlocking arises when the
predictor overfits to the features selected by the generator thus reinforcing
the generator's selection even if the selected rationales are sub-optimal. The
fundamental cause of the interlocking problem is that the rationalization
objective to be minimized is concave with respect to the generator's selection
policy. We propose a new rationalization framework, called A2R, which
introduces a third component into the architecture, a predictor driven by soft
attention as opposed to selection. The generator now realizes both soft and
hard attention over the features and these are fed into the two different
predictors. While the generator still seeks to support the original predictor
performance, it also minimizes a gap between the two predictors. As we will
show theoretically, since the attention-based predictor exhibits a better
convexity property, A2R can overcome the concavity barrier. Our experiments on
two synthetic benchmarks and two real datasets demonstrate that A2R can
significantly alleviate the interlock problem and find explanations that better
align with human judgments. We release our code at
this https URL.

    

### [[2110.13889] Heterogeneous Temporal Graph Neural Network](http://arxiv.org/abs/2110.13889)


  Graph neural networks (GNNs) have been broadly studied on dynamic graphs for
their representation learning, majority of which focus on graphs with
homogeneous structures in the spatial domain. However, many real-world graphs -
i.e., heterogeneous temporal graphs (HTGs) - evolve dynamically in the context
of heterogeneous graph structures. The dynamics associated with heterogeneity
have posed new challenges for HTG representation learning. To solve this
problem, in this paper, we propose heterogeneous temporal graph neural network
(HTGNN) to integrate both spatial and temporal dependencies while preserving
the heterogeneity to learn node representations over HTGs. Specifically, in
each layer of HTGNN, we propose a hierarchical aggregation mechanism, including
intra-relation, inter-relation, and across-time aggregations, to jointly model
heterogeneous spatial dependencies and temporal dimensions. To retain the
heterogeneity, intra-relation aggregation is first performed over each slice of
HTG to attentively aggregate information of neighbors with the same type of
relation, and then intra-relation aggregation is exploited to gather
information over different types of relations; to handle temporal dependencies,
across-time aggregation is conducted to exchange information across different
graph slices over the HTG. The proposed HTGNN is a holistic framework tailored
heterogeneity with evolution in time and space for HTG representation learning.
Extensive experiments are conducted on the HTGs built from different real-world
datasets and promising results demonstrate the outstanding performance of HTGNN
by comparison with state-of-the-art baselines. Our built HTGs and code have
been made publicly accessible at: this https URL.

    

### [[2110.13891] Dynamic Causal Bayesian Optimization](http://arxiv.org/abs/2110.13891)


  This paper studies the problem of performing a sequence of optimal
interventions in a causal dynamical system where both the target variable of
interest and the inputs evolve over time. This problem arises in a variety of
domains e.g. system biology and operational research. Dynamic Causal Bayesian
Optimization (DCBO) brings together ideas from sequential decision making,
causal inference and Gaussian process (GP) emulation. DCBO is useful in
scenarios where all causal effects in a graph are changing over time. At every
time step DCBO identifies a local optimal intervention by integrating both
observational and past interventional data collected from the system. We give
theoretical results detailing how one can transfer interventional information
across time steps and define a dynamic causal GP model which can be used to
quantify uncertainty and find optimal interventions in practice. We demonstrate
how DCBO identifies optimal interventions faster than competing approaches in
multiple settings and applications.

    

### [[2110.13905] Gradient Descent on Two-layer Nets: Margin Maximization and Simplicity Bias](http://arxiv.org/abs/2110.13905)


  The generalization mystery of overparametrized deep nets has motivated
efforts to understand how gradient descent (GD) converges to low-loss solutions
that generalize well. Real-life neural networks are initialized from small
random values and trained with cross-entropy loss for classification (unlike
the "lazy" or "NTK" regime of training where analysis was more successful), and
a recent sequence of results (Lyu and Li, 2020; Chizat and Bach, 2020; Ji and
Telgarsky, 2020) provide theoretical evidence that GD may converge to the
"max-margin" solution with zero loss, which presumably generalizes well.
However, the global optimality of margin is proved only in some settings where
neural nets are infinitely or exponentially wide. The current paper is able to
establish this global optimality for two-layer Leaky ReLU nets trained with
gradient flow on linearly separable and symmetric data, regardless of the
width. The analysis also gives some theoretical justification for recent
empirical findings (Kalimeris et al., 2019) on the so-called simplicity bias of
GD towards linear or other "simple" classes of solutions, especially early in
training. On the pessimistic side, the paper suggests that such results are
fragile. A simple data manipulation can make gradient flow converge to a linear
classifier with suboptimal margin.

    

### [[1805.08079] Faster Neural Network Training with Approximate Tensor Operations](http://arxiv.org/abs/1805.08079)


  We propose a novel technique for faster deep neural network training which
systematically applies sample-based approximation to the constituent tensor
operations, i.e., matrix multiplications and convolutions. We introduce new
sampling techniques, study their theoretical properties, and prove that they
provide the same convergence guarantees when applied to SGD training. We apply
approximate tensor operations to single and multi-node training of MLP and CNN
networks on MNIST, CIFAR-10 and ImageNet datasets. We demonstrate up to 66%
reduction in the amount of computations and communication, and up to 1.37x
faster training time while maintaining negligible or no impact on the final
test accuracy.

    

### [[1806.01380] A General Framework for Bandit Problems Beyond Cumulative Objectives](http://arxiv.org/abs/1806.01380)


  The stochastic multi-armed bandit (MAB) problem is a common model for
sequential decision problems. In the standard setup, a decision maker has to
choose at every instant between several competing arms, each of them provides a
scalar random variable, referred to as a "reward." Nearly all research on this
topic considers the total cumulative reward as the criterion of interest. This
work focuses on other natural objectives that cannot be cast as a sum over
rewards, but rather more involved functions of the reward stream. Unlike the
case of cumulative criteria, in the problems we study here the oracle policy,
that knows the problem parameters a priori and is used to "center" the regret,
is not trivial. We provide a systematic approach to such problems, and derive
general conditions under which the oracle policy is sufficiently tractable to
facilitate the design of optimism-based (upper confidence bound) learning
policies. These conditions elucidate an interesting interplay between the arm
reward distributions and the performance metric. Our main findings are
illustrated for several commonly used objectives such as conditional
value-at-risk, mean-variance trade-offs, Sharpe-ratio, and more.

    

### [[1903.07138] A Brain-inspired Algorithm for Training Highly Sparse Neural Networks](http://arxiv.org/abs/1903.07138)


  Sparse neural networks attract increasing interest as they exhibit comparable
performance to their dense counterparts while being computationally efficient.
Pruning the dense neural networks is among the most widely used methods to
obtain a sparse neural network. Driven by the high training cost of such
methods that can be unaffordable for a low-resource device, training sparse
neural networks sparsely from scratch has recently gained attention. However,
existing sparse training algorithms suffer from various issues, including poor
performance in high sparsity scenarios, computing dense gradient information
during training, or pure random topology search. In this paper, inspired by the
evolution of the biological brain and the Hebbian learning theory, we present a
new sparse training approach that evolves sparse neural networks according to
the behavior of neurons in the network. Concretely, by exploiting the cosine
similarity metric to measure the importance of the connections, our proposed
method, Cosine similarity-based and Random Topology Exploration (CTRE), evolves
the topology of sparse neural networks by adding the most important connections
to the network without calculating dense gradient in the backward. We carried
out different experiments on eight datasets, including tabular, image, and text
datasets, and demonstrate that our proposed method outperforms several
state-of-the-art sparse training algorithms in extremely sparse neural networks
by a large gap. The implementation code is available on
this https URL


### [[1904.05254] Attraction-Repulsion clustering with applications to fairness](http://arxiv.org/abs/1904.05254)


  We consider the problem of diversity enhancing clustering, i.e, developing
clustering methods which produce clusters that favour diversity with respect to
a set of protected attributes such as race, sex, age, etc. In the context of
fair clustering, diversity plays a major role when fairness is understood as
demographic parity. To promote diversity, we introduce perturbations to the
distance in the unprotected attributes that account for protected attributes in
a way that resembles attraction-repulsion of charged particles in Physics.
These perturbations are defined through dissimilarities with a tractable
interpretation. Cluster analysis based on attraction-repulsion dissimilarities
penalizes homogeneity of the clusters with respect to the protected attributes
and leads to an improvement in diversity. An advantage of our approach, which
falls into a pre-processing set-up, is its compatibility with a wide variety of
clustering methods and whit non-Euclidean data. We illustrate the use of our
procedures with both synthetic and real data and provide discussion about the
relation between diversity, fairness, and cluster structure. Our procedures are
implemented in an R package freely available at
this https URL.

    

### [[1906.08635] Energy Models for Better Pseudo-Labels: Improving Semi-Supervised Classification with the 1-Laplacian Graph Energy](http://arxiv.org/abs/1906.08635)


  Semi-supervised classification is a great focus of interest, as in real-world
scenarios obtaining labels is expensive, time-consuming and might require
expert knowledge. This has motivated the fast development of semi-supervised
techniques, whose performance is on a par with or better than supervised
approaches. A current major challenge for semi-supervised techniques is how to
better handle the network calibration and confirmation bias problems for
improving performance. In this work, we argue that energy models are an
effective alternative to such problems. With this motivation in mind, we
propose a hybrid framework for semi-supervised classification called CREPE
model (1-Lapla$\mathbf{C}$ian g$\mathbf{R}$aph $\mathbf{E}$nergy for
$\mathbf{P}$seudo-lab$\mathbf{E}$ls). Firstly, we introduce a new energy model
based on the non-smooth $\ell_1$ norm of the normalised graph 1-Laplacian. Our
functional enforces a sufficiently smooth solution and strengthens the
intrinsic relation between the labelled and unlabelled data. Secondly, we
provide a theoretical analysis for our proposed scheme and show that the
solution trajectory does converge to a non-constant steady point. Thirdly, we
derive the connection of our energy model for pseudo-labelling. We show that
our energy model produces more meaningful pseudo-labels than the ones generated
directly by a deep network. We extensively evaluate our framework, through
numerical and visual experiments, using six benchmarking datasets for natural
and medical images. We demonstrate that our technique reports state-of-the-art
results for semi-supervised classification.

    

### [[1908.03265] On the Variance of the Adaptive Learning Rate and Beyond](http://arxiv.org/abs/1908.03265)


  The learning rate warmup heuristic achieves remarkable success in stabilizing
training, accelerating convergence and improving generalization for adaptive
stochastic optimization algorithms like RMSprop and Adam. Here, we study its
mechanism in details. Pursuing the theory behind warmup, we identify a problem
of the adaptive learning rate (i.e., it has problematically large variance in
the early stage), suggest warmup works as a variance reduction technique, and
provide both empirical and theoretical evidence to verify our hypothesis. We
further propose RAdam, a new variant of Adam, by introducing a term to rectify
the variance of the adaptive learning rate. Extensive experimental results on
image classification, language modeling, and neural machine translation verify
our intuition and demonstrate the effectiveness and robustness of our proposed
method. All implementations are available at:
this https URL.

    

### [[1909.13035] Bridging Explicit and Implicit Deep Generative Models via Neural Stein Estimators](http://arxiv.org/abs/1909.13035)


  There are two types of deep generative models: explicit and implicit. The
former defines an explicit density form that allows likelihood inference; while
the latter targets a flexible transformation from random noise to generated
samples. While the two classes of generative models have shown great power in
many applications, both of them, when used alone, suffer from respective
limitations and drawbacks. To take full advantages of both models and enable
mutual compensation, we propose a novel joint training framework that bridges
an explicit (unnormalized) density estimator and an implicit sample generator
via Stein discrepancy. We show that our method 1) induces novel mutual
regularization via kernel Sobolev norm penalization and Moreau-Yosida
regularization, and 2) stabilizes the training dynamics. Empirically, we
demonstrate that proposed method can facilitate the density estimator to more
accurately identify data modes and guide the generator to output higher-quality
samples, comparing with training a single counterpart. The new approach also
shows promising results when the training samples are contaminated or limited.

    

### [[1911.04250] How to GENERALize Across Many Software Projects? (with case studies on Predicting Defect and Project Health)](http://arxiv.org/abs/1911.04250)


  Despite decades of research, SE lacks widely accepted models (that offer
precise quantitative predictions) about what factors most influence software
quality. This paper provides a "good news" result that such general models can
be generated using a new transfer learning framework called "GENERAL". Given a
tree of recursively clustered projects (using project meta-data), GENERAL
promotes a model upwards if it performs best in the lower clusters (stopping
when the promoted model performs worse than the models seen at a lower level).
The number of models found by GENERAL is minimal: one for defect prediction
(756 projects) and less than a dozen for project health (1628 projects). Hence,
via GENERAL, it is possible to make conclusions that hold across hundreds of
projects at a time. Further, the models produced in this manner offer
predictions that perform as well or better than prior state-of-the-art.
To the best of our knowledge, this is the largest demonstration of the
generalizability of quantitative predictions of project quality yet reported in
the SE literature.

    

### [[2002.00253] Bandits with Knapsacks beyond the Worst-Case](http://arxiv.org/abs/2002.00253)


  Bandits with Knapsacks (BwK) is a general model for multi-armed bandits under
supply/budget constraints. While worst-case regret bounds for BwK are
well-understood, we present three results that go beyond the worst-case
perspective. First, we provide upper and lower bounds which amount to a full
characterization for logarithmic, instance-dependent regret rates. Second, we
consider "simple regret" in BwK, which tracks algorithm's performance in a
given round, and prove that it is small in all but a few rounds. Third, we
provide a general "reduction" from BwK to bandits which takes advantage of some
known helpful structure, and apply this reduction to combinatorial
semi-bandits, linear contextual bandits, and multinomial-logit bandits. Our
results build on the BwK algorithm from \citet{AgrawalDevanur-ec14}, providing
new analyses thereof.

    

### [[2002.04758] Salvaging Federated Learning by Local Adaptation](http://arxiv.org/abs/2002.04758)


  Federated learning (FL) is a heavily promoted approach for training ML models
on sensitive data, e.g., text typed by users on their smartphones. FL is
expressly designed for training on data that are unbalanced and non-iid across
the participants. To ensure privacy and integrity of the fedeated model, latest
FL approaches use differential privacy or robust aggregation.
We look at FL from the \emph{local} viewpoint of an individual participant
and ask: (1) do participants have an incentive to participate in FL? (2) how
can participants \emph{individually} improve the quality of their local models,
without re-designing the FL framework and/or involving other participants?
First, we show that on standard tasks such as next-word prediction, many
participants gain no benefit from FL because the federated model is less
accurate on their data than the models they can train locally on their own.
Second, we show that differential privacy and robust aggregation make this
problem worse by further destroying the accuracy of the federated model for
many participants.
Then, we evaluate three techniques for local adaptation of federated models:
fine-tuning, multi-task learning, and knowledge distillation. We analyze where
each is applicable and demonstrate that all participants benefit from local
adaptation. Participants whose local models are poor obtain big accuracy
improvements over conventional FL. Participants whose local models are better
than the federated model\textemdash and who have no incentive to participate in
FL today\textemdash improve less, but sufficiently to make the adapted
federated model better than their local models.

    

### [[2002.05308] Efficient Adaptive Experimental Design for Average Treatment Effect Estimation](http://arxiv.org/abs/2002.05308)


  The goal of many scientific experiments including A/B testing is to estimate
the average treatment effect (ATE), which is defined as the difference between
the expected outcomes of two or more treatments. In this paper, we consider a
situation where an experimenter can assign a treatment to research subjects
sequentially. In adaptive experimental design, the experimenter is allowed to
change the probability of assigning a treatment using past observations for
estimating the ATE efficiently. However, with this approach, it is difficult to
apply a standard statistical method to construct an estimator because the
observations are not independent and identically distributed. We thus propose
an algorithm for efficient experiments with estimators constructed from
dependent samples. We also introduce a sequential testing framework using the
proposed estimator. To justify our proposed approach, we provide finite and
infinite sample analyses. Finally, we experimentally show that the proposed
algorithm exhibits preferable performance.

    

### [[2003.08907] Overinterpretation reveals image classification model pathologies](http://arxiv.org/abs/2003.08907)


  Image classifiers are typically scored on their test set accuracy, but high
accuracy can mask a subtle type of model failure. We find that high scoring
convolutional neural networks (CNNs) on popular benchmarks exhibit troubling
pathologies that allow them to display high accuracy even in the absence of
semantically salient features. When a model provides a high-confidence decision
without salient supporting input features, we say the classifier has
overinterpreted its input, finding too much class-evidence in patterns that
appear nonsensical to humans. Here, we demonstrate that neural networks trained
on CIFAR-10 and ImageNet suffer from overinterpretation, and we find models on
CIFAR-10 make confident predictions even when 95% of input images are masked
and humans cannot discern salient features in the remaining pixel-subsets. We
introduce Batched Gradient SIS, a new method for discovering sufficient input
subsets for complex datasets, and use this method to show the sufficiency of
border pixels in ImageNet for training and testing. Although these patterns
portend potential model fragility in real-world deployment, they are in fact
valid statistical patterns of the benchmark that alone suffice to attain high
test accuracy. Unlike adversarial examples, overinterpretation relies upon
unmodified image pixels. We find ensembling and input dropout can each help
mitigate overinterpretation.

    

### [[2006.02894] Secure Sum Outperforms Homomorphic Encryption in (Current) Collaborative Deep Learning](http://arxiv.org/abs/2006.02894)


  Deep learning (DL) approaches are achieving extraordinary results in a wide
range of domains, but often require a massive collection of private data.
Hence, methods for training neural networks on the joint data of different data
owners, that keep each party's input confidential, are called for. We address a
specific setting in federated learning, namely that of deep learning from
horizontally distributed data with a limited number of parties, where their
vulnerable intermediate results have to be processed in a privacy-preserving
manner. This setting can be found in medical and healthcare as well as
industrial applications. The predominant scheme for this is based on
homomorphic encryption (HE), and it is widely considered to be without
alternative. In contrast to this, we demonstrate that a carefully chosen, less
complex and computationally less expensive secure sum protocol in conjunction
with default secure channels exhibits superior properties in terms of both
collusion-resistance and runtime. Finally, we discuss several open research
questions in the context of collaborative DL, especially regarding privacy
risks caused by joint intermediate results.

    

### [[2008.09569] Revisiting Process versus Product Metrics: a Large Scale Analysis](http://arxiv.org/abs/2008.09569)


  Numerous methods can build predictive models from software data. However,
what methods and conclusions should we endorse as we move from analytics
in-the-small (dealing with a handful of projects) to analytics in-the-large
(dealing with hundreds of projects)?
To answer this question, we recheck prior small-scale results (about process
versus product metrics for defect prediction and the granularity of metrics)
using 722,471 commits from 700 Github projects. We find that some analytics
in-the-small conclusions still hold when scaling up to analytics in-the-large.
For example, like prior work, we see that process metrics are better predictors
for defects than product metrics (best process/product-based learners
respectively achieve recalls of 98\%/44\% and AUCs of 95\%/54\%, median
values).
That said, we warn that it is unwise to trust metric importance results from
analytics in-the-small studies since those change dramatically when moving to
analytics in-the-large. Also, when reasoning in-the-large about hundreds of
projects, it is better to use predictions from multiple models (since single
model predictions can become confused and exhibit a high variance).

    

### [[2009.05204] Transfer Learning of Graph Neural Networks with Ego-graph Information Maximization](http://arxiv.org/abs/2009.05204)


  Graph neural networks (GNNs) have achieved superior performance in various
applications, but training dedicated GNNs can be costly for large-scale graphs.
Some recent work started to study the pre-training of GNNs. However, none of
them provide theoretical insights into the design of their frameworks, or clear
requirements and guarantees towards their transferability. In this work, we
establish a theoretically grounded and practically useful framework for the
transfer learning of GNNs. Firstly, we propose a novel view towards the
essential graph information and advocate the capturing of it as the goal of
transferable GNN training, which motivates the design of EGI (Ego-Graph
Information maximization) to analytically achieve this goal. Secondly, when
node features are structure-relevant, we conduct an analysis of EGI
transferability regarding the difference between the local graph Laplacians of
the source and target graphs. We conduct controlled synthetic experiments to
directly justify our theoretical conclusions. Comprehensive experiments on two
real-world network datasets show consistent results in the analyzed setting of
direct-transfering, while those on large-scale knowledge graphs show promising
results in the more practical setting of transfering with fine-tuning.

    

### [[2009.09026] Adversarial Robustness through Bias Variance Decomposition: A New Perspective for Federated Learning](http://arxiv.org/abs/2009.09026)


  Federated learning learns a neural network model by aggregating the knowledge
from a group of distributed clients under the privacy-preserving constraint. In
this work, we show that this paradigm might inherit the adversarial
vulnerability of the centralized neural network, i.e., it has deteriorated
performance on adversarial examples when the model is deployed. This is even
more alarming when federated learning paradigm is designed to approximate the
updating behavior of a centralized neural network. To solve this problem, we
propose an adversarially robust federated learning framework, named Fed_BVA,
with improved server and client update mechanisms. This is motivated by our
observation that the generalization error in federated learning can be
naturally decomposed into the bias and variance triggered by multiple clients'
predictions. Thus, we propose to generate the adversarial examples via
maximizing the bias and variance during server update, and learn the
adversarially robust model updates with those examples during client update. As
a result, an adversarially robust neural network can be aggregated from these
improved local clients' model updates. The experiments are conducted on
multiple benchmark data sets using several prevalent neural network models, and
the empirical results show that our framework is robust against white-box and
black-box adversarial corruptions under both IID and non-IID settings.

    

### [[2009.13961] Online Action Learning in High Dimensions: A Conservative Perspective](http://arxiv.org/abs/2009.13961)


  Sequential learning problems are common in several fields of research and
practical applications. Examples include dynamic pricing and assortment, design
of auctions and incentives and permeate a large number of sequential treatment
experiments. In this paper, we extend one of the most popular learning
solutions, the $\epsilon_t$-greedy heuristics, to high-dimensional contexts
considering a conservative directive. We do this by allocating part of the time
the original rule uses to adopt completely new actions to a more focused search
in a restrictive set of promising actions. The resulting rule might be useful
for practical applications that still values surprises, although at a
decreasing rate, while also has restrictions on the adoption of unusual
actions. With high probability, we find reasonable bounds for the cumulative
regret of a conservative high-dimensional decaying $\epsilon_t$-greedy rule.
Also, we provide a lower bound for the cardinality of the set of viable actions
that implies in an improved regret bound for the conservative version when
compared to its non-conservative counterpart. Additionally, we show that
end-users have sufficient flexibility when establishing how much safety they
want, since it can be tuned without impacting theoretical properties. We
illustrate our proposal both in a simulation exercise and using a real dataset.

    

### [[2009.14471] PettingZoo: Gym for Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2009.14471)


  This paper introduces the PettingZoo library and the accompanying Agent
Environment Cycle ("AEC") games model. PettingZoo is a library of diverse sets
of multi-agent environments with a universal, elegant Python API. PettingZoo
was developed with the goal of accelerating research in Multi-Agent
Reinforcement Learning ("MARL"), by making work more interchangeable,
accessible and reproducible akin to what OpenAI's Gym library did for
single-agent reinforcement learning. PettingZoo's API, while inheriting many
features of Gym, is unique amongst MARL APIs in that it's based around the
novel AEC games model. We argue, in part through case studies on major problems
in popular MARL environments, that the popular game models are poor conceptual
models of games commonly used in MARL and accordingly can promote confusing
bugs that are hard to detect, and that the AEC games model addresses these
problems.

    

### [[2010.08281] Embedding and Extraction of Knowledge in Tree Ensemble Classifiers](http://arxiv.org/abs/2010.08281)


  The embedding and extraction of useful knowledge is a recent trend in machine
learning applications, e.g., to supplement existing datasets that are small.
Whilst, as the increasing use of machine learning models in security-critical
applications, the embedding and extraction of malicious knowledge are
equivalent to the notorious backdoor attack and its defence, respectively. This
paper studies the embedding and extraction of knowledge in tree ensemble
classifiers, and focuses on knowledge expressible with a generic form of
Boolean formulas, e.g., robustness properties and backdoor attacks. For the
embedding, it is required to be preservative(the original performance of the
classifier is preserved), verifiable(the knowledge can be attested), and
stealthy(the embedding cannot be easily detected). To facilitate this, we
propose two novel, and effective, embedding algorithms, one of which is for
black-box settings and the other for white-box settings.The embedding can be
done in PTIME. Beyond the embedding, we develop an algorithm to extract the
embedded knowledge, by reducing the problem to be solvable with an SMT
(satisfiability modulo theories) solver. While this novel algorithm can
successfully extract knowledge, the reduction leads to an NP computation.
Therefore, if applying embedding as backdoor attacks and extraction as defence,
our results suggest a complexity gap (P vs. NP) between the attack and defence
when working with tree ensemble classifiers. We apply our algorithms toa
diverse set of datasets to validate our conclusion extensively.

    

### [[2010.09541] On the Difficulty of Unbiased Alpha Divergence Minimization](http://arxiv.org/abs/2010.09541)


  Several approximate inference algorithms have been proposed to minimize an
alpha-divergence between an approximating distribution and a target
distribution. Many of these algorithms introduce bias, the magnitude of which
becomes problematic in high dimensions. Other algorithms are unbiased. These
often seem to suffer from high variance, but little is rigorously known. In
this work we study unbiased methods for alpha-divergence minimization through
the Signal-to-Noise Ratio (SNR) of the gradient estimator. We study several
representative scenarios where strong analytical results are possible, such as
fully-factorized or Gaussian distributions. We find that when alpha is not
zero, the SNR worsens exponentially in the dimensionality of the problem. This
casts doubt on the practicality of these methods. We empirically confirm these
theoretical results.

    

### [[2010.16358] AgEBO-Tabular: Joint Neural Architecture and Hyperparameter Search with Autotuned Data-Parallel Training for Tabular Data](http://arxiv.org/abs/2010.16358)


  Developing high-performing predictive models for large tabular data sets is a
challenging task. The state-of-the-art methods are based on expert-developed
model ensembles from different supervised learning methods. Recently, automated
machine learning (AutoML) is emerging as a promising approach to automate
predictive model development. Neural architecture search (NAS) is an AutoML
approach that generates and evaluates multiple neural network architectures
concurrently and improves the accuracy of the generated models iteratively. A
key issue in NAS, particularly for large data sets, is the large computation
time required to evaluate each generated architecture. While data-parallel
training is a promising approach that can address this issue, its use within
NAS is difficult. For different data sets, the data-parallel training settings
such as the number of parallel processes, learning rate, and batch size need to
be adapted to achieve high accuracy and reduction in training time. To that
end, we have developed AgEBO-Tabular, an approach to combine aging evolution
(AgE), a parallel NAS method that searches over neural architecture space, and
an asynchronous Bayesian optimization method for tuning the hyperparameters of
the data-parallel training simultaneously. We demonstrate the efficacy of the
proposed method to generate high-performing neural network models for large
tabular benchmark data sets. Furthermore, we demonstrate that the automatically
discovered neural network models using our method outperform the
state-of-the-art AutoML ensemble models in inference speed by two orders of
magnitude while reaching similar accuracy values.

    

### [[2011.09468] Gradient Starvation: A Learning Proclivity in Neural Networks](http://arxiv.org/abs/2011.09468)


  We identify and formalize a fundamental gradient descent phenomenon resulting
in a learning proclivity in over-parameterized neural networks. Gradient
Starvation arises when cross-entropy loss is minimized by capturing only a
subset of features relevant for the task, despite the presence of other
predictive features that fail to be discovered. This work provides a
theoretical explanation for the emergence of such feature imbalance in neural
networks. Using tools from Dynamical Systems theory, we identify simple
properties of learning dynamics during gradient descent that lead to this
imbalance, and prove that such a situation can be expected given certain
statistical structure in training data. Based on our proposed formalism, we
develop guarantees for a novel regularization method aimed at decoupling
feature learning dynamics, improving accuracy and robustness in cases hindered
by gradient starvation. We illustrate our findings with simple and real-world
out-of-distribution (OOD) generalization experiments.

    

### [[2011.14164] Towards Robust Partially Supervised Multi-Structure Medical Image Segmentation on Small-Scale Data](http://arxiv.org/abs/2011.14164)


  The data-driven nature of deep learning (DL) models for semantic segmentation
requires a large number of pixel-level annotations. However, large-scale and
fully labeled medical datasets are often unavailable for practical tasks.
Recently, partially supervised methods have been proposed to utilize images
with incomplete labels in the medical domain. To bridge the methodological gaps
in partially supervised learning (PSL) under data scarcity, we propose Vicinal
Labels Under Uncertainty (VLUU), a simple yet efficient framework utilizing the
human structure similarity for partially supervised medical image segmentation.
Motivated by multi-task learning and vicinal risk minimization, VLUU transforms
the partially supervised problem into a fully supervised problem by generating
vicinal labels. We systematically evaluate VLUU under the challenges of
small-scale data, dataset shift, and class imbalance on two commonly used
segmentation datasets for the tasks of chest organ segmentation and optic
disc-and-cup segmentation. The experimental results show that VLUU can
consistently outperform previous partially supervised models in these settings.
Our research suggests a new research direction in label-efficient deep learning
with partial supervision.

    

### [[2012.00901] Deep Multi-Fidelity Active Learning of High-dimensional Outputs](http://arxiv.org/abs/2012.00901)


  Many applications, such as in physical simulation and engineering design,
demand we estimate functions with high-dimensional outputs. The training
examples can be collected with different fidelities to allow a cost/accuracy
trade-off. In this paper, we consider the active learning task that identifies
both the fidelity and input to query new training examples so as to achieve the
best benefit-cost ratio. To this end, we propose DMFAL, a Deep Multi-Fidelity
Active Learning approach. We first develop a deep neural network-based
multi-fidelity model for learning with high-dimensional outputs, which can
flexibly, efficiently capture all kinds of complex relationships across the
outputs and fidelities to improve prediction. We then propose a mutual
information-based acquisition function that extends the predictive entropy
principle. To overcome the computational challenges caused by large output
dimensions, we use multi-variate Delta's method and moment-matching to estimate
the output posterior, and Weinstein-Aronszajn identity to calculate and
optimize the acquisition function. The computation is tractable, reliable and
efficient. We show the advantage of our method in several applications of
computational physics and engineering design.

    

### [[2101.00300] When Is Generalizable Reinforcement Learning Tractable?](http://arxiv.org/abs/2101.00300)


  Agents trained by reinforcement learning (RL) often fail to generalize beyond
the environment they were trained in, even when presented with new scenarios
that seem similar to the training environment. We study the query complexity
required to train RL agents that generalize to multiple environments.
Intuitively, tractable generalization is only possible when the environments
are similar or close in some sense. To capture this, we introduce Weak
Proximity, a natural structural condition that requires the environments to
have highly similar transition and reward functions and share a policy
providing optimal value. Despite such shared structure, we prove that tractable
generalization is impossible in the worst case. This holds even when each
individual environment can be efficiently solved to obtain an optimal linear
policy, and when the agent possesses a generative model. Our lower bound
applies to the more complex task of representation learning for the purpose of
efficient generalization to multiple environments. On the positive side, we
introduce Strong Proximity, a strengthened condition which we prove is
sufficient for efficient generalization.

    

### [[2101.06203] Reviving Purpose Limitation and Data Minimisation in Data-Driven Systems](http://arxiv.org/abs/2101.06203)


  This paper determines whether the two core data protection principles of data
minimisation and purpose limitation can be meaningfully implemented in
data-driven systems. While contemporary data processing practices appear to
stand at odds with these principles, we demonstrate that systems could
technically use much less data than they currently do. This observation is a
starting point for our detailed techno-legal analysis uncovering obstacles that
stand in the way of meaningful implementation and compliance as well as
exemplifying unexpected trade-offs which emerge where data protection law is
applied in practice. Our analysis seeks to inform debates about the impact of
data protection on the development of artificial intelligence in the European
Union, offering practical action points for data controllers, regulators, and
researchers.

    

### [[2101.09752] AQuA: Analytical Quality Assessment for Optimizing Video Analytics Systems](http://arxiv.org/abs/2101.09752)


  Millions of cameras at edge are being deployed to power a variety of
different deep learning applications. However, the frames captured by these
cameras are not always pristine - they can be distorted due to lighting issues,
sensor noise, compression etc. Such distortions not only deteriorate visual
quality, they impact the accuracy of deep learning applications that process
such video streams. In this work, we introduce AQuA, to protect application
accuracy against such distorted frames by scoring the level of distortion in
the frames. It takes into account the analytical quality of frames, not the
visual quality, by learning a novel metric, classifier opinion score, and uses
a lightweight, CNN-based, object-independent feature extractor. AQuA accurately
scores distortion levels of frames and generalizes to multiple different deep
learning applications. When used for filtering poor quality frames at edge, it
reduces high-confidence errors for analytics applications by 17%. Through
filtering, and due to its low overhead (14ms), AQuA can also reduce computation
time and average bandwidth usage by 25%.

    

### [[2102.03034] Hyperparameter Optimization Is Deceiving Us, and How to Stop It](http://arxiv.org/abs/2102.03034)


  Recent empirical work shows that inconsistent results based on choice of
hyperparameter optimization (HPO) configuration are a widespread problem in ML
research. When comparing two algorithms J and K searching one subspace can
yield the conclusion that J outperforms K, whereas searching another can entail
the opposite. In short, the way we choose hyperparameters can deceive us. We
provide a theoretical complement to this prior work, arguing that, to avoid
such deception, the process of drawing conclusions from HPO should be made more
rigorous. We call this process epistemic hyperparameter optimization (EHPO),
and put forth a logical framework to capture its semantics and how it can lead
to inconsistent conclusions about performance. Our framework enables us to
prove EHPO methods that are guaranteed to be defended against deception, given
bounded compute time budget t. We demonstrate our framework's utility by
proving and empirically validating a defended variant of random search.

    

### [[2102.03324] GIBBON: General-purpose Information-Based Bayesian OptimisatioN](http://arxiv.org/abs/2102.03324)


  This paper describes a general-purpose extension of max-value entropy search,
a popular approach for Bayesian Optimisation (BO). A novel approximation is
proposed for the information gain -- an information-theoretic quantity central
to solving a range of BO problems, including noisy, multi-fidelity and batch
optimisations across both continuous and highly-structured discrete spaces.
Previously, these problems have been tackled separately within
information-theoretic BO, each requiring a different sophisticated
approximation scheme, except for batch BO, for which no
computationally-lightweight information-theoretic approach has previously been
proposed. GIBBON (General-purpose Information-Based Bayesian OptimisatioN)
provides a single principled framework suitable for all the above,
out-performing existing approaches whilst incurring substantially lower
computational overheads. In addition, GIBBON does not require the problem's
search space to be Euclidean and so is the first high-performance yet
computationally light-weight acquisition function that supports batch BO over
general highly structured input spaces like molecular search and gene design.
Moreover, our principled derivation of GIBBON yields a natural interpretation
of a popular batch BO heuristic based on determinantal point processes.
Finally, we analyse GIBBON across a suite of synthetic benchmark tasks, a
molecular search loop, and as part of a challenging batch multi-fidelity
framework for problems with controllable experimental noise.

    

### [[2102.03448] Federated Reconstruction: Partially Local Federated Learning](http://arxiv.org/abs/2102.03448)


  Personalization methods in federated learning aim to balance the benefits of
federated and local training for data availability, communication cost, and
robustness to client heterogeneity. Approaches that require clients to
communicate all model parameters can be undesirable due to privacy and
communication constraints. Other approaches require always-available or
stateful clients, impractical in large-scale cross-device settings. We
introduce Federated Reconstruction, the first model-agnostic framework for
partially local federated learning suitable for training and inference at
scale. We motivate the framework via a connection to model-agnostic meta
learning, empirically demonstrate its performance over existing approaches for
collaborative filtering and next word prediction, and release an open-source
library for evaluating approaches in this setting. We also describe the
successful deployment of this approach at scale for federated collaborative
filtering in a mobile keyboard application.

    

### [[2102.04716] Better Safe Than Sorry: Preventing Delusive Adversaries with Adversarial Training](http://arxiv.org/abs/2102.04716)


  Delusive attacks aim to substantially deteriorate the test accuracy of the
learning model by slightly perturbing the features of correctly labeled
training examples. By formalizing this malicious attack as finding the
worst-case training data within a specific $\infty$-Wasserstein ball, we show
that minimizing adversarial risk on the perturbed data is equivalent to
optimizing an upper bound of natural risk on the original data. This implies
that adversarial training can serve as a principled defense against delusive
attacks. Thus, the test accuracy decreased by delusive attacks can be largely
recovered by adversarial training. To further understand the internal mechanism
of the defense, we disclose that adversarial training can resist the delusive
perturbations by preventing the learner from overly relying on non-robust
features in a natural setting. Finally, we complement our theoretical findings
with a set of experiments on popular benchmark datasets, which show that the
defense withstands six different practical attacks. Both theoretical and
empirical results vote for adversarial training when confronted with delusive
adversaries.

    

### [[2102.05242] Patterns, predictions, and actions: A story about machine learning](http://arxiv.org/abs/2102.05242)


  This graduate textbook on machine learning tells a story of how patterns in
data support predictions and consequential actions. Starting with the
foundations of decision making, we cover representation, optimization, and
generalization as the constituents of supervised learning. A chapter on
datasets as benchmarks examines their histories and scientific bases.
Self-contained introductions to causality, the practice of causal inference,
sequential decision making, and reinforcement learning equip the reader with
concepts and tools to reason about actions and their consequences. Throughout,
the text discusses historical context and societal impact. We invite readers
from all backgrounds; some experience with probability, calculus, and linear
algebra suffices.

    

### [[2102.05762] Risk-Averse Bayes-Adaptive Reinforcement Learning](http://arxiv.org/abs/2102.05762)


  In this work, we address risk-averse Bayes-adaptive reinforcement learning.
We pose the problem of optimising the conditional value at risk (CVaR) of the
total return in Bayes-adaptive Markov decision processes (MDPs). We show that a
policy optimising CVaR in this setting is risk-averse to both the parametric
uncertainty due to the prior distribution over MDPs, and the internal
uncertainty due to the inherent stochasticity of MDPs. We reformulate the
problem as a two-player stochastic game and propose an approximate algorithm
based on Monte Carlo tree search and Bayesian optimisation. Our experiments
demonstrate that our approach significantly outperforms baseline approaches for
this problem.

    

### [[2102.06442] Broad-UNet: Multi-scale feature learning for nowcasting tasks](http://arxiv.org/abs/2102.06442)


  Weather nowcasting consists of predicting meteorological components in the
short term at high spatial resolutions. Due to its influence in many human
activities, accurate nowcasting has recently gained plenty of attention. In
this paper, we treat the nowcasting problem as an image-to-image translation
problem using satellite imagery. We introduce Broad-UNet, a novel architecture
based on the core UNet model, to efficiently address this problem. In
particular, the proposed Broad-UNet is equipped with asymmetric parallel
convolutions as well as Atrous Spatial Pyramid Pooling (ASPP) module. In this
way, The the Broad-UNet model learns more complex patterns by combining
multi-scale features while using fewer parameters than the core UNet model. The
proposed model is applied on two different nowcasting tasks, i.e. precipitation
maps and cloud cover nowcasting. The obtained numerical results show that the
introduced Broad-UNet model performs more accurate predictions compared to the
other examined architectures.

    

### [[2102.06589] Generalization Bounds for Meta-Learning via PAC-Bayes and Uniform Stability](http://arxiv.org/abs/2102.06589)


  We are motivated by the problem of providing strong generalization guarantees
in the context of meta-learning. Existing generalization bounds are either
challenging to evaluate or provide vacuous guarantees in even relatively simple
settings. We derive a probably approximately correct (PAC) bound for
gradient-based meta-learning using two different generalization frameworks in
order to deal with the qualitatively different challenges of generalization at
the "base" and "meta" levels. We employ bounds for uniformly stable algorithms
at the base level and bounds from the PAC-Bayes framework at the meta level.
The result of this approach is a novel PAC bound that is tighter when the base
learner adapts quickly, which is precisely the goal of meta-learning. We show
that our bound provides a tighter guarantee than other bounds on a toy
non-convex problem on the unit sphere and a text-based classification example.
We also present a practical regularization scheme motivated by the bound in
settings where the bound is loose and demonstrate improved performance over
baseline techniques.

    

### [[2102.06604] Cockpit: A Practical Debugging Tool for the Training of Deep Neural Networks](http://arxiv.org/abs/2102.06604)


  When engineers train deep learning models, they are very much 'flying blind'.
Commonly used methods for real-time training diagnostics, such as monitoring
the train/test loss, are limited. Assessing a network's training process solely
through these performance indicators is akin to debugging software without
access to internal states through a debugger. To address this, we present
Cockpit, a collection of instruments that enable a closer look into the inner
workings of a learning machine, and a more informative and meaningful status
report for practitioners. It facilitates the identification of learning phases
and failure modes, like ill-chosen hyperparameters. These instruments leverage
novel higher-order information about the gradient distribution and curvature,
which has only recently become efficiently accessible. We believe that such a
debugging tool, which we open-source for PyTorch, is a valuable help in
troubleshooting the training process. By revealing new insights, it also more
generally contributes to explainability and interpretability of deep nets.

    

### [[2102.06648] A Critical Look at the Consistency of Causal Estimation With Deep Latent Variable Models](http://arxiv.org/abs/2102.06648)


  Using deep latent variable models in causal inference has attracted
considerable interest recently, but an essential open question is their ability
to yield consistent causal estimates. While they have demonstrated promising
results and theory exists on some simple model formulations, we also know that
causal effects are not even identifiable in general with latent variables. We
investigate this gap between theory and empirical results with analytical
considerations and extensive experiments under multiple synthetic and
real-world data sets, using the causal effect variational autoencoder (CEVAE)
as a case study. While CEVAE seems to work reliably under some simple
scenarios, it does not estimate the causal effect correctly with a misspecified
latent variable or a complex data distribution, as opposed to its original
motivation. Hence, our results show that more attention should be paid to
ensuring the correctness of causal estimates with deep latent variable models.

    

### [[2102.07650] Learning Student-Friendly Teacher Networks for Knowledge Distillation](http://arxiv.org/abs/2102.07650)


  We propose a novel knowledge distillation approach to facilitate the transfer
of dark knowledge from a teacher to a student. Contrary to most of the existing
methods that rely on effective training of student models given pretrained
teachers, we aim to learn the teacher models that are friendly to students and,
consequently, more appropriate for knowledge transfer. In other words, at the
time of optimizing a teacher model, the proposed algorithm learns the student
branches jointly to obtain student-friendly representations. Since the main
goal of our approach lies in training teacher models and the subsequent
knowledge distillation procedure is straightforward, most of the existing
knowledge distillation methods can adopt this technique to improve the
performance of diverse student models in terms of accuracy and convergence
speed. The proposed algorithm demonstrates outstanding accuracy in several
well-known knowledge distillation techniques with various combinations of
teacher and student models even in the case that their architectures are
heterogeneous and there is no prior knowledge about student models at the time
of training teacher networks.

    

### [[2102.08604] SWAD: Domain Generalization by Seeking Flat Minima](http://arxiv.org/abs/2102.08604)


  Domain generalization (DG) methods aim to achieve generalizability to an
unseen target domain by using only training data from the source domains.
Although a variety of DG methods have been proposed, a recent study shows that
under a fair evaluation protocol, called DomainBed, the simple empirical risk
minimization (ERM) approach works comparable to or even outperforms previous
methods. Unfortunately, simply solving ERM on a complex, non-convex loss
function can easily lead to sub-optimal generalizability by seeking sharp
minima. In this paper, we theoretically show that finding flat minima results
in a smaller domain generalization gap. We also propose a simple yet effective
method, named Stochastic Weight Averaging Densely (SWAD), to find flat minima.
SWAD finds flatter minima and suffers less from overfitting than does the
vanilla SWA by a dense and overfit-aware stochastic weight sampling strategy.
SWAD shows state-of-the-art performances on five DG benchmarks, namely PACS,
VLCS, OfficeHome, TerraIncognita, and DomainNet, with consistent and large
margins of +1.6% averagely on out-of-domain accuracy. We also compare SWAD with
conventional generalization methods, such as data augmentation and consistency
regularization methods, to verify that the remarkable performance improvements
are originated from by seeking flat minima, not from better in-domain
generalizability. Last but not least, SWAD is readily adaptable to existing DG
methods without modification; the combination of SWAD and an existing DG method
further improves DG performances. Source code is available at
this https URL.

    

### [[2102.10570] Symbolic regression for scientific discovery: an application to wind speed forecasting](http://arxiv.org/abs/2102.10570)


  Symbolic regression corresponds to an ensemble of techniques that allow to
uncover an analytical equation from data. Through a closed form formula, these
techniques provide great advantages such as potential scientific discovery of
new laws, as well as explainability, feature engineering as well as fast
inference. Similarly, deep learning based techniques has shown an extraordinary
ability of modeling complex patterns. The present paper aims at applying a
recent end-to-end symbolic regression technique, i.e. the equation learner
(EQL), to get an analytical equation for wind speed forecasting. We show that
it is possible to derive an analytical equation that can achieve reasonable
accuracy for short term horizons predictions only using few number of features.

    

### [[2102.12002] Adversarial Robustness with Non-uniform Perturbations](http://arxiv.org/abs/2102.12002)


  Robustness of machine learning models is critical for security related
applications, where real-world adversaries are uniquely focused on evading
neural network based detectors. Prior work mainly focus on crafting adversarial
examples (AEs) with small uniform norm-bounded perturbations across features to
maintain the requirement of imperceptibility. However, uniform perturbations do
not result in realistic AEs in domains such as malware, finance, and social
networks. For these types of applications, features typically have some
semantically meaningful dependencies. The key idea of our proposed approach is
to enable non-uniform perturbations that can adequately represent these feature
dependencies during adversarial training. We propose using characteristics of
the empirical data distribution, both on correlations between the features and
the importance of the features themselves. Using experimental datasets for
malware classification, credit risk prediction, and spam detection, we show
that our approach is more robust to real-world attacks. Finally, we present
robustness certification utilizing non-uniform perturbation bounds, and show
that non-uniform bounds achieve better certification.

    

### [[2102.12090] Continuous Mean-Covariance Bandits](http://arxiv.org/abs/2102.12090)


  Existing risk-aware multi-armed bandit models typically focus on risk
measures of individual options such as variance. As a result, they cannot be
directly applied to important real-world online decision making problems with
correlated options. In this paper, we propose a novel Continuous
Mean-Covariance Bandit (CMCB) model to explicitly take into account option
correlation. Specifically, in CMCB, there is a learner who sequentially chooses
weight vectors on given options and observes random feedback according to the
decisions. The agent's objective is to achieve the best trade-off between
reward and risk, measured with option covariance. To capture important reward
observation scenarios in practice, we consider three feedback settings, i.e.,
full-information, semi-bandit and full-bandit feedback. We propose novel
algorithms with the optimal regrets (within logarithmic factors), and provide
matching lower bounds to validate their optimalities. Our experimental results
also demonstrate the superiority of the proposed algorithms. To the best of our
knowledge, this is the first work that considers option correlation in
risk-aware bandits and explicitly quantifies how arbitrary covariance
structures impact the learning performance.

    

### [[2102.12094] Combinatorial Pure Exploration with Bottleneck Reward Function](http://arxiv.org/abs/2102.12094)


  In this paper, we study the Combinatorial Pure Exploration problem with the
Bottleneck reward function (CPE-B) under the fixed-confidence (FC) and
fixed-budget (FB) settings. In CPE-B, given a set of base arms and a collection
of subsets of base arms (super arms) following a certain combinatorial
constraint, a learner sequentially plays a base arm and observes its random
reward, with the objective of finding the optimal super arm with the maximum
bottleneck value, defined as the minimum expected reward of the base arms
contained in the super arm. CPE-B captures a variety of practical scenarios
such as network routing in communication networks, and its \emph{unique
challenges} fall on how to utilize the bottleneck property to save samples and
achieve the statistical optimality. None of the existing CPE studies (most of
them assume linear rewards) can be adapted to solve such challenges, and thus
we develop brand-new techniques to handle them. For the FC setting, we propose
novel algorithms with optimal sample complexity for a broad family of instances
and establish a matching lower bound to demonstrate the optimality (within a
logarithmic factor). For the FB setting, we design an algorithm which achieves
the state-of-the-art error probability guarantee and is the first to run
efficiently on fixed-budget path instances, compared to existing CPE
algorithms. Our experimental results on the top-$k$, path and matching
instances validate the empirical superiority of the proposed algorithms over
their baselines.

    

### [[2102.12781] Do Input Gradients Highlight Discriminative Features?](http://arxiv.org/abs/2102.12781)


  Post-hoc gradient-based interpretability methods [Simonyan et al., 2013,
Smilkov et al., 2017] that provide instance-specific explanations of model
predictions are often based on assumption (A): magnitude of input gradients --
gradients of logits with respect to input -- noisily highlight discriminative
task-relevant features. In this work, we test the validity of assumption (A)
using a three-pronged approach. First, we develop an evaluation framework,
DiffROAR, to test assumption (A) on four image classification benchmarks. Our
results suggest that (i) input gradients of standard models (i.e., trained on
original data) may grossly violate (A), whereas (ii) input gradients of
adversarially robust models satisfy (A). Second, we introduce BlockMNIST, an
MNIST-based semi-real dataset, that by design encodes a priori knowledge of
discriminative features. Our analysis on BlockMNIST leverages this information
to validate as well as characterize differences between input gradient
attributions of standard and robust models. Finally, we theoretically prove
that our empirical findings hold on a simplified version of the BlockMNIST
dataset. Specifically, we prove that input gradients of standard
one-hidden-layer MLPs trained on this dataset do not highlight
instance-specific signal coordinates, thus grossly violating assumption (A).
Our findings motivate the need to formalize and test common assumptions in
interpretability in a falsifiable manner [Leavitt and Morcos, 2020]. We believe
that the DiffROAR evaluation framework and BlockMNIST-based datasets can serve
as sanity checks to audit instance-specific interpretability methods; code and
data available at this https URL.

    

### [[2103.01615] Mini-Batch Consistent Slot Set Encoder for Scalable Set Encoding](http://arxiv.org/abs/2103.01615)


  Most existing set encoding algorithms operate under the implicit assumption
that all the set elements are accessible, and that there are ample
computational and memory resources to load the set into memory during training
and inference. However, both assumptions fail when the set is excessively large
such that it is impossible to load all set elements into memory, or when data
arrives in a stream. To tackle such practical challenges in large-scale set
encoding, the general set-function constraints of permutation invariance and
equivariance are not sufficient. We introduce a new property termed Mini-Batch
Consistency (MBC) that is required for large scale mini-batch set encoding.
Additionally, we present a scalable and efficient attention-based set encoding
mechanism that is amenable to mini-batch processing of sets, and capable of
updating set representations as data arrives. The proposed method adheres to
the required symmetries of invariance and equivariance as well as maintaining
MBC for any partition of the input set. We perform extensive experiments and
show that our method is computationally efficient and results in rich set
encoding representations for set-structured data.

    

### [[2104.00428] Storchastic: A Framework for General Stochastic Automatic Differentiation](http://arxiv.org/abs/2104.00428)


  Modelers use automatic differentiation (AD) of computation graphs to
implement complex Deep Learning models without defining gradient computations.
Stochastic AD extends AD to stochastic computation graphs with sampling steps,
which arise when modelers handle the intractable expectations common in
Reinforcement Learning and Variational Inference. However, current methods for
stochastic AD are limited: They are either only applicable to continuous random
variables and differentiable functions, or can only use simple but high
variance score-function estimators. To overcome these limitations, we introduce
Storchastic, a new framework for AD of stochastic computation graphs.
Storchastic allows the modeler to choose from a wide variety of gradient
estimation methods at each sampling step, to optimally reduce the variance of
the gradient estimates. Furthermore, Storchastic is provably unbiased for
estimation of any-order gradients, and generalizes variance reduction
techniques to higher-order gradient estimates. Finally, we implement
Storchastic as a PyTorch library at this https URL.

    

### [[2104.03736] Towards Enabling Meta-Learning from Target Models](http://arxiv.org/abs/2104.03736)


  Meta-learning can extract an inductive bias from previous learning experience
and assist the training of new tasks. It is often realized through optimizing a
meta-model with the evaluation loss of task-specific solvers. Most existing
algorithms sample non-overlapping $\mathit{support}$ sets and $\mathit{query}$
sets to train and evaluate the solvers respectively due to simplicity
($\mathcal{S}$/$\mathcal{Q}$ protocol). Different from
$\mathcal{S}$/$\mathcal{Q}$ protocol, we can also evaluate a task-specific
solver by comparing it to a target model $\mathcal{T}$, which is the optimal
model for this task or a model that behaves well enough on this task
($\mathcal{S}$/$\mathcal{T}$ protocol). Although being short of research,
$\mathcal{S}$/$\mathcal{T}$ protocol has unique advantages such as offering
more informative supervision, but it is computationally expensive. This paper
looks into this special evaluation method and takes a step towards putting it
into practice. We find that with a small ratio of tasks armed with target
models, classic meta-learning algorithms can be improved a lot without
consuming many resources. We empirically verify the effectiveness of
$\mathcal{S}$/$\mathcal{T}$ protocol in a typical application of meta-learning,
$\mathit{i.e.}$, few-shot learning. In detail, after constructing target models
by fine-tuning the pre-trained network on those hard tasks, we match the
task-specific solvers and target models via knowledge distillation.

    

### [[2105.04683] Deep Bandits Show-Off: Simple and Efficient Exploration with Deep Networks](http://arxiv.org/abs/2105.04683)


  Designing efficient exploration is central to Reinforcement Learning due to
the fundamental problem posed by the exploration-exploitation dilemma. Bayesian
exploration strategies like Thompson Sampling resolve this trade-off in a
principled way by modeling and updating the distribution of the parameters of
the action-value function, the outcome model of the environment. However, this
technique becomes infeasible for complex environments due to the computational
intractability of maintaining probability distributions over parameters of
outcome models of corresponding complexity. Moreover, the approximation
techniques introduced to mitigate this issue typically result in poor
exploration-exploitation trade-offs, as observed in the case of deep neural
network models with approximate posterior methods that have been shown to
underperform in the deep bandit scenario. In this paper we introduce Sample
Average Uncertainty (SAU), a simple and efficient uncertainty measure for
contextual bandits. While Bayesian approaches like Thompson Sampling estimate
outcomes uncertainty indirectly by first quantifying the variability over the
parameters of the outcome model, SAU is a frequentist approach that directly
estimates the uncertainty of the outcomes based on the value predictions.
Importantly, we show theoretically that the uncertainty measure estimated by
SAU asymptotically matches the uncertainty provided by Thompson Sampling, as
well as its regret bounds. Because of its simplicity SAU can be seamlessly
applied to deep contextual bandits as a very scalable drop-in replacement for
epsilon-greedy exploration. We confirm empirically our theory by showing that
SAU-based exploration outperforms current state-of-the-art deep Bayesian bandit
methods on several real-world datasets at modest computation cost. Code is
available at \url{this https URL}.

    

### [[2105.08547] Partitioned Active Learning for Heterogeneous Systems](http://arxiv.org/abs/2105.08547)


  Active learning is a subfield of machine learning that focuses on improving
the data collection efficiency of expensive-to-evaluate systems. Especially,
active learning integrated surrogate modeling has shown remarkable performance
in computationally demanding engineering systems. However, the existence of
heterogeneity in underlying systems may adversely affect the performance of
active learning. In order to improve the learning efficiency under this regime,
we propose the partitioned active learning that seeks the most informative
design points for partitioned Gaussian process modeling of heterogeneous
systems. The proposed active learning consists of two systematic subsequent
steps: the global searching scheme accelerates the exploration of active
learning by investigating the most uncertain design space, and the local
searching exploits the circumscribed information induced by the local GP. We
also propose Cholesky update driven numerical remedies for our active learning
to address the computational complexity challenge. The proposed method is
applied to numerical simulations and two real-world case studies about (i) the
cost-efficient automatic fuselage shape control in aerospace manufacturing; and
(ii) the optimal design of tribocorrosion-resistant alloys in materials
science. The results show that our approach outperforms benchmark methods with
respect to prediction accuracy and computational efficiency.

    

### [[2105.08866] Localization, Convexity, and Star Aggregation](http://arxiv.org/abs/2105.08866)


  Offset Rademacher complexities have been shown to provide tight upper bounds
for the square loss in a broad class of problems including improper statistical
learning and online learning. We show that the offset complexity can be
generalized to any loss that satisfies a certain general convexity condition.
Further, we show that this condition is closely related to both exponential
concavity and self-concordance, unifying apparently disparate results. By a
novel geometric argument, many of our bounds translate to improper learning in
a non-convex class with Audibert's star algorithm. Thus, the offset complexity
provides a versatile analytic tool that covers both convex empirical risk
minimization and improper learning under entropy conditions. Applying the
method, we recover the optimal rates for proper and improper learning with the
$p$-loss for $1 < p < \infty$, and show that improper variants of empirical
risk minimization can attain fast rates for logistic regression and other
generalized linear models.

    

### [[2105.09384] Graph Sanitation with Application to Node Classification](http://arxiv.org/abs/2105.09384)


  The past decades have witnessed the prosperity of graph mining, with a
multitude of sophisticated models and algorithms designed for various mining
tasks, such as ranking, classification, clustering and anomaly detection.
Generally speaking, the vast majority of the existing works aim to answer the
following question, that is, given a graph, what is the best way to mine it? In
this paper, we introduce the graph sanitation problem, to answer an orthogonal
question. That is, given a mining task and an initial graph, what is the best
way to improve the initially provided graph? By learning a better graph as part
of the input of the mining model, it is expected to benefit graph mining in a
variety of settings, ranging from denoising, imputation to defense. We
formulate the graph sanitation problem as a bilevel optimization problem, and
further instantiate it by semi-supervised node classification, together with an
effective solver named GaSoliNe. Extensive experimental results demonstrate
that the proposed method is (1) broadly applicable with respect to different
graph neural network models and flexible graph modification strategies, (2)
effective in improving the node classification accuracy on both the original
and contaminated graphs in various perturbation scenarios. In particular, it
brings up to 25% performance improvement over the existing robust graph neural
network methods.

    

### [[2105.10018] Scalable Multi-Robot System for Non-myopic Spatial Sampling](http://arxiv.org/abs/2105.10018)


  This paper presents a distributed scalable multi-robot planning algorithm for
non-uniform sampling of quasi-static spatial fields. We address the problem of
efficient data collection using multiple autonomous vehicles and consider the
effects of communication between multiple robots, acting independently, on the
overall sampling performance of the team. We focus on the distributed sampling
problem where the robots operate independent of their teammates, but have the
ability to communicate their current state to other neighbors within a fixed
communication range. Our proposed approach is scalable and adaptive to various
environmental scenarios, changing robot team configurations, and runs in
real-time, which are important features for many real-world applications. We
compare the performance of our proposed algorithm to baseline strategies
through simulated experiments that utilize models derived from both synthetic
and field deployment data. The results show that our sampling algorithm is
efficient even when robots in the team are operating with a limited
communication range, thus demonstrating the scalability our method in sampling
large-scale environments.

    

### [[2105.12909] Deconditional Downscaling with Gaussian Processes](http://arxiv.org/abs/2105.12909)


  Refining low-resolution (LR) spatial fields with high-resolution (HR)
information, often known as statistical downscaling, is challenging as the
diversity of spatial datasets often prevents direct matching of observations.
Yet, when LR samples are modeled as aggregate conditional means of HR samples
with respect to a mediating variable that is globally observed, the recovery of
the underlying fine-grained field can be framed as taking an "inverse" of the
conditional expectation, namely a deconditioning problem. In this work, we
propose a Bayesian formulation of deconditioning which naturally recovers the
initial reproducing kernel Hilbert space formulation from Hsu and Ramos (2019).
We extend deconditioning to a downscaling setup and devise efficient
conditional mean embedding estimator for multiresolution data. By treating
conditional expectations as inter-domain features of the underlying field, a
posterior for the latent field can be established as a solution to the
deconditioning problem. Furthermore, we show that this solution can be viewed
as a two-staged vector-valued kernel ridge regressor and show that it has a
minimax optimal convergence rate under mild assumptions. Lastly, we demonstrate
its proficiency in a synthetic and a real-world atmospheric field downscaling
problem, showing substantial improvements over existing methods.

    

### [[2105.13954] A Gradient Method for Multilevel Optimization](http://arxiv.org/abs/2105.13954)


  Although application examples of multilevel optimization have already been
discussed since the 1990s, the development of solution methods was almost
limited to bilevel cases due to the difficulty of the problem. In recent years,
in machine learning, Franceschi et al. have proposed a method for solving
bilevel optimization problems by replacing their lower-level problems with the
$T$ steepest descent update equations with some prechosen iteration number $T$.
In this paper, we have developed a gradient-based algorithm for multilevel
optimization with $n$ levels based on their idea and proved that our
reformulation asymptotically converges to the original multilevel problem. As
far as we know, this is one of the first algorithms with some theoretical
guarantee for multilevel optimization. Numerical experiments show that a
trilevel hyperparameter learning model considering data poisoning produces more
stable prediction results than an existing bilevel hyperparameter learning
model in noisy data settings.

    

### [[2105.13977] Perturbation Theory for the Information Bottleneck](http://arxiv.org/abs/2105.13977)


  Extracting relevant information from data is crucial for all forms of
learning. The information bottleneck (IB) method formalizes this, offering a
mathematically precise and conceptually appealing framework for understanding
learning phenomena. However the nonlinearity of the IB problem makes it
computationally expensive and analytically intractable in general. Here we
derive a perturbation theory for the IB method and report the first complete
characterization of the learning onset, the limit of maximum relevant
information per bit extracted from data. We test our results on synthetic
probability distributions, finding good agreement with the exact numerical
solution near the onset of learning. We explore the difference and subtleties
in our derivation and previous attempts at deriving a perturbation theory for
the learning onset and attribute the discrepancy to a flawed assumption. Our
work also provides a fresh perspective on the intimate relationship between the
IB method and the strong data processing inequality.

    

### [[2105.14033] An Inexact Projected Gradient Method with Rounding and Lifting by Nonlinear Programming for Solving Rank-One Semidefinite Relaxation of Polynomial Optimization](http://arxiv.org/abs/2105.14033)


  We consider solving high-order semidefinite programming (SDP) relaxations of
nonconvex polynomial optimization problems (POPs) that often admit degenerate
rank-one optimal solutions. Instead of solving the SDP alone, we propose a new
algorithmic framework that blends local search using the nonconvex POP into
global descent using the convex SDP. In particular, we first design a globally
convergent inexact projected gradient method (iPGM) for solving the SDP that
serves as the backbone of our framework. We then accelerate iPGM by taking
long, but safeguarded, rank-one steps generated by fast nonlinear programming
algorithms. We prove that the new framework is still globally convergent for
solving the SDP. To solve the iPGM subproblem of projecting a given point onto
the feasible set of the SDP, we design a two-phase algorithm with phase one
using a symmetric Gauss-Seidel based accelerated proximal gradient method
(sGS-APG) to generate a good initial point, and phase two using a modified
limited-memory BFGS (L-BFGS) method to obtain an accurate solution. We analyze
the convergence for both phases and establish a novel global convergence result
for the modified L-BFGS that does not require the objective function to be
twice continuously differentiable. We conduct numerical experiments for solving
second-order SDP relaxations arising from a diverse set of POPs. Our framework
demonstrates state-of-the-art efficiency, scalability, and robustness in
solving degenerate rank-one SDPs to high accuracy, even in the presence of
millions of equality constraints.

    

### [[2105.14039] Towards mental time travel: a hierarchical memory for reinforcement learning agents](http://arxiv.org/abs/2105.14039)


  Reinforcement learning agents often forget details of the past, especially
after delays or distractor tasks. Agents with common memory architectures
struggle to recall and integrate across multiple timesteps of a past event, or
even to recall the details of a single timestep that is followed by distractor
tasks. To address these limitations, we propose a Hierarchical Chunk Attention
Memory (HCAM), which helps agents to remember the past in detail. HCAM stores
memories by dividing the past into chunks, and recalls by first performing
high-level attention over coarse summaries of the chunks, and then performing
detailed attention within only the most relevant chunks. An agent with HCAM can
therefore "mentally time-travel" -- remember past events in detail without
attending to all intervening events. We show that agents with HCAM
substantially outperform agents with other memory architectures at tasks
requiring long-term recall, retention, or reasoning over memory. These include
recalling where an object is hidden in a 3D environment, rapidly learning to
navigate efficiently in a new neighborhood, and rapidly learning and retaining
new object names. Agents with HCAM can extrapolate to task sequences much
longer than they were trained on, and can even generalize zero-shot from a
meta-learning setting to maintaining knowledge across episodes. HCAM improves
agent sample efficiency, generalization, and generality (by solving tasks that
previously required specialized architectures). Our work is a step towards
agents that can learn, interact, and adapt in complex and temporally-extended
environments.

    

### [[2105.14099] Bridging the Gap Between Practice and PAC-Bayes Theory in Few-Shot Meta-Learning](http://arxiv.org/abs/2105.14099)


  Despite recent advances in its theoretical understanding, there still remains
a significant gap in the ability of existing PAC-Bayesian theories on
meta-learning to explain performance improvements in the few-shot learning
setting, where the number of training examples in the target tasks is severely
limited. This gap originates from an assumption in the existing theories which
supposes that the number of training examples in the observed tasks and the
number of training examples in the target tasks follow the same distribution,
an assumption that rarely holds in practice. By relaxing this assumption, we
develop two PAC-Bayesian bounds tailored for the few-shot learning setting and
show that two existing meta-learning algorithms (MAML and Reptile) can be
derived from our bounds, thereby bridging the gap between practice and
PAC-Bayesian theories. Furthermore, we derive a new computationally-efficient
PACMAML algorithm, and show it outperforms existing meta-learning algorithms on
several few-shot benchmark datasets.

    

### [[2105.14655] UNiTE: Unitary N-body Tensor Equivariant Network with Applications to Quantum Chemistry](http://arxiv.org/abs/2105.14655)


  Equivariant neural networks have been successful in incorporating various
types of symmetries, but are mostly limited to vector representations of
geometric objects. Despite the prevalence of higher-order tensors in various
application domains, e.g. in quantum chemistry, equivariant neural networks for
general tensors remain underexplored. Previous strategies for learning
equivariant functions on tensors mostly rely on expensive tensor factorization
which is not scalable when the dimensionality of the problem becomes large. In
this work, we propose unitary $N$-body tensor equivariant neural network
(UNiTE), an architecture for a general class of symmetric tensors called
$N$-body tensors. The proposed neural network is equivariant with respect to
the actions of a unitary group, such as the group of 3D rotations. Furthermore,
it has a linear time complexity with respect to the number of non-zero elements
in the tensor. When applied to quantum chemistry, UNiTE in combination with a
low-cost physics-based molecular representation outperforms state-of-the-art
machine learning methods on multiple benchmarks. Finally, we show that UNiTE
achieves a robust zero-shot generalization performance on diverse down stream
chemistry tasks, while being three orders of magnitude faster than conventional
numerical methods with competitive accuracy.

    

### [[2105.14835] Towards Lower Bounds on the Depth of ReLU Neural Networks](http://arxiv.org/abs/2105.14835)


  We contribute to a better understanding of the class of functions that is
represented by a neural network with ReLU activations and a given architecture.
Using techniques from mixed-integer optimization, polyhedral theory, and
tropical geometry, we provide a mathematical counterbalance to the universal
approximation theorems which suggest that a single hidden layer is sufficient
for learning tasks. In particular, we investigate whether the class of exactly
representable functions strictly increases by adding more layers (with no
restrictions on size). This problem has potential impact on algorithmic and
statistical aspects because of the insight it provides into the class of
functions represented by neural hypothesis classes. However, to the best of our
knowledge, this question has not been investigated in the neural network
literature. We also present upper bounds on the sizes of neural networks
required to represent functions in these neural hypothesis classes.

    

### [[2105.14937] Safe Pontryagin Differentiable Programming](http://arxiv.org/abs/2105.14937)


  We propose a Safe Pontryagin Differentiable Programming (Safe PDP)
methodology, which establishes a theoretical and algorithmic framework to solve
a broad class of safety-critical learning and control tasks -- problems that
require the guarantee of safety constraint satisfaction at any stage of the
learning and control progress. In the spirit of interior-point methods, Safe
PDP handles different types of system constraints on states and inputs by
incorporating them into the cost or loss through barrier functions. We prove
three fundamentals of the proposed Safe PDP: first, both the solution and its
gradient in the backward pass can be approximated by solving their more
efficient unconstrained counterparts; second, the approximation for both the
solution and its gradient can be controlled for arbitrary accuracy by a barrier
parameter; and third, importantly, all intermediate results throughout the
approximation and optimization strictly respect the constraints, thus
guaranteeing safety throughout the entire learning and control process. We
demonstrate the capabilities of Safe PDP in solving various safety-critical
tasks, including safe policy optimization, safe motion planning, and learning
MPCs from demonstrations, on different challenging systems such as 6-DoF
maneuvering quadrotor and 6-DoF rocket powered landing.

    

### [[2105.14995] Choose a Transformer: Fourier or Galerkin](http://arxiv.org/abs/2105.14995)


  In this paper, we apply the self-attention from the state-of-the-art
Transformer in Attention Is All You Need for the first time to a data-driven
operator learning problem related to partial differential equations. An effort
is put together to explain the heuristics of, and to improve the efficacy of
the attention mechanism. By employing the operator approximation theory in
Hilbert spaces, it is demonstrated for the first time that the softmax
normalization in the scaled dot-product attention is sufficient but not
necessary. Without softmax, the approximation capacity of a linearized
Transformer variant can be proved to be comparable to a Petrov-Galerkin
projection layer-wise, and the estimate is independent with respect to the
sequence length. A new layer normalization scheme mimicking the Petrov-Galerkin
projection is proposed to allow a scaling to propagate through attention
layers, which helps the model achieve remarkable accuracy in operator learning
tasks with unnormalized data. Finally, we present three operator learning
experiments, including the viscid Burgers' equation, an interface Darcy flow,
and an inverse interface coefficient identification problem. The newly proposed
simple attention-based operator learner, Galerkin Transformer, shows
significant improvements in both training cost and evaluation accuracy over its
softmax-normalized counterparts.

    

### [[2105.15075] Not All Images are Worth 16x16 Words: Dynamic Transformers for Efficient Image Recognition](http://arxiv.org/abs/2105.15075)


  Vision Transformers (ViT) have achieved remarkable success in large-scale
image recognition. They split every 2D image into a fixed number of patches,
each of which is treated as a token. Generally, representing an image with more
tokens would lead to higher prediction accuracy, while it also results in
drastically increased computational cost. To achieve a decent trade-off between
accuracy and speed, the number of tokens is empirically set to 16x16 or 14x14.
In this paper, we argue that every image has its own characteristics, and
ideally the token number should be conditioned on each individual input. In
fact, we have observed that there exist a considerable number of "easy" images
which can be accurately predicted with a mere number of 4x4 tokens, while only
a small fraction of "hard" ones need a finer representation. Inspired by this
phenomenon, we propose a Dynamic Transformer to automatically configure a
proper number of tokens for each input image. This is achieved by cascading
multiple Transformers with increasing numbers of tokens, which are sequentially
activated in an adaptive fashion at test time, i.e., the inference is
terminated once a sufficiently confident prediction is produced. We further
design efficient feature reuse and relationship reuse mechanisms across
different components of the Dynamic Transformer to reduce redundant
computations. Extensive empirical results on ImageNet, CIFAR-10, and CIFAR-100
demonstrate that our method significantly outperforms the competitive baselines
in terms of both theoretical computational efficiency and practical inference
speed. Code and pre-trained models (based on PyTorch and MindSpore) are
available at this https URL
and this https URL.

    

### [[2106.00651] Asymptotics of representation learning in finite Bayesian neural networks](http://arxiv.org/abs/2106.00651)


  Recent works have suggested that finite Bayesian neural networks may
sometimes outperform their infinite cousins because finite networks can
flexibly adapt their internal representations. However, our theoretical
understanding of how the learned hidden layer representations of finite
networks differ from the fixed representations of infinite networks remains
incomplete. Perturbative finite-width corrections to the network prior and
posterior have been studied, but the asymptotics of learned features have not
been fully characterized. Here, we argue that the leading finite-width
corrections to the average feature kernels for any Bayesian network with linear
readout and Gaussian likelihood have a largely universal form. We illustrate
this explicitly for three tractable network architectures: deep linear
fully-connected and convolutional networks, and networks with a single
nonlinear hidden layer. Our results begin to elucidate how task-relevant
learning signals shape the hidden layer representations of wide Bayesian neural
networks.

    

### [[2106.00769] Improving Compositionality of Neural Networks by Decoding Representations to Inputs](http://arxiv.org/abs/2106.00769)


  In traditional software programs, it is easy to trace program logic from
variables back to input, apply assertion statements to block erroneous
behavior, and compose programs together. Although deep learning programs have
demonstrated strong performance on novel applications, they sacrifice many of
the functionalities of traditional software programs. With this as motivation,
we take a modest first step towards improving deep learning programs by jointly
training a generative model to constrain neural network activations to "decode"
back to inputs. We call this design a Decodable Neural Network, or DecNN. Doing
so enables a form of compositionality in neural networks, where one can
recursively compose DecNN with itself to create an ensemble-like model with
uncertainty. In our experiments, we demonstrate applications of this
uncertainty to out-of-distribution detection, adversarial example detection,
and calibration -- while matching standard neural networks in accuracy. We
further explore this compositionality by combining DecNN with pretrained
models, where we show promising results that neural networks can be regularized
from using protected features.

    

### [[2106.01336] Improved Rates for Differentially Private Stochastic Convex Optimization with Heavy-Tailed Data](http://arxiv.org/abs/2106.01336)


  We study stochastic convex optimization with heavy-tailed data under the
constraint of differential privacy (DP). Most prior work on this problem is
restricted to the case where the loss function is Lipschitz. Instead, as
introduced by Wang, Xiao, Devadas, and Xu \cite{WangXDX20}, we study general
convex loss functions with the assumption that the distribution of gradients
has bounded $k$-th moments. We provide improved upper bounds on the excess
population risk under concentrated DP for convex and strongly convex loss
functions. Along the way, we derive new algorithms for private mean estimation
of heavy-tailed distributions, under both pure and concentrated DP. Finally, we
prove nearly-matching lower bounds for private stochastic convex optimization
with strongly convex losses and mean estimation, showing new separations
between pure and concentrated DP.

    

### [[2106.01921] Sample Selection Bias in Evaluation of Prediction Performance of Causal Models](http://arxiv.org/abs/2106.01921)


  Causal models are notoriously difficult to validate because they make
untestable assumptions regarding confounding. New scientific experiments offer
the possibility of evaluating causal models using prediction performance.
Prediction performance measures are typically robust to violations in causal
assumptions. However, prediction performance does depend on the selection of
training and test sets. Biased training sets can lead to optimistic assessments
of model performance. In this work, we revisit the prediction performance of
several recently proposed causal models tested on a genetic perturbation data
set of Kemmeren. We find that sample selection bias is likely a key driver of
model performance. We propose using a less-biased evaluation set for assessing
prediction performance and compare models on this new set. In this setting, the
causal models have similar or worse performance compared to standard
association-based estimators such as Lasso. Finally, we compare the performance
of causal estimators in simulation studies that reproduce the Kemmeren
structure of genetic knockout experiments but without any sample selection
bias. These results provide an improved understanding of the performance of
several causal models and offer guidance on how future studies should use
Kemmeren.

    

### [[2106.02034] DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification](http://arxiv.org/abs/2106.02034)


  Attention is sparse in vision transformers. We observe the final prediction
in vision transformers is only based on a subset of most informative tokens,
which is sufficient for accurate image recognition. Based on this observation,
we propose a dynamic token sparsification framework to prune redundant tokens
progressively and dynamically based on the input. Specifically, we devise a
lightweight prediction module to estimate the importance score of each token
given the current features. The module is added to different layers to prune
redundant tokens hierarchically. To optimize the prediction module in an
end-to-end manner, we propose an attention masking strategy to differentiably
prune a token by blocking its interactions with other tokens. Benefiting from
the nature of self-attention, the unstructured sparse tokens are still hardware
friendly, which makes our framework easy to achieve actual speed-up. By
hierarchically pruning 66% of the input tokens, our method greatly reduces
31%~37% FLOPs and improves the throughput by over 40% while the drop of
accuracy is within 0.5% for various vision transformers. Equipped with the
dynamic token sparsification framework, DynamicViT models can achieve very
competitive complexity/accuracy trade-offs compared to state-of-the-art CNNs
and vision transformers on ImageNet. Code is available at
this https URL


### [[2106.02105] A Little Robustness Goes a Long Way: Leveraging Robust Features for Targeted Transfer Attacks](http://arxiv.org/abs/2106.02105)


  Adversarial examples for neural network image classifiers are known to be
transferable: examples optimized to be misclassified by a source classifier are
often misclassified as well by classifiers with different architectures.
However, targeted adversarial examples -- optimized to be classified as a
chosen target class -- tend to be less transferable between architectures.
While prior research on constructing transferable targeted attacks has focused
on improving the optimization procedure, in this work we examine the role of
the source classifier. Here, we show that training the source classifier to be
"slightly robust" -- that is, robust to small-magnitude adversarial examples --
substantially improves the transferability of class-targeted and
representation-targeted adversarial attacks, even between architectures as
different as convolutional neural networks and transformers. The results we
present provide insight into the nature of adversarial examples as well as the
mechanisms underlying so-called "robust" classifiers.

    

### [[2106.02720] An Even More Optimal Stochastic Optimization Algorithm: Minibatching and Interpolation Learning](http://arxiv.org/abs/2106.02720)


  We present and analyze an algorithm for optimizing smooth and convex or
strongly convex objectives using minibatch stochastic gradient estimates. The
algorithm is optimal with respect to its dependence on both the minibatch size
and minimum expected loss simultaneously. This improves over the optimal method
of Lan (2012), which is insensitive to the minimum expected loss; over the
optimistic acceleration of Cotter et al. (2011), which has suboptimal
dependence on the minibatch size; and over the algorithm of Liu and Belkin
(2018), which is limited to least squares problems and is also similarly
suboptimal with respect to the minibatch size. Applied to interpolation
learning, the improvement over Cotter et al. and Liu and Belkin translates to a
linear, rather than square-root, parallelization speedup.

    

### [[2106.02734] Revisiting Hilbert-Schmidt Information Bottleneck for Adversarial Robustness](http://arxiv.org/abs/2106.02734)


  We investigate the HSIC (Hilbert-Schmidt independence criterion) bottleneck
as a regularizer for learning an adversarially robust deep neural network
classifier. In addition to the usual cross-entropy loss, we add regularization
terms for every intermediate layer to ensure that the latent representations
retain useful information for output prediction while reducing redundant
information. We show that the HSIC bottleneck enhances robustness to
adversarial attacks both theoretically and experimentally. In particular, we
prove that the HSIC bottleneck regularizer reduces the sensitivity of the
classifier to adversarial examples. Our experiments on multiple benchmark
datasets and architectures demonstrate that incorporating an HSIC bottleneck
regularizer attains competitive natural accuracy and improves adversarial
robustness, both with and without adversarial examples during training. Our
code and adversarially robust models are publicly available.

    

### [[2106.02848] Numerical Composition of Differential Privacy](http://arxiv.org/abs/2106.02848)


  We give a fast algorithm to optimally compose privacy guarantees of
differentially private (DP) algorithms to arbitrary accuracy. Our method is
based on the notion of privacy loss random variables to quantify the privacy
loss of DP algorithms. The running time and memory needed for our algorithm to
approximate the privacy curve of a DP algorithm composed with itself $k$ times
is $\tilde{O}(\sqrt{k})$. This improves over the best prior method by Koskela
et al. (2020) which requires $\tilde{\Omega}(k^{1.5})$ running time. We
demonstrate the utility of our algorithm by accurately computing the privacy
loss of DP-SGD algorithm of Abadi et al. (2016) and showing that our algorithm
speeds up the privacy computations by a few orders of magnitude compared to
prior work, while maintaining similar accuracy.

    

### [[2106.03668] Recovery Analysis for Plug-and-Play Priors using the Restricted Eigenvalue Condition](http://arxiv.org/abs/2106.03668)


  The plug-and-play priors (PnP) and regularization by denoising (RED) methods
have become widely used for solving inverse problems by leveraging pre-trained
deep denoisers as image priors. While the empirical imaging performance and the
theoretical convergence properties of these algorithms have been widely
investigated, their recovery properties have not previously been theoretically
analyzed. We address this gap by showing how to establish theoretical recovery
guarantees for PnP/RED by assuming that the solution of these methods lies near
the fixed-points of a deep neural network. We also present numerical results
comparing the recovery performance of PnP/RED in compressive sensing against
that of recent compressive sensing algorithms based on generative models. Our
numerical results suggest that PnP with a pre-trained artifact removal network
provides significantly better results compared to the existing state-of-the-art
methods.

    

### [[2106.03696] Dynamics of Stochastic Momentum Methods on Large-scale, Quadratic Models](http://arxiv.org/abs/2106.03696)


  We analyze a class of stochastic gradient algorithms with momentum on a
high-dimensional random least squares problem. Our framework, inspired by
random matrix theory, provides an exact (deterministic) characterization for
the sequence of loss values produced by these algorithms which is expressed
only in terms of the eigenvalues of the Hessian. This leads to simple
expressions for nearly-optimal hyperparameters, a description of the limiting
neighborhood, and average-case complexity.
As a consequence, we show that (small-batch) stochastic heavy-ball momentum
with a fixed momentum parameter provides no actual performance improvement over
SGD when step sizes are adjusted correctly. For contrast, in the non-strongly
convex setting, it is possible to get a large improvement over SGD using
momentum. By introducing hyperparameters that depend on the number of samples,
we propose a new algorithm sDANA (stochastic dimension adjusted Nesterov
acceleration) which obtains an asymptotically optimal average-case complexity
while remaining linearly convergent in the strongly convex setting without
adjusting parameters.

    

### [[2106.03831] Counterfactual Maximum Likelihood Estimation for Training Deep Networks](http://arxiv.org/abs/2106.03831)


  Although deep learning models have driven state-of-the-art performance on a
wide array of tasks, they are prone to spurious correlations that should not be
learned as predictive clues. To mitigate this problem, we propose a
causality-based training framework to reduce the spurious correlations caused
by observed confounders. We give theoretical analysis on the underlying general
Structural Causal Model (SCM) and propose to perform Maximum Likelihood
Estimation (MLE) on the interventional distribution instead of the
observational distribution, namely Counterfactual Maximum Likelihood Estimation
(CMLE). As the interventional distribution, in general, is hidden from the
observational data, we then derive two different upper bounds of the expected
negative log-likelihood and propose two general algorithms, Implicit CMLE and
Explicit CMLE, for causal predictions of deep learning models using
observational data. We conduct experiments on both simulated data and two
real-world tasks: Natural Language Inference (NLI) and Image Captioning. The
results show that CMLE methods outperform the regular MLE method in terms of
out-of-domain generalization performance and reducing spurious correlations,
while maintaining comparable performance on the regular evaluations.

    

### [[2106.04538] What Makes Multi-modal Learning Better than Single (Provably)](http://arxiv.org/abs/2106.04538)


  The world provides us with data of multiple modalities. Intuitively, models
fusing data from different modalities outperform their uni-modal counterparts,
since more information is aggregated. Recently, joining the success of deep
learning, there is an influential line of work on deep multi-modal learning,
which has remarkable empirical results on various applications. However,
theoretical justifications in this field are notably lacking.
Can multi-modal learning provably perform better than uni-modal?
In this paper, we answer this question under a most popular multi-modal
fusion framework, which firstly encodes features from different modalities into
a common latent space and seamlessly maps the latent representations into the
task space. We prove that learning with multiple modalities achieves a smaller
population risk than only using its subset of modalities. The main intuition is
that the former has a more accurate estimate of the latent space
representation. To the best of our knowledge, this is the first theoretical
treatment to capture important qualitative phenomena observed in real
multi-modal applications from the generalization perspective. Combining with
experiment results, we show that multi-modal learning does possess an appealing
formal guarantee.

    

### [[2106.05933] PARP: Prune, Adjust and Re-Prune for Self-Supervised Speech Recognition](http://arxiv.org/abs/2106.05933)


  Self-supervised speech representation learning (speech SSL) has demonstrated
the benefit of scale in learning rich representations for Automatic Speech
Recognition (ASR) with limited paired data, such as wav2vec 2.0. We investigate
the existence of sparse subnetworks in pre-trained speech SSL models that
achieve even better low-resource ASR results. However, directly applying widely
adopted pruning methods such as the Lottery Ticket Hypothesis (LTH) is
suboptimal in the computational cost needed. Moreover, we show that the
discovered subnetworks yield minimal performance gain compared to the original
dense network. We present Prune-Adjust-Re-Prune (PARP), which discovers and
finetunes subnetworks for much better performance, while only requiring a
single downstream ASR finetuning run. PARP is inspired by our surprising
observation that subnetworks pruned for pre-training tasks need merely a slight
adjustment to achieve a sizeable performance boost in downstream ASR tasks.
Extensive experiments on low-resource ASR verify (1) sparse subnetworks exist
in mono-lingual/multi-lingual pre-trained speech SSL, and (2) the computational
advantage and performance gain of PARP over baseline pruning methods. In
particular, on the 10min Librispeech split without LM decoding, PARP discovers
subnetworks from wav2vec 2.0 with an absolute 10.9%/12.6% WER decrease compared
to the full model. We further demonstrate the effectiveness of PARP via:
cross-lingual pruning without any phone recognition degradation, the discovery
of a multi-lingual subnetwork for 10 spoken languages in 1 finetuning run, and
its applicability to pre-trained BERT/XLNet for natural language tasks.

    

### [[2106.06048] Optimizing Bayesian Recurrent Neural Networks on an FPGA-based Accelerator](http://arxiv.org/abs/2106.06048)


  Neural networks have demonstrated their outstanding performance in a wide
range of tasks. Specifically recurrent architectures based on long-short term
memory (LSTM) cells have manifested excellent capability to model time
dependencies in real-world data. However, standard recurrent architectures
cannot estimate their uncertainty which is essential for safety-critical
applications such as in medicine. In contrast, Bayesian recurrent neural
networks (RNNs) are able to provide uncertainty estimation with improved
accuracy. Nonetheless, Bayesian RNNs are computationally and memory demanding,
which limits their practicality despite their advantages. To address this
issue, we propose an FPGA-based hardware design to accelerate Bayesian
LSTM-based RNNs. To further improve the overall algorithmic-hardware
performance, a co-design framework is proposed to explore the most fitting
algorithmic-hardware configurations for Bayesian RNNs. We conduct extensive
experiments on healthcare applications to demonstrate the improvement of our
design and the effectiveness of our framework. Compared with GPU
implementation, our FPGA-based design can achieve up to 10 times speedup with
nearly 106 times higher energy efficiency. To the best of our knowledge, this
is the first work targeting acceleration of Bayesian RNNs on FPGAs.

    

### [[2106.06426] Catch-A-Waveform: Learning to Generate Audio from a Single Short Example](http://arxiv.org/abs/2106.06426)


  Models for audio generation are typically trained on hours of recordings.
Here, we illustrate that capturing the essence of an audio source is typically
possible from as little as a few tens of seconds from a single training signal.
Specifically, we present a GAN-based generative model that can be trained on
one short audio signal from any domain (e.g. speech, music, etc.) and does not
require pre-training or any other form of external supervision. Once trained,
our model can generate random samples of arbitrary duration that maintain
semantic similarity to the training waveform, yet exhibit new compositions of
its audio primitives. This enables a long line of interesting applications,
including generating new jazz improvisations or new a-cappella rap variants
based on a single short example, producing coherent modifications to famous
songs (e.g. adding a new verse to a Beatles song based solely on the original
recording), filling-in of missing parts (inpainting), extending the bandwidth
of a speech signal (super-resolution), and enhancing old recordings without
access to any clean training example. We show that in all cases, no more than
20 seconds of training audio commonly suffice for our model to achieve
state-of-the-art results. This is despite its complete lack of prior knowledge
about the nature of audio signals in general.

    

### [[2106.06762] Solving Graph-based Public Good Games with Tree Search and Imitation Learning](http://arxiv.org/abs/2106.06762)


  Public goods games represent insightful settings for studying incentives for
individual agents to make contributions that, while costly for each of them,
benefit the wider society. In this work, we adopt the perspective of a central
planner with a global view of a network of self-interested agents and the goal
of maximizing some desired property in the context of a best-shot public goods
game. Existing algorithms for this known NP-complete problem find solutions
that are sub-optimal and cannot optimize for criteria other than social
welfare.
In order to efficiently solve public goods games, our proposed method
directly exploits the correspondence between equilibria and the Maximal
Independent Set (mIS) structural property of graphs. In particular, we define a
Markov Decision Process which incrementally generates an mIS, and adopt a
planning method to search for equilibria, outperforming existing methods.
Furthermore, we devise a graph imitation learning technique that uses
demonstrations of the search to obtain a graph neural network parametrized
policy which quickly generalizes to unseen game instances. Our evaluation
results show that this policy is able to reach 99.5% of the performance of the
planning method while being three orders of magnitude faster to evaluate on the
largest graphs tested. The methods presented in this work can be applied to a
large class of public goods games of potentially high societal impact and more
broadly to other graph combinatorial optimization problems.

    

### [[2106.07841] Randomized Exploration for Reinforcement Learning with General Value Function Approximation](http://arxiv.org/abs/2106.07841)


  We propose a model-free reinforcement learning algorithm inspired by the
popular randomized least squares value iteration (RLSVI) algorithm as well as
the optimism principle. Unlike existing upper-confidence-bound (UCB) based
approaches, which are often computationally intractable, our algorithm drives
exploration by simply perturbing the training data with judiciously chosen
i.i.d. scalar noises. To attain optimistic value function estimation without
resorting to a UCB-style bonus, we introduce an optimistic reward sampling
procedure. When the value functions can be represented by a function class
$\mathcal{F}$, our algorithm achieves a worst-case regret bound of
$\widetilde{O}(\mathrm{poly}(d_EH)\sqrt{T})$ where $T$ is the time elapsed, $H$
is the planning horizon and $d_E$ is the $\textit{eluder dimension}$ of
$\mathcal{F}$. In the linear setting, our algorithm reduces to LSVI-PHE, a
variant of RLSVI, that enjoys an $\widetilde{\mathcal{O}}(\sqrt{d^3H^3T})$
regret. We complement the theory with an empirical evaluation across known
difficult exploration tasks.

    

### [[2106.07998] Revisiting the Calibration of Modern Neural Networks](http://arxiv.org/abs/2106.07998)


  Accurate estimation of predictive uncertainty (model calibration) is
essential for the safe application of neural networks. Many instances of
miscalibration in modern neural networks have been reported, suggesting a trend
that newer, more accurate models produce poorly calibrated predictions. Here,
we revisit this question for recent state-of-the-art image classification
models. We systematically relate model calibration and accuracy, and find that
the most recent models, notably those not using convolutions, are among the
best calibrated. Trends observed in prior model generations, such as decay of
calibration with distribution shift or model size, are less pronounced in
recent architectures. We also show that model size and amount of pretraining do
not fully explain these differences, suggesting that architecture is a major
determinant of calibration properties.

    

### [[2106.08233] Spot the Difference: Detection of Topological Changes via Geometric Alignment](http://arxiv.org/abs/2106.08233)


  Geometric alignment appears in a variety of applications, ranging from domain
adaptation, optimal transport, and normalizing flows in machine learning;
optical flow and learned augmentation in computer vision and deformable
registration within biomedical imaging. A recurring challenge is the alignment
of domains whose topology is not the same; a problem that is routinely ignored,
potentially introducing bias in downstream analysis. As a first step towards
solving such alignment problems, we propose an unsupervised algorithm for the
detection of changes in image topology. The model is based on a conditional
variational auto-encoder and detects topological changes between two images
during the registration step. We account for both topological changes in the
image under spatial variation and unexpected transformations. Our approach is
validated on two tasks and datasets: detection of topological changes in
microscopy images of cells, and unsupervised anomaly detection brain imaging.

    

### [[2106.09884] Batch Multi-Fidelity Bayesian Optimization with Deep Auto-Regressive Networks](http://arxiv.org/abs/2106.09884)


  Bayesian optimization (BO) is a powerful approach for optimizing black-box,
expensive-to-evaluate functions. To enable a flexible trade-off between the
cost and accuracy, many applications allow the function to be evaluated at
different fidelities. In order to reduce the optimization cost while maximizing
the benefit-cost ratio, in this paper, we propose Batch Multi-fidelity Bayesian
Optimization with Deep Auto-Regressive Networks (BMBO-DARN). We use a set of
Bayesian neural networks to construct a fully auto-regressive model, which is
expressive enough to capture strong yet complex relationships across all the
fidelities, so as to improve the surrogate learning and optimization
performance. Furthermore, to enhance the quality and diversity of queries, we
develop a simple yet efficient batch querying method, without any combinatorial
search over the fidelities. We propose a batch acquisition function based on
Max-value Entropy Search (MES) principle, which penalizes highly correlated
queries and encourages diversity. We use posterior samples and moment matching
to fulfill efficient computation of the acquisition function and conduct
alternating optimization over every fidelity-input pair, which guarantees an
improvement at each step. We demonstrate the advantage of our approach on four
real-world hyperparameter optimization applications.

    

### [[2106.09993] Accumulative Poisoning Attacks on Real-time Data](http://arxiv.org/abs/2106.09993)


  Collecting training data from untrusted sources exposes machine learning
services to poisoning adversaries, who maliciously manipulate training data to
degrade the model accuracy. When trained on offline datasets, poisoning
adversaries have to inject the poisoned data in advance before training, and
the order of feeding these poisoned batches into the model is stochastic. In
contrast, practical systems are more usually trained/fine-tuned on sequentially
captured real-time data, in which case poisoning adversaries could dynamically
poison each data batch according to the current model state. In this paper, we
focus on the real-time settings and propose a new attacking strategy, which
affiliates an accumulative phase with poisoning attacks to secretly (i.e.,
without affecting accuracy) magnify the destructive effect of a (poisoned)
trigger batch. By mimicking online learning and federated learning on MNIST and
CIFAR-10, we show that model accuracy significantly drops by a single update
step on the trigger batch after the accumulative phase. Our work validates that
a well-designed but straightforward attacking strategy can dramatically amplify
the poisoning effects, with no need to explore complex techniques.

    

### [[2106.11113] Matrix Encoding Networks for Neural Combinatorial Optimization](http://arxiv.org/abs/2106.11113)


  Machine Learning (ML) can help solve combinatorial optimization (CO) problems
better. A popular approach is to use a neural net to compute on the parameters
of a given CO problem and extract useful information that guides the search for
good solutions. Many CO problems of practical importance can be specified in a
matrix form of parameters quantifying the relationship between two groups of
items. There is currently no neural net model, however, that takes in such
matrix-style relationship data as an input. Consequently, these types of CO
problems have been out of reach for ML engineers. In this paper, we introduce
Matrix Encoding Network (MatNet) and show how conveniently it takes in and
processes parameters of such complex CO problems. Using an end-to-end model
based on MatNet, we solve asymmetric traveling salesman (ATSP) and flexible
flow shop (FFSP) problems as the earliest neural approach. In particular, for a
class of FFSP we have tested MatNet on, we demonstrate a far superior empirical
performance to any methods (neural or not) known to date.

    

### [[2106.11535] Particle Cloud Generation with Message Passing Generative Adversarial Networks](http://arxiv.org/abs/2106.11535)


  In high energy physics (HEP), jets are collections of correlated particles
produced ubiquitously in particle collisions such as those at the CERN Large
Hadron Collider (LHC). Machine learning (ML)-based generative models, such as
generative adversarial networks (GANs), have the potential to significantly
accelerate LHC jet simulations. However, despite jets having a natural
representation as a set of particles in momentum-space, a.k.a. a particle
cloud, there exist no generative models applied to such a dataset. In this
work, we introduce a new particle cloud dataset (JetNet), and apply to it
existing point cloud GANs. Results are evaluated using (1) 1-Wasserstein
distances between high- and low-level feature distributions, (2) a newly
developed Fréchet ParticleNet Distance, and (3) the coverage and (4)
minimum matching distance metrics. Existing GANs are found to be inadequate for
physics applications, hence we develop a new message passing GAN (MPGAN), which
outperforms existing point cloud GANs on virtually every metric and shows
promise for use in HEP. We propose JetNet as a novel point-cloud-style dataset
for the ML community to experiment with, and set MPGAN as a benchmark to
improve upon for future generative models. Additionally, to facilitate research
and improve accessibility and reproducibility in this area, we release the
open-source JetNet Python package with interfaces for particle cloud datasets,
implementations for evaluation and loss metrics, and more tools for ML in HEP
development.

    

### [[2106.13695] CADDA: Class-wise Automatic Differentiable Data Augmentation for EEG Signals](http://arxiv.org/abs/2106.13695)


  Data augmentation is a key element of deep learning pipelines, as it informs
the network during training about transformations of the input data that keep
the label unchanged. Manually finding adequate augmentation methods and
parameters for a given pipeline is however rapidly cumbersome. In particular,
while intuition can guide this decision for images, the design and choice of
augmentation policies remains unclear for more complex types of data, such as
neuroscience signals. Besides, class-dependent augmentation strategies have
been surprisingly unexplored in the literature, although it is quite intuitive:
changing the color of a car image does not change the object class to be
predicted, but doing the same to the picture of an orange does. This paper
investigates gradient-based automatic data augmentation algorithms amenable to
class-wise policies with exponentially larger search spaces. Motivated by
supervised learning applications using EEG signals for which good augmentation
policies are mostly unknown, we propose a new differentiable relaxation of the
problem. In the class-agnostic setting, results show that our new relaxation
leads to optimal performance with faster training than competing gradient-based
methods, while also outperforming gradient-free methods in the class-wise
setting. This work proposes also novel differentiable augmentation operations
relevant for sleep stage classification.

    

### [[2106.13906] Compositional Reinforcement Learning from Logical Specifications](http://arxiv.org/abs/2106.13906)


  We study the problem of learning control policies for complex tasks given by
logical specifications. Recent approaches automatically generate a reward
function from a given specification and use a suitable reinforcement learning
algorithm to learn a policy that maximizes the expected reward. These
approaches, however, scale poorly to complex tasks that require high-level
planning. In this work, we develop a compositional learning approach, called
DiRL, that interleaves high-level planning and reinforcement learning. First,
DiRL encodes the specification as an abstract graph; intuitively, vertices and
edges of the graph correspond to regions of the state space and simpler
sub-tasks, respectively. Our approach then incorporates reinforcement learning
to learn neural network policies for each edge (sub-task) within a
Dijkstra-style planning algorithm to compute a high-level plan in the graph. An
evaluation of the proposed approach on a set of challenging control benchmarks
with continuous state and action spaces demonstrates that it outperforms
state-of-the-art baselines.

    

### [[2106.14308] Concentration of Contractive Stochastic Approximation and Reinforcement Learning](http://arxiv.org/abs/2106.14308)


  Using a martingale concentration inequality, concentration bounds `from time
$n_0$ on' are derived for stochastic approximation algorithms with contractive
maps and both martingale difference and Markov noises. These are applied to
reinforcement learning algorithms, in particular to asynchronous Q-learning and
TD(0).

    

### [[2106.14952] Adversarial Robustness of Streaming Algorithms through Importance Sampling](http://arxiv.org/abs/2106.14952)


  In this paper, we introduce adversarially robust streaming algorithms for
central machine learning and algorithmic tasks, such as regression and
clustering, as well as their more general counterparts, subspace embedding,
low-rank approximation, and coreset construction. For regression and other
numerical linear algebra related tasks, we consider the row arrival streaming
model. Our results are based on a simple, but powerful, observation that many
importance sampling-based algorithms give rise to adversarial robustness which
is in contrast to sketching based algorithms, which are very prevalent in the
streaming literature but suffer from adversarial attacks. In addition, we show
that the well-known merge and reduce paradigm in streaming is adversarially
robust. Since the merge and reduce paradigm allows coreset constructions in the
streaming setting, we thus obtain robust algorithms for $k$-means, $k$-median,
$k$-center, Bregman clustering, projective clustering, principal component
analysis (PCA) and non-negative matrix factorization. To the best of our
knowledge, these are the first adversarially robust results for these problems
yet require no new algorithmic implementations. Finally, we empirically confirm
the robustness of our algorithms on various adversarial attacks and demonstrate
that by contrast, some common existing algorithms are not robust.
(Abstract shortened to meet arXiv limits)

    

### [[2106.15482] Personalized Federated Learning with Gaussian Processes](http://arxiv.org/abs/2106.15482)


  Federated learning aims to learn a global model that performs well on client
devices with limited cross-client communication. Personalized federated
learning (PFL) further extends this setup to handle data heterogeneity between
clients by learning personalized models. A key challenge in this setting is to
learn effectively across clients even though each client has unique data that
is often limited in size. Here we present pFedGP, a solution to PFL that is
based on Gaussian processes (GPs) with deep kernel learning. GPs are highly
expressive models that work well in the low data regime due to their Bayesian
nature. However, applying GPs to PFL raises multiple challenges. Mainly, GPs
performance depends heavily on access to a good kernel function, and learning a
kernel requires a large training set. Therefore, we propose learning a shared
kernel function across all clients, parameterized by a neural network, with a
personal GP classifier for each client. We further extend pFedGP to include
inducing points using two novel methods, the first helps to improve
generalization in the low data regime and the second reduces the computational
cost. We derive a PAC-Bayes generalization bound on novel clients and
empirically show that it gives non-vacuous guarantees. Extensive experiments on
standard PFL benchmarks with CIFAR-10, CIFAR-100, and CINIC-10, and on a new
setup of learning under input noise show that pFedGP achieves well-calibrated
predictions while significantly outperforming baseline methods, reaching up to
21% in accuracy gain.

    

### [[2106.16048] Resilient UAV Swarm Communications with Graph Convolutional Neural Network](http://arxiv.org/abs/2106.16048)


  In this paper, we study the self-healing problem of unmanned aerial vehicle
(UAV) swarm network (USNET) that is required to quickly rebuild the
communication connectivity under unpredictable external disruptions (UEDs).
Firstly, to cope with the one-off UEDs, we propose a graph convolutional neural
network (GCN) and find the recovery topology of the USNET in an on-line manner.
Secondly, to cope with general UEDs, we develop a GCN based trajectory planning
algorithm that can make UAVs rebuild the communication connectivity during the
self-healing process. We also design a meta learning scheme to facilitate the
on-line executions of the GCN. Numerical results show that the proposed
algorithms can rebuild the communication connectivity of the USNET more quickly
than the existing algorithms under both one-off UEDs and general UEDs. The
simulation results also show that the meta learning scheme can not only enhance
the performance of the GCN but also reduce the time complexity of the on-line
executions.

    

### [[2107.00645] Global Filter Networks for Image Classification](http://arxiv.org/abs/2107.00645)


  Recent advances in self-attention and pure multi-layer perceptrons (MLP)
models for vision have shown great potential in achieving promising performance
with fewer inductive biases. These models are generally based on learning
interaction among spatial locations from raw data. The complexity of
self-attention and MLP grows quadratically as the image size increases, which
makes these models hard to scale up when high-resolution features are required.
In this paper, we present the Global Filter Network (GFNet), a conceptually
simple yet computationally efficient architecture, that learns long-term
spatial dependencies in the frequency domain with log-linear complexity. Our
architecture replaces the self-attention layer in vision transformers with
three key operations: a 2D discrete Fourier transform, an element-wise
multiplication between frequency-domain features and learnable global filters,
and a 2D inverse Fourier transform. We exhibit favorable accuracy/complexity
trade-offs of our models on both ImageNet and downstream tasks. Our results
demonstrate that GFNet can be a very competitive alternative to
transformer-style models and CNNs in efficiency, generalization ability and
robustness. Code is available at this https URL


### [[2107.01105] Memory Efficient Meta-Learning with Large Images](http://arxiv.org/abs/2107.01105)


  Meta learning approaches to few-shot classification are computationally
efficient at test time, requiring just a few optimization steps or single
forward pass to learn a new task, but they remain highly memory-intensive to
train. This limitation arises because a task's entire support set, which can
contain up to 1000 images, must be processed before an optimization step can be
taken. Harnessing the performance gains offered by large images thus requires
either parallelizing the meta-learner across multiple GPUs, which may not be
available, or trade-offs between task and image size when memory constraints
apply. We improve on both options by proposing LITE, a general and memory
efficient episodic training scheme that enables meta-training on large tasks
composed of large images on a single GPU. We achieve this by observing that the
gradients for a task can be decomposed into a sum of gradients over the task's
training images. This enables us to perform a forward pass on a task's entire
training set but realize significant memory savings by back-propagating only a
random subset of these images which we show is an unbiased approximation of the
full gradient. We use LITE to train meta-learners and demonstrate new
state-of-the-art accuracy on the real-world ORBIT benchmark and 3 of the 4
parts of the challenging VTAB+MD benchmark relative to leading meta-learners.
LITE also enables meta-learners to be competitive with transfer learning
approaches but at a fraction of the test-time computational cost, thus serving
as a counterpoint to the recent narrative that transfer learning is all you
need for few-shot classification.

    

### [[2107.01163] Unveiling the structure of wide flat minima in neural networks](http://arxiv.org/abs/2107.01163)


  The success of deep learning has revealed the application potential of neural
networks across the sciences and opened up fundamental theoretical problems. In
particular, the fact that learning algorithms based on simple variants of
gradient methods are able to find near-optimal minima of highly nonconvex loss
functions is an unexpected feature of neural networks. Moreover, such
algorithms are able to fit the data even in the presence of noise, and yet they
have excellent predictive capabilities. Several empirical results have shown a
reproducible correlation between the so-called flatness of the minima achieved
by the algorithms and the generalization performance. At the same time,
statistical physics results have shown that in nonconvex networks a multitude
of narrow minima may coexist with a much smaller number of wide flat minima,
which generalize well. Here we show that wide flat minima arise as complex
extensive structures, from the coalescence of minima around "high-margin"
(i.e., locally robust) configurations. Despite being exponentially rare
compared to zero-margin ones, high-margin minima tend to concentrate in
particular regions. These minima are in turn surrounded by other solutions of
smaller and smaller margin, leading to dense regions of solutions over long
distances. Our analysis also provides an alternative analytical method for
estimating when flat minima appear and when algorithms begin to find solutions,
as the number of model parameters varies.

    

### [[2108.04105] Towards better data discovery and collection with flow-based programming](http://arxiv.org/abs/2108.04105)


  Despite huge successes reported by the field of machine learning, such as
voice assistants or self-driving cars, businesses still observe very high
failure rate when it comes to deployment of ML in production. We argue that
part of the reason is infrastructure that was not designed for data-oriented
activities. This paper explores the potential of flow-based programming (FBP)
for simplifying data discovery and collection in software systems. We compare
FBP with the currently prevalent service-oriented paradigm to assess
characteristics of each paradigm in the context of ML deployment. We develop a
data processing application, formulate a subsequent ML deployment task, and
measure the impact of the task implementation within both programming
paradigms. Our main conclusion is that FBP shows great potential for providing
data-centric infrastructural benefits for deployment of ML. Additionally, we
provide an insight into the current trend that prioritizes model development
over data quality management.

    

### [[2109.06153] Relaxed Marginal Consistency for Differentially Private Query Answering](http://arxiv.org/abs/2109.06153)


  Many differentially private algorithms for answering database queries involve
a step that reconstructs a discrete data distribution from noisy measurements.
This provides consistent query answers and reduces error, but often requires
space that grows exponentially with dimension. Private-PGM is a recent approach
that uses graphical models to represent the data distribution, with complexity
proportional to that of exact marginal inference in a graphical model with
structure determined by the co-occurrence of variables in the noisy
measurements. Private-PGM is highly scalable for sparse measurements, but may
fail to run in high dimensions with dense measurements. We overcome the main
scalability limitation of Private-PGM through a principled approach that
relaxes consistency constraints in the estimation objective. Our new approach
works with many existing private query answering algorithms and improves
scalability or accuracy with no privacy cost.

    

### [[2109.09710] Understanding neural networks with reproducing kernel Banach spaces](http://arxiv.org/abs/2109.09710)


  Characterizing the function spaces corresponding to neural networks can
provide a way to understand their properties. In this paper we discuss how the
theory of reproducing kernel Banach spaces can be used to tackle this
challenge. In particular, we prove a representer theorem for a wide class of
reproducing kernel Banach spaces that admit a suitable integral representation
and include one hidden layer neural networks of possibly infinite width.
Further, we show that, for a suitable class of ReLU activation functions, the
norm in the corresponding reproducing kernel Banach space can be characterized
in terms of the inverse Radon transform of a bounded real measure, with norm
given by the total variation norm of the measure. Our analysis simplifies and
extends recent results in [34,29,30].

    

### [[2110.12459] Non-convex Distributionally Robust Optimization: Non-asymptotic Analysis](http://arxiv.org/abs/2110.12459)


  Distributionally robust optimization (DRO) is a widely-used approach to learn
models that are robust against distribution shift. Compared with the standard
optimization setting, the objective function in DRO is more difficult to
optimize, and most of the existing theoretical results make strong assumptions
on the loss function. In this work we bridge the gap by studying DRO algorithms
for general smooth non-convex losses. By carefully exploiting the specific form
of the DRO objective, we are able to provide non-asymptotic convergence
guarantees even though the objective function is possibly non-convex,
non-smooth and has unbounded gradient noise. In particular, we prove that a
special algorithm called the mini-batch normalized gradient descent with
momentum, can find an $\epsilon$ first-order stationary point within $O(
\epsilon^{-4} )$ gradient complexity. We also discuss the conditional
value-at-risk (CVaR) setting, where we propose a penalized DRO objective based
on a smoothed version of the CVaR that allows us to obtain a similar
convergence guarantee. We finally verify our theoretical results in a number of
tasks and find that the proposed algorithm can consistently achieve prominent
acceleration.

    

### [[2110.12786] Dictionary Learning Using Rank-One Atomic Decomposition (ROAD)](http://arxiv.org/abs/2110.12786)


  Dictionary learning aims at seeking a dictionary under which the training
data can be sparsely represented. Methods in the literature typically formulate
the dictionary learning problem as an optimization w.r.t. two variables, i.e.,
dictionary and sparse coefficients, and solve it by alternating between two
stages: sparse coding and dictionary update. The key contribution of this work
is a Rank-One Atomic Decomposition (ROAD) formulation where dictionary learning
is cast as an optimization w.r.t. a single variable which is a set of rank one
matrices. The resulting algorithm is hence single-stage. Compared with
two-stage algorithms, ROAD minimizes the sparsity of the coefficients whilst
keeping the data consistency constraint throughout the whole learning process.
An alternating direction method of multipliers (ADMM) is derived to solve the
optimization problem and the lower bound of the penalty parameter is computed
to guarantees a global convergence despite non-convexity of the optimization
formulation. From practical point of view, ROAD reduces the number of tuning
parameters required in other benchmark algorithms. Numerical tests demonstrate
that ROAD outperforms other benchmark algorithms for both synthetic data and
real data, especially when the number of training samples is small.

    

### [[2110.12899] No One Representation to Rule Them All: Overlapping Features of Training Methods](http://arxiv.org/abs/2110.12899)


  Despite being able to capture a range of features of the data, high accuracy
models trained with supervision tend to make similar predictions. This
seemingly implies that high-performing models share similar biases regardless
of training methodology, which would limit ensembling benefits and render
low-accuracy models as having little practical use. Against this backdrop,
recent work has made very different training techniques, such as large-scale
contrastive learning, yield competitively-high accuracy on generalization and
robustness benchmarks. This motivates us to revisit the assumption that models
necessarily learn similar functions. We conduct a large-scale empirical study
of models across hyper-parameters, architectures, frameworks, and datasets. We
find that model pairs that diverge more in training methodology display
categorically different generalization behavior, producing increasingly
uncorrelated errors. We show these models specialize in subdomains of the data,
leading to higher ensemble performance: with just 2 models (each with ImageNet
accuracy ~76.5%), we can create ensembles with 83.4% (+7% boost). Surprisingly,
we find that even significantly low-accuracy models can be used to improve
high-accuracy models. Finally, we show diverging training methodology yield
representations that capture overlapping (but not supersetting) feature sets
which, when combined, lead to increased downstream performance.

    

### [[2110.12911] Instance-Dependent Partial Label Learning](http://arxiv.org/abs/2110.12911)


  Partial label learning (PLL) is a typical weakly supervised learning problem,
where each training example is associated with a set of candidate labels among
which only one is true. Most existing PLL approaches assume that the incorrect
labels in each training example are randomly picked as the candidate labels.
However, this assumption is not realistic since the candidate labels are always
instance-dependent. In this paper, we consider instance-dependent PLL and
assume that each example is associated with a latent label distribution
constituted by the real number of each label, representing the degree to each
label describing the feature. The incorrect label with a high degree is more
likely to be annotated as the candidate label. Therefore, the latent label
distribution is the essential labeling information in partially labeled
examples and worth being leveraged for predictive model training. Motivated by
this consideration, we propose a novel PLL method that recovers the label
distribution as a label enhancement (LE) process and trains the predictive
model iteratively in every epoch. Specifically, we assume the true posterior
density of the latent label distribution takes on the variational approximate
Dirichlet density parameterized by an inference model. Then the evidence lower
bound is deduced for optimizing the inference model and the label distributions
generated from the variational posterior are utilized for training the
predictive model. Experiments on benchmark and real-world datasets validate the
effectiveness of the proposed method. Source code is available at
this https URL.

    

### [[2110.12985] Goal-Aware Cross-Entropy for Multi-Target Reinforcement Learning](http://arxiv.org/abs/2110.12985)


  Learning in a multi-target environment without prior knowledge about the
targets requires a large amount of samples and makes generalization difficult.
To solve this problem, it is important to be able to discriminate targets
through semantic understanding. In this paper, we propose goal-aware
cross-entropy (GACE) loss, that can be utilized in a self-supervised way using
auto-labeled goal states alongside reinforcement learning. Based on the loss,
we then devise goal-discriminative attention networks (GDAN) which utilize the
goal-relevant information to focus on the given instruction. We evaluate the
proposed methods on visual navigation and robot arm manipulation tasks with
multi-target environments and show that GDAN outperforms the state-of-the-art
methods in terms of task success ratio, sample efficiency, and generalization.
Additionally, qualitative analyses demonstrate that our proposed method can
help the agent become aware of and focus on the given instruction clearly,
promoting goal-directed behavior.

    

### [[2110.12997] Unsupervised Domain Adaptation with Dynamics-Aware Rewards in Reinforcement Learning](http://arxiv.org/abs/2110.12997)


  Unsupervised reinforcement learning aims to acquire skills without prior goal
representations, where an agent automatically explores an open-ended
environment to represent goals and learn the goal-conditioned policy. However,
this procedure is often time-consuming, limiting the rollout in some
potentially expensive target environments. The intuitive approach of training
in another interaction-rich environment disrupts the reproducibility of trained
skills in the target environment due to the dynamics shifts and thus inhibits
direct transferring. Assuming free access to a source environment, we propose
an unsupervised domain adaptation method to identify and acquire skills across
dynamics. Particularly, we introduce a KL regularized objective to encourage
emergence of skills, rewarding the agent for both discovering skills and
aligning its behaviors respecting dynamics shifts. This suggests that both
dynamics (source and target) shape the reward to facilitate the learning of
adaptive skills. We also conduct empirical experiments to demonstrate that our
method can effectively learn skills that can be smoothly deployed in target.

    

### [[2110.13006] Gradient-based Quadratic Multiform Separation](http://arxiv.org/abs/2110.13006)


  Classification as a supervised learning concept is an important content in
machine learning. It aims at categorizing a set of data into classes. There are
several commonly-used classification methods nowadays such as k-nearest
neighbors, random forest, and support vector machine. Each of them has its own
pros and cons, and none of them is invincible for all kinds of problems. In
this thesis, we focus on Quadratic Multiform Separation (QMS), a classification
method recently proposed by Michael Fan et al. (2019). Its fresh concept, rich
mathematical structure, and innovative definition of loss function set it apart
from the existing classification methods. Inspired by QMS, we propose utilizing
a gradient-based optimization method, Adam, to obtain a classifier that
minimizes the QMS-specific loss function. In addition, we provide suggestions
regarding model tuning through explorations of the relationships between
hyperparameters and accuracies. Our empirical result shows that QMS performs as
good as most classification methods in terms of accuracy. Its superior
performance is almost comparable to those of gradient boosting algorithms that
win massive machine learning competitions.

    

### [[2010.08262] Local plasticity rules can learn deep representations using self-supervised contrastive predictions](http://arxiv.org/abs/2010.08262)


  Learning in the brain is poorly understood and learning rules that respect
biological constraints, yet yield deep hierarchical representations, are still
unknown. Here, we propose a learning rule that takes inspiration from
neuroscience and recent advances in self-supervised deep learning. Learning
minimizes a simple layer-specific loss function and does not need to
back-propagate error signals within or between layers. Instead, weight updates
follow a local, Hebbian, learning rule that only depends on pre- and
post-synaptic neuronal activity, predictive dendritic input and widely
broadcasted modulation factors which are identical for large groups of neurons.
The learning rule applies contrastive predictive learning to a causal,
biological setting using saccades (i.e. rapid shifts in gaze direction). We
find that networks trained with this self-supervised and local rule build deep
hierarchical representations of images, speech and video.

    

### [[2106.01862] Self-Supervised Learning of Event-Based Optical Flow with Spiking Neural Networks](http://arxiv.org/abs/2106.01862)


  The field of neuromorphic computing promises extremely low-power and
low-latency sensing and processing. Challenges in transferring learning
algorithms from traditional artificial neural networks (ANNs) to spiking neural
networks (SNNs) have so far prevented their application to large-scale, complex
regression tasks. Furthermore, realizing a truly asynchronous and fully
neuromorphic pipeline that maximally attains the abovementioned benefits
involves rethinking the way in which this pipeline takes in and accumulates
information. In the case of perception, spikes would be passed as-is and
one-by-one between an event camera and an SNN, meaning all temporal integration
of information must happen inside the network. In this article, we tackle these
two problems. We focus on the complex task of learning to estimate optical flow
from event-based camera inputs in a self-supervised manner, and modify the
state-of-the-art ANN training pipeline to encode minimal temporal information
in its inputs. Moreover, we reformulate the self-supervised loss function for
event-based optical flow to improve its convexity. We perform experiments with
various types of recurrent ANNs and SNNs using the proposed pipeline.
Concerning SNNs, we investigate the effects of elements such as parameter
initialization and optimization, surrogate gradient shape, and adaptive
neuronal mechanisms. We find that initialization and surrogate gradient width
play a crucial part in enabling learning with sparse inputs, while the
inclusion of adaptivity and learnable neuronal parameters can improve
performance. We show that the performance of the proposed ANNs and SNNs are on
par with that of the current state-of-the-art ANNs trained in a self-supervised
manner.

    

### [[2110.13407] VLSI Implementation of Cryptographic Algorithms & Techniques: A Literature Review](http://arxiv.org/abs/2110.13407)


  Through the years, the flow of Data and its transmission have increased
tremendously and so has the security issues to it. Cryptography in recent years
with the advancement of VLSI has led to its implementation of Encryption and
Decryption techniques, where the process of translating and converting
plaintext into cypher text and vice versa is made possible. In this paper, the
review of various aspects of VLSI's implementation of encryption and decryption
are covered. To systemize the material, the information about topics such as
Private Key Encryption, Index Technique, Blowfish Algorithm, DNA cryptography,
and many more are reviewed. Ultimately, with this review, the basic
understanding of different VLSI techniques of Encryption and Decryption can be
studied and implemented.

    

### [[2110.13187] Data intensive physics analysis in Azure cloud](http://arxiv.org/abs/2110.13187)


  The Compact Muon Solenoid (CMS) experiment at the Large Hadron Collider (LHC)
is one of the largest data producers in the scientific world, with standard
data products centrally produced, and then used by often competing teams within
the collaboration. This work is focused on how a local institution, University
of California San Diego (UCSD), partnered with the Open Science Grid (OSG) to
use Azure cloud resources to augment its available computing to accelerate time
to results for multiple analyses pursued by a small group of collaborators. The
OSG is a federated infrastructure allowing many independent resource providers
to serve many independent user communities in a transparent manner.
Historically the resources would come from various research institutions,
spanning small universities to large HPC centers, based on either community
needs or grant allocations, so adding commercial clouds as resource providers
is a natural evolution. The OSG technology allows for easy integration of cloud
resources, but the data-intensive nature of CMS compute jobs required the
deployment of additional data caching infrastructure to ensure high efficiency.

    

### [[2110.13234] Let's Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud](http://arxiv.org/abs/2110.13234)


  Depending on energy sources and demand, the carbon intensity of the public
power grid fluctuates over time. Exploiting this variability is an important
factor in reducing the emissions caused by data centers. However, regional
differences in the availability of low-carbon energy sources make it hard to
provide general best practices for when to consume electricity. Moreover,
existing research in this domain focuses mostly on carbon-aware workload
migration across geo-distributed data centers, or addresses demand response
purely from the perspective of power grid stability and costs.
In this paper, we examine the potential impact of shifting computational
workloads towards times where the energy supply is expected to be less
carbon-intensive. To this end, we identify characteristics of delay-tolerant
workloads and analyze the potential for temporal workload shifting in Germany,
Great Britain, France, and California over the year 2020. Furthermore, we
experimentally evaluate two workload shifting scenarios in a simulation to
investigate the influence of time constraints, scheduling strategies, and the
accuracy of carbon intensity forecasts. To accelerate research in the domain of
carbon-aware computing and to support the evaluation of novel scheduling
algorithms, our simulation framework and datasets are publicly available.

    

### [[2110.13368] OpenACC Acceleration of an Agent-Based Biological Simulation Framework](http://arxiv.org/abs/2110.13368)


  Computational biology has increasingly turned to agent-based modeling to
explore complex biological systems. Biological diffusion (diffusion, decay,
secretion, and uptake) is a key driver of biological tissues. GPU computing can
vastly accelerate the diffusion and decay operators in the partial differential
equations used to represent biological transport in an agent-based biological
modeling system. In this paper, we utilize OpenACC to accelerate the diffusion
portion of PhysiCell, a cross-platform agent-based biosimulation framework. We
demonstrate an almost 40x speedup on the state-of-the-art NVIDIA A100 GPU
compared to a serial run on AMD's EPYC 7742. We also demonstrate 9x speedup on
the 64 core AMD EPYC 7742 multicore platform. By using OpenACC for both the
CPUs and the GPUs, we maintain a single source code base, thus creating a
portable yet performant solution. With the simulator's most significant
computational bottleneck significantly reduced, we can continue cancer
simulations over much longer times.

    

### [[2110.13551] BuffetFS: Serve Yourself Permission Checks without Remote Procedure Calls](http://arxiv.org/abs/2110.13551)


  The remote procedure call (a.k.a. RPC) latency becomes increasingly
significant in a distributed file system. We propose BuffetFS, a user-level
file system that optimizes I/O performance by eliminating the RPCs caused by
\texttt{open()} operation. By leveraging \texttt{open()} from file servers to
clients, BuffetFS can restrain the procedure calls for permission checks
locally, hence avoid RPCs during the initial stage to access a file. BuffetFS
can further reduce response time when users are accessing a large number of
small files. We implement a BuffetFS prototype and integrate it into a storage
cluster. Our preliminary evaluation results show that BuffetFS can offer up to
70\% performance gain compared to the Lustre file system.

    

### [[2110.13693] A proposed method using GPU based SDO to optimize retail warehouses](http://arxiv.org/abs/2110.13693)


  Research in warehouse optimization has gotten increased attention in the last
few years due to e-commerce. The warehouse contains a waste range of different
products. Due to the nature of the individual order, it is challenging to plan
the picking list to optimize the material flow in the process. There are also
challenges in minimizing costs and increasing production capacity, and this
complexity can be defined as a multidisciplinary optimization problem with an
IDF nature. In recent years the use of parallel computing using GPGPUs has
become increasingly popular due to the introduction of CUDA C and accompanying
applications in, e.g., Python. In the case study at the company in the field of
retail, a case study including a system design optimization (SDO) resulted in
an increase in throughput with well over 20% just by clustering different
categories and suggesting in which sequence the orders should be picked during
a given time frame. The options provided by implementing a distributed
high-performance computing network based on GPUs for subsystem optimization
have shown to be fruitful in developing a functioning SDO for warehouse
optimization. The toolchain can be used for designing new warehouses or
evaluating and tuning existing ones.

    

### [[2103.07450] Reaching Agreement in Competitive Microbial Systems](http://arxiv.org/abs/2103.07450)


  In this work, we consider distributed agreement tasks in microbial
distributed systems under stochastic population dynamics and competitive
interactions. We examine how competitive exclusion can be used to solve
distributed agreement tasks in the microbial setting. To this end, we develop a
new technique for analyzing the time to reach competitive exclusion in systems
with several competing species under biologically realistic population
dynamics. We use this technique to analyze a protocol that exploits competitive
interactions to solve approximate majority consensus efficiently in synthetic
microbial systems.
We show that direct competition dynamics reach majority consensus with high
probability when the initial gap between the species is small, i.e.,
$\Omega(\sqrt{n \log n})$, where $n$ is the initial population size of the
majority species. In contrast, we show that indirect competition alone is not
efficient: for example, solving majority consensus with high probability
requires an initial gap of $\Omega(n)$. To corroborate our analytical results,
we use computer simulations to show that these consensus dynamics occur within
practical time scales.

    

### [[2110.13264] Memory visualization tool for training neural network](http://arxiv.org/abs/2110.13264)


  Software developed helps world a better place ranging from system software,
open source, application software and so on. Software engineering does have
neural network models applied to code suggestion, bug report summarizing and so
on to demonstrate their effectiveness at a real SE task. Software and machine
learning algorithms combine to make software give better solutions and
understanding of environment. In software, there are both generalized
applications which helps solve problems for entire world and also some specific
applications which helps one particular community. To address the computational
challenge in deep learning, many tools exploit hardware features such as
multi-core CPUs and many-core GPUs to shorten the training time. Machine
learning algorithms have a greater impact in the world but there is a
considerable amount of memory utilization during the process. We propose a new
tool for analysis of memory utilized for developing and training deep learning
models. Our tool results in visual utilization of memory concurrently. Various
parameters affecting the memory utilization are analysed while training. This
tool helps in knowing better idea of processes or models which consumes more
memory.

    

### [[2110.13309] History Aware Multimodal Transformer for Vision-and-Language Navigation](http://arxiv.org/abs/2110.13309)


  Vision-and-language navigation (VLN) aims to build autonomous visual agents
that follow instructions and navigate in real scenes. To remember previously
visited locations and actions taken, most approaches to VLN implement memory
using recurrent states. Instead, we introduce a History Aware Multimodal
Transformer (HAMT) to incorporate a long-horizon history into multimodal
decision making. HAMT efficiently encodes all the past panoramic observations
via a hierarchical vision transformer (ViT), which first encodes individual
images with ViT, then models spatial relation between images in a panoramic
observation and finally takes into account temporal relation between panoramas
in the history. It, then, jointly combines text, history and current
observation to predict the next action. We first train HAMT end-to-end using
several proxy tasks including single step action prediction and spatial
relation prediction, and then use reinforcement learning to further improve the
navigation policy. HAMT achieves new state of the art on a broad range of VLN
tasks, including VLN with fine-grained instructions (R2R, RxR), high-level
instructions (R2R-Last, REVERIE), dialogs (CVDN) as well as long-horizon VLN
(R4R, R2R-Back). We demonstrate HAMT to be particularly effective for
navigation tasks with longer trajectories.

    

### [[2110.13341] How Should AI Interpret Rules? A Defense of Minimally Defeasible Interpretive Argumentation](http://arxiv.org/abs/2110.13341)


  Can artificially intelligent systems follow rules? The answer might seem an
obvious `yes', in the sense that all (current) AI strictly acts in accordance
with programming code constructed from highly formalized and well-defined
rulesets. But here I refer to the kinds of rules expressed in human language
that are the basis of laws, regulations, codes of conduct, ethical guidelines,
and so on. The ability to follow such rules, and to reason about them, is not
nearly as clear-cut as it seems on first analysis. Real-world rules are
unavoidably rife with open-textured terms, which imbue rules with a possibly
infinite set of possible interpretations. Narrowing down this set requires a
complex reasoning process that is not yet within the scope of contemporary AI.
This poses a serious problem for autonomous AI: If one cannot reason about
open-textured terms, then one cannot reason about (or in accordance with)
real-world rules. And if one cannot reason about real-world rules, then one
cannot: follow human laws, comply with regulations, act in accordance with
written agreements, or even obey mission-specific commands that are anything
more than trivial. But before tackling these problems, we must first answer a
more fundamental question: Given an open-textured rule, what is its correct
interpretation? Or more precisely: How should our artificially intelligent
systems determine which interpretation to consider correct? In this essay, I
defend the following answer: Rule-following AI should act in accordance with
the interpretation best supported by minimally defeasible interpretive
arguments (MDIA).

    

### [[2110.13386] Self-Denoising Neural Networks for Few Shot Learning](http://arxiv.org/abs/2110.13386)


  In this paper, we introduce a new architecture for few shot learning, the
task of teaching a neural network from as few as one or five labeled examples.
Inspired by the theoretical results of Alaine et al that Denoising Autoencoders
refine features to lie closer to the true data manifold, we present a new
training scheme that adds noise at multiple stages of an existing neural
architecture while simultaneously learning to be robust to this added noise.
This architecture, which we call a Self-Denoising Neural Network (SDNN), can be
applied easily to most modern convolutional neural architectures, and can be
used as a supplement to many existing few-shot learning techniques. We
empirically show that SDNNs out-perform previous state-of-the-art methods for
few shot image recognition using the Wide-ResNet architecture on the
\textit{mini}ImageNet, tiered-ImageNet, and CIFAR-FS few shot learning
datasets. We also perform a series of ablation experiments to empirically
justify the construction of the SDNN architecture. Finally, we show that SDNNs
even improve few shot performance on the task of human action detection in
video using experiments on the ActEV SDL Surprise Activities challenge.

    

### [[2110.13395] Transferring Domain-Agnostic Knowledge in Video Question Answering](http://arxiv.org/abs/2110.13395)


  Video question answering (VideoQA) is designed to answer a given question
based on a relevant video clip. The current available large-scale datasets have
made it possible to formulate VideoQA as the joint understanding of visual and
language information. However, this training procedure is costly and still less
competent with human performance. In this paper, we investigate a transfer
learning method by the introduction of domain-agnostic knowledge and
domain-specific knowledge. First, we develop a novel transfer learning
framework, which finetunes the pre-trained model by applying domain-agnostic
knowledge as the medium. Second, we construct a new VideoQA dataset with 21,412
human-generated question-answer samples for comparable transfer of knowledge.
Our experiments show that: (i) domain-agnostic knowledge is transferable and
(ii) our proposed transfer learning framework can boost VideoQA performance
effectively.

    

### [[2110.13398] Unified Instance and Knowledge Alignment Pretraining for Aspect-based Sentiment Analysis](http://arxiv.org/abs/2110.13398)


  Aspect-based Sentiment Analysis (ABSA) aims to determine the sentiment
polarity towards an aspect. Because of the expensive and limited labelled data,
the pretraining strategy has become the de-facto standard for ABSA. However,
there always exists severe domain shift between the pretraining and downstream
ABSA datasets, hindering the effective knowledge transfer when directly
finetuning and making the downstream task performs sub-optimal. To mitigate
such domain shift, we introduce a unified alignment pretraining framework into
the vanilla pretrain-finetune pipeline with both instance- and knowledge-level
alignments. Specifically, we first devise a novel coarse-to-fine retrieval
sampling approach to select target domain-related instances from the
large-scale pretraining dataset, thus aligning the instances between
pretraining and target domains (\textit{First Stage}). Then, we introduce a
knowledge guidance-based strategy to further bridge the domain gap at the
knowledge level. In practice, we formulate the model pretrained on the sampled
instances into a knowledge guidance model and a learner model, respectively. On
the target dataset, we design an on-the-fly teacher-student joint fine-tuning
approach to progressively transfer the knowledge from the knowledge guidance
model to the learner model (\textit{Second Stage}). Thereby, the learner model
can maintain more domain-invariant knowledge when learning new knowledge from
the target dataset. In the \textit{Third Stage,} the learner model is finetuned
to better adapt its learned knowledge to the target dataset. Extensive
experiments and analyses on several ABSA benchmarks demonstrate the
effectiveness and universality of our proposed pretraining framework. Notably,
our pretraining framework pushes several strong baseline models up to the new
state-of-the-art records. We release our code and models.

    

### [[2110.13409] Task-Aware Meta Learning-based Siamese Neural Network for Classifying Obfuscated Malware](http://arxiv.org/abs/2110.13409)


  Malware authors apply different obfuscation techniques on the generic feature
of malware (i.e., unique malware signature) to create new variants to avoid
detection. Existing Siamese Neural Network (SNN) based malware detection
methods fail to correctly classify different malware families when similar
generic features are shared across multiple malware variants resulting in high
false-positive rates. To address this issue, we propose a novel Task-Aware Meta
Learning-based Siamese Neural Network resilient against obfuscated malware
while able to detect malware trained with one or a few training samples. Using
entropy features of each malware signature alongside image features as task
inputs, our task-aware meta leaner generates the parameters for the feature
layers to more accurately adjust the feature embedding for different malware
families. In addition, our model utilizes meta-learning with the extracted
features of a pre-trained network (e.g., VGG-16) to avoid the bias typically
associated with a model trained with a limited number of training samples. Our
proposed approach is highly effective in recognizing unique malware signatures,
thus correctly classifying malware samples that belong to the same malware
family even in the presence of obfuscation technique applied to malware. Our
experimental results, validated with N-way on N-shot learning, show that our
model is highly effective in classification accuracy exceeding the rate>91%
compared to other similar methods.

    

### [[2110.13424] Precise URL Phishing Detection Using Neural Networks](http://arxiv.org/abs/2110.13424)


  With the development of the Internet, ways of obtaining important data such
as passwords and logins or sensitive personal data have increased. One of the
ways to extract such information is page impersonation, also called phishing.
Such websites do not provide service but collect sensitive details from the
user. Here, we present you with ways to detect such malicious URLs with state
of art accuracy with neural networks. Different from previous works, where web
content, URL or traffic statistics are examined, we analyse only the URL text,
making it faster and which detects zero-day attacks. The network is optimised
and can be used even on small devices such as Ras-Pi without a change in
performance.

    

### [[2110.13470] Subject Adaptive EEG-based Visual Recognition](http://arxiv.org/abs/2110.13470)


  This paper focuses on EEG-based visual recognition, aiming to predict the
visual object class observed by a subject based on his/her EEG signals. One of
the main challenges is the large variation between signals from different
subjects. It limits recognition systems to work only for the subjects involved
in model training, which is undesirable for real-world scenarios where new
subjects are frequently added. This limitation can be alleviated by collecting
a large amount of data for each new user, yet it is costly and sometimes
infeasible. To make the task more practical, we introduce a novel problem
setting, namely subject adaptive EEG-based visual recognition. In this setting,
a bunch of pre-recorded data of existing users (source) is available, while
only a little training data from a new user (target) are provided. At inference
time, the model is evaluated solely on the signals from the target user. This
setting is challenging, especially because training samples from source
subjects may not be helpful when evaluating the model on the data from the
target subject. To tackle the new problem, we design a simple yet effective
baseline that minimizes the discrepancy between feature distributions from
different subjects, which allows the model to extract subject-independent
features. Consequently, our model can learn the common knowledge shared among
subjects, thereby significantly improving the recognition performance for the
target subject. In the experiments, we demonstrate the effectiveness of our
method under various settings. Our code is available at
this https URL.

    

### [[2110.13473] CTRN: Class-Temporal Relational Network for Action Detection](http://arxiv.org/abs/2110.13473)


  Action detection is an essential and challenging task, especially for densely
labelled datasets of untrimmed videos. There are many real-world challenges in
those datasets, such as composite action, co-occurring action, and high
temporal variation of instance duration. For handling these challenges, we
propose to explore both the class and temporal relations of detected actions.
In this work, we introduce an end-to-end network: Class-Temporal Relational
Network (CTRN). It contains three key components: (1) The Representation
Transform Module filters the class-specific features from the mixed
representations to build graph-structured data. (2) The Class-Temporal Module
models the class and temporal relations in a sequential manner. (3)
G-classifier leverages the privileged knowledge of the snippet-wise
co-occurring action pairs to further improve the co-occurring action detection.
We evaluate CTRN on three challenging densely labelled datasets and achieve
state-of-the-art performance, reflecting the effectiveness and robustness of
our method.

    

### [[2110.13575] Automated Support for Unit Test Generation: A Tutorial Book Chapter](http://arxiv.org/abs/2110.13575)


  Unit testing is a stage of testing where the smallest segment of code that
can be tested in isolation from the rest of the system - often a class - is
tested. Unit tests are typically written as executable code, often in a format
provided by a unit testing framework such as pytest for Python.
Creating unit tests is a time and effort-intensive process with many
repetitive, manual elements. To illustrate how AI can support unit testing,
this chapter introduces the concept of search-based unit test generation. This
technique frames the selection of test input as an optimization problem - we
seek a set of test cases that meet some measurable goal of a tester - and
unleashes powerful metaheuristic search algorithms to identify the best
possible test cases within a restricted timeframe. This chapter introduces two
algorithms that can generate pytest-formatted unit tests, tuned towards
coverage of source code statements. The chapter concludes by discussing more
advanced concepts and gives pointers to further reading for how artificial
intelligence can support developers and testers when unit testing software.

    

### [[2110.13606] AUTO-DISCERN: Autonomous Driving Using Common Sense Reasoning](http://arxiv.org/abs/2110.13606)


  Driving an automobile involves the tasks of observing surroundings, then
making a driving decision based on these observations (steer, brake, coast,
etc.). In autonomous driving, all these tasks have to be automated. Autonomous
driving technology thus far has relied primarily on machine learning
techniques. We argue that appropriate technology should be used for the
appropriate task. That is, while machine learning technology is good for
observing and automatically understanding the surroundings of an automobile,
driving decisions are better automated via commonsense reasoning rather than
machine learning. In this paper, we discuss (i) how commonsense reasoning can
be automated using answer set programming (ASP) and the goal-directed s(CASP)
ASP system, and (ii) develop the AUTO-DISCERN system using this technology for
automating decision-making in driving. The goal of our research, described in
this paper, is to develop an autonomous driving system that works by simulating
the mind of a human driver. Since driving decisions are based on human-style
reasoning, they are explainable, their ethics can be ensured, and they will
always be correct, provided the system modeling and system inputs are correct.

    

### [[2110.13608] Using Traceless Genetic Programming for Solving Multiobjective Optimization Problems](http://arxiv.org/abs/2110.13608)


  Traceless Genetic Programming (TGP) is a Genetic Programming (GP) variant
that is used in cases where the focus is rather the output of the program than
the program itself. The main difference between TGP and other GP techniques is
that TGP does not explicitly store the evolved computer programs. Two genetic
operators are used in conjunction with TGP: crossover and insertion. In this
paper, we shall focus on how to apply TGP for solving multi-objective
optimization problems which are quite unusual for GP. Each TGP individual
stores the output of a computer program (tree) representing a point in the
search space. Numerical experiments show that TGP is able to solve very fast
and very well the considered test problems.

    

### [[2110.13609] Resolving Anomalies in the Behaviour of a Modularity Inducing Problem Domain with Distributional Fitness Evaluation](http://arxiv.org/abs/2110.13609)


  Discrete gene regulatory networks (GRNs) play a vital role in the study of
robustness and modularity. A common method of evaluating the robustness of GRNs
is to measure their ability to regulate a set of perturbed gene activation
patterns back to their unperturbed forms. Usually, perturbations are obtained
by collecting random samples produced by a predefined distribution of gene
activation patterns. This sampling method introduces stochasticity, in turn
inducing dynamicity. This dynamicity is imposed on top of an already complex
fitness landscape. So where sampling is used, it is important to understand
which effects arise from the structure of the fitness landscape, and which
arise from the dynamicity imposed on it. Stochasticity of the fitness function
also causes difficulties in reproducibility and in post-experimental analyses.
We develop a deterministic distributional fitness evaluation by considering
the complete distribution of gene activity patterns, so as to avoid
stochasticity in fitness assessment. This fitness evaluation facilitates
repeatability. Its determinism permits us to ascertain theoretical bounds on
the fitness, and thus to identify whether the algorithm has reached a global
optimum. It enables us to differentiate the effects of the problem domain from
those of the noisy fitness evaluation, and thus to resolve two remaining
anomalies in the behaviour of the problem domain
of~\citet{espinosa2010specialization}. We also reveal some properties of
solution GRNs that lead them to be robust and modular, leading to a deeper
understanding of the nature of the problem domain. We conclude by discussing
potential directions toward simulating and understanding the emergence of
modularity in larger, more complex domains, which is key both to generating
more useful modular solutions, and to understanding the ubiquity of modularity
in biological systems.

    

### [[2110.13665] Bootstrapping Concept Formation in Small Neural Networks](http://arxiv.org/abs/2110.13665)


  The question how neural systems (of humans) can perform reasoning is still
far from being solved. We posit that the process of forming Concepts is a
fundamental step required for this. We argue that, first, Concepts are formed
as closed representations, which are then consolidated by relating them to each
other. Here we present a model system (agent) with a small neural network that
uses realistic learning rules and receives only feedback from the environment
in which the agent performs virtual actions. First, the actions of the agent
are reflexive. In the process of learning, statistical regularities in the
input lead to the formation of neuronal pools representing relations between
the entities observed by the agent from its artificial world. This information
then influences the behavior of the agent via feedback connections replacing
the initial reflex by an action driven by these relational representations. We
hypothesize that the neuronal pools representing relational information can be
considered as primordial Concepts, which may in a similar way be present in
some pre-linguistic animals, too. We argue that systems such as this can help
formalizing the discussion about what constitutes Concepts and serve as a
starting point for constructing artificial cogitating systems.

    

### [[2110.13677] A Personalized Diagnostic Generation Framework Based on Multi-source Heterogeneous Data](http://arxiv.org/abs/2110.13677)


  Personalized diagnoses have not been possible due to sear amount of data
pathologists have to bear during the day-to-day routine. This lead to the
current generalized standards that are being continuously updated as new
findings are reported. It is noticeable that these effective standards are
developed based on a multi-source heterogeneous data, including whole-slide
images and pathology and clinical reports. In this study, we propose a
framework that combines pathological images and medical reports to generate a
personalized diagnosis result for individual patient. We use nuclei-level image
feature similarity and content-based deep learning method to search for a
personalized group of population with similar pathological characteristics,
extract structured prognostic information from descriptive pathology reports of
the similar patient population, and assign importance of different prognostic
factors to generate a personalized pathological diagnosis result. We use
multi-source heterogeneous data from TCGA (The Cancer Genome Atlas) database.
The result demonstrate that our framework matches the performance of
pathologists in the diagnosis of renal cell carcinoma. This framework is
designed to be generic, thus could be applied for other types of cancer. The
weights could provide insights to the known prognostic factors and further
guide more precise clinical treatment protocols.

    

### [[2110.13683] BioIE: Biomedical Information Extraction with Multi-head Attention Enhanced Graph Convolutional Network](http://arxiv.org/abs/2110.13683)


  Constructing large-scaled medical knowledge graphs can significantly boost
healthcare applications for medical surveillance, bring much attention from
recent research. An essential step in constructing large-scale MKG is
extracting information from medical reports. Recently, information extraction
techniques have been proposed and show promising performance in biomedical
information extraction. However, these methods only consider limited types of
entity and relation due to the noisy biomedical text data with complex entity
correlations. Thus, they fail to provide enough information for constructing
MKGs and restrict the downstream applications. To address this issue, we
propose Biomedical Information Extraction, a hybrid neural network to extract
relations from biomedical text and unstructured medical reports. Our model
utilizes a multi-head attention enhanced graph convolutional network to capture
the complex relations and context information while resisting the noise from
the data. We evaluate our model on two major biomedical relationship extraction
tasks, chemical-disease relation and chemical-protein interaction, and a
cross-hospital pan-cancer pathology report corpus. The results show that our
method achieves superior performance than baselines. Furthermore, we evaluate
the applicability of our method under a transfer learning setting and show that
BioIE achieves promising performance in processing medical text from different
formats and writing styles.

    

### [[2110.13691] An Explicit-Joint and Supervised-Contrastive Learning Framework for Few-Shot Intent Classification and Slot Filling](http://arxiv.org/abs/2110.13691)


  Intent classification (IC) and slot filling (SF) are critical building blocks
in task-oriented dialogue systems. These two tasks are closely-related and can
flourish each other. Since only a few utterances can be utilized for
identifying fast-emerging new intents and slots, data scarcity issue often
occurs when implementing IC and SF. However, few IC/SF models perform well when
the number of training samples per class is quite small. In this paper, we
propose a novel explicit-joint and supervised-contrastive learning framework
for few-shot intent classification and slot filling. Its highlights are as
follows. (i) The model extracts intent and slot representations via
bidirectional interactions, and extends prototypical network to achieve
explicit-joint learning, which guarantees that IC and SF tasks can mutually
reinforce each other. (ii) The model integrates with supervised contrastive
learning, which ensures that samples from same class are pulled together and
samples from different classes are pushed apart. In addition, the model follows
a not common but practical way to construct the episode, which gets rid of the
traditional setting with fixed way and shot, and allows for unbalanced
datasets. Extensive experiments on three public datasets show that our model
can achieve promising performance.

    

### [[2110.13710] DASentimental: Detecting depression, anxiety and stress in texts via emotional recall, cognitive networks and machine learning](http://arxiv.org/abs/2110.13710)


  Most current affect scales and sentiment analysis on written text focus on
quantifying valence (sentiment) -- the most primary dimension of emotion.
However, emotions are broader and more complex than valence. Distinguishing
negative emotions of similar valence could be important in contexts such as
mental health. This project proposes a semi-supervised machine learning model
(DASentimental) to extract depression, anxiety and stress from written text.
First, we trained the model to spot how sequences of recalled emotion words by
$N=200$ individuals correlated with their responses to the Depression Anxiety
Stress Scale (DASS-21). Within the framework of cognitive network science, we
model every list of recalled emotions as a walk over a networked mental
representation of semantic memory, with emotions connected according to free
associations in people's memory. Among several tested machine learning
approaches, we find that a multilayer perceptron neural network trained on word
sequences and semantic network distances can achieve state-of-art,
cross-validated predictions for depression ($R = 0.7$), anxiety ($R = 0.44$)
and stress ($R = 0.52$). Though limited by sample size, this first-of-its-kind
approach enables quantitative explorations of key semantic dimensions behind
DAS levels. We find that semantic distances between recalled emotions and the
dyad "sad-happy" are crucial features for estimating depression levels but are
less important for anxiety and stress. We also find that semantic distance of
recalls from "fear" can boost the prediction of anxiety but it becomes
redundant when the "sad-happy" dyad is considered. Adopting DASentimental as a
semi-supervised learning tool to estimate DAS in text, we apply it to a dataset
of 142 suicide notes. We conclude by discussing key directions for future
research enabled by artificial intelligence detecting stress, anxiety and
depression.

    

### [[2110.13715] ConE: Cone Embeddings for Multi-Hop Reasoning over Knowledge Graphs](http://arxiv.org/abs/2110.13715)


  Query embedding (QE) -- which aims to embed entities and first-order logical
(FOL) queries in low-dimensional spaces -- has shown great power in multi-hop
reasoning over knowledge graphs. Recently, embedding entities and queries with
geometric shapes becomes a promising direction, as geometric shapes can
naturally represent answer sets of queries and logical relationships among
them. However, existing geometry-based models have difficulty in modeling
queries with negation, which significantly limits their applicability. To
address this challenge, we propose a novel query embedding model, namely Cone
Embeddings (ConE), which is the first geometry-based QE model that can handle
all the FOL operations, including conjunction, disjunction, and negation.
Specifically, ConE represents entities and queries as Cartesian products of
two-dimensional cones, where the intersection and union of cones naturally
model the conjunction and disjunction operations. By further noticing that the
closure of complement of cones remains cones, we design geometric complement
operators in the embedding space for the negation operations. Experiments
demonstrate that ConE significantly outperforms existing state-of-the-art
methods on benchmark datasets.

    

### [[2102.01951] Mind the Gap: Assessing Temporal Generalization in Neural Language Models](http://arxiv.org/abs/2102.01951)


  Our world is open-ended, non-stationary, and constantly evolving; thus what
we talk about and how we talk about it change over time. This inherent dynamic
nature of language contrasts with the current static language modelling
paradigm, which trains and evaluates models on utterances from overlapping time
periods. Despite impressive recent progress, we demonstrate that Transformer-XL
language models perform worse in the realistic setup of predicting future
utterances from beyond their training period, and that model performance
becomes increasingly worse with time. We find that, while increasing model size
alone -- a key driver behind recent progress -- does not solve this problem,
having models that continually update their knowledge with new information can
indeed mitigate this performance degradation over time. Hence, given the
compilation of ever-larger language modelling datasets, combined with the
growing list of language-model-based NLP applications that require up-to-date
factual knowledge about the world, we argue that now is the right time to
rethink the static way in which we currently train and evaluate our language
models, and develop adaptive language models that can remain up-to-date with
respect to our ever-changing and non-stationary world. We publicly release our
dynamic, streaming language modelling benchmarks for WMT and arXiv to
facilitate language model evaluation that takes temporal dynamics into account.

    

### [[2103.05154] Explanations in Autonomous Driving: A Survey](http://arxiv.org/abs/2103.05154)


  The automotive industry has witnessed an increasing level of development in
the past decades; from manufacturing manually operated vehicles to
manufacturing vehicles with a high level of automation. With the recent
developments in Artificial Intelligence (AI), automotive companies now employ
blackbox AI models to enable vehicles to perceive their environments and make
driving decisions with little or no input from a human. With the hope to deploy
autonomous vehicles (AV) on a commercial scale, the acceptance of AV by society
becomes paramount and may largely depend on their degree of transparency,
trustworthiness, and compliance with regulations. The assessment of the
compliance of AVs to these acceptance requirements can be facilitated through
the provision of explanations for AVs' behaviour. Explainability is therefore
seen as an important requirement for AVs. AVs should be able to explain what
they have 'seen', done, and might do in environments in which they operate.
In this paper, we provide a comprehensive survey of the existing body of work
around explainable autonomous driving. First, we open with a motivation for
explanations by highlighting and emphasising the importance of transparency,
accountability, and trust in AVs; and examining existing regulations and
standards related to AVs. Second, we identify and categorise the different
stakeholders involved in the development, use, and regulation of AVs and elicit
their explanation requirements for AV. Third, we provide a rigorous review of
previous work on explanations for the different AV operations (i.e.,
perception, localisation, planning, control, and system management). Finally,
we identify pertinent challenges and provide recommendations, such as a
conceptual framework for AV explainability. This survey aims to provide the
fundamental knowledge required of researchers who are interested in
explainability in AVs.

    

### [[2103.12028] Quality at a Glance: An Audit of Web-Crawled Multilingual Datasets](http://arxiv.org/abs/2103.12028)


  With the success of large-scale pre-training and multilingual modeling in
Natural Language Processing (NLP), recent years have seen a proliferation of
large, web-mined text datasets covering hundreds of languages. We manually
audit the quality of 205 language-specific corpora released with five major
public datasets (CCAligned, ParaCrawl, WikiMatrix, OSCAR, mC4). Lower-resource
corpora have systematic issues: At least 15 corpora have no usable text, and a
significant fraction contains less than 50% sentences of acceptable quality. In
addition, many are mislabeled or use nonstandard/ambiguous language codes. We
demonstrate that these issues are easy to detect even for non-proficient
speakers, and supplement the human audit with automatic analyses. Finally, we
recommend techniques to evaluate and improve multilingual corpora and discuss
potential risks that come with low-quality data releases.

    

### [[2103.14712] Generating and Evaluating Explanations of Attended and Error-Inducing Input Regions for VQA Models](http://arxiv.org/abs/2103.14712)


  Attention maps, a popular heatmap-based explanation method for Visual
Question Answering (VQA), are supposed to help users understand the model by
highlighting portions of the image/question used by the model to infer answers.
However, we see that users are often misled by current attention map
visualizations that point to relevant regions despite the model producing an
incorrect answer. Hence, we propose Error Maps that clarify the error by
highlighting image regions where the model is prone to err. Error maps can
indicate when a correctly attended region may be processed incorrectly leading
to an incorrect answer, and hence, improve users' understanding of those cases.
To evaluate our new explanations, we further introduce a metric that simulates
users' interpretation of explanations to evaluate their potential helpfulness
to understand model correctness. We finally conduct user studies to see that
our new explanations help users understand model correctness better than
baselines by an expected 30\% and that our proxy helpfulness metrics correlate
strongly ($\rho>0.97$) with how well users can predict model correctness.

    

### [[2106.01609] Tail-to-Tail Non-Autoregressive Sequence Prediction for Chinese Grammatical Error Correction](http://arxiv.org/abs/2106.01609)


  We investigate the problem of Chinese Grammatical Error Correction (CGEC) and
present a new framework named Tail-to-Tail (\textbf{TtT}) non-autoregressive
sequence prediction to address the deep issues hidden in CGEC. Considering that
most tokens are correct and can be conveyed directly from source to target, and
the error positions can be estimated and corrected based on the bidirectional
context information, thus we employ a BERT-initialized Transformer Encoder as
the backbone model to conduct information modeling and conveying. Considering
that only relying on the same position substitution cannot handle the
variable-length correction cases, various operations such substitution,
deletion, insertion, and local paraphrasing are required jointly. Therefore, a
Conditional Random Fields (CRF) layer is stacked on the up tail to conduct
non-autoregressive sequence prediction by modeling the token dependencies.
Since most tokens are correct and easily to be predicted/conveyed to the
target, then the models may suffer from a severe class imbalance issue. To
alleviate this problem, focal loss penalty strategies are integrated into the
loss functions. Moreover, besides the typical fix-length error correction
datasets, we also construct a variable-length corpus to conduct experiments.
Experimental results on standard datasets, especially on the variable-length
datasets, demonstrate the effectiveness of TtT in terms of sentence-level
Accuracy, Precision, Recall, and F1-Measure on tasks of error Detection and
Correction.

    

### [[2106.03400] Believe What You See: Implicit Constraint Approach for Offline Multi-Agent Reinforcement Learning](http://arxiv.org/abs/2106.03400)


  Learning from datasets without interaction with environments (Offline
Learning) is an essential step to apply Reinforcement Learning (RL) algorithms
in real-world scenarios. However, compared with the single-agent counterpart,
offline multi-agent RL introduces more agents with the larger state and action
space, which is more challenging but attracts little attention. We demonstrate
current offline RL algorithms are ineffective in multi-agent systems due to the
accumulated extrapolation error. In this paper, we propose a novel offline RL
algorithm, named Implicit Constraint Q-learning (ICQ), which effectively
alleviates the extrapolation error by only trusting the state-action pairs
given in the dataset for value estimation. Moreover, we extend ICQ to
multi-agent tasks by decomposing the joint-policy under the implicit
constraint. Experimental results demonstrate that the extrapolation error is
successfully controlled within a reasonable range and insensitive to the number
of agents. We further show that ICQ achieves the state-of-the-art performance
in the challenging multi-agent offline tasks (StarCraft II). Our code is public
online at this https URL.

    

### [[2106.03894] Differentiable Quality Diversity](http://arxiv.org/abs/2106.03894)


  Quality diversity (QD) is a growing branch of stochastic optimization
research that studies the problem of generating an archive of solutions that
maximize a given objective function but are also diverse with respect to a set
of specified measure functions. However, even when these functions are
differentiable, QD algorithms treat them as "black boxes", ignoring gradient
information. We present the differentiable quality diversity (DQD) problem, a
special case of QD, where both the objective and measure functions are first
order differentiable. We then present MAP-Elites via a Gradient Arborescence
(MEGA), a DQD algorithm that leverages gradient information to efficiently
explore the joint range of the objective and measure functions. Results in two
QD benchmark domains and in searching the latent space of a StyleGAN show that
MEGA significantly outperforms state-of-the-art QD algorithms, highlighting
DQD's promise for efficient quality diversity optimization when gradient
information is available. Source code is available at
\url{this https URL}.

    

### [<title>How to reduce the model randomness? or stabilize the model - XGBoost</title>](https://discuss.xgboost.ai/t/how-to-reduce-the-model-randomness-or-stabilize-the-model/394/5)