
## 2021-7-30

### [<title>Multi dimensional training - XGBoost</title>](https://discuss.xgboost.ai/t/multi-dimensional-training/2397/1)

### [[2107.13665] Novel Direct Algorithm for Computing Simultaneous All-Levels Reliability of Multi-state Flow Networks](http://arxiv.org/abs/2107.13665)


  All kind of networks, e.g., Internet of Things, social networks, wireless
sensor networks, transportation networks, 4g/5G, etc., are around us to benefit
and help our daily life. The multistate flow network (MFN) is always used to
model network structures and applications. The level d reliability, Rd, of the
MFN is the success probability of sending at least d units of integer flow from
the source node to the sink node. The reliability Rd is a popular index for
designing, managing, controlling, and evaluating MFNs. The traditional indirect
algorithms must have all d-MPs (special connected vectors) or d-MCs (special
disconnected vectors) first, then use Inclusion-Exclusion Technique (IET) or
Sum-of-disjoint Product (SDP) in terms of found d-MPs or d-MCs to calculate Rd.
The above four procedures are all NP-Hard and #P-Hard and cannot calculate Rd
for all d at the same time A novel algorithm based on the binary-addition-tree
algorithm (BAT) is proposed to calculate the Rd directly for all d at the same
time without using any of the above four procedures. The time complexity and
demonstration of the proposed algorithm are analyzed, and examples are
provided. An experiment is also conducted to compare the proposed algorithm and
existing algorithms based on d-MPs, d-MCs, IET, and/or SDP to validate the
proposed algorithm.

    

### [[2107.13694] P4COM: In-Network Computation with Programmable Switches](http://arxiv.org/abs/2107.13694)


  Traditionally, switches only provide forwarding services and have no credits
on computation in distributed computing frameworks. The emerging programmable
switches make in-network computing (INC) possible, i.e., offloading some
computation to the switch data plane. While some proposals have attempted to
offload computation onto special hardwares (e.g., NetFPGA), many practical
issues have not been addressed. Therefore, we propose P4COM - a user-friendly,
memory-efficient, and fault-tolerant framework realizing in-network computation
(e.g., MapReduce) with programmable switches.
P4COM consists of three modules. First, P4COM automatically translates
application logic to switch data plane programs with a lightweight interpreter.
Second, P4COM adopts a memory management policy to efficiently utilize the
limited switch on-chip memory. Third, P4COM provides a cutting-payload
mechanism to handle packet losses. We have built a P4COM prototype with a
Barefoot Tofino switch and multiple commodity servers. Through a combination of
testbed experiments and large-scale simulations, we show that P4COM is able to
achieve line-rate processing at 10Gbps links, and can increase the data
shuffling throughput by 2-5 times for the MapReduce-style applications.

    

### [[2107.13869] Autonomous UAV Base Stations for Next Generation Wireless Networks: A Deep Learning Approach](http://arxiv.org/abs/2107.13869)


  To address the ever-growing connectivity demands of wireless communications,
the adoption of ingenious solutions, such as Unmanned Aerial Vehicles (UAVs) as
mobile Base Stations (BSs), is imperative. In general, the location of a UAV
Base Station (UAV-BS) is determined by optimization algorithms, which have high
computationally complexities and place heavy demands on UAV resources. In this
paper, we show that a Convolutional Neural Network (CNN) model can be trained
to infer the location of a UAV-BS in real time. In so doing, we create a
framework to determine the UAV locations that considers the deployment of
Mobile Users (MUs) to generate labels by using the data obtained from an
optimization algorithm. Performance evaluations reveal that once the CNN model
is trained with the given labels and locations of MUs, the proposed approach is
capable of approximating the results given by the adopted optimization
algorithm with high fidelity, outperforming Reinforcement Learning (RL)-based
approaches. We also explore future research challenges and highlight key
issues.

    

### [[2107.13968] Improving Latency with Active Queue Management (AQM) During COVID-19](http://arxiv.org/abs/2107.13968)


  During the COVID-19 pandemic the Comcast network performed well in response
to unprecedented changes in Internet usage and video communications
applications that are sensitive to network latency have exploded in popularity.
However, in today's typical networks-such as a home network-those applications
often degrade if they are sharing a network link with other network traffic.
This is a problem caused by a network design flaw often described using the
term 'buffer bloat'. Several years ago, Comcast helped to fund research and
development in the technical community into new Active Queue Management (AQM)
techniques to eliminate this issue and AQM was later built into Data Over Cable
Service Interface Specification (DOCSIS) standards.
Just prior to the pandemic, Comcast also deployed a large-scale network
performance measurement system that included a latency under load test. In
addition, Comcast happened to deploy two otherwise identical cable modems; one
with upstream AQM enabled, and the other without. This fortuitous confluence of
events has enabled Comcast to perform a comparative analysis of the differences
between cable modem gateways using AQM with those that lack that enhancement,
at a unique level of scale across many months of time and millions of devices.
The data reveals significantly better upstream latency under load performance
when AQM is used. For the device with AQM, most of the latency under load tests
resulted in 15-30 milliseconds of latency. In comparison, the device without
AQM averaged roughly 250 milliseconds of latency, between 8-16 times higher, a
highly significant difference to the end user quality of experience. These
large-scale measurement comparisons should provide additional data to justify
accelerated deployment of AQM in DOCSIS and other Internet Service Provider
networks and user-purchased home network equipment.

    

### [[2107.14112] Internet-of-Things Devices and Assistive Technologies for Healthcare: Applications, Challenges, and Opportunities](http://arxiv.org/abs/2107.14112)


  Medical conditions and cases are growing at a rapid pace, where physical
space is starting to be constrained. Hospitals and clinics no longer have the
ability to accommodate large numbers of incoming patients. It is clear that the
current state of the health industry needs to improve its valuable and limited
resources. The evolution of the Internet of Things (IoT) devices along with
assistive technologies can alleviate the problem in healthcare, by being a
convenient and easy means of accessing healthcare services wirelessly. There is
a plethora of IoT devices and potential applications that can take advantage of
the unique characteristics that these technologies can offer. However, at the
same time, these services pose novel challenges that need to be properly
addressed. In this article, we review some popular categories of IoT-based
applications for healthcare along with their devices. Then, we describe the
challenges and discuss how research can properly address the open issues and
improve the already existing implementations in healthcare. Further possible
solutions are also discussed to show their potential in being viable solutions
for future healthcare applications

    

### [[2107.14218] Gossiping with Binary Freshness Metric](http://arxiv.org/abs/2107.14218)


  We consider the binary freshness metric for gossip networks that consist of a
single source and $n$ end-nodes, where the end-nodes are allowed to share their
stored versions of the source information with the other nodes. We develop
recursive equations that characterize binary freshness in arbitrarily connected
gossip networks using the stochastic hybrid systems (SHS) approach. Next, we
study binary freshness in several structured gossip networks, namely
disconnected, ring and fully connected networks. We show that for both
disconnected and ring network topologies, when the number of nodes gets large,
the binary freshness of a node decreases down to 0 as $n^{-1}$, but the
freshness is strictly larger for the ring topology. We also show that for the
fully connected topology, the rate of decrease to 0 is slower, and it takes the
form of $n^{-\rho}$ for a $\rho$ smaller than 1, when the update rates of the
source and the end-nodes are sufficiently large. Finally, we study the binary
freshness metric for clustered gossip networks, where multiple clusters of
structured gossip networks are connected to the source node through designated
access nodes, i.e., cluster heads. We characterize the binary freshness in such
networks and numerically study how the optimal cluster sizes change with
respect to the update rates in the system.

    

### [[2102.07100] IMF: Iterative Max-Flow for Node Localizability Detection in Barycentric Linear Localization](http://arxiv.org/abs/2102.07100)


  Determining whether nodes can be uniquely localized, called localizability
detection, is a concomitant problem of network localization. Localizability
under traditional Non-Linear Localization (NLL) schema has been well explored,
whereas localizability under the emerging Barycentric coordinate-based Linear
Localization (BLL) schema has not been well touched. In this paper, we
investigate the deficiency of existing localizability theories and algorithms
in BLL, and then propose a necessary condition and a sufficient condition for
BLL node localizability. Based on these two conditions, an efficient iterative
maximum flow (IMF) algorithm is designed to identify BLL localizable nodes.
Finally, our algorithms are validated by both theoretical analysis and
experimental evaluations.

    

### [[2104.14985] A Path to Smart Radio Environments: An Industrial Viewpoint on Reconfigurable Intelligent Surfaces](http://arxiv.org/abs/2104.14985)


  With both the standardization and commercialization completed in an
unforeseen pace for the 5th generation (5G) wireless network, researchers,
engineers and executives from the academia and the industry have turned their
sights on candidate technologies to support the next generation wireless
networks. Reconfigurable intelligent surfaces (RIS), sometimes referred to as
intelligent reflecting surfaces (IRS), have been identified to be potential
components of the future wireless networks because they can reconfigure the
propagation environment for wireless signals with low-cost passive devices. In
doing so, the coverage of a cell can be expected to increase significantly as
well as the overall throughput of the network. RIS has not only become an
attractive research area but also triggered a couple of projects to develop
appropriate solutions to enable the set-up of hardware demonstrations and
prototypes. In parallel, technical discussions and activities towards
standardization already took off in some regions. Promoting RIS to be
integrated into future commercial networks and become a commercial success
requires significant standardization work taken place both at regional level
standards developing organizations (SDO) and international SDOs such as the 3rd
Generation Partnership Project (3GPP). While many research papers study how RIS
can be used and optimized, few effort is devoted to analyzing the challenges to
commercialize RIS and how RIS can be standardized. This paper intends to shed
some light on RIS from an industrial viewpoint and provide a clear roadmap to
make RIS industrially feasible.

    

### [[2107.00853] Adaptive Regularized Zero-Forcing Beamforming in Massive MIMO with Multi-Antenna Users](http://arxiv.org/abs/2107.00853)


  Modern wireless cellular networks use massive multiple-input multiple-output
technology. This involves operations with an antenna array at a base station
that simultaneously serves multiple mobile devices that also use multiple
antennas on their side. For this, various Beamforming and Detection techniques
are used, allowing each user to receive the signal intended for him from the
base station. There is an important class of linear Precoding called
Regularized Zero-Forcing. In this work, we propose a special kind of
regularization matrix with different regularizations for different UE, using
singular values of multi-antenna users. The proposed algorithm has a simple
analytical formula and is provided with theoretical study. We also show the
results in comparison with other linear Precoding algorithms on simulations
with the Quadriga channel model. The proposed approach leads to a significant
increase in quality with the same computation time as in the baseline methods.

    

### [[2107.13576] Social Processes: Self-Supervised Forecasting of Nonverbal Cues in Social Conversations](http://arxiv.org/abs/2107.13576)


  The default paradigm for the forecasting of human behavior in social
conversations is characterized by top-down approaches. These involve
identifying predictive relationships between low level nonverbal cues and
future semantic events of interest (e.g. turn changes, group leaving). A common
hurdle however, is the limited availability of labeled data for supervised
learning. In this work, we take the first step in the direction of a bottom-up
self-supervised approach in the domain. We formulate the task of Social Cue
Forecasting to leverage the larger amount of unlabeled low-level behavior cues,
and characterize the modeling challenges involved. To address these, we take a
meta-learning approach and propose the Social Process (SP) models--socially
aware sequence-to-sequence (Seq2Seq) models within the Neural Process (NP)
family. SP models learn extractable representations of non-semantic future cues
for each participant, while capturing global uncertainty by jointly reasoning
about the future for all members of the group. Evaluation on synthesized and
real-world behavior data shows that our SP models achieve higher log-likelihood
than the NP baselines, and also highlights important considerations for
applying such techniques within the domain of social human interactions.

    

### [[2107.13586] Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](http://arxiv.org/abs/2107.13586)


  This paper surveys and organizes research works in a new paradigm in natural
language processing, which we dub "prompt-based learning". Unlike traditional
supervised learning, which trains a model to take in an input x and predict an
output y as P(y|x), prompt-based learning is based on language models that
model the probability of text directly. To use these models to perform
prediction tasks, the original input x is modified using a template into a
textual string prompt x' that has some unfilled slots, and then the language
model is used to probabilistically fill the unfilled information to obtain a
final string x, from which the final output y can be derived. This framework is
powerful and attractive for a number of reasons: it allows the language model
to be pre-trained on massive amounts of raw text, and by defining a new
prompting function the model is able to perform few-shot or even zero-shot
learning, adapting to new scenarios with few or no labeled data. In this paper
we introduce the basics of this promising paradigm, describe a unified set of
mathematical notations that can cover a wide variety of existing work, and
organize existing work along several dimensions, e.g.the choice of pre-trained
models, prompts, and tuning strategies. To make the field more accessible to
interested beginners, we not only make a systematic review of existing works
and a highly structured typology of prompt-based concepts, but also release
other resources, e.g., a website this http URL including
constantly-updated survey, and paperlist.

    

### [[2107.13600] To Boost or not to Boost: On the Limits of Boosted Neural Networks](http://arxiv.org/abs/2107.13600)


  Boosting is a method for finding a highly accurate hypothesis by linearly
combining many ``weak" hypotheses, each of which may be only moderately
accurate. Thus, boosting is a method for learning an ensemble of classifiers.
While boosting has been shown to be very effective for decision trees, its
impact on neural networks has not been extensively studied. We prove one
important difference between sums of decision trees compared to sums of
convolutional neural networks (CNNs) which is that a sum of decision trees
cannot be represented by a single decision tree with the same number of
parameters while a sum of CNNs can be represented by a single CNN. Next, using
standard object recognition datasets, we verify experimentally the well-known
result that a boosted ensemble of decision trees usually generalizes much
better on testing data than a single decision tree with the same number of
parameters. In contrast, using the same datasets and boosting algorithms, our
experiments show the opposite to be true when using neural networks (both CNNs
and multilayer perceptrons (MLPs)). We find that a single neural network
usually generalizes better than a boosted ensemble of smaller neural networks
with the same total number of parameters.

    

### [[2107.13610] Large sample spectral analysis of graph-based multi-manifold clustering](http://arxiv.org/abs/2107.13610)


  In this work we study statistical properties of graph-based algorithms for
multi-manifold clustering (MMC). In MMC the goal is to retrieve the
multi-manifold structure underlying a given Euclidean data set when this one is
assumed to be obtained by sampling a distribution on a union of manifolds
$\mathcal{M} = \mathcal{M}_1 \cup\dots \cup \mathcal{M}_N$ that may intersect
with each other and that may have different dimensions. We investigate
sufficient conditions that similarity graphs on data sets must satisfy in order
for their corresponding graph Laplacians to capture the right geometric
information to solve the MMC problem. Precisely, we provide high probability
error bounds for the spectral approximation of a tensorized Laplacian on
$\mathcal{M}$ with a suitable graph Laplacian built from the observations; the
recovered tensorized Laplacian contains all geometric information of all the
individual underlying manifolds. We provide an example of a family of
similarity graphs, which we call annular proximity graphs with angle
constraints, satisfying these sufficient conditions. We contrast our family of
graphs with other constructions in the literature based on the alignment of
tangent planes. Extensive numerical experiments expand the insights that our
theory provides on the MMC problem.

    

### [[2107.13617] Pitch-Informed Instrument Assignment Using a Deep Convolutional Network with Multiple Kernel Shapes](http://arxiv.org/abs/2107.13617)


  This paper proposes a deep convolutional neural network for performing
note-level instrument assignment. Given a polyphonic multi-instrumental music
signal along with its ground truth or predicted notes, the objective is to
assign an instrumental source for each note. This problem is addressed as a
pitch-informed classification task where each note is analysed individually. We
also propose to utilise several kernel shapes in the convolutional layers in
order to facilitate learning of efficient timbre-discriminative feature maps.
Experiments on the MusicNet dataset using 7 instrument classes show that our
approach is able to achieve an average F-score of 0.904 when the original
multi-pitch annotations are used as the pitch information for the system, and
that it also excels if the note information is provided using third-party
multi-pitch estimation algorithms. We also include ablation studies
investigating the effects of the use of multiple kernel shapes and comparing
different input representations for the audio and the note-related information.

    

### [[2107.13619] A Deep Graph Reinforcement Learning Model for Improving User Experience in Live Video Streaming](http://arxiv.org/abs/2107.13619)


  In this paper we present a deep graph reinforcement learning model to predict
and improve the user experience during a live video streaming event,
orchestrated by an agent/tracker. We first formulate the user experience
prediction problem as a classification task, accounting for the fact that most
of the viewers at the beginning of an event have poor quality of experience due
to low-bandwidth connections and limited interactions with the tracker. In our
model we consider different factors that influence the quality of user
experience and train the proposed model on diverse state-action transitions
when viewers interact with the tracker. In addition, provided that past events
have various user experience characteristics we follow a gradient boosting
strategy to compute a global model that learns from different events. Our
experiments with three real-world datasets of live video streaming events
demonstrate the superiority of the proposed model against several baseline
strategies. Moreover, as the majority of the viewers at the beginning of an
event has poor experience, we show that our model can significantly increase
the number of viewers with high quality experience by at least 75% over the
first streaming minutes. Our evaluation datasets and implementation are
publicly available at this https URL


### [[2107.13625] Generalizing Fairness: Discovery and Mitigation of Unknown Sensitive Attributes](http://arxiv.org/abs/2107.13625)


  When deploying artificial intelligence (AI) in the real world, being able to
trust the operation of the AI by characterizing how it performs is an
ever-present and important topic. An important and still largely unexplored
task in this characterization is determining major factors within the real
world that affect the AI's behavior, such as weather conditions or lighting,
and either a) being able to give justification for why it may have failed or b)
eliminating the influence the factor has. Determining these sensitive factors
heavily relies on collected data that is diverse enough to cover numerous
combinations of these factors, which becomes more onerous when having many
potential sensitive factors or operating in complex environments. This paper
investigates methods that discover and separate out individual semantic
sensitive factors from a given dataset to conduct this characterization as well
as addressing mitigation of these factors' sensitivity. We also broaden
remediation of fairness, which normally only addresses socially relevant
factors, and widen it to deal with the desensitization of AI with regard to all
possible aspects of variation in the domain. The proposed methods which
discover these major factors reduce the potentially onerous demands of
collecting a sufficiently diverse dataset. In experiments using the road sign
(GTSRB) and facial imagery (CelebA) datasets, we show the promise of using this
scheme to perform this characterization and remediation and demonstrate that
our approach outperforms state of the art approaches.

    

### [[2107.13639] Imbalanced Adversarial Training with Reweighting](http://arxiv.org/abs/2107.13639)


  Adversarial training has been empirically proven to be one of the most
effective and reliable defense methods against adversarial attacks. However,
almost all existing studies about adversarial training are focused on balanced
datasets, where each class has an equal amount of training examples. Research
on adversarial training with imbalanced training datasets is rather limited. As
the initial effort to investigate this problem, we reveal the facts that
adversarially trained models present two distinguished behaviors from naturally
trained models in imbalanced datasets: (1) Compared to natural training,
adversarially trained models can suffer much worse performance on
under-represented classes, when the training dataset is extremely imbalanced.
(2) Traditional reweighting strategies may lose efficacy to deal with the
imbalance issue for adversarial training. For example, upweighting the
under-represented classes will drastically hurt the model's performance on
well-represented classes, and as a result, finding an optimal reweighting value
can be tremendously challenging. In this paper, to further understand our
observations, we theoretically show that the poor data separability is one key
reason causing this strong tension between under-represented and
well-represented classes. Motivated by this finding, we propose Separable
Reweighted Adversarial Training (SRAT) to facilitate adversarial training under
imbalanced scenarios, by learning more separable features for different
classes. Extensive experiments on various datasets verify the effectiveness of
the proposed framework.

    

### [[2107.13640] Secure Bayesian Federated Analytics for Privacy-Preserving Trend Detection](http://arxiv.org/abs/2107.13640)


  Federated analytics has many applications in edge computing, its use can lead
to better decision making for service provision, product development, and user
experience. We propose a Bayesian approach to trend detection in which the
probability of a keyword being trendy, given a dataset, is computed via Bayes'
Theorem; the probability of a dataset, given that a keyword is trendy, is
computed through secure aggregation of such conditional probabilities over
local datasets of users. We propose a protocol, named SAFE, for Bayesian
federated analytics that offers sufficient privacy for production grade use
cases and reduces the computational burden of users and an aggregator. We
illustrate this approach with a trend detection experiment and discuss how this
approach could be extended further to make it production-ready.

    

### [[2107.13646] Evaluating Relaxations of Logic for Neural Networks: A Comprehensive Study](http://arxiv.org/abs/2107.13646)


  Symbolic knowledge can provide crucial inductive bias for training neural
models, especially in low data regimes. A successful strategy for incorporating
such knowledge involves relaxing logical statements into sub-differentiable
losses for optimization. In this paper, we study the question of how best to
relax logical expressions that represent labeled examples and knowledge about a
problem; we focus on sub-differentiable t-norm relaxations of logic. We present
theoretical and empirical criteria for characterizing which relaxation would
perform best in various scenarios. In our theoretical study driven by the goal
of preserving tautologies, the Lukasiewicz t-norm performs best. However, in
our empirical analysis on the text chunking and digit recognition tasks, the
product t-norm achieves best predictive performance. We analyze this apparent
discrepancy, and conclude with a list of best practices for defining loss
functions via logic.

    

### [[2107.13648] Spot What Matters: Learning Context Using Graph Convolutional Networks for Weakly-Supervised Action Detection](http://arxiv.org/abs/2107.13648)


  The dominant paradigm in spatiotemporal action detection is to classify
actions using spatiotemporal features learned by 2D or 3D Convolutional
Networks. We argue that several actions are characterized by their context,
such as relevant objects and actors present in the video. To this end, we
introduce an architecture based on self-attention and Graph Convolutional
Networks in order to model contextual cues, such as actor-actor and
actor-object interactions, to improve human action detection in video. We are
interested in achieving this in a weakly-supervised setting, i.e. using as less
annotations as possible in terms of action bounding boxes. Our model aids
explainability by visualizing the learned context as an attention map, even for
actions and objects unseen during training. We evaluate how well our model
highlights the relevant context by introducing a quantitative metric based on
recall of objects retrieved by attention maps. Our model relies on a 3D
convolutional RGB stream, and does not require expensive optical flow
computation. We evaluate our models on the DALY dataset, which consists of
human-object interaction actions. Experimental results show that our
contextualized approach outperforms a baseline action detection approach by
more than 2 points in Video-mAP. Code is available at
\url{this https URL}

    

### [[2107.13653] Demand Forecasting in Smart Grid Using Long Short-Term Memory](http://arxiv.org/abs/2107.13653)


  Demand forecasting in power sector has become an important part of modern
demand management and response systems with the rise of smart metering enabled
grids. Long Short-Term Memory (LSTM) shows promising results in predicting time
series data which can also be applied to power load demand in smart grids. In
this paper, an LSTM based model using neural network architecture is proposed
to forecast power demand. The model is trained with hourly energy and power
usage data of four years from a smart grid. After training and prediction, the
accuracy of the model is compared against the traditional statistical time
series analysis algorithms, such as Auto-Regressive (AR), to determine the
efficiency. The mean absolute percentile error is found to be 1.22 in the
proposed LSTM model, which is the lowest among the other models. From the
findings, it is clear that the inclusion of neural network in predicting power
demand reduces the error of prediction significantly. Thus, the application of
LSTM can enable a more efficient demand response system.

    

### [[2107.13656] Characterizing the Generalization Error of Gibbs Algorithm with Symmetrized KL information](http://arxiv.org/abs/2107.13656)


  Bounding the generalization error of a supervised learning algorithm is one
of the most important problems in learning theory, and various approaches have
been developed. However, existing bounds are often loose and lack of
guarantees. As a result, they may fail to characterize the exact generalization
ability of a learning algorithm. Our main contribution is an exact
characterization of the expected generalization error of the well-known Gibbs
algorithm in terms of symmetrized KL information between the input training
samples and the output hypothesis. Such a result can be applied to tighten
existing expected generalization error bound. Our analysis provides more
insight on the fundamental role the symmetrized KL information plays in
controlling the generalization error of the Gibbs algorithm.

    

### [[2107.13657] Competitive Control](http://arxiv.org/abs/2107.13657)


  We consider control from the perspective of competitive analysis. Unlike much
prior work on learning-based control, which focuses on minimizing regret
against the best controller selected in hindsight from some specific class, we
focus on designing an online controller which competes against a clairvoyant
offline optimal controller. A natural performance metric in this setting is
competitive ratio, which is the ratio between the cost incurred by the online
controller and the cost incurred by the offline optimal controller. Using
operator-theoretic techniques from robust control, we derive a computationally
efficient state-space description of the the controller with optimal
competitive ratio in both finite-horizon and infinite-horizon settings. We
extend competitive control to nonlinear systems using Model Predictive Control
(MPC) and present numerical experiments which show that our competitive
controller can significantly outperform standard $H_2$ and $H_{\infty}$
controllers in the MPC setting.

    

### [[2107.13671] Deeper Learning By Doing: Integrating Hands-On Research Projects Into a Machine Learning Course](http://arxiv.org/abs/2107.13671)


  Machine learning has seen a vast increase of interest in recent years, along
with an abundance of learning resources. While conventional lectures provide
students with important information and knowledge, we also believe that
additional project-based learning components can motivate students to engage in
topics more deeply. In addition to incorporating project-based learning in our
courses, we aim to develop project-based learning components aligned with
real-world tasks, including experimental design and execution, report writing,
oral presentation, and peer-reviewing. This paper describes the organization of
our project-based machine learning courses with a particular emphasis on the
class project components and shares our resources with instructors who would
like to include similar elements in their courses.

    

### [[2107.13673] Relational Graph Neural Networks for Fraud Detection in a Super-Appe nvironment](http://arxiv.org/abs/2107.13673)


  Large digital platforms create environments where different types of user
interactions are captured, these relationships offer a novel source of
information for fraud detection problems. In this paper we propose a framework
of relational graph convolutional networks methods for fraudulent behaviour
prevention in the financial services of a Super-App. To this end, we apply the
framework on different heterogeneous graphs of users, devices, and credit
cards; and finally use an interpretability algorithm for graph neural networks
to determine the most important relations to the classification task of the
users. Our results show that there is an added value when considering models
that take advantage of the alternative data of the Super-App and the
interactions found in their high connectivity, further proofing how they can
leverage that into better decisions and fraud detection strategies.

    

### [[2107.13686] AutoTinyBERT: Automatic Hyper-parameter Optimization for Efficient Pre-trained Language Models](http://arxiv.org/abs/2107.13686)


  Pre-trained language models (PLMs) have achieved great success in natural
language processing. Most of PLMs follow the default setting of architecture
hyper-parameters (e.g., the hidden dimension is a quarter of the intermediate
dimension in feed-forward sub-networks) in BERT (Devlin et al., 2019). Few
studies have been conducted to explore the design of architecture
hyper-parameters in BERT, especially for the more efficient PLMs with tiny
sizes, which are essential for practical deployment on resource-constrained
devices. In this paper, we adopt the one-shot Neural Architecture Search (NAS)
to automatically search architecture hyper-parameters. Specifically, we
carefully design the techniques of one-shot learning and the search space to
provide an adaptive and efficient development way of tiny PLMs for various
latency constraints. We name our method AutoTinyBERT and evaluate its
effectiveness on the GLUE and SQuAD benchmarks. The extensive experiments show
that our method outperforms both the SOTA search-based baseline (NAS-BERT) and
the SOTA distillation-based methods (such as DistilBERT, TinyBERT, MiniLM and
MobileBERT). In addition, based on the obtained architectures, we propose a
more efficient development method that is even faster than the development of a
single PLM.

    

### [[2107.13720] Convolutional Transformer based Dual Discriminator Generative Adversarial Networks for Video Anomaly Detection](http://arxiv.org/abs/2107.13720)


  Detecting abnormal activities in real-world surveillance videos is an
important yet challenging task as the prior knowledge about video anomalies is
usually limited or unavailable. Despite that many approaches have been
developed to resolve this problem, few of them can capture the normal
spatio-temporal patterns effectively and efficiently. Moreover, existing works
seldom explicitly consider the local consistency at frame level and global
coherence of temporal dynamics in video sequences. To this end, we propose
Convolutional Transformer based Dual Discriminator Generative Adversarial
Networks (CT-D2GAN) to perform unsupervised video anomaly detection.
Specifically, we first present a convolutional transformer to perform future
frame prediction. It contains three key components, i.e., a convolutional
encoder to capture the spatial information of the input video clips, a temporal
self-attention module to encode the temporal dynamics, and a convolutional
decoder to integrate spatio-temporal features and predict the future frame.
Next, a dual discriminator based adversarial training procedure, which jointly
considers an image discriminator that can maintain the local consistency at
frame-level and a video discriminator that can enforce the global coherence of
temporal dynamics, is employed to enhance the future frame prediction. Finally,
the prediction error is used to identify abnormal video frames. Thoroughly
empirical studies on three public video anomaly detection datasets, i.e., UCSD
Ped2, CUHK Avenue, and Shanghai Tech Campus, demonstrate the effectiveness of
the proposed adversarial spatio-temporal modeling framework.

    

### [[2107.13721] Amplitude Mean of Functional Data on $\mathbb{S}^2$](http://arxiv.org/abs/2107.13721)


  Mainfold-valued functional data analysis (FDA) recently becomes an active
area of research motivated by the raising availability of trajectories or
longitudinal data observed on non-linear manifolds. The challenges of analyzing
such data comes from many aspects, including infinite dimensionality and
nonlinearity, as well as time domain or phase variability. In this paper, we
study the amplitude part of manifold-valued functions on $§^2$, which is
invariant to random time warping or re-parameterization of the function.
Utilizing the nice geometry of $§^2$, we develop a set of efficient and
accurate tools for temporal alignment of functions, geodesic and sample mean
calculation. At the heart of these tools, they rely on gradient descent
algorithms with carefully derived gradients. We show the advantages of these
newly developed tools over its competitors with extensive simulations and real
data, and demonstrate the importance of considering the amplitude part of
functions instead of mixing it with phase variability in mainfold-valued FDA.

    

### [[2107.13735] Learning the temporal evolution of multivariate densities via normalizing flows](http://arxiv.org/abs/2107.13735)


  In this work, we propose a method to learn probability distributions using
sample path data from stochastic differential equations. Specifically, we
consider temporally evolving probability distributions (e.g., those produced by
integrating local or nonlocal Fokker-Planck equations). We analyze this
evolution through machine learning assisted construction of a time-dependent
mapping that takes a reference distribution (say, a Gaussian) to each and every
instance of our evolving distribution. If the reference distribution is the
initial condition of a Fokker-Planck equation, what we learn is the time-T map
of the corresponding solution. Specifically, the learned map is a normalizing
flow that deforms the support of the reference density to the support of each
and every density snapshot in time. We demonstrate that this approach can learn
solutions to non-local Fokker-Planck equations, such as those arising in
systems driven by both Brownian and Lévy noise. We present examples with two-
and three-dimensional, uni- and multimodal distributions to validate the
method.

    

### [[2107.13743] Malware Classification Using Transfer Learning](http://arxiv.org/abs/2107.13743)


  With the rapid growth of the number of devices on the Internet, malware poses
a threat not only to the affected devices but also their ability to use said
devices to launch attacks on the Internet ecosystem. Rapid malware
classification is an important tools to combat that threat. One of the
successful approaches to classification is based on malware images and deep
learning. While many deep learning architectures are very accurate they usually
take a long time to train. In this work we perform experiments on multiple well
known, pre-trained, deep network architectures in the context of transfer
learning. We show that almost all them classify malware accurately with a very
short training period.

    

### [[2107.13772] Bayesian Optimization for Min Max Optimization](http://arxiv.org/abs/2107.13772)


  A solution that is only reliable under favourable conditions is hardly a safe
solution. Min Max Optimization is an approach that returns optima that are
robust against worst case conditions. We propose algorithms that perform Min
Max Optimization in a setting where the function that should be optimized is
not known a priori and hence has to be learned by experiments. Therefore we
extend the Bayesian Optimization setting, which is tailored to maximization
problems, to Min Max Optimization problems. While related work extends the two
acquisition functions Expected Improvement and Gaussian Process Upper
Confidence Bound; we extend the two acquisition functions Entropy Search and
Knowledge Gradient. These acquisition functions are able to gain knowledge
about the optimum instead of just looking for points that are supposed to be
optimal. In our evaluation we show that these acquisition functions allow for
better solutions - converging faster to the optimum than the benchmark
settings.

    

### [[2107.13782] Multimodal Co-learning: Challenges, Applications with Datasets, Recent Advances and Future Directions](http://arxiv.org/abs/2107.13782)


  Multimodal deep learning systems which employ multiple modalities like text,
image, audio, video, etc., are showing better performance in comparison with
individual modalities (i.e., unimodal) systems. Multimodal machine learning
involves multiple aspects: representation, translation, alignment, fusion, and
co-learning. In the current state of multimodal machine learning, the
assumptions are that all modalities are present, aligned, and noiseless during
training and testing time. However, in real-world tasks, typically, it is
observed that one or more modalities are missing, noisy, lacking annotated
data, have unreliable labels, and are scarce in training or testing and or
both. This challenge is addressed by a learning paradigm called multimodal
co-learning. The modeling of a (resource-poor) modality is aided by exploiting
knowledge from another (resource-rich) modality using transfer of knowledge
between modalities, including their representations and predictive models.
Co-learning being an emerging area, there are no dedicated reviews explicitly
focusing on all challenges addressed by co-learning. To that end, in this work,
we provide a comprehensive survey on the emerging area of multimodal
co-learning that has not been explored in its entirety yet. We review
implementations that overcome one or more co-learning challenges without
explicitly considering them as co-learning challenges. We present the
comprehensive taxonomy of multimodal co-learning based on the challenges
addressed by co-learning and associated implementations. The various techniques
employed to include the latest ones are reviewed along with some of the
applications and datasets. Our final goal is to discuss challenges and
perspectives along with the important ideas and directions for future work that
we hope to be beneficial for the entire research community focusing on this
exciting domain.

    

### [[2107.13790] Non-Markovian Reinforcement Learning using Fractional Dynamics](http://arxiv.org/abs/2107.13790)


  Reinforcement learning (RL) is a technique to learn the control policy for an
agent that interacts with a stochastic environment. In any given state, the
agent takes some action, and the environment determines the probability
distribution over the next state as well as gives the agent some reward. Most
RL algorithms typically assume that the environment satisfies Markov
assumptions (i.e. the probability distribution over the next state depends only
on the current state). In this paper, we propose a model-based RL technique for
a system that has non-Markovian dynamics. Such environments are common in many
real-world applications such as in human physiology, biological systems,
material science, and population dynamics. Model-based RL (MBRL) techniques
typically try to simultaneously learn a model of the environment from the data,
as well as try to identify an optimal policy for the learned model. We propose
a technique where the non-Markovianity of the system is modeled through a
fractional dynamical system. We show that we can quantify the difference in the
performance of an MBRL algorithm that uses bounded horizon model predictive
control from the optimal policy. Finally, we demonstrate our proposed framework
on a pharmacokinetic model of human blood glucose dynamics and show that our
fractional models can capture distant correlations on real-world datasets.

    

### [[2107.13797] HAFLO: GPU-Based Acceleration for Federated Logistic Regression](http://arxiv.org/abs/2107.13797)


  In recent years, federated learning (FL) has been widely applied for
supporting decentralized collaborative learning scenarios. Among existing FL
models, federated logistic regression (FLR) is a widely used statistic model
and has been used in various industries. To ensure data security and user
privacy, FLR leverages homomorphic encryption (HE) to protect the exchanged
data among different collaborative parties. However, HE introduces significant
computational overhead (i.e., the cost of data encryption/decryption and
calculation over encrypted data), which eventually becomes the performance
bottleneck of the whole system. In this paper, we propose HAFLO, a GPU-based
solution to improve the performance of FLR. The core idea of HAFLO is to
summarize a set of performance-critical homomorphic operators (HO) used by FLR
and accelerate the execution of these operators through a joint optimization of
storage, IO, and computation. The preliminary results show that our
acceleration on FATE, a popular FL framework, achieves a 49.9$\times$ speedup
for heterogeneous LR and 88.4$\times$ for homogeneous LR.

    

### [[2107.13821] Concept for a Technical Infrastructure for Management of Predictive Models in Industrial Applications](http://arxiv.org/abs/2107.13821)


  With the increasing number of created and deployed prediction models and the
complexity of machine learning workflows we require so called model management
systems to support data scientists in their tasks. In this work we describe our
technological concept for such a model management system. This concept includes
versioned storage of data, support for different machine learning algorithms,
fine tuning of models, subsequent deployment of models and monitoring of model
performance after deployment. We describe this concept with a close focus on
model lifecycle requirements stemming from our industry application cases, but
generalize key features that are relevant for all applications of machine
learning.

    

### [[2107.13822] Semi-supervised Learning for Data-driven Soft-sensing of Biological and Chemical Processes](http://arxiv.org/abs/2107.13822)


  Continuously operated (bio-)chemical processes increasingly suffer from
external disturbances, such as feed fluctuations or changes in market
conditions. Product quality often hinges on control of rarely measured
concentrations, which are expensive to measure. Semi-supervised regression is a
possible building block and method from machine learning to construct
soft-sensors for such infrequently measured states. Using two case studies,
i.e., the Williams-Otto process and a bioethanol production process,
semi-supervised regression is compared against standard regression to evaluate
its merits and its possible scope of application for process control in the
(bio-)chemical industry.

    

### [[2107.13832] Blind Room Parameter Estimation Using Multiple-Multichannel Speech Recordings](http://arxiv.org/abs/2107.13832)


  Knowing the geometrical and acoustical parameters of a room may benefit
applications such as audio augmented reality, speech dereverberation or audio
forensics. In this paper, we study the problem of jointly estimating the total
surface area, the volume, as well as the frequency-dependent reverberation time
and mean surface absorption of a room in a blind fashion, based on two-channel
noisy speech recordings from multiple, unknown source-receiver positions. A
novel convolutional neural network architecture leveraging both single- and
inter-channel cues is proposed and trained on a large, realistic simulated
dataset. Results on both simulated and real data show that using multiple
observations in one room significantly reduces estimation errors and variances
on all target quantities, and that using two channels helps the estimation of
surface and volume. The proposed model outperforms a recently proposed blind
volume estimation method on the considered datasets.

    

### [[2107.13833] Recurrent U-net for automatic pelvic floor muscle segmentation on 3D ultrasound](http://arxiv.org/abs/2107.13833)


  The prevalance of pelvic floor problems is high within the female population.
Transperineal ultrasound (TPUS) is the main imaging modality used to
investigate these problems. Automating the analysis of TPUS data will help in
growing our understanding of pelvic floor related problems. In this study we
present a U-net like neural network with some convolutional long short term
memory (CLSTM) layers to automate the 3D segmentation of the levator ani muscle
(LAM) in TPUS volumes. The CLSTM layers are added to preserve the inter-slice
3D information. We reach human level performance on this segmentation task.
Therefore, we conclude that we successfully automated the segmentation of the
LAM on 3D TPUS data. This paves the way towards automatic in-vivo analysis of
the LAM mechanics in the context of large study populations.

    

### [[2107.13841] Addressing materials' microstructure diversity using transfer learning](http://arxiv.org/abs/2107.13841)


  Materials' microstructures are signatures of their alloying composition and
processing history. Therefore, microstructures exist in a wide variety. As
materials become increasingly complex to comply with engineering demands,
advanced computer vision (CV) approaches such as deep learning (DL) inevitably
gain relevance for quantifying microstrucutures' constituents from micrographs.
While DL can outperform classical CV techniques for many tasks, shortcomings
are poor data efficiency and generalizability across datasets. This is
inherently in conflict with the expense associated with annotating materials
data through experts and extensive materials diversity. To tackle poor domain
generalizability and the lack of labeled data simultaneously, we propose to
apply a sub-class of transfer learning methods called unsupervised domain
adaptation (UDA). These algorithms address the task of finding domain-invariant
features when supplied with annotated source data and unannotated target data,
such that performance on the latter distribution is optimized despite the
absence of annotations. Exemplarily, this study is conducted on a lath-shaped
bainite segmentation task in complex phase steel micrographs. Here, the domains
to bridge are selected to be different metallographic specimen preparations
(surface etchings) and distinct imaging modalities. We show that a
state-of-the-art UDA approach surpasses the naïve application of source
domain trained models on the target domain (generalization baseline) to a large
extent. This holds true independent of the domain shift, despite using little
data, and even when the baseline models were pre-trained or employed data
augmentation. Through UDA, mIoU was improved over generalization baselines from
82.2%, 61.0%, 49.7% to 84.7%, 67.3%, 73.3% on three target datasets,
respectively. This underlines this techniques' potential to cope with materials
variance.

    

### [[2107.13856] Predicting battery end of life from solar off-grid system field data using machine learning](http://arxiv.org/abs/2107.13856)


  Hundreds of millions of people lack access to electricity. Decentralised
solar-battery systems are key for addressing this whilst avoiding carbon
emissions and air pollution, but are hindered by relatively high costs and
rural locations that inhibit timely preventative maintenance. Accurate
diagnosis of battery health and prediction of end of life from operational data
improves user experience and reduces costs. But lack of controlled validation
tests and variable data quality mean existing lab-based techniques fail to
work. We apply a scaleable probabilistic machine learning approach to diagnose
health in 1027 solar-connected lead-acid batteries, each running for 400-760
days, totalling 620 million data rows. We demonstrate 73% accurate prediction
of end of life, eight weeks in advance, rising to 82% at the point of failure.
This work highlights the opportunity to estimate health from existing
measurements using `big data' techniques, without additional equipment,
extending lifetime and improving performance in real-world applications.

    

### [[2107.13870] Artificial Intelligence Hybrid Deep Learning Model for Groundwater Level Prediction Using MLP-ADAM](http://arxiv.org/abs/2107.13870)


  Groundwater is the largest storage of freshwater resources, which serves as
the major inventory for most of the human consumption through agriculture,
industrial, and domestic water supply. In the fields of hydrological, some
researchers applied a neural network to forecast rainfall intensity in
space-time and introduced the advantages of neural networks compared to
numerical models. Then, many researches have been conducted applying
data-driven models. Some of them extended an Artificial Neural Networks (ANNs)
model to forecast groundwater level in semi-confined glacial sand and gravel
aquifer under variable state, pumping extraction and climate conditions with
significant accuracy. In this paper, a multi-layer perceptron is applied to
simulate groundwater level. The adaptive moment estimation optimization
algorithm is also used to this matter. The root mean squared error, mean
absolute error, mean squared error and the coefficient of determination ( ) are
used to evaluate the accuracy of the simulated groundwater level. Total value
of and RMSE are 0.9458 and 0.7313 respectively which are obtained from the
model output. Results indicate that deep learning algorithms can demonstrate a
high accuracy prediction. Although the optimization of parameters is
insignificant in numbers, but due to the value of time in modelling setup, it
is highly recommended to apply an optimization algorithm in modelling.

    

### [[2107.13875] Spatio-temporal graph neural networks for multi-site PV power forecasting](http://arxiv.org/abs/2107.13875)


  Accurate forecasting of solar power generation with fine temporal and spatial
resolution is vital for the operation of the power grid. However,
state-of-the-art approaches that combine machine learning with numerical
weather predictions (NWP) have coarse resolution. In this paper, we take a
graph signal processing perspective and model multi-site photovoltaic (PV)
production time series as signals on a graph to capture their spatio-temporal
dependencies and achieve higher spatial and temporal resolution forecasts. We
present two novel graph neural network models for deterministic multi-site PV
forecasting dubbed the graph-convolutional long short term memory (GCLSTM) and
the graph-convolutional transformer (GCTrafo) models. These methods rely solely
on production data and exploit the intuition that PV systems provide a dense
network of virtual weather stations. The proposed methods were evaluated in two
data sets for an entire year: 1) production data from 304 real PV systems, and
2) simulated production of 1000 PV systems, both distributed over Switzerland.
The proposed models outperform state-of-the-art multi-site forecasting methods
for prediction horizons of six hours ahead. Furthermore, the proposed models
outperform state-of-the-art single-site methods with NWP as inputs on horizons
up to four hours ahead.

    

### [[2107.13876] Understanding the Effects of Adversarial Personalized Ranking Optimization Method on Recommendation Quality](http://arxiv.org/abs/2107.13876)


  Recommender systems (RSs) employ user-item feedback, e.g., ratings, to match
customers to personalized lists of products. Approaches to top-k recommendation
mainly rely on Learning-To-Rank algorithms and, among them, the most widely
adopted is Bayesian Personalized Ranking (BPR), which bases on a pair-wise
optimization approach. Recently, BPR has been found vulnerable against
adversarial perturbations of its model parameters. Adversarial Personalized
Ranking (APR) mitigates this issue by robustifying BPR via an adversarial
training procedure. The empirical improvements of APR's accuracy performance on
BPR have led to its wide use in several recommender models. However, a key
overlooked aspect has been the beyond-accuracy performance of APR, i.e.,
novelty, coverage, and amplification of popularity bias, considering that
recent results suggest that BPR, the building block of APR, is sensitive to the
intensification of biases and reduction of recommendation novelty. In this
work, we model the learning characteristics of the BPR and APR optimization
frameworks to give mathematical evidence that, when the feedback data have a
tailed distribution, APR amplifies the popularity bias more than BPR due to an
unbalanced number of received positive updates from short-head items. Using
matrix factorization (MF), we empirically validate the theoretical results by
performing preliminary experiments on two public datasets to compare BPR-MF and
APR-MF performance on accuracy and beyond-accuracy metrics. The experimental
results consistently show the degradation of novelty and coverage measures and
a worrying amplification of bias.

    

### [[2107.13892] QuPeD: Quantized Personalization via Distillation with Applications to Federated Learning](http://arxiv.org/abs/2107.13892)


  Traditionally, federated learning (FL) aims to train a single global model
while collaboratively using multiple clients and a server. Two natural
challenges that FL algorithms face are heterogeneity in data across clients and
collaboration of clients with {\em diverse resources}. In this work, we
introduce a \textit{quantized} and \textit{personalized} FL algorithm QuPeD
that facilitates collective (personalized model compression) training via
\textit{knowledge distillation} (KD) among clients who have access to
heterogeneous data and resources. For personalization, we allow clients to
learn \textit{compressed personalized models} with different quantization
parameters and model dimensions/structures. Towards this, first we propose an
algorithm for learning quantized models through a relaxed optimization problem,
where quantization values are also optimized over. When each client
participating in the (federated) learning process has different requirements
for the compressed model (both in model dimension and precision), we formulate
a compressed personalization framework by introducing knowledge distillation
loss for local client objectives collaborating through a global model. We
develop an alternating proximal gradient update for solving this compressed
personalization problem, and analyze its convergence properties. Numerically,
we validate that QuPeD outperforms competing personalized FL methods, FedAvg,
and local training of clients in various heterogeneous settings.

    

### [[2107.13921] Bellamy: Reusing Performance Models for Distributed Dataflow Jobs Across Contexts](http://arxiv.org/abs/2107.13921)


  Distributed dataflow systems enable the use of clusters for scalable data
analytics. However, selecting appropriate cluster resources for a processing
job is often not straightforward. Performance models trained on historical
executions of a concrete job are helpful in such situations, yet they are
usually bound to a specific job execution context (e.g. node type, software
versions, job parameters) due to the few considered input parameters. Even in
case of slight context changes, such supportive models need to be retrained and
cannot benefit from historical execution data from related contexts.
This paper presents Bellamy, a novel modeling approach that combines
scale-outs, dataset sizes, and runtimes with additional descriptive properties
of a dataflow job. It is thereby able to capture the context of a job
execution. Moreover, Bellamy is realizing a two-step modeling approach. First,
a general model is trained on all the available data for a specific scalable
analytics algorithm, hereby incorporating data from different contexts.
Subsequently, the general model is optimized for the specific situation at
hand, based on the available data for the concrete context. We evaluate our
approach on two publicly available datasets consisting of execution data from
various dataflow jobs carried out in different environments, showing that
Bellamy outperforms state-of-the-art methods.

    

### [[2107.13943] Ranking Micro-Influencers: a Novel Multi-Task Learning and Interpretable Framework](http://arxiv.org/abs/2107.13943)


  With the rise in use of social media to promote branded products, the demand
for effective influencer marketing has increased. Brands are looking for
improved ways to identify valuable influencers among a vast catalogue; this is
even more challenging with "micro-influencers", which are more affordable than
mainstream ones but difficult to discover. In this paper, we propose a novel
multi-task learning framework to improve the state of the art in
micro-influencer ranking based on multimedia content. Moreover, since the
visual congruence between a brand and influencer has been shown to be good
measure of compatibility, we provide an effective visual method for
interpreting our models' decisions, which can also be used to inform brands'
media strategies. We compare with the current state-of-the-art on a recently
constructed public dataset and we show significant improvement both in terms of
accuracy and model complexity. The techniques for ranking and interpretation
presented in this work can be generalised to arbitrary multimedia ranking tasks
that have datasets with a similar structure.

    

### [[2107.13944] Lyapunov-based uncertainty-aware safe reinforcement learning](http://arxiv.org/abs/2107.13944)


  Reinforcement learning (RL) has shown a promising performance in learning
optimal policies for a variety of sequential decision-making tasks. However, in
many real-world RL problems, besides optimizing the main objectives, the agent
is expected to satisfy a certain level of safety (e.g., avoiding collisions in
autonomous driving). While RL problems are commonly formalized as Markov
decision processes (MDPs), safety constraints are incorporated via constrained
Markov decision processes (CMDPs). Although recent advances in safe RL have
enabled learning safe policies in CMDPs, these safety requirements should be
satisfied during both training and in the deployment process. Furthermore, it
is shown that in memory-based and partially observable environments, these
methods fail to maintain safety over unseen out-of-distribution observations.
To address these limitations, we propose a Lyapunov-based uncertainty-aware
safe RL model. The introduced model adopts a Lyapunov function that converts
trajectory-based constraints to a set of local linear constraints. Furthermore,
to ensure the safety of the agent in highly uncertain environments, an
uncertainty quantification method is developed that enables identifying
risk-averse actions through estimating the probability of constraint
violations. Moreover, a Transformers model is integrated to provide the agent
with memory to process long time horizons of information via the self-attention
mechanism. The proposed model is evaluated in grid-world navigation tasks where
safety is defined as avoiding static and dynamic obstacles in fully and
partially observable environments. The results of these experiments show a
significant improvement in the performance of the agent both in achieving
optimality and satisfying safety constraints.

    

### [[2107.13964] Mind the Performance Gap: Examining Dataset Shift During Prospective Validation](http://arxiv.org/abs/2107.13964)


  Once integrated into clinical care, patient risk stratification models may
perform worse compared to their retrospective performance. To date, it is
widely accepted that performance will degrade over time due to changes in care
processes and patient populations. However, the extent to which this occurs is
poorly understood, in part because few researchers report prospective
validation performance. In this study, we compare the 2020-2021 ('20-'21)
prospective performance of a patient risk stratification model for predicting
healthcare-associated infections to a 2019-2020 ('19-'20) retrospective
validation of the same model. We define the difference in retrospective and
prospective performance as the performance gap. We estimate how i) "temporal
shift", i.e., changes in clinical workflows and patient populations, and ii)
"infrastructure shift", i.e., changes in access, extraction and transformation
of data, both contribute to the performance gap. Applied prospectively to
26,864 hospital encounters during a twelve-month period from July 2020 to June
2021, the model achieved an area under the receiver operating characteristic
curve (AUROC) of 0.767 (95% confidence interval (CI): 0.737, 0.801) and a Brier
score of 0.189 (95% CI: 0.186, 0.191). Prospective performance decreased
slightly compared to '19-'20 retrospective performance, in which the model
achieved an AUROC of 0.778 (95% CI: 0.744, 0.815) and a Brier score of 0.163
(95% CI: 0.161, 0.165). The resulting performance gap was primarily due to
infrastructure shift and not temporal shift. So long as we continue to develop
and validate models using data stored in large research data warehouses, we
must consider differences in how and when data are accessed, measure how these
differences may affect prospective performance, and work to mitigate those
differences.

    

### [[2107.13966] Artificial Intelligence in Achieving Sustainable Development Goals](http://arxiv.org/abs/2107.13966)


  This perspective illustrates some of the AI applications that can accelerate
the achievement of SDGs and also highlights some of the considerations that
could hinder the efforts towards them. This emphasizes the importance of
establishing standard AI guidelines and regulations for the beneficial
applications of AI.

    

### [[2107.13969] Significance of Speaker Embeddings and Temporal Context for Depression Detection](http://arxiv.org/abs/2107.13969)


  Depression detection from speech has attracted a lot of attention in recent
years. However, the significance of speaker-specific information in depression
detection has not yet been explored. In this work, we analyze the significance
of speaker embeddings for the task of depression detection from speech.
Experimental results show that the speaker embeddings provide important cues to
achieve state-of-the-art performance in depression detection. We also show that
combining conventional OpenSMILE and COVAREP features, which carry
complementary information, with speaker embeddings further improves the
depression detection performance. The significance of temporal context in the
training of deep learning models for depression detection is also analyzed in
this paper.

    

### [[2107.13973] Self-Supervised Learning for Fine-Grained Image Classification](http://arxiv.org/abs/2107.13973)


  Fine-grained image classification involves identifying different
subcategories of a class which possess very subtle discriminatory features.
Fine-grained datasets usually provide bounding box annotations along with class
labels to aid the process of classification. However, building large scale
datasets with such annotations is a mammoth task. Moreover, this extensive
annotation is time-consuming and often requires expertise, which is a huge
bottleneck in building large datasets. On the other hand, self-supervised
learning (SSL) exploits the freely available data to generate supervisory
signals which act as labels. The features learnt by performing some pretext
tasks on huge unlabelled data proves to be very helpful for multiple downstream
tasks.
Our idea is to leverage self-supervision such that the model learns useful
representations of fine-grained image classes. We experimented with 3 kinds of
models: Jigsaw solving as pretext task, adversarial learning (SRGAN) and
contrastive learning based (SimCLR) model. The learned features are used for
downstream tasks such as fine-grained image classification. Our code is
available at
this http URL


### [[2107.13998] "Excavating AI" Re-excavated: Debunking a Fallacious Account of the JAFFE Dataset](http://arxiv.org/abs/2107.13998)


  Twenty-five years ago, my colleagues Miyuki Kamachi and Jiro Gyoba and I
designed and photographed JAFFE, a set of facial expression images intended for
use in a study of face perception. In 2019, without seeking permission or
informing us, Kate Crawford and Trevor Paglen exhibited JAFFE in two widely
publicized art shows. In addition, they published a nonfactual account of the
images in the essay "Excavating AI: The Politics of Images in Machine Learning
Training Sets." The present article recounts the creation of the JAFFE dataset
and unravels each of Crawford and Paglen's fallacious statements. I also
discuss JAFFE more broadly in connection with research on facial expression,
affective computing, and human-computer interaction.

    

### [[2107.14028] Estimating Respiratory Rate From Breath Audio Obtained Through Wearable Microphones](http://arxiv.org/abs/2107.14028)


  Respiratory rate (RR) is a clinical metric used to assess overall health and
physical fitness. An individual's RR can change from their baseline due to
chronic illness symptoms (e.g., asthma, congestive heart failure), acute
illness (e.g., breathlessness due to infection), and over the course of the day
due to physical exhaustion during heightened exertion. Remote estimation of RR
can offer a cost-effective method to track disease progression and
cardio-respiratory fitness over time. This work investigates a model-driven
approach to estimate RR from short audio segments obtained after physical
exertion in healthy adults. Data was collected from 21 individuals using
microphone-enabled, near-field headphones before, during, and after strenuous
exercise. RR was manually annotated by counting perceived inhalations and
exhalations. A multi-task Long-Short Term Memory (LSTM) network with
convolutional layers was implemented to process mel-filterbank energies,
estimate RR in varying background noise conditions, and predict heavy
breathing, indicated by an RR of more than 25 breaths per minute. The
multi-task model performs both classification and regression tasks and
leverages a mixture of loss functions. It was observed that RR can be estimated
with a concordance correlation coefficient (CCC) of 0.76 and a mean squared
error (MSE) of 0.2, demonstrating that audio can be a viable signal for
approximating RR.

    

### [[2107.14033] Temporal-Relational Hypergraph Tri-Attention Networks for Stock Trend Prediction](http://arxiv.org/abs/2107.14033)


  Predicting the future price trends of stocks is a challenging yet intriguing
problem given its critical role to help investors make profitable decisions. In
this paper, we present a collaborative temporal-relational modeling framework
for end-to-end stock trend prediction. The temporal dynamics of stocks is
firstly captured with an attention-based recurrent neural network. Then,
different from existing studies relying on the pairwise correlations between
stocks, we argue that stocks are naturally connected as a collective group, and
introduce the hypergraph structures to jointly characterize the stock
group-wise relationships of industry-belonging and fund-holding. A novel
hypergraph tri-attention network (HGTAN) is proposed to augment the hypergraph
convolutional networks with a hierarchical organization of intra-hyperedge,
inter-hyperedge, and inter-hypergraph attention modules. In this manner, HGTAN
adaptively determines the importance of nodes, hyperedges, and hypergraphs
during the information propagation among stocks, so that the potential
synergies between stock movements can be fully exploited. Extensive experiments
on real-world data demonstrate the effectiveness of our approach. Also, the
results of investment simulation show that our approach can achieve a more
desirable risk-adjusted return. The data and codes of our work have been
released at this https URL.

    

### [[2107.14035] ProtoTransformer: A Meta-Learning Approach to Providing Student Feedback](http://arxiv.org/abs/2107.14035)


  High-quality computer science education is limited by the difficulty of
providing instructor feedback to students at scale. While this feedback could
in principle be automated, supervised approaches to predicting the correct
feedback are bottlenecked by the intractability of annotating large quantities
of student code. In this paper, we instead frame the problem of providing
feedback as few-shot classification, where a meta-learner adapts to give
feedback to student code on a new programming question from just a few examples
annotated by instructors. Because data for meta-training is limited, we propose
a number of amendments to the typical few-shot learning framework, including
task augmentation to create synthetic tasks, and additional side information to
build stronger priors about each task. These additions are combined with a
transformer architecture to embed discrete sequences (e.g. code) to a
prototypical representation of a feedback class label. On a suite of few-shot
natural language processing tasks, we match or outperform state-of-the-art
performance. Then, on a collection of student solutions to exam questions from
an introductory university course, we show that our approach reaches an average
precision of 88% on unseen questions, surpassing the 82% precision of teaching
assistants. Our approach was successfully deployed to deliver feedback to
16,000 student exam-solutions in a programming course offered by a tier 1
university. This is, to the best of our knowledge, the first successful
deployment of a machine learning based feedback to open-ended student code.

    

### [[2107.14037] Machine Learning and Deep Learning Methods for Building Intelligent Systems in Medicine and Drug Discovery: A Comprehensive Survey](http://arxiv.org/abs/2107.14037)


  With the advancements in computer technology, there is a rapid development of
intelligent systems to understand the complex relationships in data to make
predictions and classifications. Artificail Intelligence based framework is
rapidly revolutionizing the healthcare industry. These intelligent systems are
built with machine learning and deep learning based robust models for early
diagnosis of diseases and demonstrates a promising supplementary diagnostic
method for frontline clinical doctors and surgeons. Machine Learning and Deep
Learning based systems can streamline and simplify the steps involved in
diagnosis of diseases from clinical and image-based data, thus providing
significant clinician support and workflow optimization. They mimic human
cognition and are even capable of diagnosing diseases that cannot be diagnosed
with human intelligence. This paper focuses on the survey of machine learning
and deep learning applications in across 16 medical specialties, namely Dental
medicine, Haematology, Surgery, Cardiology, Pulmonology, Orthopedics,
Radiology, Oncology, General medicine, Psychiatry, Endocrinology, Neurology,
Dermatology, Hepatology, Nephrology, Ophthalmology, and Drug discovery. In this
paper along with the survey, we discuss the advancements of medical practices
with these systems and also the impact of these systems on medical
professionals.

    

### [[2107.14038] Point-Cloud Deep Learning of Porous Media for Permeability Prediction](http://arxiv.org/abs/2107.14038)


  We propose a novel deep learning framework for predicting permeability of
porous media from their digital images. Unlike convolutional neural networks,
instead of feeding the whole image volume as inputs to the network, we model
the boundary between solid matrix and pore spaces as point clouds and feed them
as inputs to a neural network based on the PointNet architecture. This approach
overcomes the challenge of memory restriction of graphics processing units and
its consequences on the choice of batch size, and convergence. Compared to
convolutional neural networks, the proposed deep learning methodology provides
freedom to select larger batch sizes, due to reducing significantly the size of
network inputs. Specifically, we use the classification branch of PointNet and
adjust it for a regression task. As a test case, two and three dimensional
synthetic digital rock images are considered. We investigate the effect of
different components of our neural network on its performance. We compare our
deep learning strategy with a convolutional neural network from various
perspectives, specifically for maximum possible batch size. We inspect the
generalizability of our network by predicting the permeability of real-world
rock samples as well as synthetic digital rocks that are statistically
different from the samples used during training. The network predicts the
permeability of digital rocks a few thousand times faster than a Lattice
Boltzmann solver with a high level of prediction accuracy.

    

### [[2107.14053] Few-Shot and Continual Learning with Attentive Independent Mechanisms](http://arxiv.org/abs/2107.14053)


  Deep neural networks (DNNs) are known to perform well when deployed to test
distributions that shares high similarity with the training distribution.
Feeding DNNs with new data sequentially that were unseen in the training
distribution has two major challenges -- fast adaptation to new tasks and
catastrophic forgetting of old tasks. Such difficulties paved way for the
on-going research on few-shot learning and continual learning. To tackle these
problems, we introduce Attentive Independent Mechanisms (AIM). We incorporate
the idea of learning using fast and slow weights in conjunction with the
decoupling of the feature extraction and higher-order conceptual learning of a
DNN. AIM is designed for higher-order conceptual learning, modeled by a mixture
of experts that compete to learn independent concepts to solve a new task. AIM
is a modular component that can be inserted into existing deep learning
frameworks. We demonstrate its capability for few-shot learning by adding it to
SIB and trained on MiniImageNet and CIFAR-FS, showing significant improvement.
AIM is also applied to ANML and OML trained on Omniglot, CIFAR-100 and
MiniImageNet to demonstrate its capability in continual learning. Code made
publicly available at this https URL.

    

### [[2107.14060] Multi-objective optimization and explanation for stroke risk assessment in Shanxi province](http://arxiv.org/abs/2107.14060)


  Stroke is the top leading causes of death in China (Zhou et al. The Lancet
2019). A dataset from Shanxi Province is used to identify the risk of each
patient's at four states low/medium/high/attack and provide the state
transition tendency through a SHAP DeepExplainer. To improve the accuracy on an
imbalance sample set, the Quadratic Interactive Deep Neural Network (QIDNN)
model is first proposed by flexible selecting and appending of quadratic
interactive features. The experimental results showed that the QIDNN model with
7 interactive features achieve the state-of-art accuracy $83.25\%$. Blood
pressure, physical inactivity, smoking, weight and total cholesterol are the
top five important features. Then, for the sake of high recall on the most
urgent state, attack state, the stroke occurrence prediction is taken as an
auxiliary objective to benefit from multi-objective optimization. The
prediction accuracy was promoted, meanwhile the recall of the attack state was
improved by $24.9\%$ (to $84.83\%$) compared to QIDNN (from $67.93\%$) with
same features. The prediction model and analysis tool in this paper not only
gave the theoretical optimized prediction method, but also provided the
attribution explanation of risk states and transition direction of each
patient, which provided a favorable tool for doctors to analyze and diagnose
the disease.

    

### [[2107.14061] The Need and Status of Sea Turtle Conservation and Survey of Associated Computer Vision Advances](http://arxiv.org/abs/2107.14061)


  For over hundreds of millions of years, sea turtles and their ancestors have
swum in the vast expanses of the ocean. They have undergone a number of
evolutionary changes, leading to speciation and sub-speciation. However, in the
past few decades, some of the most notable forces driving the genetic variance
and population decline have been global warming and anthropogenic impact
ranging from large-scale poaching, collecting turtle eggs for food, besides
dumping trash including plastic waste into the ocean. This leads to severe
detrimental effects in the sea turtle population, driving them to extinction.
This research focusses on the forces causing the decline in sea turtle
population, the necessity for the global conservation efforts along with its
successes and failures, followed by an in-depth analysis of the modern advances
in detection and recognition of sea turtles, involving Machine Learning and
Computer Vision systems, aiding the conservation efforts.

    

### [[2107.14062] Structure and Performance of Fully Connected Neural Networks: Emerging Complex Network Properties](http://arxiv.org/abs/2107.14062)


  Understanding the behavior of Artificial Neural Networks is one of the main
topics in the field recently, as black-box approaches have become usual since
the widespread of deep learning. Such high-dimensional models may manifest
instabilities and weird properties that resemble complex systems. Therefore, we
propose Complex Network (CN) techniques to analyze the structure and
performance of fully connected neural networks. For that, we build a dataset
with 4 thousand models and their respective CN properties. They are employed in
a supervised classification setup considering four vision benchmarks. Each
neural network is approached as a weighted and undirected graph of neurons and
synapses, and centrality measures are computed after training. Results show
that these measures are highly related to the network classification
performance. We also propose the concept of Bag-Of-Neurons (BoN), a CN-based
approach for finding topological signatures linking similar neurons. Results
suggest that six neuronal types emerge in such networks, independently of the
target domain, and are distributed differently according to classification
accuracy. We also tackle specific CN properties related to performance, such as
higher subgraph centrality on lower-performing models. Our findings suggest
that CN properties play a critical role in the performance of fully connected
neural networks, with topological patterns emerging independently on a wide
range of models.

    

### [[2107.14070] Machine Learning Advances aiding Recognition and Classification of Indian Monuments and Landmarks](http://arxiv.org/abs/2107.14070)


  Tourism in India plays a quintessential role in the country's economy with an
estimated 9.2% GDP share for the year 2018. With a yearly growth rate of 6.2%,
the industry holds a huge potential for being the primary driver of the economy
as observed in the nations of the Middle East like the United Arab Emirates.
The historical and cultural diversity exhibited throughout the geography of the
nation is a unique spectacle for people around the world and therefore serves
to attract tourists in tens of millions in number every year. Traditionally,
tour guides or academic professionals who study these heritage monuments were
responsible for providing information to the visitors regarding their
architectural and historical significance. However, unfortunately this system
has several caveats when considered on a large scale such as unavailability of
sufficient trained people, lack of accurate information, failure to convey the
richness of details in an attractive format etc. Recently, machine learning
approaches revolving around the usage of monument pictures have been shown to
be useful for rudimentary analysis of heritage sights. This paper serves as a
survey of the research endeavors undertaken in this direction which would
eventually provide insights for building an automated decision system that
could be utilized to make the experience of tourism in India more modernized
for visitors.

    

### [[2107.14077] A Fair and Ethical Healthcare Artificial Intelligence System for Monitoring Driver Behavior and Preventing Road Accidents](http://arxiv.org/abs/2107.14077)


  This paper presents a new approach to prevent transportation accidents and
monitor driver's behavior using a healthcare AI system that incorporates
fairness and ethics. Dangerous medical cases and unusual behavior of the driver
are detected. Fairness algorithm is approached in order to improve
decision-making and address ethical issues such as privacy issues, and to
consider challenges that appear in the wild within AI in healthcare and
driving. A healthcare professional will be alerted about any unusual activity,
and the driver's location when necessary, is provided in order to enable the
healthcare professional to immediately help to the unstable driver. Therefore,
using the healthcare AI system allows for accidents to be predicted and thus
prevented and lives may be saved based on the built-in AI system inside the
vehicle which interacts with the ER system.

    

### [[2107.14094] Day-to-day and seasonal regularity of network passenger delay for metro networks](http://arxiv.org/abs/2107.14094)


  In an effort to improve user satisfaction and transit image, transit service
providers worldwide offer delay compensations. Smart card data enables the
estimation of passenger delays throughout the network and aid in monitoring
service performance. Notwithstanding, in order to prioritize measures for
improving service reliability and hence reducing passenger delays, it is
paramount to identify the system components - stations and track segments -
where most passenger delay occurs. To this end, we propose a novel method for
estimating network passenger delay from individual trajectories. We decompose
the delay along a passenger trajectory into its corresponding track segment
delay, initial waiting time and transfer delay. We distinguish between two
different types of passenger delay in relation to the public transit network:
average passenger delay and total passenger delay. We employ temporal
clustering on these two quantities to reveal daily and seasonal regularity in
delay patterns of the transit network. The estimation and clustering methods
are demonstrated on one year of data from Washington metro network. The data
consists of schedule information and smart card data which includes
passenger-train assignment of the metro network for the months of August 2017
to August 2018. Our findings show that the average passenger delay is
relatively stable throughout the day. The temporal clustering reveals
pronounced and recurrent and thus predictable daily and weekly patterns with
distinct characteristics for certain months.

    

### [[2107.14110] Enhancing Adversarial Robustness via Test-time Transformation Ensembling](http://arxiv.org/abs/2107.14110)


  Deep learning models are prone to being fooled by imperceptible perturbations
known as adversarial attacks. In this work, we study how equipping models with
Test-time Transformation Ensembling (TTE) can work as a reliable defense
against such attacks. While transforming the input data, both at train and test
times, is known to enhance model performance, its effects on adversarial
robustness have not been studied. Here, we present a comprehensive empirical
study of the impact of TTE, in the form of widely-used image transforms, on
adversarial robustness. We show that TTE consistently improves model robustness
against a variety of powerful attacks without any need for re-training, and
that this improvement comes at virtually no trade-off with accuracy on clean
samples. Finally, we show that the benefits of TTE transfer even to the
certified robustness domain, in which TTE provides sizable and consistent
improvements.

    

### [[2107.14135] Modifications of FastICA in Convolutive Blind Source Separation](http://arxiv.org/abs/2107.14135)


  Convolutive blind source separation (BSS) is intended to recover the unknown
components from their convolutive mixtures. Contrary to the contrast functions
used in instantaneous cases, the spatial-temporal prewhitening stage and the
para-unitary filters constraint are difficult to implement in a convolutive
context. In this paper, we propose several modifications of FastICA to
alleviate these difficulties. Our method performs the simple prewhitening step
on convolutive mixtures prior to the separation and optimizes the contrast
function under the diagonalization constraint implemented by single value
decomposition (SVD). Numerical simulations are implemented to verify the
performance of the proposed method.

    

### [[2107.14151] Modern Non-Linear Function-on-Function Regression](http://arxiv.org/abs/2107.14151)


  We introduce a new class of non-linear function-on-function regression models
for functional data using neural networks. We propose a framework using a
hidden layer consisting of continuous neurons, called a continuous hidden
layer, for functional response modeling and give two model fitting strategies,
Functional Direct Neural Network (FDNN) and Functional Basis Neural Network
(FBNN). Both are designed explicitly to exploit the structure inherent in
functional data and capture the complex relations existing between the
functional predictors and the functional response. We fit these models by
deriving functional gradients and implement regularization techniques for more
parsimonious results. We demonstrate the power and flexibility of our proposed
method in handling complex functional models through extensive simulation
studies as well as real data examples.

    

### [[2107.14153] Semi-Supervised Active Learning with Temporal Output Discrepancy](http://arxiv.org/abs/2107.14153)


  While deep learning succeeds in a wide range of tasks, it highly depends on
the massive collection of annotated data which is expensive and time-consuming.
To lower the cost of data annotation, active learning has been proposed to
interactively query an oracle to annotate a small proportion of informative
samples in an unlabeled dataset. Inspired by the fact that the samples with
higher loss are usually more informative to the model than the samples with
lower loss, in this paper we present a novel deep active learning approach that
queries the oracle for data annotation when the unlabeled sample is believed to
incorporate high loss. The core of our approach is a measurement Temporal
Output Discrepancy (TOD) that estimates the sample loss by evaluating the
discrepancy of outputs given by models at different optimization steps. Our
theoretical investigation shows that TOD lower-bounds the accumulated sample
loss thus it can be used to select informative unlabeled samples. On basis of
TOD, we further develop an effective unlabeled data sampling strategy as well
as an unsupervised learning criterion that enhances model performance by
incorporating the unlabeled data. Due to the simplicity of TOD, our active
learning approach is efficient, flexible, and task-agnostic. Extensive
experimental results demonstrate that our approach achieves superior
performances than the state-of-the-art active learning methods on image
classification and semantic segmentation tasks.

    

### [[2107.14171] Tianshou: a Highly Modularized Deep Reinforcement Learning Library](http://arxiv.org/abs/2107.14171)


  We present Tianshou, a highly modularized python library for deep
reinforcement learning (DRL) that uses PyTorch as its backend. Tianshou aims to
provide building blocks to replicate common RL experiments and has officially
supported more than 15 classic algorithms succinctly. To facilitate related
research and prove Tianshou's reliability, we release Tianshou's benchmark of
MuJoCo environments, covering 9 classic algorithms and 9/13 Mujoco tasks with
state-of-the-art performance. We open-sourced Tianshou at
this https URL, which has received over 3k stars and
become one of the most popular PyTorch-based DRL libraries.

    

### [[2107.14194] On the combined effect of class imbalance and concept complexity in deep learning](http://arxiv.org/abs/2107.14194)


  Structural concept complexity, class overlap, and data scarcity are some of
the most important factors influencing the performance of classifiers under
class imbalance conditions. When these effects were uncovered in the early
2000s, understandably, the classifiers on which they were demonstrated belonged
to the classical rather than Deep Learning categories of approaches. As Deep
Learning is gaining ground over classical machine learning and is beginning to
be used in critical applied settings, it is important to assess systematically
how well they respond to the kind of challenges their classical counterparts
have struggled with in the past two decades. The purpose of this paper is to
study the behavior of deep learning systems in settings that have previously
been deemed challenging to classical machine learning systems to find out
whether the depth of the systems is an asset in such settings. The results in
both artificial and real-world image datasets (MNIST Fashion, CIFAR-10) show
that these settings remain mostly challenging for Deep Learning systems and
that deeper architectures seem to help with structural concept complexity but
not with overlap challenges in simple artificial domains. Data scarcity is not
overcome by deeper layers, either. In the real-world image domains, where
overfitting is a greater concern than in the artificial domains, the advantage
of deeper architectures is less obvious: while it is observed in certain cases,
it is quickly cancelled as models get deeper and perform worse than their
shallower counterparts.

    

### [[2107.14203] Did the Model Change? Efficiently Assessing Machine Learning API Shifts](http://arxiv.org/abs/2107.14203)


  Machine learning (ML) prediction APIs are increasingly widely used. An ML API
can change over time due to model updates or retraining. This presents a key
challenge in the usage of the API because it is often not clear to the user if
and how the ML model has changed. Model shifts can affect downstream
application performance and also create oversight issues (e.g. if consistency
is desired). In this paper, we initiate a systematic investigation of ML API
shifts. We first quantify the performance shifts from 2020 to 2021 of popular
ML APIs from Google, Microsoft, Amazon, and others on a variety of datasets. We
identified significant model shifts in 12 out of 36 cases we investigated.
Interestingly, we found several datasets where the API's predictions became
significantly worse over time. This motivated us to formulate the API shift
assessment problem at a more fine-grained level as estimating how the API
model's confusion matrix changes over time when the data distribution is
constant. Monitoring confusion matrix shifts using standard random sampling can
require a large number of samples, which is expensive as each API call costs a
fee. We propose a principled adaptive sampling algorithm, MASA, to efficiently
estimate confusion matrix shifts. MASA can accurately estimate the confusion
matrix shifts in commercial ML APIs using up to 90% fewer samples compared to
random sampling. This work establishes ML API shifts as an important problem to
study and provides a cost-effective approach to monitor such shifts.

    

### [[2107.14226] Learning more skills through optimistic exploration](http://arxiv.org/abs/2107.14226)


  Unsupervised skill learning objectives (Gregor et al., 2016, Eysenbach et
al., 2018) allow agents to learn rich repertoires of behavior in the absence of
extrinsic rewards. They work by simultaneously training a policy to produce
distinguishable latent-conditioned trajectories, and a discriminator to
evaluate distinguishability by trying to infer latents from trajectories. The
hope is for the agent to explore and master the environment by encouraging each
skill (latent) to reliably reach different states. However, an inherent
exploration problem lingers: when a novel state is actually encountered, the
discriminator will necessarily not have seen enough training data to produce
accurate and confident skill classifications, leading to low intrinsic reward
for the agent and effective penalization of the sort of exploration needed to
actually maximize the objective. To combat this inherent pessimism towards
exploration, we derive an information gain auxiliary objective that involves
training an ensemble of discriminators and rewarding the policy for their
disagreement. Our objective directly estimates the epistemic uncertainty that
comes from the discriminator not having seen enough training examples, thus
providing an intrinsic reward more tailored to the true objective compared to
pseudocount-based methods (Burda et al., 2019). We call this exploration bonus
discriminator disagreement intrinsic reward, or DISDAIN. We demonstrate
empirically that DISDAIN improves skill learning both in a tabular grid world
(Four Rooms) and the 57 games of the Atari Suite (from pixels). Thus, we
encourage researchers to treat pessimism with DISDAIN.

    

### [[2107.14228] Open-World Entity Segmentation](http://arxiv.org/abs/2107.14228)


  We introduce a new image segmentation task, termed Entity Segmentation (ES)
with the aim to segment all visual entities in an image without considering
semantic category labels. It has many practical applications in image
manipulation/editing where the segmentation mask quality is typically crucial
but category labels are less important. In this setting, all
semantically-meaningful segments are equally treated as categoryless entities
and there is no thing-stuff distinction. Based on our unified entity
representation, we propose a center-based entity segmentation framework with
two novel modules to improve mask quality. Experimentally, both our new task
and framework demonstrate superior advantages as against existing work. In
particular, ES enables the following: (1) merging multiple datasets to form a
large training set without the need to resolve label conflicts; (2) any model
trained on one dataset can generalize exceptionally well to other datasets with
unseen domains. Our code is made publicly available at
this https URL.

    

### [[2107.14229] Guided Disentanglement in Generative Networks](http://arxiv.org/abs/2107.14229)


  Image-to-image translation (i2i) networks suffer from entanglement effects in
presence of physics-related phenomena in target domain (such as occlusions,
fog, etc), thus lowering the translation quality and variability. In this
paper, we present a comprehensive method for disentangling physics-based traits
in the translation, guiding the learning process with neural or physical
models. For the latter, we integrate adversarial estimation and genetic
algorithms to correctly achieve disentanglement. The results show our approach
dramatically increase performances in many challenging scenarios for image
translation.

    

### [[1801.02982] How To Make the Gradients Small Stochastically: Even Faster Convex and Nonconvex SGD](http://arxiv.org/abs/1801.02982)


  Stochastic gradient descent (SGD) gives an optimal convergence rate when
minimizing convex stochastic objectives $f(x)$. However, in terms of making the
gradients small, the original SGD does not give an optimal rate, even when
$f(x)$ is convex.
If $f(x)$ is convex, to find a point with gradient norm $\varepsilon$, we
design an algorithm SGD3 with a near-optimal rate
$\tilde{O}(\varepsilon^{-2})$, improving the best known rate
$O(\varepsilon^{-8/3})$ of [18].
If $f(x)$ is nonconvex, to find its $\varepsilon$-approximate local minimum,
we design an algorithm SGD5 with rate $\tilde{O}(\varepsilon^{-3.5})$, where
previously SGD variants only achieve $\tilde{O}(\varepsilon^{-4})$ [6, 15, 33].
This is no slower than the best known stochastic version of Newton's method in
all parameter regimes [30].

    

### [[1905.10848] Learning Gaussian DAGs from Network Data](http://arxiv.org/abs/1905.10848)


  Structural learning of directed acyclic graphs (DAGs) or Bayesian networks
has been studied extensively under the assumption that data are independent. We
propose a new Gaussian DAG model for dependent data which assumes the
observations are correlated according to an undirected network. Under this
model, we develop a method to estimate the DAG structure given a topological
ordering of the nodes. The proposed method jointly estimates the Bayesian
network and the correlations among observations by optimizing a scoring
function based on penalized likelihood. We show that under some mild
conditions, the proposed method produces consistent estimators after one
iteration. Extensive numerical experiments also demonstrate that by jointly
estimating the DAG structure and the sample correlation, our method achieves
much higher accuracy in structure learning. When the node ordering is unknown,
through experiments on synthetic and real data, we show that our algorithm can
be used to estimate the correlations between samples, with which we can
de-correlate the dependent data to significantly improve the performance of
classical DAG learning methods.

    

### [[1906.01005] Gated recurrent units viewed through the lens of continuous time dynamical systems](http://arxiv.org/abs/1906.01005)


  Gated recurrent units (GRUs) are specialized memory elements for building
recurrent neural networks. Despite their incredible success on various tasks,
including extracting dynamics underlying neural data, little is understood
about the specific dynamics representable in a GRU network. As a result, it is
both difficult to know a priori how successful a GRU network will perform on a
given task, and also their capacity to mimic the underlying behavior of their
biological counterparts. Using a continuous time analysis, we gain intuition on
the inner workings of GRU networks. We restrict our presentation to low
dimensions, allowing for a comprehensive visualization. We found a surprisingly
rich repertoire of dynamical features that includes stable limit cycles
(nonlinear oscillations), multi-stable dynamics with various topologies, and
homoclinic bifurcations. At the same time we were unable to train GRU networks
to produce continuous attractors, which are hypothesized to exist in biological
neural networks. We contextualize the usefulness of different kinds of observed
dynamics and support our claims experimentally.

    

### [[1908.00814] On the Merge of k-NN Graph](http://arxiv.org/abs/1908.00814)


  k-nearest neighbor graph is a fundamental data structure in many disciplines
such as information retrieval, data-mining, pattern recognition, and machine
learning, etc. In the literature, considerable research has been focusing on
how to efficiently build an approximate k-nearest neighbor graph (k-NN graph)
for a fixed dataset. Unfortunately, a closely related issue of how to merge two
existing k-NN graphs has been overlooked. In this paper, we address the issue
of k-NN graph merging in two different scenarios. In the first scenario, a
symmetric merge algorithm is proposed to combine two approximate k-NN graphs.
The algorithm facilitates large-scale processing by the efficient merging of
k-NN graphs that are produced in parallel. In the second scenario, a joint
merge algorithm is proposed to expand an existing k-NN graph with a raw
dataset. The algorithm enables the incremental construction of a hierarchical
approximate k-NN graph. Superior performance is attained when leveraging the
hierarchy for NN search of various data types, dimensionality, and distance
measures.

    

### [[1908.01656] Distributed Deep Convolutional Neural Networks for the Internet-of-Things](http://arxiv.org/abs/1908.01656)


  Severe constraints on memory and computation characterizing the
Internet-of-Things (IoT) units may prevent the execution of Deep Learning
(DL)-based solutions, which typically demand large memory and high processing
load. In order to support a real-time execution of the considered DL model at
the IoT unit level, DL solutions must be designed having in mind constraints on
memory and processing capability exposed by the chosen IoT technology. In this
paper, we introduce a design methodology aiming at allocating the execution of
Convolutional Neural Networks (CNNs) on a distributed IoT application. Such a
methodology is formalized as an optimization problem where the latency between
the data-gathering phase and the subsequent decision-making one is minimized,
within the given constraints on memory and processing load at the units level.
The methodology supports multiple sources of data as well as multiple CNNs in
execution on the same IoT system allowing the design of CNN-based applications
demanding autonomy, low decision-latency, and high Quality-of-Service.

    

### [[1909.03194] On Sample Complexity Upper and Lower Bounds for Exact Ranking from Noisy Comparisons](http://arxiv.org/abs/1909.03194)


  This paper studies the problem of finding the exact ranking from noisy
comparisons. A comparison over a set of $m$ items produces a noisy outcome
about the most preferred item, and reveals some information about the ranking.
By repeatedly and adaptively choosing items to compare, we want to fully rank
the items with a certain confidence, and use as few comparisons as possible.
Different from most previous works, in this paper, we have three main
novelties: (i) compared to prior works, our upper bounds (algorithms) and lower
bounds on the sample complexity (aka number of comparisons) require the minimal
assumptions on the instances, and are not restricted to specific models; (ii)
we give lower bounds and upper bounds on instances with unequal noise levels;
and (iii) this paper aims at the exact ranking without knowledge on the
instances, while most of the previous works either focus on approximate
rankings or study exact ranking but require prior knowledge. We first derive
lower bounds for pairwise ranking (i.e., compare two items each time), and then
propose (nearly) optimal pairwise ranking algorithms. We further make
extensions to listwise ranking (i.e., comparing multiple items each time).
Numerical results also show our improvements against the state of the art.

    

### [[1910.01845] The Complexity of Finding Stationary Points with Stochastic Gradient Descent](http://arxiv.org/abs/1910.01845)


  We study the iteration complexity of stochastic gradient descent (SGD) for
minimizing the gradient norm of smooth, possibly nonconvex functions. We
provide several results, implying that the $\mathcal{O}(\epsilon^{-4})$ upper
bound of Ghadimi and Lan~\cite{ghadimi2013stochastic} (for making the average
gradient norm less than $\epsilon$) cannot be improved upon, unless a
combination of additional assumptions is made. Notably, this holds even if we
limit ourselves to convex quadratic functions. We also show that for nonconvex
functions, the feasibility of minimizing gradients with SGD is surprisingly
sensitive to the choice of optimality criteria.

    

### [[1910.10692] Deterministic tensor completion with hypergraph expanders](http://arxiv.org/abs/1910.10692)


  We provide a novel analysis of low-rank tensor completion based on hypergraph
expanders. As a proxy for rank, we minimize the max-quasinorm of the tensor,
which generalizes the max-norm for matrices. Our analysis is deterministic and
shows that the number of samples required to approximately recover an order-$t$
tensor with at most $n$ entries per dimension is linear in $n$, under the
assumption that the rank and order of the tensor are $O(1)$. As steps in our
proof, we find a new expander mixing lemma for a $t$-partite, $t$-uniform
regular hypergraph model, and prove several new properties about tensor
max-quasinorm. To the best of our knowledge, this is the first deterministic
analysis of tensor completion. We develop a practical algorithm that solves a
relaxed version of the max-quasinorm minimization problem, and we demonstrate
its efficacy with numerical experiments.

    

### [[2003.08773] Do CNNs Encode Data Augmentations?](http://arxiv.org/abs/2003.08773)


  Data augmentations are important ingredients in the recipe for training
robust neural networks, especially in computer vision. A fundamental question
is whether neural network features encode data augmentation transformations. To
answer this question, we introduce a systematic approach to investigate which
layers of neural networks are the most predictive of augmentation
transformations. Our approach uses features in pre-trained vision models with
minimal additional processing to predict common properties transformed by
augmentation (scale, aspect ratio, hue, saturation, contrast, and brightness).
Surprisingly, neural network features not only predict data augmentation
transformations, but they predict many transformations with high accuracy.
After validating that neural networks encode features corresponding to
augmentation transformations, we show that these features are encoded in the
early layers of modern CNNs, though the augmentation signal fades in deeper
layers.

    

### [[2006.09858] Geometry of Similarity Comparisons](http://arxiv.org/abs/2006.09858)


  Many data analysis problems can be cast as distance geometry problems in
\emph{space forms} -- Euclidean, spherical, or hyperbolic spaces. Often,
absolute distance measurements are often unreliable or simply unavailable and
only proxies to absolute distances in the form of similarities are available.
Hence we ask the following: Given only \emph{comparisons} of similarities
amongst a set of entities, what can be said about the geometry of the
underlying space form? To study this question, we introduce the notions of the
\textit{ordinal capacity} of a target space form and \emph{ordinal spread} of
the similarity measurements. The latter is an indicator of complex patterns in
the measurements, while the former quantifies the capacity of a space form to
accommodate a set of measurements with a specific ordinal spread profile. We
prove that the ordinal capacity of a space form is related to its dimension and
the sign of its curvature. This leads to a lower bound on the Euclidean and
spherical embedding dimension of what we term similarity graphs. More
importantly, we show that the statistical behavior of the ordinal spread random
variables defined on a similarity graph can be used to identify its underlying
space form. We support our theoretical claims with experiments on weighted
trees, single-cell RNA expression data and spherical cartographic measurements.

    

### [[2008.05101] FATNN: Fast and Accurate Ternary Neural Networks](http://arxiv.org/abs/2008.05101)


  Ternary Neural Networks (TNNs) have received much attention due to being
potentially orders of magnitude faster in inference, as well as more power
efficient, than full-precision counterparts. However, 2 bits are required to
encode the ternary representation with only 3 quantization levels leveraged. As
a result, conventional TNNs have similar memory consumption and speed compared
with the standard 2-bit models, but have worse representational capability.
Moreover, there is still a significant gap in accuracy between TNNs and
full-precision networks, hampering their deployment to real applications. To
tackle these two challenges, in this work, we first show that, under some mild
constraints, computational complexity of the ternary inner product can be
reduced by a factor of 2. Second, to mitigate the performance gap, we
elaborately design an implementation-dependent ternary quantization algorithm.
The proposed framework is termed Fast and Accurate Ternary Neural Networks
(FATNN). Experiments on image classification demonstrate that our FATNN
surpasses the state-of-the-arts by a significant margin in accuracy. More
importantly, speedup evaluation compared with various precisions is analyzed on
several platforms, which serves as a strong benchmark for further research.

    

### [[2009.05261] End-to-end Learning for OFDM: From Neural Receivers to Pilotless Communication](http://arxiv.org/abs/2009.05261)


  Previous studies have demonstrated that end-to-end learning enables
significant shaping gains over additive white Gaussian noise (AWGN) channels.
However, its benefits have not yet been quantified over realistic wireless
channel models. This work aims to fill this gap by exploring the gains of
end-to-end learning over a frequency- and time-selective fading channel using
orthogonal frequency division multiplexing (OFDM). With imperfect channel
knowledge at the receiver, the shaping gains observed on AWGN channels vanish.
Nonetheless, we identify two other sources of performance improvements. The
first comes from a neural network (NN)-based receiver operating over a large
number of subcarriers and OFDM symbols which allows to significantly reduce the
number of orthogonal pilots without loss of bit error rate (BER). The second
comes from entirely eliminating orthognal pilots by jointly learning a neural
receiver together with either superimposed pilots (SIPs), linearly combined
with conventional quadrature amplitude modulation (QAM), or an optimized
constellation geometry. The learned geometry works for a wide range of
signal-to-noise ratios (SNRs), Doppler and delay spreads, has zero mean and
does hence not contain any form of superimposed pilots. Both schemes achieve
the same BER as the pilot-based baseline with around 7% higher throughput.
Thus, we believe that a jointly learned transmitter and receiver are a very
interesting component for beyond-5G communication systems which could remove
the need and associated control overhead for demodulation reference signals
(DMRSs).

    

### [[2009.14119] Asymmetric Loss For Multi-Label Classification](http://arxiv.org/abs/2009.14119)


  In a typical multi-label setting, a picture contains on average few positive
labels, and many negative ones. This positive-negative imbalance dominates the
optimization process, and can lead to under-emphasizing gradients from positive
labels during training, resulting in poor accuracy. In this paper, we introduce
a novel asymmetric loss ("ASL"), which operates differently on positive and
negative samples. The loss enables to dynamically down-weights and
hard-thresholds easy negative samples, while also discarding possibly
mislabeled samples. We demonstrate how ASL can balance the probabilities of
different samples, and how this balancing is translated to better mAP scores.
With ASL, we reach state-of-the-art results on multiple popular multi-label
datasets: MS-COCO, Pascal-VOC, NUS-WIDE and Open Images. We also demonstrate
ASL applicability for other tasks, such as single-label classification and
object detection. ASL is effective, easy to implement, and does not increase
the training time or complexity.
Implementation is available at: this https URL.

    

### [[2010.11270] Learning second order coupled differential equations that are subject to non-conservative forces](http://arxiv.org/abs/2010.11270)


  In this article we address the question whether it is possible to learn the
differential equations describing the physical properties of a dynamical
system, subject to non-conservative forces, from observations of its realspace
trajectory(ies) only. We introduce a network that incorporates a difference
approximation for the second order derivative in terms of residual connections
between convolutional blocks, whose shared weights represent the coefficients
of a second order ordinary differential equation. We further combine this
solver-like architecture with a convolutional network, capable of learning the
relation between trajectories of coupled oscillators and therefore allows us to
make a stable forecast even if the system is only partially observed. We
optimize this map together with the solver network, while sharing their
weights, to form a powerful framework capable of learning the complex physical
properties of a dissipative dynamical system.

    

### [[2011.08055] Scalable Reinforcement Learning Policies for Multi-Agent Control](http://arxiv.org/abs/2011.08055)


  We develop a Multi-Agent Reinforcement Learning (MARL) method to learn
scalable control policies for target tracking. Our method can handle an
arbitrary number of pursuers and targets; we show results for tasks consisting
up to 1000 pursuers tracking 1000 targets. We use a decentralized,
partially-observable Markov Decision Process framework to model pursuers as
agents receiving partial observations (range and bearing) about targets which
move using fixed, unknown policies. An attention mechanism is used to
parameterize the value function of the agents; this mechanism allows us to
handle an arbitrary number of targets. Entropy-regularized off-policy RL
methods are used to train a stochastic policy, and we discuss how it enables a
hedging behavior between pursuers that leads to a weak form of cooperation in
spite of completely decentralized control execution. We further develop a
masking heuristic that allows training on smaller problems with few
pursuers-targets and execution on much larger problems. Thorough simulation
experiments, ablation studies, and comparisons to state of the art algorithms
are performed to study the scalability of the approach and robustness of
performance to varying numbers of agents and targets.

    

### [[2011.10829] On the Convergence of Reinforcement Learning in Nonlinear Continuous State Space Problems](http://arxiv.org/abs/2011.10829)


  We consider the problem of Reinforcement Learning for nonlinear stochastic
dynamical systems. We show that in the RL setting, there is an inherent ``Curse
of Variance" in addition to Bellman's infamous ``Curse of Dimensionality", in
particular, we show that the variance in the solution grows
factorial-exponentially in the order of the approximation. A fundamental
consequence is that this precludes the search for anything other than ``local"
feedback solutions in RL, in order to control the explosive variance growth,
and thus, ensure accuracy. We further show that the deterministic optimal
control has a perturbation structure, in that the higher order terms do not
affect the calculation of lower order terms, which can be utilized in RL to get
accurate local solutions.

    

### [[2012.14415] Stochastic Approximation for Online Tensorial Independent Component Analysis](http://arxiv.org/abs/2012.14415)


  Independent component analysis (ICA) has been a popular dimension reduction
tool in statistical machine learning and signal processing. In this paper, we
present a convergence analysis for an online tensorial ICA algorithm, by
viewing the problem as a nonconvex stochastic approximation problem. For
estimating one component, we provide a dynamics-based analysis to prove that
our online tensorial ICA algorithm with a specific choice of stepsize achieves
a sharp finite-sample error bound. In particular, under a mild assumption on
the data-generating distribution and a scaling condition such that $d^4/T$ is
sufficiently small up to a polylogarithmic factor of data dimension $d$ and
sample size $T$, a sharp finite-sample error bound of $\tilde{O}(\sqrt{d/T})$
can be obtained.

    

### [[2012.15843] A Tale of Two Efficient and Informative Negative Sampling Distributions](http://arxiv.org/abs/2012.15843)


  Softmax classifiers with a very large number of classes naturally occur in
many applications such as natural language processing and information
retrieval. The calculation of full softmax is costly from the computational and
energy perspective. There have been various sampling approaches to overcome
this challenge, popularly known as negative sampling (NS). Ideally, NS should
sample negative classes from a distribution that is dependent on the input
data, the current parameters, and the correct positive class. Unfortunately,
due to the dynamically updated parameters and data samples, there is no
sampling scheme that is provably adaptive and samples the negative classes
efficiently. Therefore, alternative heuristics like random sampling, static
frequency-based sampling, or learning-based biased sampling, which primarily
trade either the sampling cost or the adaptivity of samples per iteration are
adopted. In this paper, we show two classes of distributions where the sampling
scheme is truly adaptive and provably generates negative samples in
near-constant time. Our implementation in C++ on CPU is significantly superior,
both in terms of wall-clock time and accuracy, compared to the most optimized
TensorFlow implementations of other popular negative sampling approaches on
powerful NVIDIA V100 GPU.

    

### [[2101.10143] Spectral Leakage and Rethinking the Kernel Size in CNNs](http://arxiv.org/abs/2101.10143)


  Convolutional layers in CNNs implement linear filters which decompose the
input into different frequency bands. However, most modern architectures
neglect standard principles of filter design when optimizing their model
choices regarding the size and shape of the convolutional kernel. In this work,
we consider the well-known problem of spectral leakage caused by windowing
artifacts in filtering operations in the context of CNNs. We show that the
small size of CNN kernels make them susceptible to spectral leakage, which may
induce performance-degrading artifacts. To address this issue, we propose the
use of larger kernel sizes along with the Hamming window function to alleviate
leakage in CNN architectures. We demonstrate improved classification accuracy
on multiple benchmark datasets including Fashion-MNIST, CIFAR-10, CIFAR-100 and
ImageNet with the simple use of a standard window function in convolutional
layers. Finally, we show that CNNs employing the Hamming window display
increased robustness against various adversarial attacks.

    

### [[2101.11815] Interpolating Classifiers Make Few Mistakes](http://arxiv.org/abs/2101.11815)


  This paper provides elementary analyses of the regret and generalization of
minimum-norm interpolating classifiers (MNIC). The MNIC is the function of
smallest Reproducing Kernel Hilbert Space norm that perfectly interpolates a
label pattern on a finite data set. We derive a mistake bound for MNIC and a
regularized variant that holds for all data sets. This bound follows from
elementary properties of matrix inverses. Under the assumption that the data is
independently and identically distributed, the mistake bound implies that MNIC
generalizes at a rate proportional to the norm of the interpolating solution
and inversely proportional to the number of data points. This rate matches
similar rates derived for margin classifiers and perceptrons. We derive several
plausible generative models where the norm of the interpolating classifier is
bounded or grows at a rate sublinear in $n$. We also show that as long as the
population class conditional distributions are sufficiently separable in total
variation, then MNIC generalizes with a fast rate.

    

### [[2102.06387] The Distributed Discrete Gaussian Mechanism for Federated Learning with Secure Aggregation](http://arxiv.org/abs/2102.06387)


  We consider training models on private data that are distributed across user
devices. To ensure privacy, we add on-device noise and use secure aggregation
so that only the noisy sum is revealed to the server. We present a
comprehensive end-to-end system, which appropriately discretizes the data and
adds discrete Gaussian noise before performing secure aggregation. We provide a
novel privacy analysis for sums of discrete Gaussians and carefully analyze the
effects of data quantization and modular summation arithmetic. Our theoretical
guarantees highlight the complex tension between communication, privacy, and
accuracy. Our extensive experimental results demonstrate that our solution is
essentially able to match the accuracy to central differential privacy with
less than 16 bits of precision per value.

    

### [[2102.12855] Modular Deep Reinforcement Learning for Continuous Motion Planning with Temporal Logic](http://arxiv.org/abs/2102.12855)


  This paper investigates the motion planning of autonomous dynamical systems
modeled by Markov decision processes (MDP) with unknown transition
probabilities over continuous state and action spaces. Linear temporal logic
(LTL) is used to specify high-level tasks over infinite horizon, which can be
converted into a limit deterministic generalized Büchi automaton (LDGBA) with
several accepting sets. The novelty is to design an embedded product MDP
(EP-MDP) between the LDGBA and the MDP by incorporating a synchronous
tracking-frontier function to record unvisited accepting sets of the automaton,
and to facilitate the satisfaction of the accepting conditions. The proposed
LDGBA-based reward shaping and discounting schemes for the model-free
reinforcement learning (RL) only depend on the EP-MDP states and can overcome
the issues of sparse rewards. Rigorous analysis shows that any RL method that
optimizes the expected discounted return is guaranteed to find an optimal
policy whose traces maximize the satisfaction probability. A modular deep
deterministic policy gradient (DDPG) is then developed to generate such
policies over continuous state and action spaces. The performance of our
framework is evaluated via an array of OpenAI gym environments.

    

### [[2102.12877] TELESTO: A Graph Neural Network Model for Anomaly Classification in Cloud Services](http://arxiv.org/abs/2102.12877)


  Deployment, operation and maintenance of large IT systems becomes
increasingly complex and puts human experts under extreme stress when problems
occur. Therefore, utilization of machine learning (ML) and artificial
intelligence (AI) is applied on IT system operation and maintenance -
summarized in the term AIOps. One specific direction aims at the recognition of
re-occurring anomaly types to enable remediation automation. However, due to IT
system specific properties, especially their frequent changes (e.g. software
updates, reconfiguration or hardware modernization), recognition of reoccurring
anomaly types is challenging. Current methods mainly assume a static
dimensionality of provided data. We propose a method that is invariant to
dimensionality changes of given data. Resource metric data such as CPU
utilization, allocated memory and others are modelled as multivariate time
series. The extraction of temporal and spatial features together with the
subsequent anomaly classification is realized by utilizing TELESTO, our novel
graph convolutional neural network (GCNN) architecture. The experimental
evaluation is conducted in a real-world cloud testbed deployment that is
hosting two applications. Classification results of injected anomalies on a
cassandra database node show that TELESTO outperforms the alternative GCNNs and
achieves an overall classification accuracy of 85.1%. Classification results
for the other nodes show accuracy values between 85% and 60%.

    

### [[2103.06254] Interpretable Machine Learning: Moving From Mythos to Diagnostics](http://arxiv.org/abs/2103.06254)


  Despite increasing interest in the field of Interpretable Machine Learning
(IML), a significant gap persists between the technical objectives targeted by
researchers' methods and the high-level goals of consumers' use cases. In this
work, we synthesize foundational work on IML methods and evaluation into an
actionable taxonomy. This taxonomy serves as a tool to conceptualize the gap
between researchers and consumers, illustrated by the lack of connections
between its methods and use cases components. It also provides the foundation
from which we describe a three-step workflow to better enable researchers and
consumers to work together to discover what types of methods are useful for
what use cases. Eventually, by building on the results generated from this
workflow, a more complete version of the taxonomy will increasingly allow
consumers to find relevant methods for their target use cases and researchers
to identify applicable use cases for their proposed methods.

    

### [[2104.14278] ReLearn: A Robust Machine Learning Framework in Presence of Missing Data for Multimodal Stress Detection from Physiological Signals](http://arxiv.org/abs/2104.14278)


  Continuous and multimodal stress detection has been performed recently
through wearable devices and machine learning algorithms. However, a well-known
and important challenge of working on physiological signals recorded by
conventional monitoring devices is missing data due to sensors insufficient
contact and interference by other equipment. This challenge becomes more
problematic when the user/patient is mentally or physically active or stressed
because of more frequent conscious or subconscious movements. In this paper, we
propose ReLearn, a robust machine learning framework for stress detection from
biomarkers extracted from multimodal physiological signals. ReLearn effectively
copes with missing data and outliers both at training and inference phases.
ReLearn, composed of machine learning models for feature selection, outlier
detection, data imputation, and classification, allows us to classify all
samples, including those with missing values at inference. In particular,
according to our experiments and stress database, while by discarding all
missing data, as a simplistic yet common approach, no prediction can be made
for 34% of the data at inference, our approach can achieve accurate
predictions, as high as 78%, for missing samples. Also, our experiments show
that the proposed framework obtains a cross-validation accuracy of 86.8% even
if more than 50% of samples within the features are missing.

    

### [[2105.02469] Point Cloud Audio Processing](http://arxiv.org/abs/2105.02469)


  Most audio processing pipelines involve transformations that act on
fixed-dimensional input representations of audio. For example, when using the
Short Time Fourier Transform (STFT) the DFT size specifies a fixed dimension
for the input representation. As a consequence, most audio machine learning
models are designed to process fixed-size vector inputs which often prohibits
the repurposing of learned models on audio with different sampling rates or
alternative representations. We note, however, that the intrinsic spectral
information in the audio signal is invariant to the choice of the input
representation or the sampling rate. Motivated by this, we introduce a novel
way of processing audio signals by treating them as a collection of points in
feature space, and we use point cloud machine learning models that give us
invariance to the choice of representation parameters, such as DFT size or the
sampling rate. Additionally, we observe that these methods result in smaller
models, and allow us to significantly subsample the input representation with
minimal effects to a trained model performance.

    

### [[2105.10719] Learning Baseline Values for Shapley Values](http://arxiv.org/abs/2105.10719)


  This paper aims to formulate the problem of estimating the optimal baseline
values for the Shapley value in game theory. The Shapley value measures the
attribution of each input variable of a complex model, which is computed as the
marginal benefit from the presence of this variable w.r.t.its absence under
different contexts. To this end, people usually set the input variable to its
baseline value to represent the absence of this variable (i.e.the no-signal
state of this variable). Previous studies usually determine the baseline values
in an empirical manner, which hurts the trustworthiness of the Shapley value.
In this paper, we revisit the feature representation of a deep model from the
perspective of game theory, and define the multi-variate interaction patterns
of input variables to define the no-signal state of an input variable. Based on
the multi-variate interaction, we learn the optimal baseline value of each
input variable. Experimental results have demonstrated the effectiveness of our
method.

    

### [[2105.11283] Coarse-to-Fine for Sim-to-Real: Sub-Millimetre Precision Across Wide Task Spaces](http://arxiv.org/abs/2105.11283)


  In this paper, we study the problem of zero-shot sim-to-real when the task
requires both highly precise control with sub-millimetre error tolerance, and
wide task space generalisation. Our framework involves a coarse-to-fine
controller, where trajectories begin with classical motion planning using
ICP-based pose estimation, and transition to a learned end-to-end controller
which maps images to actions and is trained in simulation with domain
randomisation. In this way, we achieve precise control whilst also generalising
the controller across wide task spaces, and keeping the robustness of
vision-based, end-to-end control. Real-world experiments on a range of
different tasks show that, by exploiting the best of both worlds, our framework
significantly outperforms purely motion planning methods, and purely
learning-based methods. Furthermore, we answer a range of questions on best
practices for precise sim-to-real transfer, such as how different image sensor
modalities and image feature representations perform.

    

### [[2106.04148] RECOWNs: Probabilistic Circuits for Trustworthy Time Series Forecasting](http://arxiv.org/abs/2106.04148)


  Time series forecasting is a relevant task that is performed in several
real-world scenarios such as product sales analysis and prediction of energy
demand. Given their accuracy performance, currently, Recurrent Neural Networks
(RNNs) are the models of choice for this task. Despite their success in time
series forecasting, less attention has been paid to make the RNNs trustworthy.
For example, RNNs can not naturally provide an uncertainty measure to their
predictions. This could be extremely useful in practice in several cases e.g.
to detect when a prediction might be completely wrong due to an unusual pattern
in the time series. Whittle Sum-Product Networks (WSPNs), prominent deep
tractable probabilistic circuits (PCs) for time series, can assist an RNN with
providing meaningful probabilities as uncertainty measure. With this aim, we
propose RECOWN, a novel architecture that employs RNNs and a discriminant
variant of WSPNs called Conditional WSPNs (CWSPNs). We also formulate a
Log-Likelihood Ratio Score as better estimation of uncertainty that is tailored
to time series and Whittle likelihoods. In our experiments, we show that
RECOWNs are accurate and trustworthy time series predictors, able to "know when
they do not know".

    

### [[2106.07832] Learning Equivariant Energy Based Models with Equivariant Stein Variational Gradient Descent](http://arxiv.org/abs/2106.07832)


  We focus on the problem of efficient sampling and learning of probability
densities by incorporating symmetries in probabilistic models. We first
introduce Equivariant Stein Variational Gradient Descent algorithm -- an
equivariant sampling method based on Stein's identity for sampling from
densities with symmetries. Equivariant SVGD explicitly incorporates symmetry
information in a density through equivariant kernels which makes the resultant
sampler efficient both in terms of sample complexity and the quality of
generated samples. Subsequently, we define equivariant energy based models to
model invariant densities that are learned using contrastive divergence. By
utilizing our equivariant SVGD for training equivariant EBMs, we propose new
ways of improving and scaling up training of energy based models. We apply
these equivariant energy models for modelling joint densities in regression and
classification tasks for image datasets, many-body particle systems and
molecular structure generation.

    

### [[2107.00821] An Experience Report on Machine Learning Reproducibility: Guidance for Practitioners and TensorFlow Model Garden Contributors](http://arxiv.org/abs/2107.00821)


  Machine learning techniques are becoming a fundamental tool for scientific
and engineering progress. These techniques are applied in contexts as diverse
as astronomy and spam filtering. However, correctly applying these techniques
requires careful engineering. Much attention has been paid to the technical
potential; relatively little attention has been paid to the software
engineering process required to bring research-based machine learning
techniques into practical utility. Technology companies have supported the
engineering community through machine learning frameworks such as TensorFLow
and PyTorch, but the details of how to engineer complex machine learning models
in these frameworks have remained hidden.
To promote best practices within the engineering community, academic
institutions and Google have partnered to launch a Special Interest Group on
Machine Learning Models (SIGMODELS) whose goal is to develop exemplary
implementations of prominent machine learning models in community locations
such as the TensorFlow Model Garden (TFMG). The purpose of this report is to
define a process for reproducing a state-of-the-art machine learning model at a
level of quality suitable for inclusion in the TFMG. We define the engineering
process and elaborate on each step, from paper analysis to model release. We
report on our experiences implementing the YOLO model family with a team of 26
student researchers, share the tools we developed, and describe the lessons we
learned along the way.

    

### [[2107.13649] Reuse Cache for Heterogeneous CPU-GPU Systems](http://arxiv.org/abs/2107.13649)


  It is generally observed that the fraction of live lines in shared last-level
caches (SLLC) is very small for chip multiprocessors (CMPs). This can be
tackled using promotion-based replacement policies like re-reference interval
prediction (RRIP) instead of LRU, dead-block predictors, or reuse-based cache
allocation schemes. In GPU systems, similar LLC issues are alleviated using
various cache bypassing techniques. These issues are worsened in heterogeneous
CPU-GPU systems because the two processors have different data access patterns
and frequencies. GPUs generally work on streaming data, but have many more
threads accessing memory as compared to CPUs. As such, most traditional cache
replacement and allocation policies prove ineffective due to the higher number
of cache accesses in GPU applications, resulting in higher allocation for GPU
cache lines, despite their minimal reuse. In this work, we implement the Reuse
Cache approach for heterogeneous CPU-GPU systems. The reuse cache is a
decoupled tag/data SLLC which is designed to only store the data that is being
accessed more than once. This design is based on the observation that most of
the cache lines in the LLC are stored but do not get reused before being
replaced. We find that the reuse cache achieves within 0.5% of the IPC gains of
a statically partitioned LLC, while decreasing the area cost of the LLC by an
average of 40%.

    

### [[2107.13814] DCG: Distributed Conjugate Gradient for Efficient Linear Equations Solving](http://arxiv.org/abs/2107.13814)


  Distributed algorithms to solve linear equations in multi-agent networks have
attracted great research attention and many iteration-based distributed
algorithms have been developed. The convergence speed is a key factor to be
considered for distributed algorithms, and it is shown dependent on the
spectral radius of the iteration matrix. However, the iteration matrix is
determined by the network structure and is hardly pre-tuned, making the
iterative-based distributed algorithms may converge very slowly when the
spectral radius is close to 1. In contrast, in centralized optimization, the
Conjugate Gradient (CG) is a widely adopted idea to speed up the convergence of
the centralized solvers, which can guarantee convergence in fixed steps. In
this paper, we propose a general distributed implementation of CG, called DCG.
DCG only needs local communication and local computation, while inheriting the
characteristic of fast convergence. DCG guarantees to converge in $4Hn$ rounds,
where $H$ is the maximum hop number of the network and $n$ is the number of
nodes. We present the applications of DCG in solving the least square problem
and network localization problem. The results show the convergence speed of DCG
is three orders of magnitude faster than the widely used Richardson iteration
method.

    

### [[2107.13843] VBR: Version Based Reclamation](http://arxiv.org/abs/2107.13843)


  Safe lock-free memory reclamation is a difficult problem. Existing solutions
follow three basic methods (or their combinations): epoch based reclamation,
hazard pointers, and optimistic reclamation. Epoch-based methods are fast, but
do not guarantee lock-freedom. Hazard pointer solutions are lock-free but
typically do not provide high performance. Optimistic methods are lock-free and
fast, but previous optimistic methods did not go all the way. While reads were
executed optimistically, writes were protected by hazard pointers. In this work
we present a new reclamation scheme called version based reclamation (VBR),
which provides a full optimistic solution to lock-free memory reclamation,
obtaining lock-freedom and high efficiency. Speculative execution is known as a
fundamental tool for improving performance in various areas of computer
science, and indeed evaluation with a lock-free linked-list, hash-table and
skip-list shows that VBR outperforms state-of-the-art existing solutions.

    

### [[2107.14101] Why blockchain and smart contracts need semantic descriptions](http://arxiv.org/abs/2107.14101)


  We argue that there is a hierarchy of levels describing to that particular
level relevant features of reality behind the content and behavior of
blockchain and smart contracts in their realistic deployment.
Choice, design, audit and legal control of these systems could be more
informed, easier and raised to a higher level, if research on foundations of
these descriptions develops and sets the formalisms, tools and standards for
such descriptions.

    

### [[2105.06737] Linearizability: A Typo](http://arxiv.org/abs/2105.06737)


  Linearizability is the de facto consistency condition for concurrent objects,
widely used in theory and practice. Loosely speaking, linearizability
classifies concurrent executions as correct if operations on shared objects
appear to take effect instantaneously during the operation execution time. This
paper calls attention to a somewhat-neglected aspect of linearizability:
restrictions on how pending invocations are handled, an issue that has become
increasingly important for software running on systems with non-volatile main
memory. Interestingly, the original published definition of linearizability
includes a typo (a symbol is missing a prime) that concerns exactly this issue.
In this paper we point out the typo and provide an amendment to make the
definition complete. We believe that pointing this typo out rigorously and
proposing a fix is important and timely.

    

### [[2106.00344] UniStore: A fault-tolerant marriage of causal and strong consistency (extended version)](http://arxiv.org/abs/2106.00344)


  Modern online services rely on data stores that replicate their data across
geographically distributed data centers. Providing strong consistency in such
data stores results in high latencies and makes the system vulnerable to
network partitions. The alternative of relaxing consistency violates crucial
correctness properties. A compromise is to allow multiple consistency levels to
coexist in the data store. In this paper we present UniStore, the first
fault-tolerant and scalable data store that combines causal and strong
consistency. The key challenge we address in UniStore is to maintain liveness
despite data center failures: this could be compromised if a strong transaction
takes a dependency on a causal transaction that is later lost because of a
failure. UniStore ensures that such situations do not arise while paying the
cost of durability for causal transactions only when necessary. We evaluate
UniStore on Amazon EC2 using both microbenchmarks and a sample application. Our
results show that UniStore effectively and scalably combines causal and strong
consistency.

    

### [[2106.08114] Leopard: Scaling BFT without Sacrificing Efficiency](http://arxiv.org/abs/2106.08114)


  With the emergence of large-scale decentralized applications, a scalable and
efficient Byzantine Fault Tolerant (BFT) protocol of hundreds of replicas is
desirable. Although the throughput of existing leader-based BFT protocols has
reached a high level of $10^5$ requests per second for a small scale of
replicas, it drops significantly when the number of replicas increases, which
leads to a lack of practicality. This paper focuses on the scalability of BFT
protocols and identifies a major bottleneck to leader-based BFT protocols due
to the excessive workload of the leader at large scales. A new metric of
scaling factor is defined to capture whether a BFT protocol will get stuck when
it scales out, which can be used to measure the performance of efficiency and
scalability of BFT protocols. We propose "Leopard", the first leader-based BFT
protocol that scales to multiple hundreds of replicas, and more importantly,
preserves a high efficiency. We remove the bottleneck by introducing a
technique of achieving a constant scaling factor, which takes full advantage of
the idle resource and adaptively balances the workload of the leader among all
replicas. We implement Leopard and evaluate its performance compared to
HotStuff, the state-of-the-art BFT protocol. We run extensive experiments on
the two systems with up to 600 replicas. The results show that Leopard achieves
significant performance improvements both on throughput and scalability. In
particular, the throughput of Leopard remains at a high level of $10^5$ when
the system scales out to 600 replicas. It achieves a $5\times$ throughput over
HotStuff when the scale is 300 (which is already the largest scale we can see
the progress of the latter in our experiments), and the gap becomes wider as
the number of replicas further increases.

    

### [[2106.13524] Cost-efficient, QoS and Security aware Placement of Smart Farming IoT Applications in Cloud-Fog Infrastructure](http://arxiv.org/abs/2106.13524)


  Smart farming is a recent innovation in the agriculture sector that can
improve the agricultural yield by using smarter, automated, and data driven
farm processes that interact with IoT devices deployed on farms. A cloud-fog
infrastructure provides an effective platform to execute IoT applications.
While fog computing satisfies the real-time processing need of delay-sensitive
IoT services by bringing virtualized services closer to the IoT devices, cloud
computing allows execution of applications with higher computational
requirements. The deployment of IoT applications is a critical challenge as
cloud and fog nodes vary in terms of their resource availability and use
different cost models. Moreover, diversity in resource, quality of service
(QoS) and security requirements of IoT applications make the problem even more
complex. In this paper, we model IoT application placement as an optimization
problem that aims at minimizing the cost while satisfying the QoS and security
constraints. The problem is formulated using Integer Linear Programming (ILP).
The ILP model is evaluated for a small-scale scenario. The evaluation shows the
impact of QoS and security requirement on the cost. We also study the impact of
relaxing security constraint on the placement decision.

    

### [[2107.13587] Fast and Scalable Image Search For Histology](http://arxiv.org/abs/2107.13587)


  The expanding adoption of digital pathology has enabled the curation of large
repositories of histology whole slide images (WSIs), which contain a wealth of
information. Similar pathology image search offers the opportunity to comb
through large historical repositories of gigapixel WSIs to identify cases with
similar morphological features and can be particularly useful for diagnosing
rare diseases, identifying similar cases for predicting prognosis, treatment
outcomes, and potential clinical trial success. A critical challenge in
developing a WSI search and retrieval system is scalability, which is uniquely
challenging given the need to search a growing number of slides that each can
consist of billions of pixels and are several gigabytes in size. Such systems
are typically slow and retrieval speed often scales with the size of the
repository they search through, making their clinical adoption tedious and are
not feasible for repositories that are constantly growing. Here we present Fast
Image Search for Histopathology (FISH), a histology image search pipeline that
is infinitely scalable and achieves constant search speed that is independent
of the image database size while being interpretable and without requiring
detailed annotations. FISH uses self-supervised deep learning to encode
meaningful representations from WSIs and a Van Emde Boas tree for fast search,
followed by an uncertainty-based ranking algorithm to retrieve similar WSIs. We
evaluated FISH on multiple tasks and datasets with over 22,000 patient cases
spanning 56 disease subtypes. We additionally demonstrate that FISH can be used
to assist with the diagnosis of rare cancer types where sufficient cases may
not be available to train traditional supervised deep models. FISH is available
as an easy-to-use, open-source software package
(this https URL).

    

### [[2107.13641] Learned upper bounds for the Time-Dependent Travelling Salesman Problem](http://arxiv.org/abs/2107.13641)


  Given a graph whose arc traversal times vary over time, the Time-Dependent
Travelling Salesman Problem consists in finding a Hamiltonian tour of least
total duration covering the vertices of the graph. The main goal of this work
is to define tight upper bounds for this problem by reusing the information
gained when solving instances with similar features. This is customary in
distribution management, where vehicle routes have to be generated over and
over again with similar input data. To this aim, we devise an upper bounding
technique based on the solution of a classical (and simpler) time-independent
Asymmetric Travelling Salesman Problem, where the constant arc costs are
suitably defined by the combined use of a Linear Program and a mix of
unsupervised and supervised Machine Learning techniques. The effectiveness of
this approach has been assessed through a computational campaign on the real
travel time functions of two European cities: Paris and London. The overall
average gap between our heuristic and the best-known solutions is about
0.001\%. For 31 instances, new best solutions have been obtained.

    

### [[2107.13668] Learning User-Interpretable Descriptions of Black-Box AI System Capabilities](http://arxiv.org/abs/2107.13668)


  Several approaches have been developed to answer specific questions that a
user may have about an AI system that can plan and act. However, the problems
of identifying which questions to ask and that of computing a
user-interpretable symbolic description of the overall capabilities of the
system have remained largely unaddressed. This paper presents an approach for
addressing these problems by learning user-interpretable symbolic descriptions
of the limits and capabilities of a black-box AI system using low-level
simulators. It uses a hierarchical active querying paradigm to generate
questions and to learn a user-interpretable model of the AI system based on its
responses. In contrast to prior work, we consider settings where imprecision of
the user's conceptual vocabulary precludes a direct expression of the agent's
capabilities. Furthermore, our approach does not require assumptions about the
internal design of the target AI system or about the methods that it may use to
compute or learn task solutions. Empirical evaluation on several game-based
simulator domains shows that this approach can efficiently learn symbolic
models of AI systems that use a deterministic black-box policy in fully
observable scenarios.

    

### [[2107.13669] Bi-Bimodal Modality Fusion for Correlation-Controlled Multimodal Sentiment Analysis](http://arxiv.org/abs/2107.13669)


  Multimodal sentiment analysis aims to extract and integrate semantic
information collected from multiple modalities to recognize the expressed
emotions and sentiment in multimodal data. This research area's major concern
lies in developing an extraordinary fusion scheme that can extract and
integrate key information from various modalities. However, one issue that may
restrict previous work to achieve a higher level is the lack of proper modeling
for the dynamics of the competition between the independence and relevance
among modalities, which could deteriorate fusion outcomes by causing the
collapse of modality-specific feature space or introducing extra noise. To
mitigate this, we propose the Bi-Bimodal Fusion Network (BBFN), a novel
end-to-end network that performs fusion (relevance increment) and separation
(difference increment) on pairwise modality representations. The two parts are
trained simultaneously such that the combat between them is simulated. The
model takes two bimodal pairs as input due to the known information imbalance
among modalities. In addition, we leverage a gated control mechanism in the
Transformer architecture to further improve the final output. Experimental
results on three datasets (CMU-MOSI, CMU-MOSEI, and UR-FUNNY) verifies that our
model significantly outperforms the SOTA. The implementation of this work is
available at this https URL.

    

### [[2107.13684] An Online Question Answering System based on Sub-graph Searching](http://arxiv.org/abs/2107.13684)


  Knowledge graphs (KGs) have been widely used for question answering (QA)
applications, especially the entity based QA. However, searching an-swers from
an entire large-scale knowledge graph is very time-consuming and it is hard to
meet the speed need of real online QA systems. In this pa-per, we design a
sub-graph searching mechanism to solve this problem by creating sub-graph
index, and each answer generation step is restricted in the sub-graph level. We
use this mechanism into a real online QA chat system, and it can bring obvious
improvement on question coverage by well answer-ing entity based questions, and
it can be with a very high speed, which en-sures the user experience of online
QA.

    

### [[2107.13704] A Theory of Consciousness from a Theoretical Computer Science Perspective 2: Insights from the Conscious Turing Machine](http://arxiv.org/abs/2107.13704)


  The quest to understand consciousness, once the purview of philosophers and
theologians, is now actively pursued by scientists of many stripes. We examine
consciousness from the perspective of theoretical computer science (TCS), a
branch of mathematics concerned with understanding the underlying principles of
computation and complexity, including the implications and surprising
consequences of resource limitations. In the spirit of Alan Turing's simple yet
powerful definition of a computer, the Turing Machine (TM), and perspective of
computational complexity theory, we formalize a modified version of the Global
Workspace Theory (GWT) of consciousness originated by cognitive neuroscientist
Bernard Baars and further developed by him, Stanislas Dehaene, Jean-Pierre
Changeaux and others. We are not looking for a complex model of the brain nor
of cognition, but for a simple computational model of (the admittedly complex
concept of) consciousness. We do this by defining the Conscious Turing Machine
(CTM), also called a conscious AI, and then we define consciousness and related
notions in the CTM. While these are only mathematical (TCS) definitions, we
suggest why the CTM has the feeling of consciousness. The TCS perspective
provides a simple formal framework to employ tools from computational
complexity theory and machine learning to help us understand consciousness and
related concepts. Previously we explored high level explanations for the
feelings of pain and pleasure in the CTM. Here we consider three examples
related to vision (blindsight, inattentional blindness, and change blindness),
followed by discussions of dreams, free will, and altered states of
consciousness.

    

### [[2107.13731] UIBert: Learning Generic Multimodal Representations for UI Understanding](http://arxiv.org/abs/2107.13731)


  To improve the accessibility of smart devices and to simplify their usage,
building models which understand user interfaces (UIs) and assist users to
complete their tasks is critical. However, unique challenges are proposed by
UI-specific characteristics, such as how to effectively leverage multimodal UI
features that involve image, text, and structural metadata and how to achieve
good performance when high-quality labeled data is unavailable. To address such
challenges we introduce UIBert, a transformer-based joint image-text model
trained through novel pre-training tasks on large-scale unlabeled UI data to
learn generic feature representations for a UI and its components. Our key
intuition is that the heterogeneous features in a UI are self-aligned, i.e.,
the image and text features of UI components, are predictive of each other. We
propose five pretraining tasks utilizing this self-alignment among different
features of a UI component and across various components in the same UI. We
evaluate our method on nine real-world downstream UI tasks where UIBert
outperforms strong multimodal baselines by up to 9.26% accuracy.

    

### [[2107.13734] An Ethical Framework for Guiding the Development of Affectively-Aware Artificial Intelligence](http://arxiv.org/abs/2107.13734)


  The recent rapid advancements in artificial intelligence research and
deployment have sparked more discussion about the potential ramifications of
socially- and emotionally-intelligent AI. The question is not if research can
produce such affectively-aware AI, but when it will. What will it mean for
society when machines -- and the corporations and governments they serve -- can
"read" people's minds and emotions? What should developers and operators of
such AI do, and what should they not do? The goal of this article is to
pre-empt some of the potential implications of these developments, and propose
a set of guidelines for evaluating the (moral and) ethical consequences of
affectively-aware AI, in order to guide researchers, industry professionals,
and policy-makers. We propose a multi-stakeholder analysis framework that
separates the ethical responsibilities of AI Developers vis-à-vis the
entities that deploy such AI -- which we term Operators. Our analysis produces
two pillars that clarify the responsibilities of each of these stakeholders:
Provable Beneficence, which rests on proving the effectiveness of the AI, and
Responsible Stewardship, which governs responsible collection, use, and storage
of data and the decisions made from such data. We end with recommendations for
researchers, developers, operators, as well as regulators and law-makers.

    

### [[2107.13738] Design-Driven Requirements for Computationally Co-Creative Game AI Design Tools](http://arxiv.org/abs/2107.13738)


  Game AI designers must manage complex interactions between the AI character,
the game world, and the player, while achieving their design visions.
Computational co-creativity tools can aid them, but first, AI and HCI
researchers must gather requirements and determine design heuristics to build
effective co-creative tools. In this work, we present a participatory design
study that categorizes and analyzes game AI designers' workflows, goals, and
expectations for such tools. We evince deep connections between game AI design
and the design of co-creative tools, and present implications for future
co-creativity tool research and development.

    

### [[2107.13742] Profile to Frontal Face Recognition in the Wild Using Coupled Conditional GAN](http://arxiv.org/abs/2107.13742)


  In recent years, with the advent of deep-learning, face recognition has
achieved exceptional success. However, many of these deep face recognition
models perform much better in handling frontal faces compared to profile faces.
The major reason for poor performance in handling of profile faces is that it
is inherently difficult to learn pose-invariant deep representations that are
useful for profile face recognition. In this paper, we hypothesize that the
profile face domain possesses a latent connection with the frontal face domain
in a latent feature subspace. We look to exploit this latent connection by
projecting the profile faces and frontal faces into a common latent subspace
and perform verification or retrieval in the latent domain. We leverage a
coupled conditional generative adversarial network (cpGAN) structure to find
the hidden relationship between the profile and frontal images in a latent
common embedding subspace. Specifically, the cpGAN framework consists of two
conditional GAN-based sub-networks, one dedicated to the frontal domain and the
other dedicated to the profile domain. Each sub-network tends to find a
projection that maximizes the pair-wise correlation between the two feature
domains in a common embedding feature subspace. The efficacy of our approach
compared with the state-of-the-art is demonstrated using the CFP, CMU
Multi-PIE, IJB-A, and IJB-C datasets. Additionally, we have also implemented a
coupled convolutional neural network (cpCNN) and an adversarial discriminative
domain adaptation network (ADDA) for profile to frontal face recognition. We
have evaluated the performance of cpCNN and ADDA and compared it with the
proposed cpGAN. Finally, we have also evaluated our cpGAN for reconstruction of
frontal faces from input profile faces contained in the VGGFace2 dataset.

    

### [[2107.13807] FREE: Feature Refinement for Generalized Zero-Shot Learning](http://arxiv.org/abs/2107.13807)


  Generalized zero-shot learning (GZSL) has achieved significant progress, with
many efforts dedicated to overcoming the problems of visual-semantic domain gap
and seen-unseen bias. However, most existing methods directly use feature
extraction models trained on ImageNet alone, ignoring the cross-dataset bias
between ImageNet and GZSL benchmarks. Such a bias inevitably results in
poor-quality visual features for GZSL tasks, which potentially limits the
recognition performance on both seen and unseen classes. In this paper, we
propose a simple yet effective GZSL method, termed feature refinement for
generalized zero-shot learning (FREE), to tackle the above problem. FREE
employs a feature refinement (FR) module that incorporates
\textit{semantic$\rightarrow$visual} mapping into a unified generative model to
refine the visual features of seen and unseen class samples. Furthermore, we
propose a self-adaptive margin center loss (SAMC-loss) that cooperates with a
semantic cycle-consistency loss to guide FR to learn class- and
semantically-relevant representations, and concatenate the features in FR to
extract the fully refined features. Extensive experiments on five benchmark
datasets demonstrate the significant performance gain of FREE over its baseline
and current state-of-the-art methods. Our codes are available at
this https URL .

    

### [[2107.13904] Cross-Camera Feature Prediction for Intra-Camera Supervised Person Re-identification across Distant Scenes](http://arxiv.org/abs/2107.13904)


  Person re-identification (Re-ID) aims to match person images across
non-overlapping camera views. The majority of Re-ID methods focus on
small-scale surveillance systems in which each pedestrian is captured in
different camera views of adjacent scenes. However, in large-scale surveillance
systems that cover larger areas, it is required to track a pedestrian of
interest across distant scenes (e.g., a criminal suspect escapes from one city
to another). Since most pedestrians appear in limited local areas, it is
difficult to collect training data with cross-camera pairs of the same person.
In this work, we study intra-camera supervised person re-identification across
distant scenes (ICS-DS Re-ID), which uses cross-camera unpaired data with
intra-camera identity labels for training. It is challenging as cross-camera
paired data plays a crucial role for learning camera-invariant features in most
existing Re-ID methods. To learn camera-invariant representation from
cross-camera unpaired training data, we propose a cross-camera feature
prediction method to mine cross-camera self supervision information from
camera-specific feature distribution by transforming fake cross-camera positive
feature pairs and minimize the distances of the fake pairs. Furthermore, we
automatically localize and extract local-level feature by a transformer. Joint
learning of global-level and local-level features forms a global-local
cross-camera feature prediction scheme for mining fine-grained cross-camera
self supervision information. Finally, cross-camera self supervision and
intra-camera supervision are aggregated in a framework. The experiments are
conducted in the ICS-DS setting on Market-SCT, Duke-SCT and MSMT17-SCT
datasets. The evaluation results demonstrate the superiority of our method,
which gains significant improvements of 15.4 Rank-1 and 22.3 mAP on Market-SCT
as compared to the second best method.

    

### [[2107.13955] Demystifying Neural Language Models' Insensitivity to Word-Order](http://arxiv.org/abs/2107.13955)


  Recent research analyzing the sensitivity of natural language understanding
models to word-order perturbations have shown that the state-of-the-art models
in several language tasks may have a unique way to understand the text that
could seldom be explained with conventional syntax and semantics. In this
paper, we investigate the insensitivity of natural language models to
word-order by quantifying perturbations and analysing their effect on neural
models' performance on language understanding tasks in GLUE benchmark. Towards
that end, we propose two metrics - the Direct Neighbour Displacement (DND) and
the Index Displacement Count (IDC) - that score the local and global ordering
of tokens in the perturbed texts and observe that perturbation functions found
in prior literature affect only the global ordering while the local ordering
remains relatively unperturbed. We propose perturbations at the granularity of
sub-words and characters to study the correlation between DND, IDC and the
performance of neural language models on natural language tasks. We find that
neural language models - pretrained and non-pretrained Transformers, LSTMs, and
Convolutional architectures - require local ordering more so than the global
ordering of tokens. The proposed metrics and the suite of perturbations allow a
systematic way to study the (in)sensitivity of neural language understanding
models to varying degree of perturbations.

    

### [[2107.13977] Underwater Acoustic Networks for Security Risk Assessment in Public Drinking Water Reservoirs](http://arxiv.org/abs/2107.13977)


  We have built a novel system for the surveillance of drinking water
reservoirs using underwater sensor networks. We implement an innovative
AI-based approach to detect, classify and localize underwater events. In this
paper, we describe the technology and cognitive AI architecture of the system
based on one of the sensor networks, the hydrophone network. We discuss the
challenges of installing and using the hydrophone network in a water reservoir
where traffic, visitors, and variable water conditions create a complex,
varying environment. Our AI solution uses an autoencoder for unsupervised
learning of latent encodings for classification and anomaly detection, and time
delay estimates for sound localization. Finally, we present the results of
experiments carried out in a laboratory pool and the water reservoir and
discuss the system's potential.

    

### [[2107.13994] Improving Robustness and Accuracy via Relative Information Encoding in 3D Human Pose Estimation](http://arxiv.org/abs/2107.13994)


  Most of the existing 3D human pose estimation approaches mainly focus on
predicting 3D positional relationships between the root joint and other human
joints (local motion) instead of the overall trajectory of the human body
(global motion). Despite the great progress achieved by these approaches, they
are not robust to global motion, and lack the ability to accurately predict
local motion with a small movement range. To alleviate these two problems, we
propose a relative information encoding method that yields positional and
temporal enhanced representations. Firstly, we encode positional information by
utilizing relative coordinates of 2D poses to enhance the consistency between
the input and output distribution. The same posture with different absolute 2D
positions can be mapped to a common representation. It is beneficial to resist
the interference of global motion on the prediction results. Second, we encode
temporal information by establishing the connection between the current pose
and other poses of the same person within a period of time. More attention will
be paid to the movement changes before and after the current pose, resulting in
better prediction performance on local motion with a small movement range. The
ablation studies validate the effectiveness of the proposed relative
information encoding method. Besides, we introduce a multi-stage optimization
method to the whole framework to further exploit the positional and temporal
enhanced representations. Our method outperforms state-of-the-art methods on
two public datasets. Code is available at
this https URL.

    

### [[2107.14000] Resisting Out-of-Distribution Data Problem in Perturbation of XAI](http://arxiv.org/abs/2107.14000)


  With the rapid development of eXplainable Artificial Intelligence (XAI),
perturbation-based XAI algorithms have become quite popular due to their
effectiveness and ease of implementation. The vast majority of
perturbation-based XAI techniques face the challenge of Out-of-Distribution
(OoD) data -- an artifact of randomly perturbed data becoming inconsistent with
the original dataset. OoD data leads to the over-confidence problem in model
predictions, making the existing XAI approaches unreliable. To our best
knowledge, the OoD data problem in perturbation-based XAI algorithms has not
been adequately addressed in the literature. In this work, we address this OoD
data problem by designing an additional module quantifying the affinity between
the perturbed data and the original dataset distribution, which is integrated
into the process of aggregation. Our solution is shown to be compatible with
the most popular perturbation-based XAI algorithms, such as RISE, OCCLUSION,
and LIME. Experiments have confirmed that our methods demonstrate a significant
improvement in general cases using both computational and cognitive metrics.
Especially in the case of degradation, our proposed approach demonstrates
outstanding performance comparing to baselines. Besides, our solution also
resolves a fundamental problem with the faithfulness indicator, a commonly used
evaluation metric of XAI algorithms that appears to be sensitive to the OoD
issue.

    

### [[2107.14042] The brain is a computer is a brain: neuroscience's internal debate and the social significance of the Computational Metaphor](http://arxiv.org/abs/2107.14042)


  The Computational Metaphor, comparing the brain to the computer and vice
versa, is the most prominent metaphor in neuroscience and artificial
intelligence (AI). Its appropriateness is highly debated in both fields,
particularly with regards to whether it is useful for the advancement of
science and technology. Considerably less attention, however, has been devoted
to how the Computational Metaphor is used outside of the lab, and particularly
how it may shape society's interactions with AI. As such, recently publicized
concerns over AI's role in perpetuating racism, genderism, and ableism suggest
that the term "artificial intelligence" is misplaced, and that a new lexicon is
needed to describe these computational systems. Thus, there is an essential
question about the Computational Metaphor that is rarely asked by
neuroscientists: whom does it help and whom does it harm? This essay invites
the neuroscience community to consider the social implications of the field's
most controversial metaphor.

    

### [[2107.14044] Ethical AI for Social Good](http://arxiv.org/abs/2107.14044)


  The concept of AI for Social Good(AI4SG) is gaining momentum in both
information societies and the AI community. Through all the advancement of
AI-based solutions, it can solve societal issues effectively. To date, however,
there is only a rudimentary grasp of what constitutes AI socially beneficial in
principle, what constitutes AI4SG in reality, and what are the policies and
regulations needed to ensure it. This paper fills the vacuum by addressing the
ethical aspects that are critical for future AI4SG efforts. Some of these
characteristics are new to AI, while others have greater importance due to its
usage.

    

### [[2107.14052] The Role of Social Movements, Coalitions, and Workers in Resisting Harmful Artificial Intelligence and Contributing to the Development of Responsible AI](http://arxiv.org/abs/2107.14052)


  There is mounting public concern over the influence that AI based systems has
in our society. Coalitions in all sectors are acting worldwide to resist hamful
applications of AI. From indigenous people addressing the lack of reliable
data, to smart city stakeholders, to students protesting the academic
relationships with sex trafficker and MIT donor Jeffery Epstein, the
questionable ethics and values of those heavily investing in and profiting from
AI are under global scrutiny. There are biased, wrongful, and disturbing
assumptions embedded in AI algorithms that could get locked in without
intervention. Our best human judgment is needed to contain AI's harmful impact.
Perhaps one of the greatest contributions of AI will be to make us ultimately
understand how important human wisdom truly is in life on earth.

    

### [[2107.14093] A Decision Model for Decentralized Autonomous Organization Platform Selection: Three Industry Case Studies](http://arxiv.org/abs/2107.14093)


  Decentralized autonomous organizations as a new form of online governance
arecollections of smart contracts deployed on a blockchain platform that
intercede groupsof people. A growing number of Decentralized Autonomous
Organization Platforms,such as Aragon and Colony, have been introduced in the
market to facilitate thedevelopment process of such organizations. Selecting
the best fitting platform ischallenging for the organizations, as a significant
number of decision criteria, such aspopularity, developer availability,
governance issues, and consistent documentation ofsuch platforms, should be
considered. Additionally, decision-makers at theorganizations are not experts
in every domain, so they must continuously acquirevolatile knowledge regarding
such platforms and keep themselves updated.Accordingly, a decision model is
required to analyze the decision criteria usingsystematic identification and
evaluation of potential alternative solutions for adevelopment project. We have
developed a theoretical framework to assist softwareengineers with a set of
Multi-Criteria Decision-Making problems in software production.This study
presents a decision model as a Multi-Criteria Decision-Making problem forthe
decentralized autonomous organization platform selection problem. Weconducted
three industry case studies in the context of three decentralizedautonomous
organizations to evaluate the effectiveness and efficiency of the decisionmodel
in assisting decision-makers.

    

### [[2107.14199] RSO: A Novel Reinforced Swarm Optimization Algorithm for Feature Selection](http://arxiv.org/abs/2107.14199)


  Swarm optimization algorithms are widely used for feature selection before
data mining and machine learning applications. The metaheuristic
nature-inspired feature selection approaches are used for single-objective
optimization tasks, though the major problem is their frequent premature
convergence, leading to weak contribution to data mining. In this paper, we
propose a novel feature selection algorithm named Reinforced Swarm Optimization
(RSO) leveraging some of the existing problems in feature selection. This
algorithm embeds the widely used Bee Swarm Optimization (BSO) algorithm along
with Reinforcement Learning (RL) to maximize the reward of a superior search
agent and punish the inferior ones. This hybrid optimization algorithm is more
adaptive and robust with a good balance between exploitation and exploration of
the search space. The proposed method is evaluated on 25 widely known UCI
datasets containing a perfect blend of balanced and imbalanced data. The
obtained results are compared with several other popular and recent feature
selection algorithms with similar classifier configurations. The experimental
outcome shows that our proposed model outperforms BSO in 22 out of 25 instances
(88%). Moreover, experimental results also show that RSO performs the best
among all the methods compared in this paper in 19 out of 25 cases (76%),
establishing the superiority of our proposed method.

    

### [[1403.1076] Is Intelligence Artificial?](http://arxiv.org/abs/1403.1076)


  Our understanding of intelligence is directed primarily at the human level.
This paper attempts to give a more unifying definition that can be applied to
the natural world in general and then Artificial Intelligence. The definition
would be used more to verify a relative intelligence, not to quantify it and
might help when making judgements on the matter. While correct behaviour is the
preferred definition, a metric that is grounded in Kolmogorov's Complexity
Theory is suggested, which leads to a measurement about entropy. A version of
an accepted AI test is then put forward as the 'acid test' and might be what a
free-thinking program would try to achieve. Recent work by the author has been
more from a direction of mechanical processes, or ones that might operate
automatically. This paper agrees that intelligence is a pro-active event, but
also notes a second aspect to it that is in the background and mechanical. The
paper suggests looking at intelligence and the conscious as being slightly
different, where consciousness is this more mechanical aspect. In fact, a
surprising conclusion can be a passive but intelligent brain being invoked by
active and less intelligent senses.

    

### [[2009.11485] CogniFNN: A Fuzzy Neural Network Framework for Cognitive Word Embedding Evaluation](http://arxiv.org/abs/2009.11485)


  Word embeddings can reflect the semantic representations, and the embedding
qualities can be comprehensively evaluated with human natural reading-related
cognitive data sources. In this paper, we proposed the CogniFNN framework,
which is the first attempt at using fuzzy neural networks to extract non-linear
and non-stationary characteristics for evaluations of English word embeddings
against the corresponding cognitive datasets. In our experiment, we used 15
human cognitive datasets across three modalities: EEG, fMRI, and eye-tracking,
and selected the mean square error and multiple hypotheses testing as metrics
to evaluate our proposed CogniFNN framework. Compared to the recent pioneer
framework, our proposed CogniFNN showed smaller prediction errors of both
context-independent (GloVe) and context-sensitive (BERT) word embeddings, and
achieved higher significant ratios with randomly generated word embeddings. Our
findings suggested that the CogniFNN framework could provide a more accurate
and comprehensive evaluation of cognitive word embeddings. It will potentially
be beneficial to the further word embeddings evaluation on extrinsic natural
language processing tasks.

    

### [[2101.00591] Progressive Correspondence Pruning by Consensus Learning](http://arxiv.org/abs/2101.00591)


  Correspondence selection aims to correctly select the consistent matches
(inliers) from an initial set of putative correspondences. The selection is
challenging since putative matches are typically extremely unbalanced, largely
dominated by outliers, and the random distribution of such outliers further
complicates the learning process for learning-based methods. To address this
issue, we propose to progressively prune the correspondences via a
local-to-global consensus learning procedure. We introduce a ``pruning'' block
that lets us identify reliable candidates among the initial matches according
to consensus scores estimated using local-to-global dynamic graphs. We then
achieve progressive pruning by stacking multiple pruning blocks sequentially.
Our method outperforms state-of-the-arts on robust line fitting, camera pose
estimation and retrieval-based image localization benchmarks by significant
margins and shows promising generalization ability to different datasets and
detector/descriptor combinations.

    

### [[2101.04640] Dimensions of Commonsense Knowledge](http://arxiv.org/abs/2101.04640)


  Commonsense knowledge is essential for many AI applications, including those
in natural language processing, visual processing, and planning. Consequently,
many sources that include commonsense knowledge have been designed and
constructed over the past decades. Recently, the focus has been on large
text-based sources, which facilitate easier integration with neural (language)
models and application to textual tasks, typically at the expense of the
semantics of the sources and their harmonization. Efforts to consolidate
commonsense knowledge have yielded partial success, with no clear path towards
a comprehensive solution. We aim to organize these sources around a common set
of dimensions of commonsense knowledge. We survey a wide range of popular
commonsense sources with a special focus on their relations. We consolidate
these relations into 13 knowledge dimensions. This consolidation allows us to
unify the separate sources and to compute indications of their coverage,
overlap, and gaps with respect to the knowledge dimensions. Moreover, we
analyze the impact of each dimension on downstream reasoning tasks that require
commonsense knowledge, observing that the temporal and desire/goal dimensions
are very beneficial for reasoning on current downstream tasks, while
distinctness and lexical knowledge have little impact. These results reveal
preferences for some dimensions in current evaluation, and potential neglect of
others.

    

### [[2104.14222] Privacy-Preserving Portrait Matting](http://arxiv.org/abs/2104.14222)


  Recently, there has been an increasing concern about the privacy issue raised
by using personally identifiable information in machine learning. However,
previous portrait matting methods were all based on identifiable portrait
images. To fill the gap, we present P3M-10k in this paper, which is the first
large-scale anonymized benchmark for Privacy-Preserving Portrait Matting.
P3M-10k consists of 10,000 high-resolution face-blurred portrait images along
with high-quality alpha mattes. We systematically evaluate both trimap-free and
trimap-based matting methods on P3M-10k and find that existing matting methods
show different generalization capabilities when following the
Privacy-Preserving Training (PPT) setting, i.e., training on face-blurred
images and testing on arbitrary images. To devise a better trimap-free portrait
matting model, we propose P3M-Net, which leverages the power of a unified
framework for both semantic perception and detail matting, and specifically
emphasizes the interaction between them and the encoder to facilitate the
matting process. Extensive experiments on P3M-10k demonstrate that P3M-Net
outperforms the state-of-the-art methods in terms of both objective metrics and
subjective visual quality. Besides, it shows good generalization capacity under
the PPT setting, confirming the value of P3M-10k for facilitating future
research and enabling potential real-world applications. The source code and
dataset are available at this https URL


### [[2107.14210] Accurate Throughput Prediction of Basic Blocks on Recent Intel Microarchitectures](http://arxiv.org/abs/2107.14210)


  Tools to predict the throughput of basic blocks on a specific
microarchitecture are useful to optimize software performance and to build
optimizing compilers. In recent work, several such tools have been proposed.
However, the accuracy of their predictions has been shown to be relatively low.
In this paper, we identify the most important factors for these inaccuracies.
To a significant degree these inaccuracies are due to elements and parameters
of the pipelines of recent CPUs that are not taken into account by previous
tools. A primary reason for this is that the necessary details are often
undocumented. In this paper, we build more precise models of relevant
components by reverse engineering using microbenchmarks. Based on these models,
we develop a simulator for predicting the throughput of basic blocks. In
addition to predicting the throughput, our simulator also provides insights
into how the code is executed.
Our tool supports all Intel Core microarchitecture generations released in
the last decade. We evaluate it on an improved version of the BHive benchmark
suite. On many recent microarchitectures, its predictions are more accurate
than the predictions of state-of-the-art tools by more than an order of
magnitude.

    