
## 2021-10-12

### [[2110.04320] Constraint-Aware Deep Reinforcement Learning for End-to-End Resource Orchestration in Mobile Networks](http://arxiv.org/abs/2110.04320)


  Network slicing is a promising technology that allows mobile network
operators to efficiently serve various emerging use cases in 5G. It is
challenging to optimize the utilization of network infrastructures while
guaranteeing the performance of network slices according to service level
agreements (SLAs). To solve this problem, we propose SafeSlicing that
introduces a new constraint-aware deep reinforcement learning (CaDRL) algorithm
to learn the optimal resource orchestration policy within two steps, i.e.,
offline training in a simulated environment and online learning with the real
network system. On optimizing the resource orchestration, we incorporate the
constraints on the statistical performance of slices in the reward function
using Lagrangian multipliers, and solve the Lagrangian relaxed problem via a
policy network. To satisfy the constraints on the system capacity, we design a
constraint network to map the latent actions generated from the policy network
to the orchestration actions such that the total resources allocated to network
slices do not exceed the system capacity. We prototype SafeSlicing on an
end-to-end testbed developed by using OpenAirInterface LTE, OpenDayLight-based
SDN, and CUDA GPU computing platform. The experimental results show that
SafeSlicing reduces more than 20% resource usage while meeting SLAs of network
slices as compared with other solutions.

    

### [[2110.04371] DispersedLedger: High-Throughput Byzantine Consensus on Variable Bandwidth Networks](http://arxiv.org/abs/2110.04371)


  The success of blockchains has sparked interest in large-scale deployments of
Byzantine fault tolerant (BFT) consensus protocols over wide area networks. A
central feature of such networks is variable communication bandwidth across
nodes and across time. We present DispersedLedger, an asynchronous BFT protocol
that provides near-optimal throughput in the presence of such variable network
bandwidth. The core idea of DispersedLedger is to enable nodes to propose,
order, and agree on blocks of transactions without having to download their
full content. By enabling nodes to agree on an ordered log of blocks, with a
guarantee that each block is available within the network and unmalleable,
DispersedLedger decouples bandwidth-intensive block downloads at different
nodes, allowing each to make progress at its own pace. We build a full system
prototype and evaluate it on real-world and emulated networks. Our results on a
geo-distributed wide-area deployment across the Internet shows that
DispersedLedger achieves 2x better throughput and 74% reduction in latency
compared to HoneyBadger, the state-of-the-art asynchronous protocol.

    

### [[2110.04478] Themis: A Network Bandwidth-Aware Collective Scheduling Policy for Distributed Training of DL Models](http://arxiv.org/abs/2110.04478)


  The continuous growth in both size and training data for modern Deep Neural
Networks (DNNs) models has led to training tasks taking days or even months.
Distributed training is a solution to reduce training time by splitting the
task across multiple NPUs (e.g., GPU/TPU). However, distributed training adds
communication overhead between the NPUs in order to synchronize the gradients
and/or activation, depending on the parallelization strategy. In today's
datacenters, for training at scale, NPUs are connected through
multi-dimensional interconnection links with different bandwidth and latency.
Hence, keeping all network dimensions busy and maximizing the network BW is a
challenging task in such a hybrid network environment, as this work identifies.
We propose Themis, a novel collective scheduling scheme that dynamically
schedules collectives (divided into chunks) to balance the communication loads
across all dimensions, further improving the network BW utilization. Our
results show that on average, Themis can improve the network BW utilization of
single All-Reduce by 1.88x (2.92x max), and improve the end-to-end training
iteration performance of real workloads such as ResNet-50, GNMT, DLRM, and
Transformer- 1T by 1.49x (1.96x max), 1.41x (1.81x max), 1.42x (1.80x max), and
1.35x (1.78x max), respectively.

    

### [[2110.04488] Demystifying the Transferability of Adversarial Attacks in Computer Networks](http://arxiv.org/abs/2110.04488)


  Deep Convolutional Neural Networks (CNN) models are one of the most popular
networks in deep learning. With their large fields of application in different
areas, they are extensively used in both academia and industry. CNN-based
models include several exciting implementations such as early breast cancer
detection or detecting developmental delays in children (e.g., autism, speech
disorders, etc.). However, previous studies demonstrate that these models are
subject to various adversarial attacks. Interestingly, some adversarial
examples could potentially still be effective against different unknown models.
This particular property is known as adversarial transferability, and prior
works slightly analyzed this characteristic in a very limited application
domain. In this paper, we aim to demystify the transferability threats in
computer networks by studying the possibility of transferring adversarial
examples. In particular, we provide the first comprehensive study which
assesses the robustness of CNN-based models for computer networks against
adversarial transferability. In our experiments, we consider five different
attacks: (1) the Iterative Fast Gradient Method (I-FGSM), (2) the
Jacobian-based Saliency Map attack (JSMA), (3) the L-BFGS attack, (4) the
Projected Gradient Descent attack (PGD), and (5) the DeepFool attack. These
attacks are performed against two well-known datasets: the N-BaIoT dataset and
the Domain Generating Algorithms (DGA) dataset. Our results show that the
transferability happens in specific use cases where the adversary can easily
compromise the victim's network with very few knowledge of the targeted model.

    

### [[2110.04648] From Fragmentation to Liberation](http://arxiv.org/abs/2110.04648)


  In this paper, I argue that "Internet fragmentation" as a phenomenon is only
meaningful in the context of the US's hegemonic control over the Internet. I
propose a broader and, I argue, more richly predictive frame: Internet
conflict. I show how this frame provides fresh analytical purchase to some of
the questions I list above, using it to contextualize several apparently
distinct phenomena. I conclude by arguing that only one question gives this
analytical frame, or any other, a higher purpose: what particular interventions
to Internet governance can produce meaningfully liberatory outcomes? Any
descriptive framework is only useful insofar as it can be mobilized to answer
this normative question.

    

### [[2006.05459] Privacy For Free: Wireless Federated Learning Via Uncoded Transmission With Adaptive Power Control](http://arxiv.org/abs/2006.05459)


  Federated Learning (FL) refers to distributed protocols that avoid direct raw
data exchange among the participating devices while training for a common
learning task. This way, FL can potentially reduce the information on the local
data sets that is leaked via communications. In order to provide formal privacy
guarantees, however, it is generally necessary to put in place additional
masking mechanisms. When FL is implemented in wireless systems via uncoded
transmission, the channel noise can directly act as a privacy-inducing
mechanism. This paper demonstrates that, as long as the privacy constraint
level, measured via differential privacy (DP), is below a threshold that
decreases with the signal-to-noise ratio (SNR), uncoded transmission achieves
privacy "for free", i.e., without affecting the learning performance. More
generally, this work studies adaptive power allocation (PA) for decentralized
gradient descent in wireless FL with the aim of minimizing the learning
optimality gap under privacy and power constraints. Both orthogonal multiple
access (OMA) and non-orthogonal multiple access (NOMA) transmission with
"over-the-air-computing" are studied, and solutions are obtained in closed form
for an offline optimization setting. Furthermore, heuristic online methods are
proposed that leverage iterative one-step-ahead optimization. The importance of
dynamic PA and the potential benefits of NOMA versus OMA are demonstrated
through extensive simulations.

    

### [[2011.04902] Windowed Backoff Algorithms for WiFi: Theory and Performance under Batched Arrivals](http://arxiv.org/abs/2011.04902)


  Binary exponential backoff (BEB) is a decades-old algorithm for coordinating
access to a shared channel. In modern networks, BEB plays an important role in
WiFi (IEEE 802.11) and other wireless communication standards.
Despite this track record, well-known theoretical results indicate that under
bursty traffic BEB yields poor makespan, and superior algorithms are possible.
To date, the degree to which these findings impact performance in wireless
networks has not been examined.
To address this issue, we investigate one of the strongest cases against BEB:
a single burst batch of packets that simultaneously contend for access to a
wireless channel. Using Network Simulator 3, we incorporate into IEEE 802.11g
several newer algorithms that, while inspired by BEB, possess makespan
guarantees that are theoretically superior. Surprisingly, we discover that
these newer algorithms underperform BEB.
Investigating further, we identify as the culprit a common abstraction
regarding the cost of collisions. Our experimental results are complemented by
analytical arguments that the number of collisions - and not solely makespan -
is an important metric to optimize. We argue that these findings have
implications for the design of backoff algorithms in wireless networks.

    

### [[2106.02927] A Generalized Framework for Joint Dynamic Optimal RF Interface Setting and Next-Hop Selection in IoT networks with Same Source Requests](http://arxiv.org/abs/2106.02927)


  Various applications which run on the machines in a network such as
Internet-of-Things require different bandwidths. So each machine may select one
of its multiple Radio Frequency (RF) interfaces for machine-to-machine or
machine-to base-station communications according to required bandwidth. We have
proposed a generalized framework for joint dynamic optimal RF interface setting
and next-hop selection, which is suitable for networks with multiple base
stations, and source nodes that have the same requests for bandwidth.
Simulation results show average data rate of the source nodes may be increased
up to 117%.

    

### [[2110.04301] Causal ImageNet: How to discover spurious features in Deep Learning?](http://arxiv.org/abs/2110.04301)


  A key reason for the lack of reliability of deep neural networks in the real
world is their heavy reliance on {\it spurious} input features that are
causally unrelated to the true label. Focusing on image classifications, we
define causal attributes as the set of visual features that are always a part
of the object while spurious attributes are the ones that are likely to {\it
co-occur} with the object but not a part of it (e.g., attribute ``fingers" for
class ``band aid"). Traditional methods for discovering spurious features
either require extensive human annotations (thus, not scalable), or are useful
on specific models. In this work, we introduce a {\it scalable} framework to
discover a subset of spurious and causal visual attributes used in inferences
of a general model and localize them on a large number of images with minimal
human supervision. Our methodology is based on this key idea: to identify
spurious or causal \textit{visual attributes} used in model predictions, we
identify spurious or causal \textit{neural features} (penultimate layer neurons
of a robust model) via limited human supervision (e.g., using top 5 activating
images per feature). We then show that these neural feature annotations {\it
generalize} extremely well to many more images {\it without} any human
supervision. We use the activation maps for these neural features as the soft
masks to highlight spurious or causal visual attributes. Using this
methodology, we introduce the {\it Causal Imagenet} dataset containing causal
and spurious masks for a large set of samples from Imagenet. We assess the
performance of several popular Imagenet models and show that they rely heavily
on various spurious features in their predictions.

    

### [[2110.04316] COVID-19 Face Mask Recognition with Advanced Face Cut Algorithm for Human Safety Measures](http://arxiv.org/abs/2110.04316)


  In the last year, the outbreak of COVID-19 has deployed computer vision and
machine learning algorithms in various fields to enhance human life
interactions. COVID-19 is a highly contaminated disease that affects mainly the
respiratory organs of the human body. We must wear a mask in this situation as
the virus can be contaminated through the air and a non-masked person can be
affected. Our proposal deploys a computer vision and deep learning framework to
recognize face masks from images or videos. We have implemented a Boundary
dependent face cut recognition algorithm that can cut the face from the image
using 27 landmarks and then the preprocessed image can further be sent to the
deep learning ResNet50 model. The experimental result shows a significant
advancement of 3.4 percent compared to the YOLOV3 mask recognition architecture
in just 10 epochs.

    

### [[2110.04318] Learning a Self-Expressive Network for Subspace Clustering](http://arxiv.org/abs/2110.04318)


  State-of-the-art subspace clustering methods are based on self-expressive
model, which represents each data point as a linear combination of other data
points. However, such methods are designed for a finite sample dataset and lack
the ability to generalize to out-of-sample data. Moreover, since the number of
self-expressive coefficients grows quadratically with the number of data
points, their ability to handle large-scale datasets is often limited. In this
paper, we propose a novel framework for subspace clustering, termed
Self-Expressive Network (SENet), which employs a properly designed neural
network to learn a self-expressive representation of the data. We show that our
SENet can not only learn the self-expressive coefficients with desired
properties on the training data, but also handle out-of-sample data. Besides,
we show that SENet can also be leveraged to perform subspace clustering on
large-scale datasets. Extensive experiments conducted on synthetic data and
real world benchmark data validate the effectiveness of the proposed method. In
particular, SENet yields highly competitive performance on MNIST, Fashion MNIST
and Extended MNIST and state-of-the-art performance on CIFAR-10. The code is
available at this https URL.

    

### [[2110.04321] Computing an Optimal Pitching Strategy in a Baseball At-Bat](http://arxiv.org/abs/2110.04321)


  The field of quantitative analytics has transformed the world of sports over
the last decade. To date, these analytic approaches are statistical at their
core, characterizing what is and what was, while using this information to
drive decisions about what to do in the future. However, as we often view team
sports, such as soccer, hockey, and baseball, as pairwise win-lose encounters,
it seems natural to model these as zero-sum games. We propose such a model for
one important class of sports encounters: a baseball at-bat, which is a matchup
between a pitcher and a batter. Specifically, we propose a novel model of this
encounter as a zero-sum stochastic game, in which the goal of the batter is to
get on base, an outcome the pitcher aims to prevent. The value of this game is
the on-base percentage (i.e., the probability that the batter gets on base). In
principle, this stochastic game can be solved using classical approaches. The
main technical challenges lie in predicting the distribution of pitch locations
as a function of pitcher intention, predicting the distribution of outcomes if
the batter decides to swing at a pitch, and characterizing the level of
patience of a particular batter. We address these challenges by proposing novel
pitcher and batter representations as well as a novel deep neural network
architecture for outcome prediction. Our experiments using Kaggle data from the
2015 to 2018 Major League Baseball seasons demonstrate the efficacy of the
proposed approach.

    

### [[2110.04328] Distinguishing rule- and exemplar-based generalization in learning systems](http://arxiv.org/abs/2110.04328)


  Despite the increasing scale of datasets in machine learning, generalization
to unseen regions of the data distribution remains crucial. Such extrapolation
is by definition underdetermined and is dictated by a learner's inductive
biases. Machine learning systems often do not share the same inductive biases
as humans and, as a result, extrapolate in ways that are inconsistent with our
expectations. We investigate two distinct such inductive biases: feature-level
bias (differences in which features are more readily learned) and
exemplar-vs-rule bias (differences in how these learned features are used for
generalization). Exemplar- vs. rule-based generalization has been studied
extensively in cognitive psychology, and, in this work, we present a protocol
inspired by these experimental approaches for directly probing this trade-off
in learning systems. The measures we propose characterize changes in
extrapolation behavior when feature coverage is manipulated in a combinatorial
setting. We present empirical results across a range of models and across both
expository and real-world image and language domains. We demonstrate that
measuring the exemplar-rule trade-off while controlling for feature-level bias
provides a more complete picture of extrapolation behavior than existing
formalisms. We find that most standard neural network models have a propensity
towards exemplar-based extrapolation and discuss the implications of these
findings for research on data augmentation, fairness, and systematic
generalization.

    

### [[2110.04330] KG-FiD: Infusing Knowledge Graph in Fusion-in-Decoder for Open-Domain Question Answering](http://arxiv.org/abs/2110.04330)


  Current Open-Domain Question Answering (ODQA) model paradigm often contains a
retrieving module and a reading module. Given an input question, the reading
module predicts the answer from the relevant passages which are retrieved by
the retriever. The recent proposed Fusion-in-Decoder (FiD), which is built on
top of the pretrained generative model T5, achieves the state-of-the-art
performance in the reading module. Although being effective, it remains
constrained by inefficient attention on all retrieved passages which contain a
lot of noise. In this work, we propose a novel method KG-FiD, which filters
noisy passages by leveraging the structural relationship among the retrieved
passages with a knowledge graph. We initiate the passage node embedding from
the FiD encoder and then use graph neural network (GNN) to update the
representation for reranking. To improve the efficiency, we build the GNN on
top of the intermediate layer output of the FiD encoder and only pass a few top
reranked passages into the higher layers of encoder and decoder for answer
generation. We also apply the proposed GNN based reranking method to enhance
the passage retrieval results in the retrieving module. Extensive experiments
on common ODQA benchmark datasets (Natural Question and TriviaQA) demonstrate
that KG-FiD can improve vanilla FiD by up to 1.5% on answer exact match score
and achieve comparable performance with FiD with only 40% of computation cost.

    

### [[2110.04337] Adversarial Token Attacks on Vision Transformers](http://arxiv.org/abs/2110.04337)


  Vision transformers rely on a patch token based self attention mechanism, in
contrast to convolutional networks. We investigate fundamental differences
between these two families of models, by designing a block sparsity based
adversarial token attack. We probe and analyze transformer as well as
convolutional models with token attacks of varying patch sizes. We infer that
transformer models are more sensitive to token attacks than convolutional
models, with ResNets outperforming Transformer models by up to $\sim30\%$ in
robust accuracy for single token attacks.

    

### [[2110.04338] Learning from non-irreducible Markov chains](http://arxiv.org/abs/2110.04338)


  Most of the existing literature on supervised learning problems focuses on
the case when the training data set is drawn from an i.i.d. sample. However,
many practical supervised learning problems are characterized by temporal
dependence and strong correlation between the marginals of the data-generating
process, suggesting that the i.i.d. assumption is not always justified. This
problem has been already considered in the context of Markov chains satisfying
the Doeblin condition. This condition, among other things, implies that the
chain is not singular in its behavior, i.e. it is irreducible. In this article,
we focus on the case when the training data set is drawn from a not necessarily
irreducible Markov chain. Under the assumption that the chain is uniformly
ergodic with respect to the $\mathrm{L}^1$-Wasserstein distance, and certain
regularity assumptions on the hypothesis class and the state space of the
chain, we first obtain a uniform convergence result for the corresponding
sample error, and then we conclude learnability of the approximate sample error
minimization algorithm and find its generalization bounds. At the end, a
relative uniform convergence result for the sample error is also discussed.

    

### [[2110.04347] Towards Sample-efficient Apprenticeship Learning from Suboptimal Demonstration](http://arxiv.org/abs/2110.04347)


  Learning from Demonstration (LfD) seeks to democratize robotics by enabling
non-roboticist end-users to teach robots to perform novel tasks by providing
demonstrations. However, as demonstrators are typically non-experts, modern LfD
techniques are unable to produce policies much better than the suboptimal
demonstration. A previously-proposed framework, SSRR, has shown success in
learning from suboptimal demonstration but relies on noise-injected
trajectories to infer an idealized reward function. A random approach such as
noise-injection to generate trajectories has two key drawbacks: 1) Performance
degradation could be random depending on whether the noise is applied to vital
states and 2) Noise-injection generated trajectories may have limited
suboptimality and therefore will not accurately represent the whole scope of
suboptimality. We present Systematic Self-Supervised Reward Regression, S3RR,
to investigate systematic alternatives for trajectory degradation. We carry out
empirical evaluations and find S3RR can learn comparable or better reward
correlation with ground-truth against a state-of-the-art learning from
suboptimal demonstration framework.

    

### [[2110.04350] FSL: Federated Supermask Learning](http://arxiv.org/abs/2110.04350)


  Federated learning (FL) allows multiple clients with (private) data to
collaboratively train a common machine learning model without sharing their
private training data. In-the-wild deployment of FL faces two major hurdles:
robustness to poisoning attacks and communication efficiency. To address these
concurrently, we propose Federated Supermask Learning (FSL). FSL server trains
a global subnetwork within a randomly initialized neural network by aggregating
local subnetworks of all collaborating clients. FSL clients share local
subnetworks in the form of rankings of network edges; more useful edges have
higher ranks. By sharing integer rankings, instead of float weights, FSL
restricts the space available to craft effective poisoning updates, and by
sharing subnetworks, FSL reduces the communication cost of training. We show
theoretically and empirically that FSL is robust by design and also
significantly communication efficient; all this without compromising clients'
privacy. Our experiments demonstrate the superiority of FSL in real-world FL
settings; in particular, (1) FSL achieves similar performances as
state-of-the-art FedAvg with significantly lower communication costs: for
CIFAR10, FSL achieves same performance as Federated Averaging while reducing
communication cost by ~35%. (2) FSL is substantially more robust to poisoning
attacks than state-of-the-art robust aggregation algorithms. We have released
the code for reproducibility.

    

### [[2110.04352] Hankel-structured Tensor Robust PCA for Multivariate Traffic Time Series Anomaly Detection](http://arxiv.org/abs/2110.04352)


  Spatiotemporal traffic data (e.g., link speed/flow) collected from sensor
networks can be organized as multivariate time series with additional spatial
attributes. A crucial task in analyzing such data is to identify and detect
anomalous observations and events from the data with complex spatial and
temporal dependencies. Robust Principal Component Analysis (RPCA) is a widely
used tool for anomaly detection. However, the traditional RPCA purely relies on
the global low-rank assumption while ignoring the local temporal correlations.
In light of this, this study proposes a Hankel-structured tensor version of
RPCA for anomaly detection in spatiotemporal data. We treat the raw data with
anomalies as a multivariate time series matrix (location $\times$ time) and
assume the denoised matrix has a low-rank structure. Then we transform the
low-rank matrix to a third-order tensor by applying temporal Hankelization. In
the end, we decompose the corrupted matrix into a low-rank Hankel tensor and a
sparse matrix. With the Hankelization operation, the model can simultaneously
capture the global and local spatiotemporal correlations and exhibit more
robust performance. We formulate the problem as an optimization problem and use
tensor nuclear norm (TNN) to approximate the tensor rank and $l_1$ norm to
approximate the sparsity. We develop an efficient solution algorithm based on
the Alternating Direction Method of Multipliers (ADMM). Despite having three
hyper-parameters, the model is easy to set in practice. We evaluate the
proposed method by synthetic data and metro passenger flow time series and the
results demonstrate the accuracy of anomaly detection.

    

### [[2110.04357] Training Transition Policies via Distribution Matching for Complex Tasks](http://arxiv.org/abs/2110.04357)


  Humans decompose novel complex tasks into simpler ones to exploit previously
learned skills. Analogously, hierarchical reinforcement learning seeks to
leverage lower-level policies for simple tasks to solve complex ones. However,
because each lower-level policy induces a different distribution of states,
transitioning from one lower-level policy to another may fail due to an
unexpected starting state. We introduce transition policies that smoothly
connect lower-level policies by producing a distribution of states and actions
that matches what is expected by the next policy. Training transition policies
is challenging because the natural reward signal -- whether the next policy can
execute its subtask successfully -- is sparse. By training transition policies
via adversarial inverse reinforcement learning to match the distribution of
expected states and actions, we avoid relying on task-based reward. To further
improve performance, we use deep Q-learning with a binary action space to
determine when to switch from a transition policy to the next pre-trained
policy, using the success or failure of the next subtask as the reward.
Although the reward is still sparse, the problem is less severe due to the
simple binary action space. We demonstrate our method on continuous bipedal
locomotion and arm manipulation tasks that require diverse skills. We show that
it smoothly connects the lower-level policies, achieving higher success rates
than previous methods that search for successful trajectories based on a reward
function, but do not match the state distribution.

    

### [[2110.04361] SubTab: Subsetting Features of Tabular Data for Self-Supervised Representation Learning](http://arxiv.org/abs/2110.04361)


  Self-supervised learning has been shown to be very effective in learning
useful representations, and yet much of the success is achieved in data types
such as images, audio, and text. The success is mainly enabled by taking
advantage of spatial, temporal, or semantic structure in the data through
augmentation. However, such structure may not exist in tabular datasets
commonly used in fields such as healthcare, making it difficult to design an
effective augmentation method, and hindering a similar progress in tabular data
setting. In this paper, we introduce a new framework, Subsetting features of
Tabular data (SubTab), that turns the task of learning from tabular data into a
multi-view representation learning problem by dividing the input features to
multiple subsets. We argue that reconstructing the data from the subset of its
features rather than its corrupted version in an autoencoder setting can better
capture its underlying latent representation. In this framework, the joint
representation can be expressed as the aggregate of latent variables of the
subsets at test time, which we refer to as collaborative inference. Our
experiments show that the SubTab achieves the state of the art (SOTA)
performance of 98.31% on MNIST in tabular setting, on par with CNN-based SOTA
models, and surpasses existing baselines on three other real-world datasets by
a significant margin.

    

### [[2110.04363] Certifying Robustness to Programmable Data Bias in Decision Trees](http://arxiv.org/abs/2110.04363)


  Datasets can be biased due to societal inequities, human biases,
under-representation of minorities, etc. Our goal is to certify that models
produced by a learning algorithm are pointwise-robust to potential dataset
biases. This is a challenging problem: it entails learning models for a large,
or even infinite, number of datasets, ensuring that they all produce the same
prediction. We focus on decision-tree learning due to the interpretable nature
of the models. Our approach allows programmatically specifying bias models
across a variety of dimensions (e.g., missing data for minorities), composing
types of bias, and targeting bias towards a specific group. To certify
robustness, we use a novel symbolic technique to evaluate a decision-tree
learner on a large, or infinite, number of datasets, certifying that each and
every dataset produces the same prediction for a specific test point. We
evaluate our approach on datasets that are commonly used in the fairness
literature, and demonstrate our approach's viability on a range of bias models.

    

### [[2110.04366] Towards a Unified View of Parameter-Efficient Transfer Learning](http://arxiv.org/abs/2110.04366)


  Fine-tuning large pre-trained language models on downstream tasks has become
the de-facto learning paradigm in NLP. However, conventional approaches
fine-tune all the parameters of the pre-trained model, which becomes
prohibitive as the model size and the number of tasks grow. Recent work has
proposed a variety of parameter-efficient transfer learning methods that only
fine-tune a small number of (extra) parameters to attain strong performance.
While effective, the critical ingredients for success and the connections among
the various methods are poorly understood. In this paper, we break down the
design of state-of-the-art parameter-efficient transfer learning methods and
present a unified framework that establishes connections between them.
Specifically, we re-frame them as modifications to specific hidden states in
pre-trained models, and define a set of design dimensions along which different
methods vary, such as the function to compute the modification and the position
to apply the modification. Through comprehensive empirical studies across
machine translation, text summarization, language understanding, and text
classification benchmarks, we utilize the unified view to identify important
design choices in previous methods. Furthermore, our unified framework enables
the transfer of design elements across different approaches, and as a result we
are able to instantiate new parameter-efficient fine-tuning methods that tune
less parameters than previous methods while being more effective, achieving
comparable results to fine-tuning all parameters on all four tasks.

    

### [[2110.04367] Hybrid Random Features](http://arxiv.org/abs/2110.04367)


  We propose a new class of random feature methods for linearizing softmax and
Gaussian kernels called hybrid random features (HRFs) that automatically adapt
the quality of kernel estimation to provide most accurate approximation in the
defined regions of interest. Special instantiations of HRFs lead to well-known
methods such as trigonometric (Rahimi and Recht, 2007) or (recently introduced
in the context of linear-attention Transformers) positive random features
(Choromanski et al., 2021). By generalizing Bochner's Theorem for
softmax/Gaussian kernels and leveraging random features for compositional
kernels, the HRF-mechanism provides strong theoretical guarantees - unbiased
approximation and strictly smaller worst-case relative errors than its
counterparts. We conduct exhaustive empirical evaluation of HRF ranging from
pointwise kernel estimation experiments, through tests on data admitting
clustering structure to benchmarking implicit-attention Transformers (also for
downstream Robotics applications), demonstrating its quality in a wide spectrum
of machine learning problems.

    

### [[2110.04369] A Loss Curvature Perspective on Training Instability in Deep Learning](http://arxiv.org/abs/2110.04369)


  In this work, we study the evolution of the loss Hessian across many
classification tasks in order to understand the effect the curvature of the
loss has on the training dynamics. Whereas prior work has focused on how
different learning rates affect the loss Hessian observed during training, we
also analyze the effects of model initialization, architectural choices, and
common training heuristics such as gradient clipping and learning rate warmup.
Our results demonstrate that successful model and hyperparameter choices allow
the early optimization trajectory to either avoid -- or navigate out of --
regions of high curvature and into flatter regions that tolerate a higher
learning rate. Our results suggest a unifying perspective on how disparate
mitigation strategies for training instability ultimately address the same
underlying failure mode of neural network optimization, namely poor
conditioning. Inspired by the conditioning perspective, we show that learning
rate warmup can improve training stability just as much as batch normalization,
layer normalization, MetaInit, GradInit, and Fixup initialization.

    

### [[2110.04372] Fair Regression under Sample Selection Bias](http://arxiv.org/abs/2110.04372)


  Recent research on fair regression focused on developing new fairness notions
and approximation methods as target variables and even the sensitive attribute
are continuous in the regression setting. However, all previous fair regression
research assumed the training data and testing data are drawn from the same
distributions. This assumption is often violated in real world due to the
sample selection bias between the training and testing data. In this paper, we
develop a framework for fair regression under sample selection bias when
dependent variable values of a set of samples from the training data are
missing as a result of another hidden process. Our framework adopts the classic
Heckman model for bias correction and the Lagrange duality to achieve fairness
in regression based on a variety of fairness notions. Heckman model describes
the sample selection process and uses a derived variable called the Inverse
Mills Ratio (IMR) to correct sample selection bias. We use fairness inequality
and equality constraints to describe a variety of fairness notions and apply
the Lagrange duality theory to transform the primal problem into the dual
convex optimization. For the two popular fairness notions, mean difference and
mean squared error difference, we derive explicit formulas without iterative
optimization, and for Pearson correlation, we derive its conditions of
achieving strong duality. We conduct experiments on three real-world datasets
and the experimental results demonstrate the approach's effectiveness in terms
of both utility and fairness metrics.

    

### [[2110.04375] Neural Link Prediction with Walk Pooling](http://arxiv.org/abs/2110.04375)


  Graph neural networks achieve high accuracy in link prediction by jointly
leveraging graph topology and node attributes. Topology, however, is
represented indirectly; state-of-the-art methods based on subgraph
classification label nodes with distance to the target link, so that, although
topological information is present, it is tempered by pooling. This makes it
challenging to leverage features like loops and motifs associated with network
formation mechanisms. We propose a link prediction algorithm based on a new
pooling scheme called WalkPool. WalkPool combines the expressivity of
topological heuristics with the feature-learning ability of neural networks. It
summarizes a putative link by random walk probabilities of adjacent paths.
Instead of extracting transition probabilities from the original graph, it
computes the transition matrix of a "predictive" latent graph by applying
attention to learned features; this may be interpreted as feature-sensitive
topology fingerprinting. WalkPool can leverage unsupervised node features or be
combined with GNNs and trained end-to-end. It outperforms state-of-the-art
methods on all common link prediction benchmarks, both homophilic and
heterophilic, with and without node attributes. Applying WalkPool to a set of
unsupervised GNNs significantly improves prediction accuracy, suggesting that
it may be used as a general-purpose graph pooling scheme.

    

### [[2110.04378] Performance optimizations on deep noise suppression models](http://arxiv.org/abs/2110.04378)


  We study the role of magnitude structured pruning as an architecture search
to speed up the inference time of a deep noise suppression (DNS) model. While
deep learning approaches have been remarkably successful in enhancing audio
quality, their increased complexity inhibits their deployment in real-time
applications. We achieve up to a 7.25X inference speedup over the baseline,
with a smooth model performance degradation. Ablation studies indicate that our
proposed network re-parameterization (i.e., size per layer) is the major driver
of the speedup, and that magnitude structured pruning does comparably to
directly training a model in the smaller size. We report inference speed
because a parameter reduction does not necessitate speedup, and we measure
model quality using an accurate non-intrusive objective speech quality metric.

    

### [[2110.04383] Learning 3D Representations of Molecular Chirality with Invariance to Bond Rotations](http://arxiv.org/abs/2110.04383)


  Molecular chirality, a form of stereochemistry most often describing relative
spatial arrangements of bonded neighbors around tetrahedral carbon centers,
influences the set of 3D conformers accessible to the molecule without changing
its 2D graph connectivity. Chirality can strongly alter (bio)chemical
interactions, particularly protein-drug binding. Most 2D graph neural networks
(GNNs) designed for molecular property prediction at best use atomic labels to
naïvely treat chirality, while E(3)-invariant 3D GNNs are invariant to
chirality altogether. To enable representation learning on molecules with
defined stereochemistry, we design an SE(3)-invariant model that processes
torsion angles of a 3D molecular conformer. We explicitly model conformational
flexibility by integrating a novel type of invariance to rotations about
internal molecular bonds into the architecture, mitigating the need for
multi-conformer data augmentation. We test our model on four benchmarks:
contrastive learning to distinguish conformers of different stereoisomers in a
learned latent space, classification of chiral centers as R/S, prediction of
how enantiomers rotate circularly polarized light, and ranking enantiomers by
their docking scores in an enantiosensitive protein pocket. We compare our
model, Chiral InterRoto-Invariant Neural Network (ChIRo), with 2D and 3D GNNs
to demonstrate that our model achieves state of the art performance when
learning chiral-sensitive functions from molecular structures.

    

### [[2110.04396] When to Call Your Neighbor? Strategic Communication in Cooperative Stochastic Bandits](http://arxiv.org/abs/2110.04396)


  In cooperative bandits, a framework that captures essential features of
collective sequential decision making, agents can minimize group regret, and
thereby improve performance, by leveraging shared information. However, sharing
information can be costly, which motivates developing policies that minimize
group regret while also reducing the number of messages communicated by agents.
Existing cooperative bandit algorithms obtain optimal performance when agents
share information with their neighbors at \textit{every time step}, i.e., full
communication. This requires $\Theta(T)$ number of messages, where $T$ is the
time horizon of the decision making process. We propose \textit{ComEx}, a novel
cost-effective communication protocol in which the group achieves the same
order of performance as full communication while communicating only $O(\log T)$
number of messages. Our key step is developing a method to identify and only
communicate the information crucial to achieving optimal performance. Further
we propose novel algorithms for several benchmark cooperative bandit frameworks
and show that our algorithms obtain \textit{state-of-the-art} performance while
consistently incurring a significantly smaller communication cost than existing
algorithms.

    

### [[2110.04397] Measure Twice, Cut Once: Quantifying Bias and Fairness in Deep Neural Networks](http://arxiv.org/abs/2110.04397)


  Algorithmic bias is of increasing concern, both to the research community,
and society at large. Bias in AI is more abstract and unintuitive than
traditional forms of discrimination and can be more difficult to detect and
mitigate. A clear gap exists in the current literature on evaluating the
relative bias in the performance of multi-class classifiers. In this work, we
propose two simple yet effective metrics, Combined Error Variance (CEV) and
Symmetric Distance Error (SDE), to quantitatively evaluate the class-wise bias
of two models in comparison to one another. By evaluating the performance of
these new metrics and by demonstrating their practical application, we show
that they can be used to measure fairness as well as bias. These demonstrations
show that our metrics can address specific needs for measuring bias in
multi-class classification.

    

### [[2110.04403] Temperature as Uncertainty in Contrastive Learning](http://arxiv.org/abs/2110.04403)


  Contrastive learning has demonstrated great capability to learn
representations without annotations, even outperforming supervised baselines.
However, it still lacks important properties useful for real-world application,
one of which is uncertainty. In this paper, we propose a simple way to generate
uncertainty scores for many contrastive methods by re-purposing temperature, a
mysterious hyperparameter used for scaling. By observing that temperature
controls how sensitive the objective is to specific embedding locations, we aim
to learn temperature as an input-dependent variable, treating it as a measure
of embedding confidence. We call this approach "Temperature as Uncertainty", or
TaU. Through experiments, we demonstrate that TaU is useful for
out-of-distribution detection, while remaining competitive with benchmarks on
linear evaluation. Moreover, we show that TaU can be learned on top of
pretrained models, enabling uncertainty scores to be generated post-hoc with
popular off-the-shelf models. In summary, TaU is a simple yet versatile method
for generating uncertainties for contrastive learning. Open source code can be
found at: this https URL.

    

### [[2110.04414] Gated recurrent units and temporal convolutional network for multilabel classification](http://arxiv.org/abs/2110.04414)


  Multilabel learning tackles the problem of associating a sample with multiple
class labels. This work proposes a new ensemble method for managing multilabel
classification: the core of the proposed approach combines a set of gated
recurrent units and temporal convolutional neural networks trained with
variants of the Adam optimization approach. Multiple Adam variants, including
novel one proposed here, are compared and tested; these variants are based on
the difference between present and past gradients, with step size adjusted for
each parameter. The proposed neural network approach is also combined with
Incorporating Multiple Clustering Centers (IMCC), which further boosts
classification performance. Multiple experiments on nine data sets representing
a wide variety of multilabel tasks demonstrate the robustness of our best
ensemble, which is shown to outperform the state-of-the-art. The MATLAB code
for generating the best ensembles in the experimental section will be available
at this https URL.

    

### [[2110.04421] DeepABM: Scalable, efficient and differentiable agent-based simulations via graph neural networks](http://arxiv.org/abs/2110.04421)


  We introduce DeepABM, a framework for agent-based modeling that leverages
geometric message passing of graph neural networks for simulating action and
interactions over large agent populations. Using DeepABM allows scaling
simulations to large agent populations in real-time and running them
efficiently on GPU architectures. To demonstrate the effectiveness of DeepABM,
we build DeepABM-COVID simulator to provide support for various
non-pharmaceutical interventions (quarantine, exposure notification,
vaccination, testing) for the COVID-19 pandemic, and can scale to populations
of representative size in real-time on a GPU. Specifically, DeepABM-COVID can
model 200 million interactions (over 100,000 agents across 180 time-steps) in
90 seconds, and is made available online to help researchers with modeling and
analysis of various interventions. We explain various components of the
framework and discuss results from one research study to evaluate the impact of
delaying the second dose of the COVID-19 vaccine in collaboration with clinical
and public health experts. While we simulate COVID-19 spread, the ideas
introduced in the paper are generic and can be easily extend to other forms of
agent-based simulations. Furthermore, while beyond scope of this document,
DeepABM enables inverse agent-based simulations which can be used to learn
physical parameters in the (micro) simulations using gradient-based
optimization with large-scale real-world (macro) data. We are optimistic that
the current work can have interesting implications for bringing ABM and AI
communities closer.

    

### [[2110.04422] Theoretically Principled Deep RL Acceleration via Nearest Neighbor Function Approximation](http://arxiv.org/abs/2110.04422)


  Recently, deep reinforcement learning (RL) has achieved remarkable empirical
success by integrating deep neural networks into RL frameworks. However, these
algorithms often require a large number of training samples and admit little
theoretical understanding. To mitigate these issues, we propose a theoretically
principled nearest neighbor (NN) function approximator that can improve the
value networks in deep RL methods. Inspired by human similarity judgments, the
NN approximator estimates the action values using rollouts on past observations
and can provably obtain a small regret bound that depends only on the intrinsic
complexity of the environment. We present (1) Nearest Neighbor Actor-Critic
(NNAC), an online policy gradient algorithm that demonstrates the practicality
of combining function approximation with deep RL, and (2) a plug-and-play NN
update module that aids the training of existing deep RL methods. Experiments
on classical control and MuJoCo locomotion tasks show that the NN-accelerated
agents achieve higher sample efficiency and stability than the baseline agents.
Based on its theoretical benefits, we believe that the NN approximator can be
further applied to other complex domains to speed-up learning.

    

### [[2110.04425] Arabic Speech Emotion Recognition Employing Wav2vec2.0 and HuBERT Based on BAVED Dataset](http://arxiv.org/abs/2110.04425)


  Recently, there have been tremendous research outcomes in the fields of
speech recognition and natural language processing. This is due to the
well-developed multi-layers deep learning paradigms such as wav2vec2.0,
Wav2vecU, WavBERT, and HuBERT that provide better representation learning and
high information capturing. Such paradigms run on hundreds of unlabeled data,
then fine-tuned on a small dataset for specific tasks. This paper introduces a
deep learning constructed emotional recognition model for Arabic speech
dialogues. The developed model employs the state of the art audio
representations include wav2vec2.0 and HuBERT. The experiment and performance
results of our model overcome the previous known outcomes.

    

### [[2110.04427] Harnessing Unlabeled Data to Improve Generalization of Biometric Gender and Age Classifiers](http://arxiv.org/abs/2110.04427)


  With significant advances in deep learning, many computer vision applications
have reached the inflection point. However, these deep learning models need
large amount of labeled data for model training and optimum parameter
estimation. Limited labeled data for model training results in over-fitting and
impacts their generalization performance. However, the collection and
annotation of large amount of data is a very time consuming and expensive
operation. Further, due to privacy and security concerns, the large amount of
labeled data could not be collected for certain applications such as those
involving medical field. Self-training, Co-training, and Self-ensemble methods
are three types of semi-supervised learning methods that can be used to exploit
unlabeled data. In this paper, we propose self-ensemble based deep learning
model that along with limited labeled data, harness unlabeled data for
improving the generalization performance. We evaluated the proposed
self-ensemble based deep-learning model for soft-biometric gender and age
classification. Experimental evaluation on CelebA and VISOB datasets suggest
gender classification accuracy of 94.46% and 81.00%, respectively, using only
1000 labeled samples and remaining 199k samples as unlabeled samples for CelebA
dataset and similarly,1000 labeled samples with remaining 107k samples as
unlabeled samples for VISOB dataset. Comparative evaluation suggest that there
is $5.74\%$ and $8.47\%$ improvement in the accuracy of the self-ensemble model
when compared with supervised model trained on the entire CelebA and VISOB
dataset, respectively. We also evaluated the proposed learning method for
age-group prediction on Adience dataset and it outperformed the baseline
supervised deep-learning learning model with a better exact accuracy of 55.55
$\pm$ 4.28 which is 3.92% more than the baseline.

    

### [[2110.04442] Deep Learning of Potential Outcomes](http://arxiv.org/abs/2110.04442)


  This review systematizes the emerging literature for causal inference using
deep neural networks under the potential outcomes framework. It provides an
intuitive introduction on how deep learning can be used to estimate/predict
heterogeneous treatment effects and extend causal inference to settings where
confounding is non-linear, time varying, or encoded in text, networks, and
images. To maximize accessibility, we also introduce prerequisite concepts from
causal inference and deep learning. The survey differs from other treatments of
deep learning and causal inference in its sharp focus on observational causal
estimation, its extended exposition of key algorithms, and its detailed
tutorials for implementing, training, and selecting among deep estimators in
Tensorflow 2 available at this http URL.

    

### [[2110.04450] Scene Editing as Teleoperation: A Case Study in 6DoF Kit Assembly](http://arxiv.org/abs/2110.04450)


  Studies in robot teleoperation have been centered around action
specifications -- from continuous joint control to discrete end-effector pose
control. However, these robot-centric interfaces often require skilled
operators with extensive robotics expertise. To make teleoperation accessible
to non-expert users, we propose the framework "Scene Editing as Teleoperation"
(SEaT), where the key idea is to transform the traditional "robot-centric"
interface into a "scene-centric" interface -- instead of controlling the robot,
users focus on specifying the task's goal by manipulating digital twins of the
real-world objects. As a result, a user can perform teleoperation without any
expert knowledge of the robot hardware. To achieve this goal, we utilize a
category-agnostic scene-completion algorithm that translates the real-world
workspace (with unknown objects) into a manipulable virtual scene
representation and an action-snapping algorithm that refines the user input
before generating the robot's action plan. To train the algorithms, we
procedurally generated a large-scale, diverse kit-assembly dataset that
contains object-kit pairs that mimic real-world object-kitting tasks. Our
experiments in simulation and on a real-world system demonstrate that our
framework improves both the efficiency and success rate for 6DoF kit-assembly
tasks. A user study demonstrates that SEaT framework participants achieve a
higher task success rate and report a lower subjective workload compared to an
alternative robot-centric interface. Video can be found at
this https URL .

    

### [[2110.04456] Deep Joint Source-Channel Coding for Wireless Image Transmission with Adaptive Rate Control](http://arxiv.org/abs/2110.04456)


  We present a novel adaptive deep joint source-channel coding (JSCC) scheme
for wireless image transmission. The proposed scheme supports multiple rates
using a single deep neural network (DNN) model and learns to dynamically
control the rate based on the channel condition and image contents.
Specifically, a policy network is introduced to exploit the tradeoff space
between the rate and signal quality. To train the policy network, the
Gumbel-Softmax trick is adopted to make the policy network differentiable and
hence the whole JSCC scheme can be trained end-to-end. To the best of our
knowledge, this is the first deep JSCC scheme that can automatically adjust its
rate using a single network model. Experiments show that our scheme
successfully learns a reasonable policy that decreases channel bandwidth
utilization for high SNR scenarios or simple image contents. For an arbitrary
target rate, our rate-adaptive scheme using a single model achieves similar
performance compared to an optimized model specifically trained for that fixed
target rate. To reproduce our results, we make the source code publicly
available at this https URL.

    

### [[2110.04458] Vision Transformer based COVID-19 Detection using Chest X-rays](http://arxiv.org/abs/2110.04458)


  COVID-19 is a global pandemic, and detecting them is a momentous task for
medical professionals today due to its rapid mutations. Current methods of
examining chest X-rays and CT scan requires profound knowledge and are time
consuming, which suggests that it shrinks the precious time of medical
practitioners when people's lives are at stake. This study tries to assist this
process by achieving state-of-the-art performance in classifying chest X-rays
by fine-tuning Vision Transformer(ViT). The proposed approach uses pretrained
models, fine-tuned for detecting the presence of COVID-19 disease on chest
X-rays. This approach achieves an accuracy score of 97.61%, precision score of
95.34%, recall score of 93.84% and, f1-score of 94.58%. This result signifies
the performance of transformer-based models on chest X-ray.

    

### [[2110.04466] ProductAE: Towards Training Larger Channel Codes based on Neural Product Codes](http://arxiv.org/abs/2110.04466)


  There have been significant research activities in recent years to automate
the design of channel encoders and decoders via deep learning. Due the
dimensionality challenge in channel coding, it is prohibitively complex to
design and train relatively large neural channel codes via deep learning
techniques. Consequently, most of the results in the literature are limited to
relatively short codes having less than 100 information bits. In this paper, we
construct ProductAEs, a computationally efficient family of deep-learning
driven (encoder, decoder) pairs, that aim at enabling the training of
relatively large channel codes (both encoders and decoders) with a manageable
training complexity. We build upon the ideas from classical product codes, and
propose constructing large neural codes using smaller code components. More
specifically, instead of directly training the encoder and decoder for a large
neural code of dimension $k$ and blocklength $n$, we provide a framework that
requires training neural encoders and decoders for the code parameters
$(k_1,n_1)$ and $(k_2,n_2)$ such that $k_1 k_2=k$ and $n_1 n_2=n$. Our training
results show significant gains, over all ranges of signal-to-noise ratio (SNR),
for a code of parameters $(100,225)$ and a moderate-length code of parameters
$(196,441)$, over polar codes under successive cancellation (SC) decoder.
Moreover, our results demonstrate meaningful gains over Turbo Autoencoder
(TurboAE) and state-of-the-art classical codes. This is the first work to
design product autoencoders and a pioneering work on training large channel
codes.

    

### [[2110.04471] Provably Efficient Black-Box Action Poisoning Attacks Against Reinforcement Learning](http://arxiv.org/abs/2110.04471)


  Due to the broad range of applications of reinforcement learning (RL),
understanding the effects of adversarial attacks against RL model is essential
for the safe applications of this model. Prior works on adversarial attacks
against RL mainly focus on either observation poisoning attacks or environment
poisoning attacks. In this paper, we introduce a new class of attacks named
action poisoning attacks, where an adversary can change the action signal
selected by the agent. Compared with existing attack models, the attacker's
ability in the proposed action poisoning attack model is more restricted, and
hence the attack model is more practical. We study the action poisoning attack
in both white-box and black-box settings. We introduce an adaptive attack
scheme called LCB-H, which works for most RL agents in the black-box setting.
We prove that the LCB-H attack can force any efficient RL agent, whose dynamic
regret scales sublinearly with the total number of steps taken, to choose
actions according to a policy selected by the attacker very frequently, with
only sublinear cost. In addition, we apply LCB-H attack against a popular
model-free RL algorithm: UCB-H. We show that, even in the black-box setting, by
spending only logarithm cost, the proposed LCB-H attack scheme can force the
UCB-H agent to choose actions according to the policy selected by the attacker
very frequently.

    

### [[2110.04483] Visualizing the embedding space to explain the effect of knowledge distillation](http://arxiv.org/abs/2110.04483)


  Recent research has found that knowledge distillation can be effective in
reducing the size of a network and in increasing generalization. A pre-trained,
large teacher network, for example, was shown to be able to bootstrap a student
model that eventually outperforms the teacher in a limited label environment.
Despite these advances, it still is relatively unclear \emph{why} this method
works, that is, what the resulting student model does 'better'. To address this
issue, here, we utilize two non-linear, low-dimensional embedding methods
(t-SNE and IVIS) to visualize representation spaces of different layers in a
network. We perform a set of extensive experiments with different architecture
parameters and distillation methods. The resulting visualizations and metrics
clearly show that distillation guides the network to find a more compact
representation space for higher accuracy already in earlier layers compared to
its non-distilled version.

    

### [[2110.04485] Application of quantum computing to a linear non-Gaussian acyclic model for novel medical knowledge discovery](http://arxiv.org/abs/2110.04485)


  Recently, with the digitalization of medicine, the utilization of real-world
medical data collected from clinical sites has been attracting attention. In
this study, quantum computing was applied to a linear non-Gaussian acyclic
model to discover causal relationships from real-world medical data alone.
Specifically, the independence measure of DirectLiNGAM, a causal discovery
algorithm, was calculated using the quantum kernel and its accuracy on
real-world medical data was verified. When DirectLiNGAM with the quantum kernel
(qLiNGAM) was applied to real-world medical data, a case was confirmed in which
the causal structure could be correctly estimated when the amount of data was
small, which was not possible with existing methods. It is suggested that
qLiNGAM may be able to discover new medical knowledge and contribute to the
solution of medical problems, even when only a small amount of data is
available.

    

### [[2110.04486] PAMA-TTS: Progression-Aware Monotonic Attention for Stable Seq2Seq TTS With Accurate Phoneme Duration Control](http://arxiv.org/abs/2110.04486)


  Sequence expansion between encoder and decoder is a critical challenge in
sequence-to-sequence TTS. Attention-based methods achieve great naturalness but
suffer from unstable issues like missing and repeating phonemes, not to mention
accurate duration control. Duration-informed methods, on the contrary, seem to
easily adjust phoneme duration but show obvious degradation in speech
naturalness. This paper proposes PAMA-TTS to address the problem. It takes the
advantage of both flexible attention and explicit duration models. Based on the
monotonic attention mechanism, PAMA-TTS also leverages token duration and
relative position of a frame, especially countdown information, i.e. in how
many future frames the present phoneme will end. They help the attention to
move forward along the token sequence in a soft but reliable control.
Experimental results prove that PAMA-TTS achieves the highest naturalness,
while has on-par or even better duration controllability than the
duration-informed model.

    

### [[2110.04487] Colour augmentation for improved semi-supervised semantic segmentation](http://arxiv.org/abs/2110.04487)


  Consistency regularization describes a class of approaches that have yielded
state-of-the-art results for semi-supervised classification. While
semi-supervised semantic segmentation proved to be more challenging, a number
of successful approaches have been recently proposed. Recent work explored the
challenges involved in using consistency regularization for segmentation
problems. In their self-supervised work Chen et al. found that colour
augmentation prevents a classification network from using image colour
statistics as a short-cut for self-supervised learning via instance
discrimination. Drawing inspiration from this we find that a similar problem
impedes semi-supervised semantic segmentation and offer colour augmentation as
a solution, improving semi-supervised semantic segmentation performance on
challenging photographic imagery.

    

### [[2110.04495] Multi-Agent MDP Homomorphic Networks](http://arxiv.org/abs/2110.04495)


  This paper introduces Multi-Agent MDP Homomorphic Networks, a class of
networks that allows distributed execution using only local information, yet is
able to share experience between global symmetries in the joint state-action
space of cooperative multi-agent systems. In cooperative multi-agent systems,
complex symmetries arise between different configurations of the agents and
their local observations. For example, consider a group of agents navigating:
rotating the state globally results in a permutation of the optimal joint
policy. Existing work on symmetries in single agent reinforcement learning can
only be generalized to the fully centralized setting, because such approaches
rely on the global symmetry in the full state-action spaces, and these can
result in correspondences across agents. To encode such symmetries while still
allowing distributed execution we propose a factorization that decomposes
global symmetries into local transformations. Our proposed factorization allows
for distributing the computation that enforces global symmetries over local
agents and local interactions. We introduce a multi-agent equivariant policy
network based on this factorization. We show empirically on symmetric
multi-agent problems that distributed execution of globally symmetric policies
improves data efficiency compared to non-equivariant baselines.

    

### [[2110.04502] EnsembleNTLDetect: An Intelligent Framework for Electricity Theft Detection in Smart Grid](http://arxiv.org/abs/2110.04502)


  Artificial intelligence-based techniques applied to the electricity
consumption data generated from the smart grid prove to be an effective
solution in reducing Non Technical Loses (NTLs), thereby ensures safety,
reliability, and security of the smart energy systems. However, imbalanced
data, consecutive missing values, large training times, and complex
architectures hinder the real time application of electricity theft detection
models. In this paper, we present EnsembleNTLDetect, a robust and scalable
electricity theft detection framework that employs a set of efficient data
pre-processing techniques and machine learning models to accurately detect
electricity theft by analysing consumers' electricity consumption patterns.
This framework utilises an enhanced Dynamic Time Warping Based Imputation
(eDTWBI) algorithm to impute missing values in the time series data and
leverages the Near-miss undersampling technique to generate balanced data.
Further, stacked autoencoder is introduced for dimensionality reduction and to
improve training efficiency. A Conditional Generative Adversarial Network
(CTGAN) is used to augment the dataset to ensure robust training and a soft
voting ensemble classifier is designed to detect the consumers with aberrant
consumption patterns. Furthermore, experiments were conducted on the real-time
electricity consumption data provided by the State Grid Corporation of China
(SGCC) to validate the reliability and efficiency of EnsembleNTLDetect over the
state-of-the-art electricity theft detection models in terms of various quality
metrics.

    

### [[2110.04503] Multi-Relation Aware Temporal Interaction Network Embedding](http://arxiv.org/abs/2110.04503)


  Temporal interaction networks are formed in many fields, e.g., e-commerce,
online education, and social network service. Temporal interaction network
embedding can effectively mine the information in temporal interaction
networks, which is of great significance to the above fields. Usually, the
occurrence of an interaction affects not only the nodes directly involved in
the interaction (interacting nodes), but also the neighbor nodes of interacting
nodes. However, existing temporal interaction network embedding methods only
use historical interaction relations to mine neighbor nodes, ignoring other
relation types. In this paper, we propose a multi-relation aware temporal
interaction network embedding method (MRATE). Based on historical interactions,
MRATE mines historical interaction relations, common interaction relations, and
interaction sequence similarity relations to obtain the neighbor based
embeddings of interacting nodes. The hierarchical multi-relation aware
aggregation method in MRATE first employs graph attention networks (GATs) to
aggregate the interaction impacts propagated through a same relation type and
then combines the aggregated interaction impacts from multiple relation types
through the self-attention mechanism. Experiments are conducted on three public
temporal interaction network datasets, and the experimental results show the
effectiveness of MRATE.

    

### [[2110.04514] Towards Open-World Feature Extrapolation: An Inductive Graph Learning Approach](http://arxiv.org/abs/2110.04514)


  We target open-world feature extrapolation problem where the feature space of
input data goes through expansion and a model trained on partially observed
features needs to handle new features in test data without further retraining.
The problem is of much significance for dealing with features incrementally
collected from different fields. To this end, we propose a new learning
paradigm with graph representation and learning. Our framework contains two
modules: 1) a backbone network (e.g., feedforward neural nets) as a lower model
takes features as input and outputs predicted labels; 2) a graph neural network
as an upper model learns to extrapolate embeddings for new features via message
passing over a feature-data graph built from observed data. Based on our
framework, we design two training strategies, a self-supervised approach and an
inductive learning approach, to endow the model with extrapolation ability and
alleviate feature-level over-fitting. We also provide theoretical analysis on
the generalization error on test data with new features, which dissects the
impact of training features and algorithms on generalization performance. Our
experiments over several classification datasets and large-scale advertisement
click prediction datasets demonstrate that our model can produce effective
embeddings for unseen features and significantly outperforms baseline methods
that adopt KNN and local aggregation.

    

### [[2110.04519] Pairwise Margin Maximization for Deep Neural Networks](http://arxiv.org/abs/2110.04519)


  The weight decay regularization term is widely used during training to
constrain expressivity, avoid overfitting, and improve generalization.
Historically, this concept was borrowed from the SVM maximum margin principle
and extended to multi-class deep networks. Carefully inspecting this principle
reveals that it is not optimal for multi-class classification in general, and
in particular when using deep neural networks. In this paper, we explain why
this commonly used principle is not optimal and propose a new regularization
scheme, called {\em Pairwise Margin Maximization} (PMM), which measures the
minimal amount of displacement an instance should take until its predicted
classification is switched. In deep neural networks, PMM can be implemented in
the vector space before the network's output layer, i.e., in the deep feature
space, where we add an additional normalization term to avoid convergence to a
trivial solution. We demonstrate empirically a substantial improvement when
training a deep neural network with PMM compared to the standard regularization
terms.

    

### [[2110.04523] An Empirical Study on Compressed Decentralized Stochastic Gradient Algorithms with Overparameterized Models](http://arxiv.org/abs/2110.04523)


  This paper considers decentralized optimization with application to machine
learning on graphs. The growing size of neural network (NN) models has
motivated prior works on decentralized stochastic gradient algorithms to
incorporate communication compression. On the other hand, recent works have
demonstrated the favorable convergence and generalization properties of
overparameterized NNs. In this work, we present an empirical analysis on the
performance of compressed decentralized stochastic gradient (DSG) algorithms
with overparameterized NNs. Through simulations on an MPI network environment,
we observe that the convergence rates of popular compressed DSG algorithms are
robust to the size of NNs. Our findings suggest a gap between theories and
practice of the compressed DSG algorithms in the existing literature.

    

### [[2110.04525] Generating Disentangled Arguments with Prompts: A Simple Event Extraction Framework that Works](http://arxiv.org/abs/2110.04525)


  Event Extraction bridges the gap between text and event signals. Based on the
assumption of trigger-argument dependency, existing approaches have achieved
state-of-the-art performance with expert-designed templates or complicated
decoding constraints. In this paper, for the first time we introduce the
prompt-based learning strategy to the domain of Event Extraction, which
empowers the automatic exploitation of label semantics on both input and output
sides. To validate the effectiveness of the proposed generative method, we
conduct extensive experiments with 11 diverse baselines. Empirical results show
that, in terms of F1 score on Argument Extraction, our simple architecture is
stronger than any other generative counterpart and even competitive with
algorithms that require template engineering. Regarding the measure of recall,
it sets new overall records for both Argument and Trigger Extractions. We
hereby recommend this framework to the community, with the code publicly
available at this https URL.

    

### [[2110.04534] Teaching Robots to Grasp Like Humans: An Interactive Approach](http://arxiv.org/abs/2110.04534)


  This work investigates how the intricate task of grasping may be learned from
humans based on demonstrations and corrections. Due to the complexity of the
task, these demonstrations are often slow and even slightly flawed,
particularly at moments when multiple aspects (i.e., end-effector movement,
orientation, and gripper width) have to be demonstrated at once. Rather than
training a person to provide better demonstrations, non-expert users are
provided with the ability to interactively modify the dynamics of their initial
demonstration through teleoperated corrective feedback. This in turn allows
them to teach motions outside of their own physical capabilities. In the end,
the goal is to obtain a faster but reliable execution of the task. The
presented framework learns the desired movement dynamics based on the current
Cartesian Position with Gaussian Processes (GP), resulting in a reactive,
time-invariant policy. Using GPs also allows online interactive corrections and
active disturbance rejection through epistemic uncertainty minimization. The
experimental evaluation of the framework is carried out on a Franka-Emika
Panda.

    

### [[2110.04541] The Inductive Bias of In-Context Learning: Rethinking Pretraining Example Design](http://arxiv.org/abs/2110.04541)


  Pretraining Neural Language Models (NLMs) over a large corpus involves
chunking the text into training examples, which are contiguous text segments of
sizes processable by the neural architecture. We highlight a bias introduced by
this common practice: we prove that the pretrained NLM can model much stronger
dependencies between text segments that appeared in the same training example,
than it can between text segments that appeared in different training examples.
This intuitive result has a twofold role. First, it formalizes the motivation
behind a broad line of recent successful NLM training heuristics, proposed for
the pretraining and fine-tuning stages, which do not necessarily appear related
at first glance. Second, our result clearly indicates further improvements to
be made in NLM pretraining for the benefit of Natural Language Understanding
tasks. As an example, we propose "kNN-Pretraining": we show that including
semantically related non-neighboring sentences in the same pretraining example
yields improved sentence representations and open domain question answering
abilities. This theoretically motivated degree of freedom for "pretraining
example design" indicates new training schemes for self-improving
representations.

    

### [[2110.04545] Towards Data-Free Domain Generalization](http://arxiv.org/abs/2110.04545)


  In this work, we investigate the unexplored intersection of domain
generalization and data-free learning. In particular, we address the question:
How can knowledge contained in models trained on different source data domains
can be merged into a single model that generalizes well to unseen target
domains, in the absence of source and target domain data? Machine learning
models that can cope with domain shift are essential for for real-world
scenarios with often changing data distributions. Prior domain generalization
methods typically rely on using source domain data, making them unsuitable for
private decentralized data. We define the novel problem of Data-Free Domain
Generalization (DFDG), a practical setting where models trained on the source
domains separately are available instead of the original datasets, and
investigate how to effectively solve the domain generalization problem in that
case. We propose DEKAN, an approach that extracts and fuses domain-specific
knowledge from the available teacher models into a student model robust to
domain shift. Our empirical evaluation demonstrates the effectiveness of our
method which achieves first state-of-the-art results in DFDG by significantly
outperforming ensemble and data-free knowledge distillation baselines.

    

### [[2110.04559] Graph Neural Networks in Real-Time Fraud Detection with Lambda Architecture](http://arxiv.org/abs/2110.04559)


  Transaction checkout fraud detection is an essential risk control components
for E-commerce marketplaces. In order to leverage graph networks to decrease
fraud rate efficiently and guarantee the information flow passed through
neighbors only from the past of the checkouts, we first present a novel
Directed Dynamic Snapshot (DDS) linkage design for graph construction and a
Lambda Neural Networks (LNN) architecture for effective inference with Graph
Neural Networks embeddings. Experiments show that our LNN on DDS graph,
outperforms baseline models significantly and is computational efficient for
real-time fraud detection.

    

### [[2110.04563] Automatic Recognition of Abdominal Organs in Ultrasound Images based on Deep Neural Networks and K-Nearest-Neighbor Classification](http://arxiv.org/abs/2110.04563)


  Abdominal ultrasound imaging has been widely used to assist in the diagnosis
and treatment of various abdominal organs. In order to shorten the examination
time and reduce the cognitive burden on the sonographers, we present a
classification method that combines the deep learning techniques and
k-Nearest-Neighbor (k-NN) classification to automatically recognize various
abdominal organs in the ultrasound images in real time. Fine-tuned deep neural
networks are used in combination with PCA dimension reduction to extract
high-level features from raw ultrasound images, and a k-NN classifier is
employed to predict the abdominal organ in the image. We demonstrate the
effectiveness of our method in the task of ultrasound image classification to
automatically recognize six abdominal organs. A comprehensive comparison of
different configurations is conducted to study the influence of different
feature extractors and classifiers on the classification accuracy. Both
quantitative and qualitative results show that with minimal training effort,
our method can "lazily" recognize the abdominal organs in the ultrasound images
in real time with an accuracy of 96.67%. Our implementation code is publicly
available at: this https URL.

    

### [[2110.04564] Human-Aware Robot Navigation via Reinforcement Learning with Hindsight Experience Replay and Curriculum Learning](http://arxiv.org/abs/2110.04564)


  In recent years, the growing demand for more intelligent service robots is
pushing the development of mobile robot navigation algorithms to allow safe and
efficient operation in a dense crowd. Reinforcement learning (RL) approaches
have shown superior ability in solving sequential decision making problems, and
recent work has explored its potential to learn navigation polices in a
socially compliant manner. However, the expert demonstration data used in
existing methods is usually expensive and difficult to obtain. In this work, we
consider the task of training an RL agent without employing the demonstration
data, to achieve efficient and collision-free navigation in a crowded
environment. To address the sparse reward navigation problem, we propose to
incorporate the hindsight experience replay (HER) and curriculum learning (CL)
techniques with RL to efficiently learn the optimal navigation policy in the
dense crowd. The effectiveness of our method is validated in a simulated
crowd-robot coexisting environment. The results demonstrate that our method can
effectively learn human-aware navigation without requiring additional
demonstration data.

    

### [[2110.04571] Widen The Backdoor To Let More Attackers In](http://arxiv.org/abs/2110.04571)


  As collaborative learning and the outsourcing of data collection become more
common, malicious actors (or agents) which attempt to manipulate the learning
process face an additional obstacle as they compete with each other. In
backdoor attacks, where an adversary attempts to poison a model by introducing
malicious samples into the training data, adversaries have to consider that the
presence of additional backdoor attackers may hamper the success of their own
backdoor. In this paper, we investigate the scenario of a multi-agent backdoor
attack, where multiple non-colluding attackers craft and insert triggered
samples in a shared dataset which is used by a model (a defender) to learn a
task. We discover a clear backfiring phenomenon: increasing the number of
attackers shrinks each attacker's attack success rate (ASR). We then exploit
this phenomenon to minimize the collective ASR of attackers and maximize
defender's robustness accuracy by (i) artificially augmenting the number of
attackers, and (ii) indexing to remove the attacker's sub-dataset from the
model for inference, hence proposing 2 defenses.

    

### [[2110.04572] X-model: Improving Data Efficiency in Deep Learning with A Minimax Model](http://arxiv.org/abs/2110.04572)


  To mitigate the burden of data labeling, we aim at improving data efficiency
for both classification and regression setups in deep learning. However, the
current focus is on classification problems while rare attention has been paid
to deep regression, which usually requires more human effort to labeling.
Further, due to the intrinsic difference between categorical and continuous
label space, the common intuitions for classification, e.g., cluster
assumptions or pseudo labeling strategies, cannot be naturally adapted into
deep regression. To this end, we first delved into the existing data-efficient
methods in deep learning and found that they either encourage invariance to
data stochasticity (e.g., consistency regularization under different
augmentations) or model stochasticity (e.g., difference penalty for predictions
of models with different dropout). To take the power of both worlds, we propose
a novel X-model by simultaneously encouraging the invariance to {data
stochasticity} and {model stochasticity}. Further, the X-model plays a minimax
game between the feature extractor and task-specific heads to further enhance
the invariance to model stochasticity. Extensive experiments verify the
superiority of the X-model among various tasks, from a single-value prediction
task of age estimation to a dense-value prediction task of keypoint
localization, a 2D synthetic, and a 3D realistic dataset, as well as a
multi-category object recognition task.

    

### [[2110.04593] Flattening Sharpness for Dynamic Gradient Projection Memory Benefits Continual Learning](http://arxiv.org/abs/2110.04593)


  The backpropagation networks are notably susceptible to catastrophic
forgetting, where networks tend to forget previously learned skills upon
learning new ones. To address such the 'sensitivity-stability' dilemma, most
previous efforts have been contributed to minimizing the empirical risk with
different parameter regularization terms and episodic memory, but rarely
exploring the usages of the weight loss landscape. In this paper, we
investigate the relationship between the weight loss landscape and
sensitivity-stability in the continual learning scenario, based on which, we
propose a novel method, Flattening Sharpness for Dynamic Gradient Projection
Memory (FS-DGPM). In particular, we introduce a soft weight to represent the
importance of each basis representing past tasks in GPM, which can be
adaptively learned during the learning process, so that less important bases
can be dynamically released to improve the sensitivity of new skill learning.
We further introduce Flattening Sharpness (FS) to reduce the generalization gap
by explicitly regulating the flatness of the weight loss landscape of all seen
tasks. As demonstrated empirically, our proposed method consistently
outperforms baselines with the superior ability to learn new skills while
alleviating forgetting effectively.

    

### [[2110.04597] A Proximal Algorithm for Sampling from Non-smooth Potentials](http://arxiv.org/abs/2110.04597)


  Markov chain Monte Carlo (MCMC) is an effective and dominant method to sample
from high-dimensional complex distributions. Yet, most existing MCMC methods
are only applicable to settings with smooth potentials (log-densities). In this
work, we examine sampling problems with non-smooth potentials. We propose a
novel MCMC algorithm for sampling from non-smooth potentials. We provide a
non-asymptotical analysis of our algorithm and establish a polynomial-time
complexity $\tilde {\cal O}(d\varepsilon^{-1})$ to obtain $\varepsilon$ total
variation distance to the target density, better than all existing results
under the same assumptions. Our method is based on the proximal bundle method
and an alternating sampling framework. This framework requires the so-called
restricted Gaussian oracle, which can be viewed as a sampling counterpart of
the proximal mapping in convex optimization. One key contribution of this work
is a fast algorithm that realizes the restricted Gaussian oracle for any convex
non-smooth potential with bounded Lipschitz constant.

    

### [[2110.04598] Self-explaining Neural Network with Plausible Explanations](http://arxiv.org/abs/2110.04598)


  Explaining the predictions of complex deep learning models, often referred to
as black boxes, is critical in high-stakes domains like healthcare. However,
post-hoc model explanations often are not understandable by clinicians and are
difficult to integrate into clinical workflow. Further, while most explainable
models use individual clinical variables as units of explanation, human
understanding often rely on higher-level concepts or feature representations.
In this paper, we propose a novel, self-explaining neural network for
longitudinal in-hospital mortality prediction using domain-knowledge driven
Sequential Organ Failure Assessment (SOFA) organ-specific scores as the atomic
units of explanation. We also design a novel procedure to quantitatively
validate the model explanations against gold standard discharge diagnosis
information of patients. Our results provide interesting insights into how each
of the SOFA organ scores contribute to mortality at different timesteps within
longitudinal patient trajectory.

    

### [[2110.04599] Embed Everything: A Method for Efficiently Co-Embedding Multi-Modal Spaces](http://arxiv.org/abs/2110.04599)


  Any general artificial intelligence system must be able to interpret, operate
on, and produce data in a multi-modal latent space that can represent audio,
imagery, text, and more. In the last decade, deep neural networks have seen
remarkable success in unimodal data distributions, while transfer learning
techniques have seen a massive expansion of model reuse across related domains.
However, training multi-modal networks from scratch remains expensive and
illusive, while heterogeneous transfer learning (HTL) techniques remain
relatively underdeveloped. In this paper, we propose a novel and cost-effective
HTL strategy for co-embedding multi-modal spaces. Our method avoids cost
inefficiencies by preprocessing embeddings using pretrained models for all
components, without passing gradients through these models. We prove the use of
this system in a joint image-audio embedding task. Our method has wide-reaching
applications, as successfully bridging the gap between different latent spaces
could provide a framework for the promised "universal" embedding.

    

### [[2110.04600] A Review of Physics-based Machine Learning in Civil Engineering](http://arxiv.org/abs/2110.04600)


  The recent development of machine learning (ML) and Deep Learning (DL)
increases the opportunities in all the sectors. ML is a significant tool that
can be applied across many disciplines, but its direct application to civil
engineering problems can be challenging. ML for civil engineering applications
that are simulated in the lab often fail in real-world tests. This is usually
attributed to a data mismatch between the data used to train and test the ML
model and the data it encounters in the real world, a phenomenon known as data
shift. However, a physics-based ML model integrates data, partial differential
equations (PDEs), and mathematical models to solve data shift problems.
Physics-based ML models are trained to solve supervised learning tasks while
respecting any given laws of physics described by general nonlinear equations.
Physics-based ML, which takes center stage across many science disciplines,
plays an important role in fluid dynamics, quantum mechanics, computational
resources, and data storage. This paper reviews the history of physics-based ML
and its application in civil engineering.

    

### [[2110.04603] Learning Single/Multi-Attribute of Object with Symmetry and Group](http://arxiv.org/abs/2110.04603)


  Attributes and objects can compose diverse compositions. To model the
compositional nature of these concepts, it is a good choice to learn them as
transformations, e.g., coupling and decoupling. However, complex
transformations need to satisfy specific principles to guarantee rationality.
Here, we first propose a previously ignored principle of attribute-object
transformation: Symmetry. For example, coupling peeled-apple with attribute
peeled should result in peeled-apple, and decoupling peeled from apple should
still output apple. Incorporating the symmetry, we propose a transformation
framework inspired by group theory, i.e., SymNet. It consists of two modules:
Coupling Network and Decoupling Network. We adopt deep neural networks to
implement SymNet and train it in an end-to-end paradigm with the group axioms
and symmetry as objectives. Then, we propose a Relative Moving Distance (RMD)
based method to utilize the attribute change instead of the attribute pattern
itself to classify attributes. Besides the compositions of single-attribute and
object, our RMD is also suitable for complex compositions of multiple
attributes and objects when incorporating attribute correlations. SymNet can be
utilized for attribute learning, compositional zero-shot learning and
outperforms the state-of-the-art on four widely-used benchmarks. Code is at
this https URL.

    

### [[2110.04604] Learning MRI Artifact Removal With Unpaired Data](http://arxiv.org/abs/2110.04604)


  Retrospective artifact correction (RAC) improves image quality post
acquisition and enhances image usability. Recent machine learning driven
techniques for RAC are predominantly based on supervised learning and therefore
practical utility can be limited as data with paired artifact-free and
artifact-corrupted images are typically insufficient or even non-existent. Here
we show that unwanted image artifacts can be disentangled and removed from an
image via an RAC neural network learned with unpaired data. This implies that
our method does not require matching artifact-corrupted data to be either
collected via acquisition or generated via simulation. Experimental results
demonstrate that our method is remarkably effective in removing artifacts and
retaining anatomical details in images with different contrasts.

    

### [[2110.04612] Personalized Automatic Speech Recognition Trained on Small Disordered Speech Datasets](http://arxiv.org/abs/2110.04612)


  This study investigates the performance of personalized automatic speech
recognition (ASR) for recognizing disordered speech using small amounts of
per-speaker adaptation data. We trained personalized models for 195 individuals
with different types and severities of speech impairment with training sets
ranging in size from <1 minute to 18-20 minutes of speech data. Word error rate
(WER) thresholds were selected to determine Success Percentage (the percentage
of personalized models reaching the target WER) in different application
scenarios. For the home automation scenario, 79% of speakers reached the target
WER with 18-20 minutes of speech; but even with only 3-4 minutes of speech, 63%
of speakers reached the target WER. Further evaluation found similar
improvement on test sets with conversational and out-of-domain, unprompted
phrases. Our results demonstrate that with only a few minutes of recordings,
individuals with disordered speech could benefit from personalized ASR.

    

### [[2110.04616] Discriminative Multimodal Learning via Conditional Priors in Generative Models](http://arxiv.org/abs/2110.04616)


  Deep generative models with latent variables have been used lately to learn
joint representations and generative processes from multi-modal data. These two
learning mechanisms can, however, conflict with each other and representations
can fail to embed information on the data modalities. This research studies the
realistic scenario in which all modalities and class labels are available for
model training, but where some modalities and labels required for downstream
tasks are missing. We show, in this scenario, that the variational lower bound
limits mutual information between joint representations and missing modalities.
We, to counteract these problems, introduce a novel conditional multi-modal
discriminative model that uses an informative prior distribution and optimizes
a likelihood-free objective function that maximizes mutual information between
joint representations and missing modalities. Extensive experimentation shows
the benefits of the model we propose, the empirical results showing that our
model achieves state-of-the-art results in representative problems such as
downstream classification, acoustic inversion and annotation generation.

    

### [[2110.04621] Universal Paralinguistic Speech Representations Using Self-Supervised Conformers](http://arxiv.org/abs/2110.04621)


  Many speech applications require understanding aspects beyond the words being
spoken, such as recognizing emotion, detecting whether the speaker is wearing a
mask, or distinguishing real from synthetic speech. In this work, we introduce
a new state-of-the-art paralinguistic representation derived from large-scale,
fully self-supervised training of a 600M+ parameter Conformer-based
architecture. We benchmark on a diverse set of speech tasks and demonstrate
that simple linear classifiers trained on top of our time-averaged
representation outperform nearly all previous results, in some cases by large
margins. Our analyses of context-window size demonstrate that, surprisingly, 2
second context-windows achieve 98% the performance of the Conformers that use
the full long-term context. Furthermore, while the best per-task
representations are extracted internally in the network, stable performance
across several layers allows a single universal representation to reach near
optimal performance on all tasks.

    

### [[2110.04622] Does Preprocessing Help Training Over-parameterized Neural Networks?](http://arxiv.org/abs/2110.04622)


  Deep neural networks have achieved impressive performance in many areas.
Designing a fast and provable method for training neural networks is a
fundamental question in machine learning.
The classical training method requires paying $\Omega(mnd)$ cost for both
forward computation and backward computation, where $m$ is the width of the
neural network, and we are given $n$ training points in $d$-dimensional space.
In this paper, we propose two novel preprocessing ideas to bypass this
$\Omega(mnd)$ barrier:
$\bullet$ First, by preprocessing the initial weights of the neural networks,
we can train the neural network in $\widetilde{O}(m^{1-\Theta(1/d)} n d)$ cost
per iteration.
$\bullet$ Second, by preprocessing the input data points, we can train the
neural network in $\widetilde{O} (m^{4/5} nd )$ cost per iteration.
From the technical perspective, our result is a sophisticated combination of
tools in different fields, greedy-type convergence analysis in optimization,
sparsity observation in practical work, high-dimensional geometric search in
data structure, concentration and anti-concentration in probability. Our
results also provide theoretical insights for a large number of previously
established fast training methods.
In addition, our classical algorithm can be generalized to the Quantum
computation model. Interestingly, we can get a similar sublinear cost per
iteration but avoid preprocessing initial weights or input data points.

    

### [[2110.04624] Iterative Refinement Graph Neural Network for Antibody Sequence-Structure Co-design](http://arxiv.org/abs/2110.04624)


  Antibodies are versatile proteins that bind to pathogens like viruses and
stimulate the adaptive immune system. The specificity of antibody binding is
determined by complementarity-determining regions (CDRs) at the tips of these
Y-shaped proteins. In this paper, we propose a generative model to
automatically design the CDRs of antibodies with enhanced binding specificity
or neutralization capabilities. Previous generative approaches formulate
protein design as a structure-conditioned sequence generation task, assuming
the desired 3D structure is given a priori. In contrast, we propose to
co-design the sequence and 3D structure of CDRs as graphs. Our model unravels a
sequence autoregressively while iteratively refining its predicted global
structure. The inferred structure in turn guides subsequent residue choices.
For efficiency, we model the conditional dependence between residues inside and
outside of a CDR in a coarse-grained manner. Our method achieves superior
log-likelihood on the test set and outperforms previous baselines in designing
antibodies capable of neutralizing the SARS-CoV-2 virus.

    

### [[2110.04627] Vector-quantized Image Modeling with Improved VQGAN](http://arxiv.org/abs/2110.04627)


  Pretraining language models with next-token prediction on massive text
corpora has delivered phenomenal zero-shot, few-shot, transfer learning and
multi-tasking capabilities on both generative and discriminative language
tasks. Motivated by this success, we explore a Vector-quantized Image Modeling
(VIM) approach that involves pretraining a Transformer to predict rasterized
image tokens autoregressively. The discrete image tokens are encoded from a
learned Vision-Transformer-based VQGAN (ViT-VQGAN). We first propose multiple
improvements over vanilla VQGAN from architecture to codebook learning,
yielding better efficiency and reconstruction fidelity. The improved ViT-VQGAN
further improves vector-quantized image modeling tasks, including
unconditional, class-conditioned image generation and unsupervised
representation learning. When trained on ImageNet at 256x256 resolution, we
achieve Inception Score (IS) of 175.1 and Fr'echet Inception Distance (FID) of
4.17, a dramatic improvement over the vanilla VQGAN, which obtains 70.6 and
17.04 for IS and FID, respectively. Based on ViT-VQGAN and unsupervised
pretraining, we further evaluate the pretrained Transformer by averaging
intermediate features, similar to Image GPT (iGPT). This ImageNet-pretrained
VIM-L significantly beats iGPT-L on linear-probe accuracy from 60.3% to 72.2%
for a similar model size. ViM-L also outperforms iGPT-XL which is trained with
extra web image data and larger model size.

    

### [[2110.04629] Evaluating Predictive Distributions: Does Bayesian Deep Learning Work?](http://arxiv.org/abs/2110.04629)


  Posterior predictive distributions quantify uncertainties ignored by point
estimates. This paper introduces \textit{The Neural Testbed}, which provides
tools for the systematic evaluation of agents that generate such predictions.
Crucially, these tools assess not only the quality of marginal predictions per
input, but also joint predictions given many inputs. Joint distributions are
often critical for useful uncertainty quantification, but they have been
largely overlooked by the Bayesian deep learning community. We benchmark
several approaches to uncertainty estimation using a neural-network-based data
generating process. Our results reveal the importance of evaluation beyond
marginal predictions. Further, they reconcile sources of confusion in the
field, such as why Bayesian deep learning approaches that generate accurate
marginal predictions perform poorly in sequential decision tasks, how
incorporating priors can be helpful, and what roles epistemic versus aleatoric
uncertainty play when evaluating performance. We also present experiments on
real-world challenge datasets, which show a high correlation with testbed
results, and that the importance of evaluating joint predictive distributions
carries over to real data. As part of this effort, we opensource The Neural
Testbed, including all implementations from this paper.

    

### [[2110.04632] DenseNet approach to segmentation and classification of dermatoscopic skin lesions images](http://arxiv.org/abs/2110.04632)


  At present, cancer is one of the most important health issues in the world.
Because early detection and appropriate treatment in cancer are very effective
in the recovery and survival of patients, image processing as a diagnostic tool
can help doctors to diagnose in the first recognition of cancer. One of the
most important steps in diagnosing a skin lesion is to automatically detect the
border of the skin image because the accuracy of the next steps depends on it.
If these subtleties are identified, they can have a great impact on the
diagnosis of the disease. Therefore, there is a good opportunity to develop
more accurate algorithms to analyze such images. This paper proposes an
improved method for segmentation and classification for skin lesions using two
architectures, the U-Net for image segmentation and the DenseNet121 for image
classification which have excellent accuracy. We tested the segmentation
architecture of our model on the ISIC-2018 dataset and the classification on
the HAM10000 dataset. Our results show that the combination of U-Net and
DenseNet121 architectures provides acceptable results in dermatoscopic image
analysis compared to previous research. Another classification examined in this
study is cancerous and non-cancerous samples. In this classification, cancerous
and non-cancerous samples were detected in DenseNet121 network with 79.49% and
93.11% accuracy respectively.

    

### [[2110.04638] An Independent Learning Algorithm for a Class of Symmetric Stochastic Games](http://arxiv.org/abs/2110.04638)


  In multi-agent reinforcement learning, independent learners are those that do
not access the action selections of other learning agents in the system. This
paper investigates the feasibility of using independent learners to find
approximate equilibrium policies in non-episodic, discounted stochastic games.
We define a property, here called the $\epsilon$-revision paths property, and
prove that a class of games exhibiting symmetry among the players has this
property for any $\epsilon \geq 0$. Building on this result, we present an
independent learning algorithm that comes with high probability guarantees of
approximate equilibrium in this class of games. This guarantee is made assuming
symmetry alone, without additional assumptions such as a zero sum, team, or
potential game structure.

    

### [[2110.04639] Multi-task learning on the edge: cost-efficiency and theoretical optimality](http://arxiv.org/abs/2110.04639)


  This article proposes a distributed multi-task learning (MTL) algorithm based
on supervised principal component analysis (SPCA) which is: (i) theoretically
optimal for Gaussian mixtures, (ii) computationally cheap and scalable.
Supporting experiments on synthetic and real benchmark data demonstrate that
significant energy gains can be obtained with no performance loss.

    

### [[2110.04644] On the Relation between Syntactic Divergence and Zero-Shot Performance](http://arxiv.org/abs/2110.04644)


  We explore the link between the extent to which syntactic relations are
preserved in translation and the ease of correctly constructing a parse tree in
a zero-shot setting. While previous work suggests such a relation, it tends to
focus on the macro level and not on the level of individual edges-a gap we aim
to address. As a test case, we take the transfer of Universal Dependencies (UD)
parsing from English to a diverse set of languages and conduct two sets of
experiments. In one, we analyze zero-shot performance based on the extent to
which English source edges are preserved in translation. In another, we apply
three linguistically motivated transformations to UD, creating more
cross-lingually stable versions of it, and assess their zero-shot parsability.
In order to compare parsing performance across different schemes, we perform
extrinsic evaluation on the downstream task of cross-lingual relation
extraction (RE) using a subset of a popular English RE benchmark translated to
Russian and Korean. In both sets of experiments, our results suggest a strong
relation between cross-lingual stability and zero-shot parsing performance.

    

### [[2110.04645] Breaking the Sample Complexity Barrier to Regret-Optimal Model-Free Reinforcement Learning](http://arxiv.org/abs/2110.04645)


  Achieving sample efficiency in online episodic reinforcement learning (RL)
requires optimally balancing exploration and exploitation. When it comes to a
finite-horizon episodic Markov decision process with $S$ states, $A$ actions
and horizon length $H$, substantial progress has been achieved towards
characterizing the minimax-optimal regret, which scales on the order of
$\sqrt{H^2SAT}$ (modulo log factors) with $T$ the total number of samples.
While several competing solution paradigms have been proposed to minimize
regret, they are either memory-inefficient, or fall short of optimality unless
the sample size exceeds an enormous threshold (e.g., $S^6A^4
\,\mathrm{poly}(H)$ for existing model-free methods).
To overcome such a large sample size barrier to efficient RL, we design a
novel model-free algorithm, with space complexity $O(SAH)$, that achieves
near-optimal regret as soon as the sample size exceeds the order of
$SA\,\mathrm{poly}(H)$. In terms of this sample size requirement (also referred
to the initial burn-in cost),
our method improves -- by at least a factor of $S^5A^3$ -- upon any prior
memory-efficient algorithm that is asymptotically regret-optimal. Leveraging
the recently introduced variance reduction strategy (also called {\em
reference-advantage decomposition}), the proposed algorithm employs an {\em
early-settled} reference update rule, with the aid of two Q-learning sequences
with upper and lower confidence bounds. The design principle of our
early-settled variance reduction method might be of independent interest to
other RL settings that involve intricate exploration-exploitation trade-offs.

    

### [[2110.04647] Learning to Follow Language Instructions with Compositional Policies](http://arxiv.org/abs/2110.04647)


  We propose a framework that learns to execute natural language instructions
in an environment consisting of goal-reaching tasks that share components of
their task descriptions. Our approach leverages the compositionality of both
value functions and language, with the aim of reducing the sample complexity of
learning novel tasks. First, we train a reinforcement learning agent to learn
value functions that can be subsequently composed through a Boolean algebra to
solve novel tasks. Second, we fine-tune a seq2seq model pretrained on web-scale
corpora to map language to logical expressions that specify the required value
function compositions. Evaluating our agent in the BabyAI domain, we observe a
decrease of 86% in the number of training steps needed to learn a second task
after mastering a single task. Results from ablation studies further indicate
that it is the combination of compositional value functions and language
representations that allows the agent to quickly generalize to new tasks.

    

### [[2110.04652] Representation Learning for Online and Offline RL in Low-rank MDPs](http://arxiv.org/abs/2110.04652)


  This work studies the question of Representation Learning in RL: how can we
learn a compact low-dimensional representation such that on top of the
representation we can perform RL procedures such as exploration and
exploitation, in a sample efficient manner. We focus on the low-rank Markov
Decision Processes (MDPs) where the transition dynamics correspond to a
low-rank transition matrix. Unlike prior works that assume the representation
is known (e.g., linear MDPs), here we need to learn the representation for the
low-rank MDP. We study both the online RL and offline RL settings. For the
online setting, operating with the same computational oracles used in FLAMBE
(Agarwal this http URL), the state-of-art algorithm for learning representations in
low-rank MDPs, we propose an algorithm REP-UCB Upper Confidence Bound driven
Representation learning for RL), which significantly improves the sample
complexity from $\widetilde{O}( A^9 d^7 / (\epsilon^{10} (1-\gamma)^{22}))$ for
FLAMBE to $\widetilde{O}( A^4 d^4 / (\epsilon^2 (1-\gamma)^{3}) )$ with $d$
being the rank of the transition matrix (or dimension of the ground truth
representation), $A$ being the number of actions, and $\gamma$ being the
discounted factor. Notably, REP-UCB is simpler than FLAMBE, as it directly
balances the interplay between representation learning, exploration, and
exploitation, while FLAMBE is an explore-then-commit style approach and has to
perform reward-free exploration step-by-step forward in time. For the offline
RL setting, we develop an algorithm that leverages pessimism to learn under a
partial coverage condition: our algorithm is able to compete against any policy
as long as it is covered by the offline distribution.

    

### [[2110.04653] Topological Data Analysis (TDA) Techniques Enhance Hand Pose Classification from ECoG Neural Recordings](http://arxiv.org/abs/2110.04653)


  Electrocorticogram (ECoG) well characterizes hand movement intentions and
gestures. In the present work we aim to investigate the possibility to enhance
hand pose classification, in a Rock-Paper-Scissor - and Rest - task, by
introducing topological descriptors of time series data. We hypothesized that
an innovative approach based on topological data analysis can extract hidden
information that are not detectable with standard Brain Computer Interface
(BCI)techniques. To investigate this hypothesis, we integrate topological
features together with power band features and feed them to several standard
classifiers, e.g. Random Forest,Gradient Boosting. Model selection is thus
completed after a meticulous phase of bayesian hyperparameter optimization.
With our method, we observed robust results in terms of ac-curacy for a
four-labels classification problem, with limited available data. Through
feature importance investigation, we conclude that topological descriptors are
able to extract useful discriminative information and provide novel
insights.Since our data are restricted to single-patient recordings,
generalization might be limited. Nevertheless, our method can be extended and
applied to a wide range of neurophysiological recordings and it might be an
intriguing point of departure for future studies.

    

### [[2110.04654] Complex Network-Based Approach for Feature Extraction and Classification of Musical Genres](http://arxiv.org/abs/2110.04654)


  Musical genre's classification has been a relevant research topic. The
association between music and genres is fundamental for the media industry,
which manages musical recommendation systems, and for music streaming services,
which may appear classified by genres. In this context, this work presents a
feature extraction method for the automatic classification of musical genres,
based on complex networks and their topological measurements. The proposed
method initially converts the musics into sequences of musical notes and then
maps the sequences as complex networks. Topological measurements are extracted
to characterize the network topology, which composes a feature vector that
applies to the classification of musical genres. The method was evaluated in
the classification of 10 musical genres by adopting the GTZAN dataset and 8
musical genres by adopting the FMA dataset. The results were compared with
methods in the literature. The proposed method outperformed all compared
methods by presenting high accuracy and low standard deviation, showing its
suitability for the musical genre's classification, which contributes to the
media industry in the automatic classification with assertiveness and
robustness. The proposed method is implemented in an open source in the Python
language and freely available at this https URL.

    

### [[2110.04656] Streaming on-device detection of device directed speech from voice and touch-based invocation](http://arxiv.org/abs/2110.04656)


  When interacting with smart devices such as mobile phones or wearables, the
user typically invokes a virtual assistant (VA) by saying a keyword or by
pressing a button on the device. However, in many cases, the VA can
accidentally be invoked by the keyword-like speech or accidental button press,
which may have implications on user experience and privacy. To this end, we
propose an acoustic false-trigger-mitigation (FTM) approach for on-device
device-directed speech detection that simultaneously handles the voice-trigger
and touch-based invocation. To facilitate the model deployment on-device, we
introduce a new streaming decision layer, derived using the notion of temporal
convolutional networks (TCN) [1], known for their computational efficiency. To
the best of our knowledge, this is the first approach that can detect
device-directed speech from more than one invocation type in a streaming
fashion. We compare this approach with streaming alternatives based on vanilla
Average layer, and canonical LSTMs, and show: (i) that all the models show only
a small degradation in accuracy compared with the invocation-specific models,
and (ii) that the newly introduced streaming TCN consistently performs better
or comparable with the alternatives, while mitigating device undirected speech
faster in time, and with (relative) reduction in runtime peak-memory over the
LSTM-based approach of 33% vs. 7%, when compared to a non-streaming
counterpart.

    

### [[2110.04659] Exploring constraints on CycleGAN-based CBCT enhancement for adaptive radiotherapy](http://arxiv.org/abs/2110.04659)


  Research exploring CycleGAN-based synthetic image generation has recently
accelerated in the medical community, as it is able to leverage unpaired
datasets effectively. However, clinical acceptance of these synthetic images
pose a significant challenge as they are subject to strict evaluation
protocols. A commonly established drawback of the CycleGAN, the introduction of
artifacts in generated images is unforgivable in the case of medical images. In
an attempt to alleviate this drawback, we explore different constraints of the
CycleGAN along with investigation of adaptive control of these constraints. The
benefits of imposing additional constraints on the CycleGAN, in the form of
structure retaining losses is also explored. A generalized frequency loss
inspired by \cite{jiang2020focal} that preserves content in the frequency
domain between source and target is investigated and compared with existing
losses such as the MIND loss arXiv:1809.04536. Synthetic images generated from
our methods are quantitatively and qualitatively investigated and outperform
the baseline CycleGAN and other approaches. Furthermore, no observable
artifacts or loss in image quality is found, which is critical for acceptance
of these synthetic images. The synthetic medical images thus generated are also
evaluated using domain-specific evaluation and using segmentation as a
downstream task, in order to clearly highlight their applicability to clinical
workflows.

    

### [[2110.04662] Cognitively Inspired Learning of Incremental Drifting Concepts](http://arxiv.org/abs/2110.04662)


  Humans continually expand their learned knowledge to new domains and learn
new concepts without any interference with past learned experiences. In
contrast, machine learning models perform poorly in a continual learning
setting, where input data distribution changes over time. Inspired by the
nervous system learning mechanisms, we develop a computational model that
enables a deep neural network to learn new concepts and expand its learned
knowledge to new domains incrementally in a continual learning setting. We rely
on the Parallel Distributed Processing theory to encode abstract concepts in an
embedding space in terms of a multimodal distribution. This embedding space is
modeled by internal data representations in a hidden network layer. We also
leverage the Complementary Learning Systems theory to equip the model with a
memory mechanism to overcome catastrophic forgetting through implementing
pseudo-rehearsal. Our model can generate pseudo-data points for experience
replay and accumulate new experiences to past learned experiences without
causing cross-task interference.

    

### [[2110.04669] Leveraging Experience in Lazy Search](http://arxiv.org/abs/2110.04669)


  Lazy graph search algorithms are efficient at solving motion planning
problems where edge evaluation is the computational bottleneck. These
algorithms work by lazily computing the shortest potentially feasible path,
evaluating edges along that path, and repeating until a feasible path is found.
The order in which edges are selected is critical to minimizing the total
number of edge evaluations: a good edge selector chooses edges that are not
only likely to be invalid, but also eliminates future paths from consideration.
We wish to learn such a selector by leveraging prior experience. We formulate
this problem as a Markov Decision Process (MDP) on the state of the search
problem. While solving this large MDP is generally intractable, we show that we
can compute oracular selectors that can solve the MDP during training. With
access to such oracles, we use imitation learning to find effective policies.
If new search problems are sufficiently similar to problems solved during
training, the learned policy will choose a good edge evaluation ordering and
solve the motion planning problem quickly. We evaluate our algorithms on a wide
range of 2D and 7D problems and show that the learned selector outperforms
baseline commonly used heuristics. We further provide a novel theoretical
analysis of lazy search in a Bayesian framework as well as regret guarantees on
our imitation learning based approach to motion planning.

    

### [[2110.04683] Mixture Model Auto-Encoders: Deep Clustering through Dictionary Learning](http://arxiv.org/abs/2110.04683)


  State-of-the-art approaches for clustering high-dimensional data utilize deep
auto-encoder architectures. Many of these networks require a large number of
parameters and suffer from a lack of interpretability, due to the black-box
nature of the auto-encoders. We introduce Mixture Model Auto-Encoders
(MixMate), a novel architecture that clusters data by performing inference on a
generative model. Derived from the perspective of sparse dictionary learning
and mixture models, MixMate comprises several auto-encoders, each tasked with
reconstructing data in a distinct cluster, while enforcing sparsity in the
latent space. Through experiments on various image datasets, we show that
MixMate achieves competitive performance compared to state-of-the-art deep
clustering algorithms, while using orders of magnitude fewer parameters.

    

### [[2110.04686] Braxlines: Fast and Interactive Toolkit for RL-driven Behavior Engineering beyond Reward Maximization](http://arxiv.org/abs/2110.04686)


  The goal of continuous control is to synthesize desired behaviors. In
reinforcement learning (RL)-driven approaches, this is often accomplished
through careful task reward engineering for efficient exploration and running
an off-the-shelf RL algorithm. While reward maximization is at the core of RL,
reward engineering is not the only -- sometimes nor the easiest -- way for
specifying complex behaviors. In this paper, we introduce \braxlines, a toolkit
for fast and interactive RL-driven behavior generation beyond simple reward
maximization that includes Composer, a programmatic API for generating
continuous control environments, and set of stable and well-tested baselines
for two families of algorithms -- mutual information maximization (MiMax) and
divergence minimization (DMin) -- supporting unsupervised skill learning and
distribution sketching as other modes of behavior specification. In addition,
we discuss how to standardize metrics for evaluating these algorithms, which
can no longer rely on simple reward maximization. Our implementations build on
a hardware-accelerated Brax simulator in Jax with minimal modifications,
enabling behavior synthesis within minutes of training. We hope Braxlines can
serve as an interactive toolkit for rapid creation and testing of environments
and behaviors, empowering explosions of future benchmark designs and new modes
of RL-driven behavior generation and their algorithmic research.

    

### [[2110.04689] Surrogate-Assisted Reference Vector Adaptation to Various Pareto Front Shapes for Many-Objective Bayesian Optimization](http://arxiv.org/abs/2110.04689)


  We propose a surrogate-assisted reference vector adaptation (SRVA) method to
solve expensive multi- and many-objective optimization problems with various
Pareto front shapes. SRVA is coupled with a multi-objective Bayesian
optimization (MBO) algorithm using reference vectors for scalarization of
objective functions. The Kriging surrogate models for MBO is used to estimate
the Pareto front shape and generate adaptive reference vectors uniformly
distributed on the estimated Pareto front. We combine SRVA with expected
improvement of penalty-based boundary intersection as an infill criterion for
MBO. The proposed algorithm is compared with two other MBO algorithms by
applying them to benchmark problems with various Pareto front shapes.
Experimental results show that the proposed algorithm outperforms the other two
in the problems whose objective functions are reasonably approximated by the
Kriging models. SRVA improves diversity of non-dominated solutions for these
problems with continuous, discontinuous, and degenerated Pareto fronts.
Besides, the proposed algorithm obtains much better solutions from early stages
of optimization especially in many-objective problems.

    

### [[1806.07788] Random Feature Stein Discrepancies](http://arxiv.org/abs/1806.07788)


  Computable Stein discrepancies have been deployed for a variety of
applications, ranging from sampler selection in posterior inference to
approximate Bayesian inference to goodness-of-fit testing. Existing
convergence-determining Stein discrepancies admit strong theoretical guarantees
but suffer from a computational cost that grows quadratically in the sample
size. While linear-time Stein discrepancies have been proposed for
goodness-of-fit testing, they exhibit avoidable degradations in testing power
-- even when power is explicitly optimized. To address these shortcomings, we
introduce feature Stein discrepancies ($\Phi$SDs), a new family of quality
measures that can be cheaply approximated using importance sampling. We show
how to construct $\Phi$SDs that provably determine the convergence of a sample
to its target and develop high-accuracy approximations -- random $\Phi$SDs
(R$\Phi$SDs) -- which are computable in near-linear time. In our experiments
with sampler selection for approximate posterior inference and goodness-of-fit
testing, R$\Phi$SDs perform as well or better than quadratic-time KSDs while
being orders of magnitude faster to compute.

    

### [[1811.00189] Unauthorized AI cannot Recognize Me: Reversible Adversarial Example](http://arxiv.org/abs/1811.00189)


  In this study, we propose a new methodology to control how user's data is
recognized and used by AI via exploiting the properties of adversarial
examples. For this purpose, we propose reversible adversarial example (RAE), a
new type of adversarial example. A remarkable feature of RAE is that the image
can be correctly recognized and used by the AI model specified by the user
because the authorized AI can recover the original image from the RAE exactly
by eliminating adversarial perturbation. On the other hand, other unauthorized
AI models cannot recognize it correctly because it functions as an adversarial
example. Moreover, RAE can be considered as one type of encryption to computer
vision since reversibility guarantees the decryption. To realize RAE, we
combine three technologies, adversarial example, reversible data hiding for
exact recovery of adversarial perturbation, and encryption for selective
control of AIs who can remove adversarial perturbation. Experimental results
show that the proposed method can achieve comparable attack ability with the
corresponding adversarial attack method and similar visual quality with the
original image, including white-box attacks and black-box attacks.

    

### [[1811.10264] PNS: Population-Guided Novelty Search for Reinforcement Learning in Hard Exploration Environments](http://arxiv.org/abs/1811.10264)


  Reinforcement Learning (RL) has made remarkable achievements, but it still
suffers from inadequate exploration strategies, sparse reward signals, and
deceptive reward functions. To alleviate these problems, a Population-guided
Novelty Search (PNS) parallel learning method is proposed in this paper. In
PNS, the population is divided into multiple sub-populations, each of which has
one chief agent and several exploring agents. The chief agent evaluates the
policies learned by exploring agents and shares the optimal policy with all
sub-populations. The exploring agents learn their policies in collaboration
with the guidance of the optimal policy and, simultaneously, upload their
policies to the chief agent. To balance exploration and exploitation, the
Novelty Search (NS) is employed in every chief agent to encourage policies with
high novelty while maximizing per-episode performance. We apply PNS to the twin
delayed deep deterministic (TD3) policy gradient algorithm. The effectiveness
of PNS to promote exploration and improve performance in continuous control
domains is demonstrated in the experimental section. Notably, PNS-TD3 achieves
rewards that far exceed the SOTA methods in environments with sparse or delayed
reward signals. We also demonstrate that PNS enables robotic agents to learn
control policies directly from pixels for sparse-reward manipulation in both
simulated and real-world settings.

    

### [[1811.11925] Stochastic Top-$K$ Subset Bandits with Linear Space and Non-Linear Feedback](http://arxiv.org/abs/1811.11925)


  Many real-world problems like Social Influence Maximization face the dilemma
of choosing the best $K$ out of $N$ options at a given time instant. This setup
can be modeled as a combinatorial bandit which chooses $K$ out of $N$ arms at
each time, with an aim to achieve an efficient trade-off between exploration
and exploitation. This is the first work for combinatorial bandits where the
feedback received can be a non-linear function of the chosen $K$ arms. The
direct use of multi-armed bandit requires choosing among $N$-choose-$K$ options
making the state space large. In this paper, we present a novel algorithm which
is computationally efficient and the storage is linear in $N$. The proposed
algorithm is a divide-and-conquer based strategy, that we call CMAB-SM.
Further, the proposed algorithm achieves a \textit{regret bound} of $\tilde
O(K^{\frac{1}{2}}N^{\frac{1}{3}}T^{\frac{2}{3}})$ for a time horizon $T$, which
is \textit{sub-linear} in all parameters $T$, $N$, and $K$. %When applied to
the problem of Social Influence Maximization, the performance of the proposed
algorithm surpasses the UCB algorithm and some more sophisticated
domain-specific methods.

    

### [[1901.08562] Sample Complexity of Estimating the Policy Gradient for Nearly Deterministic Dynamical Systems](http://arxiv.org/abs/1901.08562)


  Reinforcement learning is a promising approach to learning robotics
controllers. It has recently been shown that algorithms based on
finite-difference estimates of the policy gradient are competitive with
algorithms based on the policy gradient theorem. We propose a theoretical
framework for understanding this phenomenon. Our key insight is that many
dynamical systems (especially those of interest in robotics control tasks) are
nearly deterministic -- i.e., they can be modeled as a deterministic system
with a small stochastic perturbation. We show that for such systems,
finite-difference estimates of the policy gradient can have substantially lower
variance than estimates based on the policy gradient theorem. Finally, we
empirically evaluate our insights in an experiment on the inverted pendulum.

    

### [[1901.08568] Algorithms for Fairness in Sequential Decision Making](http://arxiv.org/abs/1901.08568)


  It has recently been shown that if feedback effects of decisions are ignored,
then imposing fairness constraints such as demographic parity or equality of
opportunity can actually exacerbate unfairness. We propose to address this
challenge by modeling feedback effects as Markov decision processes (MDPs).
First, we propose analogs of fairness properties for the MDP setting. Second,
we propose algorithms for learning fair decision-making policies for MDPs.
Finally, we demonstrate the need to account for dynamical effects using
simulations on a loan applicant MDP.

    

### [[1901.08576] Learning Interpretable Models with Causal Guarantees](http://arxiv.org/abs/1901.08576)


  Machine learning has shown much promise in helping improve the quality of
medical, legal, and financial decision-making. In these applications, machine
learning models must satisfy two important criteria: (i) they must be causal,
since the goal is typically to predict individual treatment effects, and (ii)
they must be interpretable, so that human decision makers can validate and
trust the model predictions. There has recently been much progress along each
direction independently, yet the state-of-the-art approaches are fundamentally
incompatible. We propose a framework for learning interpretable models from
observational data that can be used to predict individual treatment effects
(ITEs). In particular, our framework converts any supervised learning algorithm
into an algorithm for estimating ITEs. Furthermore, we prove an error bound on
the treatment effects predicted by our model. Finally, in an experiment on
real-world data, we show that the models trained using our framework
significantly outperform a number of baselines.

    

### [[1902.02181] Attention in Natural Language Processing](http://arxiv.org/abs/1902.02181)


  Attention is an increasingly popular mechanism used in a wide range of neural
architectures. The mechanism itself has been realized in a variety of formats.
However, because of the fast-paced advances in this domain, a systematic
overview of attention is still missing. In this article, we define a unified
model for attention architectures in natural language processing, with a focus
on those designed to work with vector representations of the textual data. We
propose a taxonomy of attention models according to four dimensions: the
representation of the input, the compatibility function, the distribution
function, and the multiplicity of the input and/or output. We present the
examples of how prior information can be exploited in attention models and
discuss ongoing research efforts and open challenges in the area, providing the
first extensive categorization of the vast body of literature in this exciting
domain.

    

### [[1905.00531] Recombinator-k-means: An evolutionary algorithm that exploits k-means++ for recombination](http://arxiv.org/abs/1905.00531)


  We introduce an evolutionary algorithm called recombinator-$k$-means for
optimizing the highly non-convex kmeans problem. Its defining feature is that
its crossover step involves all the members of the current generation,
stochastically recombining them with a repurposed variant of the $k$-means++
seeding algorithm. The recombination also uses a reweighting mechanism that
realizes a progressively sharper stochastic selection policy and ensures that
the population eventually coalesces into a single solution. We compare this
scheme with state-of-the-art alternative, a more standard genetic algorithm
with deterministic pairwise-nearest-neighbor crossover and an elitist selection
policy, of which we also provide an augmented and efficient implementation.
Extensive tests on large and challenging datasets (both synthetic and
real-word) show that for fixed population sizes recombinator-$k$-means is
generally superior in terms of the optimization objective, at the cost of a
more expensive crossover step. When adjusting the population sizes of the two
algorithms to match their running times, we find that for short times the
(augmented) pairwise-nearest-neighbor method is always superior, while at
longer times recombinator-$k$-means will match it and, on the most difficult
examples, take over. We conclude that the reweighted whole-population
recombination is more costly, but generally better at escaping local minima.
Moreover, it is algorithmically simpler and more general (it could be applied
even to $k$-medians or $k$-medoids, for example). Our implementations are
publicly available.

    

### [[1906.05173] Multi-local Collaborative AutoEncoder](http://arxiv.org/abs/1906.05173)


  The excellent performance of representation learning of autoencoders have
attracted considerable interest in various applications. However, the structure
and multi-local collaborative relationships of unlabeled data are ignored in
their encoding procedure that limits the capability of feature extraction. This
paper presents a Multi-local Collaborative AutoEncoder (MC-AE), which consists
of novel multi-local collaborative representation RBM (mcrRBM) and multi-local
collaborative representation GRBM (mcrGRBM) models. Here, the Locality
Sensitive Hashing (LSH) method is used to divide the input data into
multi-local cross blocks which contains multi-local collaborative relationships
of the unlabeled data and features since the similar multi-local instances and
features of the input data are divided into the same block. In mcrRBM and
mcrGRBM models, the structure and multi-local collaborative relationships of
unlabeled data are integrated into their encoding procedure. Then, the local
hidden features converges on the center of each local collaborative block.
Under the collaborative joint influence of each local block, the proposed MC-AE
has powerful capability of representation learning for unsupervised clustering.
However, our MC-AE model perhaps perform training process for a long time on
the large-scale and high-dimensional datasets because more local collaborative
blocks are integrate into it. Five most related deep models are compared with
our MC-AE. The experimental results show that the proposed MC-AE has more
excellent capabilities of collaborative representation and generalization than
the contrastive deep models.

    

### [[1910.09293] Approximation capabilities of neural networks on unbounded domains](http://arxiv.org/abs/1910.09293)


  In this paper, we prove that a shallow neural network with a monotone
sigmoid, ReLU, ELU, Softplus, or LeakyReLU activation function can arbitrarily
well approximate any L^p(p>=2) integrable functions defined on R*[0,1]^n. We
also prove that a shallow neural network with a sigmoid, ReLU, ELU, Softplus,
or LeakyReLU activation function expresses no nonzero integrable function
defined on the Euclidean plane. Together with a recent result that the deep
ReLU network can arbitrarily well approximate any integrable function on
Euclidean spaces, we provide a new perspective on the advantage of multiple
hidden layers in the context of ReLU networks. Lastly, we prove that the ReLU
network with depth 3 is a universal approximator in L^p(R^n).

    

### [[1912.01398] TeaNet: universal neural network interatomic potential inspired by iterative electronic relaxations](http://arxiv.org/abs/1912.01398)


  A universal interatomic potential for an arbitrary set of chemical elements
is urgently needed in computational materials science. Graph convolution neural
network (GCN) has rich expressive power, but previously was mainly employed to
transport scalars and vectors, not rank $\ge 2$ tensors. As classic interatomic
potentials were inspired by tight-binding electronic relaxation framework, we
want to represent this iterative propagation of rank $\ge 2$ tensor information
by GCN. Here we propose an architecture called the tensor embedded atom network
(TeaNet) where angular interaction is translated into graph convolution through
the incorporation of Euclidean tensors, vectors and scalars. By applying the
residual network (ResNet) architecture and training with recurrent GCN weights
initialization, a much deeper (16 layers) GCN was constructed, whose flow is
similar to an iterative electronic relaxation. Our traning dataset is generated
by density functional theory calculation of mostly chemically and structurally
randomized configurations. We demonstrate that arbitrary structures and
reactions involving the first 18 elements on the periodic table (H to Ar) can
be realized satisfactorily by TeaNet, including C-H molecular structures,
metals, amorphous SiO${}_2$, and water, showing surprisingly good performance
(energy mean absolute error 19 meV/atom) and robustness for arbitrary
chemistries involving elements from H to Ar.

    

### [[2002.08537] Adaptive Temporal Difference Learning with Linear Function Approximation](http://arxiv.org/abs/2002.08537)


  This paper revisits the temporal difference (TD) learning algorithm for the
policy evaluation tasks in reinforcement learning. Typically, the performance
of TD(0) and TD($\lambda$) is very sensitive to the choice of stepsizes.
Oftentimes, TD(0) suffers from slow convergence. Motivated by the tight link
between the TD(0) learning algorithm and the stochastic gradient methods, we
develop a provably convergent adaptive projected variant of the TD(0) learning
algorithm with linear function approximation that we term AdaTD(0). In contrast
to the TD(0), AdaTD(0) is robust or less sensitive to the choice of stepsizes.
Analytically, we establish that to reach an $\epsilon$ accuracy, the number of
iterations needed is
$\tilde{O}(\epsilon^{-2}\ln^4\frac{1}{\epsilon}/\ln^4\frac{1}{\rho})$ in the
general case, where $\rho$ represents the speed of the underlying Markov chain
converges to the stationary distribution. This implies that the iteration
complexity of AdaTD(0) is no worse than that of TD(0) in the worst case. When
the stochastic semi-gradients are sparse, we provide theoretical acceleration
of AdaTD(0). Going beyond TD(0), we develop an adaptive variant of
TD($\lambda$), which is referred to as AdaTD($\lambda$). Empirically, we
evaluate the performance of AdaTD(0) and AdaTD($\lambda$) on several standard
reinforcement learning tasks, which demonstrate the effectiveness of our new
approaches.

    

### [[2002.12898] PM2.5-GNN: A Domain Knowledge Enhanced Graph Neural Network For PM2.5 Forecasting](http://arxiv.org/abs/2002.12898)


  When predicting PM2.5 concentrations, it is necessary to consider complex
information sources since the concentrations are influenced by various factors
within a long period. In this paper, we identify a set of critical domain
knowledge for PM2.5 forecasting and develop a novel graph based model,
PM2.5-GNN, being capable of capturing long-term dependencies. On a real-world
dataset, we validate the effectiveness of the proposed model and examine its
abilities of capturing both fine-grained and long-term influences in PM2.5
process. The proposed PM2.5-GNN has also been deployed online to provide free
forecasting service.

    

### [[2003.06069] A General Framework for Learning Mean-Field Games](http://arxiv.org/abs/2003.06069)


  This paper presents a general mean-field game (GMFG) framework for
simultaneous learning and decision-making in stochastic games with a large
population. It first establishes the existence of a unique Nash Equilibrium to
this GMFG, and demonstrates that naively combining reinforcement learning with
the fixed-point approach in classical MFGs yields unstable algorithms. It then
proposes value-based and policy-based reinforcement learning algorithms (GMF-V
and GMF-P, respectively) with smoothed policies, with analysis of their
convergence properties and computational complexities. Experiments on an
equilibrium product pricing problem demonstrate that GMF-V-Q and GMF-P-TRPO,
two specific instantiations of GMF-V and GMF-P, respectively, with Q-learning
and TRPO, are both efficient and robust in the GMFG setting. Moreover, their
performance is superior in convergence speed, accuracy, and stability when
compared with existing algorithms for multi-agent reinforcement learning in the
$N$-player setting.

    

### [[2003.09198] A unified framework for spectral clustering in sparse graphs](http://arxiv.org/abs/2003.09198)


  This article considers spectral community detection in the regime of sparse
networks with heterogeneous degree distributions, for which we devise an
algorithm to efficiently retrieve communities. Specifically, we demonstrate
that a conveniently parametrized form of regularized Laplacian matrix can be
used to perform spectral clustering in sparse networks, without suffering from
its degree heterogeneity. Besides, we exhibit important connections between
this proposed matrix and the now popular non-backtracking matrix, the
Bethe-Hessian matrix, as well as the standard Laplacian matrix. Interestingly,
as opposed to competitive methods, our proposed improved parametrization
inherently accounts for the hardness of the classification problem. These
findings are summarized under the form of an algorithm capable of both
estimating the number of communities and achieving high-quality community
reconstruction.

    

### [[2004.12427] Cross-Domain Structure Preserving Projection for Heterogeneous Domain Adaptation](http://arxiv.org/abs/2004.12427)


  Heterogeneous Domain Adaptation (HDA) addresses the transfer learning
problems where data from the source and target domains are of different
modalities (e.g., texts and images) or feature dimensions (e.g., features
extracted with different methods). It is useful for multi-modal data analysis.
Traditional domain adaptation algorithms assume that the representations of
source and target samples reside in the same feature space, hence are likely to
fail in solving the heterogeneous domain adaptation problem. Contemporary
state-of-the-art HDA approaches are usually composed of complex optimization
objectives for favourable performance and are therefore computationally
expensive and less generalizable. To address these issues, we propose a novel
Cross-Domain Structure Preserving Projection (CDSPP) algorithm for HDA. As an
extension of the classic LPP to heterogeneous domains, CDSPP aims to learn
domain-specific projections to map sample features from source and target
domains into a common subspace such that the class consistency is preserved and
data distributions are sufficiently aligned. CDSPP is simple and has
deterministic solutions by solving a generalized eigenvalue problem. It is
naturally suitable for supervised HDA but has also been extended for
semi-supervised HDA where the unlabelled target domain samples are available.
Extensive experiments have been conducted on commonly used benchmark datasets
(i.e. Office-Caltech, Multilingual Reuters Collection, NUS-WIDE-ImageNet) for
HDA as well as the Office-Home dataset firstly introduced for HDA by ourselves
due to its significantly larger number of classes than the existing ones (65 vs
10, 6 and 8). The experimental results of both supervised and semi-supervised
HDA demonstrate the superior performance of our proposed method against
contemporary state-of-the-art methods.

    

### [[2005.07041] SQuARM-SGD: Communication-Efficient Momentum SGD for Decentralized Optimization](http://arxiv.org/abs/2005.07041)


  In this paper, we propose and analyze SQuARM-SGD, a communication-efficient
algorithm for decentralized training of large-scale machine learning models
over a network. In SQuARM-SGD, each node performs a fixed number of local SGD
steps using Nesterov's momentum and then sends sparsified and quantized updates
to its neighbors regulated by a locally computable triggering criterion. We
provide convergence guarantees of our algorithm for general (non-convex) and
convex smooth objectives, which, to the best of our knowledge, is the first
theoretical analysis for compressed decentralized SGD with momentum updates. We
show that the convergence rate of SQuARM-SGD matches that of vanilla SGD. We
empirically show that including momentum updates in SQuARM-SGD can lead to
better test performance than the current state-of-the-art which does not
consider momentum updates.

    

### [[2005.13815] Adversarial Classification via Distributional Robustness with Wasserstein Ambiguity](http://arxiv.org/abs/2005.13815)


  We study a model for adversarial classification based on distributionally
robust chance constraints. We show that under Wasserstein ambiguity, the model
aims to minimize the conditional value-at-risk of the distance to
misclassification, and we explore links to adversarial classification models
proposed earlier and to maximum-margin classifiers. We also provide a
reformulation of the distributionally robust model for linear classification,
and show it is equivalent to minimizing a regularized ramp loss objective.
Numerical experiments show that, despite the nonconvexity of this formulation,
standard descent methods appear to converge to the global minimizer for this
problem. Inspired by this observation, we show that, for a certain class of
distributions, the only stationary point of the regularized ramp loss
minimization problem is the global minimizer.

    

### [[2006.01980] On the Equivalence between Online and Private Learnability beyond Binary Classification](http://arxiv.org/abs/2006.01980)


  Alon et al. [2019] and Bun et al. [2020] recently showed that online
learnability and private PAC learnability are equivalent in binary
classification. We investigate whether this equivalence extends to multi-class
classification and regression. First, we show that private learnability implies
online learnability in both settings. Our extension involves studying a novel
variant of the Littlestone dimension that depends on a tolerance parameter and
on an appropriate generalization of the concept of threshold functions beyond
binary classification. Second, we show that while online learnability continues
to imply private learnability in multi-class classification, current proof
techniques encounter significant hurdles in the regression setting. While the
equivalence for regression remains open, we provide non-trivial sufficient
conditions for an online learnable class to also be privately learnable.

    

### [[2006.02608] Meta-Model-Based Meta-Policy Optimization](http://arxiv.org/abs/2006.02608)


  Model-based meta-reinforcement learning (RL) methods have recently been shown
to be a promising approach to improving the sample efficiency of RL in
multi-task settings. However, the theoretical understanding of those methods is
yet to be established, and there is currently no theoretical guarantee of their
performance in a real-world environment. In this paper, we analyze the
performance guarantee of model-based meta-RL methods by extending the theorems
proposed by Janner et al. (2019). On the basis of our theoretical results, we
propose Meta-Model-Based Meta-Policy Optimization (M3PO), a model-based meta-RL
method with a performance guarantee. We demonstrate that M3PO outperforms
existing meta-RL methods in continuous-control benchmarks.

    

### [[2006.04804] Optimal Transport Graph Neural Networks](http://arxiv.org/abs/2006.04804)


  Current graph neural network (GNN) architectures naively average or sum node
embeddings into an aggregated graph representation -- potentially losing
structural or semantic information. We here introduce OT-GNN, a model that
computes graph embeddings using parametric prototypes that highlight key facets
of different graph aspects. Towards this goal, we successfully combine optimal
transport (OT) with parametric graph models. Graph representations are obtained
from Wasserstein distances between the set of GNN node embeddings and
``prototype'' point clouds as free parameters. We theoretically prove that,
unlike traditional sum aggregation, our function class on point clouds
satisfies a fundamental universal approximation theorem. Empirically, we
address an inherent collapse optimization issue by proposing a noise
contrastive regularizer to steer the model towards truly exploiting the OT
geometry. Finally, we outperform popular methods on several molecular property
prediction tasks, while exhibiting smoother graph representations.

    

### [[2006.06267] A Generalised Linear Model Framework for $β$-Variational Autoencoders based on Exponential Dispersion Families](http://arxiv.org/abs/2006.06267)


  Although variational autoencoders (VAE) are successfully used to obtain
meaningful low-dimensional representations for high-dimensional data, the
characterization of critical points of the loss function for general
observation models is not fully understood. We introduce a theoretical
framework that is based on a connection between $\beta$-VAE and generalized
linear models (GLM). The equality between the activation function of a
$\beta$-VAE and the inverse of the link function of a GLM enables us to provide
a systematic generalization of the loss analysis for $\beta$-VAE based on the
assumption that the observation model distribution belongs to an exponential
dispersion family (EDF). As a result, we can initialize $\beta$-VAE nets by
maximum likelihood estimates (MLE) that enhance the training performance on
both synthetic and real world data sets. As a further consequence, we
analytically describe the auto-pruning property inherent in the $\beta$-VAE
objective and reason for posterior collapse.

    

### [[2006.08426] Walking in the Shadow: A New Perspective on Descent Directions for Constrained Minimization](http://arxiv.org/abs/2006.08426)


  Descent directions such as movement towards Frank-Wolfe vertices, away steps,
in-face away steps and pairwise directions have been an important design
consideration in conditional gradient descent (CGD) variants. In this work, we
attempt to demystify the impact of movement in these directions towards
attaining constrained minimizers. The best local direction of descent is the
directional derivative of the projection of the gradient, which we refer to as
the $\textit{shadow}$ of the gradient. We show that the continuous-time
dynamics of moving in the shadow are equivalent to those of PGD however
non-trivial to discretize. By projecting gradients in PGD, one not only ensures
feasibility but is also able to "wrap" around the convex region. We show that
Frank-Wolfe (FW) vertices in fact recover the maximal wrap one can obtain by
projecting gradients, thus providing a new perspective on these steps. We also
claim that the shadow steps give the best direction of descent emanating from
the convex hull of all possible away-steps. Viewing PGD movements in terms of
shadow steps gives linear convergence, dependent on the number of faces. We
combine these insights into a novel $S\small{HADOW}$-$CG$ method that uses FW
steps (i.e., wrap around the polytope) and shadow steps (i.e., optimal local
descent direction), while enjoying linear convergence. Our analysis develops
properties of the curve formed by projecting a line on a polytope, which may be
of independent interest, while providing a unifying view of various descent
directions in the CGD literature.

    

### [[2006.08464] Globally Injective ReLU Networks](http://arxiv.org/abs/2006.08464)


  Injectivity plays an important role in generative models where it enables
inference; in inverse problems and compressed sensing with generative priors it
is a precursor to well posedness. We establish sharp characterizations of
injectivity of fully-connected and convolutional ReLU layers and networks.
First, through a layerwise analysis, we show that an expansivity factor of two
is necessary and sufficient for injectivity by constructing appropriate weight
matrices. We show that global injectivity with iid Gaussian matrices, a
commonly used tractable model, requires larger expansivity between 3.4 and
10.5. We also characterize the stability of inverting an injective network via
worst-case Lipschitz constants of the inverse. We then use arguments from
differential topology to study injectivity of deep networks and prove that any
Lipschitz map can be approximated by an injective ReLU network. Finally, using
an argument based on random projections, we show that an end-to-end -- rather
than layerwise -- doubling of the dimension suffices for injectivity. Our
results establish a theoretical basis for the study of nonlinear inverse and
inference problems using neural networks.

    

### [[2006.11654] Counterfactually Guided Off-policy Transfer in Clinical Settings](http://arxiv.org/abs/2006.11654)


  Domain shift creates significant challenges for sequential decision making in
healthcare since the target domain may be data-scarce and confounded. In this
paper, we propose a method for off-policy transfer by modeling the underlying
generative process with a causal mechanism. We use informative priors from the
source domain to augment counterfactual trajectories in the target in a
principled manner. We demonstrate how this addresses data-scarcity in the
presence of unobserved confounding. The causal parametrization of our sampling
procedure guarantees that counterfactual quantities can be estimated from
scarce observational target data, maintaining intuitive stability properties.
Policy learning in the target domain is further regularized via the source
policy through KL-divergence. Through evaluation on a simulated sepsis
treatment task, our counterfactual policy transfer procedure significantly
improves the performance of a learned treatment policy when assumptions of
"no-unobserved confounding" are relaxed.

    

### [[2006.15666] Breathing K-Means](http://arxiv.org/abs/2006.15666)


  The k-means++ algorithm is the de-facto standard for finding approximate
solutions to the k-means problem. A widely used implementation is provided by
the scikit-learn Python package for machine learning. We propose the breathing
k-means algorithm, which on average significantly outperforms scikit-learn's
k-means++ w.r.t. both solution quality and execution speed. The initialization
step in the new method is done by k-means++ but without the usual (and costly)
repetitions (ten in scikit-learn). The core of the new method is a sequence of
"breathing cycles," each consisting of a "breathe in" step where the number of
centroids is increased by m and a "breathe out" step where m centroids are
removed. Each step is ended by a run of Lloyd's algorithm. The parameter m is
decreased until zero, at which point the algorithm terminates. With the default
(m = 5), breathing k-means dominates scikit-learn's k-means++. This is
demonstrated via experiments on various data sets, including all those from the
original k-means++ publication. By setting m to smaller or larger values, one
can optionally produce faster or better solutions, respectively. For larger
values of m, e.g., m = 20, breathing k-means likely is the new SOTA for the
k-means problem.

    

### [[2007.03812] Robust Multi-Agent Multi-Armed Bandits](http://arxiv.org/abs/2007.03812)


  Recent works have shown that agents facing independent instances of a
stochastic $K$-armed bandit can collaborate to decrease regret. However, these
works assume that each agent always recommends their individual best-arm
estimates to other agents, which is unrealistic in envisioned applications
(machine faults in distributed computing or spam in social recommendation
systems). Hence, we generalize the setting to include $n$ honest and $m$
malicious agents who recommend best-arm estimates and arbitrary arms,
respectively. We first show that even with a single malicious agent, existing
collaboration-based algorithms fail to improve regret guarantees over a
single-agent baseline. We propose a scheme where honest agents learn who is
malicious and dynamically reduce communication with (i.e., "block") them. We
show that collaboration indeed decreases regret for this algorithm, assuming
$m$ is small compared to $K$ but without assumptions on malicious agents'
behavior, thus ensuring that our algorithm is robust against any malicious
recommendation strategy.

    

### [[2007.10492] Assessment of COVID-19 hospitalization forecasts from a simplified SIR model](http://arxiv.org/abs/2007.10492)


  We propose the SH model, a simplified version of the well-known SIR
compartmental model of infectious diseases. With optimized parameters and
initial conditions, this time-invariant two-parameter two-dimensional model is
able to fit COVID-19 hospitalization data over several months with high
accuracy (e.g., the root relative squared error is below 10% for Belgium over
the period from 2020-03-15 to 2020-07-15). Moreover, we observed that, when the
model is trained on a suitable three-week period around the first
hospitalization peak for Belgium, it forecasts the subsequent two months with
mean absolute percentage error (MAPE) under 4%. We repeated the experiment for
each French department and found 14 of them where the MAPE was below 20%.
However, when the model is trained in the increase phase, it is less successful
at forecasting the subsequent evolution.

    

### [[2007.14110] WaveFuse: A Unified Deep Framework for Image Fusion with Discrete Wavelet Transform](http://arxiv.org/abs/2007.14110)


  We propose an unsupervised image fusion architecture for multiple application
scenarios based on the combination of multi-scale discrete wavelet transform
through regional energy and deep learning. To our best knowledge, this is the
first time the conventional image fusion method has been combined with deep
learning. The useful information of feature maps can be utilized adequately
through multi-scale discrete wavelet transform in our proposed method.Compared
with other state-of-the-art fusion method, the proposed algorithm exhibits
better fusion performance in both subjective and objective evaluation.
Moreover, it's worth mentioning that comparable fusion performance trained in
COCO dataset can be obtained by training with a much smaller dataset with only
hundreds of images chosen randomly from COCO. Hence, the training time is
shortened substantially, leading to the improvement of the model's performance
both in practicality and training efficiency.

    

### [[2007.14268] On the Convergence of Tsetlin Machines for the IDENTITY- and NOT Operators](http://arxiv.org/abs/2007.14268)


  The Tsetlin Machine (TM) is a recent machine learning algorithm with several
distinct properties, such as interpretability, simplicity, and
hardware-friendliness. Although numerous empirical evaluations report on its
performance, the mathematical analysis of its convergence is still open. In
this article, we analyze the convergence of the TM with only one clause
involved for classification. More specifically, we examine two basic logical
operators, namely, the "IDENTITY"- and "NOT" operators. Our analysis reveals
that the TM, with just one clause, can converge correctly to the intended
logical operator, learning from training data over an infinite time horizon.
Besides, it can capture arbitrarily rare patterns and select the most accurate
one when two candidate patterns are incompatible, by configuring a granularity
parameter. The analysis of the convergence of the two basic operators lays the
foundation for analyzing other logical operators. These analyses altogether,
from a mathematical perspective, provide new insights on why TMs have obtained
state-of-the-art performance on several pattern recognition problems.

    

### [[2008.11348] Variance-Reduced Splitting Schemes for Monotone Stochastic Generalized Equations](http://arxiv.org/abs/2008.11348)


  We consider monotone inclusion problems where the operators may be
expectation-valued, a class of problems that subsumes convex stochastic
optimization problems as well as subclasses of stochastic variational
inequality and equilibrium problems. A direct application of splitting schemes
is complicated by the need to resolve problems with expectation-valued maps at
each step, a concern that is addressed by using sampling. Accordingly, we
propose an avenue for addressing uncertainty in the mapping: Variance- reduced
stochastic modified forward-backward splitting scheme (vr-SMFBS). In
constrained settings, we consider structured settings when the map can be
decomposed into an expectation-valued map A and a maximal monotone map B with a
tractable resolvent. We show that the proposed schemes are equipped with a.s.
convergence guarantees, linear (strongly monotone A) and O(1/k) (monotone A)
rates of convergence while achieving optimal oracle complexity bounds. The rate
statements in monotone regimes appear to be amongst the first and rely on
leveraging the Fitzpatrick gap function for monotone inclusions. Furthermore,
the schemes rely on weaker moment requirements on noise and allow for weakening
unbiasedness requirements on oracles in strongly monotone regimes. Preliminary
numerics on a class of two-stage stochastic variational inequality problems
reflect these findings and show that the variance-reduced schemes outperform
stochastic approximation schemes and sample-average approximation approaches.
The benefits of attaining deterministic rates of convergence become even more
salient when resolvent computation is expensive.

    

### [[2008.13293] Sharp finite-sample concentration of independent variables](http://arxiv.org/abs/2008.13293)


  We show an extension of Sanov's theorem on large deviations, controlling the
tail probabilities of i.i.d. random variables with matching concentration and
anti-concentration bounds. This result has a general scope, applies to samples
of any size, and has a short information-theoretic proof using elementary
techniques.

    

### [[2009.01974] FedBE: Making Bayesian Model Ensemble Applicable to Federated Learning](http://arxiv.org/abs/2009.01974)


  Federated learning aims to collaboratively train a strong global model by
accessing users' locally trained models but not their own data. A crucial step
is therefore to aggregate local models into a global model, which has been
shown challenging when users have non-i.i.d. data. In this paper, we propose a
novel aggregation algorithm named FedBE, which takes a Bayesian inference
perspective by sampling higher-quality global models and combining them via
Bayesian model Ensemble, leading to much robust aggregation. We show that an
effective model distribution can be constructed by simply fitting a Gaussian or
Dirichlet distribution to the local models. Our empirical studies validate
FedBE's superior performance, especially when users' data are not i.i.d. and
when the neural networks go deeper. Moreover, FedBE is compatible with recent
efforts in regularizing users' model training, making it an easily applicable
module: you only need to replace the aggregation method but leave other parts
of your federated learning algorithm intact. Our code is publicly available at
this https URL.

    

### [[2009.05986] Oracle-Efficient Regret Minimization in Factored MDPs with Unknown Structure](http://arxiv.org/abs/2009.05986)


  We study regret minimization in non-episodic factored Markov decision
processes (FMDPs), where all existing algorithms make the strong assumption
that the factored structure of the FMDP is known to the learner in advance. In
this paper, we provide the first algorithm that learns the structure of the
FMDP while minimizing the regret. Our algorithm is based on the optimism in
face of uncertainty principle, combined with a simple statistical method for
structure learning, and can be implemented efficiently given oracle-access to
an FMDP planner. Moreover, we give a variant of our algorithm that remains
efficient even when the oracle is limited to non-factored actions, which is the
case with almost all existing approximate planners. Finally, we leverage our
techniques to prove a novel lower bound for the known structure case, closing
the gap to the regret bound of Chen et al. [2021].

    

### [[2009.08058] MultAV: Multiplicative Adversarial Videos](http://arxiv.org/abs/2009.08058)


  The majority of adversarial machine learning research focuses on additive
attacks, which add adversarial perturbation to input data. On the other hand,
unlike image recognition problems, only a handful of attack approaches have
been explored in the video domain. In this paper, we propose a novel attack
method against video recognition models, Multiplicative Adversarial Videos
(MultAV), which imposes perturbation on video data by multiplication. MultAV
has different noise distributions to the additive counterparts and thus
challenges the defense methods tailored to resisting additive adversarial
attacks. Moreover, it can be generalized to not only Lp-norm attacks with a new
adversary constraint called ratio bound, but also different types of physically
realizable attacks. Experimental results show that the model adversarially
trained against additive attack is less robust to MultAV.

    

### [[2009.09590] Generalized Clustering and Multi-Manifold Learning with Geometric Structure Preservation](http://arxiv.org/abs/2009.09590)


  Though manifold-based clustering has become a popular research topic, we
observe that one important factor has been omitted by these works, namely that
the defined clustering loss may corrupt the local and global structure of the
latent space. In this paper, we propose a novel Generalized Clustering and
Multi-manifold Learning (GCML) framework with geometric structure preservation
for generalized data, i.e., not limited to 2-D image data and has a wide range
of applications in speech, text, and biology domains. In the proposed
framework, manifold clustering is done in the latent space guided by a
clustering loss. To overcome the problem that the clustering-oriented loss may
deteriorate the geometric structure of the latent space, an isometric loss is
proposed for preserving intra-manifold structure locally and a ranking loss for
inter-manifold structure globally. Extensive experimental results have shown
that GCML exhibits superior performance to counterparts in terms of qualitative
visualizations and quantitative metrics, which demonstrates the effectiveness
of preserving geometric structure.

    

### [[2010.03408] Machine learning for recovery factor estimation of an oil reservoir: a tool for de-risking at a hydrocarbon asset evaluation](http://arxiv.org/abs/2010.03408)


  Well known oil recovery factor estimation techniques such as analogy,
volumetric calculations, material balance, decline curve analysis, hydrodynamic
simulations have certain limitations. Those techniques are time-consuming,
require specific data and expert knowledge. Besides, though uncertainty
estimation is highly desirable for this problem, the methods above do not
include this by default. In this work, we present a data-driven technique for
oil recovery factor estimation using reservoir parameters and representative
statistics. We apply advanced machine learning methods to historical worldwide
oilfields datasets (more than 2000 oil reservoirs). The data-driven model might
be used as a general tool for rapid and completely objective estimation of the
oil recovery factor. In addition, it includes the ability to work with partial
input data and to estimate the prediction interval of the oil recovery factor.
We perform the evaluation in terms of accuracy and prediction intervals
coverage for several tree-based machine learning techniques in application to
the following two cases: (1) using parameters only related to geometry,
geology, transport, storage and fluid properties, (2) using an extended set of
parameters including development and production data. For both cases model
proved itself to be robust and reliable. We conclude that the proposed
data-driven approach overcomes several limitations of the traditional methods
and is suitable for rapid, reliable and objective estimation of oil recovery
factor for hydrocarbon reservoir.

    

### [[2010.08666] Active Domain Adaptation via Clustering Uncertainty-weighted Embeddings](http://arxiv.org/abs/2010.08666)


  Generalizing deep neural networks to new target domains is critical to their
real-world utility. In practice, it may be feasible to get some target data
labeled, but to be cost-effective it is desirable to select a
maximally-informative subset via active learning (AL). We study the problem of
AL under a domain shift, called Active Domain Adaptation (Active DA). We
demonstrate how existing AL approaches based solely on model uncertainty or
diversity sampling are less effective for Active DA. We propose Clustering
Uncertainty-weighted Embeddings (CLUE), a novel label acquisition strategy for
Active DA that performs uncertainty-weighted clustering to identify target
instances for labeling that are both uncertain under the model and diverse in
feature space. CLUE consistently outperforms competing label acquisition
strategies for Active DA and AL across learning settings on 6 diverse domain
shifts for image classification.

    

### [[2010.13365] Robustness May Be at Odds with Fairness: An Empirical Study on Class-wise Accuracy](http://arxiv.org/abs/2010.13365)


  Convolutional neural networks (CNNs) have made significant advancement,
however, they are widely known to be vulnerable to adversarial attacks.
Adversarial training is the most widely used technique for improving
adversarial robustness to strong white-box attacks. Prior works have been
evaluating and improving the model average robustness without class-wise
evaluation. The average evaluation alone might provide a false sense of
robustness. For example, the attacker can focus on attacking the vulnerable
class, which can be dangerous, especially, when the vulnerable class is a
critical one, such as "human" in autonomous driving. We propose an empirical
study on the class-wise accuracy and robustness of adversarially trained
models. We find that there exists inter-class discrepancy for accuracy and
robustness even when the training dataset has an equal number of samples for
each class. For example, in CIFAR10, "cat" is much more vulnerable than other
classes. Moreover, this inter-class discrepancy also exists for normally
trained models, while adversarial training tends to further increase the
discrepancy. Our work aims to investigate the following questions: (a) is the
phenomenon of inter-class discrepancy universal regardless of datasets, model
architectures and optimization hyper-parameters? (b) If so, what can be
possible explanations for the inter-class discrepancy? (c) Can the techniques
proposed in the long tail classification be readily extended to adversarial
training for addressing the inter-class discrepancy?

    

### [[2011.01516] Quadratic Metric Elicitation for Fairness and Beyond](http://arxiv.org/abs/2011.01516)


  Metric elicitation is a recent framework for eliciting performance metrics
that best reflect implicit user preferences based on the application and
context. However, available elicitation strategies have been limited to linear
(or quasi-linear) functions of predictive rates, which can be practically
restrictive for many domains including fairness. This paper develops a strategy
for eliciting more flexible multiclass metrics defined by quadratic functions
of rates, designed to reflect human preferences better. We show its application
in eliciting quadratic violation-based group-fair metrics. Our strategy
requires only relative preference feedback, and that too of near-optimal
amount, and is robust to feedback noise. We further extend this strategy to
eliciting polynomial metrics -- thus broadening the use cases for metric
elicitation.

    

### [[2011.07720] Distributed Bandits: Probabilistic Communication on $d$-regular Graphs](http://arxiv.org/abs/2011.07720)


  We study the decentralized multi-agent multi-armed bandit problem for agents
that communicate with probability over a network defined by a $d$-regular
graph. Every edge in the graph has probabilistic weight $p$ to account for the
($1\!-\!p$) probability of a communication link failure. At each time step,
each agent chooses an arm and receives a numerical reward associated with the
chosen arm. After each choice, each agent observes the last obtained reward of
each of its neighbors with probability $p$. We propose a new Upper Confidence
Bound (UCB) based algorithm and analyze how agent-based strategies contribute
to minimizing group regret in this probabilistic communication setting. We
provide theoretical guarantees that our algorithm outperforms state-of-the-art
algorithms. We illustrate our results and validate the theoretical claims using
numerical simulations.

    

### [[2012.03174] Counting Substructures with Higher-Order Graph Neural Networks: Possibility and Impossibility Results](http://arxiv.org/abs/2012.03174)


  While message passing Graph Neural Networks (GNNs) have become increasingly
popular architectures for learning with graphs, recent works have revealed
important shortcomings in their expressive power. In response, several
higher-order GNNs have been proposed that substantially increase the expressive
power, albeit at a large computational cost. Motivated by this gap, we explore
alternative strategies and lower bounds. In particular, we analyze a new
recursive pooling technique of local neighborhoods that allows different
tradeoffs of computational cost and expressive power. First, we prove that this
model can count subgraphs of size $k$, and thereby overcomes a known limitation
of low-order GNNs. Second, we show how recursive pooling can exploit sparsity
to reduce the computational complexity compared to the existing higher-order
GNNs. More generally, we provide a (near) matching information-theoretic lower
bound for counting subgraphs with graph representations that pool over
representations of derived (sub-)graphs. We also discuss lower bounds on time
complexity.

    

### [[2012.06951] Attentional Biased Stochastic Gradient for Imbalanced Classification](http://arxiv.org/abs/2012.06951)


  In this paper, we present a simple yet effective method (ABSGD) for
addressing the data imbalance issue in deep learning. Our method is a simple
modification to momentum SGD where we leverage an attentional mechanism to
assign an individual importance weight to each gradient in the mini-batch.
Unlike many existing heuristic-driven methods for tackling data imbalance, our
method is grounded in {\it theoretically justified distributionally robust
optimization (DRO)}, which is guaranteed to converge to a stationary point of
an information-regularized DRO problem. The individual-level weight of a
sampled data is systematically proportional to the exponential of a scaled loss
value of the data, where the scaling factor is interpreted as the
regularization parameter in the framework of information-regularized DRO.
Compared with existing class-level weighting schemes, our method can capture
the diversity between individual examples within each class. Compared with
existing individual-level weighting methods using meta-learning that require
three backward propagations for computing mini-batch stochastic gradients, our
method is more efficient with only one backward propagation at each iteration
as in standard deep learning methods. To balance between the learning of
feature extraction layers and the learning of the classifier layer, we employ a
two-stage method that uses SGD for pretraining followed by ABSGD for learning a
robust classifier and finetuning lower layers. Our empirical studies on several
benchmark datasets demonstrate the effectiveness of the proposed method.

    

### [[2012.09400] Stochastic Compositional Gradient Descent under Compositional constraints](http://arxiv.org/abs/2012.09400)


  This work studies constrained stochastic optimization problems where the
objective and constraint functions are convex and expressed as compositions of
stochastic functions. The problem arises in the context of fair classification,
fair regression, and the design of queuing systems. Of particular interest is
the large-scale setting where an oracle provides the stochastic gradients of
the constituent functions, and the goal is to solve the problem with a minimal
number of calls to the oracle. Owing to the compositional form, the stochastic
gradients provided by the oracle do not yield unbiased estimates of the
objective or constraint gradients. Instead, we construct approximate gradients
by tracking the inner function evaluations, resulting in a quasi-gradient
saddle point algorithm. We prove that the proposed algorithm is guaranteed to
find the optimal and feasible solution almost surely. We further establish that
the proposed algorithm requires $\mathcal{O}(1/\epsilon^4)$ data samples in
order to obtain an $\epsilon$-approximate optimal point while also ensuring
zero constraint violation. The result matches the sample complexity of the
stochastic compositional gradient descent method for unconstrained problems and
improves upon the best-known sample complexity results for the constrained
settings. The efficacy of the proposed algorithm is tested on both fair
classification and fair regression problems. The numerical results show that
the proposed algorithm outperforms the state-of-the-art algorithms in terms of
the convergence rate.

    

### [[2012.11460] SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation](http://arxiv.org/abs/2012.11460)


  Many existing approaches for unsupervised domain adaptation (UDA) focus on
adapting under only data distribution shift and offer limited success under
additional cross-domain label distribution shift. Recent work based on
self-training using target pseudo-labels has shown promise, but on challenging
shifts pseudo-labels may be highly unreliable, and using them for self-training
may cause error accumulation and domain misalignment. We propose Selective
Entropy Optimization via Committee Consistency (SENTRY), a UDA algorithm that
judges the reliability of a target instance based on its predictive consistency
under a committee of random image transformations. Our algorithm then
selectively minimizes predictive entropy to increase confidence on highly
consistent target instances, while maximizing predictive entropy to reduce
confidence on highly inconsistent ones. In combination with pseudo-label based
approximate target class balancing, our approach leads to significant
improvements over the state-of-the-art on 27/31 domain shifts from standard UDA
benchmarks as well as benchmarks designed to stress-test adaptation under label
distribution shift.

    

### [[2012.14406] dalex: Responsible Machine Learning with Interactive Explainability and Fairness in Python](http://arxiv.org/abs/2012.14406)


  The increasing amount of available data, computing power, and the constant
pursuit for higher performance results in the growing complexity of predictive
models. Their black-box nature leads to opaqueness debt phenomenon inflicting
increased risks of discrimination, lack of reproducibility, and deflated
performance due to data drift. To manage these risks, good MLOps practices ask
for better validation of model performance and fairness, higher explainability,
and continuous monitoring. The necessity of deeper model transparency appears
not only from scientific and social domains, but also emerging laws and
regulations on artificial intelligence. To facilitate the development of
responsible machine learning models, we showcase dalex, a Python package which
implements the model-agnostic interface for interactive model exploration. It
adopts the design crafted through the development of various tools for
responsible machine learning; thus, it aims at the unification of the existing
solutions. This library's source code and documentation are available under
open license at this https URL.

    

### [[2101.03255] Spending Your Winning Lottery Better After Drawing It](http://arxiv.org/abs/2101.03255)


  Lottery Ticket Hypothesis (LTH) suggests that a dense neural network contains
a sparse sub-network that can match the performance of the original dense
network when trained in isolation from scratch. Most works retrain the sparse
sub-network with the same training protocols as its dense network, such as
initialization, architecture blocks, and training recipes. However, till now it
is unclear that whether these training protocols are optimal for sparse
networks.
In this paper, we demonstrate that it is unnecessary for spare retraining to
strictly inherit those properties from the dense network. Instead, by plugging
in purposeful "tweaks" of the sparse subnetwork architecture or its training
recipe, its retraining can be significantly improved than the default,
especially at high sparsity levels. Combining all our proposed "tweaks" can
yield the new state-of-the-art performance of LTH, and these modifications can
be easily adapted to other sparse training algorithms in general. Specifically,
we have achieved a significant and consistent performance gain of1.05% - 4.93%
for ResNet18 on CIFAR-100 over vanilla-LTH. Moreover, our methods are shown to
generalize across datasets (CIFAR10, CIFAR100, TinyImageNet) and architectures
(Vgg16, ResNet-18/ResNet-34, MobileNet). All codes will be publicly available.

    

### [[2101.05917] DiffPD: Differentiable Projective Dynamics](http://arxiv.org/abs/2101.05917)


  We present a novel, fast differentiable simulator for soft-body learning and
control applications. Existing differentiable soft-body simulators can be
classified into two categories based on their time integration methods:
Simulators using explicit time-stepping schemes require tiny time steps to
avoid numerical instabilities in gradient computation, and simulators using
implicit time integration typically compute gradients by employing the adjoint
method and solving the expensive linearized dynamics. Inspired by Projective
Dynamics (PD), we present Differentiable Projective Dynamics (DiffPD), an
efficient differentiable soft-body simulator based on PD with implicit time
integration. The key idea in DiffPD is to speed up backpropagation by
exploiting the prefactorized Cholesky decomposition in forward PD simulation.
In terms of contact handling, DiffPD supports two types of contacts: a
penalty-based model describing contact and friction forces and a
complementarity-based model enforcing non-penetration conditions and static
friction. We evaluate the performance of DiffPD and observe it is 4-19 times
faster compared with the standard Newton's method in various applications
including system identification, inverse design problems, trajectory
optimization, and closed-loop control. We also apply DiffPD in a
reality-to-simulation (real-to-sim) example with contact and collisions and
show its capability of reconstructing a digital twin of real-world scenes.

    

### [[2101.10502] Model-agnostic interpretation by visualization of feature perturbations](http://arxiv.org/abs/2101.10502)


  Interpretation of machine learning models has become one of the most
important research topics due to the necessity of maintaining control and
avoiding bias in these algorithms. Since many machine learning algorithms are
published every day, there is a need for novel model-agnostic interpretation
approaches that could be used to interpret a great variety of algorithms. Thus,
one advantageous way to interpret machine learning models is to feed different
input data to understand the changes in the prediction. Using such an approach,
practitioners can define relations among data patterns and a model's decision.
This work proposes a model-agnostic interpretation approach that uses
visualization of feature perturbations induced by the PSO algorithm. We
validate our approach on publicly available datasets, showing the capability to
enhance the interpretation of different classifiers while yielding very stable
results compared with state-of-the-art algorithms.

    

### [[2101.12578] Adjusting for Autocorrelated Errors in Neural Networks for Time Series](http://arxiv.org/abs/2101.12578)


  An increasing body of research focuses on using neural networks to model time
series. A common assumption in training neural networks via maximum likelihood
estimation on time series is that the errors across time steps are
uncorrelated. However, errors are actually autocorrelated in many cases due to
the temporality of the data, which makes such maximum likelihood estimations
inaccurate. In this paper, in order to adjust for autocorrelated errors, we
propose to learn the autocorrelation coefficient jointly with the model
parameters. In our experiments, we verify the effectiveness of our approach on
time series forecasting. Results across a wide range of real-world datasets
with various state-of-the-art models show that our method enhances performance
in almost all cases. Based on these results, we suggest empirical critical
values to determine the severity of autocorrelated errors. We also analyze
several aspects of our method to demonstrate its advantages. Finally, other
time series tasks are also considered to validate that our method is not
restricted to only forecasting.

    

### [[2102.07007] Bridging Graph Neural Networks and Statistical Relational Learning: Relational One-Class GCN](http://arxiv.org/abs/2102.07007)


  We consider the problem of learning Graph Convolutional Networks (GCNs) for
relational data. Specifically, we consider the classic link prediction and node
classification problems as relational modeling tasks and develop a relational
extension to GCNs. Our method constructs a secondary graph using relational
density estimation techniques where vertices correspond to the target triples.
We emphasize the importance of learning features using the secondary graph and
the advantages of employing a distance matrix over the typically used adjacency
matrix. Our comprehensive empirical evaluation demonstrates the superiority of
our approach over $12$ different GCN models, relational embedding techniques,
rule learning techniques and relational models.

    

### [[2102.07835] Topological Graph Neural Networks](http://arxiv.org/abs/2102.07835)


  Graph neural networks (GNNs) are a powerful architecture for tackling graph
learning tasks, yet have been shown to be oblivious to eminent substructures
such as cycles. We present TOGL, a novel layer that incorporates global
topological information of a graph using persistent homology. TOGL can be
easily integrated into any type of GNN and is strictly more expressive in terms
of the Weisfeiler-Lehman graph isomorphism test. Augmenting GNNs with our layer
leads to improved predictive performance for graph and node classification
tasks, both on synthetic data sets (which can be classified by humans using
their topology but not by ordinary GNNs) and on real-world data.

    

### [[2102.07976] A General Descent Aggregation Framework for Gradient-based Bi-level Optimization](http://arxiv.org/abs/2102.07976)


  In recent years, a variety of gradient-based methods have been developed to
solve Bi-Level Optimization (BLO) problems in machine learning and computer
vision areas. However, the theoretical correctness and practical effectiveness
of these existing approaches always rely on some restrictive conditions (e.g.,
Lower-Level Singleton, LLS), which could hardly be satisfied in real-world
applications. Moreover, previous literature only proves theoretical results
based on their specific iteration strategies, thus lack a general recipe to
uniformly analyze the convergence behaviors of different gradient-based BLOs.
In this work, we formulate BLOs from an optimistic bi-level viewpoint and
establish a new gradient-based algorithmic framework, named Bi-level Descent
Aggregation (BDA), to partially address the above issues. Specifically, BDA
provides a modularized structure to hierarchically aggregate both the upper-
and lower-level subproblems to generate our bi-level iterative dynamics.
Theoretically, we establish a general convergence analysis template and derive
a new proof recipe to investigate the essential theoretical properties of
gradient-based BLO methods. Furthermore, this work systematically explores the
convergence behavior of BDA in different optimization scenarios, i.e.,
considering various solution qualities (i.e., global/local/stationary solution)
returned from solving approximation subproblems. Extensive experiments justify
our theoretical results and demonstrate the superiority of the proposed
algorithm for hyper-parameter optimization and meta-learning tasks.

    

### [[2102.08649] A General Framework for the Disintegration of PAC-Bayesian Bounds](http://arxiv.org/abs/2102.08649)


  PAC-Bayesian bounds are known to be tight and informative when studying the
generalization ability of randomized classifiers. However, when applied to some
family of deterministic models such as neural networks, they require a loose
and costly derandomization step. As an alternative to this step, we introduce
new PAC-Bayesian generalization bounds that have the originality to provide
disintegrated bounds, i.e., they give guarantees over one single hypothesis
instead of the usual averaged analysis. Our bounds are easily optimizable and
can be used to design learning algorithms. We illustrate the interest of our
result on neural networks and show a significant practical improvement over the
state-of-the-art framework.

    

### [[2102.09743] Personalized Federated Learning: A Unified Framework and Universal Optimization Techniques](http://arxiv.org/abs/2102.09743)


  We study the optimization aspects of personalized Federated Learning (FL). We
propose general optimizers that can be used to solve essentially any existing
personalized FL objective, namely a tailored variant of Local SGD and variants
of accelerated coordinate descent/accelerated SVRCD. By studying a general
personalized objective that is capable of recovering essentially any existing
personalized FL objective as a special case, we develop a universal
optimization theory applicable to all strongly convex personalized FL models in
the literature. We demonstrate the practicality and/or optimality of our
methods both in terms of communication and local computation. Surprisingly
enough, our general optimization solvers and theory are capable of recovering
best-known communication and computation guarantees for solving specific
personalized FL objectives. Thus, our proposed methods can be taken as
universal optimizers that make the design of task-specific optimizers
unnecessary in many cases.

    

### [[2102.10085] Output-Weighted Sampling for Multi-Armed Bandits with Extreme Payoffs](http://arxiv.org/abs/2102.10085)


  We present a new type of acquisition functions for online decision making in
multi-armed and contextual bandit problems with extreme payoffs. Specifically,
we model the payoff function as a Gaussian process and formulate a novel type
of upper confidence bound (UCB) acquisition function that guides exploration
towards the bandits that are deemed most relevant according to the variability
of the observed rewards. This is achieved by computing a tractable likelihood
ratio that quantifies the importance of the output relative to the inputs and
essentially acts as an \textit{attention mechanism} that promotes exploration
of extreme rewards. We demonstrate the benefits of the proposed methodology
across several synthetic benchmarks, as well as a realistic example involving
noisy sensor network data. Finally, we provide a JAX library for efficient
bandit optimization using Gaussian processes.

    

### [[2102.12926] Persistent Homology and Graphs Representation Learning](http://arxiv.org/abs/2102.12926)


  This article aims to study the topological invariant properties encoded in
node graph representational embeddings by utilizing tools available in
persistent homology. Specifically, given a node embedding representation
algorithm, we consider the case when these embeddings are real-valued. By
viewing these embeddings as scalar functions on a domain of interest, we can
utilize the tools available in persistent homology to study the topological
information encoded in these representations. Our construction effectively
defines a unique persistence-based graph descriptor, on both the graph and node
levels, for every node representation algorithm. To demonstrate the
effectiveness of the proposed method, we study the topological descriptors
induced by DeepWalk, Node2Vec and Diff2Vec.

    

### [[2103.02588] IH-GAN: A Conditional Generative Model for Implicit Surface-Based Inverse Design of Cellular Structures](http://arxiv.org/abs/2103.02588)


  Variable-density cellular structures can overcome connectivity and
manufacturability issues of topologically optimized structures, particularly
those represented as discrete density maps. However, the optimization of such
cellular structures is challenging due to the multiscale design problem. Past
work addressing this problem generally either only optimizes the volume
fraction of single-type unit cells but ignoring the effects of unit cell
geometry on properties, or considers the geometry-property relation but builds
this relation via heuristics. In contrast, we propose a simple yet more
principled way to accurately model the property to geometry mapping using a
conditional deep generative model, named Inverse Homogenization Generative
Adversarial Network (IH-GAN). It learns the conditional distribution of unit
cell geometries given properties and can realize the one-to-many mapping from
geometry to properties. We further reduce the complexity of IH-GAN by using the
implicit function parameterization to represent unit cell geometries. Results
show that our method can 1) generate various unit cells that satisfy given
material properties with high accuracy (relative error <5%) and 2) improve the
optimized structural performance over the conventional topology-optimized
variable-density structure. Specifically, in the minimum compliance example,
our IH-GAN generated structure achieves an 84.4% reduction in concentrated
stress and an extra 7% reduction in displacement. In the target deformation
examples, our IH-GAN generated structure reduces the target matching error by
24.2% and 44.4% for two test cases, respectively. We also demonstrated that the
connectivity issue for multi-type unit cells can be solved by transition layer
blending.

    

### [[2103.04192] High Perceptual Quality Image Denoising with a Posterior Sampling CGAN](http://arxiv.org/abs/2103.04192)


  The vast work in Deep Learning (DL) has led to a leap in image denoising
research. Most DL solutions for this task have chosen to put their efforts on
the denoiser's architecture while maximizing distortion performance. However,
distortion driven solutions lead to blurry results with sub-optimal perceptual
quality, especially in immoderate noise levels. In this paper we propose a
different perspective, aiming to produce sharp and visually pleasing denoised
images that are still faithful to their clean sources. Formally, our goal is to
achieve high perceptual quality with acceptable distortion. This is attained by
a stochastic denoiser that samples from the posterior distribution, trained as
a generator in the framework of conditional generative adversarial networks
(CGAN). Contrary to distortion-based regularization terms that conflict with
perceptual quality, we introduce to the CGAN objective a theoretically founded
penalty term that does not force a distortion requirement on individual
samples, but rather on their mean. We showcase our proposed method with a novel
denoiser architecture that achieves the reformed denoising goal and produces
vivid and diverse outcomes in immoderate noise levels.

    

### [[2103.04807] PyRCN: A Toolbox for Exploration and Application of Reservoir Computing Networks](http://arxiv.org/abs/2103.04807)


  Reservoir Computing Networks belong to a group of machine learning techniques
that project the input space non-linearly into a high-dimensional feature
space, where the underlying task can be solved linearly. Popular variants of
RCNs, e.g.\ Extreme Learning Machines (ELMs), Echo State Networks (ESNs) and
Liquid State Machines (LSMs) are capable of solving complex tasks equivalently
to widely used deep neural networks, but with a substantially simpler training
paradigm based on linear regression. In this paper, we introduce the Python
toolbox PyRCN (Python Reservoir Computing Networks) for optimizing, training
and analyzing Reservoir Computing Networks (RCNs) on arbitrarily large
datasets. The tool is based on widely-used scientific packages, such as numpy
and scipy and complies with the scikit-learn interface specification. It
provides a platform for educational and exploratory analyses of RCNs, as well
as a framework to apply RCNs on complex tasks including sequence processing.
With only a small number of basic components, the framework allows the
implementation of a vast number of different RCN architectures. We provide
extensive code examples on how to set up RCNs for a time series prediction and
for a sequence classification task.

    

### [[2103.05456] Extended Tree Search for Robot Task and Motion Planning](http://arxiv.org/abs/2103.05456)


  Integrated task and motion planning (TAMP) is desirable for generalized
autonomy robots but it is challenging at the same time. TAMP requires the
planner to not only search in both the large symbolic task space and the
high-dimension motion space but also deal with the infeasible task actions due
to its intrinsic hierarchical process. We propose a novel decision-making
framework for TAMP by constructing an extended decision tree for both symbolic
task planning and high-dimension motion variable binding. We integrate top-k
planning for generating explicitly a skeleton space where a variety of
candidate skeleton plans are at disposal. Moreover, we effectively combine this
skeleton space with the resultant motion variable spaces into a single extended
decision space. Accordingly, we use Monte-Carlo Tree Search (MCTS) to ensure an
exploration-exploitation balance at each decision node and optimize globally to
produce optimal solutions. The proposed seamless combination of symbolic top-k
planning with streams, with the proved optimality of MCTS, leads to a powerful
planning algorithm that can handle the combinatorial complexity of long-horizon
manipulation tasks. We empirically evaluate our proposed algorithm in
challenging robot tasks with different domains that require multi-stage
decisions and show how our method can overcome the large task space and motion
space through its effective tree search compared to its most competitive
baseline method.

    

### [[2103.07945] Learning One Representation to Optimize All Rewards](http://arxiv.org/abs/2103.07945)


  We introduce the forward-backward (FB) representation of the dynamics of a
reward-free Markov decision process. It provides explicit near-optimal policies
for any reward specified a posteriori. During an unsupervised phase, we use
reward-free interactions with the environment to learn two representations via
off-the-shelf deep learning methods and temporal difference (TD) learning. In
the test phase, a reward representation is estimated either from observations
or an explicit reward description (e.g., a target state). The optimal policy
for that reward is directly obtained from these representations, with no
planning. We assume access to an exploration scheme or replay buffer for the
first phase.
The corresponding unsupervised loss is well-principled: if training is
perfect, the policies obtained are provably optimal for any reward function.
With imperfect training, the sub-optimality is proportional to the unsupervised
approximation error. The FB representation learns long-range relationships
between states and actions, via a predictive occupancy map, without having to
synthesize states as in model-based approaches.
This is a step towards learning controllable agents in arbitrary black-box
stochastic environments. This approach compares well to goal-oriented RL
algorithms on discrete and continuous mazes, pixel-based MsPacman, and the
FetchReach virtual robot arm. We also illustrate how the agent can immediately
adapt to new tasks beyond goal-oriented RL.

    

### [[2103.12019] Statistically-Robust Clustering Techniques for Mapping Spatial Hotspots: A Survey](http://arxiv.org/abs/2103.12019)


  Mapping of spatial hotspots, i.e., regions with significantly higher rates of
generating cases of certain events (e.g., disease or crime cases), is an
important task in diverse societal domains, including public health, public
safety, transportation, agriculture, environmental science, etc. Clustering
techniques required by these domains differ from traditional clustering methods
due to the high economic and social costs of spurious results (e.g., false
alarms of crime clusters). As a result, statistical rigor is needed explicitly
to control the rate of spurious detections. To address this challenge,
techniques for statistically-robust clustering (e.g., scan statistics) have
been extensively studied by the data mining and statistics communities. In this
survey we present an up-to-date and detailed review of the models and
algorithms developed by this field. We first present a general taxonomy for
statistically-robust clustering, covering key steps of data and statistical
modeling, region enumeration and maximization, and significance testing. We
further discuss different paradigms and methods within each of the key steps.
Finally, we highlight research gaps and potential future directions, which may
serve as a stepping stone in generating new ideas and thoughts in this growing
field and beyond.

    

### [[2103.12608] DIG: A Turnkey Library for Diving into Graph Deep Learning Research](http://arxiv.org/abs/2103.12608)


  Although there exist several libraries for deep learning on graphs, they are
aiming at implementing basic operations for graph deep learning. In the
research community, implementing and benchmarking various advanced tasks are
still painful and time-consuming with existing libraries. To facilitate graph
deep learning research, we introduce DIG: Dive into Graphs, a turnkey library
that provides a unified testbed for higher level, research-oriented graph deep
learning tasks. Currently, we consider graph generation, self-supervised
learning on graphs, explainability of graph neural networks, and deep learning
on 3D graphs. For each direction, we provide unified implementations of data
interfaces, common algorithms, and evaluation metrics. Altogether, DIG is an
extensible, open-source, and turnkey library for researchers to develop new
methods and effortlessly compare with common baselines using widely used
datasets and evaluation metrics. Source code is available at
this https URL.

    

### [[2103.12715] Promoting Fairness through Hyperparameter Optimization](http://arxiv.org/abs/2103.12715)


  Considerable research effort has been guided towards algorithmic fairness but
real-world adoption of bias reduction techniques is still scarce. Existing
methods are either metric- or model-specific, require access to sensitive
attributes at inference time, or carry high development or deployment costs.
This work explores the unfairness that emerges when optimizing ML models solely
for predictive performance, and how to mitigate it with a simple and easily
deployed intervention: fairness-aware hyperparameter optimization (HO). We
propose and evaluate fairness-aware variants of three popular HO algorithms:
Fair Random Search, Fair TPE, and Fairband. We validate our approach on a
real-world bank account opening fraud case-study, as well as on three datasets
from the fairness literature. Results show that, without extra training cost,
it is feasible to find models with 111% mean fairness increase and just 6%
decrease in performance when compared with fairness-blind HO.

    

### [[2103.14686] Generalization capabilities of translationally equivariant neural networks](http://arxiv.org/abs/2103.14686)


  The rising adoption of machine learning in high energy physics and lattice
field theory necessitates the re-evaluation of common methods that are widely
used in computer vision, which, when applied to problems in physics, can lead
to significant drawbacks in terms of performance and generalizability. One
particular example for this is the use of neural network architectures that do
not reflect the underlying symmetries of the given physical problem. In this
work, we focus on complex scalar field theory on a two-dimensional lattice and
investigate the benefits of using group equivariant convolutional neural
network architectures based on the translation group. For a meaningful
comparison, we conduct a systematic search for equivariant and non-equivariant
neural network architectures and apply them to various regression and
classification tasks. We demonstrate that in most of these tasks our best
equivariant architectures can perform and generalize significantly better than
their non-equivariant counterparts, which applies not only to physical
parameters beyond those represented in the training set, but also to different
lattice sizes.

    

### [[2103.15432] Towards High Fidelity Monocular Face Reconstruction with Rich Reflectance using Self-supervised Learning and Ray Tracing](http://arxiv.org/abs/2103.15432)


  Robust face reconstruction from monocular image in general lighting
conditions is challenging. Methods combining deep neural network encoders with
differentiable rendering have opened up the path for very fast monocular
reconstruction of geometry, lighting and reflectance. They can also be trained
in self-supervised manner for increased robustness and better generalization.
However, their differentiable rasterization based image formation models, as
well as underlying scene parameterization, limit them to Lambertian face
reflectance and to poor shape details. More recently, ray tracing was
introduced for monocular face reconstruction within a classic
optimization-based framework and enables state-of-the art results. However
optimization-based approaches are inherently slow and lack robustness. In this
paper, we build our work on the aforementioned approaches and propose a new
method that greatly improves reconstruction quality and robustness in general
scenes. We achieve this by combining a CNN encoder with a differentiable ray
tracer, which enables us to base the reconstruction on much more advanced
personalized diffuse and specular albedos, a more sophisticated illumination
model and a plausible representation of self-shadows. This enables to take a
big leap forward in reconstruction quality of shape, appearance and lighting
even in scenes with difficult illumination. With consistent face attributes
reconstruction, our method leads to practical applications such as relighting
and self-shadows removal. Compared to state-of-the-art methods, our results
show improved accuracy and validity of the approach.

    

### [[2104.00931] Assem-VC: Realistic Voice Conversion by Assembling Modern Speech Synthesis Techniques](http://arxiv.org/abs/2104.00931)


  Recent works on voice conversion (VC) focus on preserving the rhythm and the
intonation as well as the linguistic content. To preserve these features from
the source, we decompose current non-parallel VC systems into two encoders and
one decoder. We analyze each module with several experiments and reassemble the
best components to propose Assem-VC, a new state-of-the-art any-to-many
non-parallel VC system. We also examine that PPG and Cotatron features are
speaker-dependent, and attempt to remove speaker identity with adversarial
training. Code and audio samples are available at
this https URL.

    

### [[2104.01769] Procrustean Training for Imbalanced Deep Learning](http://arxiv.org/abs/2104.01769)


  Neural networks trained with class-imbalanced data are known to perform
poorly on minor classes of scarce training data. Several recent works attribute
this to over-fitting to minor classes. In this paper, we provide a novel
explanation of this issue. We found that a neural network tends to first
under-fit the minor classes by classifying most of their data into the major
classes in early training epochs. To correct these wrong predictions, the
neural network then must focus on pushing features of minor class data across
the decision boundaries between major and minor classes, leading to much larger
gradients for features of minor classes. We argue that such an under-fitting
phase over-emphasizes the competition between major and minor classes, hinders
the neural network from learning the discriminative knowledge that can be
generalized to test data, and eventually results in over-fitting. To address
this issue, we propose a novel learning strategy to equalize the training
progress across classes. We mix features of the major class data with those of
other data in a mini-batch, intentionally weakening their features to prevent a
neural network from fitting them first. We show that this strategy can largely
balance the training accuracy and feature gradients across classes, effectively
mitigating the under-fitting then over-fitting problem for minor class data. On
several benchmark datasets, our approach achieves the state-of-the-art
accuracy, especially for the challenging step-imbalanced cases.

    

### [[2104.06378] QA-GNN: Reasoning with Language Models and Knowledge Graphs for Question Answering](http://arxiv.org/abs/2104.06378)


  The problem of answering questions using knowledge from pre-trained language
models (LMs) and knowledge graphs (KGs) presents two challenges: given a QA
context (question and answer choice), methods need to (i) identify relevant
knowledge from large KGs, and (ii) perform joint reasoning over the QA context
and KG. In this work, we propose a new model, QA-GNN, which addresses the above
challenges through two key innovations: (i) relevance scoring, where we use LMs
to estimate the importance of KG nodes relative to the given QA context, and
(ii) joint reasoning, where we connect the QA context and KG to form a joint
graph, and mutually update their representations through graph neural networks.
We evaluate QA-GNN on the CommonsenseQA and OpenBookQA datasets, and show its
improvement over existing LM and LM+KG models, as well as its capability to
perform interpretable and structured reasoning, e.g., correctly handling
negation in questions.

    

### [[2104.08869] Ranking Structured Objects with Graph Neural Networks](http://arxiv.org/abs/2104.08869)


  Graph neural networks (GNNs) have been successfully applied in many
structured data domains, with applications ranging from molecular property
prediction to the analysis of social networks. Motivated by the broad
applicability of GNNs, we propose the family of so-called RankGNNs, a
combination of neural Learning to Rank (LtR) methods and GNNs. RankGNNs are
trained with a set of pair-wise preferences between graphs, suggesting that one
of them is preferred over the other. One practical application of this problem
is drug screening, where an expert wants to find the most promising molecules
in a large collection of drug candidates. We empirically demonstrate that our
proposed pair-wise RankGNN approach either significantly outperforms or at
least matches the ranking performance of the naive point-wise baseline
approach, in which the LtR problem is solved via GNN-based graph regression.

    

### [[2104.09864] RoFormer: Enhanced Transformer with Rotary Position Embedding](http://arxiv.org/abs/2104.09864)


  Position encoding in transformer architecture provides supervision for
dependency modeling between elements at different positions in the sequence. We
investigate various methods to encode positional information in
transformer-based language models and propose a novel implementation named
Rotary Position Embedding(RoPE). The proposed RoPE encodes absolute positional
information with rotation matrix and naturally incorporates explicit relative
position dependency in self-attention formulation. Notably, RoPE comes with
valuable properties such as flexibility of being expand to any sequence
lengths, decaying inter-token dependency with increasing relative distances,
and capability of equipping the linear self-attention with relative position
encoding. As a result, the enhanced transformer with rotary position embedding,
or RoFormer, achieves superior performance in tasks with long texts. We release
the theoretical analysis along with some preliminary experiment results on
Chinese data. The undergoing experiment for English benchmark will soon be
updated.

    

### [[2104.11846] Joint Detection and Localization of Stealth False Data Injection Attacks in Smart Grids using Graph Neural Networks](http://arxiv.org/abs/2104.11846)


  False data injection attacks (FDIA) are a main category of cyber-attacks
threatening the security of power systems. Contrary to the detection of these
attacks, less attention has been paid to identifying the attacked units of the
grid. To this end, this work jointly studies detecting and localizing the
stealth FDIA in power grids. Exploiting the inherent graph topology of power
systems as well as the spatial correlations of measurement data, this paper
proposes an approach based on the graph neural network (GNN) to identify the
presence and location of the FDIA. The proposed approach leverages the
auto-regressive moving average (ARMA) type graph filters (GFs) which can better
adapt to sharp changes in the spectral domain due to their rational type filter
composition compared to the polynomial type GFs such as Chebyshev. To the best
of our knowledge, this is the first work based on GNN that automatically
detects and localizes FDIA in power systems. Extensive simulations and
visualizations show that the proposed approach outperforms the available
methods in both detection and localization of FDIA for different IEEE test
systems. Thus, the targeted areas can be identified and preventive actions can
be taken before the attack impacts the grid.

    

### [[2104.12657] tsrobprep - an R package for robust preprocessing of time series data](http://arxiv.org/abs/2104.12657)


  Data cleaning is a crucial part of every data analysis exercise. Yet, the
currently available R packages do not provide fast and robust methods for
cleaning and preparation of time series data. The open source package tsrobprep
introduces efficient methods for handling missing values and outliers using
model based approaches. For data imputation a probabilistic replacement model
is proposed, which may consist of autoregressive components and external
inputs. For outlier detection a clustering algorithm based on finite mixture
modelling is introduced, which considers time series properties in terms of the
gradient and the underlying seasonality as features. The procedure allows to
return a probability for each observation being outlying data as well as a
specific cause for an outlier assignment in terms of the provided feature
space. The methods work robust and are fully tunable. Moreover, by providing
the auto_data_cleaning function the data preprocessing can be carried out in
one cast, without comprehensive tuning and providing suitable results. The
primary motivation of the package is the preprocessing of energy system data.
We present application for electricity load, wind and solar power data.

    

### [[2104.15061] Black-box Gradient Attack on Graph Neural Networks: Deeper Insights in Graph-based Attack and Defense](http://arxiv.org/abs/2104.15061)


  Graph Neural Networks (GNNs) have received significant attention due to their
state-of-the-art performance on various graph representation learning tasks.
However, recent studies reveal that GNNs are vulnerable to adversarial attacks,
i.e. an attacker is able to fool the GNNs by perturbing the graph structure or
node features deliberately. While being able to successfully decrease the
performance of GNNs, most existing attacking algorithms require access to
either the model parameters or the training data, which is not practical in the
real world.
In this paper, we develop deeper insights into the Mettack algorithm, which
is a representative grey-box attacking method, and then we propose a
gradient-based black-box attacking algorithm. Firstly, we show that the Mettack
algorithm will perturb the edges unevenly, thus the attack will be highly
dependent on a specific training set. As a result, a simple yet useful strategy
to defense against Mettack is to train the GNN with the validation set.
Secondly, to overcome the drawbacks, we propose the Black-Box Gradient Attack
(BBGA) algorithm. Extensive experiments demonstrate that out proposed method is
able to achieve stable attack performance without accessing the training sets
of the GNNs. Further results shows that our proposed method is also applicable
when attacking against various defense methods.

    

### [[2105.03863] Towards Theoretical Understandings of Robust Markov Decision Processes: Sample Complexity and Asymptotics](http://arxiv.org/abs/2105.03863)


  In this paper, we study the non-asymptotic and asymptotic performances of the
optimal robust policy and value function of robust Markov Decision
Processes(MDPs), where the optimal robust policy and value function are solved
only from a generative model. While prior work focusing on non-asymptotic
performances of robust MDPs is restricted in the setting of the KL uncertainty
set and $(s,a)$-rectangular assumption, we improve their results and also
consider other uncertainty sets, including $L_1$ and $\chi^2$ balls. Our
results show that when we assume $(s,a)$-rectangular on uncertainty sets, the
sample complexity is about
$\widetilde{O}\left(\frac{|\mathcal{S}|^2|\mathcal{A}|}{\varepsilon^2\rho^2(1-\gamma)^4}\right)$.
In addition, we extend our results from $(s,a)$-rectangular assumption to
$s$-rectangular assumption. In this scenario, the sample complexity varies with
the choice of uncertainty sets and is generally larger than the case under
$(s,a)$-rectangular assumption. Moreover, we also show that the optimal robust
value function is asymptotic normal with a typical rate $\sqrt{n}$ under
$(s,a)$ and $s$-rectangular assumptions from both theoretical and empirical
perspectives.

    

### [[2105.04544] Proximal Causal Learning with Kernels: Two-Stage Estimation and Moment Restriction](http://arxiv.org/abs/2105.04544)


  We address the problem of causal effect estimation in the presence of
unobserved confounding, but where proxies for the latent confounder(s) are
observed. We propose two kernel-based methods for nonlinear causal effect
estimation in this setting: (a) a two-stage regression approach, and (b) a
maximum moment restriction approach. We focus on the proximal causal learning
setting, but our methods can be used to solve a wider class of inverse problems
characterised by a Fredholm integral equation. In particular, we provide a
unifying view of two-stage and moment restriction approaches for solving this
problem in a nonlinear setting. We provide consistency guarantees for each
algorithm, and we demonstrate these approaches achieve competitive results on
synthetic data and data simulating a real-world task. In particular, our
approach outperforms earlier methods that are not suited to leveraging proxy
variables.

    

### [[2105.07510] Doc2Dict: Information Extraction as Text Generation](http://arxiv.org/abs/2105.07510)


  Typically, information extraction (IE) requires a pipeline approach: first, a
sequence labeling model is trained on manually annotated documents to extract
relevant spans; then, when a new document arrives, a model predicts spans which
are then post-processed and standardized to convert the information into a
database entry. We replace this labor-intensive workflow with a transformer
language model trained on existing database records to directly generate
structured JSON. Our solution removes the workload associated with producing
token-level annotations and takes advantage of a data source which is generally
quite plentiful (e.g. database records). As long documents are common in
information extraction tasks, we use gradient checkpointing and chunked
encoding to apply our method to sequences of up to 32,000 tokens on a single
GPU. Our Doc2Dict approach is competitive with more complex, hand-engineered
pipelines and offers a simple but effective baseline for document-level
information extraction. We release our Doc2Dict model and code to reproduce our
experiments and facilitate future work.

    

### [[2105.08769] Learning and Information in Stochastic Networks and Queues](http://arxiv.org/abs/2105.08769)


  We review the role of information and learning in the stability and
optimization of queueing systems. In recent years, techniques from supervised
learning, bandit learning and reinforcement learning have been applied to
queueing systems supported by increasing role of information in decision
making. We present observations and new results that help rationalize the
application of these areas to queueing systems.
We prove that the MaxWeight and BackPressure policies are an application of
Blackwell's Approachability Theorem. This connects queueing theoretic results
with adversarial learning. We then discuss the requirements of statistical
learning for service parameter estimation. As an example, we show how queue
size regret can be bounded when applying a perceptron algorithm to classify
service. Next, we discuss the role of state information in improved decision
making. Here we contrast the roles of epistemic information (information on
uncertain parameters) and aleatoric information (information on an uncertain
state). Finally we review recent advances in the theory of reinforcement
learning and queueing, as well as, provide discussion on current research
challenges.

    

### [[2105.10360] Multi-source Learning via Completion of Block-wise Overlapping Noisy Matrices](http://arxiv.org/abs/2105.10360)


  Matrix completion has attracted attention in many fields, including
statistics, applied mathematics, and electrical engineering. Most of the works
focus on the independent sampling models under which the observed entries are
sampled independently. Motivated by applications in the integration of
knowledge graphs derived from multi-source biomedical data such as those from
Electronic Health Records (EHR) and biomedical text, we propose the {\bf
B}lock-wise {\bf O}verlapping {\bf N}oisy {\bf M}atrix {\bf I}ntegration
(BONMI) to treat blockwise missingness of symmetric matrices representing
relatedness between entity pairs. Our idea is to exploit the orthogonal
Procrustes problem to align the eigenspace of the two sub-matrices, then
complete the missing blocks by the inner product of the two low-rank
components. Besides, we prove the statistical rate for the eigenspace of the
underlying matrix, which is comparable to the rate under the independently
missing assumption. Simulation studies show that the method performs well under
a variety of configurations. In the real data analysis, the method is applied
to two tasks: (i) the integrating of several point-wise mutual information
matrices built by English EHR and Chinese medical text data, and (ii) the
machine translation between English and Chinese medical concepts. Our method
shows an advantage over existing methods.

    

### [[2105.12584] A Comprehensive Survey on Community Detection with Deep Learning](http://arxiv.org/abs/2105.12584)


  A community reveals the features and connections of its members that are
different from those in other communities in a network. Detecting communities
is of great significance in network analysis. Despite the classical spectral
clustering and statistical inference methods, we notice a significant
development of deep learning techniques for community detection in recent years
with their advantages in handling high dimensional network data. Hence, a
comprehensive overview of community detection's latest progress through deep
learning is timely to academics and practitioners. This survey devises and
proposes a new taxonomy covering different state-of-the-art methods, including
deep learning-based models upon deep neural networks, deep nonnegative matrix
factorization and deep sparse filtering. The main category, i.e., deep neural
networks, is further divided into convolutional networks, graph attention
networks, generative adversarial networks and autoencoders. The survey also
summarizes the popular benchmark data sets, evaluation metrics, and open-source
implementations to address experimentation settings. We then discuss the
practical applications of community detection in various domains and point to
implementation scenarios. Finally, we outline future directions by suggesting
challenging topics in this fast-growing deep learning field.

    

### [[2105.14275] Greedy Bayesian Posterior Approximation with Deep Ensembles](http://arxiv.org/abs/2105.14275)


  Ensembles of independently trained neural networks are a state-of-the-art
approach to estimate predictive uncertainty in Deep Learning, and can be
interpreted as an approximation of the posterior distribution via a mixture of
delta functions. The training of ensembles relies on non-convexity of the loss
landscape and random initialization of their individual members, making the
resulting posterior approximation uncontrolled. This paper proposes a novel and
principled method to tackle this limitation, minimizing an $f$-divergence
between the true posterior and a kernel density estimator in a function space.
We analyze this objective from a combinatorial point of view, and show that it
is submodular with respect to mixture components for any $f$. Subsequently, we
consider the problem of ensemble construction, and from the marginal gain of
the total objective, we derive a novel diversity term for training ensembles
greedily. The performance of our approach is demonstrated on computer vision
out-of-distribution detection benchmarks in a range of architectures trained on
multiple datasets. The source code of our method is publicly available at
this https URL.

    

### [[2105.14491] How Attentive are Graph Attention Networks?](http://arxiv.org/abs/2105.14491)


  Graph Attention Networks (GATs) are one of the most popular GNN architectures
and are considered as the state-of-the-art architecture for representation
learning with graphs. In GAT, every node attends to its neighbors given its own
representation as the query. However, in this paper we show that GAT computes a
very limited kind of attention: the ranking of the attention scores is
unconditioned on the query node. We formally define this restricted kind of
attention as static attention and distinguish it from a strictly more
expressive dynamic attention. Because GATs use a static attention mechanism,
there are simple graph problems that GAT cannot express: in a controlled
problem, we show that static attention hinders GAT from even fitting the
training data. To remove this limitation, we introduce a simple fix by
modifying the order of operations and propose GATv2: a dynamic graph attention
variant that is strictly more expressive than GAT. We perform an extensive
evaluation and show that GATv2 outperforms GAT across 11 OGB and other
benchmarks while we match their parametric costs. Our code is available at
this https URL , and GATv2 is available as
part of the PyTorch Geometric library.

    

### [[2105.15186] Fast Policy Extragradient Methods for Competitive Games with Entropy Regularization](http://arxiv.org/abs/2105.15186)


  This paper investigates the problem of computing the equilibrium of
competitive games, which is often modeled as a constrained saddle-point
optimization problem with probability simplex constraints. Despite recent
efforts in understanding the last-iterate convergence of extragradient methods
in the unconstrained setting, the theoretical underpinnings of these methods in
the constrained settings, especially those using multiplicative updates, remain
highly inadequate, even when the objective function is bilinear. Motivated by
the algorithmic role of entropy regularization in single-agent reinforcement
learning and game theory, we develop provably efficient extragradient methods
to find the quantal response equilibrium (QRE) -- which are solutions to
zero-sum two-player matrix games with entropy regularization -- at a linear
rate. The proposed algorithms can be implemented in a decentralized manner,
where each player executes symmetric and multiplicative updates iteratively
using its own payoff without observing the opponent's actions directly. In
addition, by controlling the knob of entropy regularization, the proposed
algorithms can locate an approximate Nash equilibrium of the unregularized
matrix game at a sublinear rate without assuming the Nash equilibrium to be
unique. Our methods also lead to efficient policy extragradient algorithms for
solving (entropy-regularized) zero-sum Markov games at similar rates. All of
our convergence rates are nearly dimension-free, which are independent of the
size of the state and action spaces up to logarithm factors, highlighting the
positive role of entropy regularization for accelerating convergence.

    

### [[2106.00110] A Methodology for Exploring Deep Convolutional Features in Relation to Hand-Crafted Features with an Application to Music Audio Modeling](http://arxiv.org/abs/2106.00110)


  Understanding the features learned by deep models is important from a model
trust perspective, especially as deep systems are deployed in the real world.
Most recent approaches for deep feature understanding or model explanation
focus on highlighting input data features that are relevant for classification
decisions. In this work, we instead take the perspective of relating deep
features to well-studied, hand-crafted features that are meaningful for the
application of interest. We propose a methodology and set of systematic
experiments for exploring deep features in this setting, where input feature
importance approaches for deep feature understanding do not apply. Our
experiments focus on understanding which hand-crafted and deep features are
useful for the classification task of interest, how robust these features are
for related tasks and how similar the deep features are to the meaningful
hand-crafted features. Our proposed method is general to many application areas
and we demonstrate its utility on orchestral music audio data.

    

### [[2106.01098] Evaluation Metrics for Graph Generative Models: Problems, Pitfalls, and Practical Solutions](http://arxiv.org/abs/2106.01098)


  Graph generative models are a highly active branch of machine learning. Given
the steady development of new models of ever-increasing complexity, it is
necessary to provide a principled way to evaluate and compare them. In this
paper, we enumerate the desirable criteria for such a comparison metric and
provide an overview of the status quo of graph generative model comparison in
use today, which predominantly relies on maximum mean discrepancy (MMD). We
perform a systematic evaluation of MMD in the context of graph generative model
comparison, highlighting some of the challenges and pitfalls researchers
inadvertently may encounter. After conducting a thorough analysis of the
behaviour of MMD on synthetically-generated perturbed graphs as well as on
recently-proposed graph generative models, we are able to provide a suitable
procedure to mitigate these challenges and pitfalls. We aggregate our findings
into a list of practical recommendations for researchers to use when evaluating
graph generative models.

    

### [[2106.01883] Learning High-Precision Bounding Box for Rotated Object Detection via Kullback-Leibler Divergence](http://arxiv.org/abs/2106.01883)


  Existing rotated object detectors are mostly inherited from the horizontal
detection paradigm, as the latter has evolved into a well-developed area.
However, these detectors are difficult to perform prominently in high-precision
detection due to the limitation of current regression loss design, especially
for objects with large aspect ratios. Taking the perspective that horizontal
detection is a special case for rotated object detection, in this paper, we are
motivated to change the design of rotation regression loss from induction
paradigm to deduction methodology, in terms of the relation between rotation
and horizontal detection. We show that one essential challenge is how to
modulate the coupled parameters in the rotation regression loss, as such the
estimated parameters can influence to each other during the dynamic joint
optimization, in an adaptive and synergetic way. Specifically, we first convert
the rotated bounding box into a 2-D Gaussian distribution, and then calculate
the Kullback-Leibler Divergence (KLD) between the Gaussian distributions as the
regression loss. By analyzing the gradient of each parameter, we show that KLD
(and its derivatives) can dynamically adjust the parameter gradients according
to the characteristics of the object. It will adjust the importance (gradient
weight) of the angle parameter according to the aspect ratio. This mechanism
can be vital for high-precision detection as a slight angle error would cause a
serious accuracy drop for large aspect ratios objects. More importantly, we
have proved that KLD is scale invariant. We further show that the KLD loss can
be degenerated into the popular $l_{n}$-norm loss for horizontal detection.
Experimental results on seven datasets using different detectors show its
consistent superiority, and codes are available at
this https URL.

    

### [[2106.02528] Neural Network Surrogate Models for Absorptivity and Emissivity Spectra of Multiple Elements](http://arxiv.org/abs/2106.02528)


  Simulations of high energy density physics are expensive in terms of
computational resources. In particular, the computation of opacities of plasmas
in the non-local thermal equilibrium (NLTE) regime can consume as much as 90\%
of the total computational time of radiation hydrodynamics simulations for high
energy density physics applications. Previous work has demonstrated that a
combination of fully-connected autoencoders and a deep jointly-informed neural
network (DJINN) can successfully replace the standard NLTE calculations for the
opacity of krypton. This work expands this idea to combining multiple elements
into a single surrogate model with the focus here being on the autoencoder.

    

### [[2106.02693] Two-Sample Tests that are Safe under Optional Stopping, with an Application to Contingency Tables](http://arxiv.org/abs/2106.02693)


  We develop E variables for testing whether two data streams come from the
same source or not, and more generally, whether the difference between the
sources is larger than some minimal effect size. These E variables lead to
tests that remain safe, i.e. keep their Type-I error guarantees, under flexible
sampling scenarios such as optional stopping and continuation. In special cases
our E variables also have an optimal `growth' property under the alternative.
We illustrate the generic construction through the special case of 2x2
contingency tables, where we also allow for the incorporation of different
restrictions on a composite alternative. Comparison to p-value analysis in
simulations and a real-world example show that E variables, through their
flexibility, often allow for early stopping of data collection, thereby
retaining similar power as classical methods.

    

### [[2106.04217] Dynamic Sparse Training for Deep Reinforcement Learning](http://arxiv.org/abs/2106.04217)


  Dynamic sparse training (DST) literature demonstrates that a highly sparse
neural network can match the performance of its corresponding dense network in
supervised and unsupervised learning when it is trained from scratch while
substantially reducing the computational and memory costs. In this paper, we
show for the first time that deep reinforcement learning can also benefit from
dynamic sparse training. We demonstrate that DST can be leveraged to decrease
the long training time required by deep reinforcement learning agents without
sacrificing performance. To achieve this, we propose a DST algorithm that
adapts to the online nature and instability of the deep reinforcement learning
paradigm. We integrate our proposed algorithm with state-of-the-art deep
reinforcement learning methods. Experimental results demonstrate that our
dynamic sparse compact agents can effectively learn and achieve higher
performance than the original dense methods while reducing the parameter count
and floating-point operations (FLOPs) by 50%. More impressively, our dynamic
sparse agents have a faster learning speed. They can reach the final
performance achieved by dense agents after 40-50% of the steps required by the
latter. We evaluate our approach on OpenAI gym continuous control tasks.

    

### [[2106.04258] Interpretable agent communication from scratch (with a generic visual processor emerging on the side)](http://arxiv.org/abs/2106.04258)


  As deep networks begin to be deployed as autonomous agents, the issue of how
they can communicate with each other becomes important. Here, we train two deep
nets from scratch to perform realistic referent identification through
unsupervised emergent communication. We show that the largely interpretable
emergent protocol allows the nets to successfully communicate even about object
types they did not see at training time. The visual representations induced as
a by-product of our training regime, moreover, show comparable quality, when
re-used as generic visual features, to a recent self-supervised learning model.
Our results provide concrete evidence of the viability of (interpretable)
emergent deep net communication in a more realistic scenario than previously
considered, as well as establishing an intriguing link between this field and
self-supervised visual learning.

    

### [[2106.04693] On the Evolution of Neuron Communities in a Deep Learning Architecture](http://arxiv.org/abs/2106.04693)


  Deep learning techniques are increasingly being adopted for classification
tasks over the past decade, yet explaining how deep learning architectures can
achieve state-of-the-art performance is still an elusive goal. While all the
training information is embedded deeply in a trained model, we still do not
understand much about its performance by only analyzing the model. This paper
examines the neuron activation patterns of deep learning-based classification
models and explores whether the models' performances can be explained through
neurons' activation behavior. We propose two approaches: one that models
neurons' activation behavior as a graph and examines whether the neurons form
meaningful communities, and the other examines the predictability of neurons'
behavior using entropy. Our comprehensive experimental study reveals that both
the community quality and entropy can provide new insights into the deep
learning models' performances, thus paves a novel way of explaining deep
learning models directly from the neurons' activation pattern.

    

### [[2106.04700] Scale Free Adversarial Multi Armed Bandits](http://arxiv.org/abs/2106.04700)


  We consider the Scale-Free Adversarial Multi Armed Bandits(MAB) problem. At
the beginning of the game, the player only knows the number of arms $n$. It
does not know the scale and magnitude of the losses chosen by the adversary or
the number of rounds $T$. In each round, it sees bandit feedback about the loss
vectors $l_1,\dots, l_T \in \mathbb{R}^n$. The goal is to bound its regret as a
function of $n$ and norms of $l_1,\dots, l_T$. We design a bandit Follow The
Regularized Leader (FTRL) algorithm, that uses an adaptive learning rate and
give two different regret bounds, based on the exploration parameter used. With
non-adaptive exploration, our algorithm has a regret of
$\tilde{\mathcal{O}}(\sqrt{nL_2} + L_\infty\sqrt{nT})$ and with adaptive
exploration, it has a regret of $\tilde{\mathcal{O}}(\sqrt{nL_2} +
L_\infty\sqrt{nL_1})$. Here $L_\infty = \sup_t \| l_t\|_\infty$, $L_2 =
\sum_{t=1}^T \|l_t\|_2^2$, $L_1 = \sum_{t=1}^T \|l_t\|_1$ and the
$\tilde{\mathcal{O}}$ notation suppress logarithmic factors. These are the
first MAB bounds that adapt to the $\|\cdot\|_2$, $\|\cdot\|_1$ norms of the
losses. The second bound is the first data-dependent scale-free MAB bound as
$T$ does not directly appear in the regret. We also develop a new technique for
obtaining a rich class of local-norm lower-bounds for Bregman Divergences. This
technique plays a crucial role in our analysis for controlling the regret when
using importance weighted estimators of unbounded losses. This technique could
be of independent interest.

    

### [[2106.06153] Towards Understanding Generalization via Decomposing Excess Risk Dynamics](http://arxiv.org/abs/2106.06153)


  Generalization is one of the fundamental issues in machine learning. However,
traditional techniques like uniform convergence may be unable to explain
generalization under overparameterization. As alternative approaches,
techniques based on \emph{stability} analyze the training dynamics and drive
algorithm-dependent generalization bounds. Unfortunately, the stability-based
bounds are still far from explaining the surprising generalization in deep
learning since neural networks usually suffer from unsatisfactory stability.
This paper proposes a novel decomposition framework to improve the
stability-based bounds via a more fine-grained analysis of the signal and
noise, inspired by the observation that neural networks converge relatively
slowly when fitting noise (which indicates better stability). Concretely, we
decompose the excess risk dynamics and apply stability-based bound only on the
noise component. The decomposition framework performs well in both linear
regimes (overparameterized linear regression) and non-linear regimes (diagonal
matrix recovery). Experiments on neural networks verify the utility of the
decomposition framework.

    

### [[2106.06257] HPO-B: A Large-Scale Reproducible Benchmark for Black-Box HPO based on OpenML](http://arxiv.org/abs/2106.06257)


  Hyperparameter optimization (HPO) is a core problem for the machine learning
community and remains largely unsolved due to the significant computational
resources required to evaluate hyperparameter configurations. As a result, a
series of recent related works have focused on the direction of transfer
learning for quickly fine-tuning hyperparameters on a dataset. Unfortunately,
the community does not have a common large-scale benchmark for comparing HPO
algorithms. Instead, the de facto practice consists of empirical protocols on
arbitrary small-scale meta-datasets that vary inconsistently across
publications, making reproducibility a challenge. To resolve this major
bottleneck and enable a fair and fast comparison of black-box HPO methods on a
level playing field, we propose HPO-B, a new large-scale benchmark in the form
of a collection of meta-datasets. Our benchmark is assembled and preprocessed
from the OpenML repository and consists of 176 search spaces (algorithms)
evaluated sparsely on 196 datasets with a total of 6.4 million hyperparameter
evaluations. For ensuring reproducibility on our benchmark, we detail explicit
experimental protocols, splits, and evaluation measures for comparing methods
for both non-transfer, as well as, transfer learning HPO.

    

### [[2106.06733] LE-NAS: Learning-based Ensemble with NAS for Dose Prediction](http://arxiv.org/abs/2106.06733)


  Radiation therapy treatment planning is a complex process, as the target dose
prescription and normal tissue sparing are conflicting objectives. Automated
and accurate dose prediction for radiation therapy planning is in high demand.
In this study, we propose a novel learning-based ensemble approach, named
LE-NAS, which integrates neural architecture search (NAS) with knowledge
distillation for 3D radiotherapy dose prediction. Specifically, the prediction
network first exhaustively searches each block from enormous architecture
space. Then, multiple architectures are selected with promising performance and
diversity. To reduce the inference time, we adopt the teacher-student paradigm
by treating the combination of diverse outputs from multiple searched networks
as supervisions to guide the student network training. In addition, we apply
adversarial learning to optimize the student network to recover the knowledge
in teacher networks. To the best of our knowledge, we are the first to
investigate the combination of NAS and knowledge distillation. The proposed
method has been evaluated on the public OpenKBP dataset, and experimental
results demonstrate the effectiveness of our method and its superior
performance to the state-of-the-art method.

    

### [[2106.07677] Planning to Fairly Allocate: Probabilistic Fairness in the Restless Bandit Setting](http://arxiv.org/abs/2106.07677)


  Restless and collapsing bandits are commonly used to model constrained
resource allocation in settings featuring arms with action-dependent transition
probabilities, such as the allocation of health interventions among patients
[Whittle, 1988; Mate et al., 2020]. However, state-of-the-art
Whittle-index-based approaches to this planning problem either do not consider
fairness among arms or incentivize fairness without guaranteeing it [Mate et
al., 2021]. Additionally, their optimality guarantees only apply when arms are
indexable and threshold-optimal. We demonstrate that the incorporation of hard
fairness constraints necessitates the coupling of arms, which undermines the
tractability, and by extension, indexability of the problem. We then introduce
ProbFair, a probabilistically fair stationary policy that maximizes total
expected reward and satisfies the budget constraint, while ensuring a strictly
positive lower bound on the probability of being pulled at each timestep. We
evaluate our algorithm on a real-world application, where interventions support
continuous positive airway pressure (CPAP) therapy adherence among obstructive
sleep apnea (OSA) patients, as well as on a broader class of synthetic
transition matrices.

    

### [[2106.08315] Decentralized Local Stochastic Extra-Gradient for Variational Inequalities](http://arxiv.org/abs/2106.08315)


  We consider distributed stochastic variational inequalities (VIs) on
unbounded domain with the problem data being heterogeneous (non-IID) and
distributed across many devices. We make very general assumption on the
computational network that, in particular, covers the settings of fully
decentralized calculations with time-varying networks and centralized
topologies commonly used in Federated Learning. Moreover, multiple local
updates on the workers can be made for reducing the communication frequency
between workers. We extend stochastic extragradient method to this very general
setting and theoretically analyze its convergence rate in the strongly
monotone, monotone, and non-monotone setting when an Minty solution exists. The
provided rates have explicit dependence on\ network characteristics and how it
varies with time, data heterogeneity, variance, number of devices, and other
standard parameters. As a special case, our method and analysis apply to
distributed stochastic saddle-point problems (SPP), e.g., to training Deep
Generative Adversarial Networks (GANs) for which the decentralized training has
been reported to be extremely challenging. In experiments for decentralized
training of GANs we demonstrate the effectiveness of our proposed approach.

    

### [[2106.08858] Grounding Spatio-Temporal Language with Transformers](http://arxiv.org/abs/2106.08858)


  Language is an interface to the outside world. In order for embodied agents
to use it, language must be grounded in other, sensorimotor modalities. While
there is an extended literature studying how machines can learn grounded
language, the topic of how to learn spatio-temporal linguistic concepts is
still largely uncharted. To make progress in this direction, we here introduce
a novel spatio-temporal language grounding task where the goal is to learn the
meaning of spatio-temporal descriptions of behavioral traces of an embodied
agent. This is achieved by training a truth function that predicts if a
description matches a given history of observations. The descriptions involve
time-extended predicates in past and present tense as well as spatio-temporal
references to objects in the scene. To study the role of architectural biases
in this task, we train several models including multimodal Transformer
architectures; the latter implement different attention computations between
words and objects across space and time. We test models on two classes of
generalization: 1) generalization to randomly held-out sentences; 2)
generalization to grammar primitives. We observe that maintaining object
identity in the attention computation of our Transformers is instrumental to
achieving good performance on generalization overall, and that summarizing
object traces in a single token has little influence on performance. We then
discuss how this opens new perspectives for language-guided autonomous embodied
agents. We also release our code under open-source license as well as
pretrained models and datasets to encourage the wider community to build upon
and extend our work in the future.

    

### [[2106.09408] Predicting cognitive scores with graph neural networks through sample selection learning](http://arxiv.org/abs/2106.09408)


  Analyzing the relation between intelligence and neural activity is of the
utmost importance in understanding the working principles of the human brain in
health and disease. In existing literature, functional brain connectomes have
been used successfully to predict cognitive measures such as intelligence
quotient (IQ) scores in both healthy and disordered cohorts using machine
learning models. However, existing methods resort to flattening the brain
connectome (i.e., graph) through vectorization which overlooks its topological
properties. To address this limitation and inspired from the emerging graph
neural networks (GNNs), we design a novel regression GNN model (namely RegGNN)
for predicting IQ scores from brain connectivity. On top of that, we introduce
a novel, fully modular sample selection method to select the best samples to
learn from for our target prediction task. However, since such deep learning
architectures are computationally expensive to train, we further propose a
\emph{learning-based sample selection} method that learns how to choose the
training samples with the highest expected predictive power on unseen samples.
For this, we capitalize on the fact that connectomes (i.e., their adjacency
matrices) lie in the symmetric positive definite (SPD) matrix cone. Our results
on full-scale and verbal IQ prediction outperforms comparison methods in autism
spectrum disorder cohorts and achieves a competitive performance for
neurotypical subjects using 3-fold cross-validation. Furthermore, we show that
our sample selection approach generalizes to other learning-based methods,
which shows its usefulness beyond our GNN architecture.

    

### [[2106.10163] Steerable Partial Differential Operators for Equivariant Neural Networks](http://arxiv.org/abs/2106.10163)


  Recent work in equivariant deep learning bears strong similarities to
physics. Fields over a base space are fundamental entities in both subjects, as
are equivariant maps between these fields. In deep learning, however, these
maps are usually defined by convolutions with a kernel, whereas they are
partial differential operators (PDOs) in physics. Developing the theory of
equivariant PDOs in the context of deep learning could bring these subjects
even closer together and lead to a stronger flow of ideas. In this work, we
derive a $G$-steerability constraint that completely characterizes when a PDO
between feature vector fields is equivariant, for arbitrary symmetry groups
$G$. We then fully solve this constraint for several important groups. We use
our solutions as equivariant drop-in replacements for convolutional layers and
benchmark them in that role. Finally, we develop a framework for equivariant
maps based on Schwartz distributions that unifies classical convolutions and
differential operators and gives insight about the relation between the two.

    

### [[2106.10370] On the benefits of maximum likelihood estimation for Regression and Forecasting](http://arxiv.org/abs/2106.10370)


  We advocate for a practical Maximum Likelihood Estimation (MLE) approach
towards designing loss functions for regression and forecasting, as an
alternative to the typical approach of direct empirical risk minimization on a
specific target metric. The MLE approach is better suited to capture inductive
biases such as prior domain knowledge in datasets, and can output post-hoc
estimators at inference time that can optimize different types of target
metrics. We present theoretical results to demonstrate that our approach is
competitive with any estimator for the target metric under some general
conditions. In two example practical settings, Poisson and Pareto regression,
we show that our competitive results can be used to prove that the MLE approach
has better excess risk bounds than directly minimizing the target metric. We
also demonstrate empirically that our method instantiated with a well-designed
general purpose mixture likelihood family can obtain superior performance for a
variety of tasks across time-series forecasting and regression datasets with
different data distributions.

    

### [[2106.10424] More Efficient Adversarial Imitation Learning Algorithms With Known and Unknown Transitions](http://arxiv.org/abs/2106.10424)


  In this work, we design provably (more) efficient imitation learning
algorithms that directly optimize policies from expert demonstrations. Firstly,
when the transition function is known, we build on the nearly minimax optimal
algorithm MIMIC-MD and relax a projection operator in it. Based on this change,
we develop an adversarial imitation learning (AIL) algorithm named \emph{TAIL}
with a gradient-based optimization procedure. Accordingly, TAIL has the same
sample complexity (i.e., the number of expert trajectories)
$\widetilde{\mathcal{O}}(H^{3/2} |\mathcal{S}|/\varepsilon)$ with MIMIC-MD,
where $H$ is the planning horizon, $|\mathcal{S}|$ is the state space size and
$\varepsilon$ is desired policy value gap. In addition, TAIL is more practical
than MIMIC-MD as the former has a space complexity $\mathcal{O}
(|\mathcal{S}||\mathcal{A}|H)$ while the latter's is about $\mathcal{O}
(|\mathcal{S}|^2 |\mathcal{A}|^2 H^2)$. Secondly, under the scenario where the
transition function is unknown but the interaction is allowed, we present an
extension of TAIL named \emph{MB-TAIL}. The sample complexity of MB-TAIL is
still $\widetilde{\mathcal{O}}(H^{3/2} |\mathcal{S}|/\varepsilon)$ while the
interaction complexity (i.e., the number of interaction episodes) is
$\widetilde{\mathcal{O}} (H^3 |\mathcal{S}|^2 |\mathcal{A}| / \varepsilon^2)$.
In particular, MB-TAIL is significantly better than the best-known OAL
algorithm, which has a sample complexity $\widetilde{\mathcal{O}}(H^{2}
|\mathcal{S}|/\varepsilon^2)$ and interaction complexity
$\widetilde{\mathcal{O}} (H^4 |\mathcal{S}|^2 |\mathcal{A}| / \varepsilon^2)$.
The advances in MB-TAIL are based on a new framework that connects reward-free
exploration and AIL. To our understanding, MB-TAIL is the first algorithm that
shifts the advances in the known transition setting to the unknown transition
setting.

    

### [[2106.11514] Adapting Stepsizes by Momentumized Gradients Improves Optimization and Generalization](http://arxiv.org/abs/2106.11514)


  Adaptive gradient methods, such as Adam, have achieved tremendous success in
machine learning. Scaling gradients by square roots of the running averages of
squared past gradients, such methods are able to attain rapid training of
modern deep neural networks. Nevertheless, they are observed to generalize
worse than stochastic gradient descent (SGD) and tend to be trapped in local
minima at an early stage during training. Intriguingly, we discover that
substituting the gradient in the second moment estimation term with the
momentumized version in Adam can well solve the issues. The intuition is that
gradient with momentum contains more accurate directional information and
therefore its second moment estimation is a better choice for scaling than that
of the raw gradient. Thereby we propose AdaMomentum as a new optimizer reaching
the goal of training fast while generalizing better. We further develop a
theory to back up the improvement in optimization and generalization and
provide convergence guarantees under both convex and nonconvex settings.
Extensive experiments on a wide range of tasks and models demonstrate that
AdaMomentum exhibits state-of-the-art performance consistently.

    

### [[2106.16245] How to Train Your MAML to Excel in Few-Shot Classification](http://arxiv.org/abs/2106.16245)


  Model-agnostic meta-learning (MAML) is arguably one of the most popular
meta-learning algorithms nowadays. Nevertheless, its performance on few-shot
classification is far behind many recent algorithms dedicated to the problem.
In this paper, we point out several key facets of how to train MAML to excel in
few-shot classification. First, we find that MAML needs a large number of
gradient steps in its inner loop update, which contradicts its common usage in
few-shot classification. Second, we find that MAML is sensitive to the class
label assignments during meta-testing. Concretely, MAML meta-trains the
initialization of an $N$-way classifier. These $N$ ways, during meta-testing,
then have $N!$ different permutations to be paired with a few-shot task of $N$
novel classes. We find that these permutations lead to a huge variance of
accuracy, making MAML unstable in few-shot classification. Third, we
investigate several approaches to make MAML permutation-invariant, among which
meta-training a single vector to initialize all the $N$ weight vectors in the
classification head performs the best. On benchmark datasets like MiniImageNet
and TieredImageNet, our approach, which we name UNICORN-MAML, performs on a par
with or even outperforms state-of-the-art few-shot classification algorithms,
without sacrificing MAML's simplicity.

    

### [[2110.01595] Solon: Communication-efficient Byzantine-resilient Distributed Training via Redundant Gradients](http://arxiv.org/abs/2110.01595)


  There has been a growing need to provide Byzantine-resilience in distributed
model training. Existing robust distributed learning algorithms focus on
developing sophisticated robust aggregators at the parameter servers, but pay
less attention to balancing the communication cost and robustness. In this
paper, we propose Solon, an algorithmic framework that exploits gradient
redundancy to provide communication efficiency and Byzantine robustness
simultaneously. Our theoretical analysis shows a fundamental trade-off among
computational load, communication cost, and Byzantine robustness. We also
develop a concrete algorithm to achieve the optimal trade-off, borrowing ideas
from coding theory and sparse recovery. Empirical experiments on various
datasets demonstrate that Solon provides significant speedups over existing
methods to achieve the same accuracy, over 10 times faster than Bulyan and 80%
faster than Draco. We also show that carefully designed Byzantine attacks break
Signum and Bulyan, but do not affect the successful convergence of Solon.

    

### [[2009.02535] A Class of Optimal Structures for Node Computations in Message Passing Algorithms](http://arxiv.org/abs/2009.02535)


  Consider the computations at a node in a message passing algorithm. Assume
that the node has incoming and outgoing messages $\mathbf{x} = (x_1, x_2,
\ldots, x_n)$ and $\mathbf{y} = (y_1, y_2, \ldots, y_n)$, respectively. In this
paper, we investigate a class of structures that can be adopted by the node for
computing $\mathbf{y}$ from $\mathbf{x}$, where each $y_j, j = 1, 2, \ldots, n$
is computed via a binary tree with leaves $\mathbf{x}$ excluding $x_j$. We make
three main contributions regarding this class of structures. First, we prove
that the minimum complexity of such a structure is $3n - 6$, and if a structure
has such complexity, its minimum latency is $\delta + \lceil \log(n-2^{\delta})
\rceil$ with $\delta = \lfloor \log(n/2) \rfloor$, where the logarithm always
takes base two. Second, we prove that the minimum latency of such a structure
is $\lceil \log(n-1) \rceil$, and if a structure has such latency, its minimum
complexity is $n \log(n-1)$ when $n-1$ is a power of two. Third, given $(n,
\tau)$ with $\tau \geq \lceil \log(n-1) \rceil$, we propose a construction for
a structure which we conjecture to have the minimum complexity among structures
with latencies at most $\tau$. Our construction method runs in $O(n^3
\log^2(n))$ time, and the obtained structure has complexity at most (generally
much smaller than) $n \lceil \log(n) \rceil - 2$.

    

### [[2102.08323] AdEle: An Adaptive Congestion-and-Energy-Aware Elevator Selection for Partially Connected 3D NoCs](http://arxiv.org/abs/2102.08323)


  By lowering the number of vertical connections in fully connected 3D
networks-on-chip (NoCs), partially connected 3D NoCs (PC-3DNoCs) help alleviate
reliability and fabrication issues. This paper proposes a novel, adaptive
congestion- and energy-aware elevator-selection scheme called AdEle to improve
the traffic distribution in PC-3DNoCs. AdEle employs an offline multi-objective
simulated-annealing-based algorithm to find good elevator subsets and an online
elevator selection policy to enhance elevator selection during routing.
Compared to the state-of- the-art techniques under different real-application
traffics and configuration scenarios, AdEle improves the network latency by
10.9% on average (up to 14.6%) with less than 6.9% energy consumption overhead.

    

### [[2109.05848] Closed-Loop Neural Prostheses with On-Chip Intelligence: A Review and A Low-Latency Machine Learning Model for Brain State Detection](http://arxiv.org/abs/2109.05848)


  The application of closed-loop approaches in systems neuroscience and
therapeutic stimulation holds great promise for revolutionizing our
understanding of the brain and for developing novel neuromodulation therapies
to restore lost functions. Neural prostheses capable of multi-channel neural
recording, on-site signal processing, rapid symptom detection, and closed-loop
stimulation are critical to enabling such novel treatments. However, the
existing closed-loop neuromodulation devices are too simplistic and lack
sufficient on-chip processing and intelligence. In this paper, we first discuss
both commercial and investigational closed-loop neuromodulation devices for
brain disorders. Next, we review state-of-the-art neural prostheses with
on-chip machine learning, focusing on application-specific integrated circuits
(ASIC). System requirements, performance and hardware comparisons, design
trade-offs, and hardware optimization techniques are discussed. To facilitate a
fair comparison and guide design choices among various on-chip classifiers, we
propose a new energy-area (E-A) efficiency figure of merit that evaluates
hardware efficiency and multi-channel scalability. Finally, we present several
techniques to improve the key design metrics of tree-based on-chip classifiers,
both in the context of ensemble methods and oblique structures.

    

### [[2110.04448] A State Transfer Method That Adapts to Network Bandwidth Variations in Geographic State Machine Replication](http://arxiv.org/abs/2110.04448)


  We present a new state transfer method for geographic State Machine
Replication (SMR) that dynamically allocates the state to be transferred among
replicas according to changes in communication bandwidths. SMR is a method that
improves fault tolerance by replicating a service to multiple replicas. When a
replica is newly added or is recovered from a failure, the other replicas
transfer the current state of the service to it. However, in geographic SMR,
the communication bandwidths of replicas are different and constantly changing.
Therefore, existing state transfer methods cannot fully utilize the available
bandwidth, and their state transfer time becomes long. To overcome this
problem, our method divides the state into multiple chunks and assigns them to
replicas based on each replica's bandwidth so that the broader a replica's
bandwidth is, the more chunks it transfers. The number of assigned chunks is
dynamically updated based on the currently estimated bandwidth. The performance
evaluation on Amazon EC2 shows that the proposed method reduces the state
transfer time by up to 47% compared with the existing one.

    

### [[2110.04615] Evaluation and Ranking of Replica Deployments in Geographic State Machine Replication](http://arxiv.org/abs/2110.04615)


  Geographic state machine replication (SMR) is a replication method in which
replicas of a service are located on multiple continents to improve the fault
tolerance of a general service. Nowadays, geographic SMR is easily realized
using public cloud services; SMR provides extraordinary resilience against
catastrophic disasters. Previous studies have revealed that the geographic
distribution of the replicas has a significant influence on the performance of
the geographic SMR; however, the optimal way for a system integrator to deploy
replicas remains unknown. In this paper, we propose a method to evaluate and
rank replica deployments to assist a system integrator in deciding a final
replica deployment. In the method, we also propose a novel evaluation function
that estimates a latency of SMR protocols with round-trip time (RTT). To
demonstrate the effectiveness of the proposed method, we build thousands of
geographic SMRs on Amazon Web Services and present experimental results. The
results show that the proposed method that estimates a latency based on RTTs
can generate consistent rankings with reasonable calculation time.

    

### [[2004.10488] Decentralized Cross-Blockchain Asset Transfers](http://arxiv.org/abs/2004.10488)


  Today, several solutions for cross-blockchain asset transfers exist. However,
these solutions are either tailored to specific assets or neglect finality
guarantees that prevent assets from getting lost in transit.
In this paper, we present a cross-blockchain asset transfer protocol that
supports arbitrary assets and adheres to finality requirements. The ability to
freely transfer assets between blockchains may increase transaction throughput
and provide developers with more flexibility by allowing them to design digital
assets that leverage the capacities and capabilities of multiple blockchains.

    

### [[2101.07095] Byzantine Generals in the Permissionless Setting](http://arxiv.org/abs/2101.07095)


  Consensus protocols have traditionally been studied in a setting where all
participants are known to each other from the start of the protocol execution.
In the parlance of the 'blockchain' literature, this is referred to as the
permissioned setting. What differentiates Bitcoin from these previously studied
protocols is that it operates in a permissionless setting, i.e. it is a
protocol for establishing consensus over an unknown network of participants
that anybody can join, with as many identities as they like in any role. The
arrival of this new form of protocol brings with it many questions. Beyond
Bitcoin, what can we prove about permissionless protocols in a general sense?
How does recent work on permissionless protocols in the blockchain literature
relate to the well-developed history of research on permissioned protocols in
distributed computing?
To answer these questions, we describe a formal framework for the analysis of
both permissioned and permissionless systems. Our framework allows for
"apples-to-apples" comparisons between different categories of protocols and,
in turn, the development of theory to formally discuss their relative merits. A
major benefit of the framework is that it facilitates the application of a rich
history of proofs and techniques in distributed computing to problems in
blockchain and the study of permissionless systems. Within our framework, we
then address the questions above. We consider the Byzantine Generals Problem as
a formalisation of the problem of reaching consensus, and address a programme
of research that asks, "Under what adversarial conditions, and for what types
of permissionless protocol, is consensus possible?" We prove a number of
results for this programme, our main result being that deterministic consensus
is not possible for decentralised permissionless protocols. To close, we give a
list of eight open questions.

    

### [[2103.00167] Inferring Unobserved Events in Systems With Shared Resources and Queues](http://arxiv.org/abs/2103.00167)


  To identify the causes of performance problems or to predict process
behavior, it is essential to have correct and complete event data. This is
particularly important for distributed systems with shared resources, e.g., one
case can block another case competing for the same machine, leading to
inter-case dependencies in performance. However, due to a variety of reasons,
real-life systems often record only a subset of all events taking place. To
understand and analyze the behavior and performance of processes with shared
resources, we aim to reconstruct bounds for timestamps of events in a case that
must have happened but were not recorded by inference over events in other
cases in the system. We formulate and solve the problem by systematically
introducing multi-entity concepts in event logs and process models. We
introduce a partial-order based model of a multi-entity event log and a
corresponding compositional model for multi-entity processes. We define
PQR-systems as a special class of multi-entity processes with shared resources
and queues. We then study the problem of inferring from an incomplete event log
unobserved events and their timestamps that are globally consistent with a
PQR-system. We solve the problem by reconstructing unobserved traces of
resources and queues according to the PQR-model and derive bounds for their
timestamps using a linear program. While the problem is illustrated for
material handling systems like baggage handling systems in airports, the
approach can be applied to other settings where recording is incomplete. The
ideas have been implemented in ProM and were evaluated using both synthetic and
real-life event logs.

    

### [[2110.04413] Robustness Evaluation of Transformer-based Form Field Extractors via Form Attacks](http://arxiv.org/abs/2110.04413)


  We propose a novel framework to evaluate the robustness of transformer-based
form field extraction methods via form attacks. We introduce 14 novel form
transformations to evaluate the vulnerability of the state-of-the-art field
extractors against form attacks from both OCR level and form level, including
OCR location/order rearrangement, form background manipulation and form
field-value augmentation. We conduct robustness evaluation using real invoices
and receipts, and perform comprehensive research analysis. Experimental results
suggest that the evaluated models are very susceptible to form perturbations
such as the variation of field-values (~15% drop in F1 score), the
disarrangement of input text order(~15% drop in F1 score) and the disruption of
the neighboring words of field-values(~10% drop in F1 score). Guided by the
analysis, we make recommendations to improve the design of field extractors and
the process of data collection.

    

### [[2110.04418] Moral-Trust Violation vs Performance-Trust Violation by a Robot: Which Hurts More?](http://arxiv.org/abs/2110.04418)


  In recent years a modern conceptualization of trust in human-robot
interaction (HRI) was introduced by Ullman et al.\cite{ullman2018does}. This
new conceptualization of trust suggested that trust between humans and robots
is multidimensional, incorporating both performance aspects (i.e., similar to
the trust in human-automation interaction) and moral aspects (i.e., similar to
the trust in human-human interaction). But how does a robot violating each of
these different aspects of trust affect human trust in a robot? How does trust
in robots change when a robot commits a moral-trust violation compared to a
performance-trust violation? And whether physiological signals have the
potential to be used for assessing gain/loss of each of these two trust aspects
in a human. We aim to design an experiment to study the effects of
performance-trust violation and moral-trust violation separately in a search
and rescue task. We want to see whether two failures of a robot with equal
magnitudes would affect human trust differently if one failure is due to a
performance-trust violation and the other is a moral-trust violation.

    

### [[2110.04439] Research on Knowledge based Expert System for Medical Diagnosis](http://arxiv.org/abs/2110.04439)


  In this paper we propose the design and implementation of a generic medical
knowledge based system (MKBS) for identifying diseases from several symptoms
and signs. To diagnosis diseases, user will be asked by the system for
different questions and finally inference engine will use certainty factor to
prune out low possible solutions. In this system some important aspects like
Knowledge bases system, Knowledge representation, Inference Engine has been
addressed. New certainty fact has been introduced to get conclusion about same
firing rules. The proposed disease diagnosis system also uses a graphical user
user interface to facilitate user to interact a with expert system more easily.
The proposed system is generic and knowledge based, and it can be integrated
with any rule bases system in disease diagnosis.

    

### [[2110.04441] Natural Language for Human-Robot Collaboration: Problems Beyond Language Grounding](http://arxiv.org/abs/2110.04441)


  To enable robots to instruct humans in collaborations, we identify several
aspects of language processing that are not commonly studied in this context.
These include location, planning, and generation. We suggest evaluations for
each task, offer baselines for simple methods, and close by discussing
challenges and opportunities in studying language for collaboration.

    

### [[2110.04447] EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement](http://arxiv.org/abs/2110.04447)


  Camera-based physiological measurement is a growing field with neural models
providing state-the-art-performance. Prior research have explored various
``end-to-end'' models; however these methods still require several
preprocessing steps. These additional operations are often non-trivial to
implement making replication and deployment difficult and can even have a
higher computational budget than the ``core'' network itself. In this paper, we
propose two novel and efficient neural models for camera-based physiological
measurement called EfficientPhys that remove the need for face detection,
segmentation, normalization, color space transformation or any other
preprocessing steps. Using an input of raw video frames, our models achieve
state-of-the-art accuracy on three public datasets. We show that this is the
case whether using a transformer or convolutional backbone. We further evaluate
the latency of the proposed networks and show that our most light weight
network also achieves a 33% improvement in efficiency.

    

### [[2110.04451] Using multiple reference audios and style embedding constraints for speech synthesis](http://arxiv.org/abs/2110.04451)


  The end-to-end speech synthesis model can directly take an utterance as
reference audio, and generate speech from the text with prosody and speaker
characteristics similar to the reference audio. However, an appropriate
acoustic embedding must be manually selected during inference. Due to the fact
that only the matched text and speech are used in the training process, using
unmatched text and speech for inference would cause the model to synthesize
speech with low content quality. In this study, we propose to mitigate these
two problems by using multiple reference audios and style embedding constraints
rather than using only the target audio. Multiple reference audios are
automatically selected using the sentence similarity determined by
Bidirectional Encoder Representations from Transformers (BERT). In addition, we
use ''target'' style embedding from a Pre-trained encoder as a constraint by
considering the mutual information between the predicted and ''target'' style
embedding. The experimental results show that the proposed model can improve
the speech naturalness and content quality with multiple reference audios and
can also outperform the baseline model in ABX preference tests of style
similarity.

    

### [[2110.04452] Towards AI Logic for Social Reasoning](http://arxiv.org/abs/2110.04452)


  Artificial Intelligence (AI) logic formalizes the reasoning of intelligent
agents. In this paper, we discuss how an argumentation-based AI logic could be
used also to formalize important aspects of social reasoning. Besides reasoning
about the knowledge and actions of individual agents, social AI logic can
reason also about social dependencies among agents using the rights,
obligations and permissions of the agents. We discuss four aspects of social AI
logic. First, we discuss how rights represent relations between the obligations
and permissions of intelligent agents. Second, we discuss how to argue about
the right-to-know, a central issue in the recent discussion of privacy and
ethics. Third, we discuss how a wide variety of conflicts among intelligent
agents can be identified and (sometimes) resolved by comparing formal
arguments. Importantly, to cover a wide range of arguments occurring in daily
life, also fallacious arguments can be represented and reasoned about. Fourth,
we discuss how to argue about the freedom to act for intelligent agents.
Examples from social, legal and ethical reasoning highlight the challenges in
developing social AI logic. The discussion of the four challenges leads to a
research program for argumentation-based social AI logic, contributing towards
the future development of AI logic.

    

### [[2110.04454] Dynamic Logic of Legal Competences](http://arxiv.org/abs/2110.04454)


  We propose a new formalization of legal competences, and in particular for
the Hohfeldian categories of power and immunity, through a deontic
reinterpretation of dynamic epistemic logic. We argue that this logic
explicitly captures the norm-changing character of legal competences while
providing a sophisticated reduction of the latter to static normative
positions. The logic is completely axiomatizable, and we apply it to a concrete
case in German contract law to illustrate that it can capture the distinction
between legal ability and legal permissibility.

    

### [[2110.04465] Predicting decision-making in the future: Human versus Machine](http://arxiv.org/abs/2110.04465)


  Deep neural networks (DNNs) have become remarkably successful in data
prediction, and have even been used to predict future actions based on limited
input. This raises the question: do these systems actually "understand" the
event similar to humans? Here, we address this issue using videos taken from an
accident situation in a driving simulation. In this situation, drivers had to
choose between crashing into a suddenly-appeared obstacle or steering their car
off a previously indicated cliff. We compared how well humans and a DNN
predicted this decision as a function of time before the event. The DNN
outperformed humans for early time-points, but had an equal performance for
later time-points. Interestingly, spatio-temporal image manipulations and
Grad-CAM visualizations uncovered some expected behavior, but also highlighted
potential differences in temporal processing for the DNN.

    

### [[2110.04473] Self-adaptive Multi-task Particle Swarm Optimization](http://arxiv.org/abs/2110.04473)


  Multi-task optimization (MTO) studies how to simultaneously solve multiple
optimization problems for the purpose of obtaining better performance on each
problem. Over the past few years, evolutionary MTO (EMTO) was proposed to
handle MTO problems via evolutionary algorithms. So far, many EMTO algorithms
have been developed and demonstrated well performance on solving real-world
problems. However, there remain many works to do in adapting knowledge transfer
to task relatedness in EMTO. Different from the existing works, we develop a
self-adaptive multi-task particle swarm optimization (SaMTPSO) through the
developed knowledge transfer adaptation strategy, the focus search strategy and
the knowledge incorporation strategy. In the knowledge transfer adaptation
strategy, each task has a knowledge source pool that consists of all knowledge
sources. Each source (task) outputs knowledge to the task. And knowledge
transfer adapts to task relatedness via individuals' choice on different
sources of a pool, where the chosen probabilities for different sources are
computed respectively according to task's success rate in generating improved
solutions via these sources. In the focus search strategy, if there is no
knowledge source benefit the optimization of a task, then all knowledge sources
in the task's pool are forbidden to be utilized except the task, which helps to
improve the performance of the proposed algorithm. Note that the task itself is
as a knowledge source of its own. In the knowledge incorporation strategy, two
different forms are developed to help the SaMTPSO explore and exploit the
transferred knowledge from a chosen source, each leading to a version of the
SaMTPSO. Several experiments are conducted on two test suites. The results of
the SaMTPSO are comparing to that of 3 popular EMTO algorithms and a particle
swarm algorithm, which demonstrates the superiority of the SaMTPSO.

    

### [[2110.04507] TiKick: Toward Playing Multi-agent Football Full Games from Single-agent Demonstrations](http://arxiv.org/abs/2110.04507)


  Deep reinforcement learning (DRL) has achieved super-human performance on
complex video games (e.g., StarCraft II and Dota II). However, current DRL
systems still suffer from challenges of multi-agent coordination, sparse
rewards, stochastic environments, etc. In seeking to address these challenges,
we employ a football video game, e.g., Google Research Football (GRF), as our
testbed and develop an end-to-end learning-based AI system (denoted as TiKick
to complete this challenging task. In this work, we first generated a large
replay dataset from the self-playing of single-agent experts, which are
obtained from league training. We then developed a distributed learning system
and new offline algorithms to learn a powerful multi-agent AI from the fixed
single-agent dataset. To the best of our knowledge, Tikick is the first
learning-based AI system that can take over the multi-agent Google Research
Football full game, while previous work could either control a single agent or
experiment on toy academic scenarios. Extensive experiments further show that
our pre-trained model can accelerate the training process of the modern
multi-agent algorithm and our method achieves state-of-the-art performances on
various academic scenarios.

    

### [[2110.04538] Focus Your Distribution: Coarse-to-Fine Non-Contrastive Learning for Anomaly Detection and Localization](http://arxiv.org/abs/2110.04538)


  The essence of unsupervised anomaly detection is to learn the compact
distribution of normal samples and detect outliers as anomalies in testing.
Meanwhile, the anomalies in real-world are usually subtle and fine-grained in a
high-resolution image especially for industrial applications. Towards this end,
we propose a novel framework for unsupervised anomaly detection and
localization. Our method aims at learning dense and compact distribution from
normal images with a coarse-to-fine alignment process. The coarse alignment
stage standardizes the pixel-wise position of objects in both image and feature
levels. The fine alignment stage then densely maximizes the similarity of
features among all corresponding locations in a batch. To facilitate the
learning with only normal images, we propose a new pretext task called
non-contrastive learning for the fine alignment stage. Non-contrastive learning
extracts robust and discriminating normal image representations without making
assumptions on abnormal samples, and it thus empowers our model to generalize
to various anomalous scenarios. Extensive experiments on two typical industrial
datasets of MVTec AD and BenTech AD demonstrate that our framework is effective
in detecting various real-world defects and achieves a new state-of-the-art in
industrial unsupervised anomaly detection.

    

### [[2110.04580] Active Altruism Learning and Information Sufficiency for Autonomous Driving](http://arxiv.org/abs/2110.04580)


  Safe interaction between vehicles requires the ability to choose actions that
reveal the preferences of the other vehicles. Since exploratory actions often
do not directly contribute to their objective, an interactive vehicle must also
able to identify when it is appropriate to perform them. In this work we
demonstrate how Active Learning methods can be used to incentivise an
autonomous vehicle (AV) to choose actions that reveal information about the
altruistic inclinations of another vehicle. We identify a property, Information
Sufficiency, that a reward function should have in order to keep exploration
from unnecessarily interfering with the pursuit of an objective. We empirically
demonstrate that reward functions that do not have Information Sufficiency are
prone to inadequate exploration, which can result in sub-optimal behaviour. We
propose a reward definition that has Information Sufficiency, and show that it
facilitates an AV choosing exploratory actions to estimate altruistic tendency,
whilst also compensating for the possibility of conflicting beliefs between
vehicles.

    

### [[2110.04620] A Framework for Rationale Extraction for Deep QA models](http://arxiv.org/abs/2110.04620)


  As neural-network-based QA models become deeper and more complex, there is a
demand for robust frameworks which can access a model's rationale for its
prediction. Current techniques that provide insights on a model's working are
either dependent on adversarial datasets or are proposing models with explicit
explanation generation components. These techniques are time-consuming and
challenging to extend to existing models and new datasets. In this work, we use
`Integrated Gradients' to extract rationale for existing state-of-the-art
models in the task of Reading Comprehension based Question Answering (RCQA). On
detailed analysis and comparison with collected human rationales, we find that
though ~40-80% words of extracted rationale coincide with the human rationale
(precision), only 6-19% of human rationale is present in the extracted
rationale (recall).

    

### [[2110.04649] Interactive Hierarchical Guidance using Language](http://arxiv.org/abs/2110.04649)


  Reinforcement learning has been successful in many tasks ranging from robotic
control, games, energy management etc. In complex real world environments with
sparse rewards and long task horizons, sample efficiency is still a major
challenge. Most complex tasks can be easily decomposed into high-level planning
and low level control. Therefore, it is important to enable agents to leverage
the hierarchical structure and decompose bigger tasks into multiple smaller
sub-tasks. We introduce an approach where we use language to specify sub-tasks
and a high-level planner issues language commands to a low level controller.
The low-level controller executes the sub-tasks based on the language commands.
Our experiments show that this method is able to solve complex long horizon
planning tasks with limited human supervision. Using language has added benefit
of interpretability and ability for expert humans to take over the high-level
planning task and provide language commands if necessary.

    

### [[2110.04660] K-Splits: Improved K-Means Clustering Algorithm to Automatically Detect the Number of Clusters](http://arxiv.org/abs/2110.04660)


  This paper introduces k-splits, an improved hierarchical algorithm based on
k-means to cluster data without prior knowledge of the number of clusters.
K-splits starts from a small number of clusters and uses the most significant
data distribution axis to split these clusters incrementally into better fits
if needed. Accuracy and speed are two main advantages of the proposed method.
We experiment on six synthetic benchmark datasets plus two real-world datasets
MNIST and Fashion-MNIST, to prove that our algorithm has excellent accuracy in
finding the correct number of clusters under different conditions. We also show
that k-splits is faster than similar methods and can even be faster than the
standard k-means in lower dimensions. Finally, we suggest using k-splits to
uncover the exact position of centroids and then input them as initial points
to the k-means algorithm to fine-tune the results.

    

### [[2110.04663] Learning to Control Complex Robots Using High-Dimensional Interfaces: Preliminary Insights](http://arxiv.org/abs/2110.04663)


  Human body motions can be captured as a high-dimensional continuous signal
using motion sensor technologies. The resulting data can be surprisingly rich
in information, even when captured from persons with limited mobility. In this
work, we explore the use of limited upper-body motions, captured via motion
sensors, as inputs to control a 7 degree-of-freedom assistive robotic arm. It
is possible that even dense sensor signals lack the salient information and
independence necessary for reliable high-dimensional robot control. As the
human learns over time in the context of this limitation, intelligence on the
robot can be leveraged to better identify key learning challenges, provide
useful feedback, and support individuals until the challenges are managed. In
this short paper, we examine two uninjured participants' data from an ongoing
study, to extract preliminary results and share insights. We observe
opportunities for robot intelligence to step in, including the identification
of inconsistencies in time spent across all control dimensions, asymmetries in
individual control dimensions, and user progress in learning. Machine reasoning
about these situations may facilitate novel interface learning in the future.

    

### [[2110.04664] Using Human-Guided Causal Knowledge for More Generalized Robot Task Planning](http://arxiv.org/abs/2110.04664)


  A major challenge in research involving artificial intelligence (AI) is the
development of algorithms that can find solutions to problems that can
generalize to different environments and tasks. Unlike AI, humans are adept at
finding solutions that can transfer. We hypothesize this is because their
solutions are informed by causal models. We propose to use human-guided causal
knowledge to help robots find solutions that can generalize to a new
environment. We develop and test the feasibility of a language interface that
naïve participants can use to communicate these causal models to a planner.
We find preliminary evidence that participants are able to use our interface
and generate causal models that achieve near-generalization. We outline an
experiment aimed at testing far-generalization using our interface and describe
our longer terms goals for these causal models.

    

### [[2110.04678] An Overview of Techniques for Biomarker Discovery in Voice Signal](http://arxiv.org/abs/2110.04678)


  This paper reflects on the effect of several categories of medical conditions
on human voice, focusing on those that may be hypothesized to have effects on
voice, but for which the changes themselves may be subtle enough to have eluded
observation in standard analytical examinations of the voice signal. It
presents three categories of techniques that can potentially uncover such
elusive biomarkers and allow them to be measured and used for predictive and
diagnostic purposes. These approaches include proxy techniques, model-based
analytical techniques and data-driven AI techniques.

    

### [[2110.04685] Learning Visual Shape Control of Novel 3D Deformable Objects from Partial-View Point Clouds](http://arxiv.org/abs/2110.04685)


  If robots could reliably manipulate the shape of 3D deformable objects, they
could find applications in fields ranging from home care to warehouse
fulfillment to surgical assistance. Analytic models of elastic, 3D deformable
objects require numerous parameters to describe the potentially infinite
degrees of freedom present in determining the object's shape. Previous attempts
at performing 3D shape control rely on hand-crafted features to represent the
object shape and require training of object-specific control models. We
overcome these issues through the use of our novel DeformerNet neural network
architecture, which operates on a partial-view point cloud of the object being
manipulated and a point cloud of the goal shape to learn a low-dimensional
representation of the object shape. This shape embedding enables the robot to
learn to define a visual servo controller that provides Cartesian pose changes
to the robot end-effector causing the object to deform towards its target
shape. Crucially, we demonstrate both in simulation and on a physical robot
that DeformerNet reliably generalizes to object shapes and material stiffness
not seen during training and outperforms comparison methods for both the
generic shape control and the surgical task of retraction.

    

### [[2110.04697] An Augmented Reality Platform for Introducing Reinforcement Learning to K-12 Students with Robots](http://arxiv.org/abs/2110.04697)


  Interactive reinforcement learning, where humans actively assist during an
agent's learning process, has the promise to alleviate the sample complexity
challenges of practical algorithms. However, the inner workings and state of
the robot are typically hidden from the teacher when humans provide feedback.
To create a common ground between the human and the learning robot, in this
paper, we propose an Augmented Reality (AR) system that reveals the hidden
state of the learning to the human users. This paper describes our system's
design and implementation and concludes with a discussion on two directions for
future work which we are pursuing: 1) use of our system in AI education
activities at the K-12 level; and 2) development of a framework for an AR-based
human-in-the-loop reinforcement learning, where the human teacher can see
sensory and cognitive representations of the robot overlaid in the real world.

    

### [[2010.03855] Dense Relational Image Captioning via Multi-task Triple-Stream Networks](http://arxiv.org/abs/2010.03855)


  We introduce dense relational captioning, a novel image captioning task which
aims to generate multiple captions with respect to relational information
between objects in a visual scene. Relational captioning provides explicit
descriptions for each relationship between object combinations. This framework
is advantageous in both diversity and amount of information, leading to a
comprehensive image understanding based on relationships, e.g., relational
proposal generation. For relational understanding between objects, the
part-of-speech (POS; i.e., subject-object-predicate categories) can be a
valuable prior information to guide the causal sequence of words in a caption.
We enforce our framework to learn not only to generate captions but also to
understand the POS of each word. To this end, we propose the multi-task
triple-stream network (MTTSNet) which consists of three recurrent units
responsible for each POS which is trained by jointly predicting the correct
captions and POS for each word. In addition, we found that the performance of
MTTSNet can be improved by modulating the object embeddings with an explicit
relational module. We demonstrate that our proposed model can generate more
diverse and richer captions, via extensive experimental analysis on large scale
datasets and several metrics. Then, we present applications of our framework to
holistic image captioning, scene graph generation, and retrieval tasks.

    

### [[2011.07689] Drone LAMS: A Drone-based Face Detection Dataset with Large Angles and Many Scenarios](http://arxiv.org/abs/2011.07689)


  This work presented a new drone-based face detection dataset Drone LAMS in
order to solve issues of low performance of drone-based face detection in
scenarios such as large angles which was a predominant working condition when a
drone flies high. The proposed dataset captured images from 261 videos with
over 43k annotations and 4.0k images with pitch or yaw angle in the range of
-90° to 90°. Drone LAMS showed significant improvement over currently
available drone-based face detection datasets in terms of detection
performance, especially with large pitch and yaw angle. Detailed analysis of
how key factors, such as duplication rate, annotation method, etc., impact
dataset performance was also provided to facilitate further usage of a drone on
face detection.

    

### [[2101.03603] Target Detection and Segmentation in Circular-Scan Synthetic-Aperture-Sonar Images using Semi-Supervised Convolutional Encoder-Decoders](http://arxiv.org/abs/2101.03603)


  We propose a framework for saliency-based, multi-target detection and
segmentation of circular-scan, synthetic-aperture-sonar (CSAS) imagery. Our
framework relies on a multi-branch, convolutional encoder-decoder network ({\sc
MB-CEDN}). The encoder portion of the {\sc MB-CEDN} extracts visual contrast
features from CSAS images. These features are fed into dual decoders that
perform pixel-level segmentation to mask targets. Each decoder provides
different perspectives as to what constitutes a salient target. These opinions
are aggregated and cascaded into a deep-parsing network to refine the
segmentation.
We evaluate our framework using real-world CSAS imagery consisting of five
broad target classes. We compare against existing approaches from the
computer-vision literature. We show that our framework outperforms supervised,
deep-saliency networks designed for natural imagery. It greatly outperforms
unsupervised saliency approaches developed for natural imagery. This
illustrates that natural-image-based models may need to be altered to be
effective for this imaging-sonar modality.

    

### [[2102.05449] Adaptive Processor Frequency Adjustment for Mobile Edge Computing with Intermittent Energy Supply](http://arxiv.org/abs/2102.05449)


  With astonishing speed, bandwidth, and scale, Mobile Edge Computing (MEC) has
played an increasingly important role in the next generation of connectivity
and service delivery. Yet, along with the massive deployment of MEC servers,
the ensuing energy issue is now on an increasingly urgent agenda. In the
current context, the large scale deployment of renewable-energy-supplied MEC
servers is perhaps the most promising solution for the incoming energy issue.
Nonetheless, as a result of the intermittent nature of their power sources,
these special design MEC server must be more cautious about their energy usage,
in a bid to maintain their service sustainability as well as service standard.
Targeting optimization on a single-server MEC scenario, we in this paper
propose NAFA, an adaptive processor frequency adjustment solution, to enable an
effective plan of the server's energy usage. By learning from the historical
data revealing request arrival and energy harvest pattern, the deep
reinforcement learning-based solution is capable of making intelligent
schedules on the server's processor frequency, so as to strike a good balance
between service sustainability and service quality. The superior performance of
NAFA is substantiated by real-data-based experiments, wherein NAFA demonstrates
up to 20% increase in average request acceptance ratio and up to 50% reduction
in average request processing time.

    

### [[2102.06362] A Decentralized Approach Towards Responsible AI in Social Ecosystems](http://arxiv.org/abs/2102.06362)


  For AI technology to fulfill its full promises, we must have effective means
to ensure Responsible AI behavior and curtail potential irresponsible use,
e.g., in areas of privacy protection, human autonomy, robustness, and
prevention of biases and discrimination in automated decision making. Recent
literature in the field has identified serious shortcomings of narrow
technology focused and formalism-oriented research and has proposed an
interdisciplinary approach that brings the social context into the scope of
study. In this paper, we take a sociotechnical approach to propose a more
expansive framework of thinking about the Responsible AI challenges in both
technical and social context. Effective solutions need to bridge the gap
between a technical system with the social system that it will be deployed to.
To this end, we propose human agency and regulation as main mechanisms of
intervention and propose a decentralized computational infrastructure, or a set
of public utilities, as the computational means to bridge this gap. A
decentralized infrastructure is uniquely suited for meeting this challenge and
enable technical solutions and social institutions in a mutually reinforcing
dynamic to achieve Responsible AI goals. Our approach is novel in its
sociotechnical approach and its aim in tackling the structural issues that
cannot be solved within the narrow confines of AI technical research. We then
explore possible features of the proposed infrastructure and discuss how it may
help solve example problems recently studied in the field.

    

### [[2103.02835] A Novel Application of Image-to-Image Translation: Chromosome Straightening Framework by Learning from a Single Image](http://arxiv.org/abs/2103.02835)


  In medical imaging, chromosome straightening plays a significant role in the
pathological study of chromosomes and in the development of cytogenetic maps.
Whereas different approaches exist for the straightening task, typically
geometric algorithms are used whose outputs are characterized by jagged edges
or fragments with discontinued banding patterns. To address the flaws in the
geometric algorithms, we propose a novel framework based on image-to-image
translation to learn a pertinent mapping dependence for synthesizing
straightened chromosomes with uninterrupted banding patterns and preserved
details. In addition, to avoid the pitfall of deficient input chromosomes, we
construct an augmented dataset using only one single curved chromosome image
for training models. Based on this framework, we apply two popular
image-to-image translation architectures, U-shape networks and conditional
generative adversarial networks, to assess its efficacy. Experiments on a
dataset comprised of 642 real-world chromosomes demonstrate the superiority of
our framework, as compared to the geometric method in straightening
performance, by rendering realistic and continued chromosome details.
Furthermore, our straightened results improve the chromosome classification by
0.98%-1.39% mean accuracy.

    

### [[2103.04541] A Reinforcement Learning Based R-Tree for Spatial Data Indexing in Dynamic Environments](http://arxiv.org/abs/2103.04541)


  Learned indices have been proposed to replace classic index structures like
B-Tree with machine learning (ML) models. They require to replace both the
indices and query processing algorithms currently deployed by the databases,
and such a radical departure is likely to encounter challenges and obstacles.
In contrast, we propose a fundamentally different way of using ML techniques to
improve on the query performance of the classic R-Tree without the need of
changing its structure or query processing algorithms. Specifically, we develop
reinforcement learning (RL) based models to decide how to choose a subtree for
insertion and how to split a node when building an R-Tree, instead of relying
on hand-crafted heuristic rules currently used by R-Tree and its variants.
Experiments on real and synthetic datasets with up to more than 100 million
spatial objects clearly show that our RL based index outperforms R-Tree and its
variants in terms of query processing time.

    

### [[2103.07601] Approximating How Single Head Attention Learns](http://arxiv.org/abs/2103.07601)


  Why do models often attend to salient words, and how does this evolve
throughout training? We approximate model training as a two stage process:
early on in training when the attention weights are uniform, the model learns
to translate individual input word `i` to `o` if they co-occur frequently.
Later, the model learns to attend to `i` while the correct output is $o$
because it knows `i` translates to `o`. To formalize, we define a model
property, Knowledge to Translate Individual Words (KTIW) (e.g. knowing that `i`
translates to `o`), and claim that it drives the learning of the attention.
This claim is supported by the fact that before the attention mechanism is
learned, KTIW can be learned from word co-occurrence statistics, but not the
other way around. Particularly, we can construct a training distribution that
makes KTIW hard to learn, the learning of the attention fails, and the model
cannot even learn the simple task of copying the input words to the output. Our
approximation explains why models sometimes attend to salient words, and
inspires a toy example where a multi-head attention model can overcome the
above hard training distribution by improving learning dynamics rather than
expressiveness. We end by discussing the limitation of our approximation
framework and suggest future directions.

    

### [[2103.11517] Dual Monte Carlo Tree Search](http://arxiv.org/abs/2103.11517)


  AlphaZero, using a combination of Deep Neural Networks and Monte Carlo Tree
Search (MCTS), has successfully trained reinforcement learning agents in a
tabula-rasa way. The neural MCTS algorithm has been successful in finding
near-optimal strategies for games through self-play. However, the AlphaZero
algorithm has a significant drawback; it takes a long time to converge and
requires high computational power due to complex neural networks for solving
games like Chess, Go, Shogi, etc. Owing to this, it is very difficult to pursue
neural MCTS research without cutting-edge hardware, which is a roadblock for
many aspiring neural MCTS researchers. In this paper, we propose a new neural
MCTS algorithm, called Dual MCTS, which helps overcome these drawbacks. Dual
MCTS uses two different search trees, a single deep neural network, and a new
update technique for the search trees using a combination of the PUCB, a
sliding-window, and the epsilon-greedy algorithm. This technique is applicable
to any MCTS based algorithm to reduce the number of updates to the tree. We
show that Dual MCTS performs better than one of the most widely used neural
MCTS algorithms, AlphaZero, for various symmetric and asymmetric games.

    

### [[2103.14804] LSTM Based Sentiment Analysis for Cryptocurrency Prediction](http://arxiv.org/abs/2103.14804)


  Recent studies in big data analytics and natural language processing develop
automatic techniques in analyzing sentiment in the social media information. In
addition, the growing user base of social media and the high volume of posts
also provide valuable sentiment information to predict the price fluctuation of
the cryptocurrency. This research is directed to predicting the volatile price
movement of cryptocurrency by analyzing the sentiment in social media and
finding the correlation between them. While previous work has been developed to
analyze sentiment in English social media posts, we propose a method to
identify the sentiment of the Chinese social media posts from the most popular
Chinese social media platform Sina-Weibo. We develop the pipeline to capture
Weibo posts, describe the creation of the crypto-specific sentiment dictionary,
and propose a long short-term memory (LSTM) based recurrent neural network
along with the historical cryptocurrency price movement to predict the price
trend for future time frames. The conducted experiments demonstrate the
proposed approach outperforms the state of the art auto regressive based model
by 18.5% in precision and 15.4% in recall.

    

### [[2105.09090] Local Aggressive Adversarial Attacks on 3D Point Cloud](http://arxiv.org/abs/2105.09090)


  Deep neural networks are found to be prone to adversarial examples which
could deliberately fool the model to make mistakes. Recently, a few of works
expand this task from 2D image to 3D point cloud by using global point cloud
optimization. However, the perturbations of global point are not effective for
misleading the victim model. First, not all points are important in
optimization toward misleading. Abundant points account considerable distortion
budget but contribute trivially to attack. Second, the multi-label optimization
is suboptimal for adversarial attack, since it consumes extra energy in finding
multi-label victim model collapse and causes instance transformation to be
dissimilar to any particular instance. Third, the independent adversarial and
perceptibility losses, caring misclassification and dissimilarity separately,
treat the updating of each point equally without a focus. Therefore, once
perceptibility loss approaches its budget threshold, all points would be stock
in the surface of hypersphere and attack would be locked in local optimality.
Therefore, we propose a local aggressive adversarial attacks (L3A) to solve
above issues. Technically, we select a bunch of salient points, the high-score
subset of point cloud according to gradient, to perturb. Then a flow of
aggressive optimization strategies are developed to reinforce the unperceptive
generation of adversarial examples toward misleading victim models. Extensive
experiments on PointNet, PointNet++ and DGCNN demonstrate the state-of-the-art
performance of our method against existing adversarial attack methods.

    

### [[2105.12319] Neural Radiosity](http://arxiv.org/abs/2105.12319)


  We introduce Neural Radiosity, an algorithm to solve the rendering equation
by minimizing the norm of its residual similar as in traditional radiosity
techniques. Traditional basis functions used in radiosity techniques, such as
piecewise polynomials or meshless basis functions are typically limited to
representing isotropic scattering from diffuse surfaces. Instead, we propose to
leverage neural networks to represent the full four-dimensional radiance
distribution, directly optimizing network parameters to minimize the norm of
the residual. Our approach decouples solving the rendering equation from
rendering (perspective) images similar as in traditional radiosity techniques,
and allows us to efficiently synthesize arbitrary views of a scene. In
addition, we propose a network architecture using geometric learnable features
that improves convergence of our solver compared to previous techniques. Our
approach leads to an algorithm that is simple to implement, and we demonstrate
its effectiveness on a variety of scenes with non-diffuse surfaces.

    

### [[2106.14396] Single RGB-D Camera Teleoperation for General Robotic Manipulation](http://arxiv.org/abs/2106.14396)


  We propose a teleoperation system that uses a single RGB-D camera as the
human motion capture device. Our system can perform general manipulation tasks
such as cloth folding, hammering and 3mm clearance peg in hole. We propose the
use of non-Cartesian oblique coordinate frame, dynamic motion scaling and
reposition of operator frames to increase the flexibility of our teleoperation
system. We hypothesize that lowering the barrier of entry to teleoperation will
allow for wider deployment of supervised autonomy system, which will in turn
generates realistic datasets that unlock the potential of machine learning for
robotic manipulation. Demo of our systems are available online
this https URL


### [[2110.04461] Toward Hole-Driven Development with Liquid Haskell](http://arxiv.org/abs/2110.04461)


  Liquid Haskell is an extension to the Haskell programming language that adds
support for refinement types: data types augmented with SMT-decidable logical
predicates that refine the set of values that can inhabit a type. Furthermore,
Liquid Haskell's support for refinement reflection enables the use of Haskell
for general-purpose mechanized theorem proving. A growing list of large-scale
mechanized proof developments in Liquid Haskell take advantage of this
capability. Adding theorem-proving capabilities to a "legacy" language like
Haskell lets programmers directly verify properties of real-world Haskell
programs (taking advantage of the existing highly tuned compiler, run-time
system, and libraries), just by writing Haskell. However, more established
proof assistants like Agda and Coq offer far better support for interactive
proof development and insight into the proof state (for instance, what subgoals
still need to be proved to finish a partially-complete proof). In contrast,
Liquid Haskell provides only coarse-grained feedback to the user -- either it
reports a type error, or not -- unfortunately hindering its usability as a
theorem prover.
In this paper, we propose improving the usability of Liquid Haskell by
extending it with support for Agda-style typed holes and interactive editing
commands that take advantage of them. In Agda, typed holes allow programmers to
indicate unfinished parts of a proof, and incrementally complete the proof in a
dialogue with the compiler. While GHC Haskell already has its own Agda-inspired
support for typed holes, we posit that typed holes would be especially powerful
and useful if combined with Liquid Haskell's refinement types and SMT
automation. We discuss how typed holes might work in Liquid Haskell, and we
consider possible implementation approaches and next steps.

    

### [[1912.12659] Synthesizing Queries via Interactive Sketching](http://arxiv.org/abs/1912.12659)


  We propose a novel approach to program synthesis, focusing on synthesizing
database queries. At a high level, our proposed algorithm takes as input a
sketch with soft constraints encoding user intent, and then iteratively
interacts with the user to refine the sketch. At each step, our algorithm
proposes a candidate refinement of the sketch, which the user can either accept
or reject. By leveraging this rich form of user feedback, our algorithm is able
to both resolve ambiguity in user intent and improve scalability. In
particular, assuming the user provides accurate inputs and responses, then our
algorithm is guaranteed to converge to the true program (i.e., one that the
user approves) in polynomial time. We perform a qualitative evaluation of our
algorithm, showing how it can be used to synthesize a variety of queries on a
database of academic publications.

    

### [[2010.15030] Actris 2.0: Asynchronous Session-Type Based Reasoning in Separation Logic](http://arxiv.org/abs/2010.15030)


  Message passing is a useful abstraction for implementing concurrent programs.
For real-world systems, however, it is often combined with other programming
and concurrency paradigms, such as higher-order functions, mutable state,
shared-memory concurrency, and locks. We present Actris: a logic for proving
functional correctness of programs that use a combination of the aforementioned
features. Actris combines the power of modern concurrent separation logics with
a first-class protocol mechanism -- based on session types -- for reasoning
about message passing in the presence of other concurrency paradigms. We show
that Actris provides a suitable level of abstraction by proving functional
correctness of a variety of examples, including a channel-based merge sort, a
channel-based load-balancing mapper, and a variant of the map-reduce model,
using concise specifications. While Actris was already presented in a
conference paper (POPL'20), this paper expands the prior presentation
significantly. Moreover, it extends Actris to Actris 2.0 with a notion of
subprotocols -- based on session-type subtyping -- that permits additional
flexibility when composing channel endpoints, and that takes full advantage of
the asynchronous semantics of message passing in Actris. Soundness of Actris
2.0 is proven using a model of its protocol mechanism in the Iris framework. We
have mechanised the theory of Actris, together with custom tactics, as well as
all examples in the paper, in the Coq proof assistant.

    

### [[2101.06757] Higher Order Automatic Differentiation of Higher Order Functions](http://arxiv.org/abs/2101.06757)


  We present semantic correctness proofs of automatic differentiation (AD). We
consider a forward-mode AD method on a higher order language with algebraic
data types, and we characterise it as the unique structure preserving macro
given a choice of derivatives for basic operations. We describe a rich
semantics for differentiable programming, based on diffeological spaces. We
show that it interprets our language, and we phrase what it means for the AD
method to be correct with respect to this semantics. We show that our
characterisation of AD gives rise to an elegant semantic proof of its
correctness based on a gluing construction on diffeological spaces. We explain
how this is, in essence, a logical relations argument. Throughout, we show how
the analysis extends to AD methods for computing higher order derivatives using
a Taylor approximation.

    