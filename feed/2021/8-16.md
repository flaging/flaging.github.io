
## 2021-8-16

### [<title>Weighted RMSE Validity with XGBoost - XGBoost</title>](https://discuss.xgboost.ai/t/weighted-rmse-validity-with-xgboost/2433/1)

### [[2108.06172] 5G NB-IoT via low density LEO Constellations](http://arxiv.org/abs/2108.06172)


  5G NB-IoT is seen as a key technology for providing truly ubiquitous, global
5G coverage (1.000.000 devices/km2) for machine type communications in the
internet of things. A non-terrestrial network (NTN) variant of NB-IoT is being
standardized in the 3GPP, which along with inexpensive and non-complex
chip-sets enables the production of competitively priced IoT devices with truly
global coverage. NB-IoT allows for narrowband single carrier transmissions in
the uplink, which improves the uplink link-budget by as much as 16.8 dB over
the 180 [kHz] downlink. This allows for a long range sufficient for ground to
low earth orbit (LEO) communication without the need for complex and expensive
antennas in the IoT devices. In this paper the feasibility of 5G NB-IoT in the
context of low-density constellations of small-satellites carrying
base-stations in LEO is analyzed and required adaptations to NB-IoT are
discussed.

    

### [[2108.06316] Downlink Resource Allocation in Multiuser Cell-free MIMO Networks with User-centric Clustering](http://arxiv.org/abs/2108.06316)


  In this paper, we optimize user scheduling, power allocation and beamforming
in distributed multiple-input multiple-output (MIMO) networks implementing
user-centric clustering. We study both the coherent and non-coherent
transmission modes, formulating a weighted sum rate maximization problem for
each; finding the optimal solution to these problems is known to be NP-hard. We
use tools from fractional programming, block coordinate descent, and
compressive sensing to construct an algorithm that optimizes the beamforming
weights and user scheduling and converges in a smooth non-decreasing pattern.
Channel state information (CSI) being crucial for optimization, we highlight
the importance of employing a low-overhead pilot assignment policy for
scheduling problems. In this regard, we use a variant of hierarchical
agglomerative clustering, which provides a suboptimal, but feasible, pilot
assignment scheme; for our cell-free case, we formulate an area-based pilot
reuse factor. Our results show that our scheme provides large gains in the
long-term network sum spectral efficiency compared to benchmark schemes such as
zero-forcing and conjugate beamforming (with round-robin scheduling)
respectively. Furthermore, the results show the superiority of coherent
transmission compared to the non-coherent mode under ideal and imperfect CSI
for the area-based pilot-reuse factors we consider.

    

### [[2105.11931] Towards Scalable Verification of Deep Reinforcement Learning](http://arxiv.org/abs/2105.11931)


  Deep neural networks (DNNs) have gained significant popularity in recent
years, becoming the state of the art in a variety of domains. In particular,
deep reinforcement learning (DRL) has recently been employed to train DNNs that
realize control policies for various types of real-world systems. In this work,
we present the whiRL 2.0 tool, which implements a new approach for verifying
complex properties of interest for DRL systems. To demonstrate the benefits of
whiRL 2.0, we apply it to case studies from the communication networks domain
that have recently been used to motivate formal verification of DRL systems,
and which exhibit characteristics that are conducive for scalable verification.
We propose techniques for performing k-induction and semi-automated invariant
inference on such systems, and leverage these techniques for proving safety and
liveness properties that were previously impossible to verify due to the
scalability barriers of prior approaches. Furthermore, we show how our proposed
techniques provide insights into the inner workings and the generalizability of
DRL systems. whiRL 2.0 is publicly available online.

    

### [[2108.05916] Alzheimer's Disease Diagnosis via Deep Factorization Machine Models](http://arxiv.org/abs/2108.05916)


  The current state-of-the-art deep neural networks (DNNs) for Alzheimer's
Disease diagnosis use different biomarker combinations to classify patients,
but do not allow extracting knowledge about the interactions of biomarkers.
However, to improve our understanding of the disease, it is paramount to
extract such knowledge from the learned model. In this paper, we propose a Deep
Factorization Machine model that combines the ability of DNNs to learn complex
relationships and the ease of interpretability of a linear model. The proposed
model has three parts: (i) an embedding layer to deal with sparse categorical
data, (ii) a Factorization Machine to efficiently learn pairwise interactions,
and (iii) a DNN to implicitly model higher order interactions. In our
experiments on data from the Alzheimer's Disease Neuroimaging Initiative, we
demonstrate that our proposed model classifies cognitive normal, mild cognitive
impaired, and demented patients more accurately than competing models. In
addition, we show that valuable knowledge about the interactions among
biomarkers can be obtained.

    

### [[2108.05928] Charts and atlases for nonlinear data-driven models of dynamics on manifolds](http://arxiv.org/abs/2108.05928)


  We introduce a method for learning minimal-dimensional dynamical models from
high-dimensional time series data that lie on a low-dimensional manifold, as
arises for many processes. For an arbitrary manifold, there is no smooth global
coordinate representation, so following the formalism of differential topology
we represent the manifold as an atlas of charts. We first partition the data
into overlapping regions. Then undercomplete autoencoders are used to find
low-dimensional coordinate representations for each region. We then use the
data to learn dynamical models in each region, which together yield a global
low-dimensional dynamical model. We apply this method to examples ranging from
simple periodic dynamics to complex, nominally high-dimensional non-periodic
bursting dynamics of the Kuramoto-Sivashinsky equation. We demonstrate that it:
(1) can yield dynamical models of the lowest possible dimension, where previous
methods generally cannot; (2) exhibits computational benefits including
scalability, parallelizability, and adaptivity; and (3) separates state space
into regions of distinct behaviours.

    

### [[2108.05935] Data Quality Toolkit: Automatic assessment of data quality and remediation for machine learning datasets](http://arxiv.org/abs/2108.05935)


  The quality of training data has a huge impact on the efficiency, accuracy
and complexity of machine learning tasks. Various tools and techniques are
available that assess data quality with respect to general cleaning and
profiling checks. However these techniques are not applicable to detect data
issues in the context of machine learning tasks, like noisy labels, existence
of overlapping classes etc. We attempt to re-look at the data quality issues in
the context of building a machine learning pipeline and build a tool that can
detect, explain and remediate issues in the data, and systematically and
automatically capture all the changes applied to the data. We introduce the
Data Quality Toolkit for machine learning as a library of some key quality
metrics and relevant remediation techniques to analyze and enhance the
readiness of structured training datasets for machine learning projects. The
toolkit can reduce the turn-around times of data preparation pipelines and
streamline the data quality assessment process. Our toolkit is publicly
available via IBM API Hub [1] platform, any developer can assess the data
quality using the IBM's Data Quality for AI apis [2]. Detailed tutorials are
also available on IBM Learning Path [3].

    

### [[2108.05940] ST-PCNN: Spatio-Temporal Physics-Coupled Neural Networks for Dynamics Forecasting](http://arxiv.org/abs/2108.05940)


  Ocean current, fluid mechanics, and many other spatio-temporal physical
dynamical systems are essential components of the universe. One key
characteristic of such systems is that certain physics laws -- represented as
ordinary/partial differential equations (ODEs/PDEs) -- largely dominate the
whole process, irrespective of time or location. Physics-informed learning has
recently emerged to learn physics for accurate prediction, but they often lack
a mechanism to leverage localized spatial and temporal correlation or rely on
hard-coded physics parameters. In this paper, we advocate a physics-coupled
neural network model to learn parameters governing the physics of the system,
and further couple the learned physics to assist the learning of recurring
dynamics. A spatio-temporal physics-coupled neural network (ST-PCNN) model is
proposed to achieve three goals: (1) learning the underlying physics
parameters, (2) transition of local information between spatio-temporal
regions, and (3) forecasting future values for the dynamical system. The
physics-coupled learning ensures that the proposed model can be tremendously
improved by using learned physics parameters, and can achieve good long-range
forecasting (e.g., more than 30-steps). Experiments, using simulated and
field-collected ocean current data, validate that ST-PCNN outperforms existing
physics-informed models.

    

### [[2108.05947] Room Classification on Floor Plan Graphs using Graph Neural Networks](http://arxiv.org/abs/2108.05947)


  We present our approach to improve room classification task on floor plan
maps of buildings by representing floor plans as undirected graphs and
leveraging graph neural networks to predict the room categories. Rooms in the
floor plans are represented as nodes in the graph with edges representing their
adjacency in the map. We experiment with House-GAN dataset that consists of
floor plan maps in vector format and train multilayer perceptron and graph
neural networks. Our results show that graph neural networks, specifically
GraphSAGE and Topology Adaptive GCN were able to achieve accuracy of 80% and
81% respectively outperforming baseline multilayer perceptron by more than 15%
margin.

    

### [[2108.05955] Using Machine Learning to Predict Engineering Technology Students' Success with Computer Aided Design](http://arxiv.org/abs/2108.05955)


  Computer-aided design (CAD) programs are essential to engineering as they
allow for better designs through low-cost iterations. While CAD programs are
typically taught to undergraduate students as a job skill, such software can
also help students learn engineering concepts. A current limitation of CAD
programs (even those that are specifically designed for educational purposes)
is that they are not capable of providing automated real-time help to students.
To encourage CAD programs to build in assistance to students, we used data
generated from students using a free, open source CAD software called Aladdin
to demonstrate how student data combined with machine learning techniques can
predict how well a particular student will perform in a design task. We
challenged students to design a house that consumed zero net energy as part of
an introductory engineering technology undergraduate course. Using data from
128 students, along with the scikit-learn Python machine learning library, we
tested our models using both total counts of design actions and sequences of
design actions as inputs. We found that our models using early design sequence
actions are particularly valuable for prediction. Our logistic regression model
achieved a >60% chance of predicting if a student would succeed in designing a
zero net energy house. Our results suggest that it would be feasible for
Aladdin to provide useful feedback to students when they are approximately
halfway through their design. Further improvements to these models could lead
to earlier predictions and thus provide students feedback sooner to enhance
their learning.

    

### [[2108.05969] Scalable3-BO: Big Data meets HPC - A scalable asynchronous parallel high-dimensional Bayesian optimization framework on supercomputers](http://arxiv.org/abs/2108.05969)


  Bayesian optimization (BO) is a flexible and powerful framework that is
suitable for computationally expensive simulation-based applications and
guarantees statistical convergence to the global optimum. While remaining as
one of the most popular optimization methods, its capability is hindered by the
size of data, the dimensionality of the considered problem, and the nature of
sequential optimization. These scalability issues are intertwined with each
other and must be tackled simultaneously. In this work, we propose the
Scalable$^3$-BO framework, which employs sparse GP as the underlying surrogate
model to scope with Big Data and is equipped with a random embedding to
efficiently optimize high-dimensional problems with low effective
dimensionality. The Scalable$^3$-BO framework is further leveraged with
asynchronous parallelization feature, which fully exploits the computational
resource on HPC within a computational budget. As a result, the proposed
Scalable$^3$-BO framework is scalable in three independent perspectives: with
respect to data size, dimensionality, and computational resource on HPC. The
goal of this work is to push the frontiers of BO beyond its well-known
scalability issues and minimize the wall-clock waiting time for optimizing
high-dimensional computationally expensive applications. We demonstrate the
capability of Scalable$^3$-BO with 1 million data points, 10,000-dimensional
problems, with 20 concurrent workers in an HPC environment.

    

### [[2108.05971] Ergonomically Intelligent Physical Human-Robot Interaction: Postural Estimation, Assessment, and Optimization](http://arxiv.org/abs/2108.05971)


  Ergonomics and human comfort are essential concerns in physical human-robot
interaction applications, and common practical methods either fail in
estimating the correct posture due to occlusion or suffer from less accurate
ergonomics models in their postural optimization methods. Instead, we propose a
novel framework for posture estimation, assessment, and optimization for
ergonomically intelligent physical human-robot interaction. We show that we can
estimate human posture solely from the trajectory of the interacting robot. We
propose DULA, a differentiable ergonomics model, and use it in gradient-free
postural optimization for physical human-robot interaction tasks such as
co-manipulation and teleoperation. We evaluate our framework through human and
simulation experiments.

    

### [[2108.05974] An Operator Splitting View of Federated Learning](http://arxiv.org/abs/2108.05974)


  Over the past few years, the federated learning ($\texttt{FL}$) community has
witnessed a proliferation of new $\texttt{FL}$ algorithms. However, our
understating of the theory of $\texttt{FL}$ is still fragmented, and a
thorough, formal comparison of these algorithms remains elusive. Motivated by
this gap, we show that many of the existing $\texttt{FL}$ algorithms can be
understood from an operator splitting point of view. This unification allows us
to compare different algorithms with ease, to refine previous convergence
results and to uncover new algorithmic variants. In particular, our analysis
reveals the vital role played by the step size in $\texttt{FL}$ algorithms. The
unification also leads to a streamlined and economic way to accelerate
$\texttt{FL}$ algorithms, without incurring any communication overhead. We
perform numerical experiments on both convex and nonconvex models to validate
our findings.

    

### [[2108.06011] Datasets for Studying Generalization from Easy to Hard Examples](http://arxiv.org/abs/2108.06011)


  We describe new datasets for studying generalization from easy to hard
examples.

    

### [[2108.06036] An Information-theoretic Perspective of Hierarchical Clustering](http://arxiv.org/abs/2108.06036)


  A combinatorial cost function for hierarchical clustering was introduced by
Dasgupta \cite{dasgupta2016cost}. It has been generalized by Cohen-Addad et al.
\cite{cohen2019hierarchical} to a general form named admissible function. In
this paper, we investigate hierarchical clustering from the
\emph{information-theoretic} perspective and formulate a new objective
function. We also establish the relationship between these two perspectives. In
algorithmic aspect, we get rid of the traditional top-down and bottom-up
frameworks, and propose a new one to stratify the \emph{sparsest} level of a
cluster tree recursively in guide with our objective function. For practical
use, our resulting cluster tree is not binary. Our algorithm called HCSE
outputs a $k$-level cluster tree by a novel and interpretable mechanism to
choose $k$ automatically without any hyper-parameter. Our experimental results
on synthetic datasets show that HCSE has a great advantage in finding the
intrinsic number of hierarchies, and the results on real datasets show that
HCSE also achieves competitive costs over the popular algorithms LOUVAIN and
HLP.

    

### [[2108.06069] Zero-shot Task Transfer for Invoice Extraction via Class-aware QA Ensemble](http://arxiv.org/abs/2108.06069)


  We present VESPA, an intentionally simple yet novel zero-shot system for
layout, locale, and domain agnostic document extraction. In spite of the
availability of large corpora of documents, the lack of labeled and validated
datasets makes it a challenge to discriminatively train document extraction
models for enterprises. We show that this problem can be addressed by simply
transferring the information extraction (IE) task to a natural language
Question-Answering (QA) task without engineering task-specific architectures.
We demonstrate the effectiveness of our system by evaluating on a closed corpus
of real-world retail and tax invoices with multiple complex layouts, domains,
and geographies. The empirical evaluation shows that our system outperforms 4
prominent commercial invoice solutions that use discriminatively trained models
with architectures specifically crafted for invoice extraction. We extracted 6
fields with zero upfront human annotation or training with an Avg. F1 of 87.50.

    

### [[2108.06080] TDM: Trustworthy Decision-Making via Interpretability Enhancement](http://arxiv.org/abs/2108.06080)


  Human-robot interactive decision-making is increasingly becoming ubiquitous,
and trust is an influential factor in determining the reliance on autonomy.
However, it is not reasonable to trust systems that are beyond our
comprehension, and typical machine learning and data-driven decision-making are
black-box paradigms that impede interpretability. Therefore, it is critical to
establish computational trustworthy decision-making mechanisms enhanced by
interpretability-aware strategies. To this end, we propose a Trustworthy
Decision-Making (TDM) framework, which integrates symbolic planning into
sequential decision-making. The framework learns interpretable subtasks that
result in a complex, higher-level composite task that can be formally evaluated
using the proposed trust metric. TDM enables the subtask-level interpretability
by design and converges to an optimal symbolic plan from the learned subtasks.
Moreover, a TDM-based algorithm is introduced to demonstrate the unification of
symbolic planning with other sequential-decision making algorithms, reaping the
benefits of both. Experimental results validate the effectiveness of
trust-score-based planning while improving the interpretability of subtasks.

    

### [[2108.06084] Curriculum Learning: A Regularization Method for Efficient and Stable Billion-Scale GPT Model Pre-Training](http://arxiv.org/abs/2108.06084)


  Recent works have demonstrated great success in training high-capacity
autoregressive language models (GPT, GPT-2, GPT-3) on a huge amount of
unlabeled text corpus for text generation. Despite showing great results, this
generates two training efficiency challenges. First, training large corpora can
be extremely timing consuming, and how to present training samples to the model
to improve the token-wise convergence speed remains a challenging and open
question. Second, many of these large models have to be trained with hundreds
or even thousands of processors using data-parallelism with a very large batch
size. Despite of its better compute efficiency, it has been observed that
large-batch training often runs into training instability issue or converges to
solutions with bad generalization performance. To overcome these two
challenges, we present a study of a curriculum learning based approach, which
helps improves the pre-training convergence speed of autoregressive models.
More importantly, we find that curriculum learning, as a regularization method,
exerts a gradient variance reduction effect and enables to train autoregressive
models with much larger batch sizes and learning rates without training
instability, further improving the training speed. Our evaluations demonstrate
that curriculum learning enables training GPT-2 models (with up to 1.5B
parameters) with 8x larger batch size and 4x larger learning rate, whereas the
baseline approach struggles with training divergence. To achieve the same
validation perplexity targets during pre-training, curriculum learning reduces
the required number of tokens and wall clock time by up to 59% and 54%,
respectively. To achieve the same or better zero-shot WikiText-103/LAMBADA
evaluation results at the end of pre-training, curriculum learning reduces the
required number of tokens and wall clock time by up to 13% and 61%,
respectively.

    

### [[2108.06094] Multi-Stage Graph Peeling Algorithm for Probabilistic Core Decomposition](http://arxiv.org/abs/2108.06094)


  Mining dense subgraphs where vertices connect closely with each other is a
common task when analyzing graphs. A very popular notion in subgraph analysis
is core decomposition. Recently, Esfahani et al. presented a probabilistic core
decomposition algorithm based on graph peeling and Central Limit Theorem (CLT)
that is capable of handling very large graphs. Their proposed peeling algorithm
(PA) starts from the lowest degree vertices and recursively deletes these
vertices, assigning core numbers, and updating the degree of neighbour vertices
until it reached the maximum core. However, in many applications, particularly
in biology, more valuable information can be obtained from dense
sub-communities and we are not interested in small cores where vertices do not
interact much with others. To make the previous PA focus more on dense
subgraphs, we propose a multi-stage graph peeling algorithm (M-PA) that has a
two-stage data screening procedure added before the previous PA. After removing
vertices from the graph based on the user-defined thresholds, we can reduce the
graph complexity largely and without affecting the vertices in subgraphs that
we are interested in. We show that M-PA is more efficient than the previous PA
and with the properly set filtering threshold, can produce very similar if not
identical dense subgraphs to the previous PA (in terms of graph density and
clustering coefficient).

    

### [[2108.06098] FedPara: Low-rank Hadamard Product Parameterization for Efficient Federated Learning](http://arxiv.org/abs/2108.06098)


  To overcome the burdens on frequent model uploads and downloads during
federated learning (FL), we propose a communication-efficient
re-parameterization, FedPara. Our method re-parameterizes the model's layers
using low-rank matrices or tensors followed by the Hadamard product. Different
from the conventional low-rank parameterization, our method is not limited to
low-rank constraints. Thereby, our FedPara has a larger capacity than the
low-rank one, even with the same number of parameters. It can achieve
comparable performance to the original models while requiring 2.8 to 10.1 times
lower communication costs than the original models, which is not achievable by
the traditional low-rank parameterization. Moreover, the efficiency can be
further improved by combining our method and other efficient FL techniques
because our method is compatible with others. We also extend our method to a
personalized FL application, pFedPara, which separates parameters into global
and local ones. We show that pFedPara outperforms competing personalized FL
methods with more than three times fewer parameters.

    

### [[2108.06129] Learning Transferable Parameters for Unsupervised Domain Adaptation](http://arxiv.org/abs/2108.06129)


  Unsupervised domain adaptation (UDA) enables a learning machine to adapt from
a labeled source domain to an unlabeled domain under the distribution shift.
Thanks to the strong representation ability of deep neural networks, recent
remarkable achievements in UDA resort to learning domain-invariant features.
Intuitively, the hope is that a good feature representation, together with the
hypothesis learned from the source domain, can generalize well to the target
domain. However, the learning processes of domain-invariant features and source
hypothesis inevitably involve domain-specific information that would degrade
the generalizability of UDA models on the target domain. In this paper,
motivated by the lottery ticket hypothesis that only partial parameters are
essential for generalization, we find that only partial parameters are
essential for learning domain-invariant information and generalizing well in
UDA. Such parameters are termed transferable parameters. In contrast, the other
parameters tend to fit domain-specific details and often fail to generalize,
which we term as untransferable parameters. Driven by this insight, we propose
Transferable Parameter Learning (TransPar) to reduce the side effect brought by
domain-specific information in the learning process and thus enhance the
memorization of domain-invariant information. Specifically, according to the
distribution discrepancy degree, we divide all parameters into transferable and
untransferable ones in each training iteration. We then perform separate
updates rules for the two types of parameters. Extensive experiments on image
classification and regression tasks (keypoint detection) show that TransPar
outperforms prior arts by non-trivial margins. Moreover, experiments
demonstrate that TransPar can be integrated into the most popular deep UDA
networks and be easily extended to handle any data distribution shift
scenarios.

    

### [[2108.06148] Q-Mixing Network for Multi-Agent Pathfinding in Partially Observable Grid Environments](http://arxiv.org/abs/2108.06148)


  In this paper, we consider the problem of multi-agent navigation in partially
observable grid environments. This problem is challenging for centralized
planning approaches as they, typically, rely on the full knowledge of the
environment. We suggest utilizing the reinforcement learning approach when the
agents, first, learn the policies that map observations to actions and then
follow these policies to reach their goals. To tackle the challenge associated
with learning cooperative behavior, i.e. in many cases agents need to yield to
each other to accomplish a mission, we use a mixing Q-network that complements
learning individual policies. In the experimental evaluation, we show that such
approach leads to plausible results and scales well to large number of agents.

    

### [[2108.06156] EEEA-Net: An Early Exit Evolutionary Neural Architecture Search](http://arxiv.org/abs/2108.06156)


  The goals of this research were to search for Convolutional Neural Network
(CNN) architectures, suitable for an on-device processor with limited computing
resources, performing at substantially lower Network Architecture Search (NAS)
costs. A new algorithm entitled an Early Exit Population Initialisation (EE-PI)
for Evolutionary Algorithm (EA) was developed to achieve both goals. The EE-PI
reduces the total number of parameters in the search process by filtering the
models with fewer parameters than the maximum threshold. It will look for a new
model to replace those models with parameters more than the threshold. Thereby,
reducing the number of parameters, memory usage for model storage and
processing time while maintaining the same performance or accuracy. The search
time was reduced to 0.52 GPU day. This is a huge and significant achievement
compared to the NAS of 4 GPU days achieved using NSGA-Net, 3,150 GPU days by
the AmoebaNet model, and the 2,000 GPU days by the NASNet model. As well, Early
Exit Evolutionary Algorithm networks (EEEA-Nets) yield network architectures
with minimal error and computational cost suitable for a given dataset as a
class of network algorithms. Using EEEA-Net on CIFAR-10, CIFAR-100, and
ImageNet datasets, our experiments showed that EEEA-Net achieved the lowest
error rate among state-of-the-art NAS models, with 2.46% for CIFAR-10, 15.02%
for CIFAR-100, and 23.8% for ImageNet dataset. Further, we implemented this
image recognition architecture for other tasks, such as object detection,
semantic segmentation, and keypoint detection tasks, and, in our experiments,
EEEA-Net-C2 outperformed MobileNet-V3 on all of these various tasks. (The
algorithm code is available at this https URL).

    

### [[2108.06158] Adaptive Positive-Unlabelled Learning via Markov Diffusion](http://arxiv.org/abs/2108.06158)


  Positive-Unlabelled (PU) learning is the machine learning setting in which
only a set of positive instances are labelled, while the rest of the data set
is unlabelled. The unlabelled instances may be either unspecified positive
samples or true negative samples. Over the years, many solutions have been
proposed to deal with PU learning. Some techniques consider the unlabelled
samples as negative ones, reducing the problem to a binary classification with
a noisy negative set, while others aim to detect sets of possible negative
examples to later apply a supervised machine learning strategy (two-step
techniques). The approach proposed in this work falls in the latter category
and works in a semi-supervised fashion: motivated and inspired by previous
works, a Markov diffusion process with restart is used to assign pseudo-labels
to unlabelled instances. Afterward, a machine learning model, exploiting the
newly assigned classes, is trained. The principal aim of the algorithm is to
identify a set of instances which are likely to contain positive instances that
were originally unlabelled.

    

### [[2108.06159] Robustness testing of AI systems: A case study for traffic sign recognition](http://arxiv.org/abs/2108.06159)


  In the last years, AI systems, in particular neural networks, have seen a
tremendous increase in performance, and they are now used in a broad range of
applications. Unlike classical symbolic AI systems, neural networks are trained
using large data sets and their inner structure containing possibly billions of
parameters does not lend itself to human interpretation. As a consequence, it
is so far not feasible to provide broad guarantees for the correct behaviour of
neural networks during operation if they process input data that significantly
differ from those seen during training. However, many applications of AI
systems are security- or safety-critical, and hence require obtaining
statements on the robustness of the systems when facing unexpected events,
whether they occur naturally or are induced by an attacker in a targeted way.
As a step towards developing robust AI systems for such applications, this
paper presents how the robustness of AI systems can be practically examined and
which methods and metrics can be used to do so. The robustness testing
methodology is described and analysed for the example use case of traffic sign
recognition in autonomous driving.

    

### [[2108.06167] Follow the Prophet: Accurate Online Conversion Rate Prediction in the Face of Delayed Feedback](http://arxiv.org/abs/2108.06167)


  The delayed feedback problem is one of the imperative challenges in online
advertising, which is caused by the highly diversified feedback delay of a
conversion varying from a few minutes to several days. It is hard to design an
appropriate online learning system under these non-identical delay for
different types of ads and users. In this paper, we propose to tackle the
delayed feedback problem in online advertising by "Following the Prophet" (FTP
for short). The key insight is that, if the feedback came instantly for all the
logged samples, we could get a model without delayed feedback, namely the
"prophet". Although the prophet cannot be obtained during online learning, we
show that we could predict the prophet's predictions by an aggregation policy
on top of a set of multi-task predictions, where each task captures the
feedback patterns of different periods. We propose the objective and
optimization approach for the policy, and use the logged data to imitate the
prophet. Extensive experiments on three real-world advertising datasets show
that our method outperforms the previous state-of-the-art baselines.

    

### [[2108.06181] Detecting socially interacting groups using f-formation: A survey of taxonomy, methods, datasets, applications, challenges, and future research directions](http://arxiv.org/abs/2108.06181)


  Robots in our daily surroundings are increasing day by day. Their usability
and acceptability largely depend on their explicit and implicit interaction
capability with fellow human beings. As a result, social behavior is one of the
most sought-after qualities that a robot can possess. However, there is no
specific aspect and/or feature that defines socially acceptable behavior and it
largely depends on the situation, application, and society. In this article, we
investigate one such social behavior for collocated robots. Imagine a group of
people is interacting with each other and we want to join the group. We as
human beings do it in a socially acceptable manner, i.e., within the group, we
do position ourselves in such a way that we can participate in the group
activity without disturbing/obstructing anybody. To possess such a quality,
first, a robot needs to determine the formation of the group and then determine
a position for itself, which we humans do implicitly. The theory of f-formation
can be utilized for this purpose. As the types of formations can be very
diverse, detecting the social groups is not a trivial task. In this article, we
provide a comprehensive survey of the existing work on social interaction and
group detection using f-formation for robotics and other applications. We also
put forward a novel holistic survey framework combining all the possible
concerns and modules relevant to this problem. We define taxonomies based on
methods, camera views, datasets, detection capabilities and scale, evaluation
approaches, and application areas. We discuss certain open challenges and
limitations in current literature along with possible future research
directions based on this framework. In particular, we discuss the existing
methods/techniques and their relative merits and demerits, applications, and
provide a set of unsolved but relevant problems in this domain.

    

### [[2108.06197] A Comparison of Latent Semantic Analysis and Correspondence Analysis for Text Mining](http://arxiv.org/abs/2108.06197)


  Both latent semantic analysis (LSA) and correspondence analysis (CA) use a
singular value decomposition (SVD) for dimensionality reduction. In this
article, LSA and CA are compared from a theoretical point of view and applied
in both a toy example and an authorship attribution example. In text mining
interest goes out to the relationships among documents and terms: for example,
what terms are more often used in what documents. However, the LSA solution
displays a mix of marginal effects and these relationships. It appears that CA
has more attractive properties than LSA. One such property is that, in CA, the
effect of the margins is effectively eliminated, so that the CA solution is
optimally suited to focus on the relationships among documents and terms. Three
mechanisms are distinguished to weight documents and terms, and a unifying
framework is proposed that includes these three mechanisms and includes both CA
and LSA as special cases. In the authorship attribution example, the national
anthem of the Netherlands, the application of the discussed methods is
illustrated.

    

### [[2108.06201] Data-driven advice for interpreting local and global model predictions in bioinformatics problems](http://arxiv.org/abs/2108.06201)


  Tree-based algorithms such as random forests and gradient boosted trees
continue to be among the most popular and powerful machine learning models used
across multiple disciplines. The conventional wisdom of estimating the impact
of a feature in tree based models is to measure the \textit{node-wise reduction
of a loss function}, which (i) yields only global importance measures and (ii)
is known to suffer from severe biases. Conditional feature contributions (CFCs)
provide \textit{local}, case-by-case explanations of a prediction by following
the decision path and attributing changes in the expected output of the model
to each feature along the path. However, Lundberg et al. pointed out a
potential bias of CFCs which depends on the distance from the root of a tree.
The by now immensely popular alternative, SHapley Additive exPlanation (SHAP)
values appear to mitigate this bias but are computationally much more
expensive. Here we contribute a thorough comparison of the explanations
computed by both methods on a set of 164 publicly available classification
problems in order to provide data-driven algorithm recommendations to current
researchers. For random forests, we find extremely high similarities and
correlations of both local and global SHAP values and CFC scores, leading to
very similar rankings and interpretations. Analogous conclusions hold for the
fidelity of using global feature importance scores as a proxy for the
predictive power associated with each feature.

    

### [[2108.06206] An Intelligent Recommendation-cum-Reminder System](http://arxiv.org/abs/2108.06206)


  Intelligent recommendation and reminder systems are the need of the
fast-pacing life. Current intelligent systems such as Siri, Google Assistant,
Microsoft Cortona, etc., have limited capability. For example, if you want to
wake up at 6 am because you have an upcoming trip, you have to set the alarm
manually. Besides, these systems do not recommend or remind what else to carry,
such as carrying an umbrella during a likely rain. The present work proposes a
system that takes an email as input and returns a recommendation-cumreminder
list. As a first step, we parse the emails, recognize the entities using named
entity recognition (NER). In the second step, information retrieval over the
web is done to identify nearby places, climatic conditions, etc. Imperative
sentences from the reviews of all places are extracted and passed to the object
extraction module. The main challenge lies in extracting the objects (items) of
interest from the review. To solve it, a modified Machine Reading
Comprehension-NER (MRC-NER) model is trained to tag objects of interest by
formulating annotation rules as a query. The objects so found are recommended
to the user one day in advance. The final reminder list of objects is pruned by
our proposed model for tracking objects kept during the "packing activity."
Eventually, when the user leaves for the event/trip, an alert is sent
containing the reminding list items. Our approach achieves superior performance
compared to several baselines by as much as 30% on recall and 10% on precision.

    

### [[2108.06208] LT-OCF: Learnable-Time ODE-based Collaborative Filtering](http://arxiv.org/abs/2108.06208)


  Collaborative filtering (CF) is a long-standing problem of recommender
systems. Many novel methods have been proposed, ranging from classical matrix
factorization to recent graph convolutional network-based approaches. After
recent fierce debates, researchers started to focus on linear graph
convolutional networks (GCNs) with a layer combination, which show
state-of-the-art accuracy in many datasets. In this work, we extend them based
on neural ordinary differential equations (NODEs), because the linear GCN
concept can be interpreted as a differential equation, and present the method
of Learnable-Time ODE-based Collaborative Filtering (LT-OCF). The main novelty
in our method is that after redesigning linear GCNs on top of the NODE regime,
i) we learn the optimal architecture rather than relying on manually designed
ones, ii) we learn smooth ODE solutions that are considered suitable for CF,
and iii) we test with various ODE solvers that internally build a diverse set
of neural network connections. We also present a novel training method
specialized to our method. In our experiments with three benchmark datasets,
Gowalla, Yelp2018, and Amazon-Book, our method consistently shows better
accuracy than existing methods, e.g., a recall of 0.0411 by LightGCN vs. 0.0442
by LT-OCF and an NDCG of 0.0315 by LightGCN vs. 0.0341 by LT-OCF in
Amazon-Book. One more important discovery in our experiments that is worth
mentioning is that our best accuracy was achieved by dense connections rather
than linear connections.

    

### [[2108.06209] W2v-BERT: Combining Contrastive Learning and Masked Language Modeling for Self-Supervised Speech Pre-Training](http://arxiv.org/abs/2108.06209)


  Motivated by the success of masked language modeling~(MLM) in pre-training
natural language processing models, we propose w2v-BERT that explores MLM for
self-supervised speech representation learning. w2v-BERT is a framework that
combines contrastive learning and MLM, where the former trains the model to
discretize input continuous speech signals into a finite set of discriminative
speech tokens, and the latter trains the model to learn contextualized speech
representations via solving a masked prediction task consuming the discretized
tokens. In contrast to existing MLM-based speech pre-training frameworks such
as HuBERT, which relies on an iterative re-clustering and re-training process,
or vq-wav2vec, which concatenates two separately trained modules, w2v-BERT can
be optimized in an end-to-end fashion by solving the two self-supervised
tasks~(the contrastive task and MLM) simultaneously. Our experiments show that
w2v-BERT achieves competitive results compared to current state-of-the-art
pre-trained models on the LibriSpeech benchmarks when using the Libri-Light~60k
corpus as the unsupervised data. In particular, when compared to published
models such as conformer-based wav2vec~2.0 and HuBERT, our model shows~5\%
to~10\% relative WER reduction on the test-clean and test-other subsets. When
applied to the Google's Voice Search traffic dataset, w2v-BERT outperforms our
internal conformer-based wav2vec~2.0 by more than~30\% relatively.

    

### [[2108.06210] Recommending Insurance products by using Users' Sentiments](http://arxiv.org/abs/2108.06210)


  In today's tech-savvy world every industry is trying to formulate methods for
recommending products by combining several techniques and algorithms to form a
pool that would bring forward the most enhanced models for making the
predictions. Building on these lines is our paper focused on the application of
sentiment analysis for recommendation in the insurance domain. We tried
building the following Machine Learning models namely, Logistic Regression,
Multinomial Naive Bayes, and the mighty Random Forest for analyzing the
polarity of a given feedback line given by a customer. Then we used this
polarity along with other attributes like Age, Gender, Locality, Income, and
the list of other products already purchased by our existing customers as input
for our recommendation model. Then we matched the polarity score along with the
user's profiles and generated the list of insurance products to be recommended
in descending order. Despite our model's simplicity and the lack of the key
data sets, the results seemed very logical and realistic. So, by developing the
model with more enhanced methods and with access to better and true data
gathered from an insurance industry may be the sector could be very well
benefitted from the amalgamation of sentiment analysis with a recommendation.

    

### [[2108.06215] Sentiment Analysis of the COVID-related r/Depression Posts](http://arxiv.org/abs/2108.06215)


 this http URL is a popular social media platform among young people. Reddit
users share their stories to seek support from other users, especially during
the Covid-19 pandemic. Messages posted on Reddit and their content have
provided researchers with opportunity to analyze public concerns. In this
study, we analyzed sentiments of COVID-related messages posted on r/Depression.
Our study poses the following questions: a) What are the common topics that the
Reddit users discuss? b) Can we use these topics to classify sentiments of the
posts? c) What matters concern people more during the pandemic?
Key Words: Sentiment Classification, Depression, COVID-19, Reddit, LDA, BERT

    

### [[2108.06217] Beyond Fairness Metrics: Roadblocks and Challenges for Ethical AI in Practice](http://arxiv.org/abs/2108.06217)


  We review practical challenges in building and deploying ethical AI at the
scale of contemporary industrial and societal uses. Apart from the purely
technical concerns that are the usual focus of academic research, the
operational challenges of inconsistent regulatory pressures, conflicting
business goals, data quality issues, development processes, systems integration
practices, and the scale of deployment all conspire to create new ethical
risks. Such ethical concerns arising from these practical considerations are
not adequately addressed by existing research results. We argue that a holistic
consideration of ethics in the development and deployment of AI systems is
necessary for building ethical AI in practice, and exhort researchers to
consider the full operational contexts of AI systems when assessing ethical
risks.

    

### [[2108.06227] SimCVD: Simple Contrastive Voxel-Wise Representation Distillation for Semi-Supervised Medical Image Segmentation](http://arxiv.org/abs/2108.06227)


  Automated segmentation in medical image analysis is a challenging task that
requires a large amount of manually labeled data. However, most existing
learning-based approaches usually suffer from limited manually annotated
medical data, which poses a major practical problem for accurate and robust
medical image segmentation. In addition, most existing semi-supervised
approaches are usually not robust compared with the supervised counterparts,
and also lack explicit modeling of geometric structure and semantic
information, both of which limit the segmentation accuracy. In this work, we
present SimCVD, a simple contrastive distillation framework that significantly
advances state-of-the-art voxel-wise representation learning. We first describe
an unsupervised training strategy, which takes two views of an input volume and
predicts their signed distance maps of object boundaries in a contrastive
objective, with only two independent dropout as mask. This simple approach
works surprisingly well, performing on the same level as previous fully
supervised methods with much less labeled data. We hypothesize that dropout can
be viewed as a minimal form of data augmentation and makes the network robust
to representation collapse. Then, we propose to perform structural distillation
by distilling pair-wise similarities. We evaluate SimCVD on two popular
datasets: the Left Atrial Segmentation Challenge (LA) and the NIH pancreas CT
dataset. The results on the LA dataset demonstrate that, in two types of
labeled ratios (i.e., 20% and 10%), SimCVD achieves an average Dice score of
90.85% and 89.03% respectively, a 0.91% and 2.22% improvement compared to
previous best results. Our method can be trained in an end-to-end fashion,
showing the promise of utilizing SimCVD as a general framework for downstream
tasks, such as medical image synthesis and registration.

    

### [[2108.06228] One-shot Transfer Learning for Population Mapping](http://arxiv.org/abs/2108.06228)


  Fine-grained population distribution data is of great importance for many
applications, e.g., urban planning, traffic scheduling, epidemic modeling, and
risk control. However, due to the limitations of data collection, including
infrastructure density, user privacy, and business security, such fine-grained
data is hard to collect and usually, only coarse-grained data is available.
Thus, obtaining fine-grained population distribution from coarse-grained
distribution becomes an important problem. To complete this task, existing
methods mainly rely on sufficient fine-grained ground truth for training, which
is not often available. This limits the applications of these methods and
brings the necessity to transfer knowledge from data-sufficient cities to
data-scarce cities.
In knowledge transfer scenario, we employ single reference fine-grained
ground truth in the target city as the ground truth to inform the large-scale
urban structure and support the knowledge transfer in the target city. By this
approach, we transform the fine-grained population mapping problem into a
one-shot transfer learning problem for population mapping task.
In this paper, we propose a one-shot transfer learning framework, PSRNet, to
transfer spatial-temporal knowledge across cities in fine-grained population
mapping task from the view of network structure, data, and optimization.
Experiments on real-life datasets of 4 cities demonstrate that PSRNet has
significant advantages over 8 baselines by reducing RMSE and MAE for more than
25%. Our code and datasets are released in Github.

    

### [[2108.06231] Online Fairness-Aware Learning with Imbalanced Data Streams](http://arxiv.org/abs/2108.06231)


  Data-driven learning algorithms are employed in many online applications, in
which data become available over time, like network monitoring, stock price
prediction, job applications, etc. The underlying data distribution might
evolve over time calling for model adaptation as new instances arrive and old
instances become obsolete. In such dynamic environments, the so-called data
streams, fairness-aware learning cannot be considered as a one-off requirement,
but rather it should comprise a continual requirement over the stream. Recent
fairness-aware stream classifiers ignore the problem of class imbalance, which
manifests in many real-life applications, and mitigate discrimination mainly
because they "reject" minority instances at large due to their inability to
effectively learn all classes.
In this work, we propose \ours, an online fairness-aware approach that
maintains a valid and fair classifier over the stream. \ours~is an online
boosting approach that changes the training distribution in an online fashion
by monitoring stream's class imbalance and tweaks its decision boundary to
mitigate discriminatory outcomes over the stream. Experiments on 8 real-world
and 1 synthetic datasets from different domains with varying class imbalance
demonstrate the superiority of our method over state-of-the-art fairness-aware
stream approaches with a range (relative) increase [11.2\%-14.2\%] in balanced
accuracy, [22.6\%-31.8\%] in gmean, [42.5\%-49.6\%] in recall, [14.3\%-25.7\%]
in kappa and [89.4\%-96.6\%] in statistical parity (fairness).

    

### [[2108.06238] Jasmine: A New Active Learning Approach to Combat Cybercrime](http://arxiv.org/abs/2108.06238)


  Over the past decade, the advent of cybercrime has accelarated the research
on cybersecurity. However, the deployment of intrusion detection methods falls
short. One of the reasons for this is the lack of realistic evaluation
datasets, which makes it a challenge to develop techniques and compare them.
This is caused by the large amounts of effort it takes for a cyber analyst to
classify network connections. This has raised the need for methods (i) that can
learn from small sets of labeled data, (ii) that can make predictions on large
sets of unlabeled data, and (iii) that request the label of only specially
selected unlabeled data instances. Hence, Active Learning (AL) methods are of
interest. These approaches choose speci?fic unlabeled instances by a query
function that are expected to improve overall classi?cation performance. The
resulting query observations are labeled by a human expert and added to the
labeled set.
In this paper, we propose a new hybrid AL method called Jasmine. Firstly, it
determines how suitable each observation is for querying, i.e., how likely it
is to enhance classi?cation. These properties are the uncertainty score and
anomaly score. Secondly, Jasmine introduces dynamic updating. This allows the
model to adjust the balance between querying uncertain, anomalous and randomly
selected observations. To this end, Jasmine is able to learn the best query
strategy during the labeling process. This is in contrast to the other AL
methods in cybersecurity that all have static, predetermined query functions.
We show that dynamic updating, and therefore Jasmine, is able to consistently
obtain good and more robust results than querying only uncertainties, only
anomalies or a ?fixed combination of the two.

    

### [[2108.06249] MIND - Mainstream and Independent News Documents Corpus](http://arxiv.org/abs/2108.06249)


  This paper presents and characterizes MIND, a new Portuguese corpus comprised
of different types of articles collected from online mainstream and alternative
media sources, over a 10-month period. The articles in the corpus are organized
into five collections: facts, opinions, entertainment, satires, and conspiracy
theories. Throughout this paper, we explain how the data collection process was
conducted, and present a set of linguistic metrics that allow us to perform a
preliminary characterization of the texts included in the corpus. Also, we
deliver an analysis of the most frequent topics in the corpus, and discuss the
main differences and similarities among the collections considered. Finally, we
enumerate some tasks and applications that could benefit from this corpus, in
particular the ones (in)directly related to misinformation detection. Overall,
our contribution of a corpus and initial analysis are designed to support
future exploratory news studies, and provide a better insight into
misinformation.

    

### [[2108.06264] Bridging the gap between emotion and joint action](http://arxiv.org/abs/2108.06264)


  Our daily human life is filled with a myriad of joint action moments, be it
children playing, adults working together (i.e., team sports), or strangers
navigating through a crowd. Joint action brings individuals (and embodiment of
their emotions) together, in space and in time. Yet little is known about how
individual emotions propagate through embodied presence in a group, and how
joint action changes individual emotion. In fact, the multi-agent component is
largely missing from neuroscience-based approaches to emotion, and reversely
joint action research has not found a way yet to include emotion as one of the
key parameters to model socio-motor interaction. In this review, we first
identify the gap and then stockpile evidence showing strong entanglement
between emotion and acting together from various branches of sciences. We
propose an integrative approach to bridge the gap, highlight five research
avenues to do so in behavioral neuroscience and digital sciences, and address
some of the key challenges in the area faced by modern societies.

    

### [[2108.06265] A reduced-order modeling framework for simulating signatures of faults in a bladed disk](http://arxiv.org/abs/2108.06265)


  This paper reports a reduced-order modeling framework of bladed disks on a
rotating shaft to simulate the vibration signature of faults like cracks in
different components aiming towards simulated data-driven machine learning. We
have employed lumped and one-dimensional analytical models of the subcomponents
for better insight into the complex dynamic response. The framework seeks to
address some of the challenges encountered in analyzing and optimizing fault
detection and identification schemes for health monitoring of rotating
turbomachinery, including aero-engines. We model the bladed disks and shafts by
combining lumped elements and one-dimensional finite elements, leading to a
coupled system. The simulation results are in good agreement with previously
published data. We model the cracks in a blade analytically with their
effective reduced stiffness approximation. Multiple types of faults are
modeled, including cracks in the blades of single and two-stage bladed disks,
Fan Blade Off (FBO), and Foreign Object Damage (FOD). We have applied
aero-engine operational loading conditions to simulate realistic scenarios of
online health monitoring. The proposed reduced-order simulation framework will
have applications in probabilistic signal modeling, machine learning toward
fault signature identification, and parameter estimation with measured
vibration signals.

    

### [[2108.06266] Safe Learning in Robotics: From Learning-Based Control to Safe Reinforcement Learning](http://arxiv.org/abs/2108.06266)


  The last half-decade has seen a steep rise in the number of contributions on
safe learning methods for real-world robotic deployments from both the control
and reinforcement learning communities. This article provides a concise but
holistic review of the recent advances made in using machine learning to
achieve safe decision making under uncertainties, with a focus on unifying the
language and frameworks used in control theory and reinforcement learning
research. Our review includes: learning-based control approaches that safely
improve performance by learning the uncertain dynamics, reinforcement learning
approaches that encourage safety or robustness, and methods that can formally
certify the safety of a learned control policy. As data- and learning-based
robot control methods continue to gain traction, researchers must understand
when and how to best leverage them in real-world scenarios where safety is
imperative, such as when operating in close proximity to humans. We highlight
some of the open challenges that will drive the field of robot learning in the
coming years, and emphasize the need for realistic physics-based benchmarks to
facilitate fair comparisons between control and reinforcement learning
approaches.

    

### [[2108.06277] Towards Structured Dynamic Sparse Pre-Training of BERT](http://arxiv.org/abs/2108.06277)


  Identifying algorithms for computational efficient unsupervised training of
large language models is an important and active area of research. In this
work, we develop and study a straightforward, dynamic always-sparse
pre-training approach for BERT language modeling task, which leverages periodic
compression steps based on magnitude pruning followed by random parameter
re-allocation. This approach enables us to achieve Pareto improvements in terms
of the number of floating-point operations (FLOPs) over statically sparse and
dense models across a broad spectrum of network sizes. Furthermore, we
demonstrate that training remains FLOP-efficient when using coarse-grained
block sparsity, making it particularly promising for efficient execution on
modern hardware accelerators.

    

### [[2108.06280] Understanding Structural Vulnerability in Graph Convolutional Networks](http://arxiv.org/abs/2108.06280)


  Recent studies have shown that Graph Convolutional Networks (GCNs) are
vulnerable to adversarial attacks on the graph structure. Although multiple
works have been proposed to improve their robustness against such structural
adversarial attacks, the reasons for the success of the attacks remain unclear.
In this work, we theoretically and empirically demonstrate that structural
adversarial examples can be attributed to the non-robust aggregation scheme
(i.e., the weighted mean) of GCNs. Specifically, our analysis takes advantage
of the breakdown point which can quantitatively measure the robustness of
aggregation schemes. The key insight is that weighted mean, as the basic design
of GCNs, has a low breakdown point and its output can be dramatically changed
by injecting a single edge. We show that adopting the aggregation scheme with a
high breakdown point (e.g., median or trimmed mean) could significantly enhance
the robustness of GCNs against structural attacks. Extensive experiments on
four real-world datasets demonstrate that such a simple but effective method
achieves the best robustness performance compared to state-of-the-art models.

    

### [[2108.06283] Random Subspace Mixture Models for Interpretable Anomaly Detection](http://arxiv.org/abs/2108.06283)


  We present a new subspace-based method to construct probabilistic models for
high-dimensional data and highlight its use in anomaly detection. The approach
is based on a statistical estimation of probability density using densities of
random subspaces combined with geometric averaging. In selecting random
subspaces, equal representation of each attribute is used to ensure correct
statistical limits. Gaussian mixture models (GMMs) are used to create the
probability densities for each subspace with techniques included to mitigate
singularities allowing for the ability to handle both numerical and categorial
attributes. The number of components for each GMM is determined automatically
through Bayesian information criterion to prevent overfitting. The proposed
algorithm attains competitive AUC scores compared with prominent algorithms
against benchmark anomaly detection datasets with the added benefits of being
simple, scalable, and interpretable.

    

### [[2108.06302] Context Aware Object Geotagging](http://arxiv.org/abs/2108.06302)


  Localization of street objects from images has gained a lot of attention in
recent years. We propose an approach to improve asset geolocation from street
view imagery by enhancing the quality of the metadata associated with the
images using Structure from Motion. The predicted object geolocation is further
refined by imposing contextual geographic information extracted from
OpenStreetMap. Our pipeline is validated experimentally against the state of
the art approaches for geotagging traffic lights.

    

### [[2108.06309] Spatio-Temporal Split Learning](http://arxiv.org/abs/2108.06309)


  This paper proposes a novel split learning framework with multiple
end-systems in order to realize privacypreserving deep neural network
computation. In conventional split learning frameworks, deep neural network
computation is separated into multiple computing systems for hiding entire
network architectures. In our proposed framework, multiple computing
end-systems are sharing one centralized server in split learning computation,
where the multiple end-systems are with input and first hidden layers and the
centralized server is with the other hidden layers and output layer. This
framework, which is called as spatio-temporal split learning, is spatially
separated for gathering data from multiple end-systems and also temporally
separated due to the nature of split learning. Our performance evaluation
verifies that our proposed framework shows nearoptimal accuracy while
preserving data privacy.

    

### [[2108.06310] MeetSum: Transforming Meeting Transcript Summarization using Transformers!](http://arxiv.org/abs/2108.06310)


  Creating abstractive summaries from meeting transcripts has proven to be
challenging due to the limited amount of labeled data available for training
neural network models. Moreover, Transformer-based architectures have proven to
beat state-of-the-art models in summarizing news data. In this paper, we
utilize a Transformer-based Pointer Generator Network to generate abstract
summaries for meeting transcripts. This model uses 2 LSTMs as an encoder and a
decoder, a Pointer network which copies words from the inputted text, and a
Generator network to produce out-of-vocabulary words (hence making the summary
abstractive). Moreover, a coverage mechanism is used to avoid repetition of
words in the generated summary. First, we show that training the model on a
news summary dataset and using zero-shot learning to test it on the meeting
dataset proves to produce better results than training it on the AMI meeting
dataset. Second, we show that training this model first on out-of-domain data,
such as the CNN-Dailymail dataset, followed by a fine-tuning stage on the AMI
meeting dataset is able to improve the performance of the model significantly.
We test our model on a testing set from the AMI dataset and report the ROUGE-2
score of the generated summary to compare with previous literature. We also
report the Factual score of our summaries since it is a better benchmark for
abstractive summaries since the ROUGE-2 score is limited to measuring
word-overlaps. We show that our improved model is able to improve on previous
models by at least 5 ROUGE-2 scores, which is a substantial improvement. Also,
a qualitative analysis of the summaries generated by our model shows that these
summaries and human-readable and indeed capture most of the important
information from the transcripts.

    

### [[2108.06317] Towards Efficient Point Cloud Graph Neural Networks Through Architectural Simplification](http://arxiv.org/abs/2108.06317)


  In recent years graph neural network (GNN)-based approaches have become a
popular strategy for processing point cloud data, regularly achieving
state-of-the-art performance on a variety of tasks. To date, the research
community has primarily focused on improving model expressiveness, with
secondary thought given to how to design models that can run efficiently on
resource constrained mobile devices including smartphones or mixed reality
headsets. In this work we make a step towards improving the efficiency of these
models by making the observation that these GNN models are heavily limited by
the representational power of their first, feature extracting, layer. We find
that it is possible to radically simplify these models so long as the feature
extraction layer is retained with minimal degradation to model performance;
further, we discover that it is possible to improve performance overall on
ModelNet40 and S3DIS by improving the design of the feature extractor. Our
approach reduces memory consumption by 20$\times$ and latency by up to
9.9$\times$ for graph layers in models such as DGCNN; overall, we achieve
speed-ups of up to 4.5$\times$ and peak memory reductions of 72.5%.

    

### [[2108.06325] Continual Backprop: Stochastic Gradient Descent with Persistent Randomness](http://arxiv.org/abs/2108.06325)


  The Backprop algorithm for learning in neural networks utilizes two
mechanisms: first, stochastic gradient descent and second, initialization with
small random weights, where the latter is essential to the effectiveness of the
former. We show that in continual learning setups, Backprop performs well
initially, but over time its performance degrades. Stochastic gradient descent
alone is insufficient to learn continually; the initial randomness enables only
initial learning but not continual learning. To the best of our knowledge, ours
is the first result showing this degradation in Backprop's ability to learn. To
address this issue, we propose an algorithm that continually injects random
features alongside gradient descent using a new generate-and-test process. We
call this the Continual Backprop algorithm. We show that, unlike Backprop,
Continual Backprop is able to continually adapt in both supervised and
reinforcement learning problems. We expect that as continual learning becomes
more common in future applications, a method like Continual Backprop will be
essential where the advantages of random initialization are present throughout
learning.

    

### [[2108.06329] Low-Resource Adaptation of Open-Domain Generative Chatbots](http://arxiv.org/abs/2108.06329)


  Recent work building open-domain chatbots has demonstrated that increasing
model size improves performance. On the other hand, latency and connectivity
considerations dictate the move of digital assistants on the device. Giving a
digital assistant like Siri, Alexa, or Google Assistant the ability to discuss
just about anything leads to the need for reducing the chatbot model size such
that it fits on the user's device. We demonstrate that low parameter models can
simultaneously retain their general knowledge conversational abilities while
improving in a specific domain. Additionally, we propose a generic framework
that accounts for variety in question types, tracks reference throughout
multi-turn conversations, and removes inconsistent and potentially toxic
responses. Our framework seamlessly transitions between chatting and performing
transactional tasks, which will ultimately make interactions with digital
assistants more human-like. We evaluate our framework on 1 internal and 4
public benchmark datasets using both automatic (Perplexity) and human (SSA -
Sensibleness and Specificity Average) evaluation metrics and establish
comparable performance while reducing model parameters by 90%.

    

### [[2108.06338] Graph2MDA: a multi-modal variational graph embedding model for predicting microbe-drug associations](http://arxiv.org/abs/2108.06338)


  Accumulated clinical studies show that microbes living in humans interact
closely with human hosts, and get involved in modulating drug efficacy and drug
toxicity. Microbes have become novel targets for the development of
antibacterial agents. Therefore, screening of microbe-drug associations can
benefit greatly drug research and development. With the increase of microbial
genomic and pharmacological datasets, we are greatly motivated to develop an
effective computational method to identify new microbe-drug associations. In
this paper, we proposed a novel method, Graph2MDA, to predict microbe-drug
associations by using variational graph autoencoder (VGAE). We constructed
multi-modal attributed graphs based on multiple features of microbes and drugs,
such as molecular structures, microbe genetic sequences, and function
annotations. Taking as input the multi-modal attribute graphs, VGAE was trained
to learn the informative and interpretable latent representations of each node
and the whole graph, and then a deep neural network classifier was used to
predict microbe-drug associations. The hyperparameter analysis and model
ablation studies showed the sensitivity and robustness of our model. We
evaluated our method on three independent datasets and the experimental results
showed that our proposed method outperformed six existing state-of-the-art
methods. We also explored the meaningness of the learned latent representations
of drugs and found that the drugs show obvious clustering patterns that are
significantly consistent with drug ATC classification. Moreover, we conducted
case studies on two microbes and two drugs and found 75\%-95\% predicted
associations have been reported in PubMed literature. Our extensive performance
evaluations validated the effectiveness of our proposed method.\

    

### [[1906.05029] A Bayesian Approach to In-Game Win Probability in Soccer](http://arxiv.org/abs/1906.05029)


  In-game win probability models, which provide a sports team's likelihood of
winning at each point in a game based on historical observations, are becoming
increasingly popular. In baseball, basketball and American football, they have
become important tools to enhance fan experience, to evaluate in-game
decision-making, and to inform coaching decisions. While equally relevant in
soccer, the adoption of these models is held back by technical challenges
arising from the low-scoring nature of the sport.
In this paper, we introduce an in-game win probability model for soccer that
addresses the shortcomings of existing models. First, we demonstrate that
in-game win probability models for other sports struggle to provide accurate
estimates for soccer, especially towards the end of a game. Second, we
introduce a novel Bayesian statistical framework that estimates running win,
tie and loss probabilities by leveraging a set of contextual game state
features. An empirical evaluation on eight seasons of data for the top-five
soccer leagues demonstrates that our framework provides well-calibrated
probabilities. Furthermore, two use cases show its ability to enhance fan
experience and to evaluate performance in crucial game situations.

    

### [[1909.11855] Universal Graph Transformer Self-Attention Networks](http://arxiv.org/abs/1909.11855)


  The transformer self-attention network has been extensively used in research
domains such as computer vision, image processing, and natural language
processing. The transformer, however, has not been actively used in graph
neural networks, where constructing an advanced aggregation function is
essential. To this end, we present an effective model, named UGformer, which --
by leveraging a transformer self-attention mechanism followed by a recurrent
transition -- induces an advanced aggregation function to learn graph
representations. Experimental results show that UGformer achieves
state-of-the-art accuracies on well-known benchmark datasets for graph
classification.

    

### [[1911.13268] Adversarially Robust Low Dimensional Representations](http://arxiv.org/abs/1911.13268)


  Many machine learning systems are vulnerable to small perturbations made to
inputs either at test time or at training time. This has received much recent
interest on the empirical front due to applications where reliability and
security are critical. However, theoretical understanding of algorithms that
are robust to adversarial perturbations is limited.
In this work we focus on Principal Component Analysis (PCA), a ubiquitous
algorithmic primitive in machine learning. We formulate a natural robust
variant of PCA where the goal is to find a low dimensional subspace to
represent the given data with minimum projection error, that is in addition
robust to small perturbations measured in $\ell_q$ norm (say $q=\infty$).
Unlike PCA which is solvable in polynomial time, our formulation is
computationally intractable to optimize as it captures a variant of the
well-studied sparse PCA objective as a special case. We show the following
results:
-Polynomial time algorithm that is constant factor competitive in the
worst-case with respect to the best subspace, in terms of the projection error
and the robustness criterion.
-We show that our algorithmic techniques can also be made robust to
adversarial training-time perturbations, in addition to yielding
representations that are robust to adversarial perturbations at test time.
Specifically, we design algorithms for a strong notion of training-time
perturbations, where every point is adversarially perturbed up to a specified
amount.
-We illustrate the broad applicability of our algorithmic techniques in
addressing robustness to adversarial perturbations, both at training time and
test time. In particular, our adversarially robust PCA primitive leads to
computationally efficient and robust algorithms for both unsupervised and
supervised learning problems such as clustering and learning adversarially
robust classifiers.

    

### [[2002.04840] Efficient active learning of sparse halfspaces with arbitrary bounded noise](http://arxiv.org/abs/2002.04840)


  We study active learning of homogeneous $s$-sparse halfspaces in
$\mathbb{R}^d$ under the setting where the unlabeled data distribution is
isotropic log-concave and each label is flipped with probability at most $\eta$
for a parameter $\eta \in \big[0, \frac12\big)$, known as the bounded noise.
Even in the presence of mild label noise, i.e. $\eta$ is a small constant, this
is a challenging problem and only recently have label complexity bounds of the
form $\tilde{O}\big(s \cdot \mathrm{polylog}(d, \frac{1}{\epsilon})\big)$ been
established in [Zhang, 2018] for computationally efficient algorithms. In
contrast, under high levels of label noise, the label complexity bounds
achieved by computationally efficient algorithms are much worse: the best known
result of [Awasthi et al., 2016] provides a computationally efficient algorithm
with label complexity $\tilde{O}\big((\frac{s \ln
d}{\epsilon})^{2^{\mathrm{poly}(1/(1-2\eta))}} \big)$, which is label-efficient
only when the noise rate $\eta$ is a fixed constant. In this work, we
substantially improve on it by designing a polynomial time algorithm for active
learning of $s$-sparse halfspaces, with a label complexity of
$\tilde{O}\big(\frac{s}{(1-2\eta)^4} \mathrm{polylog} (d, \frac 1 \epsilon)
\big)$. This is the first efficient algorithm with label complexity polynomial
in $\frac{1}{1-2\eta}$ in this setting, which is label-efficient even for
$\eta$ arbitrarily close to $\frac12$. Our active learning algorithm and its
theoretical guarantees also immediately translate to new state-of-the-art label
and sample complexity results for full-dimensional active and passive halfspace
learning under arbitrary bounded noise. The key insight of our algorithm and
analysis is a new interpretation of online learning regret inequalities, which
may be of independent interest.

    

### [[2005.08859] PDE constraints on smooth hierarchical functions computed by neural networks](http://arxiv.org/abs/2005.08859)


  Neural networks are versatile tools for computation, having the ability to
approximate a broad range of functions. An important problem in the theory of
deep neural networks is expressivity; that is, we want to understand the
functions that are computable by a given network. We study real infinitely
differentiable (smooth) hierarchical functions implemented by feedforward
neural networks via composing simpler functions in two cases:
1) each constituent function of the composition has fewer inputs than the
resulting function;
2) constituent functions are in the more specific yet prevalent form of a
non-linear univariate function (e.g. tanh) applied to a linear multivariate
function.
We establish that in each of these regimes there exist non-trivial algebraic
partial differential equations (PDEs), which are satisfied by the computed
functions. These PDEs are purely in terms of the partial derivatives and are
dependent only on the topology of the network. For compositions of polynomial
functions, the algebraic PDEs yield non-trivial equations (of degrees dependent
only on the architecture) in the ambient polynomial space that are satisfied on
the associated functional varieties. Conversely, we conjecture that such PDE
constraints, once accompanied by appropriate non-singularity conditions and
perhaps certain inequalities involving partial derivatives, guarantee that the
smooth function under consideration can be represented by the network. The
conjecture is verified in numerous examples including the case of tree
architectures which are of neuroscientific interest. Our approach is a step
toward formulating an algebraic description of functional spaces associated
with specific neural networks, and may provide new, useful tools for
constructing neural networks.

    

### [[2005.08919] Modeling extra-deep electromagnetic logs using a deep neural network](http://arxiv.org/abs/2005.08919)


  Modern geosteering is heavily dependent on real-time interpretation of deep
electromagnetic (EM) measurements. We present a methodology to construct a deep
neural network (DNN) model trained to reproduce a full set of extra-deep EM
logs consisting of 22 measurements per logging position. The model is trained
in a 1D layered environment consisting of up to seven layers with different
resistivity values. A commercial simulator provided by a tool vendor is used to
generate a training dataset. The dataset size is limited because the simulator
provided by the vendor is optimized for sequential execution. Therefore, we
design a training dataset that embraces the geological rules and geosteering
specifics supported by the forward model. We use this dataset to produce an EM
simulator based on a DNN without access to the proprietary information about
the EM tool configuration or the original simulator source code. Despite
employing a relatively small training set size, the resulting DNN forward model
is quite accurate for the considered examples: a multi-layer synthetic case and
a section of a published historical operation from the Goliat Field. The
observed average evaluation time of 0.15 ms per logging position makes it also
suitable for future use as part of evaluation-hungry statistical and/or
Monte-Carlo inversion algorithms within geosteering workflows.

    

### [[2006.09226] Parameter-Based Value Functions](http://arxiv.org/abs/2006.09226)


  Traditional off-policy actor-critic Reinforcement Learning (RL) algorithms
learn value functions of a single target policy. However, when value functions
are updated to track the learned policy, they forget potentially useful
information about old policies. We introduce a class of value functions called
Parameter-Based Value Functions (PBVFs) whose inputs include the policy
parameters. They can generalize across different policies. PBVFs can evaluate
the performance of any policy given a state, a state-action pair, or a
distribution over the RL agent's initial states. First we show how PBVFs yield
novel off-policy policy gradient theorems. Then we derive off-policy
actor-critic algorithms based on PBVFs trained by Monte Carlo or Temporal
Difference methods. We show how learned PBVFs can zero-shot learn new policies
that outperform any policy seen during training. Finally our algorithms are
evaluated on a selection of discrete and continuous control tasks using shallow
policies and deep neural networks. Their performance is comparable to
state-of-the-art methods.

    

### [[2006.09773] Neural Ordinary Differential Equation Control of Dynamics on Graphs](http://arxiv.org/abs/2006.09773)


  We study the ability of neural networks to steer or control trajectories of
continuous time non-linear dynamical systems on graphs, which we represent with
neural ordinary differential equations (neural ODEs). To do so, we introduce a
neural-ODE control (NODEC) framework and find that it can learn control signals
that drive graph dynamical systems into desired target states. While we use
loss functions that do not constrain the control energy, our results show that
NODEC produces low energy control signals. Finally, we showcase the performance
and versatility of NODEC by using it to control a system of more than one
thousand coupled, non-linear ODEs.

    

### [[2007.05385] Next Waves in Veridical Network Embedding](http://arxiv.org/abs/2007.05385)


  Embedding nodes of a large network into a metric (e.g., Euclidean) space has
become an area of active research in statistical machine learning, which has
found applications in natural and social sciences. Generally, a representation
of a network object is learned in a Euclidean geometry and is then used for
subsequent tasks regarding the nodes and/or edges of the network, such as
community detection, node classification and link prediction. Network embedding
algorithms have been proposed in multiple disciplines, often with
domain-specific notations and details. In addition, different measures and
tools have been adopted to evaluate and compare the methods proposed under
different settings, often dependent of the downstream tasks. As a result, it is
challenging to study these algorithms in the literature systematically.
Motivated by the recently proposed Veridical Data Science (VDS) framework, we
propose a framework for network embedding algorithms and discuss how the
principles of predictability, computability and stability apply in this
context. The utilization of this framework in network embedding holds the
potential to motivate and point to new directions for future research.

    

### [[2007.14129] COMET: Convolutional Dimension Interaction for Collaborative Filtering](http://arxiv.org/abs/2007.14129)


  Latent factor models play a dominant role among recommendation techniques.
However, most of the existing latent factor models assume both historical
interactions and embedding dimensions are independent of each other, and thus
regrettably ignore the high-order interaction information among historical
interactions and embedding dimensions. In this paper, we propose a novel latent
factor model called COMET (COnvolutional diMEnsion inTeraction), which
simultaneously model the high-order interaction patterns among historical
interactions and embedding dimensions. To be specific, COMET stacks the
embeddings of historical interactions horizontally at first, which results in
two "embedding maps". In this way, internal interactions and dimensional
interactions can be exploited by convolutional neural networks with kernels of
different sizes simultaneously. A fully-connected multi-layer perceptron is
then applied to obtain two interaction vectors. Lastly, the representations of
users and items are enriched by the learnt interaction vectors, which can
further be used to produce the final prediction. Extensive experiments and
ablation studies on various public implicit feedback datasets clearly
demonstrate the effectiveness and the rationality of our proposed method.

    

### [[2008.08871] Deep learning-based transformation of the H&E stain into special stains](http://arxiv.org/abs/2008.08871)


  Pathology is practiced by visual inspection of histochemically stained
slides. Most commonly, the hematoxylin and eosin (H&E) stain is used in the
diagnostic workflow and it is the gold standard for cancer diagnosis. However,
in many cases, especially for non-neoplastic diseases, additional "special
stains" are used to provide different levels of contrast and color to tissue
components and allow pathologists to get a clearer diagnostic picture. In this
study, we demonstrate the utility of supervised learning-based computational
stain transformation from H&E to different special stains (Masson's Trichrome,
periodic acid-Schiff and Jones silver stain) using tissue sections from kidney
needle core biopsies. Based on evaluation by three renal pathologists, followed
by adjudication by a fourth renal pathologist, we show that the generation of
virtual special stains from existing H&E images improves the diagnosis in
several non-neoplastic kidney diseases sampled from 58 unique subjects. A
second study performed by three pathologists found that the quality of the
special stains generated by the stain transformation network was statistically
equivalent to those generated through standard histochemical staining. As the
transformation of H&E images into special stains can be achieved within 1 min
or less per patient core specimen slide, this stain-to-stain transformation
framework can improve the quality of the preliminary diagnosis when additional
special stains are needed, along with significant savings in time and cost,
reducing the burden on healthcare system and patients.

    

### [[2010.12609] Iterative Graph Self-Distillation](http://arxiv.org/abs/2010.12609)


  How to discriminatively vectorize graphs is a fundamental challenge that
attracts increasing attentions in recent years. Inspired by the recent success
of unsupervised contrastive learning, we aim to learn graph-level
representation in an unsupervised manner. Specifically, we propose a novel
unsupervised graph learning paradigm called Iterative Graph Self-Distillation
(IGSD) which iteratively performs the teacher-student distillation with graph
augmentations. Different from conventional knowledge distillation, IGSD
constructs the teacher with an exponential moving average of the student model
and distills the knowledge of itself. The intuition behind IGSD is to predict
the teacher network representation of the graph pairs under different augmented
views. As a natural extension, we also apply IGSD to semi-supervised scenarios
by jointly regularizing the network with both supervised and unsupervised
contrastive loss. Finally, we show that finetuning the IGSD-trained models with
self-training can further improve the graph representation power. Empirically,
we achieve significant and consistent performance gain on various graph
datasets in both unsupervised and semi-supervised settings, which well
validates the superiority of IGSD.

    

### [[2010.15658] Compressive Sensing and Neural Networks from a Statistical Learning Perspective](http://arxiv.org/abs/2010.15658)


  Various iterative reconstruction algorithms for inverse problems can be
unfolded as neural networks. Empirically, this approach has often led to
improved results, but theoretical guarantees are still scarce. While some
progress on generalization properties of neural networks have been made, great
challenges remain. In this chapter, we discuss and combine these topics to
present a generalization error analysis for a class of neural networks suitable
for sparse reconstruction from few linear measurements. The hypothesis class
considered is inspired by the classical iterative soft-thresholding algorithm
(ISTA). The neural networks in this class are obtained by unfolding iterations
of ISTA and learning some of the weights. Based on training samples, we aim at
learning the optimal network parameters via empirical risk minimization and
thereby the optimal network that reconstructs signals from their compressive
linear measurements. In particular, we may learn a sparsity basis that is
shared by all of the iterations/layers and thereby obtain a new approach for
dictionary learning. For this class of networks, we present a generalization
bound, which is based on bounding the Rademacher complexity of hypothesis
classes consisting of such deep networks via Dudley's integral. Remarkably,
under realistic conditions, the generalization error scales only
logarithmically in the number of layers, and at most linear in number of
measurements.

    

### [[2011.07221] Deep Interpretable Classification and Weakly-Supervised Segmentation of Histology Images via Max-Min Uncertainty](http://arxiv.org/abs/2011.07221)


  Weakly-supervised learning (WSL) has recently triggered substantial interest
as it mitigates the lack of pixel-wise annotations.
Given global image labels, WSL methods yield pixel-level predictions
(segmentations), which enable to interpret class predictions. Despite their
recent success, mostly with natural images, such methods can face important
challenges when the foreground and background regions have similar visual cues,
yielding high false-positive rates in segmentations, as is the case in
challenging histology images. WSL training is commonly driven by standard
classification losses, which implicitly maximize model confidence, and locate
the discriminative regions linked to classification decisions. Therefore, they
lack mechanisms for modeling explicitly non-discriminative regions and reducing
false-positive rates. We propose novel regularization terms, which enable the
model to seek both non-discriminative and discriminative regions, while
discouraging unbalanced segmentations. We introduce high uncertainty as a
criterion to localize non-discriminative regions that do not affect classifier
decision, and describe it with original Kullback-Leibler (KL) divergence losses
evaluating the deviation of posterior predictions from the uniform
distribution. Our KL terms encourage high uncertainty of the model when the
latter inputs the latent non-discriminative regions. Our loss integrates: (i) a
cross-entropy seeking a foreground, where model confidence about class
prediction is high; (ii) a KL regularizer seeking a background, where model
uncertainty is high; and (iii) log-barrier terms discouraging unbalanced
segmentations. Comprehensive experiments and ablation studies over the public
GlaS colon cancer data and a Camelyon16 patch-based benchmark for breast cancer
show substantial improvements over state-of-the-art WSL methods, and confirm
the effect of our new regularizers.

    

### [[2101.11751] Faster Kernel Interpolation for Gaussian Processes](http://arxiv.org/abs/2101.11751)


  A key challenge in scaling Gaussian Process (GP) regression to massive
datasets is that exact inference requires computation with a dense n x n kernel
matrix, where n is the number of data points. Significant work focuses on
approximating the kernel matrix via interpolation using a smaller set of m
inducing points. Structured kernel interpolation (SKI) is among the most
scalable methods: by placing inducing points on a dense grid and using
structured matrix algebra, SKI achieves per-iteration time of O(n + m log m)
for approximate inference. This linear scaling in n enables inference for very
large data sets; however the cost is per-iteration, which remains a limitation
for extremely large n. We show that the SKI per-iteration time can be reduced
to O(m log m) after a single O(n) time precomputation step by reframing SKI as
solving a natural Bayesian linear regression problem with a fixed set of m
compact basis functions. With per-iteration complexity independent of the
dataset size n for a fixed grid, our method scales to truly massive data sets.
We demonstrate speedups in practice for a wide range of m and n and apply the
method to GP inference on a three-dimensional weather radar dataset with over
100 million points.

    

### [[2102.03803] Lazy OCO: Online Convex Optimization on a Switching Budget](http://arxiv.org/abs/2102.03803)


  We study a variant of online convex optimization where the player is
permitted to switch decisions at most $S$ times in expectation throughout $T$
rounds. Similar problems have been addressed in prior work for the discrete
decision set setting, and more recently in the continuous setting but only with
an adaptive adversary. In this work, we aim to fill the gap and present
computationally efficient algorithms in the more prevalent oblivious setting,
establishing a regret bound of $O(T/S)$ for general convex losses and
$\widetilde O(T/S^2)$ for strongly convex losses. In addition, for stochastic
i.i.d.~losses, we present a simple algorithm that performs $\log T$ switches
with only a multiplicative $\log T$ factor overhead in its regret in both the
general and strongly convex settings. Finally, we complement our algorithms
with lower bounds that match our upper bounds in some of the cases we consider.

    

### [[2102.12756] CMDNet: Learning a Probabilistic Relaxation of Discrete Variables for Soft Detection with Low Complexity](http://arxiv.org/abs/2102.12756)


  Following the great success of Machine Learning (ML), especially Deep Neural
Networks (DNNs), in many research domains in 2010s, several ML-based approaches
were proposed for detection in large inverse linear problems, e.g., massive
MIMO systems. The main motivation behind is that the complexity of Maximum
A-Posteriori (MAP) detection grows exponentially with system dimensions.
Instead of using DNNs, essentially being a black-box, we take a slightly
different approach and introduce a probabilistic Continuous relaxation of
disCrete variables to MAP detection. Enabling close approximation and
continuous optimization, we derive an iterative detection algorithm: Concrete
MAP Detection (CMD). Furthermore, extending CMD by the idea of deep unfolding
into CMDNet, we allow for (online) optimization of a small number of parameters
to different working points while limiting complexity. In contrast to recent
DNN-based approaches, we select the optimization criterion and output of CMDNet
based on information theory and are thus able to learn approximate
probabilities of the individual optimal detector. This is crucial for soft
decoding in today's communication systems. Numerical simulation results in MIMO
systems reveal CMDNet to feature a promising accuracy complexity trade-off
compared to State of the Art. Notably, we demonstrate CMDNet's soft outputs to
be reliable for decoders.

    

### [[2104.02588] Principal Component Analysis Applied to Gradient Fields in Band Gap Optimization Problems for Metamaterials](http://arxiv.org/abs/2104.02588)


  A promising technique for the spectral design of acoustic metamaterials is
based on the formulation of suitable constrained nonlinear optimization
problems. Unfortunately, the straightforward application of classical
gradient-based iterative optimization algorithms to the numerical solution of
such problems is typically highly demanding, due to the complexity of the
underlying physical models. Nevertheless, supervised machine learning
techniques can reduce such a computational effort, e.g., by replacing the
original objective functions of such optimization problems with more-easily
computable approximations. In this framework, the present article describes the
application of a related unsupervised machine learning technique, namely,
principal component analysis, to approximate the gradient of the objective
function of a band gap optimization problem for an acoustic metamaterial, with
the aim of making the successive application of a gradient-based iterative
optimization algorithm faster. Numerical results show the effectiveness of the
proposed method.

    

### [[2104.05704] Escaping the Big Data Paradigm with Compact Transformers](http://arxiv.org/abs/2104.05704)


  With the rise of Transformers as the standard for language processing, and
their advancements in computer vision, along with their unprecedented size and
amounts of training data, many have come to believe that they are not suitable
for small sets of data. This trend leads to great concerns, including but not
limited to: limited availability of data in certain scientific domains and the
exclusion of those with limited resource from research in the field. In this
paper, we dispel the myth that transformers are "data hungry" and therefore can
only be applied to large sets of data. We show for the first time that with the
right size and tokenization, transformers can perform head-to-head with
state-of-the-art CNNs on small datasets, often with better accuracy and fewer
parameters. Our model eliminates the requirement for class token and positional
embeddings through a novel sequence pooling strategy and the use of
convolution/s. It is flexible in terms of model size, and can have as little as
0.28M parameters while achieving good results. Our model can reach 98.00%
accuracy when training from scratch on CIFAR-10, which is a significant
improvement over previous Transformer based models. It also outperforms many
modern CNN based approaches, such as ResNet, and even some recent NAS-based
approaches, such as Proxyless-NAS. Our simple and compact design democratizes
transformers by making them accessible to those with limited computing
resources and/or dealing with small datasets. Our method also works on larger
datasets, such as ImageNet (82.71% accuracy with 29% parameters of ViT), and
NLP tasks as well. Our code and pre-trained models are publicly available at
this https URL.

    

### [[2104.11178] VATT: Transformers for Multimodal Self-Supervised Learning from Raw Video, Audio and Text](http://arxiv.org/abs/2104.11178)


  We present a framework for learning multimodal representations from unlabeled
data using convolution-free Transformer architectures. Specifically, our
Video-Audio-Text Transformer (VATT) takes raw signals as inputs and extracts
multimodal representations that are rich enough to benefit a variety of
downstream tasks. We train VATT end-to-end from scratch using multimodal
contrastive losses and evaluate its performance by the downstream tasks of
video action recognition, audio event classification, image classification, and
text-to-video retrieval. Furthermore, we study a modality-agnostic
single-backbone Transformer by sharing weights among the three modalities. We
show that the convolution-free VATT outperforms state-of-the-art ConvNet-based
architectures in the downstream tasks. Especially, VATT's vision Transformer
achieves the top-1 accuracy of 82.1% on Kinetics-400, 83.6% on Kinetics-600,and
41.1% on Moments in Time, new records while avoiding supervised pre-training.
Transferring to image classification leads to 78.7% top-1 accuracy on ImageNet
compared to 64.7% by training the same Transformer from scratch, showing the
generalizability of our model despite the domain gap between videos and images.
VATT's audio Transformer also sets a new record on waveform-based audio event
recognition by achieving the mAP of 39.4% on AudioSet without any supervised
pre-training. VATT's source code is publicly available.

    

### [[2104.15050] Generative Models Improve Radiomics Reproducibility in Low Dose CTs: A Simulation Study](http://arxiv.org/abs/2104.15050)


  Radiomics is an active area of research in medical image analysis, the low
reproducibility of radiomics has limited its applicability to clinical
practice. This issue is especially prominent when radiomic features are
calculated from noisy images, such as low dose computed tomography (CT) scans.
In this article, we investigate the possibility of improving the
reproducibility of radiomic features calculated on noisy CTs by using
generative models for denoising.One traditional denoising method - non-local
means - and two generative models - encoder-decoder networks (EDN) and
conditional generative adversarial networks (CGANs) - were selected as the test
models. We added noise to the sinograms of full dose CTs to mimic low dose CTs
with two different levels of noise: low-noise CT and high-noise CT. Models were
trained on high-noise CTs and used to denoise low-noise CTs without
re-training. We also test the performance of our model in real data, using
dataset of same-day repeat low dose CTs to assess the reproducibility of
radiomic features in denoised images. The EDN and the CGAN improved the
concordance correlation coefficients (CCC) of radiomic features for low-noise
images from 0.87 to 0.92 and for high-noise images from 0.68 to 0.92
respectively. Moreover, the EDN and the CGAN improved the test-retest
reliability of radiomic features (mean CCC increased from 0.89 to 0.94) based
on real low dose CTs. The results show that denoising using EDN and CGANs can
improve the reproducibility of radiomic features calculated on noisy CTs.
Moreover, images with different noise levels can be denoised to improve the
reproducibility using these models without re-training, as long as the noise
intensity is equal or lower than that in high-noise CTs. To the authors'
knowledge, this is the first effort to improve the reproducibility of radiomic
features calculated on low dose CT scans.

    

### [[2106.01779] Preparation of Many-body Ground States by Time Evolution with Variational Microscopic Magnetic Fields and Incomplete Interactions](http://arxiv.org/abs/2106.01779)


  State preparation is of fundamental importance in quantum physics, which can
be realized by constructing the quantum circuit as a unitary that transforms
the initial state to the target, or implementing a quantum control protocol to
evolve to the target state with a designed Hamiltonian. In this work, we study
the latter on quantum many-body systems by the time evolution with fixed
couplings and variational magnetic fields. In specific, we consider to prepare
the ground states of the Hamiltonians containing certain interactions that are
missing in the Hamiltonians for the time evolution. An optimization method is
proposed to optimize the magnetic fields by "fine-graining" the discretization
of time, in order to gain high precision and stability. The back propagation
technique is utilized to obtain the gradients of the fields against the
logarithmic fidelity. Our method is tested on preparing the ground state of
Heisenberg chain with the time evolution by the XY and Ising interactions, and
its performance surpasses two baseline methods that use local and global
optimization strategies, respectively. Our work can be applied and generalized
to other quantum models such as those defined on higher dimensional lattices.
It enlightens to reduce the complexity of the required interactions for
implementing quantum control or other tasks in quantum information and
computation by means of optimizing the magnetic fields.

    

### [[2106.12758] Neural ODE to model and prognose thermoacoustic instability](http://arxiv.org/abs/2106.12758)


  In reacting flow systems, thermoacoustic instability characterized by high
amplitude pressure fluctuations, is driven by a positive coupling between the
unsteady heat release rate and the acoustic field of the combustor. When the
underlying flow is turbulent, as a control parameter of the system is varied
and the system approach thermoacoustic instability, the acoustic pressure
oscillations synchronize with heat release rate oscillations. Consequently,
during the onset of thermoacoustic instability in turbulent combustors, the
system dynamics transition from chaotic oscillations to periodic oscillations
via a state of intermittency. Thermoacoustic systems are traditionally modeled
by coupling the model for the unsteady heat source and the acoustic subsystem,
each estimated independently. The response of the unsteady heat source, the
flame, to acoustic fluctuations are characterized by introducing external
unsteady forcing. This necessitates a powerful excitation module to obtain the
nonlinear response of the flame to acoustic perturbations. Instead of
characterizing individual subsystems, we introduce a neural ordinary
differential equation (neural ODE) framework to model the thermoacoustic system
as a whole. The neural ODE model for the thermoacoustic system uses time series
of the heat release rate and the pressure fluctuations, measured simultaneously
without introducing any external perturbations, to model their coupled
interaction. Further, we use the parameters of neural ODE to define an anomaly
measure that represents the proximity of system dynamics to limit cycle
oscillations and thus provide an early warning signal for the onset of
thermoacoustic instability.

    

### [[2106.14736] Speech2Properties2Gestures: Gesture-Property Prediction as a Tool for Generating Representational Gestures from Speech](http://arxiv.org/abs/2106.14736)


  We propose a new framework for gesture generation, aiming to allow
data-driven approaches to produce more semantically rich gestures. Our approach
first predicts whether to gesture, followed by a prediction of the gesture
properties. Those properties are then used as conditioning for a modern
probabilistic gesture-generation model capable of high-quality output. This
empowers the approach to generate gestures that are both diverse and
representational. Follow-ups and more information can be found on the project
page: this https URL .

    

### [[2108.05987] Automating System Configuration](http://arxiv.org/abs/2108.05987)


  The increasing complexity of modern configurable systems makes it critical to
improve the level of automation in the process of system configuration. Such
automation can also improve the agility of the development cycle, allowing for
rapid and automated integration of decoupled workflows. In this paper, we
present a new framework for automated configuration of systems representable as
state machines. The framework leverages model checking and satisfiability
modulo theories (SMT) and can be applied to any application domain
representable using SMT formulas. Our approach can also be applied modularly,
improving its scalability. Furthermore, we show how optimization can be used to
produce configurations that are best according to some metric and also more
likely to be understandable to humans. We showcase this framework and its
flexibility by using it to configure a CGRA memory tile for various image
processing applications.

    

### [[2108.06001] HPTMT Parallel Operators for High Performance Data Science & Data Engineering](http://arxiv.org/abs/2108.06001)


  Data-intensive applications are becoming commonplace in all science
disciplines. They are comprised of a rich set of sub-domains such as data
engineering, deep learning, and machine learning. These applications are built
around efficient data abstractions and operators that suit the applications of
different domains. Often lack of a clear definition of data structures and
operators in the field has led to other implementations that do not work well
together. The HPTMT architecture that we proposed recently, identifies a set of
data structures, operators, and an execution model for creating rich data
applications that links all aspects of data engineering and data science
together efficiently. This paper elaborates and illustrates this architecture
using an end-to-end application with deep learning and data engineering parts
working together.

    

### [[2108.06004] A Distributed SGD Algorithm with Global Sketching for Deep Learning Training Acceleration](http://arxiv.org/abs/2108.06004)


  Distributed training is an effective way to accelerate the training process
of large-scale deep learning models. However, the parameter exchange and
synchronization of distributed stochastic gradient descent introduce a large
amount of communication overhead. Gradient compression is an effective method
to reduce communication overhead. In synchronization SGD compression methods,
many Top-k sparsification based gradient compression methods have been proposed
to reduce the communication. However, the centralized method based on the
parameter servers has the single point of failure problem and limited
scalability, while the decentralized method with global parameter exchanging
may reduce the convergence rate of training. In contrast with Top-$k$ based
methods, we proposed a gradient compression method with globe gradient vector
sketching, which uses the Count-Sketch structure to store the gradients to
reduce the loss of the accuracy in the training process, named global-sketching
SGD (gs-SGD). The gs-SGD has better convergence efficiency on deep learning
models and a communication complexity of O($\log d*\log P$), where $d$ is the
number of model parameters and P is the number of workers. We conducted
experiments on GPU clusters to verify that our method has better convergence
efficiency than global Top-$k$ and Sketching-based methods. In addition, gs-SGD
achieves 1.3-3.1x higher throughput compared with gTop-$k$, and 1.1-1.2x higher
throughput compared with original Sketched-SGD.

    

### [[2108.06091] BESS Aided Reconfigurable Energy Supply using Deep Reinforcement Learning for 5G and Beyond](http://arxiv.org/abs/2108.06091)


  The year of 2020 has witnessed the unprecedented development of 5G networks,
along with the widespread deployment of 5G base stations (BSs). Nevertheless,
the enormous energy consumption of BSs and the incurred huge energy cost have
become significant concerns for the mobile operators. As the continuous decline
of the renewable energy cost, equipping the power-hungry BSs with renewable
energy generators could be a sustainable solution. In this work, we propose an
energy storage aided reconfigurable renewable energy supply solution for the
BS, which could supply clean energy to the BS and store surplus energy for
backup usage. Specifically, to flexibly reconfigure the battery's
discharging/charging operations, we propose a deep reinforcement learning based
reconfiguring policy, which can adapt to the dynamical renewable energy
generations as well as the varying power demands. Our experiments using the
real-world data on renewable energy generations and power demands demonstrate
that, our reconfigurable power supply solution can achieve an energy saving
ratio of 74.8%, compared to the case with traditional power grid supply.

    

### [[2108.06123] Digital Twin of a Cloud Data Centre: OpenStack Cluster Visualisation](http://arxiv.org/abs/2108.06123)


  Data centres in contemporary times are essential as the supply of data
increases. Data centres are areas where computing systems are concentrated for
facilitating data processing, transfer and storage. At present traditional data
centres have moved more towards the cloud model thereby making the processing,
storage and harnessing of data more manageable and more accessible via the
utility and subscription-based model of computing services. From the
administrative point of view, cloud data centres are complex systems, hard to
grasp and require large amounts of time to analyse different aspects of the
cloud data centre such as maintenance and resource management. For a cloud data
centre admin, this could be a challenging problem and a highly time-consuming
task. Accordingly, there is a need to improve the useability of cloud data
centre monitoring and management tools, and the digital twin could fulfil this
need. This paper's primary objective is to construct a digital twin - a 3D
visualisation and monitoring tool - of a cloud data centre managed by
OpenStack, the well-known open-source cloud computing infrastructure software.
To evaluate our proposed tool, we garner feedback on the digital twin's
useability compared to the OpenStack dashboard. The input will be received from
cloud data centres experts as they test the digital twin and answer various
questions in an interview. The study results show that our proposed Digital
Twin will help data centre admins better monitor and manage their data centres.
It also will facilitate further research and implementation of the digital twin
of data centres to improve usability.

    

### [[2108.06322] Quantifying and Improving Performance of Distributed Deep Learning with Cloud Storage](http://arxiv.org/abs/2108.06322)


  Cloud computing provides a powerful yet low-cost environment for distributed
deep learning workloads. However, training complex deep learning models often
requires accessing large amounts of data, which can easily exceed the capacity
of local disks. Prior research often overlooks this training data problem by
implicitly assuming that data is available locally or via low latency
network-based data storage. Such implicit assumptions often do not hold in a
cloud-based training environment, where deep learning practitioners create and
tear down dedicated GPU clusters on demand, or do not have the luxury of local
storage, such as in serverless workloads. In this work, we investigate the
performance of distributed training that leverages training data residing
entirely inside cloud storage buckets. These buckets promise low storage costs,
but come with inherent bandwidth limitations that make them seem unsuitable for
an efficient training solution. To account for these bandwidth limitations, we
propose the use of two classical techniques, namely caching and pre-fetching,
to mitigate the training performance degradation. We implement a prototype,
DELI, based on the popular deep learning framework PyTorch by building on its
data loading abstractions. We then evaluate the training performance of two
deep learning workloads using Google Cloud's NVIDIA K80 GPU servers and show
that we can reduce the time that the training loop is waiting for data by
85.6%-93.5% compared to loading directly from a storage bucket - thus achieving
comparable performance to loading data directly from disk - while only storing
a fraction of the data locally at a time. In addition, DELI has the potential
of lowering the cost of running a training workload, especially on models with
long per-epoch training times.

    

### [[1804.05039] Mitigating Docker Security Issues](http://arxiv.org/abs/1804.05039)


  Docker offers an ecosystem that offers a platform for application packaging,
distributing, and managing within containers. However, the Docker platform has
not yet matured. Presently, Docker is less secured than virtual machines (VM)
and most of the other cloud technologies. The key to Dockers inadequate
security protocols is container sharing of Linux kernel, which can lead to the
risk of privileged escalations. This research will outline some significant
security vulnerabilities at Docker and counter solutions to neutralize such
attacks. There are a variety of security attacks like insider and outsider.
This research will outline both types of attacks and their mitigations
strategies. Taking some precautionary measures can save from massive disasters.
This research will also present Docker secure deployment guidelines. These
guidelines will suggest different configurations to deploy Docker containers in
a more secure way.

    

### [[2106.03617] PAIO: A Software-Defined Storage Data Plane Framework](http://arxiv.org/abs/2106.03617)


  We propose PAIO, the first general-purpose framework that enables system
designers to build custom-made Software-Defined Storage (SDS) data plane
stages. It provides the means to implement storage optimizations adaptable to
different workflows and user-defined policies, and allows straightforward
integration with existing applications and I/O layers. PAIO allows stages to be
integrated with modern SDS control planes to ensure holistic control and
system-wide optimal performance. We demonstrate the performance and
applicability of PAIO with two use cases. The first improves 99th percentile
latency by 4x in industry-standard LSM-based key-value stores. The second
ensures dynamic per-application bandwidth guarantees under shared storage
environments.

    

### [[2108.05948] Deep adversarial attack on target detection systems](http://arxiv.org/abs/2108.05948)


  Target detection systems identify targets by localizing their coordinates on
the input image of interest. This is ideally achieved by labeling each pixel in
an image as a background or a potential target pixel. Deep Convolutional Neural
Network (DCNN) classifiers have proven to be successful tools for computer
vision applications. However,prior research confirms that even state of the art
classifier models are susceptible to adversarial attacks. In this paper, we
show how to generate adversarial infrared images by adding small perturbations
to the targets region to deceive a DCNN-based target detector at remarkable
levels. We demonstrate significant progress in developing visually
imperceptible adversarial infrared images where the targets are visually
recognizable by an expert but a DCNN-based target detector cannot detect the
targets in the image.

    

### [[2108.05951] Sophisticated Students in Boston Mechanism and Gale-Shapley Algorithm for School Choice Problem](http://arxiv.org/abs/2108.05951)


  We present our experimental results of simulating the school choice problem
which deals with the assignment of students to schools based on each group's
complete preference list for the other group using two algorithms: Boston
mechanism and student-proposing Gale-Shapley algorithm. We compare the effects
of sophisticated students altering their preference lists with regards to these
two algorithms. Our simulation results show that sophisticated students can
benefit more in Boston mechanism compared to Gale-Shapley algorithm based on
multiple evaluation metrics.

    

### [[2108.06010] GQE-PRF: Generative Query Expansion with Pseudo-Relevance Feedback](http://arxiv.org/abs/2108.06010)


  Query expansion with pseudo-relevance feedback (PRF) is a powerful approach
to enhance the effectiveness in information retrieval. Recently, with the rapid
advance of deep learning techniques, neural text generation has achieved
promising success in many natural language tasks. To leverage the strength of
text generation for information retrieval, in this article, we propose a novel
approach which effectively integrates text generation models into PRF-based
query expansion. In particular, our approach generates augmented query terms
via neural text generation models conditioned on both the initial query and
pseudo-relevance feedback. Moreover, in order to train the generative model, we
adopt the conditional generative adversarial nets (CGANs) and propose the
PRF-CGAN method in which both the generator and the discriminator are
conditioned on the pseudo-relevance feedback. We evaluate the performance of
our approach on information retrieval tasks using two benchmark datasets. The
experimental results show that our approach achieves comparable performance or
outperforms traditional query expansion methods on both the retrieval and
reranking tasks.

    

### [[2108.06014] TPRM: A Topic-based Personalized Ranking Model for Web Search](http://arxiv.org/abs/2108.06014)


  Ranking models have achieved promising results, but it remains challenging to
design personalized ranking systems to leverage user profiles and semantic
representations between queries and documents. In this paper, we propose a
topic-based personalized ranking model (TPRM) that integrates user topical
profile with pretrained contextualized term representations to tailor the
general document ranking list. Experiments on the real-world dataset
demonstrate that TPRM outperforms state-of-the-art ad-hoc ranking models and
personalized ranking models significantly.

    

### [[2108.06027] PAIR: Leveraging Passage-Centric Similarity Relation for Improving Dense Passage Retrieval](http://arxiv.org/abs/2108.06027)


  Recently, dense passage retrieval has become a mainstream approach to finding
relevant information in various natural language processing tasks. A number of
studies have been devoted to improving the widely adopted dual-encoder
architecture. However, most of the previous studies only consider query-centric
similarity relation when learning the dual-encoder retriever. In order to
capture more comprehensive similarity relations, we propose a novel approach
that leverages both query-centric and PAssage-centric sImilarity Relations
(called PAIR) for dense passage retrieval. To implement our approach, we make
three major technical contributions by introducing formal formulations of the
two kinds of similarity relations, generating high-quality pseudo labeled data
via knowledge distillation, and designing an effective two-stage training
procedure that incorporates passage-centric similarity relation constraint.
Extensive experiments show that our approach significantly outperforms previous
state-of-the-art models on both MSMARCO and Natural Questions datasets.

    

### [[2108.06028] DeepIC: Coding for Interference Channels via Deep Learning](http://arxiv.org/abs/2108.06028)


  The two-user interference channel is a model for multi one-to-one
communications, where two transmitters wish to communicate with their
corresponding receivers via a shared wireless medium. Two most common and
simple coding schemes are time division (TD) and treating interference as noise
(TIN). Interestingly, it is shown that there exists an asymptotic scheme,
called Han-Kobayashi scheme, that performs better than TD and TIN. However,
Han-Kobayashi scheme has impractically high complexity and is designed for
asymptotic settings, which leads to a gap between information theory and
practice.
In this paper, we focus on designing practical codes for interference
channels. As it is challenging to analytically design practical codes with
feasible complexity, we apply deep learning to learn codes for interference
channels. We demonstrate that DeepIC, a convolutional neural network-based code
with an iterative decoder, outperforms TD and TIN by a significant margin for
two-user additive white Gaussian noise channels with moderate amount of
interference.

    

### [[2108.06038] Co-GAIL: Learning Diverse Strategies for Human-Robot Collaboration](http://arxiv.org/abs/2108.06038)


  We present a method for learning a human-robot collaboration policy from
human-human collaboration demonstrations. An effective robot assistant must
learn to handle diverse human behaviors shown in the demonstrations and be
robust when the humans adjust their strategies during online task execution.
Our method co-optimizes a human policy and a robot policy in an interactive
learning process: the human policy learns to generate diverse and plausible
collaborative behaviors from demonstrations while the robot policy learns to
assist by estimating the unobserved latent strategy of its human collaborator.
Across a 2D strategy game, a human-robot handover task, and a multi-step
collaborative manipulation task, our method outperforms the alternatives in
both simulated evaluations and when executing the tasks with a real human
operator in-the-loop. Supplementary materials and videos at
this https URL


### [[2108.06040] Knowledge Graph Reasoning with Relational Directed Graph](http://arxiv.org/abs/2108.06040)


  Reasoning on the knowledge graph (KG) aims to infer new facts from existing
ones. Methods based on the relational path in the literature have shown strong,
interpretable, and inductive reasoning ability. However, the paths are
naturally limited in capturing complex topology in KG. In this paper, we
introduce a novel relational structure, i.e., relational directed graph
(r-digraph), which is composed of overlapped relational paths, to capture the
KG's structural information. Since the digraph exhibits more complex structure
than paths, constructing and learning on the r-digraph are challenging. Here,
we propose a variant of graph neural network, i.e., RED-GNN, to address the
above challenges by learning the RElational Digraph with a variant of GNN.
Specifically, RED-GNN recursively encodes multiple r-digraphs with shared edges
and selects the strongly correlated edges through query-dependent attention
weights. We demonstrate the significant gains on reasoning both KG with unseen
entities and incompletion KG benchmarks by the r-digraph, the efficiency of
RED-GNN, and the interpretable dependencies learned on the r-digraph.

    

### [[2108.06076] Point-Voxel Transformer: An Efficient Approach To 3D Deep Learning](http://arxiv.org/abs/2108.06076)


  Due to the sparsity and irregularity of the 3D data, approaches that directly
process points have become popular. Among all point-based models,
Transformer-based models have achieved state-of-the-art performance by fully
preserving point interrelation. However, most of them spend high percentage of
total time on sparse data accessing (e.g., Farthest Point Sampling (FPS) and
neighbor points query), which becomes the computation burden. Therefore, we
present a novel 3D Transformer, called Point-Voxel Transformer (PVT) that
leverages self-attention computation in points to gather global context
features, while performing multi-head self-attention (MSA) computation in
voxels to capture local information and reduce the irregular data access.
Additionally, to further reduce the cost of MSA computation, we design a cyclic
shifted boxing scheme which brings greater efficiency by limiting the MSA
computation to non-overlapping local boxes while also preserving cross-box
connection. Our method fully exploits the potentials of Transformer
architecture, paving the road to efficient and accurate recognition results.
Evaluated on classification and segmentation benchmarks, our PVT not only
achieves strong accuracy but outperforms previous state-of-the-art
Transformer-based models with 9x measured speedup on average. For 3D object
detection task, we replace the primitives in Frustrum PointNet with PVT layer
and achieve the improvement of 8.6%.

    

### [[2108.06107] Aspect Sentiment Triplet Extraction Using Reinforcement Learning](http://arxiv.org/abs/2108.06107)


  Aspect Sentiment Triplet Extraction (ASTE) is the task of extracting triplets
of aspect terms, their associated sentiments, and the opinion terms that
provide evidence for the expressed sentiments. Previous approaches to ASTE
usually simultaneously extract all three components or first identify the
aspect and opinion terms, then pair them up to predict their sentiment
polarities. In this work, we present a novel paradigm, ASTE-RL, by regarding
the aspect and opinion terms as arguments of the expressed sentiment in a
hierarchical reinforcement learning (RL) framework. We first focus on
sentiments expressed in a sentence, then identify the target aspect and opinion
terms for that sentiment. This takes into account the mutual interactions among
the triplet's components while improving exploration and sample efficiency.
Furthermore, this hierarchical RLsetup enables us to deal with multiple and
overlapping triplets. In our experiments, we evaluate our model on existing
datasets from laptop and restaurant domains and show that it achieves
state-of-the-art performance. The implementation of this work is publicly
available at this https URL.

    

### [[2108.06161] Reinforcement Learning for Robot Navigation with Adaptive ExecutionDuration (AED) in a Semi-Markov Model](http://arxiv.org/abs/2108.06161)


  Deep reinforcement learning (DRL) algorithms have proven effective in robot
navigation, especially in unknown environments, through directly mapping
perception inputs into robot control commands. Most existing methods adopt
uniform execution duration with robots taking commands at fixed intervals. As
such, the length of execution duration becomes a crucial parameter to the
navigation algorithm. In particular, if the duration is too short, then the
navigation policy would be executed at a high frequency, with increased
training difficulty and high computational cost. Meanwhile, if the duration is
too long, then the policy becomes unable to handle complex situations, like
those with crowded obstacles. It is thus tricky to find the "sweet" duration
range; some duration values may render a DRL model to fail to find a navigation
path. In this paper, we propose to employ adaptive execution duration to
overcome this problem. Specifically, we formulate the navigation task as a
Semi-Markov Decision Process (SMDP) problem to handle adaptive execution
duration. We also improve the distributed proximal policy optimization (DPPO)
algorithm and provide its theoretical guarantee for the specified SMDP problem.
We evaluate our approach both in the simulator and on an actual robot. The
results show that our approach outperforms the other DRL-based method (with
fixed execution duration) by 10.3% in terms of the navigation success rate.

    

### [[2108.06180] SPACE: A Simulator for Physical Interactions and Causal Learning in 3D Environments](http://arxiv.org/abs/2108.06180)


  Recent advancements in deep learning, computer vision, and embodied AI have
given rise to synthetic causal reasoning video datasets. These datasets
facilitate the development of AI algorithms that can reason about physical
interactions between objects. However, datasets thus far have primarily focused
on elementary physical events such as rolling or falling. There is currently a
scarcity of datasets that focus on the physical interactions that humans
perform daily with objects in the real world. To address this scarcity, we
introduce SPACE: A Simulator for Physical Interactions and Causal Learning in
3D Environments. The SPACE simulator allows us to generate the SPACE dataset, a
synthetic video dataset in a 3D environment, to systematically evaluate
physics-based models on a range of physical causal reasoning tasks. Inspired by
daily object interactions, the SPACE dataset comprises videos depicting three
types of physical events: containment, stability and contact. These events make
up the vast majority of the basic physical interactions between objects. We
then further evaluate it with a state-of-the-art physics-based deep model and
show that the SPACE dataset improves the learning of intuitive physics with an
approach inspired by curriculum learning. Repository:
this https URL


### [[2108.06216] MAIR: Framework for mining relationships between research articles, strategies, and regulations in the field of explainable artificial intelligence](http://arxiv.org/abs/2108.06216)


  The growing number of AI applications, also for high-stake decisions,
increases the interest in Explainable and Interpretable Machine Learning
(XI-ML). This trend can be seen both in the increasing number of regulations
and strategies for developing trustworthy AI and the growing number of
scientific papers dedicated to this topic. To ensure the sustainable
development of AI, it is essential to understand the dynamics of the impact of
regulation on research papers as well as the impact of scientific discourse on
AI-related policies. This paper introduces a novel framework for joint analysis
of AI-related policy documents and eXplainable Artificial Intelligence (XAI)
research papers. The collected documents are enriched with metadata and
interconnections, using various NLP methods combined with a methodology
inspired by Institutional Grammar. Based on the information extracted from
collected documents, we showcase a series of analyses that help understand
interactions, similarities, and differences between documents at different
stages of institutionalization. To the best of our knowledge, this is the first
work to use automatic language analysis tools to understand the dynamics
between XI-ML methods and regulations. We believe that such a system
contributes to better cooperation between XAI researchers and AI policymakers.

    

### [[2108.06246] An Interpretable Algorithm for Uveal Melanoma Subtyping from Whole Slide Cytology Images](http://arxiv.org/abs/2108.06246)


  Algorithmic decision support is rapidly becoming a staple of personalized
medicine, especially for high-stakes recommendations in which access to certain
information can drastically alter the course of treatment, and thus, patient
outcome; a prominent example is radiomics for cancer subtyping. Because in
these scenarios the stakes are high, it is desirable for decision systems to
not only provide recommendations but supply transparent reasoning in support
thereof. For learning-based systems, this can be achieved through an
interpretable design of the inference pipeline. Herein we describe an automated
yet interpretable system for uveal melanoma subtyping with digital cytology
images from fine needle aspiration biopsies. Our method embeds every
automatically segmented cell of a candidate cytology image as a point in a 2D
manifold defined by many representative slides, which enables reasoning about
the cell-level composition of the tissue sample, paving the way for
interpretable subtyping of the biopsy. Finally, a rule-based slide-level
classification algorithm is trained on the partitions of the circularly
distorted 2D manifold. This process results in a simple rule set that is
evaluated automatically but highly transparent for human verification. On our
in house cytology dataset of 88 uveal melanoma patients, the proposed method
achieves an accuracy of 87.5% that compares favorably to all competing
approaches, including deep "black box" models. The method comes with a user
interface to facilitate interaction with cell-level content, which may offer
additional insights for pathological assessment.

    

### [[2108.06247] Optical Adversarial Attack](http://arxiv.org/abs/2108.06247)


  We introduce \textbf{OP}tical \textbf{AD}versarial attack (OPAD). OPAD is an
adversarial attack in the physical space aiming to fool image classifiers
without physically touching the objects (e.g., moving or painting the objects).
The principle of OPAD is to use structured illumination to alter the appearance
of the target objects. The system consists of a low-cost projector, a camera,
and a computer. The challenge of the problem is the non-linearity of the
radiometric response of the projector and the spatially varying spectral
response of the scene. Attacks generated in a conventional approach do not work
in this setting unless they are calibrated to compensate for such a
projector-camera model. The proposed solution incorporates the projector-camera
model into the adversarial attack optimization, where a new attack formulation
is derived. Experimental results prove the validity of the solution. It is
demonstrated that OPAD can optically attack a real 3D object in the presence of
background lighting for white-box, black-box, targeted, and untargeted attacks.
Theoretical analysis is presented to quantify the fundamental performance limit
of the system.

    

### [[2108.06270] Enhancing audio quality for expressive Neural Text-to-Speech](http://arxiv.org/abs/2108.06270)


  Artificial speech synthesis has made a great leap in terms of naturalness as
recent Text-to-Speech (TTS) systems are capable of producing speech with
similar quality to human recordings. However, not all speaking styles are easy
to model: highly expressive voices are still challenging even to recent TTS
architectures since there seems to be a trade-off between expressiveness in a
generated audio and its signal quality. In this paper, we present a set of
techniques that can be leveraged to enhance the signal quality of a
highly-expressive voice without the use of additional data. The proposed
techniques include: tuning the autoregressive loop's granularity during
training; using Generative Adversarial Networks in acoustic modelling; and the
use of Variational Auto-Encoders in both the acoustic model and the neural
vocoder. We show that, when combined, these techniques greatly closed the gap
in perceived naturalness between the baseline system and recordings by 39% in
terms of MUSHRA scores for an expressive celebrity voice.

    

### [[2108.06314] A Dataset for Answering Time-Sensitive Questions](http://arxiv.org/abs/2108.06314)


  Time is an important dimension in our physical world. Lots of facts can
evolve with respect to time. For example, the U.S. President might change every
four years. Therefore, it is important to consider the time dimension and
empower the existing QA models to reason over time. However, the existing QA
datasets contain rather few time-sensitive questions, hence not suitable for
diagnosing or benchmarking the model's temporal reasoning capability. In order
to promote research in this direction, we propose to construct a time-sensitive
QA dataset. The dataset is constructed by 1) mining time-evolving facts from
WikiData and align them to their corresponding Wikipedia page, 2) employing
crowd workers to verify and calibrate these noisy facts, 3) generating
question-answer pairs based on the annotated time-sensitive facts. Our dataset
poses two novel challenges: 1) the model needs to understand both explicit and
implicit mention of time information in the long document, 2) the model needs
to perform temporal reasoning like comparison, addition, subtraction. We
evaluate different SoTA long-document QA systems like BigBird and FiD on our
dataset. The best-performing model FiD can only achieve 46\% accuracy, still
far behind the human performance of 87\%. We demonstrate that these models are
still lacking the ability to perform robust temporal understanding and
reasoning. Therefore, we believe that our dataset could serve as a benchmark to
empower future studies in temporal reasoning. The dataset and code are released
in~\url{this https URL}.

    

### [[2007.12870] Three-stage intelligent support of clinical decision making for higher trust, validity, and explainability](http://arxiv.org/abs/2007.12870)


  The paper presents an approach for building consistent and applicable
clinical decision support systems (CDSSs) using a data-driven predictive model
aimed at resolving the problem of low applicability and scalability of CDSSs in
real-world applications. The approach is based on a threestage application of
domain-specific and data-driven supportive procedures that are to be integrated
into clinical business processes with higher trust and explainability of the
prediction results and recommendations. Within the considered three stages, the
regulatory policy, data-driven modes, and interpretation procedures are
integrated to enable natural domain-specific interaction with decisionmakers
with sequential narrowing of the intelligent decision support focus. The
proposed methodology enables a higher level of automation, scalability, and
semantic interpretability of CDSSs. The approach was implemented in software
solutions and tested within a case study in T2DM prediction, enabling us to
improve known clinical scales (such as FINDRISK) while keeping the
problem-specific reasoning interface similar to existing applications. Such
inheritance, together with the three-staged approach, provide higher
compatibility of the solution and leads to trust, valid, and explainable
application of data-driven solutions in real-world cases.

    

### [[2012.05370] Algorithmic Risk Assessments Can Alter Human Decision-Making Processes in High-Stakes Government Contexts](http://arxiv.org/abs/2012.05370)


  Governments are increasingly turning to algorithmic risk assessments when
making important decisions, such as whether to release criminal defendants
before trial. Policymakers assert that providing public servants with
algorithmic advice will improve human risk predictions and thereby lead to
better (e.g., fairer) decisions. Yet because many policy decisions require
balancing risk-reduction with competing goals, improving the accuracy of
predictions may not necessarily improve the quality of decisions. If risk
assessments make people more attentive to reducing risk at the expense of other
values, these algorithms would diminish the implementation of public policy
even as they lead to more accurate predictions. Through an experiment with
2,140 lay participants simulating two high-stakes government contexts, we
provide the first direct evidence that risk assessments can systematically
alter how people factor risk into their decisions. These shifts counteracted
the potential benefits of improved prediction accuracy. In the pretrial setting
of our experiment, the risk assessment made participants more sensitive to
increases in perceived risk; this shift increased the racial disparity in
pretrial detention by 1.9%. In the government loans setting of our experiment,
the risk assessment made participants more risk-averse; this shift reduced
government aid by 8.3%. These results demonstrate the potential limits and
harms of attempts to improve public policy by incorporating predictive
algorithms into multifaceted policy decisions. If these observed behaviors
occur in practice, presenting risk assessments to public servants would
generate unexpected and unjust shifts in public policy without being subject to
democratic deliberation or oversight.

    

### [[2103.06816] COVID-19 Smart Chatbot Prototype for Patient Monitoring](http://arxiv.org/abs/2103.06816)


  Many COVID-19 patients developed prolonged symptoms after the infection,
including fatigue, delirium, and headache. The long-term health impact of these
conditions is still not clear. It is necessary to develop a way to follow up
with these patients for monitoring their health status to support timely
intervention and treatment. In the lack of sufficient human resources to follow
up with patients, we propose a novel smart chatbot solution backed with machine
learning to collect information (i.e., generating digital diary) in a
personalized manner. In this article, we describe the design framework and
components of our prototype.

    

### [[2103.11686] IPAPRec: A promising tool for learning high-performance mapless navigation skills with deep reinforcement learning](http://arxiv.org/abs/2103.11686)


  This paper studies how to improve the generalization performance and learning
speed of the navigation agents trained with deep reinforcement learning (DRL).
DRL exhibits huge potential in mapless navigation, but DRL agents performing
well in training scenarios are found to perform poorly in unfamiliar real-world
scenarios. In this work, we present the representation of LiDAR readings as a
key factor behind agents' performance degradation and propose a simple but
powerful input pre-processing (IP) approach to improve the agents' performance.
As this approach uses adaptively parametric reciprocal functions to pre-process
LiDAR readings, we refer to this approach as IPAPRec and its normalized version
as IPAPRecN. IPAPRec/IPAPRecN can highlight important short-distance values and
compress the range of less-important long-distance values in laser scans, which
well addressed the issues induced by conventional representations of laser
scans. Their high performance is validated by extensive simulation and
real-world experiments. The results show that our methods can substantially
improve agents' success rates and greatly reduce the training time compared to
conventional methods.

    

### [[2104.10956] FCOS3D: Fully Convolutional One-Stage Monocular 3D Object Detection](http://arxiv.org/abs/2104.10956)


  Monocular 3D object detection is an important task for autonomous driving
considering its advantage of low cost. It is much more challenging than
conventional 2D cases due to its inherent ill-posed property, which is mainly
reflected in the lack of depth information. Recent progress on 2D detection
offers opportunities to better solving this problem. However, it is non-trivial
to make a general adapted 2D detector work in this 3D task. In this paper, we
study this problem with a practice built on a fully convolutional single-stage
detector and propose a general framework FCOS3D. Specifically, we first
transform the commonly defined 7-DoF 3D targets to the image domain and
decouple them as 2D and 3D attributes. Then the objects are distributed to
different feature levels with consideration of their 2D scales and assigned
only according to the projected 3D-center for the training procedure.
Furthermore, the center-ness is redefined with a 2D Gaussian distribution based
on the 3D-center to fit the 3D target formulation. All of these make this
framework simple yet effective, getting rid of any 2D detection or 2D-3D
correspondence priors. Our solution achieves 1st place out of all the
vision-only methods in the nuScenes 3D detection challenge of NeurIPS 2020.
Code and models are released at this https URL.

    

### [[2010.05807] PATSQL: Efficient Synthesis of SQL Queries from Example Tables with Quick Inference of Projected Columns](http://arxiv.org/abs/2010.05807)


  SQL is one of the most popular tools for data analysis, and it is now used by
an increasing number of users without having expertise in databases. Several
studies have proposed programming-by-example approaches to help such
non-experts to write correct SQL queries. While existing methods support a
variety of SQL features such as aggregation and nested query, they suffer a
significant increase in computational cost as the scale of example tables
increases. In this paper, we propose an efficient algorithm utilizing
properties known in relational algebra to synthesize SQL queries from input and
output tables. Our key insight is that a projection operator in a program
sketch can be lifted above other operators by applying transformation rules in
relational algebra, while preserving the semantics of the program. This enables
a quick inference of appropriate columns in the projection operator, which is
an essential component in synthesis but causes combinatorial explosions in
prior work. We also introduce a novel form of constraints and its top-down
propagation mechanism for efficient sketch completion. We implemented this
algorithm in our tool PATSQL and evaluated it on 226 queries from prior
benchmarks and Kaggle's tutorials. As a result, PATSQL solved 68% of the
benchmarks and found 89% of the solutions within a second. Our tool is
available at this https URL.

    

### [[2010.07080] Concise Outlines for a Complex Logic: A Proof Outline Checker for TaDA (Full Paper)](http://arxiv.org/abs/2010.07080)


  Modern separation logics allow one to prove rich properties of intricate
code, e.g. functional correctness and linearizability of non-blocking
concurrent code. However, this expressiveness leads to a complexity that makes
these logics difficult to apply. Manual proofs or proofs in interactive theorem
provers consist of a large number of steps, often with subtle side conditions.
On the other hand, automation with dedicated verifiers typically requires
sophisticated proof search algorithms that are specific to the given program
logic, resulting in limited tool support that makes it difficult to experiment
with program logics, e.g. when learning, improving, or comparing them. Proof
outline checkers fill this gap. Their input is a program annotated with the
most essential proof steps, just like the proof outlines typically presented in
papers. The tool then checks automatically that this outline represents a valid
proof in the program logic. In this paper, we systematically develop a proof
outline checker for the TaDA logic, which reduces the checking to a simpler
verification problem, for which automated tools exist. Our approach leads to
proof outline checkers that provide substantially more automation than
interactive provers, but are much simpler to develop than custom automatic
verifiers.

    