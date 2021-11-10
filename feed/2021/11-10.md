
## 2021-11-10

### [<title>XGBoost iteration_range defined differently in sklearn API and docs - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost-iteration-range-defined-differently-in-sklearn-api-and-docs/2495/5)

### [<title>XGBoost iteration_range defined differently in sklearn API and docs - XGBoost</title>](https://discuss.xgboost.ai/t/xgboost-iteration-range-defined-differently-in-sklearn-api-and-docs/2495/4)

### [[2111.04801] Improved security solutions for DDoS mitigation in 5G Multi-access Edge Computing](http://arxiv.org/abs/2111.04801)


  Multi-access Edge Computing (MEC) is a 5G-enabling solution that aims to
bring cloud-computing capabilities closer to the end-users. This paper focuses
on mitigation techniques against Distributed Denial-of-Service (DDoS) attacks
in the context of 5G MEC, providing solutions that involve the virtualized
environment and the management entities from the MEC architecture. The proposed
solutions aim to reduce the risk of affecting legitimate traffic in the context
of DDoS attacks. Our work supports the idea of using a network flow collector
that sends the data to an anomaly detection system based on artificial
intelligence techniques and, as an improvement over the previous work, it
contributes to redirecting detected anomalies for isolation to a separate
virtual machine. This virtual machine uses deep packet inspection tools to
analyze the traffic and provides services until the final verdict. We decrease
the risk of compromising the virtual machine that provides services to
legitimate users by isolating the suspicious traffic. The management entities
of the MEC architecture allow to re-instantiate or reconfigure the virtual
machines. Hence, if the machine inspecting the isolated traffic crashes because
of an attack, the damaged machine can be restored while the services provided
to legitimate users are not affected.

    

### [[2111.05034] Classifying DNS Servers based on Response Message Matrix using Machine Learning](http://arxiv.org/abs/2111.05034)


  Improperly configured domain name system (DNS) servers are sometimes used as
packet reflectors as part of a DoS or DDoS attack. Detecting packets created as
a result of this activity is logically possible by monitoring the DNS request
and response traffic. Any response that does not have a corresponding request
can be considered a reflected message; checking and tracking every DNS packet,
however, is a non-trivial operation. In this paper, we propose a detection
mechanism for DNS servers used as reflectors by using a DNS server feature
matrix built from a small number of packets and a machine learning algorithm.
The F1 score of bad DNS server detection was more than 0.9 when the test and
training data are generated within the same day, and more than 0.7 for the data
not used for the training and testing phase of the same day.

    

### [[2111.04730] Emotional Prosody Control for Speech Generation](http://arxiv.org/abs/2111.04730)


  Machine-generated speech is characterized by its limited or unnatural
emotional variation. Current text to speech systems generates speech with
either a flat emotion, emotion selected from a predefined set, average
variation learned from prosody sequences in training data or transferred from a
source style. We propose a text to speech(TTS) system, where a user can choose
the emotion of generated speech from a continuous and meaningful emotion space
(Arousal-Valence space). The proposed TTS system can generate speech from the
text in any speaker's style, with fine control of emotion. We show that the
system works on emotion unseen during training and can scale to previously
unseen speakers given his/her speech sample. Our work expands the horizon of
the state-of-the-art FastSpeech2 backbone to a multi-speaker setting and gives
it much-coveted continuous (and interpretable) affective control, without any
observable degradation in the quality of the synthesized speech.

    

### [[2111.04731] Survey of Deep Learning Methods for Inverse Problems](http://arxiv.org/abs/2111.04731)


  In this paper we investigate a variety of deep learning strategies for
solving inverse problems. We classify existing deep learning solutions for
inverse problems into three categories of Direct Mapping, Data Consistency
Optimizer, and Deep Regularizer. We choose a sample of each inverse problem
type, so as to compare the robustness of the three categories, and report a
statistical analysis of their differences. We perform extensive experiments on
the classic problem of linear regression and three well-known inverse problems
in computer vision, namely image denoising, 3D human face inverse rendering,
and object tracking, selected as representative prototypes for each class of
inverse problems. The overall results and the statistical analyses show that
the solution categories have a robustness behaviour dependent on the type of
inverse problem domain, and specifically dependent on whether or not the
problem includes measurement outliers. Based on our experimental results, we
conclude by proposing the most robust solution category for each inverse
problem class.

    

### [[2111.04732] Use of 1D-CNN for input data size reduction of LSTM in Hourly Rainfall-Runoff modeling](http://arxiv.org/abs/2111.04732)


  An architecture consisting of a serial coupling of the one-dimensional
convolutional neural network (1D-CNN) and the long short-term memory (LSTM)
network, which is referred as CNNsLSTM, was proposed for hourly-scale
rainfall-runoff modeling in this study. In CNNsLTSM, the CNN component receives
the hourly meteorological time series data for a long duration, and then the
LSTM component receives the extracted features from 1D-CNN and the hourly
meteorological time series data for a short-duration. As a case study, CNNsLSTM
was implemented for hourly rainfall-runoff modeling at the Ishikari River
watershed, Japan. The meteorological dataset, consists of precipitation, air
temperature, evapotranspiration, and long- and short-wave radiation, were
utilized as input, and the river flow was used as the target data. To evaluate
the performance of proposed CNNsLSTM, results of CNNsLSTM were compared with
those of 1D-CNN, LSTM only with hourly inputs (LSTMwHour), parallel
architecture of 1D-CNN and LSTM (CNNpLSTM), and the LSTM architecture which
uses both daily and hourly input data (LSTMwDpH). CNNsLSTM showed clear
improvements on the estimation accuracy compared to the three conventional
architectures (1D-CNN, LSTMwHour, and CNNpLSTM), and recently proposed
LSTMwDpH. In comparison to observed flows, the median of the NSE values for the
test period are 0.455-0.469 for 1D-CNN (based on NCHF=8, 16, and 32, the
numbers of the channels of the feature map of the first layer of CNN),
0.639-0.656 for CNNpLSTM (based on NCHF=8, 16, and 32), 0.745 for LSTMwHour,
0.831 for LSTMwDpH, and 0.865-0.873 for CNNsLSTM (based on NCHF=8, 16, and 32).
Furthermore, the proposed CNNsLSTM reduces the median RMSE of 1D-CNN by
50.2%-51.4%, CNNpLSTM by 37.4%-40.8%, LSTMwHour by 27.3%-29.5%, and LSTMwDpH by
10.6%-13.4%.

    

### [[2111.04746] Realizable Learning is All You Need](http://arxiv.org/abs/2111.04746)


  The equivalence of realizable and agnostic learnability is a fundamental
phenomenon in learning theory. With variants ranging from classical settings
like PAC learning and regression to recent trends such as adversarially robust
and private learning, it's surprising that we still lack a unified theory;
traditional proofs of the equivalence tend to be disparate, and rely on strong
model-specific assumptions like uniform convergence and sample compression.
In this work, we give the first model-independent framework explaining the
equivalence of realizable and agnostic learnability: a three-line blackbox
reduction that simplifies, unifies, and extends our understanding across a wide
variety of settings. This includes models with no known characterization of
learnability such as learning with arbitrary distributional assumptions or
general loss, as well as a host of other popular settings such as robust
learning, partial learning, fair learning, and the statistical query model.
More generally, we argue that the equivalence of realizable and agnostic
learning is actually a special case of a broader phenomenon we call property
generalization: any desirable property of a learning algorithm (e.g.\ noise
tolerance, privacy, stability) that can be satisfied over finite hypothesis
classes extends (possibly in some variation) to any learnable hypothesis class.

    

### [[2111.04779] ML-EXray: Visibility into ML Deployment on the Edge](http://arxiv.org/abs/2111.04779)


  Benefiting from expanding cloud infrastructure, deep neural networks (DNNs)
today have increasingly high performance when trained in the cloud. Researchers
spend months of effort competing for an extra few percentage points of model
accuracy. However, when these models are actually deployed on edge devices in
practice, very often, the performance can abruptly drop over 10% without
obvious reasons. The key challenge is that there is not much visibility into ML
inference execution on edge devices, and very little awareness of potential
issues during the edge deployment process. We present ML-EXray, an end-to-end
framework, which provides visibility into layer-level details of the ML
execution, and helps developers analyze and debug cloud-to-edge deployment
issues. More often than not, the reason for sub-optimal edge performance does
not only lie in the model itself, but every operation throughout the data flow
and the deployment process. Evaluations show that ML-EXray can effectively
catch deployment issues, such as pre-processing bugs, quantization issues,
suboptimal kernels, etc. Using ML-EXray, users need to write less than 15 lines
of code to fully examine the edge deployment pipeline. Eradicating these
issues, ML-EXray can correct model performance by up to 30%, pinpoint
error-prone layers, and guide users to optimize kernel execution latency by two
orders of magnitude. Code and APIs will be released as an open-source
multi-lingual instrumentation library and a Python deployment validation
library.

    

### [[2111.04794] Deep Learning Approach for Aggressive Driving Behaviour Detection](http://arxiv.org/abs/2111.04794)


  Driving behaviour is one of the primary causes of road crashes and accidents,
and these can be decreased by identifying and minimizing aggressive driving
behaviour. This study identifies the timesteps when a driver in different
circumstances (rush, mental conflicts, reprisal) begins to drive aggressively.
An observer (real or virtual) is needed to examine driving behaviour to
discover aggressive driving occasions; we overcome this problem by using a
smartphone's GPS sensor to detect locations and classify drivers' driving
behaviour every three minutes. To detect timeseries patterns in our dataset, we
employ RNN (GRU, LSTM) algorithms to identify patterns during the driving
course. The algorithm is independent of road, vehicle, position, or driver
characteristics. We conclude that three minutes (or more) of driving (120
seconds of GPS data) is sufficient to identify driver behaviour. The results
show high accuracy and a high F1 score.

    

### [[2111.04798] TAGLETS: A System for Automatic Semi-Supervised Learning with Auxiliary Data](http://arxiv.org/abs/2111.04798)


  Machine learning practitioners often have access to a spectrum of data:
labeled data for the target task (which is often limited), unlabeled data, and
auxiliary data, the many available labeled datasets for other tasks. We
describe TAGLETS, a system built to study techniques for automatically
exploiting all three types of data and creating high-quality, servable
classifiers. The key components of TAGLETS are: (1) auxiliary data organized
according to a knowledge graph, (2) modules encapsulating different methods for
exploiting auxiliary and unlabeled data, and (3) a distillation stage in which
the ensembled modules are combined into a servable model. We compare TAGLETS
with state-of-the-art transfer learning and semi-supervised learning methods on
four image classification tasks. Our study covers a range of settings, varying
the amount of labeled data and the semantic relatedness of the auxiliary data
to the target task. We find that the intelligent incorporation of auxiliary and
unlabeled data into multiple learning techniques enables TAGLETS to match-and
most often significantly surpass-these alternatives. TAGLETS is available as an
open-source system at this http URL.

    

### [[2111.04804] Approximating Fair Clustering with Cascaded Norm Objectives](http://arxiv.org/abs/2111.04804)


  We introduce the $(p,q)$-Fair Clustering problem. In this problem, we are
given a set of points $P$ and a collection of different weight functions $W$.
We would like to find a clustering which minimizes the $\ell_q$-norm of the
vector over $W$ of the $\ell_p$-norms of the weighted distances of points in
$P$ from the centers. This generalizes various clustering problems, including
Socially Fair $k$-Median and $k$-Means, and is closely connected to other
problems such as Densest $k$-Subgraph and Min $k$-Union.
We utilize convex programming techniques to approximate the $(p,q)$-Fair
Clustering problem for different values of $p$ and $q$. When $p\geq q$, we get
an $O(k^{(p-q)/(2pq)})$, which nearly matches a $k^{\Omega((p-q)/(pq))}$ lower
bound based on conjectured hardness of Min $k$-Union and other problems. When
$q\geq p$, we get an approximation which is independent of the size of the
input for bounded $p,q$, and also matches the recent $O((\log n/(\log\log
n))^{1/p})$-approximation for $(p, \infty)$-Fair Clustering by Makarychev and
Vakilian (COLT 2021).

    

### [[2111.04805] Solution to the Non-Monotonicity and Crossing Problems in Quantile Regression](http://arxiv.org/abs/2111.04805)


  This paper proposes a new method to address the long-standing problem of lack
of monotonicity in estimation of the conditional and structural quantile
function, also known as quantile crossing problem. Quantile regression is a
very powerful tool in data science in general and econometrics in particular.
Unfortunately, the crossing problem has been confounding researchers and
practitioners alike for over 4 decades. Numerous attempts have been made to
find an acceptable solution but no simple and general solution has been found
to date. This paper describes an elegant solution to the problem which is based
on a single mathematical equation that is easy to understand and implement in R
and Python, while greatly reducing the crossing problem. It will be very
important in all areas where quantile regression is routinely used and may also
find application in robust regression, especially in the context of machine
learning.

    

### [[2111.04807] Unsupervised Approaches for Out-Of-Distribution Dermoscopic Lesion Detection](http://arxiv.org/abs/2111.04807)


  There are limited works showing the efficacy of unsupervised
Out-of-Distribution (OOD) methods on complex medical data. Here, we present
preliminary findings of our unsupervised OOD detection algorithm, SimCLR-LOF,
as well as a recent state of the art approach (SSD), applied on medical images.
SimCLR-LOF learns semantically meaningful features using SimCLR and uses LOF
for scoring if a test sample is OOD. We evaluated on the multi-source
International Skin Imaging Collaboration (ISIC) 2019 dataset, and show results
that are competitive with SSD as well as with recent supervised approaches
applied on the same data.

    

### [[2111.04820] Explaining Hyperparameter Optimization via Partial Dependence Plots](http://arxiv.org/abs/2111.04820)


  Automated hyperparameter optimization (HPO) can support practitioners to
obtain peak performance in machine learning models. However, there is often a
lack of valuable insights into the effects of different hyperparameters on the
final model performance. This lack of explainability makes it difficult to
trust and understand the automated HPO process and its results. We suggest
using interpretable machine learning (IML) to gain insights from the
experimental data obtained during HPO with Bayesian optimization (BO). BO tends
to focus on promising regions with potential high-performance configurations
and thus induces a sampling bias. Hence, many IML techniques, such as the
partial dependence plot (PDP), carry the risk of generating biased
interpretations. By leveraging the posterior uncertainty of the BO surrogate
model, we introduce a variant of the PDP with estimated confidence bands. We
propose to partition the hyperparameter space to obtain more confident and
reliable PDPs in relevant sub-regions. In an experimental study, we provide
quantitative evidence for the increased quality of the PDPs within sub-regions.

    

### [[2111.04824] MassFormer: Tandem Mass Spectrum Prediction with Graph Transformers](http://arxiv.org/abs/2111.04824)


  Mass spectrometry is a key tool in the study of small molecules, playing an
important role in metabolomics, drug discovery, and environmental chemistry.
Tandem mass spectra capture fragmentation patterns that provide key structural
information about a molecule and help with its identification. Practitioners
often rely on spectral library searches to match unknown spectra with known
compounds. However, such search-based methods are limited by availability of
reference experimental data. In this work we show that graph transformers can
be used to accurately predict tandem mass spectra. Our model, MassFormer,
outperforms competing deep learning approaches for spectrum prediction, and
includes an interpretable attention mechanism to help explain predictions. We
demonstrate that our model can be used to improve reference library coverage on
a synthetic molecule identification task. Through quantitative analysis and
visual inspection, we verify that our model recovers prior knowledge about the
effect of collision energy on the generated spectrum. We evaluate our model on
different types of mass spectra from two independent MS datasets and show that
its performance generalizes. Code available at this http URL.

    

### [[2111.04826] Inferential SIR-GN: Scalable Graph Representation Learning](http://arxiv.org/abs/2111.04826)


  Graph representation learning methods generate numerical vector
representations for the nodes in a network, thereby enabling their use in
standard machine learning models. These methods aim to preserve relational
information, such that nodes that are similar in the graph are found close to
one another in the representation space. Similarity can be based largely on one
of two notions: connectivity or structural role. In tasks where node structural
role is important, connectivity based methods show poor performance. Recent
work has begun to focus on scalability of learning methods to massive graphs of
millions to billions of nodes and edges. Many unsupervised node representation
learning algorithms are incapable of scaling to large graphs, and are unable to
generate node representations for unseen nodes. In this work, we propose
Inferential SIR-GN, a model which is pre-trained on random graphs, then
computes node representations rapidly, including for very large networks. We
demonstrate that the model is able to capture node's structural role
information, and show excellent performance at node and graph classification
tasks, on unseen networks. Additionally, we observe the scalability of
Inferential SIR-GN is comparable to the fastest current approaches for massive
graphs.

    

### [[2111.04833] Solving Marginal MAP Exactly by Probabilistic Circuit Transformations](http://arxiv.org/abs/2111.04833)


  Probabilistic circuits (PCs) are a class of tractable probabilistic models
that allow efficient, often linear-time, inference of queries such as marginals
and most probable explanations (MPE). However, marginal MAP, which is central
to many decision-making problems, remains a hard query for PCs unless they
satisfy highly restrictive structural constraints. In this paper, we develop a
pruning algorithm that removes parts of the PC that are irrelevant to a
marginal MAP query, shrinking the PC while maintaining the correct solution.
This pruning technique is so effective that we are able to build a marginal MAP
solver based solely on iteratively transforming the circuit -- no search is
required. We empirically demonstrate the efficacy of our approach on real-world
datasets.

    

### [[2111.04835] Safe Optimal Design with Applications in Policy Learning](http://arxiv.org/abs/2111.04835)


  Motivated by practical needs in online experimentation and off-policy
learning, we study the problem of safe optimal design, where we develop a data
logging policy that efficiently explores while achieving competitive rewards
with a baseline production policy. We first show, perhaps surprisingly, that a
common practice of mixing the production policy with uniform exploration,
despite being safe, is sub-optimal in maximizing information gain. Then we
propose a safe optimal logging policy for the case when no side information
about the actions' expected rewards is available. We improve upon this design
by considering side information and also extend both approaches to a large
number of actions with a linear reward model. We analyze how our data logging
policies impact errors in off-policy learning. Finally, we empirically validate
the benefit of our designs by conducting extensive experiments.

    

### [[2111.04838] Efficient estimates of optimal transport via low-dimensional embeddings](http://arxiv.org/abs/2111.04838)


  Optimal transport distances (OT) have been widely used in recent work in
Machine Learning as ways to compare probability distributions. These are costly
to compute when the data lives in high dimension. Recent work by Paty et al.,
2019, aims specifically at reducing this cost by computing OT using low-rank
projections of the data (seen as discrete measures). We extend this approach
and show that one can approximate OT distances by using more general families
of maps provided they are 1-Lipschitz. The best estimate is obtained by
maximising OT over the given family. As OT calculations are done after mapping
data to a lower dimensional space, our method scales well with the original
data dimension. We demonstrate the idea with neural networks.

    

### [[2111.04840] Cold Brew: Distilling Graph Node Representations with Incomplete or Missing Neighborhoods](http://arxiv.org/abs/2111.04840)


  Graph Neural Networks (GNNs) have achieved state of the art performance in
node classification, regression, and recommendation tasks. GNNs work well when
high-quality and rich connectivity structure is available. However, this
requirement is not satisfied in many real world graphs where the node degrees
have power-law distributions as many nodes have either fewer or noisy
connections. The extreme case of this situation is a node may have no neighbors
at all, called Strict Cold Start (SCS) scenario. This forces the prediction
models to rely completely on the node's input features. We propose Cold Brew to
address the SCS and noisy neighbor setting compared to pointwise and other
graph-based models via a distillation approach. We introduce
feature-contribution ratio (FCR), a metric to study the viability of using
inductive GNNs to solve the SCS problem and to select the best architecture for
SCS generalization. We experimentally show FCR disentangles the contributions
of various components of graph datasets and demonstrate the superior
performance of Cold Brew on several public benchmarks and proprietary
e-commerce datasets. The source code for our approach is available at:
this https URL.

    

### [[2111.04850] Dueling RL: Reinforcement Learning with Trajectory Preferences](http://arxiv.org/abs/2111.04850)


  We consider the problem of preference based reinforcement learning (PbRL),
where, unlike traditional reinforcement learning, an agent receives feedback
only in terms of a 1 bit (0/1) preference over a trajectory pair instead of
absolute rewards for them. The success of the traditional RL framework
crucially relies on the underlying agent-reward model, which, however, depends
on how accurately a system designer can express an appropriate reward function
and often a non-trivial task. The main novelty of our framework is the ability
to learn from preference-based trajectory feedback that eliminates the need to
hand-craft numeric reward models. This paper sets up a formal framework for the
PbRL problem with non-markovian rewards, where the trajectory preferences are
encoded by a generalized linear model of dimension $d$. Assuming the transition
model is known, we then propose an algorithm with almost optimal regret
guarantee of $\tilde {\mathcal{O}}\left( SH d \log (T / \delta) \sqrt{T}
\right)$. We further, extend the above algorithm to the case of unknown
transition dynamics, and provide an algorithm with near optimal regret
guarantee $\widetilde{\mathcal{O}}((\sqrt{d} + H^2 + |\mathcal{S}|)\sqrt{dT}
+\sqrt{|\mathcal{S}||\mathcal{A}|TH} )$. To the best of our knowledge, our work
is one of the first to give tight regret guarantees for preference based RL
problems with trajectory preferences.

    

### [[2111.04857] Model-assisted deep learning of rare extreme events from partial observations](http://arxiv.org/abs/2111.04857)


  To predict rare extreme events using deep neural networks, one encounters the
so-called small data problem because even long-term observations often contain
few extreme events. Here, we investigate a model-assisted framework where the
training data is obtained from numerical simulations, as opposed to
observations, with adequate samples from extreme events. However, to ensure the
trained networks are applicable in practice, the training is not performed on
the full simulation data; instead we only use a small subset of observable
quantities which can be measured in practice. We investigate the feasibility of
this model-assisted framework on three different dynamical systems (Rossler
attractor, FitzHugh--Nagumo model, and a turbulent fluid flow) and three
different deep neural network architectures (feedforward, long short-term
memory, and reservoir computing). In each case, we study the prediction
accuracy, robustness to noise, reproducibility under repeated training, and
sensitivity to the type of input data. In particular, we find long short-term
memory networks to be most robust to noise and to yield relatively accurate
predictions, while requiring minimal fine-tuning of the hyperparameters.

    

### [[2111.04865] On Assessing The Safety of Reinforcement Learning algorithms Using Formal Methods](http://arxiv.org/abs/2111.04865)


  The increasing adoption of Reinforcement Learning in safety-critical systems
domains such as autonomous vehicles, health, and aviation raises the need for
ensuring their safety. Existing safety mechanisms such as adversarial training,
adversarial detection, and robust learning are not always adapted to all
disturbances in which the agent is deployed. Those disturbances include moving
adversaries whose behavior can be unpredictable by the agent, and as a matter
of fact harmful to its learning. Ensuring the safety of critical systems also
requires methods that give formal guarantees on the behaviour of the agent
evolving in a perturbed environment. It is therefore necessary to propose new
solutions adapted to the learning challenges faced by the agent. In this paper,
first we generate adversarial agents that exhibit flaws in the agent's policy
by presenting moving adversaries. Secondly, We use reward shaping and a
modified Q-learning algorithm as defense mechanisms to improve the agent's
policy when facing adversarial perturbations. Finally, probabilistic model
checking is employed to evaluate the effectiveness of both mechanisms. We have
conducted experiments on a discrete grid world with a single agent facing
non-learning and learning adversaries. Our results show a diminution in the
number of collisions between the agent and the adversaries. Probabilistic model
checking provides lower and upper probabilistic bounds regarding the agent's
safety in the adversarial environment.

    

### [[2111.04867] Synthesizing Collective Communication Algorithms for Heterogeneous Networks with TACCL](http://arxiv.org/abs/2111.04867)


  Large ML models and datasets have necessitated the use of multi-GPU systems
for distributed model training. To harness the power offered by multi-GPU
systems, it is critical to eliminate bottlenecks in inter-GPU communication - a
problem made challenging by the heterogeneous nature of interconnects. In this
work, we present TACCL, a synthesizer for collective communication primitives
for large-scale multi-GPU systems. TACCL encodes a profiled topology and input
size into a synthesis problem to generate optimized communication algorithms.
TACCL is built on top of the standard NVIDIA Collective Communication Library
(NCCL), allowing it to be a drop-in replacement for GPU communication in
frameworks like PyTorch with minimal changes. TACCL generates algorithms for
communication primitives like Allgather, Alltoall, and Allreduce that are up to
$3\times$ faster than NCCL. Using TACCL's algorithms speeds up the end-to-end
training of an internal mixture of experts model by $17\%$. By decomposing the
optimization problem into parts and leveraging the symmetry in multi-GPU
topologies, TACCL synthesizes collectives for up to 80-GPUs in less than 3
minutes, at least two orders of magnitude faster than other synthesis-based
state-of-the-art collective communication libraries.

    

### [[2111.04870] A toolkit for data-driven discovery of governing equations in high-noise regimes](http://arxiv.org/abs/2111.04870)


  We consider the data-driven discovery of governing equations from time-series
data in the limit of high noise. The algorithms developed describe an extensive
toolkit of methods for circumventing the deleterious effects of noise in the
context of the sparse identification of nonlinear dynamics (SINDy) framework.
We offer two primary contributions, both focused on noisy data acquired from a
system x' = f(x). First, we propose, for use in high-noise settings, an
extensive toolkit of critically enabling extensions for the SINDy regression
method, to progressively cull functionals from an over-complete library and
yield a set of sparse equations that regress to the derivate x'. These
innovations can extract sparse governing equations and coefficients from
high-noise time-series data (e.g. 300% added noise). For example, it discovers
the correct sparse libraries in the Lorenz system, with median coefficient
estimate errors equal to 1% - 3% (for 50% noise), 6% - 8% (for 100% noise); and
23% - 25% (for 300% noise). The enabling modules in the toolkit are combined
into a single method, but the individual modules can be tactically applied in
other equation discovery methods (SINDy or not) to improve results on
high-noise data. Second, we propose a technique, applicable to any model
discovery method based on x' = f(x), to assess the accuracy of a discovered
model in the context of non-unique solutions due to noisy data. Currently, this
non-uniqueness can obscure a discovered model's accuracy and thus a discovery
method's effectiveness. We describe a technique that uses linear dependencies
among functionals to transform a discovered model into an equivalent form that
is closest to the true model, enabling more accurate assessment of a discovered
model's accuracy.

    

### [[2111.04871] Query-augmented Active Metric Learning](http://arxiv.org/abs/2111.04871)


  In this paper we propose an active metric learning method for clustering with
pairwise constraints. The proposed method actively queries the label of
informative instance pairs, while estimating underlying metrics by
incorporating unlabeled instance pairs, which leads to a more accurate and
efficient clustering process. In particular, we augment the queried constraints
by generating more pairwise labels to provide additional information in
learning a metric to enhance clustering performance. Furthermore, we increase
the robustness of metric learning by updating the learned metric sequentially
and penalizing the irrelevant features adaptively. In addition, we propose a
novel active query strategy that evaluates the information gain of instance
pairs more accurately by incorporating the neighborhood structure, which
improves clustering efficiency without extra labeling cost. In theory, we
provide a tighter error bound of the proposed metric learning method utilizing
augmented queries compared with methods using existing constraints only.
Furthermore, we also investigate the improvement using the active query
strategy instead of random selection. Numerical studies on simulation settings
and real datasets indicate that the proposed method is especially advantageous
when the signal-to-noise ratio between significant features and irrelevant
features is low.

    

### [[2111.04873] An Instance-Dependent Analysis for the Cooperative Multi-Player Multi-Armed Bandit](http://arxiv.org/abs/2111.04873)


  We study the problem of information sharing and cooperation in Multi-Player
Multi-Armed bandits. We propose the first algorithm that achieves logarithmic
regret for this problem. Our results are based on two innovations. First, we
show that a simple modification to a successive elimination strategy can be
used to allow the players to estimate their suboptimality gaps, up to constant
factors, in the absence of collisions. Second, we leverage the first result to
design a communication protocol that successfully uses the small reward of
collisions to coordinate among players, while preserving meaningful
instance-dependent logarithmic regret guarantees.

    

### [[2111.04877] Papaya: Practical, Private, and Scalable Federated Learning](http://arxiv.org/abs/2111.04877)


  Cross-device Federated Learning (FL) is a distributed learning paradigm with
several challenges that differentiate it from traditional distributed learning,
variability in the system characteristics on each device, and millions of
clients coordinating with a central server being primary ones. Most FL systems
described in the literature are synchronous - they perform a synchronized
aggregation of model updates from individual clients. Scaling synchronous FL is
challenging since increasing the number of clients training in parallel leads
to diminishing returns in training speed, analogous to large-batch training.
Moreover, stragglers hinder synchronous FL training. In this work, we outline a
production asynchronous FL system design. Our work tackles the aforementioned
issues, sketches of some of the system design challenges and their solutions,
and touches upon principles that emerged from building a production FL system
for millions of clients. Empirically, we demonstrate that asynchronous FL
converges faster than synchronous FL when training across nearly one hundred
million devices. In particular, in high concurrency settings, asynchronous FL
is 5x faster and has nearly 8x less communication overhead than synchronous FL.

    

### [[2111.04879] EvoLearner: Learning Description Logics with Evolutionary Algorithms](http://arxiv.org/abs/2111.04879)


  Classifying nodes in knowledge graphs is an important task, e.g., predicting
missing types of entities, predicting which molecules cause cancer, or
predicting which drugs are promising treatment candidates. While black-box
models often achieve high predictive performance, they are only post-hoc and
locally explainable and do not allow the learned model to be easily enriched
with domain knowledge. Towards this end, learning description logic concepts
from positive and negative examples has been proposed. However, learning such
concepts often takes a long time and state-of-the-art approaches provide
limited support for literal data values, although they are crucial for many
applications. In this paper, we propose EvoLearner - an evolutionary approach
to learn ALCQ(D), which is the attributive language with complement (ALC)
paired with qualified cardinality restrictions (Q) and data properties (D). We
contribute a novel initialization method for the initial population: starting
from positive examples (nodes in the knowledge graph), we perform biased random
walks and translate them to description logic concepts. Moreover, we improve
support for data properties by maximizing information gain when deciding where
to split the data. We show that our approach significantly outperforms the
state of the art on the benchmarking framework SML-Bench for structured machine
learning. Our ablation study confirms that this is due to our novel
initialization method and support for data properties.

    

### [[2111.04881] Combining Machine Learning with Physics: A Framework for Tracking and Sorting Multiple Dark Solitons](http://arxiv.org/abs/2111.04881)


  In ultracold atom experiments, data often comes in the form of images which
suffer information loss inherent in the techniques used to prepare and measure
the system. This is particularly problematic when the processes of interest are
complicated, such as interactions among excitations in Bose-Einstein
condensates (BECs). In this paper, we describe a framework combining machine
learning (ML) models with physics-based traditional analyses to identify and
track multiple solitonic excitations in images of BECs. We use an ML-based
object detector to locate the solitonic excitations and develop a
physics-informed classifier to sort solitonic excitations into physically
motivated sub-categories. Lastly, we introduce a quality metric quantifying the
likelihood that a specific feature is a kink soliton. Our trained
implementation of this framework -- SolDet -- is publicly available as an
open-source python package. SolDet is broadly applicable to feature
identification in cold atom images when trained on a suitable user-provided
dataset.

    

### [[2111.04888] Active Sampling for Linear Regression Beyond the $\ell_2$ Norm](http://arxiv.org/abs/2111.04888)


  We study active sampling algorithms for linear regression, which aim to query
only a small number of entries of a target vector $b\in\mathbb{R}^n$ and output
a near minimizer to $\min_{x\in\mathbb{R}^d}\|Ax-b\|$, where $A\in\mathbb{R}^{n
\times d}$ is a design matrix and $\|\cdot\|$ is some loss function.
For $\ell_p$ norm regression for any $0<p<\infty$, we give an algorithm based
on Lewis weight sampling that outputs a $(1+\epsilon)$ approximate solution
using just $\tilde{O}(d^{\max(1,{p/2})}/\mathrm{poly}(\epsilon))$ queries to
$b$. We show that this dependence on $d$ is optimal, up to logarithmic factors.
Our result resolves a recent open question of Chen and Dereziński, who gave
near optimal bounds for the $\ell_1$ norm, and suboptimal bounds for $\ell_p$
regression with $p\in(1,2)$.
We also provide the first total sensitivity upper bound of
$O(d^{\max\{1,p/2\}}\log^2 n)$ for loss functions with at most degree $p$
polynomial growth. This improves a recent result of Tukan, Maalouf, and
Feldman. By combining this with our techniques for the $\ell_p$ regression
result, we obtain an active regression algorithm making $\tilde
O(d^{1+\max\{1,p/2\}}/\mathrm{poly}(\epsilon))$ queries, answering another open
question of Chen and Dereziński. For the important special case of the
Huber loss, we further improve our bound to an active sample complexity of
$\tilde O(d^{(1+\sqrt2)/2}/\epsilon^c)$ and a non-active sample complexity of
$\tilde O(d^{4-2\sqrt 2}/\epsilon^c)$, improving a previous $d^4$ bound for
Huber regression due to Clarkson and Woodruff. Our sensitivity bounds have
further implications, improving a variety of previous results using sensitivity
sampling, including Orlicz norm subspace embeddings and robust subspace
approximation. Finally, our active sampling results give the first sublinear
time algorithms for Kronecker product regression under every $\ell_p$ norm.

    

### [[2111.04894] Safe Policy Optimization with Local Generalized Linear Function Approximations](http://arxiv.org/abs/2111.04894)


  Safe exploration is a key to applying reinforcement learning (RL) in
safety-critical systems. Existing safe exploration methods guaranteed safety
under the assumption of regularity, and it has been difficult to apply them to
large-scale real problems. We propose a novel algorithm, SPO-LF, that optimizes
an agent's policy while learning the relation between a locally available
feature obtained by sensors and environmental reward/safety using generalized
linear function approximations. We provide theoretical guarantees on its safety
and optimality. We experimentally show that our algorithm is 1) more efficient
in terms of sample complexity and computational cost and 2) more applicable to
large-scale problems than previous safe RL methods with theoretical guarantees,
and 3) comparably sample-efficient and safer compared with existing advanced
deep RL methods with safety constraints.

    

### [[2111.04898] Machine Learning for Multimodal Electronic Health Records-based Research: Challenges and Perspectives](http://arxiv.org/abs/2111.04898)


  Background: Electronic Health Records (EHRs) contain rich information of
patients' health history, which usually include both structured and
unstructured data. There have been many studies focusing on distilling valuable
information from structured data, such as disease codes, laboratory test
results, and treatments. However, relying on structured data only might be
insufficient in reflecting patients' comprehensive information and such data
may occasionally contain erroneous records. Objective: With the recent advances
of machine learning (ML) and deep learning (DL) techniques, an increasing
number of studies seek to obtain more accurate results by incorporating
unstructured free-text data as well. This paper reviews studies that use
multimodal data, i.e. a combination of structured and unstructured data, from
EHRs as input for conventional ML or DL models to address the targeted tasks.
Materials and Methods: We searched in the Institute of Electrical and
Electronics Engineers (IEEE) Digital Library, PubMed, and Association for
Computing Machinery (ACM) Digital Library for articles related to ML-based
multimodal EHR studies. Results and Discussion: With the final 94 included
studies, we focus on how data from different modalities were combined and
interacted using conventional ML and DL techniques, and how these algorithms
were applied in EHR-related tasks. Further, we investigate the advantages and
limitations of these fusion methods and indicate future directions for ML-based
multimodal EHR research.

    

### [[2111.04901] Label-Aware Distribution Calibration for Long-tailed Classification](http://arxiv.org/abs/2111.04901)


  Real-world data usually present long-tailed distributions. Training on
imbalanced data tends to render neural networks perform well on head classes
while much worse on tail classes. The severe sparseness of training instances
for the tail classes is the main challenge, which results in biased
distribution estimation during training. Plenty of efforts have been devoted to
ameliorating the challenge, including data re-sampling and synthesizing new
training instances for tail classes. However, no prior research has exploited
the transferable knowledge from head classes to tail classes for calibrating
the distribution of tail classes. In this paper, we suppose that tail classes
can be enriched by similar head classes and propose a novel distribution
calibration approach named as label-Aware Distribution Calibration LADC. LADC
transfers the statistics from relevant head classes to infer the distribution
of tail classes. Sampling from calibrated distribution further facilitates
re-balancing the classifier. Experiments on both image and text long-tailed
datasets demonstrate that LADC significantly outperforms existing methods.The
visualization also shows that LADC provides a more accurate distribution
estimation.

    

### [[2111.04906] The Role of Adaptive Optimizers for Honest Private Hyperparameter Selection](http://arxiv.org/abs/2111.04906)


  Hyperparameter optimization is a ubiquitous challenge in machine learning,
and the performance of a trained model depends crucially upon their effective
selection. While a rich set of tools exist for this purpose, there are
currently no practical hyperparameter selection methods under the constraint of
differential privacy (DP). We study honest hyperparameter selection for
differentially private machine learning, in which the process of hyperparameter
tuning is accounted for in the overall privacy budget. To this end, we i) show
that standard composition tools outperform more advanced techniques in many
settings, ii) empirically and theoretically demonstrate an intrinsic connection
between the learning rate and clipping norm hyperparameters, iii) show that
adaptive optimizers like DPAdam enjoy a significant advantage in the process of
honest hyperparameter tuning, and iv) draw upon novel limiting behaviour of
Adam in the DP setting to design a new and more efficient optimizer.

    

### [[2111.04915] Practical, Provably-Correct Interactive Learning in the Realizable Setting: The Power of True Believers](http://arxiv.org/abs/2111.04915)


  We consider interactive learning in the realizable setting and develop a
general framework to handle problems ranging from best arm identification to
active classification. We begin our investigation with the observation that
agnostic algorithms \emph{cannot} be minimax-optimal in the realizable setting.
Hence, we design novel computationally efficient algorithms for the realizable
setting that match the minimax lower bound up to logarithmic factors and are
general-purpose, accommodating a wide variety of function classes including
kernel methods, H{ö}lder smooth functions, and convex functions. The sample
complexities of our algorithms can be quantified in terms of well-known
quantities like the extended teaching dimension and haystack dimension.
However, unlike algorithms based directly on those combinatorial quantities,
our algorithms are computationally efficient. To achieve computational
efficiency, our algorithms sample from the version space using Monte Carlo
"hit-and-run" algorithms instead of maintaining the version space explicitly.
Our approach has two key strengths. First, it is simple, consisting of two
unifying, greedy algorithms. Second, our algorithms have the capability to
seamlessly leverage prior knowledge that is often available and useful in
practice. In addition to our new theoretical results, we demonstrate
empirically that our algorithms are competitive with Gaussian process UCB
methods.

    

### [[2111.04927] Self-Interpretable Model with TransformationEquivariant Interpretation](http://arxiv.org/abs/2111.04927)


  In this paper, we propose a self-interpretable model SITE with
transformation-equivariant interpretations. We focus on the robustness and
self-consistency of the interpretations of geometric transformations. Apart
from the transformation equivariance, as a self-interpretable model, SITE has
comparable expressive power as the benchmark black-box classifiers, while being
able to present faithful and robust interpretations with high quality. It is
worth noticing that although applied in most of the CNN visualization methods,
the bilinear upsampling approximation is a rough approximation, which can only
provide interpretations in the form of heatmaps (instead of pixel-wise). It
remains an open question whether such interpretations can be direct to the
input space (as shown in the MNIST experiments). Besides, we consider the
translation and rotation transformations in our model. In future work, we will
explore the robust interpretations under more complex transformations such as
scaling and distortion. Moreover, we clarify that SITE is not limited to
geometric transformation (that we used in the computer vision domain), and will
explore SITEin other domains in future work.

    

### [[2111.04930] Optimizing Bayesian acquisition functions in Gaussian Processes](http://arxiv.org/abs/2111.04930)


  Bayesian Optimization is an effective method for searching the global maxima
of an objective function especially if the function is unknown. The process
comprises of using a surrogate function and choosing an acquisition function
followed by optimizing the acquisition function to find the next sampling
point. This paper analyzes different acquistion functions like Maximum
Probability of Improvement and Expected Improvement and various optimizers like
L-BFGS and TNC to optimize the acquisitions functions for finding the next
sampling point. Along with the analysis of time taken, the paper also shows the
importance of position of initial samples chosen.

    

### [[2111.04936] An Interactive Visualization Tool for Understanding Active Learning](http://arxiv.org/abs/2111.04936)


  Despite recent progress in artificial intelligence and machine learning, many
state-of-the-art methods suffer from a lack of explainability and transparency.
The ability to interpret the predictions made by machine learning models and
accurately evaluate these models is crucially important. In this paper, we
present an interactive visualization tool to elucidate the training process of
active learning. This tool enables one to select a sample of interesting data
points, view how their prediction values change at different querying stages,
and thus better understand when and how active learning works. Additionally,
users can utilize this tool to compare different active learning strategies
simultaneously and inspect why some strategies outperform others in certain
contexts. With some preliminary experiments, we demonstrate that our
visualization panel has a great potential to be used in various active learning
experiments and help users evaluate their models appropriately.

    

### [[2111.04941] Solving PDE-constrained Control Problems using Operator Learning](http://arxiv.org/abs/2111.04941)


  The modeling and control of complex physical dynamics are essential in
real-world problems. We propose a novel framework that is generally applicable
to solving PDE-constrained optimal control problems by introducing surrogate
models for PDE solution operators with special regularizers. The procedure of
the proposed framework is divided into two phases: solution operator learning
for PDE constraints (Phase 1) and searching for optimal control (Phase 2). Once
the surrogate model is trained in Phase 1, the optimal control can be inferred
in Phase 2 without intensive computations. Our framework can be applied to both
data-driven and data-free cases. We demonstrate the successful application of
our method to various optimal control problems for different control variables
with diverse PDE constraints from the Poisson equation to Burgers' equation.

    

### [[2111.04942] Learning from Multiple Time Series: A Deep Disentangled Approach to Diversified Time Series Forecasting](http://arxiv.org/abs/2111.04942)


  Time series forecasting is a significant problem in many applications, e.g.,
financial predictions and business optimization. Modern datasets can have
multiple correlated time series, which are often generated with global (shared)
regularities and local (specific) dynamics. In this paper, we seek to tackle
such forecasting problems with DeepDGL, a deep forecasting model that
disentangles dynamics into global and local temporal patterns. DeepDGL employs
an encoder-decoder architecture, consisting of two encoders to learn global and
local temporal patterns, respectively, and a decoder to make multi-step
forecasting. Specifically, to model complicated global patterns, the vector
quantization (VQ) module is introduced, allowing the global feature encoder to
learn a shared codebook among all time series. To model diversified and
heterogenous local patterns, an adaptive parameter generation module enhanced
by the contrastive multi-horizon coding (CMC) is proposed to generate the
parameters of the local feature encoder for each individual time series, which
maximizes the mutual information between the series-specific context variable
and the long/short-term representations of the corresponding time series. Our
experiments on several real-world datasets show that DeepDGL outperforms
existing state-of-the-art models.

    

### [[2111.04949] How to Train Your Neural Network: A Comparative Evaluation](http://arxiv.org/abs/2111.04949)


  The field of deep learning has witnessed a remarkable shift towards extremely
compute- and memory-intensive neural networks. These newer larger models have
enabled researchers to advance state-of-the-art tools across a variety of
fields. This phenomenon has spurred the development of algorithms for
distributed training of neural networks over a larger number of hardware
accelerators. In this paper, we discuss and compare current state-of-the-art
frameworks for large scale distributed deep learning. First, we survey current
practices in distributed learning and identify the different types of
parallelism used. Then, we present empirical results comparing their
performance on large image and language training tasks. Additionally, we
address their statistical efficiency and memory consumption behavior. Based on
our results, we discuss algorithmic and implementation portions of each
framework which hinder performance.

    

### [[2111.04964] On Representation Knowledge Distillation for Graph Neural Networks](http://arxiv.org/abs/2111.04964)


  Knowledge distillation is a promising learning paradigm for boosting the
performance and reliability of resource-efficient graph neural networks (GNNs)
using more expressive yet cumbersome teacher models. Past work on distillation
for GNNs proposed the Local Structure Preserving loss (LSP), which matches
local structural relationships across the student and teacher's node embedding
spaces. In this paper, we make two key contributions:
From a methodological perspective, we study whether preserving the global
topology of how the teacher embeds graph data can be a more effective
distillation objective for GNNs, as real-world graphs often contain latent
interactions and noisy edges. The purely local LSP objective over pre-defined
edges is unable to achieve this as it ignores relationships among disconnected
nodes. We propose two new approaches which better preserve global topology: (1)
Global Structure Preserving loss (GSP), which extends LSP to incorporate all
pairwise interactions; and (2) Graph Contrastive Representation Distillation
(G-CRD), which uses contrastive learning to align the student node embeddings
to those of the teacher in a shared representation space.
From an experimental perspective, we introduce an expanded set of benchmarks
on large-scale real-world datasets where the performance gap between teacher
and student GNNs is non-negligible. We believe this is critical for testing the
efficacy and robustness of knowledge distillation, but was missing from the LSP
study which used synthetic datasets with trivial performance gaps. Experiments
across 4 datasets and 14 heterogeneous GNN architectures show that G-CRD
consistently boosts the performance and robustness of lightweight GNN models,
outperforming the structure preserving approaches, LSP and GSP, as well as
baselines adapted from 2D computer vision.

    

### [[2111.04971] Time-Varying Channel Prediction for RIS-Assisted MU-MISO Networks via Deep Learning](http://arxiv.org/abs/2111.04971)


  To mitigate the effects of shadow fading and obstacle blocking,
reconfigurable intelligent surface (RIS) has become a promising technology to
improve the signal transmission quality of wireless communications by
controlling the reconfigurable passive elements with less hardware cost and
lower power consumption. However, accurate, low-latency and low-pilot-overhead
channel state information (CSI) acquisition remains a considerable challenge in
RIS-assisted systems due to the large number of RIS passive elements. In this
paper, we propose a three-stage joint channel decomposition and prediction
framework to require CSI. The proposed framework exploits the two-timescale
property that the base station (BS)-RIS channel is quasi-static and the
RIS-user equipment (UE) channel is fast time-varying. Specifically, in the
first stage, we use the full-duplex technique to estimate the channel between a
BS's specific antenna and the RIS, addressing the critical scaling ambiguity
problem in the channel decomposition. We then design a novel deep neural
network, namely, the sparse-connected long short-term memory (SCLSTM), and
propose a SCLSTM-based algorithm in the second and third stages, respectively.
The algorithm can simultaneously decompose the BS-RIS channel and RIS-UE
channel from the cascaded channel and capture the temporal relationship of the
RIS-UE channel for prediction. Simulation results show that our proposed
framework has lower pilot overhead than the traditional channel estimation
algorithms, and the proposed SCLSTM-based algorithm can also achieve more
accurate CSI acquisition robustly and effectively.

    

### [[2111.04972] Risk Sensitive Model-Based Reinforcement Learning using Uncertainty Guided Planning](http://arxiv.org/abs/2111.04972)


  Identifying uncertainty and taking mitigating actions is crucial for safe and
trustworthy reinforcement learning agents, especially when deployed in
high-risk environments. In this paper, risk sensitivity is promoted in a
model-based reinforcement learning algorithm by exploiting the ability of a
bootstrap ensemble of dynamics models to estimate environment epistemic
uncertainty. We propose uncertainty guided cross-entropy method planning, which
penalises action sequences that result in high variance state predictions
during model rollouts, guiding the agent to known areas of the state space with
low uncertainty. Experiments display the ability for the agent to identify
uncertain regions of the state space during planning and to take actions that
maintain the agent within high confidence areas, without the requirement of
explicit constraints. The result is a reduction in the performance in terms of
attaining reward, displaying a trade-off between risk and return.

    

### [[2111.04976] Analysis of Sectoral Profitability of the Indian Stock Market Using an LSTM Regression Model](http://arxiv.org/abs/2111.04976)


  Predictive model design for accurately predicting future stock prices has
always been considered an interesting and challenging research problem. The
task becomes complex due to the volatile and stochastic nature of the stock
prices in the real world which is affected by numerous controllable and
uncontrollable variables. This paper presents an optimized predictive model
built on long-and-short-term memory (LSTM) architecture for automatically
extracting past stock prices from the web over a specified time interval and
predicting their future prices for a specified forecast horizon, and forecasts
the future stock prices. The model is deployed for making buy and sell
transactions based on its predicted results for 70 important stocks from seven
different sectors listed in the National Stock Exchange (NSE) of India. The
profitability of each sector is derived based on the total profit yielded by
the stocks in that sector over a period from Jan 1, 2010 to Aug 26, 2021. The
sectors are compared based on their profitability values. The prediction
accuracy of the model is also evaluated for each sector. The results indicate
that the model is highly accurate in predicting future stock prices.

    

### [[2111.04981] Wasserstein Adversarially Regularized Graph Autoencoder](http://arxiv.org/abs/2111.04981)


  This paper introduces Wasserstein Adversarially Regularized Graph Autoencoder
(WARGA), an implicit generative algorithm that directly regularizes the latent
distribution of node embedding to a target distribution via the Wasserstein
metric. The proposed method has been validated in tasks of link prediction and
node clustering on real-world graphs, in which WARGA generally outperforms
state-of-the-art models based on Kullback-Leibler (KL) divergence and typical
adversarial framework.

    

### [[2111.04986] Unified Group Fairness on Federated Learning](http://arxiv.org/abs/2111.04986)


  Federated learning (FL) has emerged as an important machine learning paradigm
where a global model is trained based on the private data from distributed
clients. However, most of existing FL algorithms cannot guarantee the
performance fairness towards different clients or different groups of samples
because of the distribution shift. Recent researches focus on achieving
fairness among clients, but they ignore the fairness towards different groups
formed by sensitive attribute(s) (e.g., gender and/or race), which is important
and practical in real applications. To bridge this gap, we formulate the goal
of unified group fairness on FL which is to learn a fair global model with
similar performance on different groups. To achieve the unified group fairness
for arbitrary sensitive attribute(s), we propose a novel FL algorithm, named
Group Distributionally Robust Federated Averaging (G-DRFA), which mitigates the
distribution shift across groups with theoretical analysis of convergence rate.
Specifically, we treat the performance of the federated global model at each
group as an objective and employ the distributionally robust techniques to
maximize the performance of the worst-performing group over an uncertainty set
by group reweighting. We validate the advantages of the G-DRFA algorithm with
various kinds of distribution shift settings in experiments, and the results
show that G-DRFA algorithm outperforms the existing fair federated learning
algorithms on unified group fairness.

    

### [[2111.05008] Misspecified Gaussian Process Bandit Optimization](http://arxiv.org/abs/2111.05008)


  We consider the problem of optimizing a black-box function based on noisy
bandit feedback. Kernelized bandit algorithms have shown strong empirical and
theoretical performance for this problem. They heavily rely on the assumption
that the model is well-specified, however, and can fail without it. Instead, we
introduce a \emph{misspecified} kernelized bandit setting where the unknown
function can be $\epsilon$--uniformly approximated by a function with a bounded
norm in some Reproducing Kernel Hilbert Space (RKHS). We design efficient and
practical algorithms whose performance degrades minimally in the presence of
model misspecification. Specifically, we present two algorithms based on
Gaussian process (GP) methods: an optimistic EC-GP-UCB algorithm that requires
knowing the misspecification error, and Phased GP Uncertainty Sampling, an
elimination-type algorithm that can adapt to unknown model misspecification. We
provide upper bounds on their cumulative regret in terms of $\epsilon$, the
time horizon, and the underlying kernel, and we show that our algorithm
achieves optimal dependence on $\epsilon$ with no prior knowledge of
misspecification. In addition, in a stochastic contextual setting, we show that
EC-GP-UCB can be effectively combined with the regret bound balancing strategy
and attain similar regret bounds despite not knowing $\epsilon$.

    

### [[2111.05011] RAVE: A variational autoencoder for fast and high-quality neural audio synthesis](http://arxiv.org/abs/2111.05011)


  Deep generative models applied to audio have improved by a large margin the
state-of-the-art in many speech and music related tasks. However, as raw
waveform modelling remains an inherently difficult task, audio generative
models are either computationally intensive, rely on low sampling rates, are
complicated to control or restrict the nature of possible signals. Among those
models, Variational AutoEncoders (VAE) give control over the generation by
exposing latent variables, although they usually suffer from low synthesis
quality. In this paper, we introduce a Realtime Audio Variational autoEncoder
(RAVE) allowing both fast and high-quality audio waveform synthesis. We
introduce a novel two-stage training procedure, namely representation learning
and adversarial fine-tuning. We show that using a post-training analysis of the
latent space allows a direct control between the reconstruction fidelity and
the representation compactness. By leveraging a multi-band decomposition of the
raw waveform, we show that our model is the first able to generate 48kHz audio
signals, while simultaneously running 20 times faster than real-time on a
standard laptop CPU. We evaluate synthesis quality using both quantitative and
qualitative subjective experiments and show the superiority of our approach
compared to existing models. Finally, we present applications of our model for
timbre transfer and signal compression. All of our source code and audio
examples are publicly available.

    

### [[2111.05013] Learning to Generalize Compositionally by Transferring Across Semantic Parsing Tasks](http://arxiv.org/abs/2111.05013)


  Neural network models often generalize poorly to mismatched domains or
distributions. In NLP, this issue arises in particular when models are expected
to generalize compositionally, that is, to novel combinations of familiar words
and constructions. We investigate learning representations that facilitate
transfer learning from one compositional task to another: the representation
and the task-specific layers of the models are strategically trained
differently on a pre-finetuning task such that they generalize well on
mismatched splits that require compositionality. We apply this method to
semantic parsing, using three very different datasets, COGS, GeoQuery and SCAN,
used alternately as the pre-finetuning and target task. Our method
significantly improves compositional generalization over baselines on the test
set of the target task, which is held out during fine-tuning. Ablation studies
characterize the utility of the major steps in the proposed algorithm and
support our hypothesis.

    

### [[2111.05059] MMD-ReID: A Simple but Effective Solution for Visible-Thermal Person ReID](http://arxiv.org/abs/2111.05059)


  Learning modality invariant features is central to the problem of
Visible-Thermal cross-modal Person Reidentification (VT-ReID), where query and
gallery images come from different modalities. Existing works implicitly align
the modalities in pixel and feature spaces by either using adversarial learning
or carefully designing feature extraction modules that heavily rely on domain
knowledge. We propose a simple but effective framework, MMD-ReID, that reduces
the modality gap by an explicit discrepancy reduction constraint. MMD-ReID
takes inspiration from Maximum Mean Discrepancy (MMD), a widely used
statistical tool for hypothesis testing that determines the distance between
two distributions. MMD-ReID uses a novel margin-based formulation to match
class-conditional feature distributions of visible and thermal samples to
minimize intra-class distances while maintaining feature discriminability.
MMD-ReID is a simple framework in terms of architecture and loss formulation.
We conduct extensive experiments to demonstrate both qualitatively and
quantitatively the effectiveness of MMD-ReID in aligning the marginal and class
conditional distributions, thus learning both modality-independent and
identity-consistent features. The proposed framework significantly outperforms
the state-of-the-art methods on SYSU-MM01 and RegDB datasets. Code will be
released at this https URL


### [[2111.05062] Prediction of new outlinks for focused crawling](http://arxiv.org/abs/2111.05062)


  Discovering new hyperlinks enables Web crawlers to find new pages that have
not yet been indexed. This is especially important for focused crawlers because
they strive to provide a comprehensive analysis of specific parts of the Web,
thus prioritizing discovery of new pages over discovery of changes in content.
In the literature, changes in hyperlinks and content have been usually
considered simultaneously. However, there is also evidence suggesting that
these two types of changes are not necessarily related. Moreover, many studies
about predicting changes assume that long history of a page is available, which
is unattainable in practice. The aim of this work is to provide a methodology
for detecting new links effectively using a short history. To this end, we use
a dataset of ten crawls at intervals of one week. Our study consists of three
parts. First, we obtain insight in the data by analyzing empirical properties
of the number of new outlinks. We observe that these properties are, on
average, stable over time, but there is a large difference between emergence of
hyperlinks towards pages within and outside the domain of a target page
(internal and external outlinks, respectively). Next, we provide statistical
models for three targets: the link change rate, the presence of new links, and
the number of new links. These models include the features used earlier in the
literature, as well as new features introduced in this work. We analyze
correlation between the features, and investigate their informativeness. A
notable finding is that, if the history of the target page is not available,
then our new features, that represent the history of related pages, are most
predictive for new links in the target page. Finally, we propose ranking
methods as guidelines for focused crawlers to efficiently discover new pages,
which achieve excellent performance with respect to the corresponding targets.

    

### [[2111.05063] Tightening the Approximation Error of Adversarial Risk with Auto Loss Function Search](http://arxiv.org/abs/2111.05063)


  Numerous studies have demonstrated that deep neural networks are easily
misled by adversarial examples. Effectively evaluating the adversarial
robustness of a model is important for its deployment in practical
applications. Currently, a common type of evaluation is to approximate the
adversarial risk of a model as a robustness indicator by constructing malicious
instances and executing attacks. Unfortunately, there is an error (gap) between
the approximate value and the true value. Previous studies manually design
attack methods to achieve a smaller error, which is inefficient and may miss a
better solution. In this paper, we establish the tightening of the
approximation error as an optimization problem and try to solve it with an
algorithm. More specifically, we first analyze that replacing the non-convex
and discontinuous 0-1 loss with a surrogate loss, a necessary compromise in
calculating the approximation, is one of the main reasons for the error. Then
we propose AutoLoss-AR, the first method for searching loss functions for
tightening the approximation error of adversarial risk. Extensive experiments
are conducted in multiple settings. The results demonstrate the effectiveness
of the proposed method: the best-discovered loss functions outperform the
handcrafted baseline by 0.9%-2.9% and 0.7%-2.0% on MNIST and CIFAR-10,
respectively. Besides, we also verify that the searched losses can be
transferred to other settings and explore why they are better than the baseline
by visualizing the local loss landscape.

    

### [[2111.05070] Almost Optimal Universal Lower Bound for Learning Causal DAGs with Atomic Interventions](http://arxiv.org/abs/2111.05070)


  A well-studied challenge that arises in the structure learning problem of
causal directed acyclic graphs (DAG) is that using observational data, one can
only learn the graph up to a "Markov equivalence class" (MEC). The remaining
undirected edges have to be oriented using interventions, which can be very
expensive to perform in applications. Thus, the problem of minimizing the
number of interventions needed to fully orient the MEC has received a lot of
recent attention, and is also the focus of this work. We prove two main
results. The first is a new universal lower bound on the number of atomic
interventions that any algorithm (whether active or passive) would need to
perform in order to orient a given MEC. Our second result shows that this bound
is, in fact, within a factor of two of the size of the smallest set of atomic
interventions that can orient the MEC. Our lower bound is provably better than
previously known lower bounds. The proof of our lower bound is based on the new
notion of CBSP orderings, which are topological orderings of DAGs without
v-structures and satisfy certain special properties. Further, using simulations
on synthetic graphs and by giving examples of special graph families, we show
that our bound is often significantly better.

    

### [[2111.05073] MixACM: Mixup-Based Robustness Transfer via Distillation of Activated Channel Maps](http://arxiv.org/abs/2111.05073)


  Deep neural networks are susceptible to adversarially crafted, small and
imperceptible changes in the natural inputs. The most effective defense
mechanism against these examples is adversarial training which constructs
adversarial examples during training by iterative maximization of loss. The
model is then trained to minimize the loss on these constructed examples. This
min-max optimization requires more data, larger capacity models, and additional
computing resources. It also degrades the standard generalization performance
of a model. Can we achieve robustness more efficiently? In this work, we
explore this question from the perspective of knowledge transfer. First, we
theoretically show the transferability of robustness from an adversarially
trained teacher model to a student model with the help of mixup augmentation.
Second, we propose a novel robustness transfer method called Mixup-Based
Activated Channel Maps (MixACM) Transfer. MixACM transfers robustness from a
robust teacher to a student by matching activated channel maps generated
without expensive adversarial perturbations. Finally, extensive experiments on
multiple datasets and different learning scenarios show our method can transfer
robustness while also improving generalization on natural images.

    

### [[2111.05077] A Statistical Difference Reduction Method for Escaping Backdoor Detection](http://arxiv.org/abs/2111.05077)


  Recent studies show that Deep Neural Networks (DNNs) are vulnerable to
backdoor attacks. An infected model behaves normally on benign inputs, whereas
its prediction will be forced to an attack-specific target on adversarial data.
Several detection methods have been developed to distinguish inputs to defend
against such attacks. The common hypothesis that these defenses rely on is that
there are large statistical differences between the latent representations of
clean and adversarial inputs extracted by the infected model. However, although
it is important, comprehensive research on whether the hypothesis must be true
is lacking. In this paper, we focus on it and study the following relevant
questions: 1) What are the properties of the statistical differences? 2) How to
effectively reduce them without harming the attack intensity? 3) What impact
does this reduction have on difference-based defenses? Our work is carried out
on the three questions. First, by introducing the Maximum Mean Discrepancy
(MMD) as the metric, we identify that the statistical differences of
multi-level representations are all large, not just the highest level. Then, we
propose a Statistical Difference Reduction Method (SDRM) by adding a
multi-level MMD constraint to the loss function during training a backdoor
model to effectively reduce the differences. Last, three typical
difference-based detection methods are examined. The F1 scores of these
defenses drop from 90%-100% on the regularly trained backdoor models to 60%-70%
on the models trained with SDRM on all two datasets, four model architectures,
and four attack methods. The results indicate that the proposed method can be
used to enhance existing attacks to escape backdoor detection algorithms.

    

### [[2111.05095] Speaker Generation](http://arxiv.org/abs/2111.05095)


  This work explores the task of synthesizing speech in nonexistent
human-sounding voices. We call this task "speaker generation", and present
TacoSpawn, a system that performs competitively at this task. TacoSpawn is a
recurrent attention-based text-to-speech model that learns a distribution over
a speaker embedding space, which enables sampling of novel and diverse
speakers. Our method is easy to implement, and does not require transfer
learning from speaker ID systems. We present objective and subjective metrics
for evaluating performance on this task, and demonstrate that our proposed
objective metrics correlate with human perception of speaker similarity. Audio
samples are available on our demo page.

    

### [[2111.05097] Cross-Lingual Citations in English Papers: A Large-Scale Analysis of Prevalence, Usage, and Impact](http://arxiv.org/abs/2111.05097)


  Citation information in scholarly data is an important source of insight into
the reception of publications and the scholarly discourse. Outcomes of citation
analyses and the applicability of citation based machine learning approaches
heavily depend on the completeness of such data. One particular shortcoming of
scholarly data nowadays is that non-English publications are often not included
in data sets, or that language metadata is not available. Because of this,
citations between publications of differing languages (cross-lingual citations)
have only been studied to a very limited degree. In this paper, we present an
analysis of cross-lingual citations based on over one million English papers,
spanning three scientific disciplines and a time span of three decades. Our
investigation covers differences between cited languages and disciplines,
trends over time, and the usage characteristics as well as impact of
cross-lingual citations. Among our findings are an increasing rate of citations
to publications written in Chinese, citations being primarily to local
non-English languages, and consistency in citation intent between cross- and
monolingual citations. To facilitate further research, we make our collected
data and source code publicly available.

    

### [[2111.05100] EEGEyeNet: a Simultaneous Electroencephalography and Eye-tracking Dataset and Benchmark for Eye Movement Prediction](http://arxiv.org/abs/2111.05100)


  We present a new dataset and benchmark with the goal of advancing research in
the intersection of brain activities and eye movements. Our dataset, EEGEyeNet,
consists of simultaneous Electroencephalography (EEG) and Eye-tracking (ET)
recordings from 356 different subjects collected from three different
experimental paradigms. Using this dataset, we also propose a benchmark to
evaluate gaze prediction from EEG measurements. The benchmark consists of three
tasks with an increasing level of difficulty: left-right, angle-amplitude and
absolute position. We run extensive experiments on this benchmark in order to
provide solid baselines, both based on classical machine learning models and on
large neural networks. We release our complete code and data and provide a
simple and easy-to-use interface to evaluate new methods.

    

### [[2111.05108] "How Does It Detect A Malicious App?" Explaining the Predictions of AI-based Android Malware Detector](http://arxiv.org/abs/2111.05108)


  AI methods have been proven to yield impressive performance on Android
malware detection. However, most AI-based methods make predictions of
suspicious samples in a black-box manner without transparency on models'
inference. The expectation on models' explainability and transparency by cyber
security and AI practitioners to assure the trustworthiness increases. In this
article, we present a novel model-agnostic explanation method for AI models
applied for Android malware detection. Our proposed method identifies and
quantifies the data features relevance to the predictions by two steps: i) data
perturbation that generates the synthetic data by manipulating features'
values; and ii) optimization of features attribution values to seek significant
changes of prediction scores on the perturbed data with minimal feature values
changes. The proposed method is validated by three experiments. We firstly
demonstrate that our proposed model explanation method can aid in discovering
how AI models are evaded by adversarial samples quantitatively. In the
following experiments, we compare the explainability and fidelity of our
proposed method with state-of-the-arts, respectively.

    

### [[2111.05113] Membership Inference Attacks Against Self-supervised Speech Models](http://arxiv.org/abs/2111.05113)


  Recently, adapting the idea of self-supervised learning (SSL) on continuous
speech has started gaining attention. SSL models pre-trained on a huge amount
of unlabeled audio can generate general-purpose representations that benefit a
wide variety of speech processing tasks. Despite their ubiquitous deployment,
however, the potential privacy risks of these models have not been well
investigated. In this paper, we present the first privacy analysis on several
SSL speech models using Membership Inference Attacks (MIA) under black-box
access. The experiment results show that these pre-trained models are
vulnerable to MIA and prone to membership information leakage with high
adversarial advantage scores in both utterance-level and speaker-level.
Furthermore, we also conduct several ablation studies to understand the factors
that contribute to the success of MIA.

    

### [[2111.05120] A Deep Learning Technique using Low Sampling rate for residential Non Intrusive Load Monitoring](http://arxiv.org/abs/2111.05120)


  Individual device loads and energy consumption feedback is one of the
important approaches for pursuing users to save energy in residences. This can
help in identifying faulty devices and wasted energy by devices when left On
unused. The main challenge is to identity and estimate the energy consumption
of individual devices without intrusive sensors on each device. Non-intrusive
load monitoring (NILM) or energy disaggregation, is a blind source separation
problem which requires a system to estimate the electricity usage of individual
appliances from the aggregated household energy consumption. In this paper, we
propose a novel deep neural network-based approach for performing load
disaggregation on low frequency power data obtained from residential
households. We combine a series of one-dimensional Convolutional Neural
Networks and Long Short Term Memory (1D CNN-LSTM) to extract features that can
identify active appliances and retrieve their power consumption given the
aggregated household power value. We used CNNs to extract features from main
readings in a given time frame and then used those features to classify if a
given appliance is active at that time period or not. Following that, the
extracted features are used to model a generation problem using LSTM. We train
the LSTM to generate the disaggregated energy consumption of a particular
appliance. Our neural network is capable of generating detailed feedback of
demand-side, providing vital insights to the end-user about their electricity
consumption. The algorithm was designed for low power offline devices such as
ESP32. Empirical calculations show that our model outperforms the
state-of-the-art on the Reference Energy Disaggregation Dataset (REDD).

    

### [[2111.05123] Gated Linear Model induced U-net for surrogate modeling and uncertainty quantification](http://arxiv.org/abs/2111.05123)


  We propose a novel deep learning based surrogate model for solving
high-dimensional uncertainty quantification and uncertainty propagation
problems. The proposed deep learning architecture is developed by integrating
the well-known U-net architecture with the Gaussian Gated Linear Network (GGLN)
and referred to as the Gated Linear Network induced U-net or GLU-net. The
proposed GLU-net treats the uncertainty propagation problem as an image to
image regression and hence, is extremely data efficient. Additionally, it also
provides estimates of the predictive uncertainty. The network architecture of
GLU-net is less complex with 44\% fewer parameters than the contemporary works.
We illustrate the performance of the proposed GLU-net in solving the Darcy flow
problem under uncertainty under the sparse data scenario. We consider the
stochastic input dimensionality to be up to 4225. Benchmark results are
generated using the vanilla Monte Carlo simulation. We observe the proposed
GLU-net to be accurate and extremely efficient even when no information about
the structure of the inputs is provided to the network. Case studies are
performed by varying the training sample size and stochastic input
dimensionality to illustrate the robustness of the proposed approach.

    

### [[2111.05128] Losses, Dissonances, and Distortions](http://arxiv.org/abs/2111.05128)


  In this paper I present a study in using the losses and gradients obtained
during the training of a simple function approximator as a mechanism for
creating musical dissonance and visual distortion in a solo piano performance
setting. These dissonances and distortions become part of an artistic
performance not just by affecting the visualizations, but also by affecting the
artistic musical performance. The system is designed such that the performer
can in turn affect the training process itself, thereby creating a closed
feedback loop between two processes: the training of a machine learning model
and the performance of an improvised piano piece.

    

### [[2111.05136] Using sequential drift detection to test the API economy](http://arxiv.org/abs/2111.05136)


  The API economy refers to the widespread integration of API (advanced
programming interface) microservices, where software applications can
communicate with each other, as a crucial element in business models and
functions. The number of possible ways in which such a system could be used is
huge. It is thus desirable to monitor the usage patterns and identify when the
system is used in a way that was never used before. This provides a warning to
the system analysts and they can ensure uninterrupted operation of the system.
In this work we analyze both histograms and call graph of API usage to
determine if the usage patterns of the system has shifted. We compare the
application of nonparametric statistical and Bayesian sequential analysis to
the problem. This is done in a way that overcomes the issue of repeated
statistical tests and insures statistical significance of the alerts. The
technique was simulated and tested and proven effective in detecting the drift
in various scenarios. We also mention modifications to the technique to
decrease its memory so that it can respond more quickly when the distribution
drift occurs at a delay from when monitoring begins.

    

### [[2111.05139] Human-in-the-Loop Disinformation Detection: Stance, Sentiment, or Something Else?](http://arxiv.org/abs/2111.05139)


  Both politics and pandemics have recently provided ample motivation for the
development of machine learning-enabled disinformation (a.k.a. fake news)
detection algorithms. Existing literature has focused primarily on the
fully-automated case, but the resulting techniques cannot reliably detect
disinformation on the varied topics, sources, and time scales required for
military applications. By leveraging an already-available analyst as a
human-in-the-loop, however, the canonical machine learning techniques of
sentiment analysis, aspect-based sentiment analysis, and stance detection
become plausible methods to use for a partially-automated disinformation
detection system. This paper aims to determine which of these techniques is
best suited for this purpose and how each technique might best be used towards
this end. Training datasets of the same size and nearly identical neural
architectures (a BERT transformer as a word embedder with a single feed-forward
layer thereafter) are used for each approach, which are then tested on
sentiment- and stance-specific datasets to establish a baseline of how well
each method can be used to do the other tasks. Four different datasets relating
to COVID-19 disinformation are used to test the ability of each technique to
detect disinformation on a topic that did not appear in the training data set.
Quantitative and qualitative results from these tests are then used to provide
insight into how best to employ these techniques in practice.

    

### [[2111.05149] Ethically aligned Deep Learning: Unbiased Facial Aesthetic Prediction](http://arxiv.org/abs/2111.05149)


  Facial beauty prediction (FBP) aims to develop a machine that automatically
makes facial attractiveness assessment. In the past those results were highly
correlated with human ratings, therefore also with their bias in annotating. As
artificial intelligence can have racist and discriminatory tendencies, the
cause of skews in the data must be identified. Development of training data and
AI algorithms that are robust against biased information is a new challenge for
scientists. As aesthetic judgement usually is biased, we want to take it one
step further and propose an Unbiased Convolutional Neural Network for FBP.
While it is possible to create network models that can rate attractiveness of
faces on a high level, from an ethical point of view, it is equally important
to make sure the model is unbiased. In this work, we introduce AestheticNet, a
state-of-the-art attractiveness prediction network, which significantly
outperforms competitors with a Pearson Correlation of 0.9601. Additionally, we
propose a new approach for generating a bias-free CNN to improve fairness in
machine learning.

    

### [[2111.05174] CAESynth: Real-Time Timbre Interpolation and Pitch Control with Conditional Autoencoders](http://arxiv.org/abs/2111.05174)


  In this paper, we present a novel audio synthesizer, CAESynth, based on a
conditional autoencoder. CAESynth synthesizes timbre in real-time by
interpolating the reference sounds in their shared latent feature space, while
controlling a pitch independently. We show that training a conditional
autoencoder based on accuracy in timbre classification together with
adversarial regularization of pitch content allows timbre distribution in
latent space to be more effective and stable for timbre interpolation and pitch
conditioning. The proposed method is applicable not only to creation of musical
cues but also to exploration of audio affordance in mixed reality based on
novel timbre mixtures with environmental sounds. We demonstrate by experiments
that CAESynth achieves smooth and high-fidelity audio synthesis in real-time
through timbre interpolation and independent yet accurate pitch control for
musical cues as well as for audio affordance with environmental sound. A Python
implementation along with some generated samples are shared online.

    

### [[2111.05177] On Training Implicit Models](http://arxiv.org/abs/2111.05177)


  This paper focuses on training implicit models of infinite layers.
Specifically, previous works employ implicit differentiation and solve the
exact gradient for the backward propagation. However, is it necessary to
compute such an exact but expensive gradient for training? In this work, we
propose a novel gradient estimate for implicit models, named phantom gradient,
that 1) forgoes the costly computation of the exact gradient; and 2) provides
an update direction empirically preferable to the implicit model training. We
theoretically analyze the condition under which an ascent direction of the loss
landscape could be found, and provide two specific instantiations of the
phantom gradient based on the damped unrolling and Neumann series. Experiments
on large-scale tasks demonstrate that these lightweight phantom gradients
significantly accelerate the backward passes in training implicit models by
roughly 1.7 times, and even boost the performance over approaches based on the
exact gradient on ImageNet.

    

### [[2111.05191] Does Thermal data make the detection systems more reliable?](http://arxiv.org/abs/2111.05191)


  Deep learning-based detection networks have made remarkable progress in
autonomous driving systems (ADS). ADS should have reliable performance across a
variety of ambient lighting and adverse weather conditions. However, luminance
degradation and visual obstructions (such as glare, fog) result in poor quality
images by the visual camera which leads to performance decline. To overcome
these challenges, we explore the idea of leveraging a different data modality
that is disparate yet complementary to the visual data. We propose a
comprehensive detection system based on a multimodal-collaborative framework
that learns from both RGB (from visual cameras) and thermal (from Infrared
cameras) data. This framework trains two networks collaboratively and provides
flexibility in learning optimal features of its own modality while also
incorporating the complementary knowledge of the other. Our extensive empirical
results show that while the improvement in accuracy is nominal, the value lies
in challenging and extremely difficult edge cases which is crucial in
safety-critical applications such as AD. We provide a holistic view of both
merits and limitations of using a thermal imaging system in detection.

    

### [[2111.05193] A Survey on Green Deep Learning](http://arxiv.org/abs/2111.05193)


  In recent years, larger and deeper models are springing up and continuously
pushing state-of-the-art (SOTA) results across various fields like natural
language processing (NLP) and computer vision (CV). However, despite promising
results, it needs to be noted that the computations required by SOTA models
have been increased at an exponential rate. Massive computations not only have
a surprisingly large carbon footprint but also have negative effects on
research inclusiveness and deployment on real-world applications.
Green deep learning is an increasingly hot research field that appeals to
researchers to pay attention to energy usage and carbon emission during model
training and inference. The target is to yield novel results with lightweight
and efficient technologies. Many technologies can be used to achieve this goal,
like model compression and knowledge distillation. This paper focuses on
presenting a systematic review of the development of Green deep learning
technologies. We classify these approaches into four categories: (1) compact
networks, (2) energy-efficient training strategies, (3) energy-efficient
inference approaches, and (4) efficient data usage. For each category, we
discuss the progress that has been achieved and the unresolved challenges.

    

### [[2111.05198] Harmless interpolation in regression and classification with structured features](http://arxiv.org/abs/2111.05198)


  Overparametrized neural networks tend to perfectly fit noisy training data
yet generalize well on test data. Inspired by this empirical observation,
recent work has sought to understand this phenomenon of benign overfitting or
harmless interpolation in the much simpler linear model. Previous theoretical
work critically assumes that either the data features are statistically
independent or the input data is high-dimensional; this precludes general
nonparametric settings with structured feature maps. In this paper, we present
a general and flexible framework for upper bounding regression and
classification risk in a reproducing kernel Hilbert space. A key contribution
is that our framework describes precise sufficient conditions on the data Gram
matrix under which harmless interpolation occurs. Our results recover prior
independent-features results (with a much simpler analysis), but they
furthermore show that harmless interpolation can occur in more general settings
such as features that are a bounded orthonormal system. Furthermore, our
results show an asymptotic separation between classification and regression
performance in a manner that was previously only shown for Gaussian features.

    

### [[2111.05199] Deep diffusion-based forecasting of COVID-19 by incorporating network-level mobility information](http://arxiv.org/abs/2111.05199)


  Modeling the spatiotemporal nature of the spread of infectious diseases can
provide useful intuition in understanding the time-varying aspect of the
disease spread and the underlying complex spatial dependency observed in
people's mobility patterns. Besides, the county level multiple related time
series information can be leveraged to make a forecast on an individual time
series. Adding to this challenge is the fact that real-time data often deviates
from the unimodal Gaussian distribution assumption and may show some complex
mixed patterns. Motivated by this, we develop a deep learning-based time-series
model for probabilistic forecasting called Auto-regressive Mixed Density
Dynamic Diffusion Network(ARM3Dnet), which considers both people's mobility and
disease spread as a diffusion process on a dynamic directed graph. The Gaussian
Mixture Model layer is implemented to consider the multimodal nature of the
real-time data while learning from multiple related time series. We show that
our model, when trained with the best combination of dynamic covariate features
and mixture components, can outperform both traditional statistical and deep
learning models in forecasting the number of Covid-19 deaths and cases at the
county level in the United States.

    

### [[2111.05204] Reason first, then respond: Modular Generation for Knowledge-infused Dialogue](http://arxiv.org/abs/2111.05204)


  Large language models can produce fluent dialogue but often hallucinate
factual inaccuracies. While retrieval-augmented models help alleviate this
issue, they still face a difficult challenge of both reasoning to provide
correct knowledge and generating conversation simultaneously. In this work, we
propose a modular model, Knowledge to Response (K2R), for incorporating
knowledge into conversational agents, which breaks down this problem into two
easier steps. K2R first generates a knowledge sequence, given a dialogue
context, as an intermediate step. After this "reasoning step", the model then
attends to its own generated knowledge sequence, as well as the dialogue
context, to produce a final response. In detailed experiments, we find that
such a model hallucinates less in knowledge-grounded dialogue tasks, and has
advantages in terms of interpretability and modularity. In particular, it can
be used to fuse QA and dialogue systems together to enable dialogue agents to
give knowledgeable answers, or QA models to give conversational responses in a
zero-shot setting.

    

### [[2111.05214] A Topological Data Analysis Based Classifier](http://arxiv.org/abs/2111.05214)


  Topological Data Analysis (TDA) is an emergent field that aims to discover
topological information hidden in a dataset. TDA tools have been commonly used
to create filters and topological descriptors to improve Machine Learning (ML)
methods. This paper proposes an algorithm that applies TDA directly to
multi-class classification problems, without any further ML stage, showing
advantages for imbalanced datasets. The proposed algorithm builds a filtered
simplicial complex on the dataset. Persistent Homology (PH) is applied to guide
the selection of a sub-complex where unlabeled points obtain the label with the
majority of votes from labeled neighboring points. We select 8 datasets with
different dimensions, degrees of class overlap and imbalanced samples per
class. On average, the proposed TDABC method was better than KNN and
weighted-KNN. It behaves competitively with Local SVM and Random Forest
baseline classifiers in balanced datasets, and it outperforms all baseline
methods classifying entangled and minority classes.

    

### [[2111.05218] A research framework for writing differentiable PDE discretizations in JAX](http://arxiv.org/abs/2111.05218)


  Differentiable simulators are an emerging concept with applications in
several fields, from reinforcement learning to optimal control. Their
distinguishing feature is the ability to calculate analytic gradients with
respect to the input parameters. Like neural networks, which are constructed by
composing several building blocks called layers, a simulation often requires
computing the output of an operator that can itself be decomposed into
elementary units chained together. While each layer of a neural network
represents a specific discrete operation, the same operator can have multiple
representations, depending on the discretization employed and the research
question that needs to be addressed. Here, we propose a simple design pattern
to construct a library of differentiable operators and discretizations, by
representing operators as mappings between families of continuous functions,
parametrized by finite vectors. We demonstrate the approach on an acoustic
optimization problem, where the Helmholtz equation is discretized using Fourier
spectral methods, and differentiability is demonstrated using gradient descent
to optimize the speed of sound of an acoustic lens. The proposed framework is
open-sourced and available at \url{this https URL}

    

### [[2111.05231] MLHarness: A Scalable Benchmarking System for MLCommons](http://arxiv.org/abs/2111.05231)


  With the society's growing adoption of machine learning (ML) and deep
learning (DL) for various intelligent solutions, it becomes increasingly
imperative to standardize a common set of measures for ML/DL models with large
scale open datasets under common development practices and resources so that
people can benchmark and compare models quality and performance on a common
ground. MLCommons has emerged recently as a driving force from both industry
and academia to orchestrate such an effort. Despite its wide adoption as
standardized benchmarks, MLCommons Inference has only included a limited number
of ML/DL models (in fact seven models in total). This significantly limits the
generality of MLCommons Inference's benchmarking results because there are many
more novel ML/DL models from the research community, solving a wide range of
problems with different inputs and outputs modalities. To address such a
limitation, we propose MLHarness, a scalable benchmarking harness system for
MLCommons Inference with three distinctive features: (1) it codifies the
standard benchmark process as defined by MLCommons Inference including the
models, datasets, DL frameworks, and software and hardware systems; (2) it
provides an easy and declarative approach for model developers to contribute
their models and datasets to MLCommons Inference; and (3) it includes the
support of a wide range of models with varying inputs/outputs modalities so
that we can scalably benchmark these models across different datasets,
frameworks, and hardware systems. This harness system is developed on top of
the MLModelScope system, and will be open sourced to the community. Our
experimental results demonstrate the superior flexibility and scalability of
this harness system for MLCommons Inference benchmarking.

    

### [[2111.05232] Learning Rates for Nonconvex Pairwise Learning](http://arxiv.org/abs/2111.05232)


  Pairwise learning is receiving increasing attention since it covers many
important machine learning tasks, e.g., metric learning, AUC maximization, and
ranking. Investigating the generalization behavior of pairwise learning is thus
of significance. However, existing generalization analysis mainly focuses on
the convex objective functions, leaving the nonconvex learning far less
explored. Moreover, the current learning rates derived for generalization
performance of pairwise learning are mostly of slower order. Motivated by these
problems, we study the generalization performance of nonconvex pairwise
learning and provide improved learning rates. Specifically, we develop
different uniform convergence of gradients for pairwise learning under
different assumptions, based on which we analyze empirical risk minimizer,
gradient descent, and stochastic gradient descent pairwise learning. We first
successfully establish learning rates for these algorithms in a general
nonconvex setting, where the analysis sheds insights on the trade-off between
optimization and generalization and the role of early-stopping. We then
investigate the generalization performance of nonconvex learning with a
gradient dominance curvature condition. In this setting, we derive faster
learning rates of order $\mathcal{O}(1/n)$, where $n$ is the sample size.
Provided that the optimal population risk is small, we further improve the
learning rates to $\mathcal{O}(1/n^2)$, which, to the best of our knowledge,
are the first $\mathcal{O}(1/n^2)$-type of rates for pairwise learning, no
matter of convex or nonconvex learning. Overall, we systematically analyzed the
generalization performance of nonconvex pairwise learning.

    

### [[2111.05251] Learning Perceptual Concepts by Bootstrapping from Human Queries](http://arxiv.org/abs/2111.05251)


  Robots need to be able to learn concepts from their users in order to adapt
their capabilities to each user's unique task. But when the robot operates on
high-dimensional inputs, like images or point clouds, this is impractical: the
robot needs an unrealistic amount of human effort to learn the new concept. To
address this challenge, we propose a new approach whereby the robot learns a
low-dimensional variant of the concept and uses it to generate a larger data
set for learning the concept in the high-dimensional space. This lets it take
advantage of semantically meaningful privileged information only accessible at
training time, like object poses and bounding boxes, that allows for richer
human interaction to speed up learning. We evaluate our approach by learning
prepositional concepts that describe object state or multi-object
relationships, like above, near, or aligned, which are key to user
specification of task goals and execution constraints for robots. Using a
simulated human, we show that our approach improves sample complexity when
compared to learning concepts directly in the high-dimensional space. We also
demonstrate the utility of the learned concepts in motion planning tasks on a
7-DoF Franka Panda robot.

    

### [[2111.05257] Logarithmic Regret from Sublinear Hints](http://arxiv.org/abs/2111.05257)


  We consider the online linear optimization problem, where at every step the
algorithm plays a point $x_t$ in the unit ball, and suffers loss $\langle c_t,
x_t\rangle$ for some cost vector $c_t$ that is then revealed to the algorithm.
Recent work showed that if an algorithm receives a hint $h_t$ that has
non-trivial correlation with $c_t$ before it plays $x_t$, then it can achieve a
regret guarantee of $O(\log T)$, improving on the bound of $\Theta(\sqrt{T})$
in the standard setting. In this work, we study the question of whether an
algorithm really requires a hint at every time step. Somewhat surprisingly, we
show that an algorithm can obtain $O(\log T)$ regret with just $O(\sqrt{T})$
hints under a natural query model; in contrast, we also show that $o(\sqrt{T})$
hints cannot guarantee better than $\Omega(\sqrt{T})$ regret. We give two
applications of our result, to the well-studied setting of optimistic regret
bounds and to the problem of online learning with abstention.

    

### [[2111.05264] Unsupervised Learning for Identifying High Eigenvector Centrality Nodes: A Graph Neural Network Approach](http://arxiv.org/abs/2111.05264)


  The existing methods to calculate the Eigenvector Centrality(EC) tend to not
be robust enough for determination of EC in low time complexity or not
well-scalable for large networks, hence rendering them practically unreliable/
computationally expensive. So, it is of the essence to develop a method that is
scalable in low computational time. Hence, we propose a deep learning model for
the identification of nodes with high Eigenvector Centrality. There have been a
few previous works in identifying the high ranked nodes with supervised
learning methods, but in real-world cases, the graphs are not labelled and
hence deployment of supervised learning methods becomes a hazard and its usage
becomes impractical. So, we devise CUL(Centrality with Unsupervised Learning)
method to learn the relative EC scores in a network in an unsupervised manner.
To achieve this, we develop an Encoder-Decoder based framework that maps the
nodes to their respective estimated EC scores. Extensive experiments were
conducted on different synthetic and real-world networks. We compared CUL
against a baseline supervised method for EC estimation similar to some of the
past works. It was observed that even with training on a minuscule number of
training datasets, CUL delivers a relatively better accuracy score when
identifying the higher ranked nodes than its supervised counterpart. We also
show that CUL is much faster and has a smaller runtime than the conventional
baseline method for EC computation. The code is available at
this https URL.

    

### [[2111.05265] High-order joint embedding for multi-level link prediction](http://arxiv.org/abs/2111.05265)


  Link prediction infers potential links from observed networks, and is one of
the essential problems in network analyses. In contrast to traditional graph
representation modeling which only predicts two-way pairwise relations, we
propose a novel tensor-based joint network embedding approach on simultaneously
encoding pairwise links and hyperlinks onto a latent space, which captures the
dependency between pairwise and multi-way links in inferring potential
unobserved hyperlinks. The major advantage of the proposed embedding procedure
is that it incorporates both the pairwise relationships and subgroup-wise
structure among nodes to capture richer network information. In addition, the
proposed method introduces a hierarchical dependency among links to infer
potential hyperlinks, and leads to better link prediction. In theory we
establish the estimation consistency for the proposed embedding approach, and
provide a faster convergence rate compared to link prediction utilizing
pairwise links or hyperlinks only. Numerical studies on both simulation
settings and Facebook ego-networks indicate that the proposed method improves
both hyperlink and pairwise link prediction accuracy compared to existing link
prediction algorithms.

    

### [[2111.05267] Community detection using low-dimensional network embedding algorithms](http://arxiv.org/abs/2111.05267)


  With the increasing relevance of large networks in important areas such as
the study of contact networks for spread of disease, or social networks for
their impact on geopolitics, it has become necessary to study machine learning
tools that are scalable to very large networks, often containing millions of
nodes. One major class of such scalable algorithms is known as network
representation learning or network embedding. These algorithms try to learn
representations of network functionals (e.g.~nodes) by first running multiple
random walks and then using the number of co-occurrences of each pair of nodes
in observed random walk segments to obtain a low-dimensional representation of
nodes on some Euclidean space. The aim of this paper is to rigorously
understand the performance of two major algorithms, DeepWalk and node2vec, in
recovering communities for canonical network models with ground truth
communities. Depending on the sparsity of the graph, we find the length of the
random walk segments required such that the corresponding observed
co-occurrence window is able to perform almost exact recovery of the underlying
community assignments. We prove that, given some fixed co-occurrence window,
node2vec using random walks with a low non-backtracking probability can succeed
for much sparser networks compared to DeepWalk using simple random walks.
Moreover, if the sparsity parameter is low, we provide evidence that these
algorithms might not succeed in almost exact recovery. The analysis requires
developing general tools for path counting on random networks having an
underlying low-rank structure, which are of independent interest.

    

### [[2111.05271] Stress field prediction in fiber-reinforced composite materials using a deep learning approach](http://arxiv.org/abs/2111.05271)


  Computational stress analysis is an important step in the design of material
systems. Finite element method (FEM) is a standard approach of performing
stress analysis of complex material systems. A way to accelerate stress
analysis is to replace FEM with a data-driven machine learning based stress
analysis approach. In this study, we consider a fiber-reinforced matrix
composite material system and we use deep learning tools to find an alternative
to the FEM approach for stress field prediction. We first try to predict stress
field maps for composite material systems of fixed number of fibers with
varying spatial configurations. Specifically, we try to find a mapping between
the spatial arrangement of the fibers in the composite material and the
corresponding von Mises stress field. This is achieved by using a convolutional
neural network (CNN), specifically a U-Net architecture, using true stress maps
of systems with same number of fibers as training data. U-Net is a
encoder-decoder network which in this study takes in the composite material
image as an input and outputs the stress field image which is of the same size
as the input image. We perform a robustness analysis by taking different
initializations of the training samples to find the sensitivity of the
prediction accuracy to the small number of training samples. When the number of
fibers in the composite material system is increased for the same volume
fraction, a finer finite element mesh discretization is required to represent
the geometry accurately. This leads to an increase in the computational cost.
Thus, the secondary goal here is to predict the stress field for systems with
larger number of fibers with varying spatial configurations using information
from the true stress maps of relatively cheaper systems of smaller fiber
number.

    

### [[2111.05275] Towards a Unified Information-Theoretic Framework for Generalization](http://arxiv.org/abs/2111.05275)


  In this work, we investigate the expressiveness of the "conditional mutual
information" (CMI) framework of Steinke and Zakynthinou (2020) and the prospect
of using it to provide a unified framework for proving generalization bounds in
the realizable setting. We first demonstrate that one can use this framework to
express non-trivial (but sub-optimal) bounds for any learning algorithm that
outputs hypotheses from a class of bounded VC dimension. We prove that the CMI
framework yields the optimal bound on the expected risk of Support Vector
Machines (SVMs) for learning halfspaces. This result is an application of our
general result showing that stable compression schemes Bousquet al. (2020) of
size $k$ have uniformly bounded CMI of order $O(k)$. We further show that an
inherent limitation of proper learning of VC classes contradicts the existence
of a proper learner with constant CMI, and it implies a negative resolution to
an open problem of Steinke and Zakynthinou (2020). We further study the CMI of
empirical risk minimizers (ERMs) of class $H$ and show that it is possible to
output all consistent classifiers (version space) with bounded CMI if and only
if $H$ has a bounded star number (Hanneke and Yang (2015)). Moreover, we prove
a general reduction showing that "leave-one-out" analysis is expressible via
the CMI framework. As a corollary we investigate the CMI of the
one-inclusion-graph algorithm proposed by Haussler et al. (1994). More
generally, we show that the CMI framework is universal in the sense that for
every consistent algorithm and data distribution, the expected risk vanishes as
the number of samples diverges if and only if its evaluated CMI has sublinear
growth with the number of samples.

    

### [[2111.05277] Generalized Kernel Ridge Regression for Causal Inference with Missing-at-Random Sample Selection](http://arxiv.org/abs/2111.05277)


  I propose kernel ridge regression estimators for nonparametric dose response
curves and semiparametric treatment effects in the setting where an analyst has
access to a selected sample rather than a random sample; only for select
observations, the outcome is observed. I assume selection is as good as random
conditional on treatment and a sufficiently rich set of observed covariates,
where the covariates are allowed to cause treatment or be caused by treatment
-- an extension of missingness-at-random (MAR). I propose estimators of means,
increments, and distributions of counterfactual outcomes with closed form
solutions in terms of kernel matrix operations, allowing treatment and
covariates to be discrete or continuous, and low, high, or infinite
dimensional. For the continuous treatment case, I prove uniform consistency
with finite sample rates. For the discrete treatment case, I prove root-n
consistency, Gaussian approximation, and semiparametric efficiency.

    

### [[2111.05283] Unsupervised Spiking Instance Segmentation on Event Data using STDP](http://arxiv.org/abs/2111.05283)


  Spiking Neural Networks (SNN) and the field of Neuromorphic Engineering has
brought about a paradigm shift in how to approach Machine Learning (ML) and
Computer Vision (CV) problem. This paradigm shift comes from the adaption of
event-based sensing and processing. An event-based vision sensor allows for
sparse and asynchronous events to be produced that are dynamically related to
the scene. Allowing not only the spatial information but a high-fidelity of
temporal information to be captured. Meanwhile avoiding the extra overhead and
redundancy of conventional high frame rate approaches. However, with this
change in paradigm, many techniques from traditional CV and ML are not
applicable to these event-based spatial-temporal visual streams. As such a
limited number of recognition, detection and segmentation approaches exist. In
this paper, we present a novel approach that can perform instance segmentation
using just the weights of a Spike Time Dependent Plasticity trained Spiking
Convolutional Neural Network that was trained for object recognition. This
exploits the spatial and temporal aspects of the network's internal feature
representations adding this new discriminative capability. We highlight the new
capability by successfully transforming a single class unsupervised network for
face detection into a multi-person face recognition and instance segmentation
network.

    

### [[2111.05292] Generalization in quantum machine learning from few training data](http://arxiv.org/abs/2111.05292)


  Modern quantum machine learning (QML) methods involve variationally
optimizing a parameterized quantum circuit on a training data set, and
subsequently making predictions on a testing data set (i.e., generalizing). In
this work, we provide a comprehensive study of generalization performance in
QML after training on a limited number $N$ of training data points. We show
that the generalization error of a quantum machine learning model with $T$
trainable gates scales at worst as $\sqrt{T/N}$. When only $K \ll T$ gates have
undergone substantial change in the optimization process, we prove that the
generalization error improves to $\sqrt{K / N}$. Our results imply that the
compiling of unitaries into a polynomial number of native gates, a crucial
application for the quantum computing industry that typically uses
exponential-size training data, can be sped up significantly. We also show that
classification of quantum states across a phase transition with a quantum
convolutional neural network requires only a very small training data set.
Other potential applications include learning quantum error correcting codes or
quantum dynamical simulation. Our work injects new hope into the field of QML,
as good generalization is guaranteed from few training data.

    

### [[2111.05297] Sliced Recursive Transformer](http://arxiv.org/abs/2111.05297)


  We present a neat yet effective recursive operation on vision transformers
that can improve parameter utilization without involving additional parameters.
This is achieved by sharing weights across depth of transformer networks. The
proposed method can obtain a substantial gain (~2%) simply using naïve
recursive operation, requires no special or sophisticated knowledge for
designing principles of networks, and introduces minimum computational overhead
to the training procedure. To reduce the additional computation caused by
recursive operation while maintaining the superior accuracy, we propose an
approximating method through multiple sliced group self-attentions across
recursive layers which can reduce the cost consumption by 10~30% with minimal
performance loss. We call our model Sliced Recursive Transformer (SReT), which
is compatible with a broad range of other designs for efficient vision
transformers. Our best model establishes significant improvement on ImageNet
over state-of-the-art methods while containing fewer parameters. The proposed
sliced recursive operation allows us to build a transformer with more than 100
or even 1000 layers effortlessly under a still small size (13~15M), to avoid
difficulties in optimization when the model size is too large. The flexible
scalability has shown great potential for scaling up and constructing extremely
deep and large dimensionality vision transformers. Our code and models are
available at this https URL.

    

### [[2111.05299] Can Information Flows Suggest Targets for Interventions in Neural Circuits?](http://arxiv.org/abs/2111.05299)


  Motivated by neuroscientific and clinical applications, we empirically
examine whether observational measures of information flow can suggest
interventions. We do so by performing experiments on artificial neural networks
in the context of fairness in machine learning, where the goal is to induce
fairness in the system through interventions. Using our recently developed
$M$-information flow framework, we measure the flow of information about the
true label (responsible for accuracy, and hence desirable), and separately, the
flow of information about a protected attribute (responsible for bias, and
hence undesirable) on the edges of a trained neural network. We then compare
the flow magnitudes against the effect of intervening on those edges by
pruning. We show that pruning edges that carry larger information flows about
the protected attribute reduces bias at the output to a greater extent. This
demonstrates that $M$-information flow can meaningfully suggest targets for
interventions, answering the title's question in the affirmative. We also
evaluate bias-accuracy tradeoffs for different intervention strategies, to
analyze how one might use estimates of desirable and undesirable information
flows (here, accuracy and bias flows) to inform interventions that preserve the
former while reducing the latter.

    

### [[2111.05300] Double Control Variates for Gradient Estimation in Discrete Latent Variable Models](http://arxiv.org/abs/2111.05300)


  Stochastic gradient-based optimisation for discrete latent variable models is
challenging due to the high variance of gradients. We introduce a variance
reduction technique for score function estimators that makes use of double
control variates. These control variates act on top of a main control variate,
and try to further reduce the variance of the overall estimator. We develop a
double control variate for the REINFORCE leave-one-out estimator using Taylor
expansions. For training discrete latent variable models, such as variational
autoencoders with binary latent variables, our approach adds no extra
computational cost compared to standard training with the REINFORCE
leave-one-out estimator. We apply our method to challenging high-dimensional
toy examples and training variational autoencoders with binary latent
variables. We show that our estimator can have lower variance compared to other
state-of-the-art estimators.

    

### [[2111.05303] Identifying the atmospheric drivers of drought and heat using a smoothed deep learning approach](http://arxiv.org/abs/2111.05303)


  Europe was hit by several, disastrous heat and drought events in recent
summers. Besides thermodynamic influences, such hot and dry extremes are driven
by certain atmospheric situations including anticyclonic conditions. Effects of
climate change on atmospheric circulations are complex and many open research
questions remain in this context, e.g., on future trends of anticyclonic
conditions. Based on the combination of a catalog of labeled circulation
patterns and spatial atmospheric variables, we propose a smoothed convolutional
neural network classifier for six types of anticyclonic circulations that are
associated with drought and heat. Our work can help to identify important
drivers of hot and dry extremes in climate simulations, which allows to unveil
the impact of climate change on these drivers. We address various challenges
inherent to circulation pattern classification that are also present in other
climate patterns, e.g., subjective labels and unambiguous transition periods.

    

### [[2111.05307] Machine-learning custom-made basis functions for partial differential equations](http://arxiv.org/abs/2111.05307)


  Spectral methods are an important part of scientific computing's arsenal for
solving partial differential equations (PDEs). However, their applicability and
effectiveness depend crucially on the choice of basis functions used to expand
the solution of a PDE. The last decade has seen the emergence of deep learning
as a strong contender in providing efficient representations of complex
functions. In the current work, we present an approach for combining deep
neural networks with spectral methods to solve PDEs. In particular, we use a
deep learning technique known as the Deep Operator Network (DeepONet), to
identify candidate functions on which to expand the solution of PDEs. We have
devised an approach which uses the candidate functions provided by the DeepONet
as a starting point to construct a set of functions which have the following
properties: i) they constitute a basis, 2) they are orthonormal, and 3) they
are hierarchical i.e., akin to Fourier series or orthogonal polynomials. We
have exploited the favorable properties of our custom-made basis functions to
both study their approximation capability and use them to expand the solution
of linear and nonlinear time-dependent PDEs.

    

### [[2111.05311] Mode connectivity in the loss landscape of parameterized quantum circuits](http://arxiv.org/abs/2111.05311)


  Variational training of parameterized quantum circuits (PQCs) underpins many
workflows employed on near-term noisy intermediate scale quantum (NISQ)
devices. It is a hybrid quantum-classical approach that minimizes an associated
cost function in order to train a parameterized ansatz. In this paper we adapt
the qualitative loss landscape characterization for neural networks introduced
in \cite{goodfellow2014qualitatively,li2017visualizing} and tests for
connectivity used in \cite{draxler2018essentially} to study the loss landscape
features in PQC training. We present results for PQCs trained on a simple
regression task, using the bilayer circuit ansatz, which consists of
alternating layers of parameterized rotation gates and entangling gates.
Multiple circuits are trained with $3$ different batch gradient optimizers:
stochastic gradient descent, the quantum natural gradient, and Adam. We
identify large features in the landscape that can lead to faster convergence in
training workflows.

    

### [[2111.05321] Turing-Universal Learners with Optimal Scaling Laws](http://arxiv.org/abs/2111.05321)


  For a given distribution, learning algorithm, and performance metric, the
rate of convergence (or data-scaling law) is the asymptotic behavior of the
algorithm's test performance as a function of number of train samples. Many
learning methods in both theory and practice have power-law rates, i.e.
performance scales as $n^{-\alpha}$ for some $\alpha > 0$. Moreover, both
theoreticians and practitioners are concerned with improving the rates of their
learning algorithms under settings of interest. We observe the existence of a
"universal learner", which achieves the best possible distribution-dependent
asymptotic rate among all learning algorithms within a specified runtime (e.g.
$O(n^2)$), while incurring only polylogarithmic slowdown over this runtime.
This algorithm is uniform, and does not depend on the distribution, and yet
achieves best-possible rates for all distributions. The construction itself is
a simple extension of Levin's universal search (Levin, 1973). And much like
universal search, the universal learner is not at all practical, and is
primarily of theoretical and philosophical interest.

    

### [[2111.05323] Variational Multi-Task Learning with Gumbel-Softmax Priors](http://arxiv.org/abs/2111.05323)


  Multi-task learning aims to explore task relatedness to improve individual
tasks, which is of particular significance in the challenging scenario that
only limited data is available for each task. To tackle this challenge, we
propose variational multi-task learning (VMTL), a general probabilistic
inference framework for learning multiple related tasks. We cast multi-task
learning as a variational Bayesian inference problem, in which task relatedness
is explored in a unified manner by specifying priors. To incorporate shared
knowledge into each task, we design the prior of a task to be a learnable
mixture of the variational posteriors of other related tasks, which is learned
by the Gumbel-Softmax technique. In contrast to previous methods, our VMTL can
exploit task relatedness for both representations and classifiers in a
principled way by jointly inferring their posteriors. This enables individual
tasks to fully leverage inductive biases provided by related tasks, therefore
improving the overall performance of all tasks. Experimental results
demonstrate that the proposed VMTL is able to effectively tackle a variety of
challenging multi-task learning settings with limited training data for both
classification and regression. Our method consistently surpasses previous
methods, including strong Bayesian approaches, and achieves state-of-the-art
performance on five benchmark datasets.

    

### [[2111.05326] The Internet of Federated Things (IoFT): A Vision for the Future and In-depth Survey of Data-driven Approaches for Federated Learning](http://arxiv.org/abs/2111.05326)


  The Internet of Things (IoT) is on the verge of a major paradigm shift. In
the IoT system of the future, IoFT, the cloud will be substituted by the crowd
where model training is brought to the edge, allowing IoT devices to
collaboratively extract knowledge and build smart analytics/models while
keeping their personal data stored locally. This paradigm shift was set into
motion by the tremendous increase in computational power on IoT devices and the
recent advances in decentralized and privacy-preserving model training, coined
as federated learning (FL). This article provides a vision for IoFT and a
systematic overview of current efforts towards realizing this vision.
Specifically, we first introduce the defining characteristics of IoFT and
discuss FL data-driven approaches, opportunities, and challenges that allow
decentralized inference within three dimensions: (i) a global model that
maximizes utility across all IoT devices, (ii) a personalized model that
borrows strengths across all devices yet retains its own model, (iii) a
meta-learning model that quickly adapts to new devices or learning tasks. We
end by describing the vision and challenges of IoFT in reshaping different
industries through the lens of domain experts. Those industries include
manufacturing, transportation, energy, healthcare, quality & reliability,
business, and computing.

    

### [[2111.05328] Data Augmentation Can Improve Robustness](http://arxiv.org/abs/2111.05328)


  Adversarial training suffers from robust overfitting, a phenomenon where the
robust test accuracy starts to decrease during training. In this paper, we
focus on reducing robust overfitting by using common data augmentation schemes.
We demonstrate that, contrary to previous findings, when combined with model
weight averaging, data augmentation can significantly boost robust accuracy.
Furthermore, we compare various augmentations techniques and observe that
spatial composition techniques work the best for adversarial training. Finally,
we evaluate our approach on CIFAR-10 against $\ell_\infty$ and $\ell_2$
norm-bounded perturbations of size $\epsilon = 8/255$ and $\epsilon = 128/255$,
respectively. We show large absolute improvements of +2.93% and +2.16% in
robust accuracy compared to previous state-of-the-art methods. In particular,
against $\ell_\infty$ norm-bounded perturbations of size $\epsilon = 8/255$,
our model reaches 60.07% robust accuracy without using any external data. We
also achieve a significant performance boost with this approach while using
other architectures and datasets such as CIFAR-100, SVHN and TinyImageNet.

    

### [[1803.07859] Efficient Sampling and Structure Learning of Bayesian Networks](http://arxiv.org/abs/1803.07859)


  Bayesian networks are probabilistic graphical models widely employed to
understand dependencies in high dimensional data, and even to facilitate causal
discovery. Learning the underlying network structure, which is encoded as a
directed acyclic graph (DAG) is highly challenging mainly due to the vast
number of possible networks in combination with the acyclicity constraint.
Efforts have focussed on two fronts: constraint-based methods that perform
conditional independence tests to exclude edges and score and search approaches
which explore the DAG space with greedy or MCMC schemes. Here we synthesise
these two fields in a novel hybrid method which reduces the complexity of MCMC
approaches to that of a constraint-based method. Individual steps in the MCMC
scheme only require simple table lookups so that very long chains can be
efficiently obtained. Furthermore, the scheme includes an iterative procedure
to correct for errors from the conditional independence tests. The algorithm
offers markedly superior performance to alternatives, particularly because DAGs
can also be sampled from the posterior distribution, enabling full Bayesian
model averaging for much larger Bayesian networks.

    

### [[1908.00325] Estimating the standard error of cross-Validation-Based estimators of classifier performance](http://arxiv.org/abs/1908.00325)


  First, we analyze the variance of the Cross Validation (CV)-based estimators
used for estimating the performance of classification rules. Second, we propose
a novel estimator to estimate this variance using the Influence Function (IF)
approach that had been used previously very successfully to estimate the
variance of the bootstrap-based estimators. The motivation for this research is
that, as the best of our knowledge, the literature lacks a rigorous method for
estimating the variance of the CV-based estimators. What is available is a set
of ad-hoc procedures that have no mathematical foundation since they ignore the
covariance structure among dependent random variables. The conducted
experiments show that the IF proposed method has small RMS error with some
bias. However, surprisingly, the ad-hoc methods still work better than the
IF-based method. Unfortunately, this is due to the lack of enough smoothness if
compared to the bootstrap estimator. This opens the research for three points:
(1) more comprehensive simulation study to clarify when the IF method win or
loose; (2) more mathematical analysis to figure out why the ad-hoc methods work
well; and (3) more mathematical treatment to figure out the connection between
the appropriate amount of "smoothness" and decreasing the bias of the IF
method.

    

### [[1910.05513] On Robustness of Neural Ordinary Differential Equations](http://arxiv.org/abs/1910.05513)


  Neural ordinary differential equations (ODEs) have been attracting increasing
attention in various research domains recently. There have been some works
studying optimization issues and approximation capabilities of neural ODEs, but
their robustness is still yet unclear. In this work, we fill this important gap
by exploring robustness properties of neural ODEs both empirically and
theoretically. We first present an empirical study on the robustness of the
neural ODE-based networks (ODENets) by exposing them to inputs with various
types of perturbations and subsequently investigating the changes of the
corresponding outputs. In contrast to conventional convolutional neural
networks (CNNs), we find that the ODENets are more robust against both random
Gaussian perturbations and adversarial attack examples. We then provide an
insightful understanding of this phenomenon by exploiting a certain desirable
property of the flow of a continuous-time ODE, namely that integral curves are
non-intersecting. Our work suggests that, due to their intrinsic robustness, it
is promising to use neural ODEs as a basic block for building robust deep
network models. To further enhance the robustness of vanilla neural ODEs, we
propose the time-invariant steady neural ODE (TisODE), which regularizes the
flow on perturbed data via the time-invariant property and the imposition of a
steady-state constraint. We show that the TisODE method outperforms vanilla
neural ODEs and also can work in conjunction with other state-of-the-art
architectural methods to build more robust deep networks.
\url{this https URL}

    

### [[2001.04294] Mean-Field and Kinetic Descriptions of Neural Differential Equations](http://arxiv.org/abs/2001.04294)


  Nowadays, neural networks are widely used in many applications as artificial
intelligence models for learning tasks. Since typically neural networks process
a very large amount of data, it is convenient to formulate them within the
mean-field and kinetic theory. In this work we focus on a particular class of
neural networks, i.e. the residual neural networks, assuming that each layer is
characterized by the same number of neurons $N$, which is fixed by the
dimension of the data. This assumption allows to interpret the residual neural
network as a time-discretized ordinary differential equation, in analogy with
neural differential equations. The mean-field description is then obtained in
the limit of infinitely many input data. This leads to a Vlasov-type partial
differential equation which describes the evolution of the distribution of the
input data. We analyze steady states and sensitivity with respect to the
parameters of the network, namely the weights and the bias. In the simple
setting of a linear activation function and one-dimensional input data, the
study of the moments provides insights on the choice of the parameters of the
network. Furthermore, a modification of the microscopic dynamics, inspired by
stochastic residual neural networks, leads to a Fokker-Planck formulation of
the network, in which the concept of network training is replaced by the task
of fitting distributions. The performed analysis is validated by artificial
numerical simulations. In particular, results on classification and regression
problems are presented.

    

### [[2006.06098] Dynamical mean-field theory for stochastic gradient descent in Gaussian mixture classification](http://arxiv.org/abs/2006.06098)


  We analyze in a closed form the learning dynamics of stochastic gradient
descent (SGD) for a single-layer neural network classifying a high-dimensional
Gaussian mixture where each cluster is assigned one of two labels. This problem
provides a prototype of a non-convex loss landscape with interpolating regimes
and a large generalization gap. We define a particular stochastic process for
which SGD can be extended to a continuous-time limit that we call stochastic
gradient flow. In the full-batch limit, we recover the standard gradient flow.
We apply dynamical mean-field theory from statistical physics to track the
dynamics of the algorithm in the high-dimensional limit via a self-consistent
stochastic process. We explore the performance of the algorithm as a function
of the control parameters shedding light on how it navigates the loss
landscape.

    

### [[2006.07869] Benchmarking Multi-Agent Deep Reinforcement Learning Algorithms in Cooperative Tasks](http://arxiv.org/abs/2006.07869)


  Multi-agent deep reinforcement learning (MARL) suffers from a lack of
commonly-used evaluation tasks and criteria, making comparisons between
approaches difficult. In this work, we provide a systematic evaluation and
comparison of three different classes of MARL algorithms (independent learning,
centralised multi-agent policy gradient, value decomposition) in a diverse
range of cooperative multi-agent learning tasks. Our experiments serve as a
reference for the expected performance of algorithms across different learning
tasks, and we provide insights regarding the effectiveness of different
learning approaches. We open-source EPyMARL, which extends the PyMARL codebase
to include additional algorithms and allow for flexible configuration of
algorithm implementation details such as parameter sharing. Finally, we
open-source two environments for multi-agent research which focus on
coordination under sparse rewards.

    

### [[2006.09447] Agent Modelling under Partial Observability for Deep Reinforcement Learning](http://arxiv.org/abs/2006.09447)


  Modelling the behaviours of other agents is essential for understanding how
agents interact and making effective decisions. Existing methods for agent
modelling commonly assume knowledge of the local observations and chosen
actions of the modelled agents during execution. To eliminate this assumption,
we extract representations from the local information of the controlled agent
using encoder-decoder architectures. Using the observations and actions of the
modelled agents during training, our models learn to extract representations
about the modelled agents conditioned only on the local observations of the
controlled agent. The representations are used to augment the controlled
agent's decision policy which is trained via deep reinforcement learning; thus,
during execution, the policy does not require access to other agents'
information. We provide a comprehensive evaluation and ablations studies in
cooperative, competitive and mixed multi-agent environments, showing that our
method achieves higher returns than baseline methods which do not use the
learned representations.

    

### [[2009.04521] How Good is your Explanation? Algorithmic Stability Measures to Assess the Quality of Explanations for Deep Neural Networks](http://arxiv.org/abs/2009.04521)


  A plethora of methods have been proposed to explain how deep neural networks
reach their decisions but comparatively, little effort has been made to ensure
that the explanations produced by these methods are objectively relevant. While
several desirable properties for trustworthy explanations have been formulated,
objective measures have been harder to derive. Here, we propose two new
measures to evaluate explanations borrowed from the field of algorithmic
stability: mean generalizability MeGe and relative consistency ReCo. We conduct
extensive experiments on different network architectures, common explainability
methods, and several image datasets to demonstrate the benefits of the proposed
this http URL comparison to ours, popular fidelity measures are not sufficient to
guarantee trustworthy explanations.Finally, we found that 1-Lipschitz networks
produce explanations with higher MeGe and ReCo than common neural networks
while reaching similar accuracy. This suggests that 1-Lipschitz networks are a
relevant direction towards predictors that are more explainable and
trustworthy.

    

### [[2010.08729] Ensemble Kalman Variational Objectives: Nonlinear Latent Trajectory Inference with A Hybrid of Variational Inference and Ensemble Kalman Filter](http://arxiv.org/abs/2010.08729)


  Variational inference (VI) combined with Bayesian nonlinear filtering
produces state-of-the-art results for latent time-series modeling. A body of
recent work has focused on sequential Monte Carlo (SMC) and its variants, e.g.,
forward filtering backward simulation (FFBSi). Although these studies have
succeeded, serious problems remain in particle degeneracy and biased gradient
estimators. In this paper, we propose Ensemble Kalman Variational Objective
(EnKO), a hybrid method of VI and the ensemble Kalman filter (EnKF), to infer
state space models (SSMs). Our proposed method can efficiently identify latent
dynamics because of its particle diversity and unbiased gradient estimators. We
demonstrate that our EnKO outperforms SMC-based methods in terms of predictive
ability and particle efficiency for three benchmark nonlinear system
identification tasks.

    

### [[2010.12163] Improved Worst-Case Regret Bounds for Randomized Least-Squares Value Iteration](http://arxiv.org/abs/2010.12163)


  This paper studies regret minimization with randomized value functions in
reinforcement learning. In tabular finite-horizon Markov Decision Processes, we
introduce a clipping variant of one classical Thompson Sampling (TS)-like
algorithm, randomized least-squares value iteration (RLSVI). Our
$\tilde{\mathrm{O}}(H^2S\sqrt{AT})$ high-probability worst-case regret bound
improves the previous sharpest worst-case regret bounds for RLSVI and matches
the existing state-of-the-art worst-case TS-based regret bounds.

    

### [[2010.14689] Bayesian Deep Learning via Subnetwork Inference](http://arxiv.org/abs/2010.14689)


  The Bayesian paradigm has the potential to solve core issues of deep neural
networks such as poor calibration and data inefficiency. Alas, scaling Bayesian
inference to large weight spaces often requires restrictive approximations. In
this work, we show that it suffices to perform inference over a small subset of
model weights in order to obtain accurate predictive posteriors. The other
weights are kept as point estimates. This subnetwork inference framework
enables us to use expressive, otherwise intractable, posterior approximations
over such subsets. In particular, we implement subnetwork linearized Laplace as
a simple, scalable Bayesian deep learning method: We first obtain a MAP
estimate of all weights and then infer a full-covariance Gaussian posterior
over a subnetwork using the linearized Laplace approximation. We propose a
subnetwork selection strategy that aims to maximally preserve the model's
predictive uncertainty. Empirically, our approach compares favorably to
ensembles and less expressive posterior approximations over full networks.

    

### [[2011.03303] Deep coastal sea elements forecasting using U-Net based models](http://arxiv.org/abs/2011.03303)


  The supply and demand of energy is influenced by meteorological conditions.
The relevance of accurate weather forecasts increases as the demand for
renewable energy sources increases. The energy providers and policy makers
require weather information to make informed choices and establish optimal
plans according to the operational objectives. Due to the recent development of
deep learning techniques applied to satellite imagery, weather forecasting that
uses remote sensing data has also been the subject of major progress. The
present paper investigates multiple steps ahead frame prediction for coastal
sea elements in the Netherlands using U-Net based architectures. Hourly data
from the Copernicus observation programme spanned over a period of 2 years has
been used to train the models and make the forecasting, including seasonal
predictions. We propose a variation of the U-Net architecture and further
extend this novel model using residual connections, parallel convolutions and
asymmetric convolutions in order to introduce three additional architectures.
In particular, we show that the architecture equipped with parallel and
asymmetric convolutions as well as skip connections outperforms the other three
discussed models.

    

### [[2011.04218] Automorphic Equivalence-aware Graph Neural Network](http://arxiv.org/abs/2011.04218)


  Distinguishing the automorphic equivalence of nodes in a graph plays an
essential role in many scientific domains, e.g., computational biologist and
social network analysis. However, existing graph neural networks (GNNs) fail to
capture such an important property. To make GNN aware of automorphic
equivalence, we first introduce a localized variant of this concept --
ego-centered automorphic equivalence (Ego-AE). Then, we design a novel variant
of GNN, i.e., GRAPE, that uses learnable AE-aware aggregators to explicitly
differentiate the Ego-AE of each node's neighbors with the aids of various
subgraph templates. While the design of subgraph templates can be hard, we
further propose a genetic algorithm to automatically search them from graph
data. Moreover, we theoretically prove that GRAPE is expressive in terms of
generating distinct representations for nodes with different Ego-AE features,
which fills in a fundamental gap of existing GNN variants. Finally, we
empirically validate our model on eight real-world graph data, including social
network, e-commerce co-purchase network, and citation network, and show that it
consistently outperforms existing GNNs. The source code is public available at
this https URL.

    

### [[2011.04315] Coupled regularized sample covariance matrix estimator for multiple classes](http://arxiv.org/abs/2011.04315)


  The estimation of covariance matrices of multiple classes with limited
training data is a difficult problem. The sample covariance matrix (SCM) is
known to perform poorly when the number of variables is large compared to the
available number of samples. In order to reduce the mean squared error (MSE) of
the SCM, regularized (shrinkage) SCM estimators are often used. In this work,
we consider regularized SCM (RSCM) estimators for multiclass problems that
couple together two different target matrices for regularization: the pooled
(average) SCM of the classes and the scaled identity matrix. Regularization
toward the pooled SCM is beneficial when the population covariances are
similar, whereas regularization toward the identity matrix guarantees that the
estimators are positive definite. We derive the MSE optimal tuning parameters
for the estimators as well as propose a method for their estimation under the
assumption that the class populations follow (unspecified) elliptical
distributions with finite fourth-order moments. The MSE performance of the
proposed coupled RSCMs are evaluated with simulations and in a regularized
discriminant analysis (RDA) classification set-up on real data. The results
based on three different real data sets indicate comparable performance to
cross-validation but with a significant speed-up in computation time.

    

### [[2012.00113] The FEDHC Bayesian network learning algorithm](http://arxiv.org/abs/2012.00113)


  The paper proposes a new hybrid Bayesian network learning algorithm, termed
Forward Early Dropping Hill Climbing (FEDHC), devised to work with either
continuous or categorical variables. Specifically for the case of continuous
data, a robust to outliers version of FEDHC, that can be adopted by other BN
learning algorithms, is proposed. Further, the paper manifests that the only
implementation of MMHC in the statistical software \textit{R}, is prohibitively
expensive and a new implementation is offered. The FEDHC is tested via Monte
Carlo simulations that distinctly show it is computationally efficient, and
produces Bayesian networks of similar to, or of higher accuracy than MMHC and
PCHC. Finally, an application of FEDHC, PCHC and MMHC algorithms to real data,
from the field of economics, is demonstrated using the statistical software
\textit{R}.

    

### [[2012.02276] A feedforward neural network for modelling of average pressure frequency response](http://arxiv.org/abs/2012.02276)


  The Helmholtz equation has been used for modelling the sound pressure field
under a harmonic load. Computing harmonic sound pressure fields by means of
solving Helmholtz equation can quickly become unfeasible if one wants to study
many different geometries for ranges of frequencies. We propose a machine
learning approach, namely a feedforward dense neural network, for computing the
average sound pressure over a frequency range. The data is generated with
finite elements, by numerically computing the response of the average sound
pressure, by an eigenmode decomposition of the pressure. We analyze the
accuracy of the approximation and determine how much training data is needed in
order to reach a certain accuracy in the predictions of the average pressure
response.

    

### [[2012.04035] ATOM3D: Tasks On Molecules in Three Dimensions](http://arxiv.org/abs/2012.04035)


  Computational methods that operate on three-dimensional molecular structure
have the potential to solve important questions in biology and chemistry. In
particular, deep neural networks have gained significant attention, but their
widespread adoption in the biomolecular domain has been limited by a lack of
either systematic performance benchmarks or a unified toolkit for interacting
with molecular data. To address this, we present ATOM3D, a collection of both
novel and existing benchmark datasets spanning several key classes of
biomolecules. We implement several classes of three-dimensional molecular
learning methods for each of these tasks and show that they consistently
improve performance relative to methods based on one- and two-dimensional
representations. The specific choice of architecture proves to be critical for
performance, with three-dimensional convolutional networks excelling at tasks
involving complex geometries, graph networks performing well on systems
requiring detailed positional information, and the more recently developed
equivariant networks showing significant promise. Our results indicate that
many molecular problems stand to gain from three-dimensional molecular
learning, and that there is potential for improvement on many tasks which
remain underexplored. To lower the barrier to entry and facilitate further
developments in the field, we also provide a comprehensive suite of tools for
dataset processing, model training, and evaluation in our open-source atom3d
Python package. All datasets are available for download from
this https URL .

    

### [[2012.05326] Privacy Amplification by Decentralization](http://arxiv.org/abs/2012.05326)


  Analyzing data owned by several parties while achieving a good trade-off
between utility and privacy is a key challenge in federated learning and
analytics. In this work, we introduce a novel relaxation of local differential
privacy (LDP) that naturally arises in fully decentralized algorithms, i.e.,
when participants exchange information by communicating along the edges of a
network graph without central coordinator. This relaxation, that we call
network DP, captures the fact that users have only a local view of the system.
To show the relevance of network DP, we study a decentralized model of
computation where a token performs a walk on the network graph and is updated
sequentially by the party who receives it. For tasks such as real summation,
histogram computation and optimization with gradient descent, we propose simple
algorithms on ring and complete topologies. We prove that the privacy-utility
trade-offs of our algorithms under network DP significantly improve upon what
is achievable under LDP (sometimes even matching the utility of the trusted
curator model), showing for the first time that formal privacy gains can be
obtained from full decentralization. Our experiments illustrate the improved
utility of our approach for decentralized training with stochastic gradient
descent.

    

### [[2101.06296] Sensitivity Prewarping for Local Surrogate Modeling](http://arxiv.org/abs/2101.06296)


  In the continual effort to improve product quality and decrease operations
costs, computational modeling is increasingly being deployed to determine
feasibility of product designs or configurations. Surrogate modeling of these
computer experiments via local models, which induce sparsity by only
considering short range interactions, can tackle huge analyses of complicated
input-output relationships. However, narrowing focus to local scale means that
global trends must be re-learned over and over again. In this article, we
propose a framework for incorporating information from a global sensitivity
analysis into the surrogate model as an input rotation and rescaling
preprocessing step. We discuss the relationship between several sensitivity
analysis methods based on kernel regression before describing how they give
rise to a transformation of the input variables. Specifically, we perform an
input warping such that the "warped simulator" is equally sensitive to all
input directions, freeing local models to focus on local dynamics. Numerical
experiments on observational data and benchmark test functions, including a
high-dimensional computer simulator from the automotive industry, provide
empirical validation.

    

### [[2101.06855] GraphAttacker: A General Multi-Task GraphAttack Framework](http://arxiv.org/abs/2101.06855)


  Graph neural networks (GNNs) have been successfully exploited in graph
analysis tasks in many real-world applications. The competition between attack
and defense methods also enhances the robustness of GNNs. In this competition,
the development of adversarial training methods put forward higher requirement
for the diversity of attack examples. By contrast, most attack methods with
specific attack strategies are difficult to satisfy such a requirement. To
address this problem, we propose GraphAttacker, a novel generic graph attack
framework that can flexibly adjust the structures and the attack strategies
according to the graph analysis tasks. GraphAttacker generates adversarial
examples through alternate training on three key components: the multi-strategy
attack generator (MAG), the similarity discriminator (SD), and the attack
discriminator (AD), based on the generative adversarial network (GAN).
Furthermore, we introduce a novel similarity modification rate SMR to conduct a
stealthier attack considering the change of node similarity distribution.
Experiments on various benchmark datasets demonstrate that GraphAttacker can
achieve state-of-the-art attack performance on graph analysis tasks of node
classification, graph classification, and link prediction, no matter the
adversarial training is conducted or not. Moreover, we also analyze the unique
characteristics of each task and their specific response in the unified attack
framework. The project code is available at
this https URL.

    

### [[2102.05311] CIFS: Improving Adversarial Robustness of CNNs via Channel-wise Importance-based Feature Selection](http://arxiv.org/abs/2102.05311)


  We investigate the adversarial robustness of CNNs from the perspective of
channel-wise activations. By comparing \textit{non-robust} (normally trained)
and \textit{robustified} (adversarially trained) models, we observe that
adversarial training (AT) robustifies CNNs by aligning the channel-wise
activations of adversarial data with those of their natural counterparts.
However, the channels that are \textit{negatively-relevant} (NR) to predictions
are still over-activated when processing adversarial data. Besides, we also
observe that AT does not result in similar robustness for all classes. For the
robust classes, channels with larger activation magnitudes are usually more
\textit{positively-relevant} (PR) to predictions, but this alignment does not
hold for the non-robust classes. Given these observations, we hypothesize that
suppressing NR channels and aligning PR ones with their relevances further
enhances the robustness of CNNs under AT. To examine this hypothesis, we
introduce a novel mechanism, i.e., \underline{C}hannel-wise
\underline{I}mportance-based \underline{F}eature \underline{S}election (CIFS).
The CIFS manipulates channels' activations of certain layers by generating
non-negative multipliers to these channels based on their relevances to
predictions. Extensive experiments on benchmark datasets including CIFAR10 and
SVHN clearly verify the hypothesis and CIFS's effectiveness of robustifying
CNNs. \url{this https URL}

    

### [[2102.06477] HNPE: Leveraging Global Parameters for Neural Posterior Estimation](http://arxiv.org/abs/2102.06477)


  Inferring the parameters of a stochastic model based on experimental
observations is central to the scientific method. A particularly challenging
setting is when the model is strongly indeterminate, i.e. when distinct sets of
parameters yield identical observations. This arises in many practical
situations, such as when inferring the distance and power of a radio source (is
the source close and weak or far and strong?) or when estimating the amplifier
gain and underlying brain activity of an electrophysiological experiment. In
this work, we present hierarchical neural posterior estimation (HNPE), a novel
method for cracking such indeterminacy by exploiting additional information
conveyed by an auxiliary set of observations sharing global parameters. Our
method extends recent developments in simulation-based inference (SBI) based on
normalizing flows to Bayesian hierarchical models. We validate quantitatively
our proposal on a motivating example amenable to analytical solutions and then
apply it to invert a well known non-linear model from computational
neuroscience, using both simulated and real EEG data.

    

### [[2102.11938] Baby Intuitions Benchmark (BIB): Discerning the goals, preferences, and actions of others](http://arxiv.org/abs/2102.11938)


  To achieve human-like common sense about everyday life, machine learning
systems must understand and reason about the goals, preferences, and actions of
other agents in the environment. By the end of their first year of life, human
infants intuitively achieve such common sense, and these cognitive achievements
lay the foundation for humans' rich and complex understanding of the mental
states of others. Can machines achieve generalizable, commonsense reasoning
about other agents like human infants? The Baby Intuitions Benchmark (BIB)
challenges machines to predict the plausibility of an agent's behavior based on
the underlying causes of its actions. Because BIB's content and paradigm are
adopted from developmental cognitive science, BIB allows for direct comparison
between human and machine performance. Nevertheless, recently proposed,
deep-learning-based agency reasoning models fail to show infant-like reasoning,
leaving BIB an open challenge.

    

### [[2103.03874] Measuring Mathematical Problem Solving With the MATH Dataset](http://arxiv.org/abs/2103.03874)


  Many intellectual endeavors require mathematical problem solving, but this
skill remains beyond the capabilities of computers. To measure this ability in
machine learning models, we introduce MATH, a new dataset of 12,500 challenging
competition mathematics problems. Each problem in MATH has a full step-by-step
solution which can be used to teach models to generate answer derivations and
explanations. To facilitate future research and increase accuracy on MATH, we
also contribute a large auxiliary pretraining dataset which helps teach models
the fundamentals of mathematics. Even though we are able to increase accuracy
on MATH, our results show that accuracy remains relatively low, even with
enormous Transformer models. Moreover, we find that simply increasing budgets
and model parameter counts will be impractical for achieving strong
mathematical reasoning if scaling trends continue. While scaling Transformers
is automatically solving most other text-based tasks, scaling is not currently
solving MATH. To have more traction on mathematical problem solving we will
likely need new algorithmic advancements from the broader research community.

    

### [[2103.06268] CUAD: An Expert-Annotated NLP Dataset for Legal Contract Review](http://arxiv.org/abs/2103.06268)


  Many specialized domains remain untouched by deep learning, as large labeled
datasets require expensive expert annotators. We address this bottleneck within
the legal domain by introducing the Contract Understanding Atticus Dataset
(CUAD), a new dataset for legal contract review. CUAD was created with dozens
of legal experts from The Atticus Project and consists of over 13,000
annotations. The task is to highlight salient portions of a contract that are
important for a human to review. We find that Transformer models have nascent
performance, but that this performance is strongly influenced by model design
and training dataset size. Despite these promising results, there is still
substantial room for improvement. As one of the only large, specialized NLP
benchmarks annotated by experts, CUAD can serve as a challenging research
benchmark for the broader NLP community.

    

### [[2103.06443] Where is your place, Visual Place Recognition?](http://arxiv.org/abs/2103.06443)


  Visual Place Recognition (VPR) is often characterized as being able to
recognize the same place despite significant changes in appearance and
viewpoint. VPR is a key component of Spatial Artificial Intelligence, enabling
robotic platforms and intelligent augmentation platforms such as augmented
reality devices to perceive and understand the physical world. In this paper,
we observe that there are three "drivers" that impose requirements on spatially
intelligent agents and thus VPR systems: 1) the particular agent including its
sensors and computational resources, 2) the operating environment of this
agent, and 3) the specific task that the artificial agent carries out. In this
paper, we characterize and survey key works in the VPR area considering those
drivers, including their place representation and place matching choices. We
also provide a new definition of VPR based on the visual overlap -- akin to
spatial view cells in the brain -- that enables us to find similarities and
differences to other research areas in the robotics and computer vision fields.
We identify numerous open challenges and suggest areas that require more
in-depth attention in future works.

    

### [[2103.07364] A Unified Game-Theoretic Interpretation of Adversarial Robustness](http://arxiv.org/abs/2103.07364)


  This paper provides a unified view to explain different adversarial attacks
and defense methods, i.e. the view of multi-order interactions between input
variables of DNNs. Based on the multi-order interaction, we discover that
adversarial attacks mainly affect high-order interactions to fool the DNN.
Furthermore, we find that the robustness of adversarially trained DNNs comes
from category-specific low-order interactions. Our findings provide a potential
method to unify adversarial perturbations and robustness, which can explain the
existing defense methods in a principle way. Besides, our findings also make a
revision of previous inaccurate understanding of the shape bias of
adversarially learned features.

    

### [[2103.10685] Controllable Generation from Pre-trained Language Models via Inverse Prompting](http://arxiv.org/abs/2103.10685)


  Large-scale pre-trained language models have demonstrated strong capabilities
of generating realistic text. However, it remains challenging to control the
generation results. Previous approaches such as prompting are far from
sufficient, which limits the usage of language models. To tackle this
challenge, we propose an innovative method, inverse prompting, to better
control text generation. The core idea of inverse prompting is to use generated
text to inversely predict the prompt during beam search, which enhances the
relevance between the prompt and the generated text and provides better
controllability. Empirically, we pre-train a large-scale Chinese language model
to perform a systematic study using human evaluation on the tasks of
open-domain poem generation and open-domain long-form question answering. Our
results show that our proposed method substantially outperforms the baselines
and that our generation quality is close to human performance on some of the
tasks.
Narrators can try our poem generation demo at
this https URL, while our QA demo can be found at
this https URL. For researchers, the code is provided in
this https URL.

    

### [[2105.07253] Regret Minimization Experience Replay in Off-Policy Reinforcement Learning](http://arxiv.org/abs/2105.07253)


  In reinforcement learning, experience replay stores past samples for further
reuse. Prioritized sampling is a promising technique to better utilize these
samples. Previous criteria of prioritization include TD error, recentness and
corrective feedback, which are mostly heuristically designed. In this work, we
start from the regret minimization objective, and obtain an optimal
prioritization strategy for Bellman update that can directly maximize the
return of the policy. The theory suggests that data with higher hindsight TD
error, better on-policiness and more accurate Q value should be assigned with
higher weights during sampling. Thus most previous criteria only consider this
strategy partially. We not only provide theoretical justifications for previous
criteria, but also propose two new methods to compute the prioritization
weight, namely ReMERN and ReMERT. ReMERN learns an error network, while ReMERT
exploits the temporal ordering of states. Both methods outperform previous
prioritized sampling algorithms in challenging RL benchmarks, including MuJoCo,
Atari and Meta-World.

    

### [[2105.09938] Measuring Coding Challenge Competence With APPS](http://arxiv.org/abs/2105.09938)


  While programming is one of the most broadly applicable skills in modern
society, modern machine learning models still cannot code solutions to basic
problems. Despite its importance, there has been surprisingly little work on
evaluating code generation, and it can be difficult to accurately assess code
generation performance rigorously. To meet this challenge, we introduce APPS, a
benchmark for code generation. Unlike prior work in more restricted settings,
our benchmark measures the ability of models to take an arbitrary natural
language specification and generate satisfactory Python code. Similar to how
companies assess candidate software developers, we then evaluate models by
checking their generated code on test cases. Our benchmark includes 10,000
problems, which range from having simple one-line solutions to being
substantial algorithmic challenges. We fine-tune large language models on both
GitHub and our training set, and we find that the prevalence of syntax errors
is decreasing exponentially as models improve. Recent models such as GPT-Neo
can pass approximately 20% of the test cases of introductory problems, so we
find that machine learning models are now beginning to learn how to code. As
the social significance of automatic code generation increases over the coming
years, our benchmark can provide an important measure for tracking
advancements.

    

### [[2105.13304] Anomalous phase separation dynamics in a correlated electron system: machine-learning enabled large-scale kinetic Monte Carlo simulations](http://arxiv.org/abs/2105.13304)


  Phase separation plays a central role in the emergence of novel
functionalities of correlated electron materials. The structure of the
mixed-phase states depends strongly on the nonequilibrium phase-separation
dynamics, which has so far yet to be systematically investigated, especially on
the theoretical side. With the aid of modern machine learning methods, we
demonstrate the first-ever large-scale kinetic Monte Carlo simulations of the
phase separation process for the Falicov-Kimball model, which is one of the
canonical strongly correlated electron systems. We uncover an unusual
phase-separation scenario where domain coarsening occurs simultaneously at two
different scales: the growth of checkerboard clusters at smaller length scales
and the expansion of super-clusters, which are aggregates of the checkerboard
clusters of the same sign, at a larger scale. We show that the emergence of
super-clusters is due to a hidden dynamical breaking of the sublattice
symmetry. Arrested growth of the checkerboard patterns and of the
super-clusters is shown to result from a correlation-induced self-trapping
mechanism. Glassy behaviors similar to the one reported in this work could be
generic for other correlated electron systems.

    

### [[2105.14417] Overparameterization of deep ResNet: zero loss and mean-field analysis](http://arxiv.org/abs/2105.14417)


  Finding parameters in a deep neural network (NN) that fit training data is a
nonconvex optimization problem, but a basic first-order optimization method
(gradient descent) finds a global optimizer with perfect fit (zero-loss) in
many practical situations. We examine this phenomenon for the case of Residual
Neural Networks (ResNet) with smooth activation functions in a limiting regime
in which both the number of layers (depth) and the number of weights in each
layer (width) go to infinity. First, we use a mean-field-limit argument to
prove that the gradient descent for parameter training becomes a gradient flow
for a probability distribution that is characterized by a partial differential
equation (PDE) in the large-NN limit. Next, we show that under certain
assumptions, the solution to the PDE converges in the training time to a
zero-loss solution. Together, these results suggest that the training of the
ResNet gives a near-zero loss if the ResNet is large enough. We give estimates
of the depth and width needed to reduce the loss below a given threshold, with
high probability.

    

### [[2106.01357] Diffusion Schrödinger Bridge with Applications to Score-Based Generative Modeling](http://arxiv.org/abs/2106.01357)


  Progressively applying Gaussian noise transforms complex data distributions
to approximately Gaussian. Reversing this dynamic defines a generative model.
When the forward noising process is given by a Stochastic Differential Equation
(SDE), Song et al. (2021) demonstrate how the time inhomogeneous drift of the
associated reverse-time SDE may be estimated using score-matching. A limitation
of this approach is that the forward-time SDE must be run for a sufficiently
long time for the final distribution to be approximately Gaussian. In contrast,
solving the Schrödinger Bridge problem (SB), i.e. an entropy-regularized
optimal transport problem on path spaces, yields diffusions which generate
samples from the data distribution in finite time. We present Diffusion SB
(DSB), an original approximation of the Iterative Proportional Fitting (IPF)
procedure to solve the SB problem, and provide theoretical analysis along with
generative modeling experiments. The first DSB iteration recovers the
methodology proposed by Song et al. (2021), with the flexibility of using
shorter time intervals, as subsequent DSB iterations reduce the discrepancy
between the final-time marginal of the forward (resp. backward) SDE with
respect to the prior (resp. data) distribution. Beyond generative modeling, DSB
offers a widely applicable computational optimal transport tool as the
continuous state-space analogue of the popular Sinkhorn algorithm (Cuturi,
2013).

    

### [[2106.01596] Semantic-Aware Contrastive Learning for Multi-object Medical Image Segmentation](http://arxiv.org/abs/2106.01596)


  Medical image segmentation, or computing voxelwise semantic masks, is a
fundamental yet challenging task to compute a voxel-level semantic mask. To
increase the ability of encoder-decoder neural networks to perform this task
across large clinical cohorts, contrastive learning provides an opportunity to
stabilize model initialization and enhance encoders without labels. However,
multiple target objects (with different semantic meanings) may exist in a
single image, which poses a problem for adapting traditional contrastive
learning methods from prevalent 'image-level classification' to 'pixel-level
segmentation'. In this paper, we propose a simple semantic-aware contrastive
learning approach leveraging attention masks to advance multi-object semantic
segmentation. Briefly, we embed different semantic objects to different
clusters rather than the traditional image-level embeddings. We evaluate our
proposed method on a multi-organ medical image segmentation task with both
in-house data and MICCAI Challenge 2015 BTCV datasets. Compared with current
state-of-the-art training strategies, our proposed pipeline yields a
substantial improvement of 5.53% and 6.09% on Dice score for both medical image
segmentation cohorts respectively (p-value<0.01). The performance of the
proposed method is further assessed on natural images via the PASCAL VOC 2012
dataset, and achieves a substantial improvement of 2.75% on mIoU
(p-value<0.01).

    

### [[2106.02067] Learning to Draw: Emergent Communication through Sketching](http://arxiv.org/abs/2106.02067)


  Evidence that visual communication preceded written language and provided a
basis for it goes back to prehistory, in forms such as cave and rock paintings
depicting traces of our distant ancestors. Emergent communication research has
sought to explore how agents can learn to communicate in order to
collaboratively solve tasks. Existing research has focused on language, with a
learned communication channel transmitting sequences of discrete tokens between
the agents. In this work, we explore a visual communication channel between
agents that are allowed to draw with simple strokes. Our agents are
parameterised by deep neural networks, and the drawing procedure is
differentiable, allowing for end-to-end training. In the framework of a
referential communication game, we demonstrate that agents can not only
successfully learn to communicate by drawing, but with appropriate inductive
biases, can do so in a fashion that humans can interpret. We hope to encourage
future research to consider visual communication as a more flexible and
directly interpretable alternative of training collaborative agents.

    

### [[2106.02795] Learnable Fourier Features for Multi-Dimensional Spatial Positional Encoding](http://arxiv.org/abs/2106.02795)


  Attentional mechanisms are order-invariant. Positional encoding is a crucial
component to allow attention-based deep model architectures such as Transformer
to address sequences or images where the position of information matters. In
this paper, we propose a novel positional encoding method based on learnable
Fourier features. Instead of hard-coding each position as a token or a vector,
we represent each position, which can be multi-dimensional, as a trainable
encoding based on learnable Fourier feature mapping, modulated with a
multi-layer perceptron. The representation is particularly advantageous for a
spatial multi-dimensional position, e.g., pixel positions on an image, where
$L_2$ distances or more complex positional relationships need to be captured.
Our experiments based on several public benchmark tasks show that our learnable
Fourier feature representation for multi-dimensional positional encoding
outperforms existing methods by both improving the accuracy and allowing faster
convergence.

    

### [[2106.04696] Curriculum Design for Teaching via Demonstrations: Theory and Applications](http://arxiv.org/abs/2106.04696)


  We consider the problem of teaching via demonstrations in sequential
decision-making settings. In particular, we study how to design a personalized
curriculum over demonstrations to speed up the learner's convergence. We
provide a unified curriculum strategy for two popular learner models: Maximum
Causal Entropy Inverse Reinforcement Learning (MaxEnt-IRL) and Cross-Entropy
Behavioral Cloning (CrossEnt-BC). Our unified strategy induces a ranking over
demonstrations based on a notion of difficulty scores computed w.r.t. the
teacher's optimal policy and the learner's current policy. Compared to the
state of the art, our strategy doesn't require access to the learner's internal
dynamics and still enjoys similar convergence guarantees under mild technical
conditions. Furthermore, we adapt our curriculum strategy to the setting where
no teacher agent is present using task-specific difficulty scores. Experiments
on a synthetic car driving environment and navigation-based environments
demonstrate the effectiveness of our curriculum strategy.

    

### [[2106.06663] TDGIA:Effective Injection Attacks on Graph Neural Networks](http://arxiv.org/abs/2106.06663)


  Graph Neural Networks (GNNs) have achieved promising performance in various
real-world applications. However, recent studies have shown that GNNs are
vulnerable to adversarial attacks. In this paper, we study a
recently-introduced realistic attack scenario on graphs -- graph injection
attack (GIA). In the GIA scenario, the adversary is not able to modify the
existing link structure and node attributes of the input graph, instead the
attack is performed by injecting adversarial nodes into it. We present an
analysis on the topological vulnerability of GNNs under GIA setting, based on
which we propose the Topological Defective Graph Injection Attack (TDGIA) for
effective injection attacks. TDGIA first introduces the topological defective
edge selection strategy to choose the original nodes for connecting with the
injected ones. It then designs the smooth feature optimization objective to
generate the features for the injected nodes. Extensive experiments on
large-scale datasets show that TDGIA can consistently and significantly
outperform various attack baselines in attacking dozens of defense GNN models.
Notably, the performance drop on target GNNs resultant from TDGIA is more than
double the damage brought by the best attack solution among hundreds of
submissions on KDD-CUP 2020.

    

### [[2106.10711] Transfer Bayesian Meta-learning via Weighted Free Energy Minimization](http://arxiv.org/abs/2106.10711)


  Meta-learning optimizes the hyperparameters of a training procedure, such as
its initialization, kernel, or learning rate, based on data sampled from a
number of auxiliary tasks. A key underlying assumption is that the auxiliary
tasks, known as meta-training tasks, share the same generating distribution as
the tasks to be encountered at deployment time, known as meta-test tasks. This
may, however, not be the case when the test environment differ from the
meta-training conditions. To address shifts in task generating distribution
between meta-training and meta-testing phases, this paper introduces weighted
free energy minimization (WFEM) for transfer meta-learning. We instantiate the
proposed approach for non-parametric Bayesian regression and classification via
Gaussian Processes (GPs). The method is validated on a toy sinusoidal
regression problem, as well as on classification using miniImagenet and CUB
data sets, through comparison with standard meta-learning of GP priors as
implemented by PACOH.

    

### [[2106.12575] Weisfeiler and Lehman Go Cellular: CW Networks](http://arxiv.org/abs/2106.12575)


  Graph Neural Networks (GNNs) are limited in their expressive power, struggle
with long-range interactions and lack a principled way to model higher-order
structures. These problems can be attributed to the strong coupling between the
computational graph and the input graph structure. The recently proposed
Message Passing Simplicial Networks naturally decouple these elements by
performing message passing on the clique complex of the graph. Nevertheless,
these models can be severely constrained by the rigid combinatorial structure
of Simplicial Complexes (SCs). In this work, we extend recent theoretical
results on SCs to regular Cell Complexes, topological objects that flexibly
subsume SCs and graphs. We show that this generalisation provides a powerful
set of graph "lifting" transformations, each leading to a unique hierarchical
message passing procedure. The resulting methods, which we collectively call CW
Networks (CWNs), are strictly more powerful than the WL test and not less
powerful than the 3-WL test. In particular, we demonstrate the effectiveness of
one such scheme, based on rings, when applied to molecular graph problems. The
proposed architecture benefits from provably larger expressivity than commonly
used GNNs, principled modelling of higher-order signals and from compressing
the distances between nodes. We demonstrate that our model achieves
state-of-the-art results on a variety of molecular datasets.

    

### [[2106.14343] High-probability Bounds for Non-Convex Stochastic Optimization with Heavy Tails](http://arxiv.org/abs/2106.14343)


  We consider non-convex stochastic optimization using first-order algorithms
for which the gradient estimates may have heavy tails. We show that a
combination of gradient clipping, momentum, and normalized gradient descent
yields convergence to critical points in high-probability with best-known rates
for smooth losses when the gradients only have bounded $\mathfrak{p}$th moments
for some $\mathfrak{p}\in(1,2]$. We then consider the case of second-order
smooth losses, which to our knowledge have not been studied in this setting,
and again obtain high-probability bounds for any $\mathfrak{p}$. Moreover, our
results hold for arbitrary smooth norms, in contrast to the typical SGD
analysis which requires a Hilbert space norm. Further, we show that after a
suitable "burn-in" period, the objective value will monotonically decrease for
every iteration until a critical point is identified, which provides intuition
behind the popular practice of learning rate "warm-up" and also yields a
last-iterate guarantee.

    

### [[2111.04357] Can semi-supervised learning reduce the amount of manual labelling required for effective radio galaxy morphology classification?](http://arxiv.org/abs/2111.04357)


  In this work, we examine the robustness of state-of-the-art semi-supervised
learning (SSL) algorithms when applied to morphological classification in
modern radio astronomy. We test whether SSL can achieve performance comparable
to the current supervised state of the art when using many fewer labelled data
points and if these results generalise to using truly unlabelled data. We find
that although SSL provides additional regularisation, its performance degrades
rapidly when using very few labels, and that using truly unlabelled data leads
to a significant drop in performance.

    

### [[2009.11939] Deep Multi-Scale Feature Learning for Defocus Blur Estimation](http://arxiv.org/abs/2009.11939)


  This paper presents an edge-based defocus blur estimation method from a
single defocused image. We first distinguish edges that lie at depth
discontinuities (called depth edges, for which the blur estimate is ambiguous)
from edges that lie at approximately constant depth regions (called pattern
edges, for which the blur estimate is well-defined). Then, we estimate the
defocus blur amount at pattern edges only, and explore an interpolation scheme
based on guided filters that prevents data propagation across the detected
depth edges to obtain a dense blur map with well-defined object boundaries.
Both tasks (edge classification and blur estimation) are performed by deep
convolutional neural networks (CNNs) that share weights to learn meaningful
local features from multi-scale patches centered at edge locations. Experiments
on naturally defocused images show that the proposed method presents
qualitative and quantitative results that outperform state-of-the-art (SOTA)
methods, with a good compromise between running time and accuracy.

    

### [[2104.10777] Viking: Variational Bayesian Variance Tracking](http://arxiv.org/abs/2104.10777)


  We consider the problem of time series forecasting in an adaptive setting. We
focus on the inference of state-space models under unknown and potentially
time-varying noise variances. We introduce an augmented model in which the
variances are represented as auxiliary gaussian latent variables in a tracking
mode. As variances are nonnegative, a transformation is chosen and applied to
these latent variables. The inference relies on the online variational Bayesian
methodology, which consists in minimizing a Kullback-Leibler divergence at each
time step. We observe that the minimum of the Kullback-Leibler divergence is an
extension of the Kalman filter taking into account the variance uncertainty. We
design a novel algorithm, named Viking, using these optimal recursive updates.
For auxiliary latent variables, we use second-order bounds whose optimum admit
closed-form solutions. Experiments on synthetic data show that Viking behaves
well and is robust to misspecification.

    

### [[2111.04913] vlang: Mapping Verilog Netlists to Modern Technologies](http://arxiv.org/abs/2111.04913)


  Portability of hardware designs between Programmable Logic Devices (PLD) can
be accomplished through the use of device-agnostic hardware description
languages (HDL) such as Verilog or VHDL. Hardware designers can use HDLs to
migrate hardware designs between devices and explore performance, area and
power tradeoffs, as well as, port designs to an alternative device. However, if
design files are corrupt or missing, the portability of the design is lost.
While reverse engineering efforts may be able to recover an HDL-netlist of the
original design, HDL-netlists use device-specific primitives, restricting
portability. Additionally, the recovered design may benefit from other
computational technologies (e.g., $\mu$P, GPGPUs), but is restricted to the
domain of PLDs. In this work, we provide a new framework, vlang, which
automatically maps Verilog-netlists into LLVM's intermediate representation
(IR). The remapped design can use the LLVM-framework to target many device
technologies such as: x86-64 assembly, RISC-V, ARM or to other PLDs with a
modern high-level synthesis tool. Our framework is able to preserve the exact
functionality of the original design within the software executable. The
vlang-produced software executable can be used with other software programs, or
to verify the functionality and correctness of the remapped design. We evaluate
our work with a suite of hardware designs from OpenCores. We compare our
framework against state-of-the-art simulators, thereby outlining our
framework's ability to produce a fully-functional, cycle accurate
software-executable. We also explore the usage of vlang as a front-end for
high-level synthesis tools.

    

### [[2111.05002] Phantom: A High-Performance Computational Core for Sparse Convolutional Neural Networks](http://arxiv.org/abs/2111.05002)


  Sparse convolutional neural networks (CNNs) have gained significant traction
over the past few years as sparse CNNs can drastically decrease the model size
and computations, if exploited befittingly, as compared to their dense
counterparts. Sparse CNNs often introduce variations in the layer shapes and
sizes, which can prevent dense accelerators from performing well on sparse CNN
models. Recently proposed sparse accelerators like SCNN, Eyeriss v2, and
SparTen, actively exploit the two-sided or full sparsity, that is, sparsity in
both weights and activations, for performance gains. These accelerators,
however, either have inefficient micro-architecture, which limits their
performance, have no support for non-unit stride convolutions and
fully-connected (FC) layers, or suffer massively from systematic load
imbalance. To circumvent these issues and support both sparse and dense models,
we propose Phantom, a multi-threaded, dynamic, and flexible neural
computational core. Phantom uses sparse binary mask representation to actively
lookahead into sparse computations, and dynamically schedule its computational
threads to maximize the thread utilization and throughput. We also generate a
two-dimensional (2D) mesh architecture of Phantom neural computational cores,
which we refer to as Phantom-2D accelerator, and propose a novel dataflow that
supports all layers of a CNN, including unit and non-unit stride convolutions,
and FC layers. In addition, Phantom-2D uses a two-level load balancing strategy
to minimize the computational idling, thereby, further improving the hardware
utilization. To show support for different types of layers, we evaluate the
performance of the Phantom architecture on VGG16 and MobileNet. Our simulations
show that the Phantom-2D accelerator attains a performance gain of 12x, 4.1x,
1.98x, and 2.36x, over dense architectures, SCNN, SparTen, and Eyeriss v2,
respectively.

    

### [[2111.05301] Adaptable Register File Organization for Vector Processors](http://arxiv.org/abs/2111.05301)


  Modern scientific applications are getting more diverse, and the vector
lengths in those applications vary widely. Contemporary Vector Processors (VPs)
are designed either for short vector lengths, e.g., Fujitsu A64FX with 512-bit
ARM SVE vector support, or long vectors, e.g., NEC Aurora Tsubasa with 16Kbits
Maximum Vector Length (MVL). Unfortunately, both approaches have drawbacks. On
the one hand, short vector length VP designs struggle to provide high
efficiency for applications featuring long vectors with high Data Level
Parallelism (DLP). On the other hand, long vector VP designs waste resources
and underutilize the Vector Register File (VRF) when executing low DLP
applications with short vector lengths. Therefore, those long vector VP
implementations are limited to a specialized subset of applications, where
relatively high DLP must be present to achieve excellent performance with high
efficiency. To overcome these limitations, we propose an Adaptable Vector
Architecture (AVA) that leads to having the best of both worlds. AVA is
designed for short vectors (MVL=16 elements) and is thus area and
energy-efficient. However, AVA has the functionality to reconfigure the MVL,
thereby allowing to exploit the benefits of having a longer vector (up to 128
elements) microarchitecture when abundant DLP is present. We model AVA on the
gem5 simulator and evaluate the performance with six applications taken from
the RiVEC Benchmark Suite. To obtain area and power consumption metrics, we
model AVA on McPAT for 22nm technology. Our results show that by reconfiguring
our small VRF (8KB) plus our novel issue queue scheme, AVA yields a 2X speedup
over the default configuration for short vectors. Additionally, AVA shows
competitive performance when compared to a long vector VP, while saving 50% of
area.

    

### [[2111.04837] D$^2$ABS: A Framework for Dynamic Dependence Abstraction of Distributed Programs](http://arxiv.org/abs/2111.04837)


  As modern software systems are increasingly developed for running in
distributed environments, it is crucial to provide fundamental techniques such
as dependence analysis for checking, diagnosing, and evolving those systems.
However, traditional dependence analysis is either inapplicable or of very
limited utility for distributed programs due to the decoupled components of
these programs that run in concurrent processes at physically separated
machines. Motivated by the need for dependence analysis of distributed software
and the diverse cost-effectiveness needs of dependence-based applications, this
paper presents D$^2$ABS, a framework of dynamic dependence abstraction for
distributed programs. By partial-ordering distributed method-execution events
and inferring causality from the ordered events, D$^2$ABS abstracts
method-level dependencies both within and across process boundaries. Further,
by exploiting message-passing semantics across processes, and incorporating
static dependencies and statement coverage within individual components, we
present three additional instantiations of D$^2$ABS that trade efficiency for
better precision. We present the design of the D$^2$ABS framework and evaluate
the four instantiations of D$^2$ABS on distributed systems of various
architectures and scales using our implementation for Java. Our empirical
results show that D$^2$ABS is significantly more effective than existing
options while offering varied levels of cost-effectiveness tradeoffs. As our
framework essentially computes whole-system run-time dependencies, it naturally
empowers a range of other dependence-based applications.

    

### [[2111.04872] Performance Evaluation of Python ParallelProgramming Models: Charm4Py and mpi4py](http://arxiv.org/abs/2111.04872)


  Python is rapidly becoming the lingua franca of machine learning and
scientific computing. With the broad use of frameworks such as Numpy, SciPy,
and TensorFlow, scientific computing and machine learning are seeing a
productivity boost on systems without a requisite loss in performance. While
high-performance libraries often provide adequate performance within a node,
distributed computing is required to scale Python across nodes and make it
genuinely competitive in large-scale high-performance computing. Many
frameworks, such as Charm4Py, DaCe, Dask, Legate Numpy, mpi4py, and Ray, scale
Python across nodes. However, little is known about these frameworks' relative
strengths and weaknesses, leaving practitioners and scientists without enough
information about which frameworks are suitable for their requirements. In this
paper, we seek to narrow this knowledge gap by studying the relative
performance of two such frameworks: Charm4Py and mpi4py.
We perform a comparative performance analysis of Charm4Py and mpi4py using
CPU and GPU-based microbenchmarks other representative mini-apps for scientific
computing.

    

### [[2111.04959] DataX: A system for Data eXchange and transformation of streams](http://arxiv.org/abs/2111.04959)


  The exponential growth in smart sensors and rapid progress in 5G networks is
creating a world awash with data streams. However, a key barrier to building
performant multi-sensor, distributed stream processing applications is high
programming complexity. We propose DataX, a novel platform that improves
programmer productivity by enabling easy exchange, transformations, and fusion
of data streams. DataX abstraction simplifies the application's specification
and exposes parallelism and dependencies among the application functions
(microservices). DataX runtime automatically sets up appropriate data
communication mechanisms, enables effortless reuse of microservices and data
streams across applications, and leverages serverless computing to transform,
fuse, and auto-scale microservices. DataX makes it easy to write, deploy and
reliably operate distributed applications at scale. Synthesizing these
capabilities into a single platform is substantially more transformative than
any available stream processing system.

    

### [[2111.04994] Analysis of Work-Stealing and Parallel Cache Complexity](http://arxiv.org/abs/2111.04994)


  Parallelism has become extremely popular over the past decade, and there have
been a lot of new parallel algorithms and software. The randomized
work-stealing (RWS) scheduler plays a crucial role in this ecosystem. In this
paper, we study two important topics related to the randomized work-stealing
scheduler.
Our first contribution is a simplified, classroom-ready version of analysis
for the RWS scheduler. The theoretical efficiency of the RWS scheduler has been
analyzed for a variety of settings, but most of them are quite complicated. In
this paper, we show a new analysis, which we believe is easy to understand, and
can be especially useful in education. We avoid using the potential function in
the analysis, and we assume a highly asynchronous setting, which is more
realistic for today's parallel machines.
Our second and main contribution is some new parallel cache complexity for
algorithms using the RWS scheduler. Although the sequential I/O model has been
well-studied over the past decades, so far very few results have extended it to
the parallel setting. The parallel cache bounds of many existing algorithms are
affected by a polynomial of the span, which causes a significant overhead for
high-span algorithms. Our new analysis decouples the span from the analysis of
the parallel cache complexity. This allows us to show new parallel cache bounds
for a list of classic algorithms. Our results are only a polylogarithmic factor
off the lower bounds, and significantly improve previous results.

    

### [[2111.05111] Population Protocols for Graph Class Identification Problems](http://arxiv.org/abs/2111.05111)


  In this paper, we focus on graph class identification problems in the
population protocol model. A graph class identification problem aims to decide
whether a given communication graph is in the desired class (e.g. whether the
given communication graph is a ring graph). Angluin et al. proposed graph class
identification protocols with directed graphs and designated initial states
under global fairness [Angluin et al., DCOSS2005]. We consider graph class
identification problems for undirected graphs on various assumptions such as
initial states of agents, fairness of the execution, and initial knowledge of
agents. In particular, we focus on lines, rings, $k$-regular graphs, stars,
trees, and bipartite graphs. With designated initial states, we propose graph
class identification protocols for $k$-regular graphs, and trees under global
fairness, and propose a graph class identification protocol for stars under
weak fairness. Moreover, we show that, even if agents know the number of agents
$n$, there is no graph class identification protocol for lines, rings,
$k$-regular graphs, trees, or bipartite graphs under weak fairness. On the
other hand, with arbitrary initial states, we show that there is no graph class
identification protocol for lines, rings, $k$-regular graphs, stars, trees, or
bipartite graphs.

    

### [[2111.05167] Tarema: Adaptive Resource Allocation for Scalable Scientific Workflows in Heterogeneous Clusters](http://arxiv.org/abs/2111.05167)


  Scientific workflow management systems like Nextflow support large-scale data
analysis by abstracting away the details of scientific workflows. In these
systems, workflows consist of several abstract tasks, of which instances are
run in parallel and transform input partitions into output partitions. Resource
managers like Kubernetes execute such workflow tasks on cluster
infrastructures. However, these resource managers only consider the number of
CPUs and the amount of available memory when assigning tasks to resources; they
do not consider hardware differences beyond these numbers, while computational
speed and memory access rates can differ significantly.
We propose Tarema, a system for allocating task instances to heterogeneous
cluster resources during the execution of scalable scientific workflows. First,
Tarema profiles the available infrastructure with a set of benchmark programs
and groups cluster nodes with similar performance. Second, Tarema uses online
monitoring data of tasks, assigning labels to tasks depending on their resource
usage. Third, Tarema uses the node groups and task labels to dynamically assign
task instances evenly to resources based on resource demand. Our evaluation of
a prototype implementation for Kubernetes, using five real-world Nextflow
workflows from the popular nf-core framework and two 15-node clusters
consisting of different virtual machines, shows a mean reduction of isolated
job runtimes by 19.8% compared to popular schedulers in widely-used resource
managers and 4.54% compared to the heuristic SJFN, while providing a better
cluster usage. Moreover, executing two long-running workflows in parallel and
on restricted resources shows that Tarema is able to reduce the runtimes even
more while providing a fair cluster usage.

    

### [[2111.05296] Resistance Distance and Control Performance for Google bittide Synchronization](http://arxiv.org/abs/2111.05296)


  We discuss control of bittide distributed systems, which are designed to
provide logical synchronization between networked machines by observing data
flow rates between adjacent systems at the physical network layer and
controlling local reference clock frequencies. We analyze the performance of
approximate proportional-integral control of the synchronization mechanism and
develop a simple continuous-time model to show the resulting dynamics are
stable for any positive choice of gains. We then construct explicit formulae to
show that closed-loop performance measured using the L2 norm is a product of
two terms, one depending only on resistance distances in the graph, and the
other depending only on controller gains.

    

### [[2111.04734] Mixed Transformer U-Net For Medical Image Segmentation](http://arxiv.org/abs/2111.04734)


  Though U-Net has achieved tremendous success in medical image segmentation
tasks, it lacks the ability to explicitly model long-range dependencies.
Therefore, Vision Transformers have emerged as alternative segmentation
structures recently, for their innate ability of capturing long-range
correlations through Self-Attention (SA). However, Transformers usually rely on
large-scale pre-training and have high computational complexity. Furthermore,
SA can only model self-affinities within a single sample, ignoring the
potential correlations of the overall dataset. To address these problems, we
propose a novel Transformer module named Mixed Transformer Module (MTM) for
simultaneous inter- and intra- affinities learning. MTM first calculates
self-affinities efficiently through our well-designed Local-Global
Gaussian-Weighted Self-Attention (LGG-SA). Then, it mines inter-connections
between data samples through External Attention (EA). By using MTM, we
construct a U-shaped model named Mixed Transformer U-Net (MT-UNet) for accurate
medical image segmentation. We test our method on two different public
datasets, and the experimental results show that the proposed method achieves
better performance over other state-of-the-art methods. The code is available
at: this https URL.

    

### [[2111.04740] BRACS: A Dataset for BReAst Carcinoma Subtyping in H&E Histology Images](http://arxiv.org/abs/2111.04740)


  Breast cancer is the most commonly diagnosed cancer and registers the highest
number of deaths for women with cancer. Recent advancements in diagnostic
activities combined with large-scale screening policies have significantly
lowered the mortality rates for breast cancer patients. However, the manual
inspection of tissue slides by the pathologists is cumbersome, time-consuming,
and is subject to significant inter- and intra-observer variability. Recently,
the advent of whole-slide scanning systems have empowered the rapid
digitization of pathology slides, and enabled to develop digital workflows.
These advances further enable to leverage Artificial Intelligence (AI) to
assist, automate, and augment pathological diagnosis. But the AI techniques,
especially Deep Learning (DL), require a large amount of high-quality annotated
data to learn from. Constructing such task-specific datasets poses several
challenges, such as, data-acquisition level constrains, time-consuming and
expensive annotations, and anonymization of private information. In this paper,
we introduce the BReAst Carcinoma Subtyping (BRACS) dataset, a large cohort of
annotated Hematoxylin & Eosin (H&E)-stained images to facilitate the
characterization of breast lesions. BRACS contains 547 Whole-Slide Images
(WSIs), and 4539 Regions of Interest (ROIs) extracted from the WSIs. Each WSI,
and respective ROIs, are annotated by the consensus of three board-certified
pathologists into different lesion categories. Specifically, BRACS includes
three lesion types, i.e., benign, malignant and atypical, which are further
subtyped into seven categories. It is, to the best of our knowledge, the
largest annotated dataset for breast cancer subtyping both at WSI- and
ROI-level. Further, by including the understudied atypical lesions, BRACS
offers an unique opportunity for leveraging AI to better understand their
characteristics.

    

### [[2111.04785] Visual Question Answering based on Formal Logic](http://arxiv.org/abs/2111.04785)


  Visual question answering (VQA) has been gaining a lot of traction in the
machine learning community in the recent years due to the challenges posed in
understanding information coming from multiple modalities (i.e., images,
language). In VQA, a series of questions are posed based on a set of images and
the task at hand is to arrive at the answer. To achieve this, we take a
symbolic reasoning based approach using the framework of formal logic. The
image and the questions are converted into symbolic representations on which
explicit reasoning is performed. We propose a formal logic framework where (i)
images are converted to logical background facts with the help of scene graphs,
(ii) the questions are translated to first-order predicate logic clauses using
a transformer based deep learning model, and (iii) perform satisfiability
checks, by using the background knowledge and the grounding of predicate
clauses, to obtain the answer. Our proposed method is highly interpretable and
each step in the pipeline can be easily analyzed by a human. We validate our
approach on the CLEVR and the GQA dataset. We achieve near perfect accuracy of
99.6% on the CLEVR dataset comparable to the state of art models, showcasing
that formal logic is a viable tool to tackle visual question answering. Our
model is also data efficient, achieving 99.1% accuracy on CLEVR dataset when
trained on just 10% of the training data.

    

### [[2111.04845] Hybrid BYOL-ViT: Efficient approach to deal with small Datasets](http://arxiv.org/abs/2111.04845)


  Supervised learning can learn large representational spaces, which are
crucial for handling difficult learning tasks. However, due to the design of
the model, classical image classification approaches struggle to generalize to
new problems and new situations when dealing with small datasets. In fact,
supervised learning can lose the location of image features which leads to
supervision collapse in very deep architectures. In this paper, we investigate
how self-supervision with strong and sufficient augmentation of unlabeled data
can train effectively the first layers of a neural network even better than
supervised learning, with no need for millions of labeled data. The main goal
is to disconnect pixel data from annotation by getting generic task-agnostic
low-level features. Furthermore, we look into Vision Transformers (ViT) and
show that the low-level features derived from a self-supervised architecture
can improve the robustness and the overall performance of this emergent
architecture. We evaluated our method on one of the smallest open-source
datasets STL-10 and we obtained a significant boost of performance from 41.66%
to 83.25% when inputting low-level features from a self-supervised learning
architecture to the ViT instead of the raw images.

    

### [[2111.04862] Explaining Face Presentation Attack Detection Using Natural Language](http://arxiv.org/abs/2111.04862)


  A large number of deep neural network based techniques have been developed to
address the challenging problem of face presentation attack detection (PAD).
Whereas such techniques' focus has been on improving PAD performance in terms
of classification accuracy and robustness against unseen attacks and
environmental conditions, there exists little attention on the explainability
of PAD predictions. In this paper, we tackle the problem of explaining PAD
predictions through natural language. Our approach passes feature
representations of a deep layer of the PAD model to a language model to
generate text describing the reasoning behind the PAD prediction. Due to the
limited amount of annotated data in our study, we apply a light-weight LSTM
network as our natural language generation model. We investigate how the
quality of the generated explanations is affected by different loss functions,
including the commonly used word-wise cross entropy loss, a sentence
discriminative loss, and a sentence semantic loss. We perform our experiments
using face images from a dataset consisting of 1,105 bona-fide and 924
presentation attack samples. Our quantitative and qualitative results show the
effectiveness of our model for generating proper PAD explanations through text
as well as the power of the sentence-wise losses. To the best of our knowledge,
this is the first introduction of a joint biometrics-NLP task. Our dataset can
be obtained through our GitHub page.

    

### [[2111.04880] User Centered Design (VI): Human Factors Approaches for Intelligent Human-Computer Interaction](http://arxiv.org/abs/2111.04880)


  Starting from the design philosophy of "user-centered design", this paper
analyzes the human factors characteristics of intelligent human-computer
interaction (iHCI) and proposes a concept of "user-oriented iHCI". It further
proposes a new human factors framework for iHCI based on the theories of joint
cognitive systems, situation awareness, and intelligent agents. With the help
of the new concept and framework, the paper analyzes the human factors issues
in the ecosystem of autonomous vehicle co-driving and layouts future research
agenda. Finally, the paper analyzes the two important research areas in iHCI
(i.e., user intention recognition, human-computer collaboration) and points out
the focus of human factors research in the future.

    

### [[2111.04885] Lymph Node Detection in T2 MRI with Transformers](http://arxiv.org/abs/2111.04885)


  Identification of lymph nodes (LN) in T2 Magnetic Resonance Imaging (MRI) is
an important step performed by radiologists during the assessment of
lymphoproliferative diseases. The size of the nodes play a crucial role in
their staging, and radiologists sometimes use an additional contrast sequence
such as diffusion weighted imaging (DWI) for confirmation. However, lymph nodes
have diverse appearances in T2 MRI scans, making it tough to stage for
metastasis. Furthermore, radiologists often miss smaller metastatic lymph nodes
over the course of a busy day. To deal with these issues, we propose to use the
DEtection TRansformer (DETR) network to localize suspicious metastatic lymph
nodes for staging in challenging T2 MRI scans acquired by different scanners
and exam protocols. False positives (FP) were reduced through a bounding box
fusion technique, and a precision of 65.41\% and sensitivity of 91.66\% at 4 FP
per image was achieved. To the best of our knowledge, our results improve upon
the current state-of-the-art for lymph node detection in T2 MRI scans.

    

### [[2111.04909] FPM: A Collection of Large-scale Foundation Pre-trained Language Models](http://arxiv.org/abs/2111.04909)


  Recent work in language modeling has shown that training large-scale
Transformer models has promoted the latest developments in natural language
processing applications. However, there is very little work to unify the
current effective models. In this work, we use the current effective model
structure to launch a model set through the current most mainstream technology.
We think this will become the basic model in the future. For Chinese, using the
GPT-2[9] model, a 10.3 billion parameter language model was trained on the
Chinese dataset, and, in particular, a 2.9 billion parameter language model
based on dialogue data was trained; the BERT model was trained on the Chinese
dataset with 495 million parameters; the Transformer model has trained a
language model with 5.6 billion parameters on the Chinese dataset. In English,
corresponding training work has also been done. Using the GPT-2 model, a
language model with 6.4 billion parameters was trained on the English dataset;
the BERT[3] model trained a language model with 1.24 billion parameters on the
English dataset, and in particular, it trained a 688 million parameter based on
single card training technology Language model; Transformer model trained a
language model with 5.6 billion parameters on the English dataset. In the TNEWS
classification task evaluated by CLUE[13], the BERT-C model exceeded the 59.46%
accuracy of ALBERT-xxlarge with an accuracy rate of 59.99%, an increase of
0.53%. In the QQP classification task evaluated by GLUE[11], the accuracy rate
of 78.95% surpassed the accuracy rate of BERT-Large of 72.1%, an increase of
6.85%. Compared with the current accuracy rate of ERNIE, the first place in the
GLUE evaluation of 75.2%, an increase of 3.75%.

    

### [[2111.04916] Building an AI-ready RSE Workforce](http://arxiv.org/abs/2111.04916)


  Artificial Intelligence has been transforming industries and academic
research across the globe, and research software development is no exception.
Machine learning and deep learning are being applied in every aspect of the
research software development lifecycles, from new algorithm design paradigms
to software development processes. In this paper, we discuss our views on
today's challenges and opportunities that AI has presented on research software
development and engineers, and the approaches we, at the University of Florida,
are taking to prepare our workforce for the new era of AI.

    

### [[2111.04933] DSBERT:Unsupervised Dialogue Structure learning with BERT](http://arxiv.org/abs/2111.04933)


  Unsupervised dialogue structure learning is an important and meaningful task
in natural language processing. The extracted dialogue structure and process
can help analyze human dialogue, and play a vital role in the design and
evaluation of dialogue systems. The traditional dialogue system requires
experts to manually design the dialogue structure, which is very costly. But
through unsupervised dialogue structure learning, dialogue structure can be
automatically obtained, reducing the cost of developers constructing dialogue
process. The learned dialogue structure can be used to promote the dialogue
generation of the downstream task system, and improve the logic and consistency
of the dialogue robot's this http URL this paper, we propose a Bert-based
unsupervised dialogue structure learning algorithm DSBERT (Dialogue Structure
BERT). Different from the previous SOTA models VRNN and SVRNN, we combine BERT
and AutoEncoder, which can effectively combine context information. In order to
better prevent the model from falling into the local optimal solution and make
the dialogue state distribution more uniform and reasonable, we also propose
three balanced loss functions that can be used for dialogue structure learning.
Experimental results show that DSBERT can generate a dialogue structure closer
to the real structure, can distinguish sentences with different semantics and
map them to different hidden states.

    

### [[2111.04951] American Hate Crime Trends Prediction with Event Extraction](http://arxiv.org/abs/2111.04951)


  Social media platforms may provide potential space for discourses that
contain hate speech, and even worse, can act as a propagation mechanism for
hate crimes. The FBI's Uniform Crime Reporting (UCR) Program collects hate
crime data and releases statistic report yearly. These statistics provide
information in determining national hate crime trends. The statistics can also
provide valuable holistic and strategic insight for law enforcement agencies or
justify lawmakers for specific legislation. However, the reports are mostly
released next year and lag behind many immediate needs. Recent research mainly
focuses on hate speech detection in social media text or empirical studies on
the impact of a confirmed crime. This paper proposes a framework that first
utilizes text mining techniques to extract hate crime events from New York
Times news, then uses the results to facilitate predicting American
national-level and state-level hate crime trends. Experimental results show
that our method can significantly enhance the prediction performance compared
with time series or regression methods without event-related factors. Our
framework broadens the methods of national-level and state-level hate crime
trends prediction.

    

### [[2111.04983] Dynamic Parameterized Network for CTR Prediction](http://arxiv.org/abs/2111.04983)


  Learning to capture feature relations effectively and efficiently is
essential in click-through rate (CTR) prediction of modern recommendation
systems. Most existing CTR prediction methods model such relations either
through tedious manually-designed low-order interactions or through inflexible
and inefficient high-order interactions, which both require extra DNN modules
for implicit interaction modeling. In this paper, we proposed a novel plug-in
operation, Dynamic Parameterized Operation (DPO), to learn both explicit and
implicit interaction instance-wisely. We showed that the introduction of DPO
into DNN modules and Attention modules can respectively benefit two main tasks
in CTR prediction, enhancing the adaptiveness of feature-based modeling and
improving user behavior modeling with the instance-wise locality. Our Dynamic
Parameterized Networks significantly outperforms state-of-the-art methods in
the offline experiments on the public dataset and real-world production
dataset, together with an online A/B test. Furthermore, the proposed Dynamic
Parameterized Networks has been deployed in the ranking system of one of the
world's largest e-commerce companies, serving the main traffic of hundreds of
millions of active users.

    

### [[2111.04988] Ultra-Low Power Keyword Spotting at the Edge](http://arxiv.org/abs/2111.04988)


  Keyword spotting (KWS) has become an indispensable part of many intelligent
devices surrounding us, as audio is one of the most efficient ways of
interacting with these devices. The accuracy and performance of KWS solutions
have been the main focus of the researchers, and thanks to deep learning,
substantial progress has been made in this domain. However, as the use of KWS
spreads into IoT devices, energy efficiency becomes a very critical requirement
besides the performance. We believe KWS solutions that would seek power
optimization both in the hardware and the neural network (NN) model
architecture are advantageous over many solutions in the literature where
mostly the architecture side of the problem is considered. In this work, we
designed an optimized KWS CNN model by considering end-to-end energy efficiency
for the deployment at MAX78000, an ultra-low-power CNN accelerator. With the
combined hardware and model optimization approach, we achieve 96.3\% accuracy
for 12 classes while only consuming 251 uJ per inference. We compare our
results with other small-footprint neural network-based KWS solutions in the
literature. Additionally, we share the energy consumption of our model in
power-optimized ARM Cortex-M4F to depict the effectiveness of the chosen
hardware for the sake of clarity.

    

### [[2111.04997] Learning Numerical Action Models from Noisy Input Data](http://arxiv.org/abs/2111.04997)


  This paper presents the PlanMiner-N algorithm, a domain learning technique
based on the PlanMiner domain learning algorithm. The algorithm presented here
improves the learning capabilities of PlanMiner when using noisy data as input.
The PlanMiner algorithm is able to infer arithmetic and logical expressions to
learn numerical planning domains from the input data, but it was designed to
work under situations of incompleteness making it unreliable when facing noisy
input data. In this paper, we propose a series of enhancements to the learning
process of PlanMiner to expand its capabilities to learn from noisy data. These
methods preprocess the input data by detecting noise and filtering it and study
the learned action models learned to find erroneous preconditions/effects in
them. The methods proposed in this paper were tested using a set of domains
from the International Planning Competition (IPC). The results obtained
indicate that PlanMiner-N improves the performance of PlanMiner greatly when
facing noisy input data.

    

### [[2111.05014] GDCA: GAN-based single image super resolution with Dual discriminators and Channel Attention](http://arxiv.org/abs/2111.05014)


  Single Image Super-Resolution (SISR) is a very active research field. This
paper addresses SISR by using a GAN-based approach with dual discriminators and
incorporating it with an attention mechanism. The experimental results show
that GDCA can generate sharper and high pleasing images compare to other
conventional methods.

    

### [[2111.05017] An effective hybrid search algorithm for the multiple traveling repairman problem with profits](http://arxiv.org/abs/2111.05017)


  As an extension of the traveling repairman problem with profits, the multiple
traveling repairman problem with profits consists of multiple repairmen who
visit a subset of all customers to maximize the revenues collected through the
visited customers. To solve this challenging problem, an effective hybrid
search algorithm based on the memetic algorithm framework is proposed. It
integrates two distinguished features: a dedicated arc-based crossover to
generate high-quality offspring solutions and a fast evaluation technique to
reduce the complexity of exploring the classical neighborhoods. We show the
competitiveness of the algorithm on 470 benchmark instances compared to the
leading reference algorithms and report new best records for 137 instances as
well as equal best results for other 330 instances. We investigate the
importance of the key search components for the algorithm.

    

### [[2111.05068] Neural News Recommendation with Event Extraction](http://arxiv.org/abs/2111.05068)


  A key challenge of online news recommendation is to help users find articles
they are interested in. Traditional news recommendation methods usually use
single news information, which is insufficient to encode news and user
representation. Recent research uses multiple channel news information, e.g.,
title, category, and body, to enhance news and user representation. However,
these methods only use various attention mechanisms to fuse multi-view
embeddings without considering deep digging higher-level information contained
in the context. These methods encode news content on the word level and jointly
train the attention parameters in the recommendation network, leading to more
corpora being required to train the model. We propose an Event Extraction-based
News Recommendation (EENR) framework to overcome these shortcomings, utilizing
event extraction to abstract higher-level information. EENR also uses a
two-stage strategy to reduce parameters in subsequent parts of the
recommendation network. We train the Event Extraction module by external
corpora in the first stage and apply the trained model to the news
recommendation dataset to predict event-level information, including event
types, roles, and arguments, in the second stage. Then we fuse multiple channel
information, including event information, news title, and category, to encode
news and users. Extensive experiments on a real-world dataset show that our
EENR method can effectively improve the performance of news recommendations.
Finally, we also explore the reasonability of utilizing higher abstract level
information to substitute news body content.

    

### [[2111.05071] Conformity Assessments and Post-market Monitoring: A Guide to the Role of Auditing in the Proposed European AI Regulation](http://arxiv.org/abs/2111.05071)


  The proposed European Artificial Intelligence Act (AIA) is the first attempt
to elaborate a general legal framework for AI carried out by any major global
economy. As such, the AIA is likely to become a point of reference in the
larger discourse on how AI systems can (and should) be regulated. In this
article, we describe and discuss the two primary enforcement mechanisms
proposed in the AIA: the conformity assessments that providers of high-risk AI
systems are expected to conduct, and the post-market monitoring plans that
providers must establish to document the performance of high-risk AI systems
throughout their lifetimes. We argue that AIA can be interpreted as a proposal
to establish a Europe-wide ecosystem for conducting AI auditing, albeit in
other words. Our analysis offers two main contributions. First, by describing
the enforcement mechanisms included in the AIA in terminology borrowed from
existing literature on AI auditing, we help providers of AI systems understand
how they can prove adherence to the requirements set out in the AIA in
practice. Second, by examining the AIA from an auditing perspective, we seek to
provide transferable lessons from previous research about how to refine further
the regulatory approach outlined in the AIA. We conclude by highlighting seven
aspects of the AIA where amendments (or simply clarifications) would be
helpful. These include, above all, the need to translate vague concepts into
verifiable criteria and to strengthen the institutional safeguards concerning
conformity assessments based on internal checks.

    

### [[2111.05157] Self-checking Logical Agents](http://arxiv.org/abs/2111.05157)


  This paper presents a comprehensive framework for run-time self-checking of
logical agents, by means of temporal axioms to be dynamically checked. These
axioms are specified by using an agent-oriented interval temporal logic defined
to this purpose. We define syntax, semantics and pragmatics for this new logic,
specifically tailored for application to agents. In the resulting framework, we
encompass and extend our past work.

    

### [[2111.05318] A Differentiable Recipe for Learning Visual Non-Prehensile Planar Manipulation](http://arxiv.org/abs/2111.05318)


  Specifying tasks with videos is a powerful technique towards acquiring novel
and general robot skills. However, reasoning over mechanics and dexterous
interactions can make it challenging to scale learning contact-rich
manipulation. In this work, we focus on the problem of visual non-prehensile
planar manipulation: given a video of an object in planar motion, find
contact-aware robot actions that reproduce the same object motion. We propose a
novel architecture, Differentiable Learning for Manipulation (\ours), that
combines video decoding neural models with priors from contact mechanics by
leveraging differentiable optimization and finite difference based simulation.
Through extensive simulated experiments, we investigate the interplay between
traditional model-based techniques and modern deep learning approaches. We find
that our modular and fully differentiable architecture performs better than
learning-only methods on unseen objects and motions.
\url{this https URL}.

    

### [[1909.12425] A Re-classification of Information Seeking Tasks and Their Computational Solutions](http://arxiv.org/abs/1909.12425)


  This article presents a re-classification of information seeking (IS) tasks,
concepts, and algorithms. The proposed taxonomy provides new dimensions to look
into information seeking tasks and methods. The new dimensions include the
number of search iterations, search goal types, and procedures to reach these
goals. Differences along these dimensions for the information seeking tasks
call for suitable computational solutions. The article then reviews machine
learning solutions that match each new category. The paper ends with a review
of evaluation campaigns for IS systems.

    

### [[2109.09331] Modular Design Patterns for Hybrid Actors](http://arxiv.org/abs/2109.09331)


  Recently, a boxology (graphical language) with design patterns for hybrid AI
was proposed, combining symbolic and sub-symbolic learning and reasoning. In
this paper, we extend this boxology with actors and their interactions. The
main contributions of this paper are: 1) an extension of the taxonomy to
describe distributed hybrid AI systems with actors and interactions; and 2)
showing examples using a few design patterns relevant in multi-agent systems
and human-agent interaction.

    

### [[2111.05290] Stateful Dynamic Partial Order Reduction for Model Checking Event-Driven Applications that Do Not Terminate](http://arxiv.org/abs/2111.05290)


  Event-driven architectures are broadly used for systems that must respond to
events in the real world. Event-driven applications are prone to concurrency
bugs that involve subtle errors in reasoning about the ordering of events.
Unfortunately, there are several challenges in using existing model-checking
techniques on these systems. Event-driven applications often loop indefinitely
and thus pose a challenge for stateless model checking techniques. On the other
hand, deploying purely stateful model checking can explore large sets of
equivalent executions.
In this work, we explore a new technique that combines dynamic partial order
reduction with stateful model checking to support non-terminating applications.
Our work is (1) the first dynamic partial order reduction algorithm for
stateful model checking that is sound for non-terminating applications and (2)
the first dynamic partial reduction algorithm for stateful model checking of
event-driven applications. We experimented with the IoTCheck dataset: a study
of interactions in smart home app pairs. This dataset consists of app pairs
originated from 198 real-world smart home apps. Overall, our DPOR algorithm
successfully reduced the search space for the app pairs, enabling 69 pairs of
apps that did not finish without DPOR to finish and providing a 7X average
speedup.

    

### [<title>Wrong results after conversion to CoreML - XGBoost</title>](https://discuss.xgboost.ai/t/wrong-results-after-conversion-to-coreml/1387/5)